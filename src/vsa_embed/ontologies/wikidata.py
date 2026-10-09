"""Wikidata (CC0) as a frame ontology: the ontology of the E9 natural track T8 (decision 63, methodology M5).

The general relation benchmarks of the toolkit (BEAR, LRE relations, PopQA, TwoHopFact, Entity Inferences) are built on
Wikidata entities; T8 gives those entities frames so that the read / write / meta benchmarks can run on a trained store.

- **Records** (`WikidataClient`, `fetch_records`): English label, description, aliases, sitelink count and English
  Wikipedia title of each entity, and its truthy (best-rank) item-valued statements for a fixed property list
  (`PROPERTIES`), fetched from the Wikidata Query Service (SPARQL) in batches; names without a QID are resolved with the
  `wbsearchentities` API. Every raw response is cached gzip-compressed under the cache directory
  (`~/data/vsa-llm/wikidata/`), and the processed snapshot (`records.jsonl.gz`) is pinned by its sha256 before any
  linking, so a build never reads the live service.
- **Aliases** (`WikidataAliasPolicy`, `select_concepts`): the linker is case-insensitive, so a name is an alias only when
  the text writes it as a name: in the screened FineWeb-Edu text at most `max_lowercase_share` of its occurrences are
  all-lowercase ("The Great", "Friends", "Apple" fail; "Kyriakos Mitsotakis" passes). Keys shared by two candidate
  entities, function words, names shorter than `min_chars` and names without a letter are dropped. The counts come from a
  token trie (`KeyTrie`: lowercase alphanumeric runs and single punctuation marks, as `mesh_novel.mention_key`), every
  occurrence counted, with the documents' evaluation bucket (sha256 of the FineWeb document id) kept apart.
- **Frames**: one edge per (property, value) for the properties of `PROPERTIES` in their order (instance of first), at most
  `max_values` values per property (most-linked fillers first), at most `max_degree` edges. Fillers are Wikidata items from
  a bounded dictionary (the `max_atomics` most-used fillers); an edge whose filler is outside it is dropped and counted.

T8 concepts are the selected entities (benchmark subjects and objects plus the fillers' entities, at most
`max_concepts`, entities mentioned in the FineWeb-Edu text first); the fillers are atomics (`wd:Q…`).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .mesh import FUNCTION_WORDS
from .wordnet import FrameOntology

USER_AGENT = "VSA-LLM-research/0.1 (T8 Wikidata track; https://github.com/bing1100/VSA-LLM; bingxuhu@gmail.com)"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
API_ENDPOINT = "https://www.wikidata.org/w/api.php"
ENTITY_PREFIX = "http://www.wikidata.org/entity/"
QID = re.compile(r"^Q[1-9][0-9]*$")
TOKEN = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]")
END = ""                       # trie terminal (never a token)

# The fixed property list (order = frame order and relation index). Each property is used by at least one benchmark
# relation (column 3; BEAR = its 60 relations, LRE = the factual relations, PQ = PopQA, 2H = TwoHopFact) or types the
# entity (P31, P279); `t8_wikidata_corpus.property_coverage` reports the benchmark relations each one serves.
PROPERTIES: list[tuple[str, str, str]] = [
    ("P31", "instance_of", "type of every entity"),
    ("P279", "subclass_of", "type of class entities (fillers)"),
    ("P106", "occupation", "LRE person_occupation; PQ occupation"),
    ("P27", "country_of_citizenship", "BEAR P27; 2H president-cntry"),
    ("P19", "place_of_birth", "BEAR P19; PQ place of birth; 2H *-birthcity"),
    ("P20", "place_of_death", "BEAR P20"),
    ("P17", "country", "LRE city_in_country, landmark_in_country; PQ country; 2H *-cntry"),
    ("P131", "located_in_admin", "BEAR P131"),
    ("P30", "continent", "BEAR P30; LRE landmark_on_continent"),
    ("P36", "capital", "BEAR P36; LRE country_capital_city; PQ capital; 2H cntry-capital"),
    ("P1376", "capital_of", "BEAR P1376; PQ capital of; 2H capital-cntry"),
    ("P37", "official_language", "BEAR P37; LRE country_language"),
    ("P38", "currency", "LRE country_currency"),
    ("P103", "native_language", "BEAR P103; LRE person_native_language"),
    ("P1412", "languages_spoken", "BEAR P1412"),
    ("P50", "author", "BEAR P50; PQ author; 2H novel-author"),
    ("P57", "director", "BEAR P57; PQ director; 2H movie-director"),
    ("P58", "screenwriter", "BEAR P58; PQ screenwriter"),
    ("P162", "producer", "BEAR P162; PQ producer"),
    ("P86", "composer", "PQ composer"),
    ("P175", "performer", "BEAR P175; 2H song-singer"),
    ("P495", "country_of_origin", "BEAR P495; LRE food_from_country; 2H movie-origcntry"),
    ("P136", "genre", "PQ genre"),
    ("P361", "part_of", "LRE person_band_lead_singer (band membership)"),
    ("P527", "has_part", "inverse of part of"),
    ("P749", "parent_organization", "organizations"),
    ("P159", "headquarters_location", "LRE company_hq; 2H *-hqcity"),
    ("P108", "employer", "BEAR P108"),
    ("P69", "educated_at", "BEAR P69; LRE person_university; 2H *-uguniv"),
    ("P463", "member_of", "BEAR P463; LRE person_band_lead_singer"),
    ("P364", "original_language", "BEAR P364"),
    ("P413", "position_played", "BEAR P413; LRE person_plays_position_in_sport"),
    ("P54", "member_of_sports_team", "athletes"),
    ("P118", "league", "teams and athletes"),
    ("P641", "sport", "BEAR P641; LRE person_plays_pro_sport; PQ sport"),
    ("P26", "spouse", "BEAR P26; 2H *-spouse"),
    ("P22", "father", "LRE person_father; PQ father; 2H *-father"),
    ("P25", "mother", "LRE person_mother; PQ mother; 2H *-mother"),
    ("P40", "child", "inverse of father / mother"),
    ("P3373", "sibling", "BEAR P3373"),
    ("P112", "founded_by", "2H *-founder"),
    ("P169", "chief_executive_officer", "LRE company_ceo; 2H orgz-ceo"),
    ("P176", "manufacturer", "BEAR P176; LRE product_by_company"),
    ("P178", "developer", "BEAR P178; LRE product_by_company; 2H vdgame-dev"),
    ("P1303", "instrument", "BEAR P1303; LRE person_plays_instrument"),
    ("P59", "constellation", "LRE star_constellation"),
    ("P140", "religion", "PQ religion"),
    ("P35", "head_of_state", "2H cntry-president"),
    ("P6", "head_of_government", "BEAR P6"),
    ("P414", "stock_exchange", "2H *-stockexch"),
    ("P85", "anthem", "2H cntry-anthem"),
    ("P674", "characters", "2H novel-mainchar, movie-mainchar"),
    ("P1441", "present_in_work", "BEAR P1441; 2H mainchar-novel, mainchar-movie"),
    ("P170", "creator", "BEAR P170; 2H mainchar-creator"),
    ("P127", "owned_by", "BEAR P127"),
    ("P137", "operator", "BEAR P137"),
    ("P264", "record_label", "musicians"),
    ("P272", "production_company", "BEAR P272"),
    ("P449", "original_broadcaster", "BEAR P449"),
    ("P179", "part_of_the_series", "BEAR P179"),
    ("P171", "parent_taxon", "BEAR P171"),
    ("P105", "taxon_rank", "BEAR P105"),
    ("P177", "crosses", "BEAR P177"),
    ("P206", "next_to_body_of_water", "BEAR P206"),
    ("P403", "mouth_of_the_watercourse", "BEAR P403"),
    ("P4552", "mountain_range", "BEAR P4552"),
    ("P610", "highest_point", "BEAR P610"),
    ("P115", "home_venue", "BEAR P115"),
    ("P1532", "country_for_sport", "BEAR P1532"),
    ("P2632", "place_of_detention", "BEAR P2632"),
    ("P291", "place_of_publication", "BEAR P291"),
    ("P344", "director_of_photography", "BEAR P344"),
    ("P412", "voice_type", "BEAR P412"),
    ("P427", "taxonomic_type", "BEAR P427"),
    ("P466", "occupant", "BEAR P466"),
    ("P509", "cause_of_death", "BEAR P509"),
    ("P53", "family", "BEAR P53"),
    ("P611", "religious_order", "BEAR P611"),
    ("P676", "lyrics_by", "BEAR P676"),
    ("P6886", "writing_language", "BEAR P6886"),
    ("P7937", "form_of_creative_work", "BEAR P7937"),
    ("P7959", "historic_county", "BEAR P7959"),
    ("P87", "librettist", "BEAR P87"),
    ("P98", "editor", "BEAR P98"),
    ("P185", "doctoral_student", "BEAR P185"),
    ("P190", "twinned_administrative_body", "BEAR P190"),
    ("P462", "color", "PQ color"),
    ("P812", "academic_major", "2H author-ugmajor"),
]
PROPERTY_NAME = {pid: name for pid, name, _ in PROPERTIES}
PROPERTY_ID = {name: pid for pid, name, _ in PROPERTIES}
# Wording of the E9 dimension-3 items on T8 (`e9_tracks.track_lexicon`): two property prompts and a statement per relation
# (the statement's prefix is the held-out wording, so it never equals a prompt). `ARTICLE_RELATIONS` take "a"/"an" before
# the filler ("{x} is" → " a city").
_T = {
    "instance_of": (["{x} is", "In Wikidata, {x} is recorded as"], "Wikidata lists {x} as {y}."),
    "subclass_of": (["{x} is a kind of", "{x} is a type of"], "{x} is a subclass of {y}."),
    "occupation": (["{x} works as", "By profession, {x} is"], "{x} earns a living as {y}."),
    "country_of_citizenship": (["{x} is a citizen of", "{x} holds the citizenship of"], "{x} has the nationality of {y}."),
    "place_of_birth": (["{x} was born in", "The birthplace of {x} is"], "{x} was born and raised in {y}."),
    "place_of_death": (["{x} died in", "The place where {x} died is"], "{x} passed away in {y}."),
    "country": (["{x} is located in the country of", "{x} is in"], "{x} can be found in {y}."),
    "located_in_admin": (["{x} is located in", "{x} lies within"], "{x} is an area within {y}."),
    "continent": (["{x} is on the continent of", "{x} is located on"], "{x} is part of {y}."),
    "capital": (["The capital of {x} is", "{x} has its capital in"], "{x} is governed from {y}."),
    "capital_of": (["{x} is the capital of", "{x} serves as the capital of"], "{x} is the seat of government of {y}."),
    "official_language": (["The official language of {x} is", "In {x}, the official language is"], "{x} uses the official language {y}."),
    "currency": (["The currency of {x} is the", "In {x}, people pay with the"], "{x} uses the currency {y}."),
    "native_language": (["The native language of {x} is", "{x}'s mother tongue is"], "{x} is a native speaker of {y}."),
    "languages_spoken": (["{x} speaks", "{x} can speak"], "{x} is fluent in {y}."),
    "author": (["{x} was written by", "The author of {x} is"], "{x} is a work by {y}."),
    "director": (["{x} was directed by", "The director of {x} is"], "{x} is a film by {y}."),
    "screenwriter": (["{x} was written for the screen by", "The screenwriter of {x} is"], "{x} has a screenplay by {y}."),
    "producer": (["{x} was produced by", "The producer of {x} is"], "{x} is a production of {y}."),
    "composer": (["The music of {x} was composed by", "The composer of {x} is"], "{x} has music by {y}."),
    "performer": (["{x} is performed by", "The performer of {x} is"], "{x} is a recording by {y}."),
    "country_of_origin": (["{x} originates from", "The country of origin of {x} is"], "{x} comes from {y}."),
    "genre": (["The genre of {x} is", "{x} belongs to the genre of"], "{x} is a work of {y}."),
    "part_of": (["{x} is part of", "{x} belongs to"], "{x} is one component of {y}."),
    "has_part": (["{x} includes", "One part of {x} is"], "{x} has {y} as a part."),
    "parent_organization": (["The parent organization of {x} is", "{x} is a subsidiary of"], "{x} is controlled by its parent {y}."),
    "headquarters_location": (["{x} is headquartered in", "The headquarters of {x} are in"], "{x} has its main offices in {y}."),
    "employer": (["{x} works for", "The employer of {x} is"], "{x} is employed by {y}."),
    "educated_at": (["{x} was educated at", "{x} studied at"], "{x} is an alumnus of {y}."),
    "member_of": (["{x} is a member of", "{x} belongs to"], "{x} has membership in {y}."),
    "original_language": (["The original language of {x} is", "{x} was originally made in"], "{x} was first released in {y}."),
    "position_played": (["{x} plays in the position of", "On the field, {x} plays as"], "{x} is known for playing as {y}."),
    "member_of_sports_team": (["{x} played for", "{x} is a player of"], "{x} was on the roster of {y}."),
    "league": (["{x} plays in the league", "{x} competes in"], "{x} is a member of the league {y}."),
    "sport": (["{x} plays the sport of", "The sport of {x} is"], "{x} competes in {y}."),
    "spouse": (["{x} is married to", "The spouse of {x} is"], "{x} has been the partner in marriage of {y}."),
    "father": (["The father of {x} is", "{x}'s father is"], "{x} is a son or daughter of {y}."),
    "mother": (["The mother of {x} is", "{x}'s mother is"], "{x} was born to {y}."),
    "child": (["{x} is the parent of", "A child of {x} is"], "{x} has a child named {y}."),
    "sibling": (["{x} is the sibling of", "A sibling of {x} is"], "{x} grew up with the sibling {y}."),
    "founded_by": (["{x} was founded by", "The founder of {x} is"], "{x} owes its founding to {y}."),
    "chief_executive_officer": (["The CEO of {x} is", "{x} is led by its chief executive"], "{x} has {y} as its chief executive officer."),
    "manufacturer": (["{x} is manufactured by", "{x} is made by"], "{x} is a product of {y}."),
    "developer": (["{x} was developed by", "The developer of {x} is"], "{x} is software from {y}."),
    "instrument": (["{x} plays the", "The instrument of {x} is the"], "{x} is a player of the {y}."),
    "constellation": (["{x} is part of the constellation", "{x} lies in the constellation"], "{x} is a star of {y}."),
    "religion": (["The religion of {x} is", "{x} follows the religion of"], "{x} is an adherent of {y}."),
    "head_of_state": (["The head of state of {x} is", "{x} is headed by"], "{x} has {y} as its head of state."),
    "head_of_government": (["The head of government of {x} is", "{x} is governed by"], "{x} has {y} as its head of government."),
    "stock_exchange": (["{x} is listed on the", "{x} trades on the"], "{x} is a company listed on {y}."),
    "anthem": (["The national anthem of {x} is", "The anthem of {x} is"], "{x} has the anthem {y}."),
    "characters": (["{x} features the character", "A character in {x} is"], "{x} has a character called {y}."),
    "present_in_work": (["{x} appears in", "{x} is a character in"], "{x} is featured in {y}."),
    "creator": (["{x} was created by", "The creator of {x} is"], "{x} is a creation of {y}."),
    "owned_by": (["{x} is owned by", "The owner of {x} is"], "{x} is property of {y}."),
    "operator": (["{x} is operated by", "The operator of {x} is"], "{x} is run by {y}."),
    "record_label": (["{x} is signed to the record label", "{x} records for"], "{x} has released records on {y}."),
    "production_company": (["{x} was produced by the company", "The production company of {x} is"], "{x} is a production of {y}."),
    "original_broadcaster": (["{x} was originally broadcast on", "{x} first aired on"], "{x} was shown on {y}."),
    "part_of_the_series": (["{x} is part of the series", "{x} belongs to the series"], "{x} is an installment of {y}."),
    "parent_taxon": (["The parent taxon of {x} is", "{x} belongs to the taxon"], "{x} is classified under {y}."),
    "taxon_rank": (["The taxon rank of {x} is", "{x} is ranked as"], "{x} is classified at the rank of {y}."),
    "crosses": (["{x} crosses", "{x} spans"], "{x} goes over {y}."),
    "next_to_body_of_water": (["{x} is located next to", "{x} lies on the shore of"], "{x} is near {y}."),
    "mouth_of_the_watercourse": (["{x} flows into", "The mouth of {x} is in"], "{x} empties into {y}."),
    "mountain_range": (["{x} is part of the mountain range", "{x} lies in the"], "{x} is a peak of {y}."),
    "highest_point": (["The highest point of {x} is", "The tallest peak in {x} is"], "{x} reaches its highest point at {y}."),
    "home_venue": (["The home venue of {x} is", "{x} plays its home games at"], "{x} is based at {y}."),
    "country_for_sport": (["{x} represents the country", "In sport, {x} competes for"], "{x} plays for {y}."),
    "place_of_detention": (["{x} was detained in", "{x} was imprisoned in"], "{x} was held at {y}."),
    "place_of_publication": (["{x} was published in", "The place of publication of {x} is"], "{x} first appeared in {y}."),
    "director_of_photography": (["The cinematographer of {x} is", "{x} was shot by"], "{x} was filmed by {y}."),
    "voice_type": (["The voice type of {x} is", "{x} sings as"], "{x} has the voice of {y}."),
    "taxonomic_type": (["The type species of {x} is", "The taxonomic type of {x} is"], "{x} is typified by {y}."),
    "occupant": (["{x} is used by", "The occupant of {x} is"], "{x} is home to {y}."),
    "cause_of_death": (["{x} died of", "The cause of death of {x} was"], "{x} was killed by {y}."),
    "family": (["{x} belongs to the family", "{x} is a member of the family"], "{x} is classified in {y}."),
    "religious_order": (["{x} belongs to the religious order", "{x} was a member of the order"], "{x} took vows in {y}."),
    "lyrics_by": (["The lyrics of {x} were written by", "The lyricist of {x} is"], "{x} has words by {y}."),
    "writing_language": (["{x} writes in", "The writing language of {x} is"], "{x} is a writer in {y}."),
    "form_of_creative_work": (["{x} is a work in the form of", "The form of {x} is"], "{x} takes the form of {y}."),
    "historic_county": (["{x} is in the historic county of", "Historically, {x} belongs to"], "{x} lies in {y}."),
    "librettist": (["The libretto of {x} was written by", "The librettist of {x} is"], "{x} has a libretto by {y}."),
    "editor": (["{x} was edited by", "The editor of {x} is"], "{x} is edited by {y}."),
    "doctoral_student": (["{x} supervised the doctoral student", "A doctoral student of {x} was"], "{x} was the doctoral advisor of {y}."),
    "twinned_administrative_body": (["{x} is twinned with", "The sister city of {x} is"], "{x} has a partnership with {y}."),
    "color": (["The color of {x} is", "{x} is colored"], "{x} has the color {y}."),
    "academic_major": (["{x} majored in", "The academic major of {x} was"], "{x} studied {y}."),
}
ARTICLE_RELATIONS = frozenset({"instance_of", "occupation", "position_played", "taxon_rank", "voice_type", "form_of_creative_work"})


def relation_templates() -> dict[str, Any]:
    """`RelationTemplates` per T8 relation (imported lazily: `tracks.common`)."""
    from ..tracks.common import RelationTemplates
    return {name: RelationTemplates(list(prompts), statement) for name, (prompts, statement) in _T.items()}


@contextlib.contextmanager
def gzip_writer(path: Path) -> Iterator[Any]:
    """A gzip file with reproducible bytes (no file name, mtime 0 in the header)."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
        yield handle


def qid_number(qid: str) -> int:
    return int(qid[1:]) if QID.match(qid) else 1 << 62


def entity_key(text: str) -> str:
    """The matcher's normal form of a name: lowercase alphanumeric runs and single punctuation marks, space-joined."""
    return " ".join(token.lower() for token in TOKEN.findall(text))


def doc_bucket(document_id: str) -> int:
    """Evaluation bucket of a FineWeb document: sha256 of its id, first 8 hex digits, mod 10,000."""
    return int(hashlib.sha256(document_id.encode()).hexdigest()[:8], 16) % 10_000


# -- the Wikidata client --------------------------------------------------------------------------------------------

class WikidataClient:
    """SPARQL and `wbsearchentities` with a response cache (gzip JSON, one file per request, keyed by the sha256 of the
    request) and polite pacing: one request at a time, at most one per `min_interval` (SPARQL) / `api_interval` (API)
    seconds, `Retry-After` and exponential back-off on 429 / 5xx. API reads are sent without `maxlag` (it is meant for
    edits, and the query-service lag it includes blocks every read while WDQS lags). `offline=True` reads the cache only."""

    def __init__(self, cache_dir: Path, *, user_agent: str = USER_AGENT, min_interval: float = 1.0,
                 api_interval: float = 0.25, session: Any = None, offline: bool = False, log: Any = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.user_agent, self.min_interval, self.api_interval = user_agent, float(min_interval), float(api_interval)
        self.offline, self._session, self._last = offline, session, 0.0
        self.log = log or (lambda message: None)
        self.stats: Counter = Counter()

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests
            self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})
        return self._session

    def _path(self, kind: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / kind / digest[:2] / f"{digest}.json.gz"

    @staticmethod
    def _read(path: Path) -> Any:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with gzip_writer(partial) as handle:
            handle.write(json.dumps(payload, sort_keys=True).encode())
        partial.rename(path)

    def _request(self, method: str, url: str, *, interval: float, **kwargs: Any) -> Any:
        delay = 5.0
        for attempt in range(10):
            wait = self._last + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                response = self.session.request(method, url, timeout=180, **kwargs)
            except Exception as error:                        # network hiccup: back off and retry
                self.log(f"wikidata: {type(error).__name__} ({error}); retry in {delay:.0f} s")
                self.stats["network_retries"] += 1
                time.sleep(delay); delay = min(delay * 2, 300.0)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                retry = response.headers.get("Retry-After")
                pause = float(retry) if retry and retry.isdigit() else delay
                self.log(f"wikidata: HTTP {response.status_code}; retry in {pause:.0f} s")
                self.stats[f"http_{response.status_code}"] += 1
                time.sleep(pause); delay = min(delay * 2, 300.0)
                continue
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError:                                # a response cut off by the service's time limit
                self.stats["truncated"] += 1
                if attempt >= 2:
                    raise RuntimeError(f"wikidata: truncated responses from {url}")
                time.sleep(delay); delay = min(delay * 2, 300.0)
                continue
            if isinstance(payload, dict) and payload.get("error", {}).get("code") == "maxlag":
                self.stats["maxlag"] += 1
                time.sleep(float(response.headers.get("Retry-After", 5)))
                continue
            return payload
        raise RuntimeError(f"wikidata: {url} failed after 10 attempts")

    def sparql(self, query: str, *, kind: str = "sparql") -> dict[str, Any]:
        """A SPARQL SELECT (POST, JSON results), cached under `<cache>/sparql/<kind>/`."""
        path = self._path(f"sparql/{kind}", query)
        if path.exists():
            self.stats["sparql_cached"] += 1
            return self._read(path)
        if self.offline:
            raise FileNotFoundError(f"offline: {path} not cached")
        payload = self._request("POST", SPARQL_ENDPOINT, interval=self.min_interval, data={"query": query},
                                headers={"Accept": "application/sparql-results+json"})
        self._write(path, {"query": query, "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "response": payload})
        self.stats["sparql_fetched"] += 1
        return {"query": query, "response": payload}

    def search(self, name: str, *, limit: int = 5, language: str = "en") -> list[dict[str, Any]]:
        """`wbsearchentities` hits for a name (items, English), cached under `<cache>/api/wbsearchentities/`."""
        params = {"action": "wbsearchentities", "search": name, "language": language, "uselang": language, "type": "item",
                  "limit": int(limit), "format": "json"}
        key = json.dumps(params, sort_keys=True)
        path = self._path("api/wbsearchentities", key)
        if path.exists():
            self.stats["search_cached"] += 1
            return self._read(path)["response"].get("search", [])
        if self.offline:
            raise FileNotFoundError(f"offline: {path} not cached")
        payload = self._request("GET", API_ENDPOINT, interval=self.api_interval, params=params)
        self._write(path, {"params": params, "retrieved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "response": payload})
        self.stats["search_fetched"] += 1
        return payload.get("search", [])


def _values(qids: Sequence[str]) -> str:
    return " ".join(f"wd:{q}" for q in qids)


def _bindings(result: dict[str, Any]) -> list[dict[str, str]]:
    return [{k: v["value"] for k, v in row.items()} for row in result["response"]["results"]["bindings"]]


def _qid(uri: str) -> str:
    return uri[len(ENTITY_PREFIX):] if uri.startswith(ENTITY_PREFIX) else uri


def _batches(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def terms_query(qids: Sequence[str]) -> str:
    return ("SELECT ?item ?label ?description ?sitelinks ?article WHERE {\n"
            f"  VALUES ?item {{ {_values(qids)} }}\n"
            "  OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label) = \"en\") }\n"
            "  OPTIONAL { ?item schema:description ?description . FILTER(LANG(?description) = \"en\") }\n"
            "  OPTIONAL { ?item wikibase:sitelinks ?sitelinks . }\n"
            "  OPTIONAL { ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }\n"
            "}")


def aliases_query(qids: Sequence[str]) -> str:
    return ("SELECT ?item ?alias WHERE {\n"
            f"  VALUES ?item {{ {_values(qids)} }}\n"
            "  ?item skos:altLabel ?alias . FILTER(LANG(?alias) = \"en\")\n"
            "}")


def claims_query(qids: Sequence[str], properties: Sequence[str]) -> str:
    props = " ".join(f"wdt:{p}" for p in properties)
    return ("SELECT ?item ?p ?value WHERE {\n"
            f"  VALUES ?item {{ {_values(qids)} }}\n"
            f"  VALUES ?p {{ {props} }}\n"
            "  ?item ?p ?value .\n"
            "  FILTER(STRSTARTS(STR(?value), \"http://www.wikidata.org/entity/Q\"))\n"
            "}")


def redirects_query(qids: Sequence[str]) -> str:
    return ("SELECT ?item ?target WHERE {\n"
            f"  VALUES ?item {{ {_values(qids)} }}\n"
            "  ?item owl:sameAs ?target .\n"
            "}")


class EntityStore:
    """Per-entity results of batched queries (`<cache>/store/<kind>.jsonl`: one `{"qid", "data", "batch"}` line per
    entity, appended after each batch; `batch` = the sha256 of the raw response's query, whose gzip file is the
    provenance). Later calls query only the entities not stored yet, so differently composed entity sets reuse earlier
    batches. One writer per kind at a time."""

    def __init__(self, client: WikidataClient, kind: str, *, keys_are_qids: bool = True) -> None:
        self.client, self.kind, self.keys_are_qids = client, kind, keys_are_qids
        self.path = client.cache_dir / "store" / f"{kind}.jsonl"
        self.data: dict[str, Any] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:          # a torn last line of an interrupted run: refetched
                        continue
                    self.data[row["qid"]] = row["data"]

    def fetch(self, qids: Iterable[str], *, batch: int, query: Any, parse: Any, empty: Any,
              progress: Any = None) -> dict[str, Any]:
        wanted = (sorted({q for q in qids if QID.match(q)}, key=qid_number) if self.keys_are_qids
                  else sorted({q for q in qids if q}))
        todo = [q for q in wanted if q not in self.data]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for index, chunk in enumerate(_batches(todo, batch)):
            self._fetch_chunk(chunk, query=query, parse=parse, empty=empty)
            if progress is not None:
                progress(f"{self.kind}: batch {index + 1} of {(len(todo) + batch - 1) // batch} ({len(chunk)} entities)")
        return {q: self.data[q] for q in wanted}

    def _fetch_chunk(self, chunk: list[str], *, query: Any, parse: Any, empty: Any) -> None:
        """One batch; a batch whose response fails (truncated by the service's time limit) is split in halves."""
        text = query(chunk)
        try:
            result = self.client.sparql(text, kind=self.kind)
        except RuntimeError:
            if len(chunk) == 1:
                raise
            self.client.stats["split_batches"] += 1
            middle = len(chunk) // 2
            self._fetch_chunk(chunk[:middle], query=query, parse=parse, empty=empty)
            self._fetch_chunk(chunk[middle:], query=query, parse=parse, empty=empty)
            return
        parsed = {q: empty() for q in chunk}
        parse(_bindings(result), parsed)
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self.path.open("a") as handle:
            for q in chunk:
                handle.write(json.dumps({"qid": q, "data": parsed[q], "batch": digest}, sort_keys=True, ensure_ascii=False) + "\n")
        self.data.update(parsed)


def _parse_terms(rows: list[dict[str, str]], out: dict[str, Any]) -> None:
    for row in rows:
        record = out[_qid(row["item"])]
        if row.get("label"):
            record["label"] = row["label"]
        if row.get("description"):
            record["description"] = row["description"]
        if row.get("sitelinks"):
            record["sitelinks"] = int(row["sitelinks"])
        if row.get("article"):
            record["enwiki"] = row["article"].rsplit("/", 1)[-1]


def _parse_aliases(rows: list[dict[str, str]], out: dict[str, Any]) -> None:
    for row in rows:
        out[_qid(row["item"])].append(row["alias"])
    for q in out:
        out[q] = sorted(set(out[q]))


def _parse_claims(rows: list[dict[str, str]], out: dict[str, Any]) -> None:
    for row in rows:
        value = _qid(row["value"])
        if QID.match(value):
            out[_qid(row["item"])].setdefault(row["p"].rsplit("/", 1)[-1], []).append(value)
    for q, claims in out.items():
        out[q] = {p: sorted(set(v), key=qid_number) for p, v in sorted(claims.items())}


def fetch_terms(client: WikidataClient, qids: Iterable[str], *, batch: int = 2000, progress: Any = None
                ) -> dict[str, dict[str, Any]]:
    """qid → {label, description, aliases, sitelinks, enwiki} (English; None / [] / 0 where Wikidata has none)."""
    qids = list(qids)
    terms = EntityStore(client, "terms").fetch(
        qids, batch=batch, query=terms_query, parse=_parse_terms, progress=progress,
        empty=lambda: {"label": None, "description": None, "sitelinks": 0, "enwiki": None})
    aliases = EntityStore(client, "aliases").fetch(qids, batch=batch, query=aliases_query, parse=_parse_aliases,
                                                   empty=list, progress=progress)
    return {q: {**terms[q], "aliases": aliases[q]} for q in terms}


def fetch_claims(client: WikidataClient, qids: Iterable[str], properties: Sequence[str], *, batch: int = 1000,
                 kind: str = "claims", progress: Any = None) -> dict[str, dict[str, list[str]]]:
    """qid → {property: sorted item values} (truthy statements, item values only). The store `kind` must always be
    queried with the same property list."""
    return EntityStore(client, kind).fetch(qids, batch=batch, query=lambda chunk: claims_query(chunk, properties),
                                           parse=_parse_claims, empty=dict, progress=progress)


def homonyms_query(labels: Sequence[str]) -> str:
    values = " ".join(json.dumps(label, ensure_ascii=False) + "@en" for label in labels)
    return ("SELECT ?label ?item ?sitelinks WHERE {\n"
            f"  VALUES ?label {{ {values} }}\n"
            "  ?item rdfs:label ?label ; wikibase:sitelinks ?sitelinks .\n"
            "  FILTER(?sitelinks >= 3)\n"
            "}")


def fetch_homonyms(client: WikidataClient, labels: Iterable[str], *, batch: int = 500, progress: Any = None
                   ) -> dict[str, list[list[Any]]]:
    """English label → [[item, sitelinks], …] of every item with exactly that label and ≥ 3 sitelinks."""
    def parse(rows: list[dict[str, str]], out: dict[str, Any]) -> None:
        for row in rows:
            if row["label"] in out:
                out[row["label"]].append([_qid(row["item"]), int(row["sitelinks"])])
        for label in out:
            out[label] = sorted(out[label], key=lambda pair: (-pair[1], qid_number(pair[0])))
    return EntityStore(client, "homonyms", keys_are_qids=False).fetch(labels, batch=batch, query=homonyms_query, parse=parse,
                                                                      empty=list, progress=progress)


def fetch_redirects(client: WikidataClient, qids: Iterable[str], *, batch: int = 2000, progress: Any = None) -> dict[str, str]:
    """Redirected (merged) QIDs → their targets."""
    def parse(rows: list[dict[str, str]], out: dict[str, Any]) -> None:
        for row in rows:
            out[_qid(row["item"])] = _qid(row["target"])
    found = EntityStore(client, "redirects").fetch(qids, batch=batch, query=redirects_query, parse=parse, empty=lambda: None,
                                                   progress=progress)
    return {q: t for q, t in found.items() if t}


# -- records snapshot -----------------------------------------------------------------------------------------------

def write_records(path: Path, records: dict[str, dict[str, Any]]) -> str:
    """`records.jsonl.gz` (one entity per line, sorted by QID number; gzip mtime 0) → its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip_writer(path) as handle:
        for qid in sorted(records, key=qid_number):
            handle.write((json.dumps({"qid": qid, **records[qid]}, sort_keys=True, ensure_ascii=False) + "\n").encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path, *, expected_sha256: str | None = None) -> dict[str, dict[str, Any]]:
    data = Path(path).expanduser().read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{path}: sha256 {digest} != pinned {expected_sha256} (the frozen Wikidata snapshot changed)")
    records = {}
    for line in gzip.decompress(data).decode().splitlines():
        if line.strip():
            row = json.loads(line)
            records[row.pop("qid")] = row
    return records


# -- mention counting over FineWeb-Edu --------------------------------------------------------------------------------

def capital_positions(name: str) -> tuple[int, ...]:
    """Token positions (`TOKEN` tokens) of a name that start with an uppercase letter."""
    return tuple(i for i, token in enumerate(TOKEN.findall(name)) if token[:1].isupper())


class KeyTrie:
    """Whole-token occurrences of a fixed list of keys (`entity_key` forms; key id = list index), every occurrence counted
    (overlapping and nested matches included), with whether the occurrence is *cased*: written with the name's capitals
    (`masks[key]`: the token positions every name of that key capitalizes; "the United States" needs "United" and
    "States" capitalized, "The American" needs "The" too). Without a mask (or an empty one) any uppercase letter counts."""

    def __init__(self, keys: Sequence[str], masks: Sequence[tuple[int, ...] | None] | None = None) -> None:
        self.size = len(keys)
        self.masks = list(masks) if masks is not None else None
        self.root: dict[str, Any] = {}
        for index, key in enumerate(keys):
            if not key:
                continue
            node = self.root
            for token in key.split(" "):
                node = node.setdefault(token, {})
            node[END] = index

    def _cased(self, key: int, tokens: list[str], lowered: list[str], i: int, j: int) -> bool:
        mask = self.masks[key] if self.masks is not None else None
        if mask:
            return all(tokens[i + p][:1].isupper() for p in mask if i + p <= j)
        return any(tokens[k] != lowered[k] for k in range(i, j + 1))

    def matches(self, text: str) -> Iterator[tuple[int, bool]]:
        """(key id, cased) per occurrence."""
        tokens = TOKEN.findall(text)
        lowered = [t.lower() for t in tokens]
        root, n = self.root, len(lowered)
        for i in range(n):
            node = root.get(lowered[i])
            j = i
            while node is not None:
                found = node.get(END)
                if found is not None:
                    yield found, self._cased(found, tokens, lowered, i, j)
                j += 1
                if j >= n:
                    break
                node = node.get(lowered[j])

    def first(self, text: str) -> bool:
        """Whether `text` holds any key."""
        for _ in self.matches(text):
            return True
        return False

    def found(self, text: str) -> set[int]:
        return {key for key, _ in self.matches(text)}


@dataclass
class MentionCounts:
    """Per key and side (0 = training buckets, 1 = evaluation buckets): occurrences, all-lowercase occurrences and
    documents."""

    occurrences: np.ndarray
    lowercase: np.ndarray
    documents: np.ndarray
    read: np.ndarray                       # documents read per side

    @classmethod
    def zeros(cls, size: int) -> "MentionCounts":
        return cls(np.zeros((2, size), np.int64), np.zeros((2, size), np.int64), np.zeros((2, size), np.int64),
                   np.zeros(2, np.int64))

    def add(self, other: "MentionCounts") -> None:
        self.occurrences += other.occurrences; self.lowercase += other.lowercase
        self.documents += other.documents; self.read += other.read


_TRIE: KeyTrie | None = None


def _init_trie(keys: list[str], masks: list[tuple[int, ...] | None] | None = None) -> None:
    global _TRIE
    _TRIE = KeyTrie(keys, masks)


def _count_rows(job: tuple[str, int, int, int, int, int]) -> MentionCounts:
    """Rows [first, last) of one shard (`offset` = the shard's first row in stream order; rows before `skip` in stream
    order are skipped): count every key on the training (bucket ≥ `eval_buckets`) and evaluation side."""
    import pyarrow.parquet as pq
    path, first, last, offset, skip, eval_buckets = job
    counts = MentionCounts.zeros(_TRIE.size)
    parquet = pq.ParquetFile(path)
    position = 0
    for group in range(parquet.num_row_groups):
        rows = parquet.metadata.row_group(group).num_rows
        if position + rows > first and position < last:
            table = parquet.read_row_group(group, columns=["id", "text"])
            ids, texts = table.column(0).to_pylist(), table.column(1).to_pylist()
            for i, (doc_id, text) in enumerate(zip(ids, texts)):
                row = position + i
                if not first <= row < last or offset + row < skip:
                    continue
                side = 1 if doc_bucket(doc_id) < eval_buckets else 0
                counts.read[side] += 1
                seen: set[int] = set()
                for key, cased in _TRIE.matches(text):
                    counts.occurrences[side, key] += 1
                    if not cased:
                        counts.lowercase[side, key] += 1
                    seen.add(key)
                if seen:
                    counts.documents[side, list(seen)] += 1
        position += rows
        if position >= last:
            break
    return counts


def count_fineweb_mentions(shards: Sequence[str], keys: Sequence[str], *, skip: int, eval_buckets: int,
                           limit: int | None = None, workers: int = 4, rows_per_job: int = 20_000,
                           masks: Sequence[tuple[int, ...] | None] | None = None) -> MentionCounts:
    """Occurrences of `keys` in the concatenated shards (stream order), documents [skip, skip + limit), split by the
    documents' evaluation bucket; row-range jobs, summed in job order (deterministic). `lowercase` counts occurrences
    not written with the name's capitals (`masks`, see `KeyTrie`)."""
    import pyarrow.parquet as pq
    jobs: list[tuple[str, int, int, int, int, int]] = []
    offset = 0
    end = None if limit is None else skip + limit
    for shard in shards:
        rows = pq.ParquetFile(shard).metadata.num_rows
        lo = max(skip, offset) - offset
        hi = rows if end is None else min(end, offset + rows) - offset
        for first in range(lo, hi, rows_per_job):
            jobs.append((str(shard), first, min(hi, first + rows_per_job), offset, int(skip), int(eval_buckets)))
        offset += rows
        if end is not None and offset >= end:
            break
    total = MentionCounts.zeros(len(keys))
    masks = list(masks) if masks is not None else None
    if workers <= 1:
        _init_trie(list(keys), masks)
        for job in jobs:
            total.add(_count_rows(job))
        return total
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init_trie, initargs=(list(keys), masks)) as pool:
        for part in pool.map(_count_rows, jobs):
            total.add(part)
    return total


# -- aliases and the concept selection --------------------------------------------------------------------------------

@dataclass(frozen=True)
class WikidataAliasPolicy:
    """Which names of an entity link (defaults: the pre-registered T8 policy)."""

    min_chars: int = 4
    max_lowercase_share: float = 0.10       # of the training-side occurrences
    min_case_evidence: int = 5              # occurrences needed to judge the case; below it only the label is kept
    min_domain_mentions: int = 1            # an alias other than the label must occur at least this often
    max_aliases: int = 8                    # per entity, most frequent first (the label always first when it passes)
    min_alias_words: int = 2                # an alias other than the label needs this many words ("Christ", "USA" do not link)
    max_homonym_ratio: float = 0.5          # a label another item also has, with ≥ this share of its sitelinks, does not link
    function_words: frozenset[str] = field(default=FUNCTION_WORDS)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "WikidataAliasPolicy":
        config = dict(config or {})
        keys = {"min_chars", "max_lowercase_share", "min_case_evidence", "min_domain_mentions", "max_aliases", "min_alias_words",
                "max_homonym_ratio"}
        unknown = set(config) - keys
        if unknown:
            raise ValueError(f"unknown alias policy keys {sorted(unknown)}")
        return cls(**config)


def entity_names(record: dict[str, Any]) -> list[str]:
    """Label first, then aliases (whitespace collapsed, case-insensitive duplicates dropped)."""
    names, seen = [], set()
    for name in [record.get("label")] + list(record.get("aliases") or []):
        if not name:
            continue
        text = " ".join(str(name).split())
        if text and text.lower() not in seen:
            seen.add(text.lower()); names.append(text)
    return names


def candidate_keys(records: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, list[tuple[str, str]]], list[tuple[int, ...]]]:
    """(sorted keys of every name of every record, qid → [(name, key)], per key the capital positions all its names share)."""
    by_record = {qid: [(name, entity_key(name)) for name in entity_names(record)] for qid, record in records.items()}
    masks: dict[str, set[int]] = {}
    for names in by_record.values():
        for name, key in names:
            if key:
                positions = set(capital_positions(name))
                masks[key] = positions if key not in masks else masks[key] & positions
    keys = sorted(masks)
    return keys, by_record, [tuple(sorted(masks[k])) for k in keys]


def alias_decision(name: str, key: str, *, is_label: bool, owners: int, train: int, lowercase: int,
                   policy: WikidataAliasPolicy, sitelinks: int = 0, homonym_sitelinks: int = 0) -> str:
    """'keep' or the reason a name does not link. `homonym_sitelinks` = the most sitelinks of another Wikidata item with
    the same English label (labels only)."""
    if len(name) < policy.min_chars:
        return "short"
    if not re.search(r"[a-z]", key):
        return "no_letter"
    if name.lower() in policy.function_words:
        return "function_word"
    if not any(c.isupper() for c in name):
        return "no_capital"                # Wikidata writes common nouns in lowercase ("march", "human")
    if not is_label and len(re.findall(r"[A-Za-z0-9]+", name)) < policy.min_alias_words:
        return "single_word_alias"
    if owners > 1:
        return "shared"
    if is_label and homonym_sitelinks and homonym_sitelinks >= policy.max_homonym_ratio * max(sitelinks, 1):
        return "homonym"
    if train >= policy.min_case_evidence and lowercase > policy.max_lowercase_share * train:
        return "lowercase_usage"
    if not is_label:
        if train < policy.min_domain_mentions:
            return "unseen_alias"
        if train < policy.min_case_evidence:
            return "too_rare_to_judge"
    return "keep"


SELECTION_FIELDS = ("qid", "alias", "key", "label", "train", "eval", "train_documents", "lowercase", "sources")


def select_concepts(records: dict[str, dict[str, Any]], counts: dict[str, dict[str, int]], sources: dict[str, list[str]], *,
                    policy: WikidataAliasPolicy, max_concepts: int, min_entity_mentions: int, source_order: Sequence[str],
                    pool: Iterable[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The T8 concept selection (rows sorted by QID number, then alias).

    `counts[key]` = {"train", "eval", "train_documents", "lowercase"} from the FineWeb screen; `sources[qid]` = the benchmark
    roles of an entity (e.g. "bear:subject", "filler"); `pool` = the candidate QIDs (default: every record).
    1. Aliases per entity by `alias_decision` (keys shared by two candidates are dropped for both).
    2. An entity is a candidate if it keeps ≥ 1 alias; its mention count = training-side occurrences of its kept aliases.
    3. Entities with ≥ `min_entity_mentions` mentions first, then the rest, each in a round robin over `source_order`
       (an entity is listed under its first source in that order; within a source by mentions, then QID), until
       `max_concepts`."""
    pool = sorted(set(pool) if pool is not None else set(records), key=qid_number)
    names = {qid: [(n, entity_key(n)) for n in entity_names(records[qid])] for qid in pool}
    owners: Counter = Counter(key for qid in pool for key in {k for _, k in names[qid] if k})
    stats: Counter = Counter()
    kept: dict[str, list[dict[str, Any]]] = {}
    for qid in pool:
        label = records[qid].get("label")
        rows = []
        for name, key in names[qid]:
            c = counts.get(key, {})
            decision = alias_decision(name, key, is_label=(name == label), owners=owners[key], train=int(c.get("train", 0)),
                                      lowercase=int(c.get("lowercase", 0)), policy=policy,
                                      sitelinks=int(records[qid].get("sitelinks") or 0),
                                      homonym_sitelinks=int(records[qid].get("homonym_sitelinks") or 0))
            stats[f"alias_{decision}"] += 1
            if decision == "keep":
                rows.append({"qid": qid, "alias": name, "key": key, "label": label or "", "train": int(c.get("train", 0)),
                             "eval": int(c.get("eval", 0)), "train_documents": int(c.get("train_documents", 0)),
                             "lowercase": int(c.get("lowercase", 0)), "sources": ",".join(sources.get(qid, []))})
        rows.sort(key=lambda r: (r["alias"] != label, -r["train"], r["alias"]))
        if rows:
            kept[qid] = rows[:policy.max_aliases]
    mentions = {qid: sum(r["train"] for r in rows) for qid, rows in kept.items()}
    rank = {s: i for i, s in enumerate(source_order)}

    def primary(qid: str) -> str:
        roles = [s.split(":", 1)[0] if ":" in s else s for s in sources.get(qid, ["filler"])]
        known = [r for r in roles if r in rank]
        return min(known, key=rank.get) if known else source_order[-1]

    chosen: list[str] = []
    for frequent in (True, False):
        queues: dict[str, list[str]] = defaultdict(list)
        for qid in kept:
            if (mentions[qid] >= min_entity_mentions) == frequent:
                queues[primary(qid)].append(qid)
        for q in queues.values():
            q.sort(key=lambda x: (-mentions[x], qid_number(x)))
        cursors = {s: 0 for s in queues}
        while len(chosen) < max_concepts and any(cursors[s] < len(queues[s]) for s in queues):
            for s in source_order:
                if s in queues and cursors[s] < len(queues[s]) and len(chosen) < max_concepts:
                    chosen.append(queues[s][cursors[s]]); cursors[s] += 1
    selected = sorted(chosen, key=qid_number)
    rows = [r for qid in selected for r in sorted(kept[qid], key=lambda r: r["alias"])]
    stats.update({"candidates": len(pool), "entities_with_alias": len(kept),
                  "entities_frequent": sum(1 for q in kept if mentions[q] >= min_entity_mentions),
                  "selected": len(selected), "selected_frequent": sum(1 for q in selected if mentions[q] >= min_entity_mentions),
                  "selected_aliases": len(rows)})
    stats["selected_by_source"] = dict(Counter(primary(q) for q in selected))
    return rows, dict(stats)


def write_selection(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """The selection as TSV (rows as given) → its sha256."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SELECTION_FIELDS)
        for row in rows:
            writer.writerow([row[k] for k in SELECTION_FIELDS])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_selection(path: Path, *, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    import csv
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{path}: sha256 {digest} != pinned {expected_sha256} (the frozen selection changed)")
    reader = csv.DictReader(data.decode().splitlines(), delimiter="\t")
    return [{**row, **{k: int(row[k]) for k in ("train", "eval", "train_documents", "lowercase")}} for row in reader]


# -- the frame ontology ----------------------------------------------------------------------------------------------

def build_wikidata_ontology(records: dict[str, dict[str, Any]], selection: Sequence[dict[str, Any]], *,
                            properties: Sequence[str] | None = None, max_atomics: int = 8192, max_degree: int = 16,
                            max_values: int = 3, max_values_by_property: dict[str, int] | None = None,
                            source: dict[str, Any] | None = None) -> FrameOntology:
    """Concepts = the selected entities (sorted by QID number); aliases = their selected names; frames over Wikidata
    filler items (`wd:Q…` atomics: the `max_atomics` most-used fillers, ties by QID number). Within a property, values
    are ordered by the filler's sitelink count (then QID) and at most `max_values` (per property:
    `max_values_by_property`) are kept; values without an English label (not verbalizable) are skipped, and edges whose
    filler is outside the dictionary are dropped."""
    properties = list(properties or [pid for pid, _, _ in PROPERTIES])
    unknown = [p for p in properties if p not in PROPERTY_NAME]
    if unknown:
        raise ValueError(f"properties without a T8 relation name: {unknown}")
    limits = {p: int((max_values_by_property or {}).get(p, max_values)) for p in properties}
    chosen = sorted({row["qid"] for row in selection}, key=qid_number)
    missing = [q for q in chosen if q not in records]
    if missing:
        raise ValueError(f"selected entities without a record: {missing[:5]}")

    def popularity(qid: str) -> tuple[int, int]:
        return (-int((records.get(qid) or {}).get("sitelinks") or 0), qid_number(qid))

    raw: dict[str, list[tuple[str, str]]] = {}
    unlabeled = 0
    for qid in chosen:
        claims = records[qid].get("claims") or {}
        edges = []
        for pid in properties:
            values = {v for v in claims.get(pid, []) if v != qid}
            labeled = {v for v in values if (records.get(v) or {}).get("label")}   # a filler must be verbalizable
            unlabeled += len(values) - len(labeled)
            edges += [(pid, v) for v in sorted(labeled, key=popularity)[:limits[pid]]]
        raw[qid] = edges
    use: Counter[str] = Counter(v for edges in raw.values() for _, v in edges)
    fillers = [q for q, _ in sorted(use.items(), key=lambda kv: (-kv[1], qid_number(kv[0])))[:max_atomics]]
    atomic_names = [f"wd:{q}" for q in fillers]
    atomic_index = {q: i for i, q in enumerate(fillers)}
    relation_names = [PROPERTY_NAME[p] for p in properties]
    relation_index = {p: i for i, p in enumerate(properties)}
    frames: list[list[tuple[int, int]]] = []
    dropped = Counter()
    for qid in chosen:
        frame: list[tuple[int, int]] = []
        for pid, value in raw[qid]:
            if value not in atomic_index:
                dropped["filler_outside_dictionary"] += 1
                continue
            if len(frame) >= max_degree:
                dropped["over_max_degree"] += 1
                continue
            frame.append((relation_index[pid], atomic_index[value]))
        frames.append(frame)
    concept = {q: i for i, q in enumerate(chosen)}
    alias_pairs = [(row["alias"], concept[row["qid"]]) for row in selection]
    ontology = FrameOntology("wikidata", chosen, relation_names, atomic_names, frames, alias_pairs,
                             {"source": dict(source or {}), "records": len(chosen)})
    def kind(qid: str) -> str | None:
        types = sorted((records.get(qid) or {}).get("claims", {}).get("P31", []), key=popularity)
        return types[0] if types else None

    ontology.metadata.update({
        "headings": [records[q].get("label") or q for q in chosen],
        "descriptions": [records[q].get("description") or "" for q in chosen],
        "sitelinks": [int(records[q].get("sitelinks") or 0) for q in chosen],
        "filler_headings": {q: (records.get(q) or {}).get("label") or q for q in fillers},
        "filler_types": {q: kind(q) for q in fillers},
        "property_ids": dict(zip(relation_names, properties)),
        "max_atomics": max_atomics, "max_degree": max_degree, "max_values": max_values,
        "empty_frames": sum(1 for f in frames if not f), "dropped_edges": {**dict(dropped), "filler_without_label": unlabeled},
        "edges": sum(len(f) for f in frames), "raw_edges": sum(len(e) for e in raw.values()),
    })
    return ontology


def verbalize_frame(frame: Sequence[tuple[int, int]], relation_names: Sequence[str], atomic_names: Sequence[str],
                    filler_headings: dict[str, str]) -> list[list[str]]:
    """[[relation name, filler label], …] of a frame."""
    out = []
    for relation, atom in frame:
        name = atomic_names[atom]
        qid = name.split(":", 1)[1] if name.startswith("wd:") else name
        out.append([relation_names[relation], filler_headings.get(qid, qid)])
    return out
