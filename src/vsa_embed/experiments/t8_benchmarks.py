"""T8 (decision 63, WP TK-B2): the Wikidata-entity benchmarks — raw files, their entities, and their items on T8.

The general relation benchmarks of the toolkit (methodology §3.1, §3.3, §3.4) are built on Wikidata entities; T8 gives
those entities frames. This module reads the five benchmarks' raw files (pinned commits) in the layout of the TK-B1
adapters (`~/data/vsa-llm/benchmarks/<name>/raw/<repository path>`; a file already there — e.g. fetched by TK-B1 from
the same commit — is kept and only hashed; T8's own record of the files it uses is `<name>/t8-source.json`). BEAR is
read from its GitHub release (`bear/raw-github/`: per-relation JSONL with subject and object QIDs, which the Hugging Face
parquet of TK-B1 does not carry). It extracts their subject and
object entities (Wikidata QIDs where the set gives them; names otherwise, resolved to QIDs with the Wikidata
`wbsearchentities` API through `ontologies.wikidata.WikidataClient`'s cache), and maps the benchmark items onto the
built track (stage `items`: each item's subject concept and whether it is held out).

| benchmark | source (pinned) | licence | entities given as |
|---|---|---|---|
| `entity-inferences` (Onoe et al., ACL 2023) | GitHub `yasumasaonoe/entity_knowledge_propagation` | see repository | names (fictitious entities in part; ECBD 2020–21 entities) |
| `lre` (Hernandez et al., ICLR 2024) | GitHub `evandez/relations` (`data/factual`) | MIT | names |
| `bear` (Wiland et al., Findings NAACL 2024) | GitHub `lm-pub-quiz/BEAR` (`BEAR/`) | CC BY-SA 4.0 | QIDs |
| `popqa` (Mallen et al., ACL 2023) | HF `akariasai/PopQA` (`test.tsv`) | see dataset card (MIT in the paper's repository) | QIDs (`s_uri`, `o_uri`) |
| `twohopfact` (Yang et al., ACL 2024) | HF `soheeyang/TwoHopFact` (`TwoHopFact.csv`) | CC BY 4.0 | QIDs (`e1/e2/e3` URIs) when present, else names |

Item format (the ranking format of the TK-B harness; `RANK_SCHEMA`): one JSON object per line,
`{"id", "set", "kind": "rank", "context", "options", "answer", "terms": [{"surface", "concept", "frame", "definition"}],
"meta"}` — `context` is the prompt, `options` the candidate continuations (scored by log-likelihood), `answer` the index of
the gold option, `terms` the T8 concepts the prompt mentions (`concept` = the QID, `frame` = its verbalized T8 frame,
`definition` = its Wikidata description), `meta` the benchmark's own fields plus `t8` (subject concept, held-out flag,
whether the subject has a frame, entry id).

    python -m vsa_embed.experiments.t8_benchmarks fetch [--root ~/data/vsa-llm/benchmarks]
    python -m vsa_embed.experiments.t8_benchmarks entities --config experiments/t8-wikidata/t8.yaml
    python -m vsa_embed.experiments.t8_benchmarks items --config experiments/t8-wikidata/t8.yaml --run experiments/t8-wikidata/runs/v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

BENCHMARK_ROOT = Path("~/data/vsa-llm/benchmarks").expanduser()
RANK_SCHEMA = "tk-rank/1"
GITHUB_RAW = "https://raw.githubusercontent.com"
HF_RESOLVE = "https://huggingface.co/datasets"
QID = re.compile(r"Q[1-9][0-9]*")


@dataclass(frozen=True)
class Source:
    """A benchmark's pinned raw files: `files` are repository paths, saved under `<root>/<name>/<subdir>/` with the same
    path."""

    name: str
    kind: str                  # "github" or "hf"
    repo: str
    revision: str
    files: tuple[str, ...]
    licence: str
    citation: str
    subdir: str = "raw"

    def folder(self, root: Path) -> Path:
        return Path(root).expanduser() / self.name / self.subdir

    def url(self, path: str) -> str:
        if self.kind == "github":
            return f"{GITHUB_RAW}/{self.repo}/{self.revision}/{path}"
        return f"{HF_RESOLVE}/{self.repo}/resolve/{self.revision}/{path}"


LRE_FACTUAL = ("city_in_country", "company_ceo", "company_hq", "country_capital_city", "country_currency", "country_language",
               "country_largest_city", "food_from_country", "landmark_in_country", "landmark_on_continent",
               "person_band_lead_singer", "person_father", "person_mother", "person_native_language", "person_occupation",
               "person_plays_instrument", "person_plays_position_in_sport", "person_plays_pro_sport", "person_university",
               "pokemon_evolutions", "presidents_birth_year", "presidents_election_year", "product_by_company",
               "star_constellation", "superhero_archnemesis", "superhero_person")
BEAR_RELATIONS = ("P103", "P105", "P108", "P115", "P127", "P1303", "P131", "P137", "P1376", "P1412", "P1441", "P1532", "P162",
                  "P170", "P171", "P175", "P176", "P177", "P178", "P179", "P185", "P19", "P190", "P20", "P206", "P26", "P2632",
                  "P27", "P272", "P291", "P30", "P3373", "P344", "P36", "P364", "P37", "P403", "P412", "P413", "P427", "P449",
                  "P4552", "P463", "P466", "P50", "P509", "P53", "P57", "P58", "P6", "P610", "P611", "P641", "P676", "P6886",
                  "P69", "P7937", "P7959", "P87", "P98")
EI_FILES = ("disaster_explicit_attribute_independent", "disaster_implicit_attribute_independent",
            "fake_person_explicit_attribute_dependent", "fake_person_implicit_attribute_dependent_adjective",
            "tv_show_explicit_attribute_dependent", "tv_show_implicit_attribute_dependent_adjective")
ECBD_FILES = ("all_ent_2020_2021_np_500samples", "all_ent_2020_2021_np_easy", "all_ent_2020_2021_random_500samples",
              "all_ent_2020_2021_random_easy", "specificity_popular_20np_20random")

SOURCES: dict[str, Source] = {
    "entity-inferences": Source(
        "entity-inferences", "github", "yasumasaonoe/entity_knowledge_propagation", "938295c1ead0622350f2e6c9d6122964af123e05",
        tuple(f"data/entity_inferences/{f}.json" for f in EI_FILES) + tuple(f"data/ecbd/{f}.json" for f in ECBD_FILES) + ("README.md",),
        "no licence file in the repository (research use; cite Onoe et al. 2023)",
        "Onoe, Zhang, Padmanabhan, Durrett, Choi. Can LMs Learn New Entities from Descriptions? ACL 2023"),
    "lre": Source(
        "lre", "github", "evandez/relations", "1b9ec3cf2b8368e42bde7e80fcef384312a6ec07",
        tuple(f"data/factual/{r}.json" for r in LRE_FACTUAL) + ("LICENSE", "README.md"), "MIT",
        "Hernandez, Sharma, Haklay, Meng, Wattenberg, Andreas, Belinkov, Bau. Linearity of Relation Decoding in "
        "Transformer Language Models. ICLR 2024"),
    "bear": Source(
        "bear", "github", "lm-pub-quiz/BEAR", "bc123e6eb66e7c07eb78d60ab64d6949b8fb224f",
        tuple(f"BEAR/{r}.jsonl" for r in BEAR_RELATIONS) + ("BEAR/metadata_relations.json", "relation_info.json",
                                                          "bear_lite_indices.json", "license.txt", "README.md"),
        "CC BY-SA 4.0 (license.txt)",
        "Wiland, Ploner, Akbik. BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked "
        "Language Models. Findings of NAACL 2024", subdir="raw-github"),
    "popqa": Source(
        "popqa", "hf", "akariasai/PopQA", "098765c79ea10a2cb19c828324e33281b8336ec0", ("test.tsv", "README.md"),
        "see the dataset card (the paper's repository is MIT)",
        "Mallen, Asai, Zhong, Das, Khashabi, Hajishirzi. When Not to Trust Language Models. ACL 2023"),
    "twohopfact": Source(
        "twohopfact", "hf", "soheeyang/TwoHopFact", "59a84cd883f71641a03fb6fa50da92a9603e7845", ("TwoHopFact.csv", "README.md"),
        "CC BY 4.0",
        "Yang, Gribovskaya, Kassner, Geva, Riedel. Do Large Language Models Latently Perform Multi-Hop Reasoning? ACL 2024"),
}
USER_AGENT = "VSA-LLM-research/0.1 (T8 Wikidata track; https://github.com/bing1100/VSA-LLM; bingxuhu@gmail.com)"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(root: Path = BENCHMARK_ROOT, names: Iterable[str] | None = None, *, session: Any = None) -> dict[str, Any]:
    """Download every pinned file that is not on disk yet (`<root>/<name>/<subdir>/<path>`) and write
    `<root>/<name>/t8-source.json` (repository, revision, licence, per-file sha256 and whether T8 downloaded it or found
    it). Files already present are kept and only hashed."""
    import requests
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", USER_AGENT)
    out = {}
    for name in names or SOURCES:
        source = SOURCES[name]
        folder = source.folder(root)
        files = {}
        for path in source.files:
            target = folder / path
            status = "present"
            if not target.exists():
                response = session.get(source.url(path), timeout=300)
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".part")
                partial.write_bytes(response.content)
                partial.rename(target)
                status = "downloaded"
            files[path] = {"sha256": file_sha256(target), "bytes": target.stat().st_size, "status": status}
        record = {"name": name, "kind": source.kind, "repo": source.repo, "revision": source.revision, "licence": source.licence,
                  "citation": source.citation, "subdir": source.subdir,
                  "files": {f"{source.subdir}/{path}": info for path, info in files.items()}}
        (folder.parent / "t8-source.json").write_text(json.dumps(record, indent=2) + "\n")
        out[name] = {k: v for k, v in record.items() if k != "files"} | {"files": len(files),
                                                                         "downloaded": sum(f["status"] == "downloaded" for f in files.values())}
    return out


# -- raw readers ------------------------------------------------------------------------------------------------------

def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _qid_of(uri: str | None) -> str | None:
    found = QID.search(uri or "")
    return found.group(0) if found else None


def bear_rows(root: Path = BENCHMARK_ROOT) -> Iterator[dict[str, Any]]:
    """BEAR instances with their relation's templates and answer space (metadata_relations.json)."""
    folder = SOURCES["bear"].folder(root) / "BEAR"
    metadata = json.loads((folder / "metadata_relations.json").read_text())
    for relation in BEAR_RELATIONS:
        meta = metadata[relation]
        for index, row in enumerate(_jsonl(folder / f"{relation}.jsonl")):
            yield {"relation": relation, "index": index, **row, "templates": meta["templates"],
                   "answer_space_labels": meta["answer_space_labels"], "answer_space_ids": meta["answer_space_ids"]}


def popqa_rows(root: Path = BENCHMARK_ROOT) -> Iterator[dict[str, Any]]:
    with open(SOURCES["popqa"].folder(root) / "test.tsv", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            yield {**row, "s_qid": _qid_of(row["s_uri"]), "o_qid": _qid_of(row["o_uri"]),
                   "possible_answers": json.loads(row["possible_answers"]), "s_pop": int(row["s_pop"]), "o_pop": int(row["o_pop"])}


def twohop_rows(root: Path = BENCHMARK_ROOT) -> Iterator[dict[str, Any]]:
    csv.field_size_limit(1 << 30)
    with open(SOURCES["twohopfact"].folder(root) / "TwoHopFact.csv", newline="") as handle:
        yield from csv.DictReader(handle)


def lre_rows(root: Path = BENCHMARK_ROOT) -> Iterator[dict[str, Any]]:
    folder = SOURCES["lre"].folder(root) / "data" / "factual"
    for relation in LRE_FACTUAL:
        data = json.loads((folder / f"{relation}.json").read_text())
        for index, sample in enumerate(data["samples"]):
            yield {"relation": relation, "index": index, "subject": sample["subject"], "object": sample["object"],
                   "prompt_templates": data["prompt_templates"], "prompt_templates_zs": data.get("prompt_templates_zs", []),
                   "properties": data.get("properties", {})}


def ei_rows(root: Path = BENCHMARK_ROOT, *, ecbd: bool = False) -> Iterator[dict[str, Any]]:
    """Entity Inferences rows (`ecbd=True`: the ECBD 2020–21 rows), each with `file`, the class QIDs of its `qid` field
    and, for ECBD, the Wikipedia title encoded in `ex_id` (`<title>_<page id>_<i>_<j>`)."""
    folder = SOURCES["entity-inferences"].folder(root) / "data" / ("ecbd" if ecbd else "entity_inferences")
    for name in (ECBD_FILES if ecbd else EI_FILES):
        for row in _jsonl(folder / f"{name}.json"):
            classes = QID.findall(str(row.get("qid", "")))
            title = row["ex_id"].rsplit("_", 3)[0] if ecbd else None
            yield {**row, "file": name, "classes": classes, "wiki_title": title}


# -- entity references and name resolution ----------------------------------------------------------------------------

# LRE factual relation → (Wikidata properties that state it, subject type) for disambiguation and item mapping; empty
# where Wikidata has no direct property (largest city, Pokémon evolution, superhero relations, years).
LRE_PROPERTIES: dict[str, tuple[tuple[str, ...], str | None]] = {
    "city_in_country": (("P17",), None), "company_ceo": (("P169",), None), "company_hq": (("P159",), None),
    "country_capital_city": (("P36",), None), "country_currency": (("P38",), None), "country_language": (("P37",), None),
    "country_largest_city": ((), None), "food_from_country": (("P495",), None), "landmark_in_country": (("P17",), None),
    "landmark_on_continent": (("P30",), None), "person_band_lead_singer": (("P463", "P361"), "Q5"),
    "person_father": (("P22",), "Q5"), "person_mother": (("P25",), "Q5"), "person_native_language": (("P103", "P1412"), "Q5"),
    "person_occupation": (("P106",), "Q5"), "person_plays_instrument": (("P1303",), "Q5"),
    "person_plays_position_in_sport": (("P413",), "Q5"), "person_plays_pro_sport": (("P641",), "Q5"),
    "person_university": (("P69",), "Q5"), "pokemon_evolutions": ((), None), "presidents_birth_year": ((), "Q5"),
    "presidents_election_year": ((), "Q5"), "product_by_company": (("P176", "P178", "P170"), None),
    "star_constellation": (("P59",), None), "superhero_archnemesis": ((), None), "superhero_person": ((), None),
}
LRE_NON_ENTITY_OBJECTS = {"presidents_birth_year", "presidents_election_year"}
# PopQA `prop` → property; TwoHopFact relation categories (the part after the dash) → property.
POPQA_PROPERTIES = {"occupation": "P106", "place of birth": "P19", "genre": "P136", "father": "P22", "country": "P17",
                    "producer": "P162", "director": "P57", "capital of": "P1376", "screenwriter": "P58", "composer": "P86",
                    "color": "P462", "religion": "P140", "sport": "P641", "author": "P50", "mother": "P25", "capital": "P36"}
TWOHOP_PROPERTIES = {"author": "P50", "dev": "P178", "spouse": "P26", "uguniv": "P69", "singer": "P175", "hqcntry": "P17",
                     "founder": "P112", "birthcity": "P19", "birthcntry": "P27", "hqcity": "P159", "father": "P22", "mother": "P25",
                     "cntry": "P17", "orgz": "P108", "stockexch": "P414", "anthem": "P85", "president": "P35", "capital": "P36",
                     "mainchar": "P674", "novel": "P1441", "movie": "P1441", "creator": "P170", "origcntry": "P495",
                     "director": "P57", "ugmajor": "P812", "ceo": "P169", "actor": "P161", "univ": "P69"}
RESOLVE_PROPERTIES = ("P31", "P17", "P169", "P159", "P36", "P38", "P37", "P495", "P30", "P463", "P361", "P22", "P25", "P103",
                      "P1412", "P106", "P1303", "P413", "P641", "P69", "P176", "P178", "P170", "P59", "P39")


def entity_references(root: Path = BENCHMARK_ROOT) -> list[dict[str, Any]]:
    """Every entity the benchmarks name: {benchmark, role, qid (None when only a name is given), name, relation, …}."""
    refs: list[dict[str, Any]] = []
    for row in bear_rows(root):
        refs.append({"benchmark": "bear", "role": "subject", "qid": row["sub_id"], "name": row["sub_label"], "relation": row["relation"]})
        refs.append({"benchmark": "bear", "role": "object", "qid": row["obj_id"], "name": row["obj_label"], "relation": row["relation"]})
    for row in popqa_rows(root):
        refs.append({"benchmark": "popqa", "role": "subject", "qid": row["s_qid"], "name": row["subj"], "relation": row["prop"]})
        refs.append({"benchmark": "popqa", "role": "object", "qid": row["o_qid"], "name": row["obj"], "relation": row["prop"]})
    for row in twohop_rows(root):
        for role, key in (("subject", "e1"), ("bridge", "e2"), ("object", "e3")):
            refs.append({"benchmark": "twohopfact", "role": role, "qid": _qid_of(row[f"{key}.wikidata_qid"]),
                         "name": row[f"{key}.value"], "relation": row["fact_comp_type"]})
    for row in lre_rows(root):
        refs.append({"benchmark": "lre", "role": "subject", "qid": None, "name": row["subject"], "relation": row["relation"],
                     "object": row["object"]})
        if row["relation"] not in LRE_NON_ENTITY_OBJECTS:
            refs.append({"benchmark": "lre", "role": "object", "qid": None, "name": row["object"], "relation": row["relation"]})
    for ecbd in (False, True):
        for row in ei_rows(root, ecbd=ecbd):
            fake = not row["classes"] or row["category"] == "fake_person"
            refs.append({"benchmark": "ecbd" if ecbd else "entity_inferences", "role": "subject", "qid": None,
                         "name": row["ent_str"], "relation": row["category"], "classes": row["classes"],
                         "wiki_title": row["wiki_title"], "fictitious": fake and not ecbd})
    return refs


def search_names(refs: Sequence[dict[str, Any]]) -> list[str]:
    """Names to resolve with `wbsearchentities` (every reference without a QID; ECBD also by its Wikipedia title)."""
    names = set()
    for ref in refs:
        if ref["qid"] is None and not ref.get("fictitious"):
            names.add(ref["name"])
            if ref.get("wiki_title"):
                names.add(ref["wiki_title"].replace("_", " "))
    return sorted(names)


def choose_candidate(candidates: Sequence[str], *, claims: dict[str, dict[str, list[str]]], properties: Sequence[str] = (),
                     objects: Sequence[str] = (), types: Sequence[str] = (), titles: dict[str, str | None] | None = None,
                     title: str | None = None) -> tuple[str | None, str]:
    """(QID, method) among `wbsearchentities` candidates (search order): `sitelink` — its English Wikipedia title is the
    benchmark's; `verified` — one of `properties` has one of the object's candidates as value; `typed` — its P31 is one
    of `types`; `top` — the first hit; (None, "unresolved") without hits."""
    if not candidates:
        return None, "unresolved"
    if title and titles:
        wanted = title.replace(" ", "_")
        for q in candidates:
            if (titles.get(q) or "") == wanted:
                return q, "sitelink"
    if properties and objects:
        targets = set(objects)
        for q in candidates:
            if any(set(claims.get(q, {}).get(p, [])) & targets for p in properties):
                return q, "verified"
    if types:
        for q in candidates:
            if set(claims.get(q, {}).get("P31", [])) & set(types):
                return q, "typed"
    return candidates[0], "top"


def resolve_references(refs: list[dict[str, Any]], client: Any, *, limit: int = 5, object_candidates: int = 3,
                       progress: Any = None) -> list[dict[str, Any]]:
    """Fill `qid` / `method` / `candidates` of every name-only reference (in place; returns refs)."""
    from vsa_embed.ontologies.wikidata import fetch_claims, fetch_terms
    names = search_names(refs)
    hits: dict[str, list[str]] = {}
    for index, name in enumerate(names):
        hits[name] = [h["id"] for h in client.search(name, limit=limit) if QID.fullmatch(h.get("id", ""))]
        if progress is not None and (index + 1) % 500 == 0:
            progress(f"wbsearchentities: {index + 1:,} of {len(names):,} names")
    pool = sorted({q for found in hits.values() for q in found}, key=lambda q: int(q[1:]))
    claims = fetch_claims(client, pool, RESOLVE_PROPERTIES, kind="candidate_claims", progress=progress)
    titled = sorted({q for ref in refs if ref.get("wiki_title")
                     for q in hits.get(ref["wiki_title"].replace("_", " "), []) + hits.get(ref["name"], [])},
                    key=lambda q: int(q[1:]))
    titles = {q: t["enwiki"] for q, t in fetch_terms(client, titled, progress=progress).items()} if titled else {}
    for ref in refs:
        if ref["qid"] is not None or ref.get("fictitious"):
            ref.setdefault("method", "given" if ref["qid"] else "fictitious")
            continue
        candidates = list(hits.get(ref["name"], []))
        if ref.get("wiki_title"):
            candidates = list(dict.fromkeys(hits.get(ref["wiki_title"].replace("_", " "), []) + candidates))
        properties: tuple[str, ...] = ()
        types: tuple[str, ...] = ()
        objects: list[str] = []
        if ref["benchmark"] == "lre":
            properties, subject_type = LRE_PROPERTIES[ref["relation"]]
            types = (subject_type,) if subject_type else ()
            if ref["role"] == "subject" and ref.get("object") is not None:
                objects = hits.get(ref["object"], [])[:object_candidates]
            if ref["role"] == "object":
                properties, types = (), ()
        elif ref.get("classes"):
            types = tuple(ref["classes"])
        qid, method = choose_candidate(candidates, claims=claims, properties=properties, objects=objects, types=types,
                                       titles=titles, title=ref.get("wiki_title"))
        ref.update(qid=qid, method=method, candidates=candidates)
    # LRE objects: the value a verified subject has (else the object's own top hit).
    verified_objects: dict[tuple[str, str], str] = {}
    for ref in refs:
        if ref["benchmark"] == "lre" and ref["role"] == "subject" and ref.get("method") == "verified":
            props = LRE_PROPERTIES[ref["relation"]][0]
            values = {v for p in props for v in claims.get(ref["qid"], {}).get(p, [])}
            match = [q for q in hits.get(ref["object"], [])[:object_candidates] if q in values]
            if match:
                verified_objects.setdefault((ref["relation"], ref["object"]), match[0])
    for ref in refs:
        if ref["benchmark"] == "lre" and ref["role"] == "object" and (ref["relation"], ref["name"]) in verified_objects:
            ref.update(qid=verified_objects[(ref["relation"], ref["name"])], method="verified")
    return refs


def entity_table(refs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """qid → {names, roles (benchmark:role), benchmarks}."""
    table: dict[str, dict[str, Any]] = {}
    for ref in refs:
        if not ref.get("qid"):
            continue
        row = table.setdefault(ref["qid"], {"names": set(), "roles": set(), "benchmarks": set()})
        row["names"].add(ref["name"]); row["roles"].add(f"{ref['benchmark']}:{ref['role']}"); row["benchmarks"].add(ref["benchmark"])
    return {q: {k: sorted(v) for k, v in row.items()} for q, row in sorted(table.items(), key=lambda kv: int(kv[0][1:]))}


def wikidata_client(config: dict[str, Any], *, log: Any = print, offline: bool = False) -> Any:
    from vsa_embed.ontologies.wikidata import WikidataClient
    spec = config["wikidata"]
    return WikidataClient(Path(spec["cache"]).expanduser(), min_interval=float(spec.get("min_interval", 1.0)),
                          api_interval=float(spec.get("api_interval", 0.1)), log=log, offline=offline)


def run_entities(config: dict[str, Any], *, log: Any = print) -> dict[str, Any]:
    """Stage `entities`: references of the five benchmarks, names resolved; writes `resolution.jsonl.gz` and
    `benchmark_entities.jsonl` to `wikidata.dir` and returns the counts (also `entities.json`)."""
    spec = config["wikidata"]
    out = Path(spec["dir"]).expanduser()
    client = wikidata_client(config, log=log)
    refs = entity_references(Path(config["benchmarks"]["root"]).expanduser())
    resolve_references(refs, client, limit=int(spec.get("search_limit", 5)), progress=log)
    out.mkdir(parents=True, exist_ok=True)
    from vsa_embed.ontologies.wikidata import gzip_writer
    with gzip_writer(out / "resolution.jsonl.gz") as handle:
        for ref in refs:
            handle.write((json.dumps(ref, sort_keys=True, ensure_ascii=False) + "\n").encode())
    table = entity_table(refs)
    with (out / "benchmark_entities.jsonl").open("w") as handle:
        for qid, row in table.items():
            handle.write(json.dumps({"qid": qid, **row}, ensure_ascii=False) + "\n")
    methods = Counter((r["benchmark"], r["role"], r.get("method")) for r in refs)
    summary = {"references": len(refs), "entities": len(table),
               "by_benchmark": {b: len({r["qid"] for r in refs if r["benchmark"] == b and r.get("qid")})
                                for b in sorted({r["benchmark"] for r in refs})},
               "methods": {f"{b}/{r}/{m}": n for (b, r, m), n in sorted(methods.items(), key=lambda kv: str(kv[0]))},
               "client": dict(client.stats)}
    (out / "entities.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# -- items on T8 ----------------------------------------------------------------------------------------------------------

MASK = re.compile(r"<extra_id_\d+>")
DISTRACTORS = 9


@dataclass
class TrackView:
    """What item mapping needs from a built T8 track: concept → entry, held-out flag, training frequency, frames."""

    concepts: dict[str, int]                    # QID → concept index
    entry_of: dict[int, int]                    # concept → entry (single-concept entries)
    heldout: set[str]
    frequency: np.ndarray                       # per entry (ℓ_min of the build)
    frames: list[list[list[str]]]               # per concept: [[relation, filler label], …]
    fillers: list[set[str]]                     # per concept: filler QIDs
    descriptions: list[str]
    aliases: dict[str, int]                     # normalized alias → concept (single-concept aliases)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    redirects: dict[str, str] = field(default_factory=dict)       # merged QIDs of the benchmarks → their targets

    def resolve(self, qid: str | None) -> str | None:
        return self.redirects.get(qid, qid) if qid else qid

    def term(self, surface: str, qid: str | None, definition: str | None = None) -> dict[str, Any]:
        qid = self.resolve(qid)
        concept = self.concepts.get(qid) if qid else None
        record = self.records.get(qid or "", {})
        return {"surface": surface, "concept": qid if concept is not None else None,
                "frame": self.frames[concept] if concept is not None else None,
                "definition": definition if definition is not None else (self.descriptions[concept] if concept is not None
                                                                          else record.get("description"))}

    def status(self, qid: str | None, *, surface: str | None = None, answer: str | None = None) -> dict[str, Any]:
        from vsa_embed.span_channel import normalize_alias
        qid, answer = self.resolve(qid), self.resolve(answer)
        concept = self.concepts.get(qid) if qid else None
        if concept is None:
            return {"subject": qid, "in_track": False, "heldout": False, "entry": None, "frame_edges": 0,
                    "answer": answer, "answer_in_frame": False, "linked": False, "train_frequency": None}
        entry = self.entry_of.get(concept)
        return {"subject": qid, "in_track": True, "heldout": qid in self.heldout, "entry": entry,
                "frame_edges": len(self.frames[concept]), "answer": answer,
                "answer_in_frame": bool(answer and answer in self.fillers[concept]),
                "linked": bool(surface) and self.aliases.get(normalize_alias(surface)) == concept,
                "train_frequency": int(self.frequency[entry]) if entry is not None else None}


def track_view(config: dict[str, Any], run_dir: Path) -> TrackView:
    import torch
    from vsa_embed.ontologies.wikidata import verbalize_frame
    from vsa_embed.span_channel import AliasTable, normalize_alias
    from .t8_wikidata_corpus import build_track_ontology, load_records, load_redirects
    records = load_records(config)
    ontology = build_track_ontology(config, records)
    heldout_names = [n for n in (Path(run_dir) / "holdout_concepts.txt").read_text().splitlines() if n]
    index = {q: i for i, q in enumerate(ontology.concept_names)}
    table = AliasTable.from_pairs(ontology.alias_pairs, holdout=[index[q] for q in heldout_names], include_holdout=True)
    channel = torch.load(Path(config["paths"]["data_root"]).expanduser() / "ontology.pt", weights_only=False)
    if channel["alias_table_sha256"] != table.digest():
        raise ValueError("the replayed T8 alias table does not match the built ontology.pt")
    entry_of = {concepts[0]: e for e, concepts in enumerate(table.entry_concepts) if len(concepts) == 1}
    fillers_meta = ontology.metadata["filler_headings"]
    frames = [verbalize_frame(f, ontology.relation_names, ontology.atomic_names, fillers_meta) for f in ontology.frames]
    filler_qids = [{ontology.atomic_names[a].split(":", 1)[1] for _, a in f} for f in ontology.frames]
    aliases = {}
    for alias, concept in ontology.alias_pairs:
        aliases.setdefault(normalize_alias(alias), set()).add(concept)
    return TrackView(index, entry_of, set(heldout_names), np.asarray(channel["train_frequency"]), frames, filler_qids,
                     list(ontology.metadata["descriptions"]), {a: next(iter(c)) for a, c in aliases.items() if len(c) == 1},
                     records, load_redirects(config))


def _distractors(gold: str, pool: Sequence[str], rng: Any, *, exclude: Iterable[str] = (), k: int = DISTRACTORS) -> list[str]:
    banned = {gold.lower()} | {e.lower() for e in exclude}
    candidates = sorted({p for p in pool if p and p.lower() not in banned})
    return rng.sample(candidates, min(k, len(candidates)))


def _rank(rng: Any, gold: str, distractors: Sequence[str]) -> tuple[list[str], int]:
    options = [gold, *distractors]
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(0)


def _fill(template: str, subject: str) -> tuple[str, str] | None:
    """BEAR template → (context, continuation format) when [Y] closes the sentence; None otherwise."""
    if not template.rstrip().endswith("[Y]."):
        return None
    prefix = template[:template.rindex("[Y]")].rstrip()
    return prefix.replace("[X]", subject), " {y}."


def bear_items(view: TrackView, root: Path, seed: int) -> Iterator[dict[str, Any]]:
    """One item per instance and template; options = the relation's answer space (BEAR's own); statements whose [Y] is
    not last are scored as whole sentences (empty context)."""
    for row in bear_rows(root):
        status = view.status(row["sub_id"], surface=row["sub_label"], answer=row["obj_id"])
        for k, template in enumerate(row["templates"]):
            filled = _fill(template, row["sub_label"])
            if filled:
                context, form = filled
                options = [form.format(y=label) for label in row["answer_space_labels"]]
            else:
                context = ""
                options = [template.replace("[X]", row["sub_label"]).replace("[Y]", label) for label in row["answer_space_labels"]]
            yield {"id": f"bear-{row['relation']}-{row['index']}-t{k}", "set": "bear", "kind": "rank", "context": context,
                   "options": options, "answer": int(row["answer_idx"]),
                   "terms": [view.term(row["sub_label"], row["sub_id"])],
                   "meta": {"relation": row["relation"], "property": row["relation"], "template": k, "sub_id": row["sub_id"],
                            "obj_id": row["obj_id"], "obj_label": row["obj_label"], "t8": status}}


def lre_items(view: TrackView, root: Path, refs: dict[tuple[str, str, str], dict[str, Any]], seed: int) -> Iterator[dict[str, Any]]:
    """One item per factual sample (first prompt template): the gold object and 9 other objects of the relation."""
    import random
    pools: dict[str, list[str]] = defaultdict(list)
    rows = list(lre_rows(root))
    for row in rows:
        pools[row["relation"]].append(row["object"])
    for row in rows:
        rng = random.Random(f"{seed}-lre-{row['relation']}-{row['index']}")
        subject = refs.get(("lre", "subject", row["relation"], row["subject"]), {})
        obj = refs.get(("lre", "object", row["relation"], row["object"]), {})
        options, answer = _rank(rng, " " + row["object"], [" " + d for d in _distractors(row["object"], pools[row["relation"]], rng)])
        properties = LRE_PROPERTIES[row["relation"]][0]
        yield {"id": f"lre-{row['relation']}-{row['index']}", "set": "lre-relations", "kind": "rank",
               "context": row["prompt_templates"][0].format(row["subject"]), "options": options, "answer": answer,
               "terms": [view.term(row["subject"], subject.get("qid"))],
               "meta": {"relation": row["relation"], "properties": list(properties), "subject": row["subject"], "object": row["object"],
                        "resolution": subject.get("method"), "object_qid": obj.get("qid"),
                        "t8": view.status(subject.get("qid"), surface=row["subject"], answer=obj.get("qid"))}}


def popqa_items(view: TrackView, root: Path, seed: int) -> Iterator[dict[str, Any]]:
    """PopQA as ranking: "Q: <question> A:" with the first possible answer and 9 objects of the same property that are
    not possible answers. (PopQA itself is scored by generation; the meta layer may use `s_pop` for when-to-call.)"""
    import random
    rows = list(popqa_rows(root))
    pools: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        pools[row["prop"]].append(row["obj"])
    for row in rows:
        rng = random.Random(f"{seed}-popqa-{row['id']}")
        gold = row["possible_answers"][0] if row["possible_answers"] else row["obj"]
        distractors = _distractors(gold, pools[row["prop"]], rng, exclude=row["possible_answers"])
        options, answer = _rank(rng, " " + gold, [" " + d for d in distractors])
        yield {"id": f"popqa-{row['id']}", "set": "popqa", "kind": "rank", "context": f"Q: {row['question']} A:",
               "options": options, "answer": answer, "terms": [view.term(row["subj"], row["s_qid"])],
               "meta": {"prop": row["prop"], "property": POPQA_PROPERTIES.get(row["prop"]), "s_pop": row["s_pop"],
                        "o_pop": row["o_pop"], "possible_answers": row["possible_answers"], "s_qid": row["s_qid"], "o_qid": row["o_qid"],
                        "t8": view.status(row["s_qid"], surface=row["subj"], answer=row["o_qid"])}}


def twohop_items(view: TrackView, root: Path, seed: int) -> Iterator[dict[str, Any]]:
    """Per TwoHopFact row three items — `2hop` (r2(r1(e1)) → e3), `hop1` (r1(e1) → e2) and `hop2` (r2(e2) → e3) — with 9
    distractors of the same relation category; the subject term is e1 (`hop2`: e2)."""
    import random
    rows = list(twohop_rows(root))
    e2_pool: dict[str, list[str]] = defaultdict(list)
    e3_pool: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        e2_pool[row["r1.category"]].append(row["e2.value"]); e3_pool[row["r2.category"]].append(row["e3.value"])
    for row in rows:
        rng = random.Random(f"{seed}-twohop-{row['uid']}")
        e1, e2, e3 = (_qid_of(row[f"{k}.wikidata_qid"]) for k in ("e1", "e2", "e3"))
        base = {"fact_comp_type": row["fact_comp_type"], "r1": row["r1.category"], "r2": row["r2.category"], "e1": e1, "e2": e2, "e3": e3,
                "e1_value": row["e1.value"], "e2_value": row["e2.value"], "e3_value": row["e3.value"]}
        for name, prompt, gold, pool, subject, subject_qid, answer_qid in (
                ("2hop", row["r2(r1(e1)).prompt"], row["e3.value"], e3_pool[row["r2.category"]], row["e1.value"], e1, e3),
                ("hop1", row["r1(e1).prompt"], row["e2.value"], e2_pool[row["r1.category"]], row["e1.value"], e1, e2),
                ("hop2", row["r2(e2).prompt"], row["e3.value"], e3_pool[row["r2.category"]], row["e2.value"], e2, e3)):
            options, answer = _rank(rng, " " + gold, [" " + d for d in _distractors(gold, pool, rng)])
            terms = [view.term(subject, subject_qid)]
            if name == "2hop":
                terms.append(view.term(row["e2.value"], e2))       # the bridge entity: chained recall reads its frame
            yield {"id": f"twohopfact-{row['uid']}-{name}", "set": f"twohopfact-{name}", "kind": "rank", "context": prompt,
                   "options": options, "answer": answer, "terms": terms,
                   "meta": {**base, "t8": view.status(subject_qid, surface=subject, answer=answer_qid),
                            "bridge": view.status(e2, surface=row["e2.value"], answer=e3) if name == "2hop" else None}}


def ei_items(view: TrackView, root: Path, refs: dict[tuple[str, str, str, str], dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Entity Inferences probes: the masked span → each label; context = the probe up to the mask, options = label + the
    rest of the probe; the entity's definition (mask filled) is its term definition (the read-to-learn input)."""
    for row in ei_rows(root):
        probe = row["probe_sentences"]["template_0"]
        sentence = probe["probe_sentence"]
        labels = [MASK.sub("", label).strip() for label in probe["labels"]]
        head, _, tail = sentence.partition("<extra_id_0>")
        definition = row["definition"].replace("<extra_id_0>", MASK.sub("", row["def_target"]).strip(), 1)
        definition = " ".join(MASK.sub("", definition).split())
        ref = refs.get(("entity_inferences", "subject", row["category"], row["ent_str"]), {})
        context = head.rstrip()
        options = [(" " if context else "") + label + tail for label in labels]
        yield {"id": f"ei-{row['ex_id']}", "set": "entity-inferences", "kind": "rank", "context": context,
               "options": options, "answer": labels.index(row["label"]),
               "terms": [view.term(row["ent_str"], ref.get("qid"), definition)],
               "meta": {"category": row["category"], "file": row["file"], "attribute": row["attribute"],
                        "fictitious": bool(ref.get("fictitious")), "resolution": ref.get("method"),
                        "t8": view.status(ref.get("qid"), surface=row["ent_str"])}}


def _coverage(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    status = [i["meta"]["t8"] for i in items]
    subjects = {s["subject"] for s in status if s["subject"]}
    covered = {s["subject"] for s in status if s["in_track"]}
    held = {s["subject"] for s in status if s["heldout"]}
    framed = {s["subject"] for s in status if s["in_track"] and s["frame_edges"] > 0}
    return {"items": len(items), "items_subject_resolved": sum(1 for s in status if s["subject"]),
            "items_subject_in_track": sum(1 for s in status if s["in_track"]),
            "items_subject_with_frame": sum(1 for s in status if s["in_track"] and s["frame_edges"] > 0),
            "items_subject_heldout": sum(1 for s in status if s["heldout"]),
            "items_subject_linked": sum(1 for s in status if s["linked"]),
            "items_answer_in_frame": sum(1 for s in status if s["answer_in_frame"]),
            "items_subject_seen_in_training": sum(1 for s in status if s["in_track"] and not s["heldout"] and (s["train_frequency"] or 0) > 0),
            "subjects": len(subjects), "subjects_in_track": len(covered), "subjects_with_frame": len(framed),
            "subjects_heldout": len(held)}


def run_items(config: dict[str, Any], run_dir: Path, *, output: Path | None = None, manifest: Path | None = None,
              log: Any = print) -> dict[str, Any]:
    """Stage `items`: every benchmark item mapped onto T8 (`<benchmarks root>/t8-items/<set>.jsonl`, `manifest.json` with
    sha256 and coverage per set; a copy of the manifest next to the run folder)."""
    root = Path(config["benchmarks"]["root"]).expanduser()
    output = Path(output or config["benchmarks"]["items"]).expanduser()
    seed = int(config["seed"])
    view = track_view(config, run_dir)
    resolution = Path(config["wikidata"]["dir"]).expanduser() / "resolution.jsonl.gz"
    refs: dict[tuple[str, ...], dict[str, Any]] = {}
    with gzip.open(resolution, "rt") as handle:
        for line in handle:
            ref = json.loads(line)
            refs.setdefault((ref["benchmark"], ref["role"], ref["relation"], ref["name"]), ref)
    output.mkdir(parents=True, exist_ok=True)
    generators = {"bear": bear_items(view, root, seed), "lre": lre_items(view, root, refs, seed),
                  "popqa": popqa_items(view, root, seed), "twohopfact": twohop_items(view, root, seed),
                  "entity-inferences": ei_items(view, root, refs)}
    sets: dict[str, Any] = {}
    for name, items in generators.items():
        by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_set[item["set"]].append(item)
        for set_name, rows in by_set.items():
            path = output / f"{set_name}.jsonl"
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sets[set_name] = {"file": path.name, "sha256": file_sha256(path), **_coverage(rows)}
            log(f"items: {set_name}: {sets[set_name]}")
    ecbd = [r for r in refs.values() if r["benchmark"] == "ecbd"]
    ecbd_status = [view.status(r.get("qid"), surface=r["name"]) for r in ecbd]
    document = {"schema": RANK_SCHEMA, "track": "t8", "run": str(run_dir), "seed": seed, "distractors": DISTRACTORS,
                "format": {"id": "item id", "set": "set name", "kind": "rank", "context": "prompt", "options": "continuations",
                           "answer": "index of the gold option",
                           "terms": "[{surface, concept (QID in T8 or null), frame ([[relation, filler label]]), definition}]",
                           "meta": "benchmark fields + t8 {subject, in_track, heldout, entry, frame_edges, answer, answer_in_frame, "
                                   "linked, train_frequency}"},
                "sets": sets, "ecbd_entities": {"names": len(ecbd), "in_track": sum(s["in_track"] for s in ecbd_status),
                                                "heldout": sum(s["heldout"] for s in ecbd_status),
                                                "note": "ECBD is a masked-span perplexity probe: entity coverage only, no ranking items"},
                "rules": {"bear": "BEAR's answer space and templates (statement scoring when [Y] is not last)",
                          "lre-relations": f"first prompt template; gold + {DISTRACTORS} other objects of the relation (seeded)",
                          "popqa": f"'Q: question A:'; first possible answer + {DISTRACTORS} objects of the property that are not possible answers",
                          "twohopfact": f"2hop / hop1 / hop2 prompts; gold + {DISTRACTORS} values of the same relation category",
                          "entity-inferences": "probe up to the mask; options = label + rest of the probe; the filled definition is the term's definition"}}
    (output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    target = Path(manifest) if manifest else Path(run_dir).parent.parent / "items-manifest.json"
    target.write_text(json.dumps(document, indent=2) + "\n")
    return {"output": str(output), "manifest": str(target), "sets": {k: {kk: v[kk] for kk in ("items", "items_subject_in_track",
                                                                                             "items_subject_heldout")}
                                                                      for k, v in sets.items()}}


# -- CLI ----------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    from .t1_open_corpus import load_config
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("fetch", help="download the pinned raw benchmark files (kept when present)")
    get.add_argument("--root", type=Path, default=BENCHMARK_ROOT)
    get.add_argument("--names", nargs="*", default=None, choices=sorted(SOURCES))
    ent = sub.add_parser("entities", help="the benchmarks' entities, names resolved to QIDs")
    ent.add_argument("--config", type=Path, required=True)
    items = sub.add_parser("items", help="map the benchmark items onto the built T8 track")
    items.add_argument("--config", type=Path, required=True)
    items.add_argument("--run", type=Path, required=True, help="the T8 build's run folder (holdout list)")
    items.add_argument("--output", type=Path, default=None, help="default: <benchmarks root>/t8-items")
    items.add_argument("--manifest", type=Path, default=None, help="committed copy of the manifest (default: <run>/../items-manifest.json)")
    args = parser.parse_args(argv)
    log = lambda message: print(message, file=sys.stderr, flush=True)      # noqa: E731
    if args.command == "fetch":
        print(json.dumps(fetch(args.root, args.names), indent=2))
    elif args.command == "entities":
        print(json.dumps(run_entities(load_config(args.config), log=log), indent=2))
    else:
        print(json.dumps(run_items(load_config(args.config), args.run, output=args.output, manifest=args.manifest, log=log),
                         indent=2, default=str))


if __name__ == "__main__":
    main()
