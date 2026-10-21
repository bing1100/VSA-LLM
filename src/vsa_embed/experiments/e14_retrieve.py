"""E14 RD-13 — retrieve-then-read: Wikipedia text for the T4 held-out concepts (pre-registration addendum 10,
`experiments/e14-logic/preregistration-addendum-10-retrieve.md`).

The definition text of a ChEBI concept states about 12% of its gold facts (RD-1). This tool fetches more text per concept
from the English Wikipedia through the MediaWiki API (the precedent of the E11 Wiktionary survey, `DATA_SOURCES.md` §15b):
`list=search` for the concept's name and its ChEBI synonyms, then `prop=extracts` (plain text, intro and sections) of the
candidate pages. Every response is cached under `~/data/vsa-llm/wikipedia/` as jsonl with page id, revision id and fetch
time, so the study is reproducible from the cache by revision id. Licence: CC BY-SA 4.0 (Wikipedia contributors;
attribution by page title and revision id in every row). Rate: one descriptive User-Agent with a contact, at most
`--rps` requests per second, retries on 429 / 5xx.

- `fetch` — build the cache for an E11 item directory: per concept the queries, the candidate pages (search rank, exact
  title, chemistry cue, disambiguation flag) and the chosen top-k pages (`concepts.jsonl`); `pages.jsonl` holds every
  page's extract once; `searches.jsonl` every search; `requests.jsonl` the request log. Resumable.
- `stats` — coverage: how many concepts got a page, how many a *plausible* one (an exact title match or a chemistry cue
  in the intro), extract lengths.
- `build` — the texts the study reads: the primary page's extract cut to whole sentences within `--words` words, written
  as "<headword>: <text>" (`retrieved.jsonl`, style `wiki`); E11's CPU readers (oracle, stated, typeprior) on that text, on
  the whole extract (`wiki-full`, stated reader only) and on the definition text (`frames.jsonl`, E11 row format); the
  *stated* rate per style (`coverage.json`, `report.md`); `manifest.json` with sha256s. `e14_reading recall --family rd13
  --retrieved DIR` then runs the arms.

    python -m vsa_embed.experiments.e14_retrieve fetch --items DIR [--cache DIR] [--k 3] [--rps 4] [--limit N]
    python -m vsa_embed.experiments.e14_retrieve stats [--cache DIR] [--output DIR]
    python -m vsa_embed.experiments.e14_retrieve build --items DIR --output DIR [--cache DIR] [--words 300] [--limit N] [--smoke]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import e11_read_to_learn as e11
from .e5_common import write_json

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "VSA-LLM-research/0.1 (bingxuhu@gmail.com) E14 retrieve-then-read"
LICENCE = "CC BY-SA 4.0 (Wikipedia contributors; attribution: page title and revision id)"
DATA = Path("~/data/vsa-llm").expanduser()
CACHE = DATA / "wikipedia/t4-heldout-v1"
CHEBI_CACHE = DATA / "e14/chebi-255-records.json.gz"
STYLE = "wiki"
FULL_STYLE = "wiki-full"
READERS = ("oracle", "stated", "typeprior")
MAX_SYNONYMS = 3
SEARCH_LIMIT = 5
MAX_PAGES_PER_CONCEPT = 8

# A page is *plausibly* about a chemical if its intro holds one of these words (the disambiguation cue).
CHEMISTRY_CUES = frozenset("""
chemical chemically chemistry compound compounds molecule molecules molecular acid acids ion ions salt salts ester esters
enzyme enzymes protein proteins drug drugs medication pharmaceutical formula organic inorganic alkaloid peptide hormone
metabolite metabolites lipid lipids sugar carbohydrate amino polymer element mineral steroid vitamin antibiotic receptor
inhibitor neurotransmitter fatty glycoside synthesized synthesised solvent reagent oxide hydroxide nucleotide nucleoside
amine ketone aldehyde alcohol phenol toxin pesticide herbicide fungicide insecticide dye pigment isomer enantiomer
derivative substance reaction oxidation reduction catalyst
""".split())
STOP_SECTIONS = {"see also", "references", "external links", "notes", "further reading", "bibliography", "sources",
                 "citations", "footnotes"}
HEADING = re.compile(r"^\s*(=+)\s*(.*?)\s*\1\s*$")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -- the API client and its cache ------------------------------------------------------------------------------------------

class WikiClient:
    """MediaWiki API calls with a jsonl cache under `cache`: `pages.jsonl` (one row per page id: title, revision id,
    timestamp, disambiguation flag, plain-text extract, fetch time), `searches.jsonl` (one row per query) and
    `requests.jsonl` (every request: parameters, time, status, bytes). A cached page or search is never refetched, so a
    second run is reproducible offline. `get` can be replaced (tests)."""

    def __init__(self, cache: Path, *, rps: float = 4.0, get: Callable[[dict[str, str]], dict[str, Any]] | None = None,
                 log: Callable[[str], None] = print) -> None:
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max(rps, 0.01)
        self._get = get or self._http_get
        self.log = log
        self.last_request = 0.0
        self.requests = 0
        self.pages: dict[int, dict[str, Any]] = {r["pageid"]: r for r in e11._read_jsonl(self.cache / "pages.jsonl")}
        self.searches: dict[str, list[dict[str, Any]]] = {r["query"]: r["results"] for r in e11._read_jsonl(self.cache / "searches.jsonl")}

    # -- transport --
    def _http_get(self, params: dict[str, str]) -> dict[str, Any]:
        url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        delay = 2.0
        for attempt in range(6):
            wait = self.min_interval - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            status, body = None, b""
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    status, body = response.status, response.read()
            except urllib.error.HTTPError as error:
                status = error.code
                if status not in {429, 500, 502, 503, 504} or attempt == 5:
                    raise
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == 5:
                    raise
                self.log(f"    network error ({error}); retry {attempt + 1}")
            self.requests += 1
            _append_jsonl(self.cache / "requests.jsonl", {"time": _utc(), "params": params, "status": status, "bytes": len(body)})
            if status == 200:
                return json.loads(body)
            self.log(f"    HTTP {status}; retry {attempt + 1} after {delay:.0f} s")
            time.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")

    # -- calls --
    def search(self, query: str, *, limit: int = SEARCH_LIMIT) -> list[dict[str, Any]]:
        if query in self.searches:
            return self.searches[query]
        data = self._get({"action": "query", "list": "search", "srsearch": query, "srlimit": str(limit), "srnamespace": "0",
                          "srprop": "wordcount"})
        results = [{"pageid": int(r["pageid"]), "title": r["title"], "wordcount": r.get("wordcount")}
                   for r in data.get("query", {}).get("search", [])]
        self.searches[query] = results
        _append_jsonl(self.cache / "searches.jsonl", {"query": query, "time": _utc(), "results": results})
        return results

    def page(self, pageid: int) -> dict[str, Any]:
        if pageid in self.pages:
            return self.pages[pageid]
        data = self._get({"action": "query", "pageids": str(pageid), "prop": "extracts|revisions|pageprops", "explaintext": "1",
                          "exsectionformat": "wiki", "exlimit": "1", "rvprop": "ids|timestamp", "ppprop": "disambiguation"})
        raw = (data.get("query", {}).get("pages") or [{}])[0]
        revision = (raw.get("revisions") or [{}])[0]
        row = {"pageid": pageid, "title": raw.get("title"), "revid": revision.get("revid"), "timestamp": revision.get("timestamp"),
               "disambiguation": "disambiguation" in (raw.get("pageprops") or {}), "missing": bool(raw.get("missing")),
               "extract": raw.get("extract") or "", "fetched_at": _utc(), "licence": LICENCE}
        self.pages[pageid] = row
        _append_jsonl(self.cache / "pages.jsonl", row)
        return row


# -- queries, candidates, the disambiguation rule ------------------------------------------------------------------------------

def chebi_records(path: Path = CHEBI_CACHE) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _usable_synonym(s: str, surface: str) -> bool:
    return bool(s) and s.lower() != surface.lower() and 3 <= len(s) <= 60 and re.search(r"[A-Za-z]{3}", s) is not None \
        and not re.fullmatch(r"[A-Z0-9()\[\]{},.+\-/ ]+", s)       # formulae, SMILES-like and all-caps codes are not queries


def queries_for(surface: str, record: dict[str, Any] | None, *, max_synonyms: int = MAX_SYNONYMS) -> list[str]:
    """The concept's name, then up to `max_synonyms` distinct ChEBI synonyms (shortest first; the ChEBI name itself first
    if it differs from the item surface)."""
    out = [surface]
    if record:
        pool = ([record["name"]] if record.get("name") else []) + sorted(record.get("synonyms") or [], key=lambda s: (len(s), s))
        for s in pool:
            if _usable_synonym(s, surface) and s.lower() not in {q.lower() for q in out}:
                out.append(s)
            if len(out) > max_synonyms:
                break
    return out


def intro_of(extract: str) -> str:
    """The text before the first section heading (`exsectionformat=wiki`)."""
    for i, line in enumerate(extract.split("\n")):
        if HEADING.match(line):
            return "\n".join(extract.split("\n")[:i]).strip()
    return extract.strip()


def chemistry_cue(text: str, *, chars: int = 1500) -> str | None:
    """The first chemistry cue word in the first `chars` characters of `text` (None: no cue)."""
    for word in re.findall(r"[a-z]+", text[:chars].lower()):
        if word in CHEMISTRY_CUES:
            return word
    return None


def _norm_title(text: str) -> str:
    return re.sub(r"[\s_]+", " ", text.strip().lower())


def rank_candidates(candidates: Sequence[dict[str, Any]], queries: Sequence[str]) -> list[dict[str, Any]]:
    """The disambiguation rule. A candidate is dropped if it is a disambiguation page, missing, or has an empty extract.
    The rest are ordered by: an exact title match to one of the queries first; then a chemistry cue in the intro; then
    the query's rank (the name before its synonyms) and the search rank. Each kept candidate records its `exact`, `cue`
    and `reason`."""
    wanted = {_norm_title(q) for q in queries}
    kept = []
    for c in candidates:
        if c.get("disambiguation") or c.get("missing") or not (c.get("extract") or "").strip():
            continue
        c = dict(c)
        c["exact"] = _norm_title(c["title"]) in wanted
        c["cue"] = chemistry_cue(intro_of(c["extract"]))
        c["reason"] = "exact title" if c["exact"] else (f"cue '{c['cue']}'" if c["cue"] else "search rank only")
        kept.append(c)
    kept.sort(key=lambda c: (not c["exact"], c["cue"] is None, c["query_rank"], c["result_rank"]))
    for rank, c in enumerate(kept):
        c["rank"] = rank
    return kept


def retrieve_concept(client: WikiClient, surface: str, record: dict[str, Any] | None, *, k: int = 3,
                     max_pages: int = MAX_PAGES_PER_CONCEPT) -> dict[str, Any]:
    """Search the name, then synonyms until `k` plausible pages (exact title or chemistry cue) are in hand; fetch at most
    `max_pages` candidate pages. Returns the queries, every candidate with its flags, and the top-k page ids."""
    queries = queries_for(surface, record)
    candidates: dict[int, dict[str, Any]] = {}
    used = []
    for qi, q in enumerate(queries):
        used.append(q)
        for ri, hit in enumerate(client.search(q)):
            if hit["pageid"] in candidates or len(candidates) >= max_pages:
                continue
            page = client.page(hit["pageid"])
            candidates[hit["pageid"]] = {"pageid": hit["pageid"], "title": page["title"] or hit["title"], "revid": page["revid"],
                                         "timestamp": page["timestamp"], "query": q, "query_rank": qi, "result_rank": ri,
                                         "disambiguation": page["disambiguation"], "missing": page["missing"],
                                         "extract": page["extract"], "words": len(page["extract"].split())}
        ranked = rank_candidates(list(candidates.values()), queries)
        if sum(1 for c in ranked if c["exact"] or c["cue"]) >= k:
            break
    ranked = rank_candidates(list(candidates.values()), queries)
    slim = [{key: c[key] for key in ("pageid", "title", "revid", "timestamp", "query", "query_rank", "result_rank", "exact", "cue",
                                     "reason", "rank", "words")} for c in ranked]
    top = slim[:k]
    return {"surface": surface, "queries": used, "queries_all": queries, "candidates": slim, "dropped": len(candidates) - len(ranked),
            "pages": [c["pageid"] for c in top], "primary": top[0]["pageid"] if top else None,
            "plausible": bool(top) and bool(top[0]["exact"] or top[0]["cue"])}


# -- the text the study reads --------------------------------------------------------------------------------------------------

def clean_paragraphs(extract: str) -> list[str]:
    """Paragraphs of a plain-text extract in order, headings removed, stopping at the first navigational section (see
    also, references, ...)."""
    out = []
    for block in re.split(r"\n\s*\n", extract):
        lines = []
        for line in block.split("\n"):
            m = HEADING.match(line)
            if m:
                if m.group(2).strip().lower() in STOP_SECTIONS:
                    return out
                continue
            lines.append(line.strip())
        text = re.sub(r"\s*\(\s*\)", "", " ".join(l for l in lines if l))     # "()" left by a removed pronunciation
        if text:
            out.append(text)
    return out


def truncate_words(paragraphs: Sequence[str], budget: int) -> str:
    """Whole sentences, in order, while the word count stays within `budget`; the first sentence is always kept (cut to
    the budget if it alone is longer). Paragraph breaks are kept as newlines."""
    kept: list[list[str]] = []
    total = 0
    for paragraph in paragraphs:
        current: list[str] = []
        for sentence in SENTENCE.split(paragraph):
            n = len(sentence.split())
            if total + n > budget:
                if total == 0:
                    current.append(" ".join(sentence.split()[:budget]))
                    total = budget
                if current:
                    kept.append(current)
                return "\n".join(" ".join(p) for p in kept)
            current.append(sentence)
            total += n
        if current:
            kept.append(current)
    return "\n".join(" ".join(p) for p in kept)


def retrieved_text(headword: str, extract: str, *, words: int) -> tuple[str, int]:
    """("<headword>: <text within the budget>", words of the whole cleaned extract)."""
    paragraphs = clean_paragraphs(extract)
    return f"{headword}: {truncate_words(paragraphs, words)}", sum(len(p.split()) for p in paragraphs)


# -- commands ----------------------------------------------------------------------------------------------------------------------

def run_fetch(args: argparse.Namespace) -> dict[str, Any]:
    read_set = e11.load_read_set(args.items).limit(args.limit)
    cache = Path(args.cache)
    client = WikiClient(cache, rps=args.rps)
    records = chebi_records(Path(args.chebi)) if Path(args.chebi).exists() else {}
    done = {r["concept"]: r for r in e11._read_jsonl(cache / "concepts.jsonl")}
    t0 = time.monotonic()
    for i, c in enumerate(read_set.concepts):
        if c["concept"] in done:
            continue
        surface = c.get("item_surface") or c["surface"]
        result = retrieve_concept(client, surface, records.get(c.get("source_concept") or ""), k=args.k)
        row = {"concept": c["concept"], "source_concept": c.get("source_concept"), "fetched_at": _utc(), **result}
        done[c["concept"]] = row
        _append_jsonl(cache / "concepts.jsonl", row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(read_set.concepts)} concepts, {client.requests} requests, {time.monotonic() - t0:.0f} s", flush=True)
    source = {"api": API, "user_agent": USER_AGENT, "licence": LICENCE, "items": str(args.items), "k": args.k, "rps": args.rps,
              "search_limit": SEARCH_LIMIT, "max_synonyms": MAX_SYNONYMS, "max_pages_per_concept": MAX_PAGES_PER_CONCEPT,
              "retrieved": _utc(), "concepts": len(done), "pages": len(client.pages), "searches": len(client.searches),
              "requests_this_run": client.requests}
    write_json(cache / "source.json", source)
    print(f"fetched {len(done)} concepts, {len(client.pages)} pages cached, {client.requests} requests in {time.monotonic() - t0:.0f} s")
    return source


def coverage_stats(concepts: Sequence[dict[str, Any]], pages: dict[int, dict[str, Any]]) -> dict[str, Any]:
    n = len(concepts)
    with_page = [c for c in concepts if c["primary"] is not None]
    plausible = [c for c in with_page if c["plausible"]]
    exact = [c for c in with_page if c["candidates"][0]["exact"]]
    cue_only = [c for c in plausible if not c["candidates"][0]["exact"]]
    full_words = [sum(len(p.split()) for p in clean_paragraphs(pages[c["primary"]]["extract"])) for c in with_page]
    intro_words = [len(intro_of(pages[c["primary"]]["extract"]).split()) for c in with_page]
    q = lambda xs, p: float(np.percentile(xs, p)) if xs else None  # noqa: E731
    return {"concepts": n, "with_page": len(with_page), "plausible": len(plausible), "exact_title": len(exact), "cue_only": len(cue_only),
            "share_with_page": len(with_page) / n if n else None, "share_plausible": len(plausible) / n if n else None,
            "share_exact_title": len(exact) / n if n else None,
            "pages_per_concept": float(np.mean([len(c["pages"]) for c in concepts])) if concepts else None,
            "candidates_per_concept": float(np.mean([len(c["candidates"]) for c in concepts])) if concepts else None,
            "dropped_per_concept": float(np.mean([c["dropped"] for c in concepts])) if concepts else None,
            "queries_per_concept": float(np.mean([len(c["queries"]) for c in concepts])) if concepts else None,
            "primary_from_synonym": sum(c["candidates"][0]["query_rank"] > 0 for c in with_page),
            "extract_words": {"mean": float(np.mean(full_words)) if full_words else None, "median": q(full_words, 50),
                              "p10": q(full_words, 10), "p90": q(full_words, 90), "min": min(full_words) if full_words else None,
                              "max": max(full_words) if full_words else None},
            "intro_words": {"mean": float(np.mean(intro_words)) if intro_words else None, "median": q(intro_words, 50)},
            "cues": Counter(c["candidates"][0]["cue"] for c in plausible if c["candidates"][0]["cue"]).most_common(10)}


def run_stats(args: argparse.Namespace) -> dict[str, Any]:
    cache = Path(args.cache)
    concepts = e11._read_jsonl(cache / "concepts.jsonl")
    pages = {r["pageid"]: r for r in e11._read_jsonl(cache / "pages.jsonl")}
    stats = coverage_stats(concepts, pages)
    stats["pages_cached"] = len(pages)
    stats["source"] = json.loads((cache / "source.json").read_text()) if (cache / "source.json").exists() else None
    output = Path(args.output or cache)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "stats.json", stats)
    f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
    lines = ["# E14 RD-13 — Wikipedia retrieval coverage", "", f"Cache: `{cache}`; {len(pages)} pages cached.", "",
             "| measure | value |", "|---|---:|"]
    for key in ("concepts", "with_page", "share_with_page", "plausible", "share_plausible", "exact_title", "share_exact_title", "cue_only",
                "primary_from_synonym", "pages_per_concept", "candidates_per_concept", "dropped_per_concept", "queries_per_concept"):
        lines.append(f"| {key} | {f(stats[key])} |")
    for key in ("extract_words", "intro_words"):
        lines.append(f"| {key} | " + ", ".join(f"{k} {f(v)}" for k, v in stats[key].items()) + " |")
    lines += ["", "Cue words of the plausible primary pages: " + ", ".join(f"{w} {n}" for w, n in stats["cues"]), ""]
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return stats


def stated_rates(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per style: the share of gold edges whose partner the finder found (the `stated` reader's frame against the
    `oracle` frame), overall and per relation; concepts with at least one stated edge."""
    by: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for r in rows:
        by[(r["style"], r["concept"])][r["reader"]] = r
    out: dict[str, Any] = {}
    for style in sorted({s for s, _ in by}):
        gold: Counter = Counter()
        hit: Counter = Counter()
        concepts = stated_concepts = 0
        for (s, _), m in by.items():
            if s != style or "oracle" not in m:
                continue
            concepts += 1
            g = {tuple(e) for e in (m["oracle"].get("frame") or [])}
            st = {tuple(e) for e in ((m.get("stated") or {}).get("frame") or [])}
            stated_concepts += bool(st)
            for e in g:
                gold[e[0]] += 1
                hit[e[0]] += e in st
        total = sum(gold.values())
        out[style] = {"concepts": concepts, "concepts_with_stated_edge": stated_concepts, "gold_edges": total,
                      "stated": sum(hit.values()) / total if total else None,
                      "by_relation": {rel: {"gold": n, "stated": hit[rel] / n} for rel, n in gold.most_common()}}
    return out


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    read_set = e11.load_read_set(args.items).limit(args.limit)
    track = read_set.manifest["track"]
    base_style = args.style or (e11.PRIMARY_STYLE.get(track) if e11.PRIMARY_STYLE.get(track) in read_set.styles else read_set.styles[0])
    cache = Path(args.cache)
    retrieval = {r["concept"]: r for r in e11._read_jsonl(cache / "concepts.jsonl")}
    pages = {r["pageid"]: r for r in e11._read_jsonl(cache / "pages.jsonl")}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_defs = {d["concept"]: d for d in read_set.definitions if d["style"] == base_style}
    retrieved, full = [], []
    for c in read_set.concepts:
        r = retrieval.get(c["concept"])
        if not r or r["primary"] is None:
            continue
        page = pages[r["primary"]]
        headword = (base_defs.get(c["concept"]) or {}).get("headword") or c.get("item_surface") or c["surface"]
        text, full_words = retrieved_text(headword, page["extract"], words=args.words)
        source = f"Wikipedia, \"{page['title']}\" (page id {page['pageid']}, revision {page['revid']}, {page['timestamp']})"
        common = {"concept": c["concept"], "headword": headword, "source": source, "licence": LICENCE, "display_name": headword,
                  "pageid": page["pageid"], "revid": page["revid"], "title": page["title"], "plausible": r["plausible"],
                  "choice": r["candidates"][0]["reason"]}
        retrieved.append({**common, "style": STYLE, "text": text, "words": len(text.split()), "full_words": full_words})
        full.append({**common, "style": FULL_STYLE, "text": f"{headword}: " + "\n".join(clean_paragraphs(page["extract"]))})
    e11._write_jsonl(output / "retrieved.jsonl", retrieved)
    ctx = e11.track_context(track)
    covered = {d["concept"] for d in retrieved}
    definitions = retrieved + full + [d for d in read_set.definitions if d["style"] == base_style and d["concept"] in covered]
    subset = e11.ReadSet(read_set.path, read_set.manifest, [c for c in read_set.concepts if c["concept"] in covered], definitions,
                         [i for i in read_set.items if i["concept"] in covered])
    results, mentions = e11.run_readers(None, subset, ctx, READERS, (STYLE, base_style), log=print)
    full_results, full_mentions = e11.run_readers(None, subset, ctx, ("oracle", "stated"), (FULL_STYLE,), log=print)
    rows = e11.frame_rows(results + full_results, subset, ctx, {**mentions, **full_mentions})
    e11._write_jsonl(output / "frames.jsonl", rows)
    coverage = {"concepts": len(read_set.concepts), "covered": len(covered), "plausible": sum(d["plausible"] for d in retrieved),
                "words": args.words, "retrieved_words": {"mean": float(np.mean([d["words"] for d in retrieved])) if retrieved else None,
                                                         "full_mean": float(np.mean([d["full_words"] for d in retrieved])) if retrieved else None},
                "stated": stated_rates(rows), "readers": e11.summarize_frames(rows)}
    write_json(output / "coverage.json", coverage)
    files = {name: sha256_of(output / name) for name in ("retrieved.jsonl", "frames.jsonl", "coverage.json")}
    for name in ("concepts.jsonl", "pages.jsonl", "searches.jsonl"):
        if (cache / name).exists():
            files[f"cache/{name}"] = sha256_of(cache / name)
    manifest = {"schema": "e14-retrieve/1", "kind": read_set.kind, "track": track, "style": STYLE, "full_style": FULL_STYLE,
                "base_style": base_style, "items": str(args.items), "cache": str(cache), "created": time.strftime("%Y-%m-%d"),
                "words": args.words, "readers": list(READERS), "concepts": len(read_set.concepts), "covered": len(covered),
                "plausible": coverage["plausible"], "smoke": bool(args.smoke), "licence": [LICENCE], "sha256": files,
                "text_committed": False, "note": "retrieved.jsonl (CC BY-SA text) is kept out of git; rebuild with `build` from the cache"}
    write_json(output / "manifest.json", manifest)
    (output / ".gitignore").write_text("retrieved.jsonl\n")
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    lines = (["**SMOKE TEST — not a result.**", ""] if args.smoke else []) + [
        f"# E14 RD-13 — retrieved text for {track} {read_set.kind}: what it states", "",
        f"{len(covered)} of {len(read_set.concepts)} concepts have a Wikipedia page ({coverage['plausible']} plausible); text cut to "
        f"{args.words} words (mean {f(coverage['retrieved_words']['mean'])} words; whole extract mean {f(coverage['retrieved_words']['full_mean'])}).", "",
        "## Stated rate (share of gold edges whose partner the finder found)", "",
        "| style | concepts | with a stated edge | gold edges | stated |", "|---|---:|---:|---:|---:|"]
    for style, s in coverage["stated"].items():
        lines.append(f"| {style} | {s['concepts']} | {s['concepts_with_stated_edge']} | {s['gold_edges']} | {f(s['stated'])} |")
    relations = sorted({r for s in coverage["stated"].values() for r in s["by_relation"]})
    lines += ["", "| relation | " + " | ".join(coverage["stated"]) + " |", "|---|" + "---:|" * len(coverage["stated"])]
    for rel in relations:
        lines.append(f"| {rel} | " + " | ".join(f(s["by_relation"].get(rel, {}).get("stated")) for s in coverage["stated"].values()) + " |")
    lines += ["", "## Readers (micro precision / recall / F1 against the gold frame)", "",
              "| style | reader | frames | empty | edges | precision | recall | F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for style, by_reader in coverage["readers"].items():
        for reader, m in by_reader.items():
            lines.append(f"| {style} | {reader} | {m['frames']} | {m['empty']} | {m['edges_mean']:.2f} | {f(m['precision'])} | "
                         f"{f(m['recall'])} | {f(m['f1'])} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return coverage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="build the Wikipedia cache for an E11 item directory (network, CPU)")
    f.add_argument("--items", type=Path, required=True)
    f.add_argument("--cache", type=Path, default=CACHE)
    f.add_argument("--chebi", type=Path, default=CHEBI_CACHE, help="ChEBI records (synonyms for the queries)")
    f.add_argument("--k", type=int, default=3)
    f.add_argument("--rps", type=float, default=4.0, help="requests per second at most")
    f.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("stats", help="coverage of the cache")
    s.add_argument("--cache", type=Path, default=CACHE)
    s.add_argument("--output", type=Path, default=None, help="default: the cache directory")
    b = sub.add_parser("build", help="the retrieved read set, the CPU readers' frames and the stated rate")
    b.add_argument("--items", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--cache", type=Path, default=CACHE)
    b.add_argument("--style", default=None, help="the definition style to compare with (default: the track's primary style)")
    b.add_argument("--words", type=int, default=300, help="word budget of the retrieved text (whole sentences)")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    {"fetch": run_fetch, "stats": run_stats, "build": run_build}[args.command](args)


if __name__ == "__main__":
    main()
