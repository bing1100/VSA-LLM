"""MeSH vocabulary that is new to the host: the ontology of the E9 new-vocabulary track T7 (author decision 55).

T1-open linked every MeSH descriptor in PubMed and the channel did nothing there: descriptors are ordinary
biomedical English that the hosts already model (R9 novelty index −5 to −8%). T7 keeps the natural text (PubMed
2025–26 abstracts) and the open, typed ontology (MeSH 2026) but links only names the host has (almost) never seen:

- **Candidates** — every term of the MeSH 2026 supplementary concept records (SCRs: drugs and other chemicals,
  diseases, organisms; `supp2026`) and, optionally, of descriptors (`desc2026`), under the T1 alias policy (no bare
  abbreviations, function words, publication types) plus a minimum length.
- **Selection by corpus frequency** (`select_aliases`) — an alias is kept if it occurs at least
  `min_domain_mentions` times in the track's training-side PubMed text and at most `max_general_mentions` times in a
  sample of general web text (FineWeb-Edu, the hosts' kind of pretraining text). The counts come from a fixed
  whole-token matcher (`MentionCounter`; lowercase alphanumeric runs and single punctuation marks, so "N,N-dimethyl"
  and "n, n - dimethyl" agree), are written with the selection (`selection.tsv`) and pinned by its sha256 before the
  linker is built, so the linker is frozen before any linking.
- **Frames** — SCR: `mapped_to` (its "heading mapped to" descriptors: the class it is indexed under, e.g.
  teclistamab → Antibodies, Bispecific), `pharmacological_action`, `record_class` (chemical / disease / organism / …)
  and the `branch_top` / `branch_second` tree categories of its mapped headings; descriptor: `parent`,
  `pharmacological_action`, `see_also` and its branches (as T1). Fillers are descriptors from a bounded dictionary
  (most-used first; an out-of-dictionary filler falls back to its nearest in-dictionary tree ancestor), so a new
  name is composed from fillers the host knows well.

The linked entries are the selected records only; the descriptors appear as fillers.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .mesh import FUNCTION_WORDS, _parents, _parse
from .wordnet import FrameOntology

SCR_CLASSES = {"1": "chemical", "2": "protocol", "3": "disease", "4": "organism", "5": "population group", "6": "anatomy"}
RELATIONS = ["mapped_to", "parent", "pharmacological_action", "see_also", "record_class", "branch_top", "branch_second"]
TOKEN = re.compile(r"[a-z0-9]+|[^a-z0-9\s]")


# -- parsing -------------------------------------------------------------------------------------------------------

def _xml_date(element: ET.Element | None) -> str | None:
    """`<Year>/<Month>/<Day>` → "YYYY-MM-DD" (a missing month or day reads as 01); None without a year."""
    year = element.findtext("Year") if element is not None else None
    if not year:
        return None
    return f"{int(year):04d}-{int(element.findtext('Month') or 1):02d}-{int(element.findtext('Day') or 1):02d}"


def parse_supplementary(path: Path) -> list[dict[str, Any]]:
    """SCRs of a MeSH `supp<year>` XML file (gzip or plain): UI, name, class, year introduced (and the full date,
    `introduced_date`: the record-level `DateIntroduced`, which replaced `DateCreated` in the 2025+ XML), mapped headings
    (`*` = major heading, the marker dropped), pharmacological actions and every term with its lexical tag."""
    opener = gzip.open if Path(path).suffix == ".gz" else open
    records = []
    with opener(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag != "SupplementalRecord":
                continue
            mapped = [m.findtext("DescriptorReferredTo/DescriptorUI") or "" for m in
                      element.findall("HeadingMappedToList/HeadingMappedTo")]
            actions = [a.findtext("DescriptorReferredTo/DescriptorUI") for a in
                       element.findall("PharmacologicalActionList/PharmacologicalAction")]
            term_info = [{"string": t.findtext("String"), "permuted": t.get("IsPermutedTermYN") == "Y",
                          "lexical_tag": t.get("LexicalTag") or "NON",
                          "record_preferred": t.get("RecordPreferredTermYN") == "Y"}
                         for t in element.findall("ConceptList/Concept/TermList/Term") if t.findtext("String")]
            year = element.findtext("DateIntroduced/Year")
            records.append({"ui": element.findtext("SupplementalRecordUI"),
                            "name": element.findtext("SupplementalRecordName/String"),
                            "scr_class": element.get("SCRClass") or "", "introduced": int(year) if year else None,
                            "introduced_date": _xml_date(element.find("DateIntroduced")),
                            "mapped": [m.lstrip("*") for m in mapped if m], "mapped_major": [m.startswith("*") for m in mapped if m],
                            "actions": [a for a in actions if a], "term_info": term_info,
                            "note": " ".join((element.findtext("Note") or "").split())})
            element.clear()
    return records


# -- mention counting ----------------------------------------------------------------------------------------------

def mention_key(text: str) -> str:
    """The matcher's normal form: lowercase alphanumeric runs and single punctuation marks, space-joined."""
    return " ".join(TOKEN.findall(text.lower()))


class MentionCounter:
    """Whole-token occurrences of a fixed set of keys (`mention_key` forms) in texts, every occurrence counted
    (overlapping matches included: a frequency screen, not the linker)."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = {k for k in keys if k}
        self.longest: dict[str, int] = {}
        for key in self.keys:
            first, n = key.split(" ", 1)[0], key.count(" ") + 1
            self.longest[first] = max(self.longest.get(first, 0), n)

    def count(self, text: str, into: Counter | None = None) -> Counter:
        counts = into if into is not None else Counter()
        tokens = TOKEN.findall(text.lower())
        for i, token in enumerate(tokens):
            longest = self.longest.get(token)
            if not longest:
                continue
            for n in range(1, min(longest, len(tokens) - i) + 1):
                key = " ".join(tokens[i:i + n])
                if key in self.keys:
                    counts[key] += 1
        return counts

    def mentions(self, text: str) -> bool:
        """Whether `text` holds at least one key (stops at the first)."""
        tokens = TOKEN.findall(text.lower())
        for i, token in enumerate(tokens):
            longest = self.longest.get(token)
            if longest and any(" ".join(tokens[i:i + n]) in self.keys for n in range(1, min(longest, len(tokens) - i) + 1)):
                return True
        return False


def count_mentions(texts: Iterable[str], keys: Iterable[str]) -> tuple[Counter, int]:
    """(occurrences per key, documents read)."""
    counter = MentionCounter(keys)
    counts: Counter = Counter()
    documents = 0
    for text in texts:
        counter.count(text, counts)
        documents += 1
    return counts, documents


_WORKER_COUNTER: MentionCounter | None = None


def _init_counter(keys: list[str]) -> None:
    global _WORKER_COUNTER
    _WORKER_COUNTER = MentionCounter(keys)


def _count_pubmed_file(job: tuple[str, int, frozenset[int]]) -> tuple[Counter, int]:
    """Training side of one extracted PubMed file: the rows assigned to it (`skip` = PMIDs read earlier in stream
    order, excluded or below the minimum), with PMID bucket ≥ `eval_buckets`, each PMID once."""
    import pyarrow.parquet as pq
    from vsa_embed.data.pubmed import pmid_bucket
    path, eval_buckets, skip = job
    counts: Counter = Counter()
    documents, seen = 0, set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["pmid", "text"], batch_size=2048):
        data = batch.to_pydict()
        for pmid, text in zip(data["pmid"], data["text"]):
            pmid = int(pmid)
            if pmid in seen or pmid in skip or pmid_bucket(pmid) < eval_buckets:
                continue
            seen.add(pmid)
            _WORKER_COUNTER.count(text, counts)
            documents += 1
    return counts, documents


def _count_rows(job: tuple[str, int, int]) -> tuple[Counter, int]:
    """Rows [first, last) of a parquet file's `text` column."""
    import pyarrow.parquet as pq
    path, first, last = job
    counts: Counter = Counter()
    parquet = pq.ParquetFile(path)
    position, documents = 0, 0
    for group in range(parquet.num_row_groups):
        rows = parquet.metadata.row_group(group).num_rows
        if position + rows > first and position < last:
            texts = parquet.read_row_group(group, columns=["text"]).column(0).to_pylist()
            for i, text in enumerate(texts):
                if first <= position + i < last:
                    _WORKER_COUNTER.count(text, counts)
                    documents += 1
        position += rows
        if position >= last:
            break
    return counts, documents


def _pool_counts(jobs: list[Any], function: Any, keys: Sequence[str], workers: int) -> tuple[Counter, int]:
    total: Counter = Counter()
    documents = 0
    if workers <= 1:
        _init_counter(list(keys))
        results = map(function, jobs)
        for counts, n in results:
            total.update(counts); documents += n
        return total, documents
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init_counter,
                             initargs=(list(keys),)) as pool:
        for counts, n in pool.map(function, jobs):          # in job order: the sums do not depend on scheduling
            total.update(counts); documents += n
    return total, documents


def count_pubmed_mentions(paths: Sequence[Path], keys: Sequence[str], *, eval_buckets: int,
                          exclude: frozenset[int] = frozenset(), min_pmid: int | None = None,
                          workers: int = 4) -> tuple[Counter, int]:
    """Occurrences of `keys` in the training side of the extracted PubMed files, read as
    `t1_open_corpus.pubmed_documents(split="train")` reads them: in file order, each PMID once (its first file; an
    update file can hold a revised record of a PMID in a later file), `exclude`d PMIDs and PMIDs below `min_pmid`
    skipped. One job per file; each job skips the PMIDs of earlier files."""
    import pyarrow.parquet as pq
    jobs, earlier = [], set(int(p) for p in exclude)
    for path in paths:
        pmids = {int(p) for p in pq.read_table(path, columns=["pmid"]).column(0).to_pylist()}
        low = {p for p in pmids if min_pmid is not None and p < min_pmid}
        jobs.append((str(path), int(eval_buckets), frozenset((pmids & earlier) | low)))
        earlier |= pmids
    return _pool_counts(jobs, _count_pubmed_file, keys, workers)


def count_general_mentions(shards: Sequence[str], keys: Sequence[str], *, skip: int, limit: int, workers: int = 4,
                           rows_per_job: int = 20_000) -> tuple[Counter, int]:
    """Occurrences of `keys` in documents [skip, skip + limit) of the concatenated general shards (the C3 stream
    order: `c3_corpus.iter_texts(shards, skip=skip, limit=limit)`), in row-range jobs."""
    import pyarrow.parquet as pq
    jobs: list[tuple[str, int, int]] = []
    start, end, offset = skip, skip + limit, 0
    for shard in shards:
        rows = pq.ParquetFile(shard).metadata.num_rows
        lo, hi = max(start, offset), min(end, offset + rows)
        for first in range(lo, hi, rows_per_job):
            jobs.append((str(shard), first - offset, min(hi, first + rows_per_job) - offset))
        offset += rows
        if offset >= end:
            break
    return _pool_counts(jobs, _count_rows, keys, workers)


# -- candidate aliases and the frequency selection -----------------------------------------------------------------

@dataclass(frozen=True)
class NovelVocabularyPolicy:
    """Which MeSH terms are candidates, and the frequency selection (defaults: the pre-registered T7 policy)."""

    scr_classes: tuple[str, ...] = ("1", "3", "4")          # chemicals and drugs, diseases, organisms
    include_descriptors: bool = True
    descriptor_min_year: int | None = None
    scr_min_year: int | None = None
    exclude_descriptor_classes: tuple[str, ...] = ("2",)    # publication types
    drop_lexical_tags: tuple[str, ...] = ("ABB", "ACR")     # bare abbreviations / acronyms (unless the record's name)
    min_chars: int = 4
    min_domain_mentions: int = 5
    max_general_mentions: int = 0
    function_words: frozenset[str] = field(default=FUNCTION_WORDS)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "NovelVocabularyPolicy":
        config = dict(config or {})
        keys = {"scr_classes", "include_descriptors", "descriptor_min_year", "scr_min_year", "exclude_descriptor_classes",
                "drop_lexical_tags", "min_chars", "min_domain_mentions", "max_general_mentions"}
        unknown = set(config) - keys
        if unknown:
            raise ValueError(f"unknown policy keys {sorted(unknown)}")
        values = {k: tuple(str(x) for x in v) if isinstance(v, list) else v for k, v in config.items()}
        return cls(**values)


def candidate_aliases(descriptors: Sequence[dict[str, Any]], scrs: Sequence[dict[str, Any]],
                      policy: NovelVocabularyPolicy = NovelVocabularyPolicy()) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """(alias text, record UI) pairs of every candidate term, and counts of each decision."""
    stats: Counter[str] = Counter()
    pairs: list[tuple[str, str]] = []

    def take(record: dict[str, Any]) -> None:
        kept = set()
        for term in record["term_info"]:
            text = " ".join(term["string"].split())
            key = mention_key(text)
            if len(text) < policy.min_chars:
                stats["dropped_short"] += 1; continue
            if text.lower() in policy.function_words:
                stats["dropped_function_word"] += 1; continue
            if not re.search(r"[a-z]", key):
                stats["dropped_no_letter"] += 1; continue
            if term["lexical_tag"] in policy.drop_lexical_tags and not term["record_preferred"]:
                stats["dropped_abbreviation"] += 1; continue
            if text.lower() not in kept:
                kept.add(text.lower())
                pairs.append((text, record["ui"]))
        stats["candidate_terms"] += len(kept)

    if policy.include_descriptors:
        for record in descriptors:
            if record["descriptor_class"] in policy.exclude_descriptor_classes:
                stats["descriptors_excluded_class"] += 1; continue
            if policy.descriptor_min_year and (record.get("introduced") or 0) < policy.descriptor_min_year:
                stats["descriptors_before_min_year"] += 1; continue
            stats["candidate_descriptors"] += 1
            take(record)
    for record in scrs:
        if record["scr_class"] not in policy.scr_classes:
            stats["scrs_excluded_class"] += 1; continue
        if policy.scr_min_year and (record.get("introduced") or 0) < policy.scr_min_year:
            stats["scrs_before_min_year"] += 1; continue
        stats["candidate_scrs"] += 1
        take(record)
    return pairs, dict(stats)


def select_aliases(pairs: Sequence[tuple[str, str]], domain: Counter, general: Counter,
                   policy: NovelVocabularyPolicy = NovelVocabularyPolicy()) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Selection rows (alias, UI, key, domain and general counts), sorted by (UI, alias).

    The rule is per **record**, so that a concept the host knows under its usual name never enters through a rare
    spelling variant: a record is selected if the general occurrences of all its candidate names together are
    ≤ `max_general_mentions` and their domain occurrences together are ≥ `min_domain_mentions`; its selected aliases
    are the names that occur in the domain text at least once. A key shared by several records is dropped (the T7
    linker never merges records) and does not count for any of them."""
    owners: dict[str, set[str]] = {}
    for alias, ui in pairs:
        owners.setdefault(mention_key(alias), set()).add(ui)
    stats: Counter[str] = Counter()
    by_record: dict[str, list[tuple[str, str]]] = {}
    for alias, ui in pairs:
        key = mention_key(alias)
        if len(owners[key]) > 1:
            stats["dropped_shared_key"] += 1; continue
        by_record.setdefault(ui, []).append((alias, key))
    rows = []
    for ui, names in by_record.items():
        keys = {key for _, key in names}
        general_total = sum(int(general.get(k, 0)) for k in keys)
        domain_total = sum(int(domain.get(k, 0)) for k in keys)
        if general_total > policy.max_general_mentions:
            stats["records_general_frequent"] += 1; continue
        if domain_total < policy.min_domain_mentions:
            stats["records_domain_rare"] += 1; continue
        for alias, key in names:
            d = int(domain.get(key, 0))
            if d == 0:
                stats["aliases_unseen_in_domain"] += 1; continue
            rows.append({"alias": alias, "ui": ui, "key": key, "domain": d, "general": int(general.get(key, 0))})
    rows.sort(key=lambda r: (r["ui"], r["alias"]))
    stats["selected_aliases"] = len(rows)
    stats["selected_records"] = len({r["ui"] for r in rows})
    return rows, dict(stats)


SELECTION_FIELDS = ("ui", "alias", "key", "domain", "general")


def write_selection(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write the selection as TSV (sorted rows); returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SELECTION_FIELDS)
        for row in rows:
            writer.writerow([row[k] for k in SELECTION_FIELDS])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_selection(path: Path, *, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"{path}: sha256 {digest} != pinned {expected_sha256} (the frozen selection changed)")
    lines = data.decode().splitlines()
    reader = csv.DictReader(lines, delimiter="\t")
    return [{**row, "domain": int(row["domain"]), "general": int(row["general"])} for row in reader]


# -- the frame ontology --------------------------------------------------------------------------------------------

def build_novel_ontology(descriptors: Sequence[dict[str, Any]], scrs: Sequence[dict[str, Any]],
                         selection: Sequence[dict[str, Any]], *, max_atomics: int = 8192, max_degree: int = 16,
                         source: dict[str, Any] | None = None) -> FrameOntology:
    """Concepts = the selected records (sorted by UI); aliases = their selected terms; frames over descriptors."""
    by_ui = {r["ui"]: r for r in descriptors}
    scr_by_ui = {r["ui"]: r for r in scrs}
    parents = _parents(list(descriptors))
    chosen = sorted({row["ui"] for row in selection})
    missing = [ui for ui in chosen if ui not in by_ui and ui not in scr_by_ui]
    if missing:
        raise ValueError(f"selected records not in the MeSH files: {missing[:5]}")

    def raw_edges(ui: str) -> list[tuple[str, str]]:
        """(relation, filler) before the dictionary bound: fillers are descriptor UIs, `branch:X` or `class:X`."""
        edges: list[tuple[str, str]] = []
        if ui in scr_by_ui:
            record = scr_by_ui[ui]
            headings = [m for m in record["mapped"] if m in by_ui]
            edges += [("mapped_to", m) for m in headings]
            edges += [("pharmacological_action", a) for a in record["actions"] if a in by_ui]
            edges.append(("record_class", f"class:{SCR_CLASSES.get(record['scr_class'], record['scr_class'])}"))
            trees = [t for m in headings for t in by_ui[m]["trees"]]
        else:
            record = by_ui[ui]
            edges += [("parent", p) for p in parents[ui]]
            edges += [("pharmacological_action", a) for a in record["actions"] if a in by_ui]
            edges += [("see_also", r) for r in record["related"] if r in by_ui]
            trees = record["trees"]
        for tree in trees:
            edges += [("branch_top", f"branch:{tree[0]}"), ("branch_second", f"branch:{tree.split('.')[0]}")]
        return edges

    edges = {ui: raw_edges(ui) for ui in chosen}
    use: Counter[str] = Counter(f for es in edges.values() for _, f in es if not f.startswith(("branch:", "class:")))
    symbols = sorted({f for es in edges.values() for _, f in es if f.startswith(("branch:", "class:"))})
    budget = max(0, max_atomics - len(symbols))
    fillers = [ui for ui, _ in sorted(use.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]]
    atomic_names = [f"mesh:{ui}" for ui in fillers] + symbols
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    cache: dict[str, int | None] = {}

    def atom(filler: str, depth: int = 0) -> int | None:
        if filler.startswith(("branch:", "class:")):
            return atomic_index.get(filler)
        if f"mesh:{filler}" in atomic_index:
            return atomic_index[f"mesh:{filler}"]
        if filler in cache:
            return cache[filler]
        found = None
        if depth < 12:
            for parent in parents.get(filler, []):
                found = atom(parent, depth + 1)
                if found is not None:
                    break
        cache[filler] = found
        return found

    relation_index = {name: i for i, name in enumerate(RELATIONS)}
    frames: list[list[tuple[int, int]]] = []
    for ui in chosen:
        frame, seen = [], set()
        for relation, filler in edges[ui]:
            index = atom(filler)
            edge = (relation_index[relation], index)
            if index is not None and edge not in seen and len(frame) < max_degree:
                seen.add(edge); frame.append(edge)
        frames.append(frame)
    concept = {ui: i for i, ui in enumerate(chosen)}
    alias_pairs = [(row["alias"], concept[row["ui"]]) for row in selection]
    names = [scr_by_ui[ui]["name"] if ui in scr_by_ui else by_ui[ui]["name"] for ui in chosen]
    ontology = FrameOntology("mesh-novel", chosen, list(RELATIONS), atomic_names, frames, alias_pairs,
                             {"source": dict(source or {}), "records": len(chosen)})
    ontology.metadata.update({
        "headings": names,
        "kind": ["scr" if ui in scr_by_ui else "descriptor" for ui in chosen],
        "record_class": [SCR_CLASSES.get(scr_by_ui[ui]["scr_class"], "") if ui in scr_by_ui else "descriptor" for ui in chosen],
        "introduced": [(scr_by_ui[ui] if ui in scr_by_ui else by_ui[ui]).get("introduced") for ui in chosen],
        "filler_headings": {ui: by_ui[ui]["name"] for ui in fillers},
        "max_atomics": max_atomics, "max_degree": max_degree,
        "empty_frames": sum(1 for f in frames if not f),
    })
    return ontology


def load_mesh_records(descriptor_path: Path, supplementary_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return sorted(_parse(Path(descriptor_path)), key=lambda r: r["ui"]), sorted(parse_supplementary(Path(supplementary_path)),
                                                                              key=lambda r: r["ui"])
