"""T8: the Wikidata-framed natural track (decision 63, WP TK-B2; methodology M5) with the M1 holdout ("ROOD-text").

Every result built from this corpus is labelled **"Wikidata entities (Wikidata + FineWeb-Edu)"**. The ontology is
`ontologies/wikidata.py` (frames from Wikidata truthy statements of a fixed property list); the text is FineWeb-Edu
(the two local sample-10BT shards of C3); the linking, holdout closure, host relinks, feasibility tables and
`ontology.pt` are T1-open's (`t1_open_corpus`: `relink_for_host`, `choose_track_holdout`, `assert_alias_disjoint`,
`feasibility_report`), with a FineWeb document stream (`EntityDocuments`) in place of PubMed.

Stages (`--stage`):

- `wikidata`: the frozen Wikidata snapshot `records.jsonl.gz` (`wikidata.dir`): the benchmark entities
  (`t8_benchmarks entities`), redirects followed; English label, description, aliases, sitelink count, English Wikipedia
  title and truthy item-valued statements of `wikidata.PROPERTIES`; then terms for every filler of those statements and
  statements for the fillers with ≥ `wikidata.min_filler_sitelinks` sitelinks (concept candidates too: "the fillers'
  entities"), and terms for their fillers, so every dictionary filler has a label. The config pins its sha256 (`wikidata.records_sha256`) before the screen.
- `screen`: every name of every record is counted in the FineWeb-Edu stream (`wikidata.count_fineweb_mentions`;
  documents after C3's evaluation documents; per side of the document-id evaluation split; occurrences not written with
  the name's capitals apart), the homonyms of every candidate label found in the text are looked up (other Wikidata
  items with the same English label and their sitelinks; cached), the alias policy is applied and at most `ontology.max_concepts` entities are selected, entities mentioned at
  least `ontology.min_entity_mentions` times first (`wikidata.select_concepts`). Writes `selection.tsv` (pinned by
  `ontology.selection_sha256` before `build`), `screen.json`, and the per-name counts (`screen_counts.tsv.gz`, under
  `wikidata.dir`; the holdout's exclusion cost reads it).
- `build`:
  1. adapter → frames and the alias table (`wikidata_adapter`);
  2. FineWeb streams (`EntityDocuments`), in C3 stream order after C3's evaluation documents: a document is
     evaluation-side when its id's sha256 bucket is < `data.eval_buckets` / 10,000. Training text interleaves
     training-side documents that mention a selected name (`entities`) with training-side documents that mention none
     (`general`) so that `mix.domain_share` of the tokens are `entities` (chars/token calibrated on evaluation
     documents). Evaluation: `eval-entities` = evaluation-side documents that mention a selected name **and** every
     document that mentions a held-out entity (ROOD: they are never trained on), in stream order;
     `eval-general` = C3's evaluation documents (locality); `eval` = their mix.
  3. pre-sample (first `presample_tokens` of the training mix, before any exclusion) → frozen holdout: T1-open's
     stratified choice + closure (node-, alias-disjoint) over **ROOD-eligible** entries — single-concept entries whose
     entity is no frame filler (node-disjoint: no other frame names it) and whose Wikidata names (all of them, ≥
     `data.rood_min_chars` characters) occur in at most `data.rood_max_documents` training-side screen documents (the
     exclusion cost), and whose closure stays eligible. Pinned by `expected_holdout_sha256`.
  4. **document exclusion (ROOD)**: every training document (both streams) that mentions a held-out entity under any of
     its Wikidata names is dropped (`exclusion_keys`).
  5. corpora for the reference tokenizer and every `hosts` relink (`relink_for_host`); `ontology.pt`;
  6. **leakage audit**: the realized training documents re-read in order — exclusion names found (must be 0), document
     ids shared with `eval-entities` (must be 0), and held-out spans when the training documents are linked with the
     *full* alias table (must be 0; `audit` in `summary.json`);
  7. cardinality, feasibility per ℓ_min on `eval-entities`, report.

    PYTHONPATH=src python -m vsa_embed.experiments.t8_wikidata_corpus --config experiments/t8-wikidata/t8.yaml --stage wikidata
    PYTHONPATH=src python -m vsa_embed.experiments.t8_wikidata_corpus --config experiments/t8-wikidata/t8.yaml --stage screen --output experiments/t8-wikidata/screen
    PYTHONPATH=src python -m vsa_embed.experiments.t8_wikidata_corpus --config experiments/t8-wikidata/t8.yaml --output experiments/t8-wikidata/runs/v1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import islice
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Sequence

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.experiments.c3_corpus import iter_texts
from vsa_embed.experiments.t1_open_corpus import (alias_statistics, assert_alias_disjoint, chars_per_token, choose_track_holdout,
                                                  containment_index, feasibility_report, general_settings, guard_reuse,
                                                  holdout_closure, load_config, mix_documents, record_shares, relink_for_host,
                                                  train_frequency, verify_shards)
from vsa_embed.ontologies import wikidata as wd
from vsa_embed.ontologies.wordnet import FrameOntology
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import AliasTable, alias_subtoken_lengths, cardinality_report

TRACK_LABEL = "Wikidata entities (Wikidata + FineWeb-Edu)"
SOURCES = ("entities", "general")
DOMAIN_SPLIT = "eval-entities"
SOURCE_ORDER = ("entity_inferences", "ecbd", "lre", "bear", "popqa", "twohopfact", "filler")
COUNT_FIELDS = ("key", "train", "eval", "train_documents", "eval_documents", "lowercase")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- ontology ---------------------------------------------------------------------------------------------------------

def records_path(config: dict[str, Any]) -> Path:
    return Path(config["wikidata"]["dir"]).expanduser() / "records.jsonl.gz"


def load_records(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return wd.read_records(records_path(config), expected_sha256=config["wikidata"].get("records_sha256"))


def wikidata_adapter(config: dict[str, Any], records: dict[str, dict[str, Any]] | None = None) -> FrameOntology:
    """The `ontology` section (plus the top-level `wikidata` paths, merged in by `build_track_ontology`) → FrameOntology."""
    selection = wd.read_selection(Path(config["selection"]), expected_sha256=config.get("selection_sha256"))
    records = records if records is not None else wd.read_records(Path(config["records"]).expanduser(),
                                                                  expected_sha256=config.get("records_sha256"))
    return wd.build_wikidata_ontology(records, selection, properties=config.get("properties"),
                                      max_atomics=int(config["max_atomics"]), max_degree=int(config["max_degree"]),
                                      max_values=int(config.get("max_values", 3)),
                                      max_values_by_property=config.get("max_values_by_property"),
                                      source={"records": config["records"], "records_sha256": config.get("records_sha256"),
                                              "selection": config["selection"], "selection_sha256": config.get("selection_sha256")})


def ontology_config(config: dict[str, Any]) -> dict[str, Any]:
    """The adapter's view: the `ontology` section with the records path and its pin."""
    return {**config["ontology"], "records": str(records_path(config)), "records_sha256": config["wikidata"].get("records_sha256")}


def build_track_ontology(config: dict[str, Any], records: dict[str, dict[str, Any]] | None = None) -> FrameOntology:
    """T8's ontology from the full track config (the replay `e9_tracks.track_frame_ontology` uses)."""
    return wikidata_adapter(ontology_config(config), records)


# -- documents ---------------------------------------------------------------------------------------------------------

def iter_records(shards: Sequence[str], *, skip: int = 0) -> Iterator[tuple[str, str]]:
    """(document id, text) in C3 stream order (the shards concatenated), documents before `skip` left out."""
    import pyarrow.parquet as pq
    seen = 0
    for path in shards:
        parquet = pq.ParquetFile(path)            # kept referenced while iterating (see c3_corpus.iter_texts)
        for batch in parquet.iter_batches(columns=["id", "text"], batch_size=2048):
            ids, texts = batch.column(0).to_pylist(), batch.column(1).to_pylist()
            for doc_id, text in zip(ids, texts):
                seen += 1
                if seen <= skip:
                    continue
                yield doc_id, text


@dataclass
class EntityDocuments:
    """Factories for every document stream of T8 (fresh iterator per call); `relink_for_host`'s interface."""

    shards: list[str]
    skip: int                                   # C3 evaluation documents: never trained on; `eval-general`
    eval_buckets: int
    eval_general_docs: int
    mention_keys: list[str]                     # the selected names (`entity_key` forms): the `entities` filter
    eval_domain_docs: int | None = None         # evaluation-side documents that mention a selected name
    eval_rood_docs: int | None = None           # documents that name a held-out entity (ROOD), any side
    exclusion_keys: list[str] | None = None     # ROOD: names of the held-out entities (None before the holdout)
    domain_share: float = 0.5
    calibration: tuple[float, float] = (4.6, 4.6)
    label: str = TRACK_LABEL
    domain_split: ClassVar[str] = DOMAIN_SPLIT
    sources: ClassVar[tuple[str, ...]] = SOURCES
    _mention: Any = field(default=None, repr=False)
    _exclude: Any = field(default=None, repr=False)

    @property
    def track_label(self) -> str:
        return self.label

    def mention(self) -> wd.KeyTrie:
        if self._mention is None:
            self._mention = wd.KeyTrie(self.mention_keys)
        return self._mention

    def exclude(self) -> wd.KeyTrie | None:
        if self.exclusion_keys is None:
            return None
        if self._exclude is None:
            self._exclude = wd.KeyTrie(self.exclusion_keys)
        return self._exclude

    def with_exclusion(self, keys: Sequence[str]) -> "EntityDocuments":
        return replace(self, exclusion_keys=sorted(set(keys)), _exclude=None)

    @staticmethod
    def _digest(keys: Sequence[str] | None) -> str | None:
        return None if keys is None else hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()

    def signature(self, stream: str) -> str:
        spec: dict[str, Any] = {"stream": stream, "general": [Path(s).name for s in self.shards], "skip": self.skip}
        if stream != "eval-general":
            spec.update(eval_buckets=self.eval_buckets, mentions=self._digest(self.mention_keys),
                        exclusion=self._digest(self.exclusion_keys))
        if stream in ("train", "eval", "presample"):
            spec.update(domain_share=self.domain_share, calibration=[round(c, 6) for c in self.calibration])
        if stream in ("eval", DOMAIN_SPLIT):
            spec.update(eval_domain_docs=self.eval_domain_docs, eval_rood_docs=self.eval_rood_docs)
        if stream in ("eval", "eval-general"):
            spec["eval_general_docs"] = self.eval_general_docs
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    # Every stream classifies each document once: side (bucket), mention (any selected name), excluded (any held-out name).
    def classified(self) -> Iterator[tuple[str, str, bool, bool, bool]]:
        """(id, text, evaluation side, mentions a selected name, mentions a held-out name) in stream order."""
        mention, exclude = self.mention(), self.exclude()
        for doc_id, text in iter_records(self.shards, skip=self.skip):
            evaluation = wd.doc_bucket(doc_id) < self.eval_buckets
            excluded = exclude is not None and exclude.first(text)
            yield doc_id, text, evaluation, (not excluded) and mention.first(text), excluded

    def train_records(self, kind: str) -> Iterator[tuple[str, str]]:
        """(id, text) of the training side of one stream: `entities` (mentions a selected name) or `general` (none);
        ROOD-excluded documents are dropped from both."""
        want = kind == "entities"
        for doc_id, text, evaluation, mentions, excluded in self.classified():
            if evaluation or excluded or mentions != want:
                continue
            yield doc_id, text

    def train_entities(self) -> Iterator[str]:
        return (text for _, text in self.train_records("entities"))

    def train_general(self) -> Iterator[str]:
        return (text for _, text in self.train_records("general"))

    def eval_records(self) -> Iterator[tuple[str, str]]:
        """`eval-entities`, in stream order: the first `eval_domain_docs` evaluation-side documents that mention a selected
        name, and the first `eval_rood_docs` documents that name a held-out entity (training-side ones included: ROOD keeps
        them out of training); the stream ends when both quotas are met."""
        bucket, rood = 0, 0

        def full(count: int, limit: int | None) -> bool:
            return limit is not None and count >= limit

        for doc_id, text, evaluation, mentions, excluded in self.classified():
            if full(bucket, self.eval_domain_docs) and (self.exclusion_keys is None or full(rood, self.eval_rood_docs)):
                return
            if excluded:
                if not full(rood, self.eval_rood_docs):
                    rood += 1
                    yield doc_id, text
            elif evaluation and mentions and not full(bucket, self.eval_domain_docs):
                bucket += 1
                yield doc_id, text

    def eval_domain(self) -> Iterator[str]:
        return (text for _, text in self.eval_records())

    def eval_general(self) -> Iterator[str]:
        return iter_texts(self.shards, limit=self.eval_general_docs)

    def mixed(self, domain: Iterator[str], general: Iterator[str], log: list[int] | None = None) -> Iterator[str]:
        return mix_documents([domain, general], [self.domain_share, 1.0 - self.domain_share], list(self.calibration), log=log)

    def train(self, log: list[int] | None = None) -> Iterator[str]:
        return self.mixed(self.train_entities(), self.train_general(), log)

    def eval_mixed(self, log: list[int] | None = None) -> Iterator[str]:
        return self.mixed(self.eval_domain(), self.eval_general(), log)

    def train_ids(self) -> Iterator[tuple[str, str, int]]:
        """(id, text, source) of the training mix in the mixer's order (the audit re-reads the realized documents)."""
        log: list[int] = []
        entities = self.train_records("entities")
        general = self.train_records("general")
        ids: list[list[str]] = [[], []]

        def texts(stream: Iterator[tuple[str, str]], index: int) -> Iterator[str]:
            for doc_id, text in stream:
                ids[index].append(doc_id)
                yield text

        cursor = [0, 0]
        for text in self.mixed(texts(entities, 0), texts(general, 1), log):
            source = log[-1]
            doc_id = ids[source][cursor[source]]
            cursor[source] += 1
            yield doc_id, text, source


def track_documents(config: dict[str, Any], mention_keys: list[str], *, calibration: tuple[float, float] | None = None
                    ) -> EntityDocuments:
    general, data = general_settings(config), config["data"]
    if int(data["eval_general_docs"]) > general["eval_docs"]:
        raise ValueError("eval_general_docs exceeds the C3 evaluation documents")
    return EntityDocuments(shards=general["shards"], skip=general["eval_docs"], eval_buckets=int(data["eval_buckets"]),
                           eval_general_docs=int(data["eval_general_docs"]), mention_keys=mention_keys,
                           eval_domain_docs=data.get("eval_domain_docs"), eval_rood_docs=data.get("eval_rood_docs"),
                           domain_share=float(config["mix"]["domain_share"]),
                           calibration=calibration or (4.6, 4.6), label=str(config.get("track_label") or TRACK_LABEL))


# -- stage `wikidata` --------------------------------------------------------------------------------------------------

def run_wikidata(config: dict[str, Any], *, log: Callable[[str], None] = _log) -> dict[str, Any]:
    """The frozen Wikidata snapshot (`records.jsonl.gz`) → summary (also `records.json`)."""
    from .t8_benchmarks import wikidata_client
    spec = config["wikidata"]
    out = Path(spec["dir"]).expanduser()
    client = wikidata_client(config, log=log)
    rows = [json.loads(line) for line in (out / "benchmark_entities.jsonl").read_text().splitlines() if line.strip()]
    benchmark = [r["qid"] for r in rows]
    properties = [p for p, _, _ in wd.PROPERTIES]
    redirects = wd.fetch_redirects(client, benchmark, progress=log)
    entities = sorted({redirects.get(q, q) for q in benchmark}, key=wd.qid_number)
    log(f"wikidata: {len(benchmark):,} benchmark QIDs, {len(redirects):,} redirected → {len(entities):,} entities")
    (out / "redirects.json").write_text(json.dumps(dict(sorted(redirects.items())), indent=0) + "\n")
    terms = wd.fetch_terms(client, entities, progress=log)
    claims = wd.fetch_claims(client, entities, properties, progress=log)
    records: dict[str, dict[str, Any]] = {q: {**terms[q], "claims": claims[q], "kind": "benchmark"} for q in entities}
    frontier = sorted({v for q in entities for vs in claims[q].values() for v in vs} - set(records), key=wd.qid_number)
    rounds = int(spec.get("filler_rounds", 1))
    min_sitelinks = int(spec.get("min_filler_sitelinks", 0))
    for depth in range(1, rounds + 1):
        log(f"wikidata: filler round {depth}: {len(frontier):,} entities")
        terms = wd.fetch_terms(client, frontier, progress=log)
        # Fillers with statements are concept candidates ("the fillers' entities"); a filler with fewer sitelinks keeps its
        # label only (it can still be a frame filler).
        known = [q for q in frontier if int(terms[q].get("sitelinks") or 0) >= min_sitelinks]
        claims = wd.fetch_claims(client, known, properties, progress=log)
        for q in frontier:
            records[q] = {**terms[q], "claims": claims.get(q, {}), "kind": f"filler{depth}" if q in claims else "label_only"}
        frontier = sorted({v for q in known for vs in claims[q].values() for v in vs} - set(records), key=wd.qid_number)
    log(f"wikidata: labels of the last fillers: {len(frontier):,} entities")
    terms = wd.fetch_terms(client, frontier, progress=log)
    for q in frontier:
        records[q] = {**terms[q], "claims": {}, "kind": "label_only"}
    digest = wd.write_records(records_path(config), records)
    kinds = Counter(r["kind"] for r in records.values())
    summary = {"records": len(records), "by_kind": dict(kinds), "records_sha256": digest,
               "with_label": sum(1 for r in records.values() if r.get("label")),
               "statements": sum(len(v) for r in records.values() for v in r["claims"].values()),
               "redirected": len(redirects), "properties": properties, "client": dict(client.stats),
               "retrieved": __import__("time").strftime("%Y-%m-%d")}
    (out / "records.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# -- stage `screen` ----------------------------------------------------------------------------------------------------

def benchmark_sources(config: dict[str, Any]) -> dict[str, list[str]]:
    """qid → benchmark roles (`bear:subject`, …) from `benchmark_entities.jsonl` (redirect targets carry their sources)."""
    out = Path(config["wikidata"]["dir"]).expanduser()
    rows = [json.loads(line) for line in (out / "benchmark_entities.jsonl").read_text().splitlines() if line.strip()]
    redirects = load_redirects(config)
    sources: dict[str, list[str]] = {}
    for r in rows:
        target = redirects.get(r["qid"], r["qid"])
        sources[target] = sorted(set(sources.get(target, [])) | set(r["roles"]))
    return sources


def load_redirects(config: dict[str, Any]) -> dict[str, str]:
    """Redirected (merged) benchmark QIDs → their targets (`redirects.json` of the `wikidata` stage)."""
    path = Path(config["wikidata"]["dir"]).expanduser() / "redirects.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_counts(path: Path, keys: Sequence[str], counts: wd.MentionCounts) -> str:
    """Per-name counts of the screen (gzip TSV, mtime 0) → sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wd.gzip_writer(path) as handle:
        handle.write(("\t".join(COUNT_FIELDS) + "\n").encode())
        for i, key in enumerate(keys):
            row = [key, counts.occurrences[0, i], counts.occurrences[1, i], counts.documents[0, i], counts.documents[1, i],
                   counts.lowercase[0, i]]
            handle.write(("\t".join(map(str, row)) + "\n").encode())
    return _sha256(path)


def read_counts(path: Path, *, expected_sha256: str | None = None) -> dict[str, dict[str, int]]:
    path = Path(path).expanduser()
    if expected_sha256 and _sha256(path) != expected_sha256:
        raise ValueError(f"{path}: sha256 differs from the pinned {expected_sha256}")
    out = {}
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            row = dict(zip(header, values))
            out[row["key"]] = {k: int(row[k]) for k in header if k != "key"}
    return out


def run_screen(config: dict[str, Any], output_dir: Path, *, log: Callable[[str], None] = _log) -> dict[str, Any]:
    records = load_records(config)
    spec, data = config["ontology"], config["data"]
    general = general_settings(config)
    verify_shards(general["shards"], general["shard_sha256"])
    keys, _, masks = wd.candidate_keys(records)
    settings = config.get("screen") or {}
    log(f"screen: {len(records):,} records, {len(keys):,} names; counting in FineWeb-Edu")
    counts = wd.count_fineweb_mentions(general["shards"], keys, skip=general["eval_docs"], eval_buckets=int(data["eval_buckets"]),
                                       limit=settings.get("docs"), workers=int(settings.get("workers", config.get("workers", 4))),
                                       masks=masks)
    counts_path = Path(config["wikidata"]["dir"]).expanduser() / "screen_counts.tsv.gz"
    counts_digest = write_counts(counts_path, keys, counts)
    table = {k: {"train": int(counts.occurrences[0, i]), "eval": int(counts.occurrences[1, i]),
                 "train_documents": int(counts.documents[0, i]), "lowercase": int(counts.lowercase[0, i])}
             for i, k in enumerate(keys)}
    sources = benchmark_sources(config)
    policy = wd.WikidataAliasPolicy.from_config(spec.get("aliases"))
    pool = [q for q, r in records.items() if r.get("label") and r.get("kind") != "label_only"]
    # Homonyms (other Wikidata items with exactly the same English label) of every candidate label that occurs in the
    # training-side text: the alias policy does not link a label a comparably known other item also bears.
    from .t8_benchmarks import wikidata_client
    labels = sorted({records[q]["label"] for q in pool if table.get(wd.entity_key(records[q]["label"]), {}).get("train", 0) >= 1})
    log(f"screen: homonyms of {len(labels):,} labels that occur in the text")
    homonyms = wd.fetch_homonyms(wikidata_client(config, log=log), labels, progress=log)
    for q in pool:
        others = [n for item, n in homonyms.get(records[q]["label"], []) if item != q]
        records[q]["homonym_sitelinks"] = max(others, default=0)
    rows, stats = wd.select_concepts(records, table, sources, policy=policy, max_concepts=int(spec["max_concepts"]),
                                     min_entity_mentions=int(spec["min_entity_mentions"]),
                                     source_order=list(spec.get("source_order") or SOURCE_ORDER), pool=pool)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = wd.write_selection(output_dir / "selection.tsv", rows)
    coverage = fineweb_coverage(records, table, sources, {r["qid"] for r in rows})
    selected = sorted({r["qid"] for r in rows}, key=wd.qid_number)
    by_benchmark = Counter(role.split(":")[0] for q in selected for role in sorted({r.split(":")[0] for r in sources.get(q, [])}))
    result = {"policy": {k: v for k, v in policy.__dict__.items() if k != "function_words"},
              "records": len(records), "names": len(keys), "documents_read": {"train": int(counts.read[0]), "eval": int(counts.read[1])},
              "selection": stats, "selection_sha256": digest, "counts": str(counts_path), "counts_sha256": counts_digest,
              "selected_by_benchmark": dict(by_benchmark), "fineweb_coverage": coverage, "homonym_labels_checked": len(labels),
              "labels_with_homonym": sum(1 for q in pool if records[q].get("homonym_sitelinks")),
              "selected_mentions": int(sum(r["train"] for r in rows)),
              "max_concepts": int(spec["max_concepts"]), "min_entity_mentions": int(spec["min_entity_mentions"])}
    (output_dir / "screen.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def fineweb_coverage(records: dict[str, dict[str, Any]], table: dict[str, dict[str, int]], sources: dict[str, list[str]],
                     selected: set[str]) -> dict[str, dict[str, int]]:
    """Per benchmark role (`bear:subject`, …): entities, how many any of whose Wikidata names occurs in the training-side
    FineWeb-Edu text at least once / at least 5 times, and how many are selected concepts."""
    out: dict[str, Counter] = {}
    for qid, roles in sources.items():
        record = records.get(qid)
        mentions = sum(table.get(wd.entity_key(n), {}).get("train", 0) for n in wd.entity_names(record)) if record else 0
        for role in roles:
            c = out.setdefault(role, Counter())
            c["entities"] += 1
            c["in_records"] += record is not None
            c["mentioned"] += mentions >= 1
            c["mentioned_5"] += mentions >= 5
            c["selected"] += qid in selected
    return {role: dict(c) for role, c in sorted(out.items())}


# -- holdout -------------------------------------------------------------------------------------------------------------

def exclusion_names(record: dict[str, Any], *, min_chars: int) -> list[str]:
    """The ROOD exclusion keys of an entity: every Wikidata name (label and aliases) of ≥ `min_chars` characters with a
    letter, in `entity_key` form."""
    keys = []
    for name in wd.entity_names(record):
        key = wd.entity_key(name)
        if len(name) >= min_chars and any(c.isalpha() for c in key):
            keys.append(key)
    return sorted(set(keys))


def rood_eligibility(table: AliasTable, ontology: FrameOntology, records: dict[str, dict[str, Any]],
                     key_counts: dict[str, dict[str, int]], *, min_chars: int, max_documents: int) -> dict[str, Any]:
    """Entries a ROOD holdout may choose: single-concept entries whose entity fills no frame (node-disjoint), whose
    exclusion names occur in ≤ `max_documents` training-side screen documents, and whose own closure (alias containment,
    shared concepts) keeps both properties."""
    fillers = {name.split(":", 1)[1] for name in ontology.atomic_names if name.startswith("wd:")}
    cost: dict[int, int] = {}
    ok: set[int] = set()
    reasons: Counter = Counter()
    for entry, concepts in enumerate(table.entry_concepts):
        if len(concepts) != 1:
            reasons["shared_entry"] += 1; continue
        qid = ontology.concept_names[concepts[0]]
        if qid in fillers:
            reasons["frame_filler"] += 1; continue
        cost[entry] = sum(key_counts.get(k, {}).get("train_documents", 0) for k in exclusion_names(records[qid], min_chars=min_chars))
        if cost[entry] > max_documents:
            reasons["exclusion_cost"] += 1; continue
        ok.add(entry)
    contained_in = containment_index(table)
    eligible = set()
    for entry in sorted(ok):
        closure = holdout_closure({entry}, table, contained_in)
        if closure <= ok:
            eligible.add(entry)
        else:
            reasons["closure_not_eligible"] += 1
    return {"eligible": eligible, "cost": cost, "reasons": dict(reasons), "fillers": fillers}


def choose_rood_holdout(counts: Counter, lengths: dict[int, int], table: AliasTable, ontology: FrameOntology,
                        records: dict[str, dict[str, Any]], key_counts: dict[str, dict[str, int]], *, fraction: float,
                        min_count: int, seed: int, max_containing: int | None, min_chars: int,
                        max_documents: int) -> dict[str, Any]:
    """T1-open's stratified holdout + closure over the ROOD-eligible entries (`rood_eligibility`); the result also holds
    the exclusion keys of every held-out entity and the eligibility counts. Node-disjointness is asserted."""
    eligibility = rood_eligibility(table, ontology, records, key_counts, min_chars=min_chars, max_documents=max_documents)
    eligible_counts = Counter({e: c for e, c in counts.items() if e in eligibility["eligible"]})
    holdout = choose_track_holdout(eligible_counts, lengths, table, ontology.concept_names, fraction=fraction, min_count=min_count,
                                   seed=seed, max_containing=max_containing)
    held_qids = [ontology.concept_names[c] for c in holdout["concepts"]]
    filling = sorted(set(held_qids) & eligibility["fillers"])
    if filling:
        raise AssertionError(f"held-out entities fill frames (not node-disjoint): {filling[:5]}")
    keys = sorted({k for q in held_qids for k in exclusion_names(records[q], min_chars=min_chars)})
    estimated = sum(eligibility["cost"].get(e, 0) for e in holdout["heldout_entries"])
    return {**holdout, "exclusion_keys": keys, "eligibility": {"eligible_entries": len(eligibility["eligible"]),
                                                                "eligible_with_count": len(eligible_counts),
                                                                "excluded": eligibility["reasons"]},
            "exclusion_cost_documents": int(estimated)}


# -- leakage audit -------------------------------------------------------------------------------------------------------

def leakage_audit(documents: EntityDocuments, builds: list[tuple[str, Path, dict[str, Any]]], *, full: AliasTable,
                  heldout_entries: Sequence[int], workers: int, scratch: Path) -> dict[str, Any]:
    """Re-read the realized training documents of every build (the longest prefix of the training mix) and check:
    held-out names found in them (ROOD filter), document ids shared with `eval-entities`, and held-out spans when they
    are linked with the full alias table (reference tokenizer). Each count must be 0."""
    exclude = documents.exclude()
    eval_ids = {doc_id for doc_id, _ in documents.eval_records()}
    longest = 0
    for _, root, _ in builds:
        manifest = json.loads((root / "train" / "manifest.json").read_text())
        longest = max(longest, int(manifest["documents"]) + int(manifest.get("skipped_documents", 0)))
    found_names, shared_ids, read = 0, 0, 0
    texts: list[str] = []
    for doc_id, text, _ in islice(documents.train_ids(), longest):
        read += 1
        if exclude is not None and exclude.first(text):
            found_names += 1
        if doc_id in eval_ids:
            shared_ids += 1
        texts.append(text)
    tokenizer_name = builds[0][0]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    scratch.mkdir(parents=True, exist_ok=True)
    linked = build_corpus(iter(texts), scratch, tokenizer_name=tokenizer_name, table=full, eos_id=tokenizer.eos_token_id,
                          max_tokens=10**12, workers=workers, vocab_size=len(tokenizer))
    spans = TokenCorpus.open(scratch).spans
    held = np.zeros(len(full.entry_concepts), dtype=bool)
    held[list(heldout_entries)] = True
    heldout_spans = int(held[spans["entry"].astype(np.int64)].sum())
    shutil.rmtree(scratch, ignore_errors=True)
    result = {"training_documents_read": read, "eval_entities_documents": len(eval_ids), "heldout_names_in_training": found_names,
              "training_ids_in_eval_entities": shared_ids, "heldout_spans_full_table": heldout_spans,
              "full_table_linked_tokens": int(linked["tokens"]), "full_table_spans": int(linked["spans"])}
    result["passed"] = found_names == 0 and shared_ids == 0 and heldout_spans == 0
    return result


# -- stage `build` -------------------------------------------------------------------------------------------------------

def run(config: dict[str, Any], output_dir: Path, *, log: Callable[[str], None] = _log) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir) if not (output_dir / "resolved_config.yaml").exists() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    data = config["data"]
    data_root = Path(config["paths"]["data_root"]).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    workers = int(config["workers"])
    tokenizer_name = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    general = general_settings(config)
    verify_shards(general["shards"], general["shard_sha256"])

    # 1. ontology and linker.
    records = load_records(config)
    ontology = build_track_ontology(config, records)
    base_table = AliasTable.from_pairs(ontology.alias_pairs)
    alias_lengths = alias_subtoken_lengths(base_table, tokenizer)
    entry_length: dict[int, int] = {}
    for alias, entry in base_table.alias_to_entry.items():
        entry_length[entry] = max(entry_length.get(entry, 0), alias_lengths[alias])
    alias_stats = alias_statistics(base_table, list(config["cardinality_tokenizers"]))
    selection = wd.read_selection(Path(config["ontology"]["selection"]), expected_sha256=config["ontology"].get("selection_sha256"))
    mention_keys = sorted({row["key"] for row in selection})
    log(f"build: {len(ontology.concept_names):,} concepts, {len(base_table.alias_to_entry):,} aliases, {ontology.metadata['edges']:,} edges")

    # 2. streams (no exclusion yet); chars/token calibration on evaluation documents.
    documents = track_documents(config, mention_keys)
    calibration_docs = int(config["mix"]["calibration_docs"])
    calibration = (chars_per_token(list(islice(documents.eval_domain(), calibration_docs)), tokenizer),
                   chars_per_token(list(islice(documents.eval_general(), calibration_docs)), tokenizer))
    documents.calibration = calibration

    # 3. pre-sample → frozen ROOD holdout.
    presample_log: list[int] = []
    presample_signature = documents.signature("presample")
    guard_reuse(data_root / "presample", base_table, int(data["presample_tokens"]), presample_signature)
    presample_manifest = build_corpus(documents.train(presample_log), data_root / "presample", tokenizer_name=tokenizer_name,
                                      table=base_table, eos_id=tokenizer.eos_token_id, max_tokens=int(data["presample_tokens"]),
                                      workers=workers, reuse=True,
                                      extra_manifest={"track": documents.track_label, "max_tokens_requested": int(data["presample_tokens"]),
                                                      "stream_signature": presample_signature})
    presample_shares = record_shares(data_root / "presample", presample_log, tokenizer.eos_token_id, SOURCES)
    presample = TokenCorpus.open(data_root / "presample")
    long_enough = presample.spans["length"] >= int(data["holdout_min_subtokens"])
    counts = Counter(presample.spans["entry"][long_enough].tolist())
    key_counts = read_counts(Path(config["wikidata"]["dir"]).expanduser() / "screen_counts.tsv.gz",
                             expected_sha256=config["ontology"].get("counts_sha256"))
    holdout = choose_rood_holdout(counts, entry_length, base_table, ontology, records, key_counts,
                                  fraction=float(data["holdout_fraction"]), min_count=int(data["holdout_min_count"]),
                                  seed=int(config["seed"]), max_containing=data.get("holdout_max_containing"),
                                  min_chars=int(data["rood_min_chars"]), max_documents=int(data["rood_max_documents"]))
    expected = data.get("expected_holdout_sha256")
    if expected and expected != holdout["sha256"]:
        raise ValueError(f"holdout sha256 {holdout['sha256']} != pinned {expected}: the frozen holdout changed")
    headings = ontology.metadata["headings"]
    chosen = set(holdout["chosen_entries"])
    (output_dir / "holdout_concepts.txt").write_text("\n".join(holdout["names"]) + "\n")
    rows = ["entry\tconcept\theading\treason\tpresample_count"]
    for entry in holdout["heldout_entries"]:
        for concept in base_table.entry_concepts[entry]:
            rows.append(f"{entry}\t{ontology.concept_names[concept]}\t{headings[concept]}\t"
                        f"{'chosen' if entry in chosen else 'closure'}\t{counts.get(entry, 0)}")
    (output_dir / "holdout_entries.tsv").write_text("\n".join(rows) + "\n")
    (output_dir / "exclusion_keys.txt").write_text("\n".join(holdout["exclusion_keys"]) + "\n")
    log(f"build: holdout {len(holdout['heldout_entries']):,} entries ({holdout['eligibility']}), "
        f"{len(holdout['exclusion_keys']):,} exclusion names")

    # 4. ROOD: the training streams drop every document that names a held-out entity.
    documents = documents.with_exclusion(holdout["exclusion_keys"])
    full = AliasTable.from_pairs(ontology.alias_pairs, holdout=holdout["concepts"], include_holdout=True)
    train_table = full.without_holdout()
    if sorted(full.heldout_entries()) != holdout["heldout_entries"]:
        raise AssertionError("holdout entries differ between the closure and the alias table")
    assert_alias_disjoint(full, train_table)

    # 5. corpora per tokenizer.
    min_subtokens = int(data["min_subtokens"])
    shared = dict(documents=documents, full=full, train_table=train_table, ontology=ontology, holdout_sha256=holdout["sha256"],
                  train_min_subtokens=int(data["train_min_subtokens"]), min_subtokens=min_subtokens, workers=workers)
    reference = relink_for_host(tokenizer_name, data_root, train_tokens=int(data["train_tokens"]),
                                eval_mix_tokens=data.get("eval_mix_tokens"), **shared)
    builds = [(tokenizer_name, data_root, reference)]
    for host in config.get("hosts") or []:
        root = data_root / "hosts" / host["name"]
        info = relink_for_host(host["tokenizer"], root, train_tokens=int(host["train_tokens"]),
                               eval_mix_tokens=host.get("eval_mix_tokens", data.get("eval_mix_tokens")), **shared)
        builds.append((host["tokenizer"], root, info))

    # 6. leakage audit.
    audit = leakage_audit(documents, builds, full=full, heldout_entries=holdout["heldout_entries"], workers=workers,
                          scratch=data_root / "audit-full-table")
    log(f"build: leakage audit {audit}")
    if not audit["passed"]:
        raise AssertionError(f"leakage audit failed: {audit}")

    # 7. cardinality (full table) on evaluation samples per source and mixed; feasibility per ℓ_min.
    cardinality_docs = int(data["cardinality_docs"])
    samples = {"entities": list(islice(documents.eval_domain(), cardinality_docs)),
               "general": list(islice(documents.eval_general(), cardinality_docs))}
    samples["mixed"] = list(documents.mixed(iter(samples["entities"]), iter(samples["general"])))
    by_source: dict[str, dict[str, Any]] = {source: {} for source in samples}
    for name in config["cardinality_tokenizers"]:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        for source, texts in samples.items():
            by_source[source][name] = cardinality_report(full, tok, texts, thresholds=(1, 2, 3, 4))
    (output_dir / "cardinality.json").write_text(json.dumps(by_source["mixed"], indent=2) + "\n")
    (output_dir / "cardinality_by_source.json").write_text(json.dumps(by_source, indent=2) + "\n")
    criteria = config["feasibility"]
    feasibility = {}
    for label, root, info in builds:
        frequencies = {}
        for threshold in (1, 2, 3, 4):
            if threshold >= int(data["train_min_subtokens"]):
                frequencies[threshold] = (train_frequency(root / "train", len(full.entry_concepts), threshold), "measured")
            else:
                frequencies[threshold] = (np.zeros(len(full.entry_concepts), dtype=np.int64), "not stored (train_min_subtokens)")
        feasibility[label] = feasibility_report(root / criteria.get("split", DOMAIN_SPLIT), heldout_entries=holdout["heldout_entries"],
                                                entry_count=len(full.entry_concepts), frequencies=frequencies, criteria=criteria)
    (output_dir / "feasibility.json").write_text(json.dumps({"criteria": criteria, "by_tokenizer": feasibility}, indent=2) + "\n")

    summary = {
        "track": documents.track_label,
        "ontology": {**{k: v for k, v in ontology.metadata.items() if k in ("source", "max_atomics", "max_degree", "max_values",
                                                                             "records", "empty_frames", "dropped_edges", "edges",
                                                                             "raw_edges")},
                     "concepts": len(ontology.concept_names), "atomics": len(ontology.atomic_names),
                     "relations": len(ontology.relation_names), "aliases": len(full.alias_to_entry),
                     "entries": len(full.entry_concepts), "alias_subtokens": alias_stats},
        "mix": {"domain_share": documents.domain_share, "chars_per_token": {"entities": calibration[0], "general": calibration[1]},
                "calibration_tokenizer": tokenizer_name, "general_shards": general["shards"], "general_eval_docs_skipped": general["eval_docs"],
                "eval_buckets": documents.eval_buckets, "eval_domain_docs": documents.eval_domain_docs,
                "eval_rood_docs": documents.eval_rood_docs},
        "presample": {**presample_manifest, "source_shares": presample_shares},
        "holdout": {"concepts": len(holdout["concepts"]), "chosen_entries": len(holdout["chosen_entries"]),
                    "closure_entries": len(holdout["closure_entries"]), "heldout_entries": len(holdout["heldout_entries"]),
                    "eligible_entries": holdout["eligible_entries"], "excluded_hub_entries": holdout["excluded_hub_entries"],
                    "rood": holdout["eligibility"], "exclusion_keys": len(holdout["exclusion_keys"]),
                    "exclusion_cost_documents_screen": holdout["exclusion_cost_documents"], "sha256": holdout["sha256"]},
        "corpora": {label: info for label, _, info in builds},
        "hosts": {h["tokenizer"]: str(data_root / "hosts" / h["name"]) for h in config.get("hosts") or []},
        "audit": audit, "alias_table_sha256": full.digest(), "data_root": str(data_root),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (output_dir / "report.md").write_text(render_report(summary, by_source, feasibility, criteria))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", track=documents.track_label)
    return summary


# -- report ------------------------------------------------------------------------------------------------------------

def _cardinality_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['min_subtokens']} | {r['linkable_entries']:,} | {r['linked_entries']:,} | {r['span_occurrences']:,} | "
                     f"{r['covered_token_fraction']:.3f} | {r['entries_seen_once']:,} | {r['entries_seen_2_to_9']:,} | "
                     f"{r['entries_seen_10_plus']:,} |")
    return lines


def render_report(summary: dict[str, Any], by_source: dict[str, dict[str, Any]], feasibility: dict[str, Any],
                  criteria: dict[str, Any]) -> str:
    h, onto, audit = summary["holdout"], summary["ontology"], summary["audit"]
    lines = [f"# {summary['track']} — corpus build (T8)", "",
             "Ontology: Wikidata entities of the general relation benchmarks (Entity Inferences / ECBD, LRE relations, BEAR, "
             "PopQA, TwoHopFact) and their frames' filler entities; frames from Wikidata truthy statements of a fixed property "
             "list (`ontologies/wikidata.py`). Corpus: FineWeb-Edu (the C3 sample-10BT shards) documents that mention a "
             "selected name, mixed 50/50 (tokens) with documents that mention none. Data: Wikidata (CC0); FineWeb-Edu "
             "(ODC-By 1.0). The hosts' pretraining likely contains FineWeb-Edu, so the text is not new to them; the holdout is "
             "the controlled part.", "",
             f"Ontology: {onto['concepts']:,} concepts, {onto['relations']} relations, {onto['atomics']:,} filler atomics, "
             f"{onto['edges']:,} edges ({onto.get('empty_frames', 0):,} empty frames; dropped {onto.get('dropped_edges')}). "
             f"Linker: {onto['aliases']:,} aliases over {onto['entries']:,} entries.", "",
             "| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |", "|---|---:|---:|---|"]
    for name, s in onto["alias_subtokens"].items():
        lines.append(f"| {name} | {s['aliases']:,} | {s['aliases_2plus_subtokens']:,} | {' / '.join(f'{v:,}' for v in s['histogram_1_2_3_4_5plus'])} |")
    lines += ["", "## Holdout (M1, ROOD)", "",
              f"{h['heldout_entries']:,} held-out entries ({h['chosen_entries']:,} chosen of {h['eligible_entries']:,} eligible with "
              f"the pre-sample count, {h['closure_entries']:,} added by the closure); sha256 `{h['sha256']}`. ROOD eligibility: "
              f"{h['rood']['eligible_entries']:,} entries (excluded: {h['rood']['excluded']}). Node-disjoint: no held-out entity "
              f"fills any frame. Alias-disjoint: no held-out name is, or is a whole-word part of, a training alias. Document "
              f"exclusion: every training document naming a held-out entity under any of its {h['exclusion_keys']:,} Wikidata "
              "names is dropped (those documents feed `eval-entities`).", "",
              "**Leakage audit** (the realized training documents re-read): "
              f"held-out names found {audit['heldout_names_in_training']}, training documents in `eval-entities` "
              f"{audit['training_ids_in_eval_entities']}, held-out spans with the full alias table {audit['heldout_spans_full_table']} "
              f"({audit['training_documents_read']:,} documents) — **{'passed' if audit['passed'] else 'FAILED'}**.", "",
              "## Corpora", "",
              f"Evaluation side: documents whose id's sha256 bucket is < {summary['mix']['eval_buckets']} / 10,000, plus the ROOD "
              f"documents; `eval-entities` keeps the first {summary['mix'].get('eval_domain_docs') or 'all'} evaluation-side and "
              f"{summary['mix'].get('eval_rood_docs') or 'all'} ROOD documents. General evaluation: "
              f"C3's first {summary['mix']['general_eval_docs_skipped']:,} documents. Calibration (chars per "
              f"{summary['mix']['calibration_tokenizer']} token): entities {summary['mix']['chars_per_token']['entities']:.3f}, "
              f"general {summary['mix']['chars_per_token']['general']:.3f}.", "",
              "| tokenizer | corpus | tokens | documents | spans | entity-text token share |", "|---|---|---:|---:|---:|---:|"]
    for tok, info in summary["corpora"].items():
        for name, manifest in info["corpora"].items():
            share = (info["source_shares"].get(name) or {}).get("share", {}).get("entities")
            lines.append(f"| {tok} | {name} | {manifest['tokens']:,} | {manifest['documents']:,} | {manifest['spans']:,} | "
                         f"{'—' if share is None else f'{share:.3f}'} |")
    lines += ["", "## Span cardinality (evaluation samples, full alias table)", ""]
    for source, tables in by_source.items():
        for name, rows in tables.items():
            lines += [f"### {source} — {name}", "", *_cardinality_table(rows), ""]
    lines += ["## Feasibility per ℓ_min", "",
              f"Criterion (on `{criteria.get('split', DOMAIN_SPLIT)}`, per tokenizer, as T1-open / T7): ≥ {criteria['heldout_min_entries_5plus']} "
              f"held-out entries with ≥ 5 occurrences and ≥ {criteria['heldout_min_occurrences']:,} held-out occurrences; ≥ "
              f"{criteria['rare_min_entries']} rare entries (training frequency 1–9) and ≥ {criteria['rare_min_occurrences']:,} "
              "rare occurrences.", "",
              "| tokenizer | ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | verdict | eval windows needed |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for tok, rows in feasibility.items():
        for r in rows:
            s = r["split"]
            needed = f"{r['eval_windows_needed']:,}" if r["eval_windows_needed"] else "not reached"
            lines.append(f"| {tok} | {r['min_subtokens']} | {s['tokens_considered']:,} | {s['heldout_occurrences']:,} | "
                         f"{s['heldout_entries_5plus']:,} | {s['rare_occurrences']:,} | {s['rare_entries_linked']:,} | {r['verdict']} | {needed} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("wikidata", "screen", "build"), default="build")
    parser.add_argument("--output", type=Path, help="run folder (screen, build)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.stage == "wikidata":
        print(json.dumps(run_wikidata(config), indent=2, default=str)[:4000])
        return
    if args.output is None:
        parser.error("--output is required for --stage screen and build")
    if args.stage == "screen":
        print(json.dumps(run_screen(config, args.output), indent=2, default=str)[:4000])
        return
    print(json.dumps(run(config, args.output), indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
