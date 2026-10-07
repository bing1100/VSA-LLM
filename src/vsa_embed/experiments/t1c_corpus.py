"""T1c: the licensed clinical track — SNOMED CT International 2022-05-31 × MIMIC-III 1.4 notes (decision 58).

Every result built from this corpus is labelled **"clinical (SNOMED CT + MIMIC-III)"**. It supersedes decision 1's
open substitute (T1-open, MeSH × PubMed) where the licensed sources apply; T1-open stays as its open counterpart.

**Licence and DUA (non-negotiable).** MIMIC-III is PhysioNet credentialed data and SNOMED CT is licensed. Nothing
derived from either is committed: the extracted notes, corpora, `ontology.pt`, the holdout lists and the alias table
live under `paths.data_root` / `paths.notes_dir` (`~/data/vsa-llm/t1c/`, mode 700). The committed run folder gets
aggregates only — counts, token totals, shares, the holdout's sha256, the cardinality and feasibility tables and the
report; `holdout_concepts.txt` and `holdout_entries.tsv` (SNOMED identifiers and names) are written to the data
root. No stage prints note text or individual rows, and no MIMIC text is sent to any external service.

Stages (`--stage`):

- `inventory`: names, sizes, row counts and column headers of the clinical files (`data.clinical_inventory`).
- `extract`: `NOTEEVENTS.csv.gz` → shuffled Parquet shards split by patient (`data.mimic.extract_notes`): rows with
  `ISERROR` set are dropped; `mimic.eval_buckets` / 10,000 of patients (by `SUBJECT_ID` hash) are evaluation-only.
- `build` (T1-open's recipe, `t1_open_corpus`, with MIMIC notes as the domain stream):
  1. the SNOMED CT adapter (`ontologies.snomed`) → frames and the linker alias table (aliases curated by
     `SnomedAliasPolicy`: decision 19's lesson for a case-insensitive linker at ℓ_min = 2);
  2. document streams: training = train-side notes (shard order) interleaved with the C3 FineWeb-Edu stream so that
     `mix.domain_share` of the tokens are notes; evaluation = evaluation-patient notes (`eval-mimic`), C3's
     evaluation documents (`eval-general`, the locality corpus) and their mix (`eval`);
  3. pre-sample (first `presample_tokens` of the training mix) → frozen holdout (node- and alias-disjoint, the
     T1-open rule: stratified 10% of eligible entries + closure over alias containment and shared concepts),
     pinned by `expected_holdout_sha256` once frozen;
  4. corpora for the reference tokenizer (SmolLM2: `data_root/{train,eval,eval-mimic,eval-general}`) and every
     configured host relink (`hosts`: the Qwen3 tokenizer under `data_root/hosts/qwen3`, same holdout);
  5. `ontology.pt`, cardinality per source and tokenizer, feasibility per stratum (held-out, unseen, rare,
     3+-subtoken) on `eval-mimic`, report.
- `definitions`: how many concepts have a text definition (SNOMED `sct2_TextDefinition`, UMLS `MRDEF` through the
  SNOMED CT US atoms of `MRCONSO`), overall, among the track's concepts and among the held-out concepts (counts only).

    PYTHONPATH=src python -m vsa_embed.experiments.t1c_corpus --config experiments/t1c-clinical/t1c.yaml --stage inventory --output experiments/t1c-clinical/runs/inventory-v1
    PYTHONPATH=src python -m vsa_embed.experiments.t1c_corpus --config experiments/t1c-clinical/t1c.yaml --stage extract
    PYTHONPATH=src python -m vsa_embed.experiments.t1c_corpus --config experiments/t1c-clinical/t1c.yaml --output experiments/t1c-clinical/runs/v1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator

import numpy as np
import yaml
from transformers import AutoTokenizer

from vsa_embed.data.corpus import TokenCorpus, build_corpus, eval_windows
from vsa_embed.data.mimic import extract_notes, iter_notes, notes_signature
from vsa_embed.experiments.c3_corpus import iter_texts
from vsa_embed.experiments.t1_open_corpus import (alias_statistics, assert_alias_disjoint, chars_per_token,
                                                  choose_track_holdout, general_settings, guard_reuse, load_config,
                                                  mix_documents, record_shares, relink_for_host, train_frequency,
                                                  verify_shards, window_mask)
from vsa_embed.ontologies.snomed import DEFAULT_HIERARCHIES, SnomedAliasPolicy, build_snomed_ontology
from vsa_embed.ontologies.wordnet import FrameOntology
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import AliasTable, alias_subtoken_lengths, cardinality_report

TRACK_LABEL = "clinical (SNOMED CT + MIMIC-III)"
SOURCES = ("mimic", "general")
DOMAIN_SPLIT = "eval-mimic"
RARE = (1, 9)
# Feasibility strata (trainer names; `training.lm.stratum_masks`): what makes a span count for each.
STRATA = ("after_heldout", "after_unseen", "after_rare_seen", "after_len3plus")


# -- ontology ---------------------------------------------------------------------------------------------------

def snomed_adapter(config: dict[str, Any]) -> FrameOntology:
    return build_snomed_ontology(Path(config["path"]).expanduser(),
                                 hierarchies=tuple(config.get("hierarchies") or DEFAULT_HIERARCHIES),
                                 max_atomics=int(config["max_atomics"]), max_degree=int(config["max_degree"]),
                                 min_relation_edges=int(config.get("min_relation_edges", 200)),
                                 policy=SnomedAliasPolicy.from_config(config.get("aliases")))


ONTOLOGY_ADAPTERS: dict[str, Callable[[dict[str, Any]], FrameOntology]] = {"snomed": snomed_adapter}


def build_track_ontology(config: dict[str, Any]) -> FrameOntology:
    adapter = config.get("adapter", "snomed")
    if adapter not in ONTOLOGY_ADAPTERS:
        raise ValueError(f"unknown ontology adapter {adapter!r}; known: {sorted(ONTOLOGY_ADAPTERS)}")
    return ONTOLOGY_ADAPTERS[adapter](config)


# -- documents --------------------------------------------------------------------------------------------------

@dataclass
class ClinicalDocuments:
    """Factories for every document stream of T1c (fresh iterator per call); `relink_for_host`'s interface."""

    notes_dir: Path
    general_shards: list[str]
    general_skip: int
    eval_general_docs: int
    eval_domain_docs: int | None = None
    domain_share: float = 0.5
    calibration: tuple[float, float] = (3.5, 4.6)          # chars per reference-tokenizer token (mimic, general)
    categories: tuple[str, ...] | None = None
    _notes_signature: str | None = field(default=None, repr=False)
    domain_split: ClassVar[str] = DOMAIN_SPLIT
    track_label: ClassVar[str] = TRACK_LABEL
    sources: ClassVar[tuple[str, ...]] = SOURCES

    def _category_filter(self) -> frozenset[str] | None:
        return frozenset(self.categories) if self.categories else None

    def signature(self, stream: str) -> str:
        """Fingerprint of everything that decides a stream's documents (guards corpus reuse)."""
        if self._notes_signature is None:
            self._notes_signature = notes_signature(self.notes_dir)
        spec: dict[str, Any] = {"stream": stream, "notes": self._notes_signature, "categories": self.categories,
                                "general": [Path(s).name for s in self.general_shards], "general_skip": self.general_skip}
        if stream in ("train", "eval"):
            spec.update(domain_share=self.domain_share, calibration=[round(c, 6) for c in self.calibration])
        if stream in ("eval", DOMAIN_SPLIT):
            spec["eval_domain_docs"] = self.eval_domain_docs
        if stream in ("eval", "eval-general"):
            spec["eval_general_docs"] = self.eval_general_docs
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def eval_domain(self) -> Iterator[str]:
        return iter_notes(self.notes_dir, "eval", limit=self.eval_domain_docs, categories=self._category_filter())

    def eval_general(self) -> Iterator[str]:
        return iter_texts(self.general_shards, limit=self.eval_general_docs)

    def train_domain(self) -> Iterator[str]:
        return iter_notes(self.notes_dir, "train", categories=self._category_filter())

    def train_general(self) -> Iterator[str]:
        return iter_texts(self.general_shards, skip=self.general_skip)

    def mixed(self, domain: Iterator[str], general: Iterator[str], log: list[int] | None = None) -> Iterator[str]:
        return mix_documents([domain, general], [self.domain_share, 1.0 - self.domain_share], list(self.calibration), log=log)

    def train(self, log: list[int] | None = None) -> Iterator[str]:
        return self.mixed(self.train_domain(), self.train_general(), log)

    def eval_mixed(self, log: list[int] | None = None) -> Iterator[str]:
        return self.mixed(self.eval_domain(), self.eval_general(), log)


def track_documents(config: dict[str, Any], *, calibration: tuple[float, float] | None = None) -> ClinicalDocuments:
    general, data = general_settings(config), config["data"]
    notes_dir = Path(config["paths"]["notes_dir"]).expanduser()
    if not (notes_dir / "extract.json").exists():
        raise FileNotFoundError(f"{notes_dir}/extract.json missing: run --stage extract first")
    if int(data["eval_general_docs"]) > general["eval_docs"]:
        raise ValueError("eval_general_docs exceeds the C3 evaluation documents")
    categories = config.get("mimic", {}).get("categories")
    return ClinicalDocuments(notes_dir=notes_dir, general_shards=general["shards"], general_skip=general["eval_docs"],
                             eval_general_docs=int(data["eval_general_docs"]), eval_domain_docs=data.get("eval_domain_docs"),
                             domain_share=float(config["mix"]["domain_share"]), calibration=calibration or (3.5, 4.6),
                             categories=tuple(categories) if categories else None)


# -- feasibility per stratum ------------------------------------------------------------------------------------

def stratum_table(spans: dict[str, np.ndarray], *, threshold: int, heldout: np.ndarray, frequency: np.ndarray,
                  mask: np.ndarray | None = None) -> dict[str, dict[str, int]]:
    """Per feasibility stratum: span occurrences, distinct entries and entries with ≥ 5 occurrences, among spans of
    ≥ `threshold` subtokens (as the trainer's strata: held-out; unseen = training frequency 0, not held out;
    rare-seen = 1–9; 3+-subtoken = any entry, span of ≥ 3 subtokens)."""
    keep = spans["length"] >= threshold
    if mask is not None:
        keep &= mask
    entries = spans["entry"][keep].astype(np.int64)
    lengths = spans["length"][keep]
    n = heldout.size
    held = heldout[entries]
    count = frequency[entries]
    members = {"after_heldout": held, "after_unseen": ~held & (count == 0),
               "after_rare_seen": ~held & (count >= RARE[0]) & (count <= RARE[1]), "after_len3plus": lengths >= 3}
    out = {}
    for name in STRATA:
        chosen = entries[members[name]]
        per_entry = np.bincount(chosen, minlength=n)
        out[name] = {"occurrences": int(chosen.size), "entries": int((per_entry > 0).sum()),
                     "entries_5plus": int((per_entry >= 5).sum())}
    out["all"] = {"occurrences": int(entries.size), "entries": int(np.unique(entries).size), "entries_5plus": 0}
    return out


def stratum_verdict(row: dict[str, int], criteria: dict[str, Any]) -> str:
    occurrences, entries = int(criteria["min_occurrences"]), int(criteria["min_entries"])
    if row["occurrences"] >= occurrences and row["entries_5plus"] >= entries:
        return "feasible"
    if row["occurrences"] >= occurrences and row["entries"] >= entries:
        return "feasible (entries counted with ≥ 1 occurrence)"
    if row["occurrences"] >= int(criteria.get("exploratory_occurrences", 1000)) and row["entries"] >= int(criteria.get("exploratory_entries", 50)):
        return "exploratory-feasible"
    return "infeasible"


def feasibility_by_stratum(corpus_dir: Path, *, heldout_entries: list[int], frequency: np.ndarray, criteria: dict[str, Any],
                           threshold: int) -> dict[str, Any]:
    """Each stratum on the whole evaluation split and in the trainer's evenly spread windows, with verdicts against
    decision 17's bar (≥ `min_occurrences` occurrences, ≥ `min_entries` entries with ≥ 5 occurrences) and the
    fewest windows that reach it."""
    corpus = TokenCorpus.open(corpus_dir)
    heldout = np.zeros(frequency.size, dtype=bool)
    heldout[heldout_entries] = True
    length = int(criteria.get("window_length", 1024))
    windows = int(criteria.get("trainer_windows", 2048))
    whole = max(1, (len(corpus) - length - 1) // length)
    counts = sorted({min(c, whole) for c in (1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768)} | {whole})
    tables = {c: stratum_table(corpus.spans, threshold=threshold, heldout=heldout, frequency=frequency,
                               mask=window_mask(corpus.spans, eval_windows(corpus, count=c, length=length), length))
              for c in counts}
    split = stratum_table(corpus.spans, threshold=threshold, heldout=heldout, frequency=frequency)
    rows = {}
    for name in STRATA:
        needed = next((c for c in counts if stratum_verdict(tables[c][name], criteria) == "feasible"), None)
        view = tables[min(windows, whole)][name]
        rows[name] = {"split": split[name], "verdict_split": stratum_verdict(split[name], criteria),
                      "trainer_windows": {"windows": min(windows, whole), **view},
                      "verdict_trainer_windows": stratum_verdict(view, criteria), "windows_needed": needed}
    return {"min_subtokens": threshold, "tokens": len(corpus), "windows_for_whole_split": whole, "strata": rows,
            "linked_spans_split": split["all"]["occurrences"]}


# -- stages -----------------------------------------------------------------------------------------------------

def run_inventory(config: dict[str, Any], output_dir: Path, *, rows: bool = True) -> dict[str, Any]:
    from vsa_embed.data.clinical_inventory import DEFAULT_GROUPS, inventory
    git_at_start = prepare_output_dir(output_dir) if not (output_dir / "resolved_config.yaml").exists() else None
    result = inventory(Path(config["paths"]["licensed_root"]).expanduser(), DEFAULT_GROUPS, rows=rows,
                       workers=int(config.get("inventory_workers", 2)))
    (output_dir / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", track=TRACK_LABEL, stage="inventory")
    return result["totals"]


def run_extract(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["mimic"]
    out = Path(config["paths"]["notes_dir"]).expanduser()
    if (out / "extract.json").exists():
        return json.loads((out / "extract.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    return extract_notes(Path(spec["noteevents"]).expanduser(), out, eval_buckets=int(spec["eval_buckets"]),
                         shards=int(spec["shards"]), chunk_rows=int(spec.get("chunk_rows", 50_000)))


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir) if not (output_dir / "resolved_config.yaml").exists() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    paths, data = config["paths"], config["data"]
    data_root = _private_dir(Path(paths["data_root"]).expanduser())
    workers = int(config["workers"])
    tokenizer_name = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    general = general_settings(config)
    verify_shards(general["shards"], general["shard_sha256"])

    # 1. ontology and linker.
    ontology = build_track_ontology(config["ontology"])
    base_table = AliasTable.from_pairs(ontology.alias_pairs)
    alias_lengths = alias_subtoken_lengths(base_table, tokenizer)
    entry_length: dict[int, int] = {}
    for alias, entry in base_table.alias_to_entry.items():
        entry_length[entry] = max(entry_length.get(entry, 0), alias_lengths[alias])
    alias_stats = alias_statistics(base_table, list(config["cardinality_tokenizers"]))

    # 2. streams; chars/token calibration on evaluation documents (never trained on).
    documents = track_documents(config)
    calibration_docs = int(config["mix"]["calibration_docs"])
    calibration = (chars_per_token(list(islice(documents.eval_domain(), calibration_docs)), tokenizer),
                   chars_per_token(list(islice(documents.eval_general(), calibration_docs)), tokenizer))
    documents.calibration = calibration

    # 3. frequency pre-sample (prefix of the mixed training stream) → frozen holdout.
    presample_log: list[int] = []
    presample_signature = documents.signature("train")
    guard_reuse(data_root / "presample", base_table, int(data["presample_tokens"]), presample_signature)
    presample_manifest = build_corpus(documents.train(presample_log), data_root / "presample", tokenizer_name=tokenizer_name,
                                      table=base_table, eos_id=tokenizer.eos_token_id, max_tokens=int(data["presample_tokens"]),
                                      workers=workers, reuse=True,
                                      extra_manifest={"track": TRACK_LABEL, "max_tokens_requested": int(data["presample_tokens"]),
                                                      "stream_signature": presample_signature})
    presample_shares = record_shares(data_root / "presample", presample_log, tokenizer.eos_token_id, SOURCES)
    presample = TokenCorpus.open(data_root / "presample")
    long_enough = presample.spans["length"] >= int(data["holdout_min_subtokens"])
    counts = Counter(presample.spans["entry"][long_enough].tolist())
    holdout = choose_track_holdout(counts, entry_length, base_table, ontology.concept_names,
                                   fraction=float(data["holdout_fraction"]), min_count=int(data["holdout_min_count"]),
                                   seed=int(config["seed"]), max_containing=data.get("holdout_max_containing"))
    expected = data.get("expected_holdout_sha256")
    if expected and expected != holdout["sha256"]:
        raise ValueError(f"holdout sha256 {holdout['sha256']} != pinned {expected}: the frozen holdout changed")
    # Licensed: SNOMED identifiers and names stay in the data root.
    headings = ontology.metadata.get("headings", ontology.concept_names)
    chosen = set(holdout["chosen_entries"])
    (data_root / "holdout_concepts.txt").write_text("\n".join(holdout["names"]) + "\n")
    rows = ["entry\tconcept\theading\treason\tpresample_count"]
    for entry in holdout["heldout_entries"]:
        for concept in base_table.entry_concepts[entry]:
            rows.append(f"{entry}\t{ontology.concept_names[concept]}\t{headings[concept]}\t"
                        f"{'chosen' if entry in chosen else 'closure'}\t{counts.get(entry, 0)}")
    (data_root / "holdout_entries.tsv").write_text("\n".join(rows) + "\n")

    # 4. tables sharing entry ids; corpora per tokenizer.
    full = AliasTable.from_pairs(ontology.alias_pairs, holdout=holdout["concepts"], include_holdout=True)
    train_table = full.without_holdout()
    if sorted(full.heldout_entries()) != holdout["heldout_entries"]:
        raise AssertionError("holdout entries differ between the closure and the alias table")
    assert_alias_disjoint(full, train_table)
    min_subtokens = int(data["min_subtokens"])
    shared = dict(documents=documents, full=full, train_table=train_table, ontology=ontology, holdout_sha256=holdout["sha256"],
                  train_min_subtokens=int(data["train_min_subtokens"]), min_subtokens=min_subtokens, workers=workers)
    reference = relink_for_host(tokenizer_name, data_root, train_tokens=int(data["train_tokens"]),
                                eval_mix_tokens=data.get("eval_mix_tokens"), **shared)
    builds = [(tokenizer_name, data_root, reference, int(data["train_tokens"]))]
    for host in config.get("hosts") or []:
        root = _private_dir(data_root / "hosts" / host["name"])
        info = relink_for_host(host["tokenizer"], root, train_tokens=int(host["train_tokens"]),
                               eval_mix_tokens=host.get("eval_mix_tokens", data.get("eval_mix_tokens")), **shared)
        builds.append((host["tokenizer"], root, info, int(host["train_tokens"])))

    # 5. cardinality (full table) on evaluation samples per source and mixed; aggregates only.
    cardinality_docs = int(data["cardinality_docs"])
    samples = {"mimic": list(islice(documents.eval_domain(), cardinality_docs)),
               "general": list(islice(documents.eval_general(), cardinality_docs))}
    samples["mixed"] = list(documents.mixed(iter(samples["mimic"]), iter(samples["general"])))
    by_source: dict[str, dict[str, Any]] = {source: {} for source in samples}
    for name in config["cardinality_tokenizers"]:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        for source, texts in samples.items():
            by_source[source][name] = cardinality_report(full, tok, texts, thresholds=(1, 2, 3, 4))
    del samples
    (output_dir / "cardinality.json").write_text(json.dumps(by_source["mixed"], indent=2) + "\n")
    (output_dir / "cardinality_by_source.json").write_text(json.dumps(by_source, indent=2) + "\n")

    # Feasibility per stratum on each tokenizer's MIMIC evaluation split, at ℓ_min = min_subtokens (and 3).
    criteria = config["feasibility"]
    feasibility = {}
    for label, root, info, configured in builds:
        built = info["corpora"]["train"]["tokens"]
        if built < 0.99 * configured:
            raise ValueError(f"{label}: training corpus has {built:,} of {configured:,} tokens (the stream ran out)")
        feasibility[label] = [feasibility_by_stratum(root / DOMAIN_SPLIT, heldout_entries=holdout["heldout_entries"],
                                                     frequency=train_frequency(root / "train", len(full.entry_concepts), threshold),
                                                     criteria=criteria, threshold=threshold)
                              for threshold in sorted({min_subtokens, 3})]
    (output_dir / "feasibility.json").write_text(json.dumps({"criteria": criteria, "by_tokenizer": feasibility}, indent=2) + "\n")

    keys = ("source", "concepts", "alias_policy", "alias_stats", "hierarchy_counts", "relation_edges", "frames_truncated",
            "max_atomics", "max_degree", "min_relation_edges")
    summary = {
        "track": TRACK_LABEL,
        "ontology": {**{k: v for k, v in ontology.metadata.items() if k in keys},
                     "atomics": len(ontology.atomic_names), "relations": len(ontology.relation_names),
                     "aliases": len(full.alias_to_entry), "entries": len(full.entry_concepts), "alias_subtokens": alias_stats,
                     "concepts_with_text_definition": int(sum(1 for d in ontology.metadata.get("definitions", []) if d))},
        "notes": json.loads((documents.notes_dir / "extract.json").read_text()),
        "mix": {"domain_share": documents.domain_share, "chars_per_token": {"mimic": calibration[0], "general": calibration[1]},
                "calibration_tokenizer": tokenizer_name, "general_shards": [Path(s).name for s in general["shards"]],
                "general_eval_docs_skipped": general["eval_docs"], "eval_domain_docs": documents.eval_domain_docs,
                "categories": documents.categories},
        "presample": {**presample_manifest, "source_shares": presample_shares},
        "holdout": {"concepts": len(holdout["concepts"]), "chosen_entries": len(holdout["chosen_entries"]),
                    "closure_entries": len(holdout["closure_entries"]), "heldout_entries": len(holdout["heldout_entries"]),
                    "eligible_entries": holdout["eligible_entries"], "excluded_hub_entries": holdout["excluded_hub_entries"],
                    "sha256": holdout["sha256"], "names_file": str(data_root / "holdout_concepts.txt")},
        "corpora": {label: info for label, _, info, _ in builds},
        "hosts": {h["tokenizer"]: str(data_root / "hosts" / h["name"]) for h in config.get("hosts") or []},
        "alias_table_sha256": full.digest(), "data_root": str(data_root),
    }
    summary["notes"].pop("source", None)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (output_dir / "report.md").write_text(render_report(summary, by_source, feasibility, criteria))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", track=TRACK_LABEL)
    return summary


def run_definitions(config: dict[str, Any], output_dir: Path, *, umls: bool = True) -> dict[str, Any]:
    """Counts of concepts with a text definition (SNOMED CT; UMLS MRDEF via the SNOMED CT US atoms): overall, among the
    track's concepts and among its held-out concepts. Writes `definitions.json` (counts only)."""
    from vsa_embed.ontologies.snomed import SnomedFiles, read_release
    spec = config["ontology"]
    release = read_release(SnomedFiles.find(Path(spec["path"]).expanduser()))
    ontology = build_track_ontology(spec)
    track = set(ontology.concept_names)
    names_file = Path(config["paths"]["data_root"]).expanduser() / "holdout_concepts.txt"
    held = set(names_file.read_text().split()) if names_file.exists() else set()
    snomed_defined = {c for c, n in release.definitions.items() if n}
    result: dict[str, Any] = {"snomed_text_definition": {
        "active_concepts": len(release.active), "defined_active": len(snomed_defined),
        "track_concepts": len(track), "defined_track": len(snomed_defined & track),
        "heldout_concepts": len(held), "defined_heldout": len(snomed_defined & held)}}
    meta = config.get("umls", {}).get("meta_dir")
    if umls and meta:
        meta = Path(meta).expanduser()
        cui_of: dict[str, set[str]] = {}
        for path in sorted(meta.glob("MRCONSO.RRF.*gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    f = line.split("|")
                    if f[11] == "SNOMEDCT_US" and f[13] in track:
                        cui_of.setdefault(f[13], set()).add(f[0])
        defined_cuis: Counter[str] = Counter()
        sources: dict[str, set[str]] = {}
        with gzip.open(meta / "MRDEF.RRF.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                f = line.split("|")
                if f[6] in ("O", "E", "Y"):            # suppressed definitions are not used
                    continue
                defined_cuis[f[0]] += 1
                sources.setdefault(f[0], set()).add(f[4])
        defined = {c for c, cuis in cui_of.items() if any(defined_cuis[u] for u in cuis)}
        source_counts: Counter[str] = Counter()
        for c in defined:
            source_counts.update({s for u in cui_of[c] for s in sources.get(u, ())})
        result["umls_mrdef"] = {"track_concepts_with_cui": len(cui_of), "defined_track": len(defined),
                                "defined_heldout": len(defined & held), "either_track": len(defined | (snomed_defined & track)),
                                "either_heldout": len((defined | snomed_defined) & held),
                                "definition_sources_top": dict(source_counts.most_common(12))}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "definitions.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# -- report -----------------------------------------------------------------------------------------------------

def _cardinality_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['min_subtokens']} | {r['linkable_entries']:,} | {r['linked_entries']:,} | {r['span_occurrences']:,} | "
                     f"{r['covered_token_fraction']:.3f} | {r['entries_seen_once']:,} | {r['entries_seen_2_to_9']:,} | "
                     f"{r['entries_seen_10_plus']:,} | {r['mean_distinct_entries_per_window']:.1f} |")
    return lines


def render_report(summary: dict[str, Any], by_source: dict[str, dict[str, Any]], feasibility: dict[str, Any],
                  criteria: dict[str, Any]) -> str:
    h, onto, notes = summary["holdout"], summary["ontology"], summary["notes"]
    lines = [f"# T1c corpus — {TRACK_LABEL}", "",
             "Ontology: SNOMED CT International Edition 2022-05-31 (RF2 snapshot; active concepts of the selected top-level "
             "hierarchies; frames from the active inferred relationships). Corpus: MIMIC-III 1.4 `NOTEEVENTS` (rows with "
             "`ISERROR` dropped), split by patient, mixed 50/50 (tokens) with FineWeb-Edu (the C3 stream). Licensed and "
             "credentialed sources: this report holds aggregates only; every derived table stays under the data root.", "",
             f"Concepts {onto['concepts']:,} ({onto.get('hierarchy_counts')}); {onto['relations']} relations; "
             f"{onto['atomics']:,} atomics; {onto['concepts_with_text_definition']:,} concepts with a SNOMED text definition.", "",
             f"Linker: {onto['aliases']:,} aliases over {onto['entries']:,} entries; alias policy {onto.get('alias_policy')}; "
             f"decisions {onto.get('alias_stats')}.", "",
             "| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |", "|---|---:|---:|---|"]
    for name, s in onto["alias_subtokens"].items():
        lines.append(f"| {name} | {s['aliases']:,} | {s['aliases_2plus_subtokens']:,} | {' / '.join(f'{v:,}' for v in s['histogram_1_2_3_4_5plus'])} |")
    lines += ["", f"Notes: {notes['rows']:,} rows; {notes['dropped_iserror']:,} `ISERROR` rows and {notes['dropped_empty']:,} empty "
              f"notes dropped; patients train / eval {notes['patients']['train']:,} / {notes['patients']['eval']:,} "
              f"(on both sides: {notes['patients_on_both_sides']}); notes train / eval {notes['notes']['train']:,} / "
              f"{notes['notes']['eval']:,}.", "",
              "| category | train notes | eval notes | train characters |", "|---|---:|---:|---:|"]
    for category in sorted(notes["categories"]["train"], key=lambda c: -notes["category_characters"]["train"][c]):
        lines.append(f"| {category} | {notes['categories']['train'][category]:,} | {notes['categories']['eval'].get(category, 0):,} | "
                     f"{notes['category_characters']['train'][category]:,} |")
    lines += ["", f"Holdout: {h['heldout_entries']:,} entries ({h['chosen_entries']:,} chosen of {h['eligible_entries']:,} eligible, "
              f"{h['closure_entries']:,} added by the closure, {h['excluded_hub_entries']:,} hub entries not eligible); "
              f"sha256 `{h['sha256']}` (names in the data root, not committed).", "", "## Corpora", "",
              f"Mixing calibration (chars per {summary['mix']['calibration_tokenizer']} token): notes "
              f"{summary['mix']['chars_per_token']['mimic']:.3f}, general {summary['mix']['chars_per_token']['general']:.3f}; "
              f"`{DOMAIN_SPLIT}` = the first {summary['mix']['eval_domain_docs'] or 'all'} evaluation-patient notes (shuffled order).", "",
              "| tokenizer | corpus | tokens | documents | spans | note token share |", "|---|---|---:|---:|---:|---:|"]
    for tok, info in summary["corpora"].items():
        for name, manifest in info["corpora"].items():
            share = (info["source_shares"].get(name) or {}).get("share", {}).get("mimic")
            lines.append(f"| {tok} | {name} | {manifest['tokens']:,} | {manifest['documents']:,} | {manifest['spans']:,} | "
                         f"{'—' if share is None else f'{share:.3f}'} |")
    lines += ["", "## Span cardinality (evaluation samples, full alias table)", ""]
    for source, tables in by_source.items():
        for name, rows in tables.items():
            lines += [f"### {source} — {name}", "", *_cardinality_table(rows), ""]
    lines += ["## Feasibility per stratum", "",
              f"Bar (decision 17, WP-T1's power argument): ≥ {criteria['min_occurrences']:,} span occurrences and ≥ "
              f"{criteria['min_entries']} entries with ≥ 5 occurrences on `{DOMAIN_SPLIT}`; WP-C7's lower bar (≥ "
              f"{criteria.get('exploratory_occurrences', 1000):,} occurrences, ≥ {criteria.get('exploratory_entries', 50)} entries) marks "
              f"a stratum exploratory-feasible. Strata as the trainer's: held-out; unseen (training frequency 0, not held out); "
              f"rare-seen (1–9); 3+-subtoken (any entry). `windows needed` = the fewest evenly spread {criteria.get('window_length', 1024)}-token "
              f"windows that meet the bar; E9 evaluates on {criteria.get('trainer_windows', 2048):,}.", "",
              "| tokenizer | ℓ_min | stratum | occurrences (split) | entries | entries ≥ 5 | verdict (split) | occurrences / entries ≥ 5 in the E9 windows | verdict (E9 windows) | windows needed |",
              "|---|---:|---|---:|---:|---:|---|---|---|---:|"]
    for tok, blocks in feasibility.items():
        for block in blocks:
            for name, row in block["strata"].items():
                s, w = row["split"], row["trainer_windows"]
                lines.append(f"| {tok} | {block['min_subtokens']} | {name} | {s['occurrences']:,} | {s['entries']:,} | "
                             f"{s['entries_5plus']:,} | {row['verdict_split']} | {w['occurrences']:,} / {w['entries_5plus']:,} "
                             f"({w['windows']:,} windows) | {row['verdict_trainer_windows']} | "
                             f"{row['windows_needed'] if row['windows_needed'] else 'not reached'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("inventory", "extract", "build", "definitions"), default="build")
    parser.add_argument("--output", type=Path, help="run folder (inventory, build, definitions)")
    parser.add_argument("--no-rows", action="store_true", help="inventory without line counts")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.stage == "extract":
        result = run_extract(config)
        print(json.dumps({k: result[k] for k in ("rows", "dropped_iserror", "dropped_empty", "notes", "patients")}, indent=2))
        return
    if args.output is None:
        parser.error(f"--output is required for --stage {args.stage}")
    if args.stage == "inventory":
        print(json.dumps(run_inventory(config, args.output, rows=not args.no_rows), indent=2))
    elif args.stage == "definitions":
        print(json.dumps(run_definitions(config, args.output), indent=2))
    else:
        summary = run(config, args.output)
        print(json.dumps({"holdout": summary["holdout"]["sha256"], "alias_table": summary["alias_table_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
