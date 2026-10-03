"""T1-open: the clinical track on open data (MeSH 2026 × PubMed), mixed 50/50 with general text.

Every result built from this corpus is labelled **"open-clinical (MeSH + PubMed)"** (author decision
of 2026-10-02). SNOMED CT / UMLS can replace MeSH later through `ONTOLOGY_ADAPTERS` (an adapter maps
its config section to a `FrameOntology` whose `alias_pairs` feed the linker); MIMIC-IV notes would
be another domain stream. Outputs mirror `c3_corpus` (train / eval corpora, `ontology.pt` with the
same keys, cardinality tables), so the trainer and the E4 tooling work unchanged.

Stages:

- `fetch`: download the configured PubMed baseline files (each verified against its `.md5`
  sidecar), extract title + abstract to parquet (`data/pubmed.py`), download PubMedQA `pqa_labeled`.
- `build`:
  1. ontology adapter → frames and the linker alias table; alias subtoken statistics per tokenizer.
  2. Document streams. PubMed is split by a PMID hash: `eval_buckets` / 10,000 of PMIDs are
     evaluation-only; PubMedQA PMIDs never enter training. General text is the C3 FineWeb-Edu
     stream read from `c3.yaml` (same shards and sha256 checks), its evaluation documents skipped
     in training and reused as the general half of evaluation. Training text interleaves the two
     streams so that `domain_share` of the tokens is PubMed (estimated from characters with a
     chars/token calibration on evaluation documents; the realized shares are measured afterwards).
  3. Frequency pre-sample on the first `presample_tokens` of the mixed training stream → frozen
     holdout: `holdout_fraction` of eligible entries (≥ `holdout_min_count` pre-sample spans of
     ≥ `holdout_min_subtokens` subtokens; aliases contained in at most `holdout_max_containing`
     other entries' aliases), stratified by log-frequency (`c3_corpus.choose_holdout`). Closure:
     an entry with an alias that contains a held-out alias as a whole-word substring is held out
     too (alias-disjoint: a held-out surface form never sits inside a training span), and so is any
     entry sharing a concept with a held-out entry (the c3 rule). The sorted held-out descriptor
     list is hashed; `expected_holdout_sha256` pins it across builds.
  4. GPT-2 corpora under `data_root`: `train` (held-out aliases removed), `eval` (50/50 mix,
     all aliases), `eval-pubmed`, `eval-general`; `presample`.
  5. Host corpora (`hosts`) under `data_root/hosts/<name>/`: the same documents, tables and entry
     ids re-linked with the host tokenizer (`relink_for_host`), each with its own `ontology.pt`.
  6. `ontology.pt`; cardinality tables (tokenizer × ℓ_min, per source); feasibility verdict per
     ℓ_min; report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from vsa_embed.data.corpus import TokenCorpus, build_corpus, eval_windows, tokenizer_fingerprint
from vsa_embed.data.pubmed import (baseline_names, download_baseline, extract_baseline, file_digest, iter_pubmed,
                                   pmid_bucket, text_paths)
from vsa_embed.experiments.c3_corpus import choose_holdout, iter_texts
from vsa_embed.ontologies.mesh import MeshAliasPolicy, build_mesh_track_ontology
from vsa_embed.ontologies.wordnet import FrameOntology
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import AliasTable, alias_subtoken_lengths, cardinality_report

TRACK_LABEL = "open-clinical (MeSH + PubMed)"
SOURCES = ("pubmed", "general")
RARE = (1, 9)          # the trainer's `after_rare` stratum: training frequency 1–9


# -- ontology adapters --------------------------------------------------------------------------

def mesh_adapter(config: dict[str, Any]) -> FrameOntology:
    return build_mesh_track_ontology(Path(config["path"]).expanduser(), max_atomics=int(config["max_atomics"]),
                                     max_degree=int(config["max_degree"]),
                                     policy=MeshAliasPolicy.from_config(config.get("aliases")))


ONTOLOGY_ADAPTERS: dict[str, Callable[[dict[str, Any]], FrameOntology]] = {"mesh": mesh_adapter}


def build_track_ontology(config: dict[str, Any]) -> FrameOntology:
    adapter = config.get("adapter", "mesh")
    if adapter not in ONTOLOGY_ADAPTERS:
        raise ValueError(f"unknown ontology adapter {adapter!r}; known: {sorted(ONTOLOGY_ADAPTERS)}")
    return ONTOLOGY_ADAPTERS[adapter](config)


# -- documents ----------------------------------------------------------------------------------

def pubmed_documents(paths: list[Path], *, eval_buckets: int, split: str, exclude: frozenset[int] = frozenset(),
                     limit: int | None = None) -> Iterator[str]:
    """Texts of the `split` ("eval" or "train") side of the PMID-hash split, in file order, each
    PMID once; `exclude` PMIDs are skipped."""
    if split not in {"eval", "train"}:
        raise ValueError("split must be eval or train")
    seen: set[int] = set()
    produced = 0
    for record in iter_pubmed(paths):
        pmid = int(record["pmid"])
        if pmid in seen or pmid in exclude:
            continue
        seen.add(pmid)
        if (pmid_bucket(pmid) < eval_buckets) != (split == "eval"):
            continue
        if limit is not None and produced >= limit:
            return
        produced += 1
        yield record["text"]


def mix_documents(streams: list[Iterator[str]], shares: list[float], chars_per_token: list[float], *,
                  log: list[int] | None = None) -> Iterator[str]:
    """Interleave document streams so that each stream's estimated token share tracks `shares`
    (next document from the stream furthest below its share; ties → lower index). Deterministic;
    stops as soon as the chosen stream is exhausted, so the mix never drifts to one source.
    `log` receives the source index of every emitted document."""
    if len(streams) != len(shares) or len(shares) != len(chars_per_token):
        raise ValueError("streams, shares and chars_per_token must have equal length")
    active = [i for i, share in enumerate(shares) if share > 0]
    tokens = [0.0] * len(streams)
    while True:
        source = min(active, key=lambda i: (tokens[i] / shares[i], i))
        try:
            text = next(streams[source])
        except StopIteration:
            return
        tokens[source] += len(text) / chars_per_token[source]
        if log is not None:
            log.append(source)
        yield text


def chars_per_token(texts: list[str], tokenizer: Any) -> float:
    lengths = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return sum(len(t) for t in texts) / max(1, sum(len(ids) for ids in lengths))


@dataclass
class TrackDocuments:
    """Factories for every document stream of the track (fresh iterator per call)."""

    pubmed_paths: list[Path]
    eval_buckets: int
    general_shards: list[str]
    general_skip: int                        # C3 evaluation documents, never used for training
    eval_general_docs: int
    eval_pubmed_docs: int | None = None
    exclude_pmids: frozenset[int] = frozenset()
    domain_share: float = 0.5
    calibration: tuple[float, float] = (4.7, 4.6)   # chars per reference-tokenizer token (pubmed, general)

    def signature(self, stream: str) -> str:
        """Fingerprint of everything that decides a stream's documents (guards corpus reuse)."""
        spec: dict[str, Any] = {"stream": stream, "pubmed": [p.name for p in self.pubmed_paths], "eval_buckets": self.eval_buckets,
                                "general": [Path(s).name for s in self.general_shards], "general_skip": self.general_skip}
        if stream in ("train", "eval"):
            spec.update(domain_share=self.domain_share, calibration=[round(c, 6) for c in self.calibration])
        if stream == "train":
            spec["exclude"] = hashlib.sha256(",".join(map(str, sorted(self.exclude_pmids))).encode()).hexdigest()
        if stream in ("eval", "eval-pubmed"):
            spec["eval_pubmed_docs"] = self.eval_pubmed_docs
        if stream in ("eval", "eval-general"):
            spec["eval_general_docs"] = self.eval_general_docs
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def eval_pubmed(self) -> Iterator[str]:
        return pubmed_documents(self.pubmed_paths, eval_buckets=self.eval_buckets, split="eval",
                                limit=self.eval_pubmed_docs)

    def eval_general(self) -> Iterator[str]:
        return iter_texts(self.general_shards, limit=self.eval_general_docs)

    def train_pubmed(self) -> Iterator[str]:
        return pubmed_documents(self.pubmed_paths, eval_buckets=self.eval_buckets, split="train",
                                exclude=self.exclude_pmids)

    def train_general(self) -> Iterator[str]:
        return iter_texts(self.general_shards, skip=self.general_skip)

    def mixed(self, domain: Iterator[str], general: Iterator[str], log: list[int] | None = None) -> Iterator[str]:
        return mix_documents([domain, general], [self.domain_share, 1.0 - self.domain_share], list(self.calibration), log=log)

    def train(self, log: list[int] | None = None) -> Iterator[str]:
        return self.mixed(self.train_pubmed(), self.train_general(), log)

    def eval_mixed(self, log: list[int] | None = None) -> Iterator[str]:
        return self.mixed(self.eval_pubmed(), self.eval_general(), log)


def source_token_shares(corpus_dir: Path, log: list[int], eos_id: int) -> dict[str, Any]:
    """Realized tokens per source of a mixed corpus: documents are EOS-terminated in stream order,
    so the first `documents` entries of the mixer's log label them. Exact unless the builder
    dropped documents (then flagged approximate and estimated from document counts)."""
    corpus = TokenCorpus.open(corpus_dir)
    documents = int(corpus.manifest["documents"])
    ends = np.flatnonzero(np.asarray(corpus.tokens) == eos_id)
    labels = np.asarray(log[:documents], dtype=np.int64)
    exact = ends.size == documents and int(corpus.manifest.get("skipped_documents", 0)) == 0
    if exact:
        lengths = np.diff(np.concatenate([[-1], ends]))
        per_source = np.bincount(labels, weights=lengths, minlength=len(SOURCES))
    else:
        per_source = np.bincount(labels, minlength=len(SOURCES)) * (len(corpus) / max(1, documents))
    total = float(per_source.sum())
    return {"exact": bool(exact), "documents": {s: int((labels == i).sum()) for i, s in enumerate(SOURCES)},
            "tokens": {s: int(per_source[i]) for i, s in enumerate(SOURCES)},
            "share": {s: float(per_source[i] / total) if total else 0.0 for i, s in enumerate(SOURCES)}}


def record_shares(corpus_dir: Path, log: list[int], eos_id: int) -> dict[str, Any] | None:
    """Measure and store `sources.json` after a build; a reused corpus (empty log) reads it back."""
    path = corpus_dir / "sources.json"
    if log:
        shares = source_token_shares(corpus_dir, log, eos_id)
        path.write_text(json.dumps(shares, indent=2) + "\n")
        return shares
    return json.loads(path.read_text()) if path.exists() else None


def guard_reuse(out_dir: Path, table: AliasTable, max_tokens: int, signature: str | None = None) -> None:
    """`build_corpus(reuse=True)` reuses any corpus with the same alias table; refuse one built for a
    different token budget or document stream (e.g. a slice build, or another eval split, in the
    same data root)."""
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("alias_table_sha256") != table.digest():
            return                                   # build_corpus rebuilds it
        if manifest.get("max_tokens_requested") != max_tokens:
            raise ValueError(f"{out_dir} was built with max_tokens {manifest.get('max_tokens_requested')}, not {max_tokens}; "
                             "use a fresh data_root")
        if signature is not None and manifest.get("stream_signature") != signature:
            raise ValueError(f"{out_dir} was built from a different document stream; use a fresh data_root")


# -- holdout ------------------------------------------------------------------------------------

def _is_word(char: str) -> bool:
    return char.isalnum() or char == "_"


def whole_word_substrings(alias: str) -> set[str]:
    """Proper substrings of `alias` that start and end at word boundaries (the linker's left
    boundary rule, and a non-word character or the end on the right)."""
    n = len(alias)
    starts = [0] + [i for i in range(1, n) if not _is_word(alias[i - 1]) and alias[i] != " "]
    ends = [j for j in range(1, n) if not _is_word(alias[j]) and alias[j - 1] != " "] + [n]
    return {alias[i:j] for i in starts for j in ends if j > i and (i, j) != (0, n)}


def containment_index(table: AliasTable) -> dict[int, set[int]]:
    """entry A → entries B ≠ A having an alias that contains an alias of A as a whole-word substring."""
    contained_in: dict[int, set[int]] = {}
    for alias, entry in table.alias_to_entry.items():
        for part in whole_word_substrings(alias):
            inner = table.alias_to_entry.get(part)
            if inner is not None and inner != entry:
                contained_in.setdefault(inner, set()).add(entry)
    return contained_in


def holdout_closure(entries: set[int], table: AliasTable, contained_in: dict[int, set[int]]) -> set[int]:
    """Smallest superset of `entries` closed under (i) containment of a held-out alias and
    (ii) sharing a concept with a held-out entry."""
    by_concept: dict[int, set[int]] = {}
    for entry, concepts in enumerate(table.entry_concepts):
        for concept in concepts:
            by_concept.setdefault(concept, set()).add(entry)
    held, frontier = set(entries), list(entries)
    while frontier:
        entry = frontier.pop()
        neighbours = set(contained_in.get(entry, ()))
        for concept in table.entry_concepts[entry]:
            neighbours |= by_concept[concept]
        for other in neighbours - held:
            held.add(other); frontier.append(other)
    return held


def choose_track_holdout(counts: Counter, lengths: dict[int, int], table: AliasTable, concept_names: list[str], *,
                         fraction: float, min_count: int, seed: int, max_containing: int | None) -> dict[str, Any]:
    """c3's stratified choice over entries whose aliases are contained in at most `max_containing`
    other entries' aliases (so the closure stays local), then the closure; hash of sorted names."""
    contained_in = containment_index(table)
    hubs = {e for e in counts if max_containing is not None and len(contained_in.get(e, ())) > max_containing}
    eligible_counts = Counter({e: c for e, c in counts.items() if e not in hubs})
    chosen = choose_holdout(eligible_counts, lengths, table, concept_names, fraction=fraction, min_count=min_count, seed=seed)
    held_entries = holdout_closure(set(chosen["chosen_entries"]), table, contained_in)
    concepts = sorted({c for e in held_entries for c in table.entry_concepts[e]})
    names = sorted(concept_names[c] for c in concepts)
    excluded_hubs = sum(1 for e in hubs if counts[e] >= min_count and lengths.get(e, 0) >= 2)
    return {"chosen_entries": chosen["chosen_entries"], "closure_entries": sorted(held_entries - set(chosen["chosen_entries"])),
            "heldout_entries": sorted(held_entries), "concepts": concepts, "names": names,
            "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "eligible_entries": chosen["eligible_entries"], "excluded_hub_entries": excluded_hubs}


def assert_alias_disjoint(full: AliasTable, train_table: AliasTable) -> None:
    """No held-out alias is a training alias or a whole-word part of one."""
    held_entries = full.heldout_entries()
    held_aliases = {a for a, e in full.alias_to_entry.items() if e in held_entries}
    leaked = held_aliases & set(train_table.alias_to_entry)
    for alias in train_table.alias_to_entry:
        leaked |= whole_word_substrings(alias) & held_aliases
    if leaked:
        raise AssertionError(f"held-out aliases reachable in training: {sorted(leaked)[:10]}")


# -- per-tokenizer corpora ----------------------------------------------------------------------

def channel_ontology(full: AliasTable, ontology: FrameOntology, frequency: np.ndarray, holdout_sha256: str) -> dict[str, Any]:
    """The trainer's channel ontology (same keys as C3's `ontology.pt`)."""
    schedule = full.entry_schedule(ontology.frames)
    return {
        "entry_count": len(full.entry_concepts), "atomic_count": len(ontology.atomic_names),
        "relation_count": len(ontology.relation_names), "offsets": schedule.offsets,
        "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": sorted(full.heldout_entries()),
        "train_frequency": frequency.tolist(), "alias_table_sha256": full.digest(), "holdout_sha256": holdout_sha256,
        "relation_names": ontology.relation_names, "atomic_names": ontology.atomic_names,
        "entry_concepts": full.entry_concepts, "concept_names": ontology.concept_names,
    }


def train_frequency(train_dir: Path, entry_count: int, min_subtokens: int) -> np.ndarray:
    spans = TokenCorpus.open(train_dir).spans
    return np.bincount(spans["entry"][spans["length"] >= min_subtokens], minlength=entry_count)


def relink_for_host(tokenizer_name: str, out_dir: Path, *, documents: TrackDocuments, full: AliasTable,
                    train_table: AliasTable, ontology: FrameOntology, holdout_sha256: str, train_tokens: int,
                    train_min_subtokens: int, min_subtokens: int, workers: int, eval_mix_tokens: int | None = None,
                    corpora: tuple[str, ...] = ("eval", "eval-pubmed", "eval-general", "train")) -> dict[str, Any]:
    """Link the track's documents for one tokenizer: training (held-out aliases removed) and
    evaluation corpora sharing entry ids with every other tokenizer, plus that tokenizer's
    `ontology.pt` (training frequencies differ per tokenizer because span lengths do)."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    eos, vocab = tokenizer.eos_token_id, len(tokenizer)
    # A tokenizer that normalizes its input (Qwen: NFC) is decode-checked against the normalized text and its corpora
    # record its fingerprint; GPT-2 and SmolLM2 have no normalizer, so their relinks are unchanged.
    from .host_corpus import tokenizer_normalization
    normalization = tokenizer_normalization(tokenizer)
    extra = {"normalization": normalization} if normalization else {}
    fingerprint = {"tokenizer_sha256": tokenizer_fingerprint(tokenizer)} if normalization else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    plan: dict[str, tuple[Callable[[list[int]], Iterator[str]], AliasTable, int, int]] = {
        "train": (documents.train, train_table, train_tokens, train_min_subtokens),
        "eval": (documents.eval_mixed, full, int(eval_mix_tokens or 10**12), 1),
        "eval-pubmed": (lambda log: documents.eval_pubmed(), full, 10**12, 1),
        "eval-general": (lambda log: documents.eval_general(), full, 10**12, 1),
    }
    manifests: dict[str, Any] = {}
    shares: dict[str, Any] = {}
    for name in corpora:
        if name not in plan:
            raise ValueError(f"unknown corpus {name!r}")
        stream, table, max_tokens, min_length = plan[name]
        signature = documents.signature(name)
        guard_reuse(out_dir / name, table, max_tokens, signature)
        log: list[int] = []
        manifests[name] = build_corpus(stream(log), out_dir / name, tokenizer_name=tokenizer_name, table=table, eos_id=eos,
                                       max_tokens=max_tokens, min_subtokens=min_length, workers=workers, vocab_size=vocab,
                                       reuse=True, extra_manifest={"track": TRACK_LABEL, "max_tokens_requested": max_tokens,
                                                                   "stream_signature": signature, **fingerprint}, **extra)
        if name in ("train", "eval"):
            shares[name] = record_shares(out_dir / name, log, eos)
    frequency = train_frequency(out_dir / "train", len(full.entry_concepts), min_subtokens)
    torch.save(channel_ontology(full, ontology, frequency, holdout_sha256), out_dir / "ontology.pt")
    return {"tokenizer": tokenizer_name, "corpora": manifests, "source_shares": shares,
            "linked_entries_in_train_at_min_subtokens": int((frequency > 0).sum()),
            "ontology_sha256": file_digest(out_dir / "ontology.pt")}


# -- feasibility --------------------------------------------------------------------------------

def window_mask(spans: dict[str, np.ndarray], starts: list[int], length: int) -> np.ndarray:
    """Spans lying fully inside one of the trainer's evaluation windows (as `TokenCorpus.window`)."""
    begins = np.asarray(sorted(starts), dtype=np.int64)
    index = np.searchsorted(begins, spans["inject"], side="right") - 1
    valid = index >= 0
    begin = begins[np.clip(index, 0, None)]
    return valid & (spans["inject"] < begin + length) & (spans["start"] >= begin)


def stratum_counts(spans: dict[str, np.ndarray], *, threshold: int, heldout: np.ndarray, frequency: np.ndarray,
                   mask: np.ndarray | None = None) -> dict[str, int]:
    keep = spans["length"] >= threshold
    if mask is not None:
        keep &= mask
    entries = spans["entry"][keep].astype(np.int64)
    n = heldout.size
    held = heldout[entries]
    rare_entry = (frequency >= RARE[0]) & (frequency <= RARE[1]) & ~heldout
    rare = rare_entry[entries]
    held_counts = np.bincount(entries[held], minlength=n)
    rare_counts = np.bincount(entries[rare], minlength=n)
    return {"tokens_considered": 0, "linked_spans": int(keep.sum()),
            "heldout_occurrences": int(held.sum()), "heldout_entries_linked": int((held_counts > 0).sum()),
            "heldout_entries_5plus": int((held_counts >= 5).sum()),
            "rare_occurrences": int(rare.sum()), "rare_entries_linked": int((rare_counts > 0).sum()),
            "rare_entries_5plus": int((rare_counts >= 5).sum())}


def verdict(counts: dict[str, int], criteria: dict[str, Any], *, rare_measured: bool) -> dict[str, Any]:
    checks = {
        "heldout_entries_5plus": counts["heldout_entries_5plus"] >= int(criteria["heldout_min_entries_5plus"]),
        "heldout_occurrences": counts["heldout_occurrences"] >= int(criteria["heldout_min_occurrences"]),
        "rare_entries_linked": counts["rare_entries_linked"] >= int(criteria["rare_min_entries"]),
        "rare_occurrences": counts["rare_occurrences"] >= int(criteria["rare_min_occurrences"]),
    }
    heldout_ok = checks["heldout_entries_5plus"] and checks["heldout_occurrences"]
    rare_ok = checks["rare_entries_linked"] and checks["rare_occurrences"]
    if not rare_measured:
        label = "held-out stratum feasible; rare stratum pending the full build" if heldout_ok else "infeasible (held-out stratum)"
    elif heldout_ok and rare_ok:
        label = "feasible"
    elif heldout_ok:
        label = "feasible for the held-out stratum only (rare stratum underpowered)"
    else:
        label = "infeasible"
    return {"checks": checks, "verdict": label}


def feasibility_report(corpus_dir: Path, *, heldout_entries: list[int], entry_count: int,
                       frequencies: dict[int, tuple[np.ndarray, str]], criteria: dict[str, Any],
                       thresholds: tuple[int, ...] = (1, 2, 3, 4)) -> list[dict[str, Any]]:
    """Per ℓ_min: held-out and rare-stratum counts on the whole split and on the trainer's default
    evaluation windows, and the verdict. `frequencies[ℓ]` = (training frequency at ℓ, provenance)."""
    corpus = TokenCorpus.open(corpus_dir)
    heldout = np.zeros(entry_count, dtype=bool)
    heldout[heldout_entries] = True
    windows, length = int(criteria.get("trainer_windows", 1024)), int(criteria.get("window_length", 1024))
    whole_windows = max(1, (len(corpus) - length - 1) // length)
    candidates = sorted({min(c, whole_windows) for c in (windows, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576)} | {whole_windows})
    masks = {count: window_mask(corpus.spans, eval_windows(corpus, count=count, length=length), length) for count in candidates}
    rows = []
    for threshold in thresholds:
        frequency, provenance = frequencies[threshold]
        measured = provenance == "measured"
        whole = stratum_counts(corpus.spans, threshold=threshold, heldout=heldout, frequency=frequency)
        whole["tokens_considered"] = len(corpus)
        view = stratum_counts(corpus.spans, threshold=threshold, heldout=heldout, frequency=frequency, mask=masks[min(windows, whole_windows)])
        view["tokens_considered"] = min(windows, whole_windows) * length
        # Fewest evenly spread trainer windows (`eval.windows`) whose spans meet the measurable criteria.
        needed = None
        for count in candidates:
            checks = verdict(stratum_counts(corpus.spans, threshold=threshold, heldout=heldout, frequency=frequency,
                                            mask=masks[count]), criteria, rare_measured=measured)["checks"]
            if checks["heldout_entries_5plus"] and checks["heldout_occurrences"] and \
                    (not measured or (checks["rare_entries_linked"] and checks["rare_occurrences"])):
                needed = count
                break
        rows.append({"min_subtokens": threshold, "frequency_source": provenance, "split": whole,
                     "trainer_windows": {"windows": min(windows, whole_windows), "length": length, **view},
                     **verdict(whole, criteria, rare_measured=measured),
                     "trainer_windows_verdict": verdict(view, criteria, rare_measured=measured)["verdict"],
                     "windows_for_whole_split": whole_windows, "eval_windows_needed": needed})
    return rows


# -- stages -------------------------------------------------------------------------------------

def general_settings(config: dict[str, Any]) -> dict[str, Any]:
    """The C3 general stream (shards, their sha256 prefixes, evaluation documents) read from c3.yaml."""
    c3 = yaml.safe_load(Path(config["paths"]["general_config"]).read_text())
    return {"shards": [str(Path(p).expanduser()) for p in c3["paths"]["shards"]],
            "shard_sha256": c3["paths"].get("shard_sha256") or {}, "eval_docs": int(c3["data"]["eval_docs"]),
            "c3_seed": c3.get("seed")}


def verify_shards(shards: list[str], expected: dict[str, str]) -> None:
    for shard in shards:   # refuse corrupt downloads (as c3_corpus)
        prefix = expected.get(Path(shard).name)
        if prefix and not file_digest(Path(shard)).startswith(prefix):
            raise ValueError(f"{shard} sha256 does not start with {prefix}")


def pubmed_names(config: dict[str, Any]) -> list[str]:
    spec = config["pubmed"]
    return baseline_names(spec["prefix"], int(spec["files"][0]), int(spec["files"][1]))


def fetch(config: dict[str, Any]) -> dict[str, Any]:
    """Download and extract the PubMed baseline files; download PubMedQA (pqa_labeled)."""
    paths, spec = config["paths"], config["pubmed"]
    names = pubmed_names(config)
    raw, text = Path(paths["pubmed_raw"]).expanduser(), Path(paths["pubmed_text"]).expanduser()
    downloads = download_baseline(names, raw, base_url=spec.get("base_url", "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"))
    extracted = extract_baseline(names, raw, text, workers=int(config["workers"]),
                                 min_abstract_words=int(spec["min_abstract_words"]), english_only=bool(spec["english_only"]))
    result = {"downloads": downloads, "extracted": extracted}
    qa = config.get("pubmedqa")
    if qa:
        from huggingface_hub import hf_hub_download
        local = Path(paths["pubmedqa"]).expanduser()
        file = hf_hub_download(qa["repo"], qa["file"], repo_type="dataset", revision=qa["revision"], local_dir=local)
        result["pubmedqa"] = {"path": file, "sha256": file_digest(Path(file))}
        if qa.get("test_ground_truth_url"):   # the official 500-item test split (pubmedqa GitHub, pinned commit)
            import requests
            response = requests.get(qa["test_ground_truth_url"], timeout=60)
            response.raise_for_status()
            target = local / "test_ground_truth.json"
            target.write_bytes(response.content)
            result["pubmedqa"]["test_ground_truth"] = {"path": str(target), "sha256": file_digest(target),
                                                       "items": len(json.loads(response.content))}
    (Path(paths["pubmed_text"]).expanduser() / "fetch.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def pubmedqa_pmids(config: dict[str, Any]) -> frozenset[int]:
    qa = config.get("pubmedqa")
    if not qa:
        return frozenset()
    import pyarrow.parquet as pq
    path = Path(config["paths"]["pubmedqa"]).expanduser() / qa["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run --stage fetch first")
    return frozenset(int(p) for p in pq.read_table(path, columns=["pubid"]).column(0).to_pylist())


def track_documents(config: dict[str, Any], *, calibration: tuple[float, float] | None = None) -> TrackDocuments:
    general, data = general_settings(config), config["data"]
    text_dir = Path(config["paths"]["pubmed_text"]).expanduser()
    paths = text_paths(pubmed_names(config), text_dir)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"extracted PubMed files missing (run --stage fetch first): {missing[:3]}")
    if int(data["eval_general_docs"]) > general["eval_docs"]:
        raise ValueError("eval_general_docs exceeds the C3 evaluation documents")
    return TrackDocuments(
        pubmed_paths=paths, eval_buckets=int(config["pubmed"]["eval_buckets"]), general_shards=general["shards"],
        general_skip=general["eval_docs"], eval_general_docs=int(data["eval_general_docs"]),
        eval_pubmed_docs=data.get("eval_pubmed_docs"), exclude_pmids=pubmedqa_pmids(config),
        domain_share=float(config["mix"]["domain_share"]), calibration=calibration or (4.7, 4.6))


def alias_statistics(table: AliasTable, tokenizer_names: list[str]) -> dict[str, Any]:
    out = {}
    for name in tokenizer_names:
        lengths = alias_subtoken_lengths(table, AutoTokenizer.from_pretrained(name, local_files_only=True))
        histogram = Counter(min(v, 5) for v in lengths.values())
        out[name] = {"aliases": len(lengths), "aliases_2plus_subtokens": sum(1 for v in lengths.values() if v >= 2),
                     "histogram_1_2_3_4_5plus": [histogram.get(k, 0) for k in range(1, 6)]}
    return out


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir) if not (output_dir / "resolved_config.yaml").exists() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    paths, data = config["paths"], config["data"]
    data_root = Path(paths["data_root"]).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
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
    calibration = (chars_per_token(list(islice(documents.eval_pubmed(), calibration_docs)), tokenizer),
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
    presample_shares = record_shares(data_root / "presample", presample_log, tokenizer.eos_token_id)
    presample = TokenCorpus.open(data_root / "presample")
    long_enough = presample.spans["length"] >= int(data["holdout_min_subtokens"])
    counts = Counter(presample.spans["entry"][long_enough].tolist())
    holdout = choose_track_holdout(counts, entry_length, base_table, ontology.concept_names,
                                   fraction=float(data["holdout_fraction"]), min_count=int(data["holdout_min_count"]),
                                   seed=int(config["seed"]), max_containing=data.get("holdout_max_containing"))
    expected = data.get("expected_holdout_sha256")
    if expected and expected != holdout["sha256"]:
        raise ValueError(f"holdout sha256 {holdout['sha256']} != pinned {expected}: the frozen holdout changed")
    headings = ontology.metadata.get("headings", ontology.concept_names)
    chosen = set(holdout["chosen_entries"])
    (output_dir / "holdout_concepts.txt").write_text("\n".join(holdout["names"]) + "\n")
    rows = ["entry\tconcept\theading\treason\tpresample_count"]
    for entry in holdout["heldout_entries"]:
        for concept in base_table.entry_concepts[entry]:
            rows.append(f"{entry}\t{ontology.concept_names[concept]}\t{headings[concept]}\t"
                        f"{'chosen' if entry in chosen else 'closure'}\t{counts.get(entry, 0)}")
    (output_dir / "holdout_entries.tsv").write_text("\n".join(rows) + "\n")

    # 4–5. tables sharing entry ids; corpora per tokenizer.
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
    # Training budget the feasibility verdict refers to (a slice build is compared with the full one).
    builds = [(tokenizer_name, data_root, reference, int(data.get("reference_train_tokens") or data["train_tokens"]))]
    for host in config.get("hosts") or []:
        host_documents = replace(documents, eval_pubmed_docs=host.get("eval_pubmed_docs", documents.eval_pubmed_docs),
                                 eval_general_docs=int(host.get("eval_general_docs", documents.eval_general_docs)))
        root = data_root / "hosts" / host["name"]
        info = relink_for_host(host["tokenizer"], root, train_tokens=int(host["train_tokens"]),
                               eval_mix_tokens=host.get("eval_mix_tokens", data.get("eval_mix_tokens")),
                               **{**shared, "documents": host_documents})
        builds.append((host["tokenizer"], root, info, int(host.get("reference_train_tokens") or host["train_tokens"])))
    l1_manifest = None
    if int(data.get("l1_slice_tokens") or 0) and int(data["train_min_subtokens"]) > 1:
        l1_manifest = build_corpus(documents.train(), data_root / "train-l1", tokenizer_name=tokenizer_name, table=train_table,
                                   eos_id=tokenizer.eos_token_id, max_tokens=int(data["l1_slice_tokens"]), workers=workers,
                                   min_subtokens=1, reuse=True)

    # 6. cardinality (full table, as C3) on evaluation samples per source and mixed.
    cardinality_docs = int(data["cardinality_docs"])
    samples = {"pubmed": list(islice(documents.eval_pubmed(), cardinality_docs)),
               "general": list(islice(documents.eval_general(), cardinality_docs))}
    samples["mixed"] = list(documents.mixed(iter(samples["pubmed"]), iter(samples["general"])))
    by_source: dict[str, dict[str, Any]] = {source: {} for source in samples}
    for name in config["cardinality_tokenizers"]:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        for source, texts in samples.items():
            by_source[source][name] = cardinality_report(full, tok, texts, thresholds=(1, 2, 3, 4))
    (output_dir / "cardinality.json").write_text(json.dumps(by_source["mixed"], indent=2) + "\n")
    (output_dir / "cardinality_by_source.json").write_text(json.dumps(by_source, indent=2) + "\n")

    # Feasibility per ℓ_min on each tokenizer's PubMed evaluation split.
    criteria = config["feasibility"]
    feasibility = {}
    for label, root, info, configured in builds:
        built = info["corpora"]["train"]["tokens"]
        frequencies = {}
        for threshold in (1, 2, 3, 4):
            if threshold >= int(data["train_min_subtokens"]):
                frequency = train_frequency(root / "train", len(full.entry_concepts), threshold)
                # A train corpus at ≥ 99% of its budget is complete (the last document overshoots or stops short).
                provenance = "measured" if built >= 0.99 * configured else f"slice ({built:,} of {configured:,} tokens)"
            else:
                frequency = np.zeros(len(full.entry_concepts), dtype=np.int64)
                provenance = "not stored (train_min_subtokens)"
            frequencies[threshold] = (frequency, provenance)
        feasibility[label] = feasibility_report(root / criteria.get("split", "eval-pubmed"), heldout_entries=holdout["heldout_entries"],
                                                entry_count=len(full.entry_concepts), frequencies=frequencies, criteria=criteria)
    (output_dir / "feasibility.json").write_text(json.dumps({"criteria": criteria, "by_tokenizer": feasibility}, indent=2) + "\n")

    summary = {
        "track": TRACK_LABEL,
        "ontology": {**{k: v for k, v in ontology.metadata.items() if k in ("source", "descriptors", "alias_policy", "alias_stats",
                                                                             "max_atomics", "max_degree")},
                     "concepts": len(ontology.concept_names), "atomics": len(ontology.atomic_names),
                     "relations": len(ontology.relation_names), "aliases": len(full.alias_to_entry),
                     "entries": len(full.entry_concepts), "alias_subtokens": alias_stats},
        "mix": {"domain_share": documents.domain_share, "chars_per_token": {"pubmed": calibration[0], "general": calibration[1]},
                "calibration_tokenizer": tokenizer_name, "general_shards": general["shards"], "general_eval_docs_skipped": general["eval_docs"],
                "pubmed_eval_buckets": documents.eval_buckets, "eval_pubmed_docs": documents.eval_pubmed_docs,
                "excluded_pubmedqa_pmids": len(documents.exclude_pmids)},
        "presample": {**presample_manifest, "source_shares": presample_shares},
        "holdout": {"concepts": len(holdout["concepts"]), "chosen_entries": len(holdout["chosen_entries"]),
                    "closure_entries": len(holdout["closure_entries"]), "heldout_entries": len(holdout["heldout_entries"]),
                    "eligible_entries": holdout["eligible_entries"], "excluded_hub_entries": holdout["excluded_hub_entries"],
                    "sha256": holdout["sha256"]},
        "corpora": {label: info for label, _, info, _ in builds},
        "hosts": {h["tokenizer"]: str(data_root / "hosts" / h["name"]) for h in config.get("hosts") or []},
        "train_l1_slice": l1_manifest,
        "alias_table_sha256": full.digest(), "data_root": str(data_root),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (output_dir / "report.md").write_text(render_report(summary, by_source, feasibility, criteria))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", track=TRACK_LABEL)
    return summary


# -- report -------------------------------------------------------------------------------------

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
    h, onto = summary["holdout"], summary["ontology"]
    lines = [f"# T1-open corpus — {TRACK_LABEL}", "",
             "Ontology: MeSH 2026 descriptors (frames from tree positions, pharmacological actions, see-also); "
             "corpus: PubMed 2026 baseline abstracts mixed 50/50 (tokens) with FineWeb-Edu (the C3 stream). "
             "Pretrained hosts may have seen PubMed; the newest baseline files were used to limit overlap.", "",
             "Data: PubMed, courtesy of the U.S. National Library of Medicine (2026 baseline snapshot, not updated); "
             "MeSH 2026, U.S. National Library of Medicine.", "",
             f"Linker: {onto['aliases']:,} aliases over {onto['entries']:,} entries (alias policy {onto.get('alias_policy')}).", "",
             "| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |", "|---|---:|---:|---|"]
    for name, s in onto["alias_subtokens"].items():
        lines.append(f"| {name} | {s['aliases']:,} | {s['aliases_2plus_subtokens']:,} | {' / '.join(f'{v:,}' for v in s['histogram_1_2_3_4_5plus'])} |")
    lines += ["", f"Holdout: {h['heldout_entries']:,} entries ({h['chosen_entries']:,} chosen of {h['eligible_entries']:,} eligible, "
              f"{h['closure_entries']:,} added by the closure, {h['excluded_hub_entries']:,} hub entries not eligible); "
              f"sha256 `{h['sha256']}`.", "", "## Corpora", "",
              f"PubMed evaluation pool: PMIDs whose sha256 bucket is < {summary['mix']['pubmed_eval_buckets']} / 10,000 (never trained on); "
              f"`eval-pubmed` uses {summary['mix'].get('eval_pubmed_docs') or 'all'} of them. General text: the C3 FineWeb-Edu stream, "
              f"its first {summary['mix']['general_eval_docs_skipped']:,} documents (C3's evaluation documents) never trained on. "
              f"Mixing calibration (chars per {summary['mix']['calibration_tokenizer']} token): PubMed "
              f"{summary['mix']['chars_per_token']['pubmed']:.3f}, general {summary['mix']['chars_per_token']['general']:.3f}.", "",
              "| tokenizer | corpus | tokens | documents | spans | PubMed token share |", "|---|---|---:|---:|---:|---:|"]
    for tok, info in summary["corpora"].items():
        for name, manifest in info["corpora"].items():
            share = info["source_shares"].get(name, {}).get("share", {}).get("pubmed")
            lines.append(f"| {tok} | {name} | {manifest['tokens']:,} | {manifest['documents']:,} | {manifest['spans']:,} | "
                         f"{'—' if share is None else f'{share:.3f}'} |")
    lines += ["", "## Span cardinality (evaluation samples, full alias table)", ""]
    for source, tables in by_source.items():
        for name, rows in tables.items():
            lines += [f"### {source} — {name}", "", *_cardinality_table(rows), ""]
    split = criteria.get("split", "eval-pubmed")
    lines += ["## Feasibility per ℓ_min", "",
              f"**Criterion** (on the `{split}` split, per tokenizer): ≥ {criteria['heldout_min_entries_5plus']} held-out entries with "
              f"≥ 5 occurrences and ≥ {criteria['heldout_min_occurrences']:,} held-out span occurrences; ≥ {criteria['rare_min_entries']} "
              f"distinct rare entries (training frequency 1–9, the trainer's `after_rare` stratum) linked and "
              f"≥ {criteria['rare_min_occurrences']:,} rare-entry occurrences.", "",
              "**Why these numbers.** E4 gate item 1 compares the held-out (and seen-rare) stratum loss between conditions with a "
              "paired bootstrap over examples, Holm-corrected over the condition grid. The unit that varies is the span occurrence "
              "(its 8 following tokens are strongly correlated), so the occurrence count sets the standard error: with a per-occurrence "
              "paired loss difference of SD ≈ 0.5–1 nat, 2,000 occurrences give SE ≈ 0.011–0.022 nat, i.e. a minimum detectable "
              "difference of ≈ 0.04–0.08 nat at 80% power after Holm over ≈ 4 comparisons — the size of effect a channel must show "
              "to matter. Occurrences cluster by concept, so at least 300 distinct concepts with ≥ 5 occurrences each keep a "
              "concept-cluster bootstrap stable and stop a few frequent concepts from carrying the stratum. The same two numbers "
              "apply to the rare stratum (gate item 1, second half).", "",
              "Rare-stratum columns are shown only when the training corpus has its full budget (a slice cannot measure training "
              "frequencies). `eval windows needed` = the fewest evenly spread 1,024-token windows (`eval.windows` of the trainer) "
              "whose spans meet the measurable criteria; the trainer default is "
              f"{criteria.get('trainer_windows', 1024):,}.", "",
              "| tokenizer | ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | training frequencies | verdict (whole split) | eval windows needed | windows for whole split |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for tok, rows in feasibility.items():
        for r in rows:
            s = r["split"]
            measured = r["frequency_source"] == "measured"
            rare = (f"{s['rare_occurrences']:,} | {s['rare_entries_linked']:,}" if measured else "— | —")
            needed = f"{r['eval_windows_needed']:,}" if r["eval_windows_needed"] else "not reached"
            lines.append(f"| {tok} | {r['min_subtokens']} | {s['tokens_considered']:,} | {s['heldout_occurrences']:,} | "
                         f"{s['heldout_entries_5plus']:,} | {rare} | {r['frequency_source']} | {r['verdict']} | {needed} | "
                         f"{r['windows_for_whole_split']:,} |")
    return "\n".join(lines) + "\n"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    """YAML config; `extends: <relative path>` deep-merges this file over its base (lists replace)."""
    config = yaml.safe_load(path.read_text())
    base = config.pop("extends", None)
    return _merge(load_config(path.parent / base), config) if base else config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("fetch", "build"), default="build")
    parser.add_argument("--output", type=Path, help="run folder (build)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.stage == "fetch":
        print(json.dumps(fetch(config), indent=2, default=str)[:3000])
        return
    if args.output is None:
        parser.error("--output is required for --stage build")
    print(json.dumps(run(config, args.output), indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
