"""Per-host corpora for continued pretraining (WP-host): the C3 linker and holdout under a host tokenizer.

Given the C3 run folder (its `resolved_config.yaml` and the frozen `holdout_concepts.txt`) and a
pretrained host:

1. Rebuild the WordNet ontology and alias table with C3's settings and the frozen holdout, and verify
   that they are C3's: the holdout sha256 and the alias-table digests (evaluation and training views)
   must equal those C3 recorded (`ontology.pt`, `summary.json`, corpus manifests), and the entries,
   held-out entries and frames must equal C3's `ontology.pt`. Entry ids are therefore shared by every
   tokenizer, and the holdout is the same frozen one.
2. Build the host's evaluation corpus (all aliases, so held-out concepts are linked; C3's evaluation
   documents, i.e. the first `eval_docs` documents of the shard list; spans of every length) and its
   training corpus (held-out aliases removed; the documents after the evaluation documents, up to
   `train_tokens`; spans stored from ℓ ≥ `min_subtokens`, the host's ℓ_min).
3. Write the host `ontology.pt` (C3's entries and frames; `train_frequency` recounted from this host's
   training spans; tokenizer, vocabulary size) and the host's cardinality tables: the C3 sample table
   (same documents as C3's `cardinality.json`, which it must reproduce) and full-corpus tables by
   ℓ_min and by training-frequency stratum.

Hosts sharing a tokenizer share a corpus (SmolLM2-135M and -360M; verified by tokenizer fingerprint).
Token ids are stored as uint16 when the tokenizer has ≤ 65,536 ids, else uint32 (Qwen2.5: 151,665).
Large artifacts go to `paths.data_root`; the manifest, summary and report go to the run folder.

    python -m vsa_embed.experiments.host_corpus --host smollm2 --c3-run experiments/c3-general-corpus/runs/v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from nltk.corpus import wordnet as wn
from transformers import AutoConfig, AutoTokenizer

from vsa_embed.data.corpus import TokenCorpus, build_corpus, tokenizer_fingerprint
from vsa_embed.experiments.c3_corpus import iter_texts
from vsa_embed.ontologies.wordnet import build_wordnet_ontology
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import AliasTable, cardinality_report

# One corpus per tokenizer; `models` are the pretrained hosts it serves (same tokenizer, verified).
HOSTS: dict[str, dict[str, Any]] = {
    "smollm2": {"tokenizer": "HuggingFaceTB/SmolLM2-135M",
                "models": ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"]},
    "qwen2.5": {"tokenizer": "Qwen/Qwen2.5-0.5B", "models": ["Qwen/Qwen2.5-0.5B"]},
    # Qwen3 base models: Qwen2.5's byte-level BPE plus 4 added tokens (ids 151,665–151,668): another fingerprint.
    "qwen3": {"tokenizer": "Qwen/Qwen3-0.6B-Base",
              "models": ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]},
}
DEFAULTS = {"train_tokens": 130_000_000, "min_subtokens": 2, "eval_min_subtokens": 1, "workers": 6}
FREQUENCY_BINS = (("heldout", None, None), ("unseen", 0, 1), ("rare", 1, 10), ("mid", 10, 100), ("frequent", 100, None))


def default_paths(host: str) -> tuple[Path, Path]:
    """(data root, run folder) of a host corpus."""
    return (Path(f"~/data/vsa-llm/c3/wordnet-{host}-v1").expanduser(),
            Path(f"experiments/c3-general-corpus/runs/host-{host}-v1"))


def tokenizer_normalization(tokenizer: Any) -> str | None:
    """The Unicode normal form the tokenizer applies to its input (Qwen2.5: NFC), or None.

    Any other normalizer would break the decode round-trip check of `build_corpus`, so it is refused.
    """
    normalizer = json.loads(tokenizer.backend_tokenizer.to_str()).get("normalizer")
    if normalizer is None:
        return None
    parts = normalizer.get("normalizers", []) if normalizer.get("type") == "Sequence" else [normalizer]
    forms = [p["type"] for p in parts if p.get("type") in {"NFC", "NFD", "NFKC", "NFKD"}]
    if len(forms) != len(parts) or len(forms) > 1:
        raise ValueError(f"unsupported tokenizer normalizer {normalizer}")
    return forms[0] if forms else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_c3_record(c3_run: Path) -> dict[str, Any]:
    """C3's configuration, frozen holdout and every recorded digest that exists."""
    config = yaml.safe_load((c3_run / "resolved_config.yaml").read_text())
    names = [line for line in (c3_run / "holdout_concepts.txt").read_text().splitlines() if line]
    data_root = Path(config["paths"]["data_root"]).expanduser()
    record: dict[str, Any] = {"config": config, "holdout_names": names, "data_root": data_root,
                              "summary": None, "ontology": None, "ontology_sha256": None, "manifests": {},
                              "cardinality": None}
    if (c3_run / "summary.json").exists():
        record["summary"] = json.loads((c3_run / "summary.json").read_text())
    if (c3_run / "cardinality.json").exists():
        record["cardinality"] = json.loads((c3_run / "cardinality.json").read_text())
    if (data_root / "ontology.pt").exists():
        record["ontology"] = torch.load(data_root / "ontology.pt", weights_only=False)
        record["ontology_sha256"] = _sha256_file(data_root / "ontology.pt")
    for split in ("eval", "train"):
        if (data_root / split / "manifest.json").exists():
            record["manifests"][split] = json.loads((data_root / split / "manifest.json").read_text())
    return record


def rebuild_c3_tables(c3_config: dict[str, Any], holdout_names: list[str]) -> dict[str, Any]:
    """C3's ontology, frozen holdout (from its names) and the evaluation/training alias tables."""
    ontology = build_wordnet_ontology(wn, max_atomics=int(c3_config["ontology"]["max_atomics"]),
                                      max_degree=int(c3_config["ontology"]["max_degree"]))
    index = ontology.concept_index
    missing = [n for n in holdout_names if n not in index]
    if missing:
        raise ValueError(f"{len(missing)} held-out concepts are not in the rebuilt ontology (e.g. {missing[:3]})")
    concepts = sorted(index[n] for n in holdout_names)
    names = sorted(ontology.concept_names[c] for c in concepts)        # as c3_corpus.choose_holdout hashes them
    full = AliasTable.from_pairs(ontology.alias_pairs, holdout=concepts, include_holdout=True)
    return {"ontology": ontology, "holdout_concepts": concepts, "full": full, "train_table": full.without_holdout(),
            "holdout_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()}


def verify_against_c3(tables: dict[str, Any], record: dict[str, Any], *, require_record: bool = True) -> dict[str, Any]:
    """Compare the rebuilt holdout and tables with everything C3 recorded; raise on any mismatch.

    Returns one row per check: the rebuilt value, C3's value and its source. With `require_record`,
    at least the holdout sha256 and the evaluation alias-table digest must have a C3 source.
    """
    full, train_table = tables["full"], tables["train_table"]
    ontology, summary, manifests = record["ontology"], record["summary"], record["manifests"]
    expected: dict[str, list[tuple[str, Any]]] = {"holdout_sha256": [], "alias_table_sha256": [],
                                                   "train_alias_table_sha256": []}
    if ontology is not None:
        expected["holdout_sha256"].append(("c3 ontology.pt", ontology["holdout_sha256"]))
        expected["alias_table_sha256"].append(("c3 ontology.pt", ontology["alias_table_sha256"]))
    if summary is not None:
        expected["holdout_sha256"].append(("c3 summary.json", summary["holdout"]["sha256"]))
        expected["alias_table_sha256"].append(("c3 summary.json", summary["eval_corpus"]["alias_table_sha256"]))
        expected["train_alias_table_sha256"].append(("c3 summary.json", summary["train_corpus"]["alias_table_sha256"]))
    if "eval" in manifests:
        expected["alias_table_sha256"].append(("c3 eval/manifest.json", manifests["eval"]["alias_table_sha256"]))
    if "train" in manifests:
        expected["train_alias_table_sha256"].append(("c3 train/manifest.json", manifests["train"]["alias_table_sha256"]))
    rebuilt = {"holdout_sha256": tables["holdout_sha256"], "alias_table_sha256": full.digest(),
               "train_alias_table_sha256": train_table.digest()}
    checks: dict[str, Any] = {}
    for key, sources in expected.items():
        for source, value in sources:
            if value != rebuilt[key]:
                raise ValueError(f"{key}: rebuilt {rebuilt[key][:16]}… != {source} {str(value)[:16]}…")
        checks[key] = {"value": rebuilt[key], "verified_against": [s for s, _ in sources]}
    if require_record and not (expected["holdout_sha256"] and expected["alias_table_sha256"]):
        raise ValueError("no C3 record of the holdout sha256 and alias-table digest to verify against "
                         "(run C3 first, or pass --allow-unverified)")
    if ontology is not None:
        schedule = full.entry_schedule(tables["ontology"].frames)
        same = {
            "entry_concepts": [tuple(c) for c in ontology["entry_concepts"]] == [tuple(c) for c in full.entry_concepts],
            "heldout_entries": list(ontology["heldout_entries"]) == sorted(full.heldout_entries()),
            "frames": all(torch.equal(torch.as_tensor(ontology[k]), torch.as_tensor(getattr(schedule, k)))
                          for k in ("offsets", "relations", "fillers")),
            "concept_names": list(ontology["concept_names"]) == list(tables["ontology"].concept_names),
            "atomic_names": list(ontology["atomic_names"]) == list(tables["ontology"].atomic_names),
            "relation_names": list(ontology["relation_names"]) == list(tables["ontology"].relation_names),
        }
        bad = [k for k, ok in same.items() if not ok]
        if bad:
            raise ValueError(f"rebuilt ontology differs from C3 ontology.pt in {bad}")
        checks["ontology_pt"] = {"identical": sorted(same), "c3_ontology_sha256": record["ontology_sha256"]}
    return checks


def covered_tokens(start: np.ndarray, end: np.ndarray) -> int:
    """Number of distinct token positions inside the union of the closed intervals [start, end]."""
    if start.size == 0:
        return 0
    order = np.argsort(start, kind="stable")
    s, e = start[order].astype(np.int64), end[order].astype(np.int64)
    previous = np.concatenate([[-1], np.maximum.accumulate(e)[:-1]])
    return int(np.clip(e - np.maximum(s - 1, previous), 0, None).sum())


def corpus_cardinality(corpus: TokenCorpus, thresholds: list[int], frequency: np.ndarray,
                       heldout: set[int]) -> list[dict[str, Any]]:
    """Full-corpus cardinality per ℓ_min: spans, distinct entries, covered-token fraction, and the
    distinct entries / spans in each training-frequency stratum (held-out; unseen in training;
    rare 1–9; mid 10–99; frequent ≥ 100 — the strata of `training.lm`)."""
    spans = corpus.spans
    held = np.zeros(frequency.shape[0], dtype=bool)
    held[list(heldout)] = True
    rows = []
    for threshold in thresholds:
        keep = spans["length"] >= threshold
        entries = spans["entry"][keep].astype(np.int64)
        row: dict[str, Any] = {"min_subtokens": threshold, "tokens": len(corpus), "span_occurrences": int(keep.sum()),
                               "linked_entries": int(np.unique(entries).size),
                               "covered_token_fraction": covered_tokens(spans["start"][keep], spans["end"][keep]) / max(1, len(corpus))}
        counts = frequency[entries]
        for name, lo, hi in FREQUENCY_BINS:
            if name == "heldout":
                mask = held[entries]
            else:
                mask = ~held[entries] & (counts >= lo) & ((counts < hi) if hi is not None else True)
            row[f"{name}_entries"] = int(np.unique(entries[mask]).size)
            row[f"{name}_spans"] = int(mask.sum())
        rows.append(row)
    return rows


def _host_settings(host: str | None, tokenizer: str | None, models: list[str] | None) -> tuple[str, str, list[str]]:
    if host in HOSTS:
        return host, tokenizer or HOSTS[host]["tokenizer"], models or list(HOSTS[host]["models"])
    if not (host and tokenizer):
        raise ValueError(f"unknown host {host!r}: choose one of {sorted(HOSTS)} or give --host <slug> --tokenizer <name>")
    return host, tokenizer, models or [tokenizer]


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    c = json.loads(json.dumps(config))
    c.setdefault("experiment", "c3-host-corpus")
    host, tokenizer, models = _host_settings(c.get("host"), c.get("tokenizer"), c.get("models"))
    c.update(host=host, tokenizer=tokenizer, models=models)
    c.setdefault("workers", DEFAULTS["workers"])
    c.setdefault("paths", {}).setdefault("data_root", str(default_paths(host)[0]))
    data = c.setdefault("data", {})
    for key in ("train_tokens", "min_subtokens", "eval_min_subtokens"):
        data.setdefault(key, DEFAULTS[key])
    c.setdefault("verify", {}).setdefault("require_c3_record", True)
    return c


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = resolve_config(config)
    c3_run = Path(config["c3_run"])
    record = load_c3_record(c3_run)
    c3 = record["config"]
    config["c3"] = {"run": str(c3_run), "config_sha256": hashlib.sha256((c3_run / "resolved_config.yaml").read_bytes()).hexdigest(),
                    "seed": c3["seed"], "shards": c3["paths"]["shards"], "eval_docs": c3["data"]["eval_docs"],
                    "ontology": c3["ontology"], "cardinality_docs": c3["data"]["cardinality_docs"]}
    git_at_start = prepare_output_dir(output_dir) if not (output_dir / "resolved_config.yaml").exists() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    timings: dict[str, float] = {}
    clock = time.monotonic()

    # 0. the host tokenizer, and every model it serves must share it.
    tokenizer_name, data = config["tokenizer"], config["data"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    fingerprint, normalization, vocab_size = tokenizer_fingerprint(tokenizer), tokenizer_normalization(tokenizer), len(tokenizer)
    model_vocab_sizes = {}
    for model in config["models"]:
        if model != tokenizer_name and tokenizer_fingerprint(AutoTokenizer.from_pretrained(model, local_files_only=True)) != fingerprint:
            raise ValueError(f"{model} does not share the tokenizer of {tokenizer_name}")
        model_vocab_sizes[model] = int(AutoConfig.from_pretrained(model, local_files_only=True).vocab_size)
        if model_vocab_sizes[model] < vocab_size:
            raise ValueError(f"{model} has {model_vocab_sizes[model]} embedding rows < {vocab_size} tokenizer ids")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError(f"{tokenizer_name} has no EOS token")

    # 1. C3's tables and holdout, verified.
    shards = [str(Path(p).expanduser()) for p in c3["paths"]["shards"]]
    for shard in shards:
        expected = (c3["paths"].get("shard_sha256") or {}).get(Path(shard).name)
        if expected and not _sha256_file(Path(shard)).startswith(expected):
            raise ValueError(f"{shard} does not match C3's sha256 prefix {expected}")
    tables = rebuild_c3_tables(c3, record["holdout_names"])
    checks = verify_against_c3(tables, record, require_record=bool(config["verify"]["require_c3_record"]))
    full, train_table, ontology = tables["full"], tables["train_table"], tables["ontology"]
    heldout_entries = sorted(full.heldout_entries())
    timings["tables_s"] = time.monotonic() - clock; clock = time.monotonic()

    # 2. evaluation and training corpora under the host tokenizer.
    data_root = Path(config["paths"]["data_root"]).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    extra = {"host": config["host"], "tokenizer_sha256": fingerprint, "vocab_size": vocab_size,
             "models": config["models"], "c3_run": str(c3_run), "holdout_sha256": tables["holdout_sha256"]}
    eval_docs, workers = int(c3["data"]["eval_docs"]), int(config["workers"])
    eval_manifest = build_corpus(iter_texts(shards, limit=eval_docs), data_root / "eval", tokenizer_name=tokenizer_name,
                                 table=full, eos_id=eos, max_tokens=10**12, workers=workers,
                                 min_subtokens=int(data["eval_min_subtokens"]), vocab_size=vocab_size,
                                 normalization=normalization, extra_manifest={**extra, "split": "eval"}, reuse=True)
    timings["eval_s"] = time.monotonic() - clock; clock = time.monotonic()
    min_subtokens = int(data["min_subtokens"])
    train_manifest = build_corpus(iter_texts(shards, skip=eval_docs), data_root / "train", tokenizer_name=tokenizer_name,
                                  table=train_table, eos_id=eos, max_tokens=int(data["train_tokens"]), workers=workers,
                                  min_subtokens=min_subtokens, vocab_size=vocab_size, normalization=normalization,
                                  extra_manifest={**extra, "split": "train", "min_subtokens_stored": min_subtokens},
                                  reuse=True)
    timings["train_s"] = time.monotonic() - clock; clock = time.monotonic()
    for split, manifest in (("eval", eval_manifest), ("train", train_manifest)):
        if manifest["tokenizer"] != tokenizer_name or manifest.get("tokenizer_sha256") != fingerprint:
            raise ValueError(f"{data_root / split} was built with another tokenizer; use a new data root")

    # 3. host ontology and cardinality.
    train_corpus, eval_corpus = TokenCorpus.open(data_root / "train"), TokenCorpus.open(data_root / "eval")
    held = set(heldout_entries)
    if bool(np.isin(train_corpus.spans["entry"], heldout_entries).any()):
        raise AssertionError("held-out entries linked in the training corpus")
    lengths = train_corpus.spans["length"]
    frequency = np.bincount(train_corpus.spans["entry"][lengths >= min_subtokens].astype(np.int64),
                            minlength=len(full.entry_concepts))
    schedule = full.entry_schedule(ontology.frames)
    channel_ontology = {
        "entry_count": len(full.entry_concepts), "atomic_count": len(ontology.atomic_names),
        "relation_count": len(ontology.relation_names), "offsets": schedule.offsets,
        "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": heldout_entries,
        "train_frequency": frequency.tolist(), "alias_table_sha256": full.digest(), "holdout_sha256": tables["holdout_sha256"],
        "relation_names": ontology.relation_names, "atomic_names": ontology.atomic_names,
        "entry_concepts": full.entry_concepts, "concept_names": ontology.concept_names,
        "tokenizer": tokenizer_name, "tokenizer_sha256": fingerprint, "vocab_size": vocab_size,
        "model_vocab_sizes": model_vocab_sizes, "models": config["models"], "host": config["host"],
        "min_subtokens": min_subtokens, "c3_run": str(c3_run), "c3_ontology_sha256": record["ontology_sha256"],
    }
    torch.save(channel_ontology, data_root / "ontology.pt")
    sample = cardinality_report(full, tokenizer, iter_texts(shards, limit=int(c3["data"]["cardinality_docs"])),
                                thresholds=(1, 2, 3, 4))
    c3_sample = (record["cardinality"] or {}).get(tokenizer_name)
    cardinality = {"tokenizer": tokenizer_name, "sample": sample,
                   "sample_matches_c3": None if c3_sample is None else c3_sample == json.loads(json.dumps(sample)),
                   "eval": corpus_cardinality(eval_corpus, [1, 2, 3, 4], frequency, held),
                   "train": corpus_cardinality(train_corpus, [t for t in (1, 2, 3, 4) if t >= min_subtokens], frequency, held)}
    (output_dir / "cardinality.json").write_text(json.dumps(cardinality, indent=2) + "\n")
    timings["cardinality_s"] = time.monotonic() - clock
    c3_eval = record["manifests"].get("eval") or (record["summary"] or {}).get("eval_corpus") or {}
    disk = sum(p.stat().st_size for p in data_root.rglob("*") if p.is_file())
    summary = {
        "host": config["host"], "tokenizer": tokenizer_name, "tokenizer_sha256": fingerprint, "vocab_size": vocab_size,
        "token_dtype": train_manifest["dtype"], "normalization": normalization, "models": model_vocab_sizes,
        "verification": checks,
        "holdout": {"concepts": len(tables["holdout_concepts"]), "heldout_entries": len(heldout_entries),
                    "sha256": tables["holdout_sha256"],
                    "heldout_spans_in_eval": int(np.isin(eval_corpus.spans["entry"], heldout_entries).sum()),
                    "heldout_spans_in_train": 0},
        "eval_documents": {"host": eval_manifest["documents"], "c3": c3_eval.get("documents"),
                           "host_skipped": eval_manifest["skipped_documents"], "c3_skipped": c3_eval.get("skipped_documents"),
                           "same_documents": (eval_manifest["documents"] == c3_eval.get("documents")
                                              and eval_manifest["skipped_documents"] == 0 == c3_eval.get("skipped_documents"))},
        "train_corpus": train_manifest, "eval_corpus": eval_manifest,
        "linked_entries_in_train_at_min_subtokens": int((frequency > 0).sum()),
        "data_root": str(data_root), "disk_bytes": disk, "timings": timings,
        "ontology_sha256": _sha256_file(data_root / "ontology.pt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (output_dir / "report.md").write_text(_report(summary, cardinality))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu")
    return summary


def _report(summary: dict[str, Any], cardinality: dict[str, Any]) -> str:
    train, evaluation = summary["train_corpus"], summary["eval_corpus"]
    lines = [f"# Host corpus `{summary['host']}` (C3 linker and holdout under {summary['tokenizer']})", "",
             f"Serves: {', '.join(summary['models'])}. Tokenizer sha256 `{summary['tokenizer_sha256'][:16]}…`, "
             f"{summary['vocab_size']:,} ids → `{summary['token_dtype']}`; input normalization: {summary['normalization']}.", "",
             f"Train tokens: **{train['tokens']:,}** ({train['documents']:,} documents); eval tokens: **{evaluation['tokens']:,}** "
             f"({evaluation['documents']:,} documents; C3: {summary['eval_documents']['c3']}).",
             f"Held-out entries: {summary['holdout']['heldout_entries']:,} (holdout sha256 `{summary['holdout']['sha256'][:16]}…`, "
             f"{summary['holdout']['heldout_spans_in_eval']:,} held-out spans in eval, 0 in train).", "",
             "## Verification against C3", "", "| Check | Value | Verified against |", "|---|---|---|"]
    for key, check in summary["verification"].items():
        if key == "ontology_pt":
            lines.append(f"| C3 ontology.pt | identical {', '.join(check['identical'])} | `{str(check['c3_ontology_sha256'])[:16]}…` |")
        else:
            lines.append(f"| {key} | `{check['value'][:16]}…` | {', '.join(check['verified_against']) or '—'} |")
    lines += ["", "## Span cardinality, C3 sample (first C3 `cardinality_docs` documents)", "",
              f"Equal to C3's `cardinality.json` row for this tokenizer: **{cardinality['sample_matches_c3']}**.", "",
              "| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cardinality["sample"]:
        lines.append(f"| {r['min_subtokens']} | {r['linkable_entries']:,} | {r['linked_entries']:,} | {r['span_occurrences']:,} | "
                     f"{r['covered_token_fraction']:.3f} | {r['entries_seen_once']:,} | {r['entries_seen_2_to_9']:,} | "
                     f"{r['entries_seen_10_plus']:,} | {r['mean_distinct_entries_per_window']:.1f} |")
    for split in ("eval", "train"):
        lines += ["", f"## Span cardinality, full {split} corpus (strata by training frequency at ℓ ≥ "
                      f"{min(r['min_subtokens'] for r in cardinality['train'])})", "",
                  "| ℓ_min | spans | linked entries | covered-token fraction | held-out entries / spans | unseen | rare 1–9 | mid 10–99 | frequent ≥ 100 |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in cardinality[split]:
            cells = [f"{r[f'{n}_entries']:,} / {r[f'{n}_spans']:,}" for n, _, _ in FREQUENCY_BINS]
            lines.append(f"| {r['min_subtokens']} | {r['span_occurrences']:,} | {r['linked_entries']:,} | "
                         f"{r['covered_token_fraction']:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help=f"one of {sorted(HOSTS)}, or a new slug with --tokenizer")
    parser.add_argument("--c3-run", type=Path, required=True, help="C3 run folder (resolved_config.yaml, holdout_concepts.txt)")
    parser.add_argument("--tokenizer", default=None); parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None, help="run folder (default runs/host-<slug>-v1)")
    parser.add_argument("--data-root", default=None, help="default ~/data/vsa-llm/c3/wordnet-<slug>-v1")
    parser.add_argument("--train-tokens", type=int, default=DEFAULTS["train_tokens"])
    parser.add_argument("--min-subtokens", type=int, default=DEFAULTS["min_subtokens"], help="host ℓ_min for training spans")
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    parser.add_argument("--allow-unverified", action="store_true", help="run without a C3 record to verify against")
    args = parser.parse_args(argv)
    host, tokenizer, models = _host_settings(args.host, args.tokenizer, args.models)
    data_root, output = default_paths(host)
    config = {"host": host, "tokenizer": tokenizer, "models": models, "c3_run": str(args.c3_run), "workers": args.workers,
              "paths": {"data_root": args.data_root or str(data_root)},
              "data": {"train_tokens": args.train_tokens, "min_subtokens": args.min_subtokens,
                       "eval_min_subtokens": DEFAULTS["eval_min_subtokens"]},
              "verify": {"require_c3_record": not args.allow_unverified}}
    print(json.dumps(run(config, args.output or output), indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
