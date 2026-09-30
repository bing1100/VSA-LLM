"""C3: build the general-track corpus (FineWeb-Edu × WordNet), freeze the holdout, report cardinality.

1. Build the WordNet frame ontology and its alias table.
2. Link a pre-sample of training documents to count how often each entry occurs, then choose
   `holdout_fraction` of the eligible entries (≥ `min_count` occurrences, an alias of ≥ 2 GPT-2
   subtokens), stratified by log-frequency. Their concepts form the frozen holdout; its sorted name
   list is hashed. Any entry containing a held-out concept is held out too (a polysemous alias
   would otherwise leak the held-out string).
3. Build the evaluation corpus (all aliases, so held-out concepts are linked) and the training
   corpus (held-out aliases removed); both share entry ids.
4. Write the channel ontology (entry frames, counts, held-out entries, training frequencies) and the
   cardinality tables for each tokenizer.

Large artifacts go under `paths.data_root`; manifests, the holdout list and reports go to the run folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer

from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.ontologies.wordnet import build_wordnet_ontology
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import AliasTable, alias_subtoken_lengths, cardinality_report


def iter_texts(paths: list[str], *, skip: int = 0, limit: int | None = None, column: str = "text") -> Iterator[str]:
    seen = produced = 0
    for path in paths:
        # Keep the ParquetFile referenced while iterating: letting it be garbage-collected
        # mid-iteration crashes pyarrow (segfault observed).
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=[column], batch_size=2048):
            for text in batch.column(0).to_pylist():
                seen += 1
                if seen <= skip:
                    continue
                if limit is not None and produced >= limit:
                    return
                produced += 1
                yield text


def choose_holdout(counts: Counter, lengths: dict[int, int], table: AliasTable, ontology_names: list[str], *,
                   fraction: float, min_count: int, seed: int) -> dict[str, Any]:
    eligible = sorted(e for e, c in counts.items() if c >= min_count and lengths.get(e, 0) >= 2)
    rng = np.random.default_rng(seed)
    by_bin: dict[int, list[int]] = {}
    for entry in eligible:
        by_bin.setdefault(int(math.log2(counts[entry])), []).append(entry)
    chosen: list[int] = []
    for _, entries in sorted(by_bin.items()):
        k = max(1, round(len(entries) * fraction))
        chosen += sorted(rng.choice(entries, size=min(k, len(entries)), replace=False).tolist())
    concepts = sorted({c for e in chosen for c in table.entry_concepts[e]})
    names = sorted(ontology_names[c] for c in concepts)
    return {"chosen_entries": sorted(chosen), "concepts": concepts, "names": names,
            "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(), "eligible_entries": len(eligible)}


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir)
    paths, data = config["paths"], config["data"]
    data_root = Path(paths["data_root"]).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    tokenizer_name = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    eos = tokenizer.eos_token_id
    shards = [str(Path(p).expanduser()) for p in paths["shards"]]

    ontology = build_wordnet_ontology(wn, max_atomics=int(config["ontology"]["max_atomics"]),
                                      max_degree=int(config["ontology"]["max_degree"]))
    base_table = AliasTable.from_pairs(ontology.alias_pairs)
    alias_lengths = alias_subtoken_lengths(base_table, tokenizer)
    entry_length: dict[int, int] = {}
    for alias, entry in base_table.alias_to_entry.items():
        entry_length[entry] = max(entry_length.get(entry, 0), alias_lengths[alias])

    # 2. frequency pre-sample on training documents (after the evaluation documents).
    eval_docs, presample_docs = int(data["eval_docs"]), int(data["presample_docs"])
    presample_dir = data_root / "presample"
    build_corpus(iter_texts(shards, skip=eval_docs, limit=presample_docs), presample_dir,
                 tokenizer_name=tokenizer_name, table=base_table, eos_id=eos, max_tokens=10**12,
                 workers=int(config["workers"]))
    presample = TokenCorpus.open(presample_dir)
    counts = Counter(presample.spans["entry"].tolist())
    holdout = choose_holdout(counts, entry_length, base_table, ontology.concept_names,
                             fraction=float(data["holdout_fraction"]), min_count=int(data["holdout_min_count"]),
                             seed=int(config["seed"]))
    (output_dir / "holdout_concepts.txt").write_text("\n".join(holdout["names"]) + "\n")

    # 3. tables sharing entry ids; evaluation and training corpora.
    full = AliasTable.from_pairs(ontology.alias_pairs, holdout=holdout["concepts"], include_holdout=True)
    train_table = full.without_holdout()
    heldout_entries = sorted(full.heldout_entries())
    eval_manifest = build_corpus(iter_texts(shards, limit=eval_docs), data_root / "eval", tokenizer_name=tokenizer_name,
                                 table=full, eos_id=eos, max_tokens=10**12, workers=int(config["workers"]))
    # Training spans are stored from ℓ ≥ train_min_subtokens (single-token spans cover ~45% of
    # tokens with WordNet); an all-lengths slice serves the ℓ_min = 1 feasibility sweep.
    train_manifest = build_corpus(iter_texts(shards, skip=eval_docs), data_root / "train", tokenizer_name=tokenizer_name,
                                  table=train_table, eos_id=eos, max_tokens=int(data["train_tokens"]),
                                  workers=int(config["workers"]), min_subtokens=int(data["train_min_subtokens"]))
    l1_manifest = build_corpus(iter_texts(shards, skip=eval_docs), data_root / "train-l1", tokenizer_name=tokenizer_name,
                               table=train_table, eos_id=eos, max_tokens=int(data["l1_slice_tokens"]),
                               workers=int(config["workers"]), min_subtokens=1)

    # 4. channel ontology and cardinality.
    train_corpus = TokenCorpus.open(data_root / "train")
    lengths = train_corpus.spans["length"]
    frequency = np.bincount(train_corpus.spans["entry"][lengths >= int(data["min_subtokens"])],
                            minlength=len(full.entry_concepts))
    schedule = full.entry_schedule(ontology.frames)
    channel_ontology = {
        "entry_count": len(full.entry_concepts), "atomic_count": len(ontology.atomic_names),
        "relation_count": len(ontology.relation_names), "offsets": schedule.offsets,
        "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": heldout_entries,
        "train_frequency": frequency.tolist(), "alias_table_sha256": full.digest(), "holdout_sha256": holdout["sha256"],
        "relation_names": ontology.relation_names, "atomic_names": ontology.atomic_names,
        "entry_concepts": full.entry_concepts, "concept_names": ontology.concept_names,
    }
    torch.save(channel_ontology, data_root / "ontology.pt")
    sample_texts = list(iter_texts(shards, limit=int(data["cardinality_docs"])))
    cardinality = {}
    for name in config["cardinality_tokenizers"]:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        cardinality[name] = cardinality_report(full, tok, sample_texts, thresholds=(1, 2, 3, 4))
    (output_dir / "cardinality.json").write_text(json.dumps(cardinality, indent=2) + "\n")
    held = set(heldout_entries)
    heldout_eval_spans = int(np.isin(TokenCorpus.open(data_root / "eval").spans["entry"], list(held)).sum())
    summary = {
        "ontology": {**ontology.metadata, "concepts": len(ontology.concept_names), "atomics": len(ontology.atomic_names),
                     "relations": len(ontology.relation_names), "aliases": len(full.alias_to_entry),
                     "entries": len(full.entry_concepts)},
        "holdout": {"concepts": len(holdout["concepts"]), "chosen_entries": len(holdout["chosen_entries"]),
                    "heldout_entries": len(heldout_entries), "eligible_entries": holdout["eligible_entries"],
                    "sha256": holdout["sha256"], "heldout_spans_in_eval": heldout_eval_spans},
        "train_corpus": train_manifest, "train_l1_slice": l1_manifest, "eval_corpus": eval_manifest,
        "linked_entries_in_train_at_min_subtokens": int((frequency > 0).sum()),
        "data_root": str(data_root), "ontology_sha256": hashlib.sha256((data_root / "ontology.pt").read_bytes()).hexdigest(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    lines = ["# C3 general corpus (FineWeb-Edu × WordNet)", "",
             f"Train tokens: **{train_manifest['tokens']:,}**; eval tokens: **{eval_manifest['tokens']:,}**.",
             f"Entries: {len(full.entry_concepts):,}; held-out entries: {len(heldout_entries):,} "
             f"(holdout sha256 `{holdout['sha256'][:16]}…`, {heldout_eval_spans:,} held-out spans in eval).", "",
             "## Span cardinality (sample of documents)", ""]
    for name, rows in cardinality.items():
        lines += [f"### {name}", "", "| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['min_subtokens']} | {r['linkable_entries']:,} | {r['linked_entries']:,} | {r['span_occurrences']:,} | "
                         f"{r['covered_token_fraction']:.3f} | {r['entries_seen_once']:,} | {r['entries_seen_2_to_9']:,} | "
                         f"{r['entries_seen_10_plus']:,} | {r['mean_distinct_entries_per_window']:.1f} |")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(yaml.safe_load(args.config.read_text()), args.output), indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
