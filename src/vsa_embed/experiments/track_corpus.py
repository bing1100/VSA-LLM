"""C7: build an application track's corpora, holdout, cardinality, feasibility verdict and task items.

One builder for tracks T2–T6 (E8 recipe steps 1–3, 5, 7), driven by `experiments/<track>/<track>.yaml`;
outputs mirror `c3_corpus.py`, so the trainer and the E4 tooling read them unchanged. Optional keys:
`data.expected_holdout_sha256` (fail unless the holdout hashes to it; one holdout across builds with
different tokenizers), `feasibility_strict` (WP-T1's criteria, reported in addition to `FEASIBILITY`) and
`linker.alias_normalization` (`span_channel.ALIAS_NORMALIZATIONS`; "identifier" keeps `_` in code-symbol
aliases — then `ontology.pt` records the mode, `alias_table.json` is written next to it, and the report
compares linking with the default mode).

1. `track.prepare()` writes the domain documents (`docs/eval.jsonl.gz`, `docs/train.jsonl.gz`).
2. Ontology adapter → frames and aliases. The track's synthetic concepts (invented names, frames
   only; the E5.4 zero-shot scenario) are appended after a contamination check: an alias found in a
   domain or general text sample drops the concept.
3. Holdout, frozen and hashed: fixed by the track (T5) or chosen as in C3 from a presample of
   training documents (`holdout_fraction` of the entries with ≥ `holdout_min_count` occurrences and
   an alias of ≥ 2 subtokens, stratified by log-frequency). Node- and alias-disjoint: every alias of a
   held-out concept leaves the training linker, and an entry sharing an alias with a held-out
   concept is held out with it. Synthetic concepts are always held out.
4. Corpora with the host tokenizer (all span lengths stored, so `ℓ_min` is applied when sampling):
   `eval` (domain documents), `eval-general` (general text linked with the track table, for the
   locality check), `train-domain` (≤ `train_domain_tokens`), `train-general` (fills the training
   stream to `train_total_tokens` when the domain corpus is smaller) and `train` = their
   concatenation, so the domain/general mix is exact in tokens.
5. `ontology.pt` (C3 keys plus track keys), cardinality per tokenizer × `ℓ_min` on a domain sample,
   the feasibility verdict per `ℓ_min` under the fixed criteria `FEASIBILITY` (the same for every
   track), and the task items with the holdout list and hashes in `items_dir`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from vsa_embed.data.concat import concat_corpora
from vsa_embed.data.corpus import TokenCorpus, build_corpus, eval_windows
from vsa_embed.experiments.c3_corpus import choose_holdout, iter_texts
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import LINKER_VERSION, AliasTable, CausalLinker, alias_subtoken_lengths, cardinality_report
from vsa_embed.tracks import Track, load_track
from vsa_embed.tracks.common import (SyntheticConcept, names_sha256, occurrences, split_counts, text_vocabulary,
                                     wordnet_forbidden, write_jsonl)

# Feasibility criteria (experiments.md §0.10, E4.7), identical for every track. A threshold ℓ_min is
# feasible if, in the trainer's evaluation sample (`W` evenly spread windows of `window` tokens of the
# domain evaluation corpus, for the smallest W in `eval_windows` that works), the gate's strata have
# enough distinct concepts and occurrences: ≥ 1,000 spans ≈ 8,000 after-span target tokens per stratum,
# which bounds the paired-bootstrap 95% CI half-width near 0.02–0.05 nats for a token-level loss
# difference sd of 1 nat and a design effect up to 4; ≥ 50 distinct entries keeps the stratum from
# resting on a handful of concepts. The linked text must stay a minority (a rare-word channel) but not
# vanish, and training must link enough distinct entries for composition to have signal. "Rare" is the
# gate's *seen* rare stratum (training frequency 1–9 at that ℓ_min); linked entries never seen in
# training ("unseen", not held out) are counted separately — the trainer's `after_rare` stratum
# currently includes them (count < 10), which matters for long-tail tracks.
FEASIBILITY = {
    "window": 1024, "eval_windows": [256, 512, 1024],
    "min_heldout_entries": 50, "min_heldout_spans": 1000,
    "min_rare_entries": 50, "min_rare_spans": 1000, "rare_below": 10,
    "min_train_entries": 1000,
    "min_covered_fraction": 0.002, "max_covered_fraction": 0.5,
}


def feasibility(eval_corpus: TokenCorpus, train_corpus: TokenCorpus, heldout: set[int], *, entry_count: int,
                thresholds: tuple[int, ...] = (1, 2, 3, 4), criteria: dict[str, Any] = FEASIBILITY) -> list[dict[str, Any]]:
    """Per-`ℓ_min` strata counts in the evaluation windows and the verdict (see FEASIBILITY)."""
    window = int(criteria["window"])
    held = np.zeros(entry_count, dtype=bool)
    held[list(heldout)] = True
    rows = []
    for threshold in thresholds:
        frequency = np.bincount(train_corpus.spans["entry"][train_corpus.spans["length"] >= threshold],
                                minlength=entry_count)
        train_entries = int(((frequency > 0) & ~held).sum())
        samples = []
        for count in criteria["eval_windows"]:
            starts = eval_windows(eval_corpus, count=count, length=window)
            held_entries, rare_entries, unseen_entries, linked_entries = set(), set(), set(), set()
            held_spans = rare_spans = unseen_spans = spans_total = covered = tokens = 0
            for start in starts:
                _, spans = eval_corpus.window(start, window, min_subtokens=threshold)
                entries = spans["entry"].astype(np.int64)
                tokens += window
                mask = np.zeros(window, dtype=bool)
                for s, e in zip(spans["start"], spans["end"]):
                    mask[int(s):int(e) + 1] = True
                covered += int(mask.sum())
                is_held = held[entries]
                seen = frequency[entries]
                is_rare = ~is_held & (seen >= 1) & (seen < int(criteria["rare_below"]))
                is_unseen = ~is_held & (seen == 0)
                held_entries.update(entries[is_held].tolist()); rare_entries.update(entries[is_rare].tolist())
                unseen_entries.update(entries[is_unseen].tolist()); unseen_spans += int(is_unseen.sum())
                linked_entries.update(entries.tolist())
                held_spans += int(is_held.sum()); rare_spans += int(is_rare.sum()); spans_total += int(entries.size)
            fraction = covered / max(1, tokens)
            checks = {
                "heldout_entries": len(held_entries) >= criteria["min_heldout_entries"],
                "heldout_spans": held_spans >= criteria["min_heldout_spans"],
                "rare_entries": len(rare_entries) >= criteria["min_rare_entries"],
                "rare_spans": rare_spans >= criteria["min_rare_spans"],
                "train_entries": train_entries >= criteria["min_train_entries"],
                "covered_fraction": criteria["min_covered_fraction"] <= fraction <= criteria["max_covered_fraction"],
            }
            partial = ("heldout_entries", "heldout_spans", "train_entries", "covered_fraction")
            samples.append({"windows_requested": count, "windows": len(starts), "tokens": tokens,
                            "heldout_entries": len(held_entries), "heldout_spans": held_spans,
                            "rare_entries": len(rare_entries), "rare_spans": rare_spans,
                            "unseen_entries": len(unseen_entries), "unseen_spans": unseen_spans,
                            "linked_entries": len(linked_entries), "spans": spans_total,
                            "covered_fraction": fraction, "checks": checks, "pass": all(checks.values()),
                            "pass_heldout_locality": all(checks[k] for k in partial)})
        passing = [s for s in samples if s["pass"]]
        partial_passing = [s for s in samples if s["pass_heldout_locality"]]
        rows.append({"min_subtokens": threshold, "train_entries": train_entries,
                     "train_spans": int(frequency.sum()), "samples": samples,
                     "feasible": bool(passing), "eval_windows_needed": passing[0]["windows"] if passing else None,
                     "heldout_locality_windows_needed": partial_passing[0]["windows"] if partial_passing else None,
                     "failed_checks": [] if passing else sorted(k for k, v in samples[-1]["checks"].items() if not v)})
    return rows


def recommend_min_subtokens(rows: list[dict[str, Any]], default: int = 2) -> dict[str, Any]:
    """ℓ_min = 2 (the E4 default) if feasible, else ℓ_min = 1 (single-token variant), else infeasible.

    An infeasible track also reports whether the held-out stratum and locality alone are powered
    (`heldout_locality`: the E4 gate without its seen-rare comparison), which is the author's call."""
    by = {r["min_subtokens"]: r for r in rows}
    for threshold in (default, 1):
        if threshold in by and by[threshold]["feasible"]:
            return {"min_subtokens": threshold, "verdict": "feasible",
                    "eval_windows": by[threshold]["eval_windows_needed"]}
    row = by.get(default, rows[0])
    return {"min_subtokens": None, "verdict": "infeasible", "failed_checks": row["failed_checks"],
            "heldout_locality": {"min_subtokens": row["min_subtokens"],
                                 "eval_windows": row.get("heldout_locality_windows_needed")}}


def _verify(paths: list[str], expected: dict[str, str] | None) -> None:
    for shard in paths:   # refuse corrupt downloads, as C3 does
        prefix = (expected or {}).get(Path(shard).name)
        if prefix:
            digest = hashlib.sha256()
            with open(shard, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 24), b""):
                    digest.update(chunk)
            if not digest.hexdigest().startswith(prefix):
                raise ValueError(f"{shard} sha256 {digest.hexdigest()[:16]} != expected {prefix}")


def _take(texts: Iterator[str], limit: int) -> list[str]:
    out = []
    for text in texts:
        if len(out) >= limit:
            break
        out.append(text)
    return out


def build_part(texts: Iterator[str], out_dir: Path, *, table: AliasTable, max_tokens: int, request: dict[str, Any],
               tokenizer_sha256: str | None = None, **build: Any) -> dict[str, Any]:
    """`build_corpus` with resumption only if the earlier build had the same request (sources and
    token budget) as well as the same alias table; otherwise the directory is rebuilt. `tokenizer_sha256`
    (the tokenizer fingerprint) is recorded in the manifest, where the trainer checks it against its host."""
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("request") != request:
        manifest_path.unlink()
    extra = {"request": request, **({"tokenizer_sha256": tokenizer_sha256} if tokenizer_sha256 else {})}
    return build_corpus(texts, out_dir, table=table, max_tokens=max_tokens, extra_manifest=extra, reuse=True, **build)


def _metadata_scalars(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool)) or v is None}


def add_synthetic(ontology: Any, synthetic: list[SyntheticConcept]) -> list[int]:
    """Append synthetic concepts (frames given by names) to the ontology; return their indices."""
    relation_index = {name: i for i, name in enumerate(ontology.relation_names)}
    atomic_index = {name: i for i, name in enumerate(ontology.atomic_names)}
    indices = []
    for concept in synthetic:
        frame = []
        for relation, atom in concept.frame:
            if relation not in relation_index or atom not in atomic_index:
                raise KeyError(f"synthetic concept {concept.name!r}: unknown edge ({relation}, {atom})")
            edge = (relation_index[relation], atomic_index[atom])
            if edge not in frame:
                frame.append(edge)
        indices.append(len(ontology.concept_names))
        ontology.concept_names.append(f"synthetic:{concept.name}")
        ontology.frames.append(frame)
        ontology.alias_pairs.extend((alias, indices[-1]) for alias in concept.aliases)
    return indices


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir) if not (output_dir / "resolved_config.yaml").exists() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    paths, data = config["paths"], config["data"]
    data_root = Path(paths["data_root"]).expanduser()
    data_root.mkdir(parents=True, exist_ok=True)
    items_dir = Path(config["items_dir"])
    track: Track = load_track(config["track"], config, data_root)
    general = [str(Path(p).expanduser()) for p in paths["general_shards"]]
    _verify(general, paths.get("general_shard_sha256"))
    tokenizer_name = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    eos, vocab_size, workers = tokenizer.eos_token_id, len(tokenizer), int(config["workers"])
    # A tokenizer that normalizes its input (Qwen: NFC) is decode-checked against the normalized text (as
    # host_corpus); SmolLM2 and GPT-2 have no normalizer, so their builds pass nothing and are unchanged. Every
    # corpus records the tokenizer fingerprint (`TokenCorpus` manifests; the trainer refuses another host's).
    from vsa_embed.data.corpus import tokenizer_fingerprint
    from vsa_embed.experiments.host_corpus import tokenizer_normalization
    normalization_form = tokenizer_normalization(tokenizer)
    fingerprint = tokenizer_fingerprint(tokenizer)
    encode = dict(tokenizer_name=tokenizer_name, eos_id=eos, workers=workers, vocab_size=vocab_size, tokenizer_sha256=fingerprint,
                  **({"normalization": normalization_form} if normalization_form else {}))

    # 1. domain documents.
    documents_summary = track.prepare()

    # 2. ontology, synthetic concepts (contamination-checked).
    ontology = track.ontology()
    sample_docs = int(data["contamination_docs"])
    domain_sample = _take(track.documents("train"), sample_docs) + _take(track.documents("eval"), sample_docs)
    general_sample = list(iter_texts(general, limit=sample_docs))
    forbidden = wordnet_forbidden() | text_vocabulary(domain_sample) | text_vocabulary(general_sample)
    candidates = track.synthetic(ontology, forbidden)
    existing = {a.lower() for a, _ in ontology.alias_pairs}
    hits = occurrences(sorted({a for c in candidates for a in c.aliases}), domain_sample + general_sample)
    synthetic = [c for c in candidates if not any(hits[a] for a in c.aliases) and not any(a.lower() in existing for a in c.aliases)]
    synthetic_concepts = add_synthetic(ontology, synthetic)
    del domain_sample, general_sample

    # 3. holdout.
    normalization = (config.get("linker") or {}).get("alias_normalization", "default")   # "identifier": T2 code symbols
    base_table = AliasTable.from_pairs(ontology.alias_pairs, normalization=normalization)
    alias_lengths = alias_subtoken_lengths(base_table, tokenizer)
    entry_length: dict[int, int] = {}
    for alias, entry in base_table.alias_to_entry.items():
        entry_length[entry] = max(entry_length.get(entry, 0), alias_lengths[alias])
    fixed = track.fixed_holdout(ontology)
    if fixed is None:
        presample_dir = data_root / "presample"
        build_part(track.documents("train"), presample_dir, table=base_table, max_tokens=int(data["presample_tokens"]),
                   request={"source": "domain train presample", "max_tokens": int(data["presample_tokens"])}, **encode)
        counts = Counter(TokenCorpus.open(presample_dir).spans["entry"].tolist())
        synthetic_entries_base = {base_table.alias_to_entry[a.lower()] for c in synthetic for a in c.aliases
                                  if a.lower() in base_table.alias_to_entry}
        for entry in synthetic_entries_base:
            counts.pop(entry, None)
        holdout = choose_holdout(counts, entry_length, base_table, ontology.concept_names,
                                 fraction=float(data["holdout_fraction"]), min_count=int(data["holdout_min_count"]),
                                 seed=int(config["seed"]))
        holdout["method"] = "presample"
    else:
        names = sorted(ontology.concept_names[c] for c in fixed)
        holdout = {"concepts": sorted(fixed), "names": names, "sha256": names_sha256(names),
                   "eligible_entries": None, "chosen_entries": [], "method": "fixed by the track"}
    real_holdout = [c for c in holdout["concepts"] if c not in set(synthetic_concepts)]
    holdout_names = sorted(ontology.concept_names[c] for c in real_holdout)
    holdout_sha = names_sha256(holdout_names)
    expected_holdout = data.get("expected_holdout_sha256")      # optional pin (one holdout across several builds)
    if expected_holdout and holdout_sha != expected_holdout:
        raise ValueError(f"holdout sha256 {holdout_sha} differs from data.expected_holdout_sha256 {expected_holdout}")
    synthetic_sha = names_sha256(c.name for c in synthetic)
    full = AliasTable.from_pairs(ontology.alias_pairs, holdout=real_holdout + synthetic_concepts, include_holdout=True,
                                 normalization=normalization)
    train_table = full.without_holdout()
    heldout_entries = sorted(full.heldout_entries())
    synthetic_set = set(synthetic_concepts)
    synthetic_entries = sorted(i for i in heldout_entries if set(full.entry_concepts[i]) & synthetic_set)
    real_heldout_entries = sorted(set(heldout_entries) - set(synthetic_entries))

    # 4. corpora.
    build = encode
    eval_manifest = build_part(track.documents("eval"), data_root / "eval", table=full, max_tokens=int(data["eval_tokens"]),
                               request={"source": "domain eval", "max_tokens": int(data["eval_tokens"])}, **build)
    general_skip, general_eval_docs = int(data["general_skip_docs"]), int(data["general_eval_docs"])
    eval_general_manifest = build_part(iter_texts(general, skip=general_skip, limit=general_eval_docs), data_root / "eval-general",
                                       table=full, max_tokens=10**12,
                                       request={"source": "general eval", "shards": general, "skip": general_skip,
                                                "documents": general_eval_docs}, **build)
    domain_manifest = build_part(track.documents("train"), data_root / "train-domain", table=train_table,
                                 max_tokens=int(data["train_domain_tokens"]),
                                 request={"source": "domain train", "max_tokens": int(data["train_domain_tokens"])}, **build)
    general_tokens = max(0, int(data["train_total_tokens"]) - domain_manifest["tokens"])
    parts = [data_root / "train-domain"]
    general_manifest = None
    if general_tokens > 0:
        general_manifest = build_part(iter_texts(general, skip=general_skip + general_eval_docs), data_root / "train-general",
                                      table=train_table, max_tokens=general_tokens,
                                      request={"source": "general train", "shards": general, "skip": general_skip + general_eval_docs,
                                               "max_tokens": general_tokens}, **build)
        parts.append(data_root / "train-general")
    mix = {"domain_tokens": domain_manifest["tokens"], "general_tokens": general_manifest["tokens"] if general_manifest else 0}
    mix["domain_fraction"] = mix["domain_tokens"] / max(1, mix["domain_tokens"] + mix["general_tokens"])
    train_manifest = concat_corpora(parts, data_root / "train", extra_manifest={"mix": mix, "tokenizer_sha256": fingerprint},
                                    reuse=True)

    # 5. channel ontology, cardinality, feasibility.
    train_corpus, eval_corpus = TokenCorpus.open(data_root / "train"), TokenCorpus.open(data_root / "eval")
    entry_count = len(full.entry_concepts)
    min_subtokens = int(data["min_subtokens"])
    by_length = {l: np.bincount(train_corpus.spans["entry"][train_corpus.spans["length"] >= l], minlength=entry_count)
                 for l in (1, 2, 3, 4)}
    frequency = by_length[min_subtokens]
    schedule = full.entry_schedule(ontology.frames)
    channel_ontology = {
        "entry_count": entry_count, "atomic_count": len(ontology.atomic_names),
        "relation_count": len(ontology.relation_names), "offsets": schedule.offsets,
        "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": heldout_entries,
        "train_frequency": frequency.tolist(), "alias_table_sha256": full.digest(), "holdout_sha256": holdout_sha,
        "relation_names": ontology.relation_names, "atomic_names": ontology.atomic_names,
        "entry_concepts": full.entry_concepts, "concept_names": ontology.concept_names,
        # track keys
        "track": config["track"], "tokenizer": tokenizer_name, "min_subtokens": min_subtokens,
        "heldout_real_entries": real_heldout_entries, "synthetic_entries": synthetic_entries,
        "synthetic_sha256": synthetic_sha, "train_frequency_by_min_subtokens": {l: v.tolist() for l, v in by_length.items()},
    }
    if normalization != "default":
        # a consumer rebuilding the table from alias pairs must use the same mode; the sidecar is the table itself
        from vsa_embed.evaluation.channel_probes import save_alias_table
        channel_ontology["alias_normalization"] = normalization
        save_alias_table(full, data_root / "alias_table.json")
    torch.save(channel_ontology, data_root / "ontology.pt")
    cardinality_texts = _take(track.documents("eval"), int(data["cardinality_docs"]))
    cardinality = {}
    for name in config["cardinality_tokenizers"]:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        cardinality[name] = cardinality_report(full, tok, cardinality_texts, thresholds=(1, 2, 3, 4))
    (output_dir / "cardinality.json").write_text(json.dumps(cardinality, indent=2) + "\n")
    linking = None
    if normalization != "default":       # how much the mode matters: the same sample linked with the default mode
        default_full = AliasTable.from_pairs(ontology.alias_pairs, holdout=real_holdout + synthetic_concepts,
                                             include_holdout=True)
        linking = {"tokenizer": tokenizer_name, "documents": len(cardinality_texts), "concepts": len(ontology.concept_names)}
        for mode, table in (("default", default_full), (normalization, full)):
            row = cardinality_report(table, tokenizer, cardinality_texts, thresholds=(1,))[0]
            linked_concepts = len({c for e in _linked_entries(table, tokenizer, cardinality_texts) for c in table.entry_concepts[e]})
            linking[mode] = {"linked_entries": row["linked_entries"], "span_occurrences": row["span_occurrences"],
                             "covered_token_fraction": row["covered_token_fraction"], "linked_concepts": linked_concepts,
                             "linked_concept_fraction": linked_concepts / max(1, len(ontology.concept_names))}
        (output_dir / "linking_by_normalization.json").write_text(json.dumps(linking, indent=2) + "\n")
    feasible_rows = feasibility(eval_corpus, train_corpus, set(real_heldout_entries), entry_count=entry_count)
    recommendation = recommend_min_subtokens(feasible_rows)
    (output_dir / "feasibility.json").write_text(json.dumps({"criteria": FEASIBILITY, "rows": feasible_rows,
                                                             "recommendation": recommendation}, indent=2) + "\n")
    strict_rows = None
    if config.get("feasibility_strict"):      # optional: WP-T1's bar on the whole domain evaluation split
        from vsa_embed.experiments.t1_open_corpus import feasibility_report
        strict_rows = feasibility_report(data_root / "eval", heldout_entries=real_heldout_entries, entry_count=entry_count,
                                         frequencies={l: (v, "measured") for l, v in by_length.items()},
                                         criteria=config["feasibility_strict"])
        (output_dir / "feasibility_strict.json").write_text(json.dumps({"criteria": config["feasibility_strict"],
                                                                        "rows": strict_rows}, indent=2) + "\n")
    general_eval = TokenCorpus.open(data_root / "eval-general")
    general_linked = {l: int((general_eval.spans["length"] >= l).sum()) for l in (1, 2, 3, 4)}

    # 6. items, holdout list and hashes.
    concept_frequency: dict[str, int] = {}
    for entry, concepts in enumerate(full.entry_concepts):
        for concept in concepts:
            name = ontology.concept_names[concept]
            concept_frequency[name] = concept_frequency.get(name, 0) + int(frequency[entry])
    context = {"ontology": ontology, "holdout_concepts": real_holdout, "synthetic": synthetic,
               "synthetic_concepts": synthetic_concepts, "concept_frequency": concept_frequency, "table": full}
    items = track.items(context)
    items_dir.mkdir(parents=True, exist_ok=True)
    (items_dir / "holdout_concepts.txt").write_text("\n".join(holdout_names) + "\n")
    (output_dir / "holdout_concepts.txt").write_text("\n".join(holdout_names) + "\n")
    synthetic_file = write_jsonl(items_dir / "synthetic_concepts.jsonl", (c.to_json() for c in synthetic))
    item_files = {"synthetic_concepts.jsonl": {**synthetic_file, "splits": {"synthetic": len(synthetic)}}}
    for stem, rows in sorted(items.items()):
        item_files[f"{stem}.jsonl"] = {**write_jsonl(items_dir / f"{stem}.jsonl", rows), "splits": split_counts(rows)}
    holdout_record = {"track": config["track"], "holdout_sha256": holdout_sha, "heldout_concepts": len(holdout_names),
                      "method": holdout["method"], "synthetic_sha256": synthetic_sha, "synthetic_concepts": len(synthetic),
                      "alias_table_sha256": full.digest(), "linker_version": LINKER_VERSION, "files": item_files}
    (items_dir / "holdout.json").write_text(json.dumps(holdout_record, indent=2) + "\n")

    summary = {
        "track": config["track"], "documents": documents_summary,
        "ontology": {**_metadata_scalars(ontology.metadata), "concepts": len(ontology.concept_names),
                     "atomics": len(ontology.atomic_names), "relations": len(ontology.relation_names),
                     "aliases": len(full.alias_to_entry), "entries": entry_count},
        "holdout": {"concepts": len(holdout_names), "heldout_entries": len(real_heldout_entries),
                    "eligible_entries": holdout.get("eligible_entries"), "method": holdout["method"], "sha256": holdout_sha},
        "synthetic": {"candidates": len(candidates), "kept": len(synthetic), "sha256": synthetic_sha,
                      "entries": len(synthetic_entries),
                      "dropped_for_text_hits": sum(1 for c in candidates if any(hits[a] for a in c.aliases))},
        "eval_corpus": eval_manifest, "eval_general_corpus": eval_general_manifest, "train_corpus": train_manifest,
        "mix": mix, "general_text_spans_by_min_subtokens": general_linked,
        "feasibility": recommendation, "items": {k: {"rows": v["rows"], "splits": v["splits"]} for k, v in item_files.items()},
        "data_root": str(data_root), "ontology_sha256": hashlib.sha256((data_root / "ontology.pt").read_bytes()).hexdigest(),
    }
    if strict_rows is not None:
        summary["feasibility_strict"] = {r["min_subtokens"]: r["verdict"] for r in strict_rows}
    if linking is not None:
        summary["alias_normalization"] = normalization
        summary["linking_by_normalization"] = linking
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    report = render_report(config, summary, cardinality, feasible_rows)
    if strict_rows is not None:
        report += render_strict(config["feasibility_strict"], strict_rows)
    if linking is not None:
        report += render_linking(linking, normalization)
    (output_dir / "report.md").write_text(report)
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu")
    return summary


def render_report(config: dict[str, Any], summary: dict[str, Any], cardinality: dict[str, Any],
                  rows: list[dict[str, Any]]) -> str:
    mix, rec = summary["mix"], summary["feasibility"]
    lines = [f"# {config['experiment']}", "",
             f"Train tokens: **{summary['train_corpus']['tokens']:,}** (domain {mix['domain_tokens']:,}, general "
             f"{mix['general_tokens']:,}; domain fraction {mix['domain_fraction']:.2f}); domain eval tokens: "
             f"**{summary['eval_corpus']['tokens']:,}**; general eval tokens: {summary['eval_general_corpus']['tokens']:,}.",
             f"Ontology: {summary['ontology']['concepts']:,} concepts, {summary['ontology']['atomics']:,} atomics, "
             f"{summary['ontology']['relations']} relations, {summary['ontology']['aliases']:,} aliases, "
             f"{summary['ontology']['entries']:,} entries.",
             f"Holdout ({summary['holdout']['method']}): {summary['holdout']['concepts']:,} concepts, "
             f"{summary['holdout']['heldout_entries']:,} entries, sha256 `{summary['holdout']['sha256'][:16]}…`; "
             f"synthetic zero-shot concepts: {summary['synthetic']['kept']:,} (sha256 `{summary['synthetic']['sha256'][:16]}…`).",
             "", f"**Feasibility verdict: {rec['verdict']}**" + (f" at ℓ_min = {rec['min_subtokens']} with "
             f"{rec['eval_windows']} evaluation windows." if rec["verdict"] == "feasible" else
             f" (failed: {rec.get('failed_checks')}). Held-out stratum and locality alone at ℓ_min = "
             f"{rec['heldout_locality']['min_subtokens']}: " + (f"powered with {rec['heldout_locality']['eval_windows']} "
             "evaluation windows." if rec["heldout_locality"]["eval_windows"] else "not powered.")),
             "", "## Feasibility (domain evaluation windows; criteria in `feasibility.json`)", "",
             "| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|"]
    for row in rows:
        for s in row["samples"]:
            lines.append(f"| {row['min_subtokens']} | {s['windows']} | {s['heldout_entries']:,} | {s['heldout_spans']:,} | "
                         f"{s['rare_entries']:,} | {s['rare_spans']:,} | {s['unseen_entries']:,} | {s['unseen_spans']:,} | "
                         f"{s['covered_fraction']:.3f} | {row['train_entries']:,} | "
                         f"{'yes' if s['pass'] else 'no'} |")
    lines += ["", "## Span cardinality (domain evaluation sample)", ""]
    for name, table_rows in cardinality.items():
        lines += [f"### {name}", "", "| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in table_rows:
            lines.append(f"| {r['min_subtokens']} | {r['linkable_entries']:,} | {r['linked_entries']:,} | {r['span_occurrences']:,} | "
                         f"{r['covered_token_fraction']:.3f} | {r['entries_seen_once']:,} | {r['entries_seen_2_to_9']:,} | "
                         f"{r['entries_seen_10_plus']:,} | {r['mean_distinct_entries_per_window']:.1f} |")
        lines.append("")
    lines += ["## Items", "", "| file | rows | splits |", "|---|---:|---|"]
    for name, info in summary["items"].items():
        lines.append(f"| `{name}` | {info['rows']:,} | {info['splits']} |")
    lines.append("")
    return "\n".join(lines)


def _linked_entries(table: AliasTable, tokenizer: Any, texts: list[str]) -> set[int]:
    """Entries linked (any span length) in `texts`."""
    linker = CausalLinker(table, min_subtokens=1)
    linked: set[int] = set()
    for text in texts:
        offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        linked.update(span.entry for span in linker.link(text, offsets))
    return linked


def render_linking(linking: dict[str, Any], normalization: str) -> str:
    lines = [f"## Alias normalization: `{normalization}` vs the default mode", "",
             f"The same {linking['documents']:,} domain evaluation documents linked with both alias tables "
             f"({linking['tokenizer']}, ℓ_min = 1; `linking_by_normalization.json`). The default mode turns `_` into a "
             "space, so a snake_case alias can never match code.", "",
             "| mode | linked entries | span occurrences | covered-token fraction | linked concepts | of all concepts |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode in ("default", normalization):
        r = linking[mode]
        lines.append(f"| {mode} | {r['linked_entries']:,} | {r['span_occurrences']:,} | {r['covered_token_fraction']:.3f} | "
                     f"{r['linked_concepts']:,} | {r['linked_concept_fraction']:.3f} |")
    return "\n".join(lines) + "\n"


def render_strict(criteria: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Report section for WP-T1's stricter bar (open decision 17), on the whole domain evaluation split."""
    lines = ["## Feasibility against WP-T1's bar (whole domain evaluation split)", "",
             f"≥ {criteria['heldout_min_entries_5plus']} held-out entries with ≥ 5 occurrences and ≥ "
             f"{criteria['heldout_min_occurrences']:,} held-out occurrences; ≥ {criteria['rare_min_entries']} rare (training "
             f"frequency 1–9) entries linked and ≥ {criteria['rare_min_occurrences']:,} rare occurrences "
             "(`t1_open_corpus.feasibility_report`; details in `feasibility_strict.json`).", "",
             "| ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | verdict | eval windows needed |",
             "|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        s = r["split"]
        needed = f"{r['eval_windows_needed']:,}" if r["eval_windows_needed"] else "not reached"
        lines.append(f"| {r['min_subtokens']} | {s['tokens_considered']:,} | {s['heldout_occurrences']:,} | "
                     f"{s['heldout_entries_5plus']:,} | {s['rare_occurrences']:,} | {s['rare_entries_linked']:,} | "
                     f"{r['verdict']} | {needed} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(yaml.safe_load(args.config.read_text()), args.output), indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
