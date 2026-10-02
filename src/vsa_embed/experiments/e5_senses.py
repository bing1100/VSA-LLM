"""E5.2 sense alignment of edge attention on jointly trained models (D5.2; experiments.md E5.2, H-C).

SemCor sentences (WSD Evaluation Framework, Raganato et al. 2017) are linked with the run's
evaluation linker. A tagged word counts when the entry injected at its last subtoken is a
polysemous union entry (≥ 2 concepts) that contains the gold synset. Its frame is the union of the
member concepts' frames, so every edge is owned by one or more senses. With the run's own context
query (P1, exactly as the channel builds it) the composer's edge weights give an attention mass per
sense, `mass(c) = Σ_e w_e · [c owns e] / |owners(e)|` (shared edges split equally, as in E1).

Metrics per item, aggregated with paired bootstrap CIs over items:
- `sense_accuracy`: argmax mass = gold (ties go to the WordNet-earlier sense, the E1 rule; the tie
  rate is reported, and a static composer ties whenever two senses own as many edges);
- baselines: WordNet first sense (`wn1`, E1's MFS) and SemCor most-frequent sense cross-fitted over
  document halves (`semcor_mfs`; counts from the other half, so no item counts itself);
- `gold_mass_share` − `uniform_share` (the share the gold sense would get with uniform weights):
  the lift attention gives the gold sense, robust to ties.
Split by entry status (held-out entries are composed with zero update; seen entries trained).

Per-concept frames come from `ontology["concept_frames"]` when present, else from rebuilding the
WordNet ontology (`build_wordnet_ontology`, `--max-degree` as C3); either way they must reproduce
the run's entry schedule exactly, or the run stops.

    python -m vsa_embed.experiments.e5_senses --run RUN --output OUT [--dataset semcor|ALL] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ..compose import FrameSchedule
from ..data.semcor import WSDInstance, load_wsd_instances
from ..evaluation import channel_probes as cp
from ..span_channel import AliasTable
from ..statistics import paired_bootstrap_ci
from .e5_common import (E5Run, finish_output, fmt, fmt_ci, occurrence_weights, open_run, span_contexts, start_output,
                        status_of, write_json)

WN_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def concept_frames(ontology: dict[str, Any], table: AliasTable, *, wordnet: Any = None,
                   max_degree: int = 16) -> list[list[tuple[int, int]]]:
    """Per-concept frames that reproduce the ontology's entry schedule (union frames) exactly."""
    if ontology.get("concept_frames") is not None:
        frames = [[tuple(edge) for edge in frame] for frame in ontology["concept_frames"]]
    else:
        from ..ontologies.wordnet import build_wordnet_ontology
        if wordnet is None:
            from nltk.corpus import wordnet
        rebuilt = build_wordnet_ontology(wordnet, max_atomics=int(ontology["atomic_count"]), max_degree=max_degree)
        if rebuilt.concept_names != list(ontology["concept_names"]) or rebuilt.atomic_names != list(ontology["atomic_names"]):
            raise ValueError("the rebuilt WordNet ontology has other concepts or atomics than the run's ontology")
        frames = rebuilt.frames
    schedule = table.entry_schedule(frames)
    expected = FrameSchedule(torch.as_tensor(ontology["offsets"]), torch.as_tensor(ontology["relations"]),
                             torch.as_tensor(ontology["fillers"]))
    if not (torch.equal(schedule.offsets, expected.offsets) and torch.equal(schedule.relations, expected.relations)
            and torch.equal(schedule.fillers, expected.fillers)):
        raise ValueError("per-concept frames do not reproduce the run's entry schedule (check --max-degree)")
    return frames


def edge_owners(schedule: FrameSchedule, entry: int, concepts: Sequence[int], frames: Sequence[Sequence[tuple[int, int]]]) -> list[list[int]]:
    """For each edge of `entry` (schedule order), the member concepts whose own frame has it."""
    start, end = int(schedule.offsets[entry]), int(schedule.offsets[entry + 1])
    sets = {c: set(map(tuple, frames[c])) for c in concepts}
    return [[c for c in concepts if (int(r), int(f)) in sets[c]]
            for r, f in zip(schedule.relations[start:end].tolist(), schedule.fillers[start:end].tolist())]


def sense_masses(weights: Sequence[float], owners: Sequence[Sequence[int]]) -> Counter:
    mass: Counter = Counter()
    for w, own in zip(weights, owners):
        for c in own:
            mass[c] += float(w) / len(own)
    return mass


def wordnet_rank(wordnet: Any) -> Callable[[str, str, str], int]:
    """Rank of a synset among the senses of (lemma, POS) in WordNet order; other synsets after them."""
    cache: dict[tuple[str, str], list[str]] = {}

    def rank(lemma: str, pos: str, synset: str) -> int:
        key = (lemma.lower(), pos)
        if key not in cache:
            cache[key] = [s.name() for s in wordnet.synsets(key[0].replace(" ", "_"), WN_POS.get(pos))] if pos in WN_POS else []
        order = cache[key]
        return order.index(synset) if synset in order else len(order) + 1

    return rank


def document_fold(sentence_id: str) -> int:
    """0/1 by a hash of the SemCor document id (`d000` of `d000.s001`)."""
    return int(hashlib.sha256(sentence_id.rsplit(".", 1)[0].encode()).hexdigest(), 16) % 2


def link_instances(run: E5Run, instances: Sequence[WSDInstance], synset_of: Callable[[str], str | None],
                   concept_index: dict[str, int]) -> list[dict[str, Any]]:
    """Tagged words whose injected entry is a polysemous union entry containing the gold sense."""
    table = run.table
    by_sentence: dict[str, list[WSDInstance]] = defaultdict(list)
    for inst in instances:
        by_sentence[inst.sentence_id].append(inst)
    items = []
    sentence_ids = list(by_sentence)
    for start in range(0, len(sentence_ids), 256):
        chunk = sentence_ids[start:start + 256]
        texts = [by_sentence[s][0].text for s in chunk]
        encoded = run.tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False, truncation=True,
                                max_length=run.adapter.max_length)
        for sid, text, offsets in zip(chunk, texts, encoded["offset_mapping"]):
            offsets = [tuple(o) for o in offsets]
            spans = run.adapter.linker.link(text, offsets)
            for inst in by_sentence[sid]:
                index = cp.read_index(offsets, inst.start, inst.end)
                entries = [s.entry for s in spans if s.inject_token == index]
                if len(entries) != 1:
                    continue
                entry = entries[0]
                concepts = list(table.entry_concepts[entry])
                gold = next((concept_index.get(s) for s in (synset_of(k) for k in inst.keys) if s in concept_index
                             and concept_index[s] in concepts), None)
                if len(concepts) < 2 or gold is None:
                    continue
                items.append({"id": inst.instance_id, "sentence": sid, "text": text, "inject": index, "entry": entry,
                              "concepts": concepts, "gold": gold, "lemma": inst.lemma, "pos": inst.pos,
                              "fold": document_fold(sid)})
    return items


@torch.no_grad()
def attention_masses(run: E5Run, items: list[dict[str, Any]], frames: Sequence[Sequence[tuple[int, int]]], *,
                     batch: int = 64) -> None:
    """Adds per-item edge weights, sense masses, gold share and uniform share."""
    composer, schedule = run.composer, run.composer.schedule
    tokenizer = run.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    owners_cache: dict[int, list[list[int]]] = {}
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        encoded = tokenizer([it["text"] for it in chunk], add_special_tokens=False, padding=True, truncation=True,
                            max_length=run.adapter.max_length, return_tensors="pt")
        ids = encoded["input_ids"].to(run.device)
        spans = {"batch": torch.arange(len(chunk), device=run.device),
                 "inject": torch.tensor([it["inject"] for it in chunk], device=run.device),
                 "entry": torch.tensor([it["entry"] for it in chunk], device=run.device)}
        context = span_contexts(run.model, ids, spans)
        weights, _, segments = occurrence_weights(composer, spans["entry"], context)
        weights, segments = weights.cpu(), segments.cpu()
        for k, item in enumerate(chunk):
            w = weights[segments == k].tolist()
            if item["entry"] not in owners_cache:
                owners_cache[item["entry"]] = edge_owners(schedule, item["entry"], item["concepts"], frames)
            owners = owners_cache[item["entry"]]
            mass = sense_masses(w, owners)
            uniform = sense_masses([1.0] * len(owners), owners)
            total, uniform_total = sum(mass.values()), sum(uniform.values())
            item["mass"] = {int(c): float(mass[c]) for c in item["concepts"]}
            item["gold_share"] = float(mass[item["gold"]] / total) if total else None
            item["uniform_share"] = float(uniform[item["gold"]] / uniform_total) if uniform_total else None


def score_items(items: list[dict[str, Any]], rank: Callable[[str, str, str], int], concept_names: Sequence[str],
                heldout: set[int], frequency: np.ndarray | None) -> None:
    """Predictions of attention, WordNet first sense and cross-fitted SemCor MFS; correctness flags."""
    counts: dict[tuple[int, str, str], Counter] = defaultdict(Counter)
    for it in items:
        counts[(it["fold"], it["lemma"].lower(), it["pos"])][it["gold"]] += 1
    for it in items:
        order = sorted(it["concepts"], key=lambda c: (rank(it["lemma"], it["pos"], concept_names[c]), concept_names[c]))
        mass = it["mass"]
        best = max(mass.values())
        top = [c for c in order if abs(mass[c] - best) < 1e-9]
        it["predicted"], it["tie"] = top[0], len(top) > 1
        it["wn1"] = order[0]
        other = counts[(1 - it["fold"], it["lemma"].lower(), it["pos"])]
        known = [c for c in order if other[c] > 0]
        it["semcor_mfs"] = max(known, key=lambda c: (other[c], -order.index(c))) if known else order[0]
        for name in ("predicted", "wn1", "semcor_mfs"):
            it[f"{name}_correct"] = int(it[name] == it["gold"])
        it["status"] = status_of(it["entry"], heldout, frequency)


def summarize(items: list[dict[str, Any]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"n": 0}
        attention = np.asarray([r["predicted_correct"] for r in rows], dtype=float)
        out: dict[str, Any] = {"n": len(rows), "lemmas": len({(r["lemma"].lower(), r["pos"]) for r in rows}),
                               "entries": len({r["entry"] for r in rows}), "sense_accuracy": float(attention.mean()),
                               "tie_rate": float(np.mean([r["tie"] for r in rows]))}
        for baseline in ("wn1", "semcor_mfs"):
            values = np.asarray([r[f"{baseline}_correct"] for r in rows], dtype=float)
            out[f"{baseline}_accuracy"] = float(values.mean())
            out[f"sense_minus_{baseline}"] = paired_bootstrap_ci(attention, values, resamples=resamples, seed=seed)
        shares = [(r["gold_share"], r["uniform_share"]) for r in rows if r.get("gold_share") is not None]
        if shares:
            gold, uniform = np.asarray(shares).T
            out["gold_mass_share"] = float(gold.mean()); out["uniform_share"] = float(uniform.mean())
            out["gold_share_lift"] = paired_bootstrap_ci(gold, uniform, resamples=resamples, seed=seed)
        return out

    summary = {"all": block(items)}
    for name, members in (("heldout", {"heldout"}), ("seen", {"rare", "mid", "frequent", "seen"})):
        summary[name] = block([it for it in items if it["status"] in members])
    return summary


def render(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# E5.2 sense alignment — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Run `{source['run']}`; composition `{(source.get('channel') or {}).get('composition')}`, context window "
             f"{(source.get('channel') or {}).get('context_window')}; dataset {header['dataset']}: {header['instances']:,} tagged "
             f"instances, {header['linked']:,} on polysemous linked entries containing the gold sense.", "",
             "| subset | items | lemmas | attention acc. | WN first sense | SemCor MFS (cross-fit) | attention − WN1 | attention − MFS | gold-mass lift | tie rate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b in summary.items():
        if not b.get("n"):
            lines.append(f"| {name} | 0 | | | | | | | | |"); continue
        lines.append(f"| {name} | {b['n']:,} | {b['lemmas']:,} | {fmt(b['sense_accuracy'])} | {fmt(b['wn1_accuracy'])} | "
                     f"{fmt(b['semcor_mfs_accuracy'])} | {fmt_ci(b['sense_minus_wn1'])} | {fmt_ci(b['sense_minus_semcor_mfs'])} | "
                     f"{fmt_ci(b.get('gold_share_lift'))} | {fmt(b['tie_rate'], 3)} |")
    lines += ["", "Ties go to the WordNet-earlier sense (E1 rule); a static composer predicts by edge counts.", ""]
    return "\n".join(lines)


def run_experiment(args: argparse.Namespace, *, wordnet: Any = None) -> dict[str, Any]:
    if wordnet is None:
        from nltk.corpus import wordnet
    config = {"experiment": "e5.2-sense-alignment", "run": str(args.run), "checkpoint": args.checkpoint,
              "dataset": args.dataset, "limit": args.limit, "max_degree": args.max_degree, "seed": args.seed,
              "resamples": args.resamples, "probes_root": str(args.probes_root)}
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device)
    if run.composer is None:
        raise ValueError(f"E5.2 needs a composition channel; {run.path} has channel mode {run.mode!r}")
    base = Path(args.probes_root) / "wsd" / "WSD_Evaluation_Framework"
    if args.dataset == "semcor":
        xml, keys = base / "Training_Corpora" / "SemCor" / "semcor.data.xml", base / "Training_Corpora" / "SemCor" / "semcor.gold.key.txt"
    else:
        xml = base / "Evaluation_Datasets" / args.dataset / f"{args.dataset}.data.xml"
        keys = base / "Evaluation_Datasets" / args.dataset / f"{args.dataset}.gold.key.txt"
    instances = load_wsd_instances(xml, keys)[:args.limit]
    resolved: dict[str, str | None] = {}

    def synset_of(key: str) -> str | None:
        if key not in resolved:
            try:
                resolved[key] = wordnet.lemma_from_key(key).synset().name()
            except Exception:
                resolved[key] = None
        return resolved[key]

    started = time.monotonic()
    names = list(run.ontology["concept_names"])
    frames = concept_frames(run.ontology, run.table, wordnet=wordnet, max_degree=args.max_degree)
    items = link_instances(run, instances, synset_of, {n: i for i, n in enumerate(names)})
    attention_masses(run, items, frames)
    score_items(items, wordnet_rank(wordnet), names, set(run.adapter.heldout_entries), run.adapter.train_frequency)
    summary = summarize(items, resamples=args.resamples, seed=args.seed)
    header = {"source": run.describe(), "dataset": args.dataset, "instances": len(instances), "linked": len(items)}
    columns = ("id", "entry", "gold", "predicted", "wn1", "semcor_mfs", "predicted_correct", "wn1_correct",
               "semcor_mfs_correct", "gold_share", "uniform_share", "tie", "status", "lemma", "pos", "fold")
    with (args.output / "items.jsonl").open("w") as handle:
        for it in items:
            handle.write(json.dumps({c: it[c] for c in columns}) + "\n")
    write_json(args.output / "summary.json", {**header, "summary": summary, "seconds": time.monotonic() - started})
    (args.output / "report.md").write_text(render(summary, header))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return {**header, "summary": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--dataset", default="semcor", help="semcor or an evaluation dataset name (ALL, semeval2007, …)")
    parser.add_argument("--limit", type=int, default=None, help="first N tagged instances (smoke tests)")
    parser.add_argument("--max-degree", type=int, default=16, help="max_degree of the WordNet ontology (C3: 16)")
    parser.add_argument("--probes-root", type=Path, default=Path("~/data/vsa-llm/probes").expanduser())
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    result = run_experiment(args)
    print(json.dumps(result["summary"]["all"], indent=2, default=str))


if __name__ == "__main__":
    main()
