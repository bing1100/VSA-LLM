"""Experiment E3 (D3): developmental recovery of collapsed WordNet senses and relation sub-types.

The WordNet frame ontology is collapsed before training, and growth policies must recover what was
removed, while composing frozen GPT-2 concept anchors (same targets and node-disjoint splits as E2):

- *sense collapse*: every synset filler is replaced by an atom for its first lemma name, so e.g.
  `bank.n.01` and `bank.n.09` become one `lemma:bank` atom (a polysemous atom);
- *relation collapse*: {part, member, substance} meronym → meronym, the three holonyms → holonym,
  {hypernym, instance_hypernym} → hypernym.

Policies: none; uniform enlargement (wider atoms, parameters matched to M3's final count); random
splits (count matched to M3); coherence-only screening (splits without the test); M3 (permutation
null, parent fallback with synced parents, consolidation); and, for relations, a minimal 01d-style
pool of K stem-cell relation experts with edge-dependent sparsemax routing.

Metrics: split precision / recall against the planted collapsed atoms (relations), false-split
rate on monosemous atoms, ARI of the training-usage partition against the original synsets
(relation labels), held-out MRR and its change per added parameter.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from nltk.corpus import wordnet as wn
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsa_embed.algebra import HRRAlgebra
from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.developmental import DevelopmentalConfig, DevelopmentalDictionary, top_between_usage_direction
from vsa_embed.experiments.e0_identifiability import _random_split, _split_candidates_untested, adjusted_rand_index
from vsa_embed.experiments.e2_frontier import wordnet_dataset
from vsa_embed.experiments.frame_anchors import centred_targets, node_disjoint_split, regression_metrics
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.statistics import mean_confidence_interval

RELATION_COLLAPSE = {"part_meronym": "meronym", "member_meronym": "meronym", "substance_meronym": "meronym",
                     "part_holonym": "holonym", "member_holonym": "holonym", "substance_holonym": "holonym",
                     "hypernym": "hypernym", "instance_hypernym": "hypernym"}


def collapse(data, onto, target: str):
    """Return (collapsed schedule, atomic_count, relation_count, edge truth, planted ids, monosemous ids)."""
    schedule = data.schedule
    if target == "atomics":
        names = onto.atomic_names
        key_of = {}
        for i, name in enumerate(names):
            if name.startswith("synset:"):
                lemma = wn.synset(name[len("synset:"):]).lemma_names()[0].lower()
                key_of[i] = f"lemma:{lemma}"
            else:
                key_of[i] = name
        keys = sorted(set(key_of.values()))
        index = {k: j for j, k in enumerate(keys)}
        mapping = torch.tensor([index[key_of[i]] for i in range(len(names))])
        fillers = mapping[schedule.fillers]
        truth = schedule.fillers.clone()
        originals = defaultdict(set)
        for orig, new in zip(schedule.fillers.tolist(), fillers.tolist()):
            originals[new].add(orig)
        new_schedule = FrameSchedule(schedule.offsets, schedule.relations, fillers)
        return new_schedule, len(keys), data.relation_count, truth, originals
    names = onto.relation_names
    collapsed_names = sorted({RELATION_COLLAPSE.get(n, n) for n in names})
    index = {n: j for j, n in enumerate(collapsed_names)}
    mapping = torch.tensor([index[RELATION_COLLAPSE.get(n, n)] for n in names])
    relations = mapping[schedule.relations]
    originals = defaultdict(set)
    for orig, new in zip(schedule.relations.tolist(), relations.tolist()):
        originals[new].add(orig)
    return (FrameSchedule(schedule.offsets, relations, schedule.fillers), data.atomic_count, len(collapsed_names),
            schedule.relations.clone(), originals)


class StemCellComposer(FrameComposer):
    """Relations as a fixed pool of K HRR experts with edge-dependent sparsemax routing (01d, minimal)."""

    def __init__(self, *args, experts: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.atomics.shape[1]
        self.experts = nn.Parameter(torch.randn(experts, dim) / dim**0.5)
        self.router = nn.Linear(self.relation_count + dim, experts)
        self.algebra = HRRAlgebra()

    def routing(self, edge_index: torch.Tensor) -> torch.Tensor:
        relations = F.one_hot(self.schedule.relations[edge_index], self.relation_count).float()
        atoms = self.atomic_vectors()[self.schedule.fillers[edge_index]]
        return sparsemax(self.router(torch.cat([relations, atoms], -1)))

    def bound_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        atoms = self.atomic_vectors()[self.schedule.fillers[edge_index]]
        weights = self.routing(edge_index)
        bound = self.algebra.bind(self.experts[None], atoms[:, None])     # edges × experts × dim
        return (weights[..., None] * bound).sum(1)


def sparsemax(logits: torch.Tensor) -> torch.Tensor:
    z, _ = torch.sort(logits, dim=-1, descending=True)
    k = torch.arange(1, logits.shape[-1] + 1, device=logits.device, dtype=logits.dtype)
    cumulative = z.cumsum(-1)
    support = (1 + k * z) > cumulative
    k_z = support.sum(-1, keepdim=True)
    tau = (cumulative.gather(-1, k_z - 1) - 1) / k_z
    return torch.clamp(logits - tau, min=0)


def train(composer, targets, train_idx, policy, dev_config, schedule_cfg, seed, random_count=0):
    optimizer = torch.optim.Adam([p for p in composer.parameters() if p.requires_grad], lr=float(schedule_cfg["learning_rate"]))
    tracker = DevelopmentalDictionary(composer, optimizer, dev_config) if policy in {"m3", "coherence_only", "random"} else None
    g = torch.Generator().manual_seed(seed)
    warmup, growth, final = (int(schedule_cfg[k]) for k in ("warmup_steps", "growth_steps", "final_steps"))
    random_done = 0
    idx = torch.from_numpy(train_idx)
    for step in range(warmup + growth + final):
        in_growth = warmup <= step < warmup + growth
        batch = idx[torch.randint(len(idx), (int(schedule_cfg["batch"]),), generator=g)].to(targets.device)
        if tracker is not None and in_growth:
            tracker.begin()
        prediction = composer(batch)
        loss = F.mse_loss(prediction, targets[batch]) + (1 - F.cosine_similarity(prediction, targets[batch])).mean()
        optimizer.zero_grad(); loss.backward()
        if tracker is not None and in_growth:
            tracker.observe()
        optimizer.step()
        if tracker is None:
            continue
        if policy == "m3":
            tracker.grow()
            if step == warmup + growth - 1:
                tracker.consolidate()
        elif in_growth and tracker.step % dev_config.screen_every == 0:
            if policy == "coherence_only":
                _split_candidates_untested(tracker); tracker._screen()
            elif policy == "random":
                quota = random_count * tracker.step // growth - random_done
                for _ in range(max(0, quota)):
                    _random_split(tracker, g); random_done += 1
        if tracker is not None and dev_config.route_unobserved == "parent":
            tracker.sync_parents()
    return tracker


def split_quality(tracker, originals, truth, original_ids, final_ids, train_concepts, schedule, min_usages):
    edge_concept = torch.repeat_interleave(torch.arange(schedule.concept_count), schedule.degrees)
    is_train = torch.isin(edge_concept, torch.from_numpy(train_concepts))
    usage_count = Counter(original_ids[is_train].tolist())
    planted = {a for a, origs in originals.items()
               if sum(1 for o in origs if Counter(truth[(original_ids == a) & is_train].tolist())[o] >= min_usages) >= 2}
    monosemous = {a for a, origs in originals.items() if len(origs) == 1 and usage_count[a] >= min_usages}
    cards = tracker.cards if tracker else []
    retired = {c["retired"] for c in cards if c["event"] == "merge"}
    raw = {c["parent"] for c in cards if c["event"] == "split"}
    split = {v for v in raw if v not in retired and final_ids[(original_ids == v) & is_train].unique().numel() > 1}
    aris = []
    for a in sorted(split & planted):
        edges = ((original_ids == a) & is_train).nonzero().flatten()
        aris.append(adjusted_rand_index(truth[edges].tolist(), final_ids[edges].tolist()))
    return {"planted": len(planted), "monosemous": len(monosemous), "splits": len(split), "raw_splits": len(raw),
            "precision": len(split & planted) / len(split) if split else float("nan"),
            "recall": len(split & planted) / len(planted) if planted else float("nan"),
            "false_split_rate": len(split & monosemous) / max(1, len(monosemous)),
            "ari": float(np.mean(aris)) if aris else float("nan")}


def run_condition(data, onto, target, policy, seed, config, device, *, random_count=0, width=None):
    schedule, atomic_count, relation_count, truth, originals = collapse(data, onto, target)
    split = node_disjoint_split(len(data.concept_ids), seed=seed)
    targets = centred_targets(data.anchors, split["train"]).to(device)
    torch.manual_seed(seed + 100)
    dim = width or int(config["dimension"])
    cls = StemCellComposer if policy == "stem_cell" else FrameComposer
    composer = cls(schedule, atomic_count, relation_count, dim, operator=config["operator"],
                   output_dimension=targets.shape[1]).to(device)
    params_before = sum(p.numel() for p in composer.parameters() if p.requires_grad)
    dev = DevelopmentalConfig(**config["developmental"], target=target, seed=seed)
    tracker = train(composer, targets, split["train"], policy, dev, config["schedule"], seed, random_count)
    with torch.no_grad():
        index = torch.from_numpy(split["test"]).to(device)
        metrics = regression_metrics(composer(index).cpu(), targets[index].cpu())
    original_ids = schedule.fillers if target == "atomics" else schedule.relations
    final = composer.schedule
    final_ids = (final.fillers if target == "atomics" else final.relations).cpu()
    quality = split_quality(tracker, originals, truth, original_ids, final_ids, split["train"], schedule,
                            int(config["min_usages"]))
    if policy == "stem_cell":
        with torch.no_grad():
            experts = composer.routing(torch.arange(schedule.relations.numel(), device=device)).argmax(-1).cpu()
        aris = [adjusted_rand_index(truth[schedule.relations == r].tolist(), experts[schedule.relations == r].tolist())
                for r, origs in originals.items() if len(origs) > 1]
        quality["ari"] = float(np.mean(aris)) if aris else float("nan")
    params_after = sum(p.numel() for p in composer.parameters() if p.requires_grad)
    return {"target": target, "policy": policy, "seed": seed, "params_before": params_before, "params_after": params_after,
            **{f"test_{k}": v for k, v in metrics.items()}, **quality}, tracker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    git_at_start = prepare_output_dir(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["host"], local_files_only=True)
    host = AutoModelForCausalLM.from_pretrained(config["host"], local_files_only=True)
    data = wordnet_dataset(config, host, tokenizer, device)
    from vsa_embed.ontologies.wordnet import build_wordnet_ontology
    onto = build_wordnet_ontology(wn, max_atomics=int(config["max_atomics"]))
    del host; torch.cuda.empty_cache()
    rows, cards = [], {}
    for target in config["targets"]:
        for seed in config["seeds"]:
            m3_row, tracker = run_condition(data, onto, target, "m3", int(seed), config, device)
            rows.append(m3_row); cards[f"{target}-{seed}"] = tracker.cards
            policies = ["none", "coherence_only", "random", "uniform"] + (["stem_cell"] if target == "relations" else [])
            for policy in policies:
                if policy == "uniform":
                    growth = m3_row["params_after"] - m3_row["params_before"]
                    width = int(round(int(config["dimension"]) * (1 + growth / max(1, m3_row["params_before"]))))
                    row, _ = run_condition(data, onto, target, "none", int(seed), config, device, width=width)
                    row["policy"] = "uniform"
                else:
                    row, _ = run_condition(data, onto, target, policy, int(seed), config, device, random_count=m3_row["splits"])
                rows.append(row)
            with (args.output / "metrics.jsonl").open("a") as handle:
                for row in rows[-(len(policies) + 1):]:
                    handle.write(json.dumps(row) + "\n")
    summary = {}
    for target in config["targets"]:
        table = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["target"] == target:
                for k in ("precision", "recall", "false_split_rate", "ari", "test_mrr", "test_cosine", "splits", "params_after"):
                    table[r["policy"]][k].append(r[k])
        means = {p: {k: float(np.nanmean(v)) if not all(np.isnan(v)) else float("nan") for k, v in d.items()} for p, d in table.items()}
        def paired(a, b, metric):
            values = [next(r[metric] for r in rows if r["target"] == target and r["policy"] == a and r["seed"] == s)
                      - next(r[metric] for r in rows if r["target"] == target and r["policy"] == b and r["seed"] == s) for s in config["seeds"]]
            return mean_confidence_interval(values)
        gate = config["gate"]
        m3 = means["m3"]
        summary[target] = {
            "means": means, "m3_minus_random_ari": paired("m3", "random", "ari") if target else None,
            "m3_minus_uniform_mrr": paired("m3", "uniform", "test_mrr"), "m3_minus_none_mrr": paired("m3", "none", "test_mrr"),
            "gate_passed": bool(m3["ari"] > max(means["random"]["ari"] if not np.isnan(means["random"]["ari"]) else 0, 0)
                                and m3["false_split_rate"] <= gate["false_split_rate"]
                                and paired("m3", "uniform", "test_mrr")["mean"] > 0),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output / "cards.json").write_text(json.dumps(cards, indent=2, default=str) + "\n")
    ci = lambda v: f"{v['mean']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]" if v and v.get("ci_low") is not None else "n/a"
    lines = ["# E3 — developmental recovery on WordNet", ""]
    for target, s in summary.items():
        lines += [f"## {target}", "", "| Policy | Splits | Precision | Recall | ARI | False-split rate | Test MRR | Params |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for policy, m in s["means"].items():
            lines.append(f"| {policy} | {m['splits']:.1f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['ari']:.3f} | "
                         f"{m['false_split_rate']:.3f} | {m['test_mrr']:.4f} | {m['params_after']:.0f} |")
        lines += ["", f"M3 − uniform enlargement (test MRR): {ci(s['m3_minus_uniform_mrr'])}; M3 − none: {ci(s['m3_minus_none_mrr'])}; "
                  f"M3 − random (ARI): {ci(s['m3_minus_random_ari'])}.", f"Gate: **{'PASS' if s['gate_passed'] else 'FAIL'}**.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    write_run_metadata(args.output, config, git_at_start=git_at_start, device=device)


if __name__ == "__main__":
    main()
