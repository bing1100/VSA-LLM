"""Experiment E0: synthetic identifiability of M1 (context), M2 (factored mapping) and M3 (splits).

D0.1 — contextual teachers of increasing context strength plus a static teacher; learners M0,
relation salience, M1 without context (P0), M1 with context, and a free per-concept query
(cannot transfer). Held-out concepts are composition-disjoint; held-out contexts are unseen
prototypes.
D0.3 — rank-`k*` bilinear teacher; M2 free / induced / hybrid factors over a rank sweep.
D0.2 — planted polysemous atomics or relation sub-types under collapsed labels; growth policies
none, random (matched count), coherence-only, M3 (permutation null), M3 (Anderson–Darling),
oracle (true labels).

See `resources/plan-improvement/experiments.md` §E0 for the gates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.developmental import DevelopmentalConfig, DevelopmentalDictionary, top_between_usage_direction
from vsa_embed.factorization import geometry_metrics
from vsa_embed.provenance import apply_thread_setting, prepare_output_dir, write_run_metadata
from vsa_embed.statistics import mean_confidence_interval
from vsa_embed.synthetic import make_frame_teacher, make_polysemy_teacher


# -- shared helpers -------------------------------------------------------------------------------

def adjusted_rand_index(labels_true: list[int], labels_pred: list[int]) -> float:
    """Adjusted Rand index of two flat clusterings."""
    from math import comb
    if len(labels_true) != len(labels_pred) or not labels_true:
        raise ValueError("labelings must be equal-length and non-empty")
    table: dict[tuple[int, int], int] = {}
    for t, p in zip(labels_true, labels_pred):
        table[(t, p)] = table.get((t, p), 0) + 1
    rows: dict[int, int] = {}; cols: dict[int, int] = {}
    for (t, p), n in table.items():
        rows[t] = rows.get(t, 0) + n; cols[p] = cols.get(p, 0) + n
    index = sum(comb(n, 2) for n in table.values())
    row_sum = sum(comb(n, 2) for n in rows.values()); col_sum = sum(comb(n, 2) for n in cols.values())
    expected = row_sum * col_sum / comb(len(labels_true), 2) if len(labels_true) > 1 else 0.0
    maximum = (row_sum + col_sum) / 2
    return 1.0 if maximum == expected else (index - expected) / (maximum - expected)


def _cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()


def _fit(composer: FrameComposer, concepts: torch.Tensor, targets: torch.Tensor, contexts: torch.Tensor | None,
         *, steps: int, lr: float, batch: int, generator: torch.Generator) -> None:
    optimizer = torch.optim.Adam([p for p in composer.parameters() if p.requires_grad], lr=lr)
    n = concepts.numel()
    for _ in range(steps):
        index = torch.randint(n, (min(batch, n),), generator=generator)
        optimizer.zero_grad()
        prediction = composer(concepts[index], None if contexts is None else contexts[index])
        loss = _cosine_loss(prediction, targets[index]) + 1e-3 * composer.delta_penalty()
        loss.backward(); optimizer.step()


@torch.no_grad()
def _evaluate(composer: FrameComposer, samples: dict[str, torch.Tensor], use_context: bool) -> dict[str, float]:
    contexts = samples["contexts"] if use_context else None
    rows, weights, _ = composer.compose(samples["concepts"], contexts, return_weights=True)
    prediction = composer.projector(rows) if composer.projector is not None else rows
    metrics = {
        "cosine": float(F.cosine_similarity(prediction, samples["targets"], dim=-1).mean()),
        "knn_overlap": geometry_metrics(prediction, samples["targets"])["knn_overlap"],
    }
    teacher = samples["weights"]
    if weights.numel() == teacher.numel() and float(weights.std()) > 1e-6 and float(teacher.std()) > 1e-6:
        metrics["weight_correlation"] = float(torch.corrcoef(torch.stack([weights, teacher]))[0, 1])
    else:
        metrics["weight_correlation"] = float("nan")
    return metrics


# -- D0.1 and D0.3 --------------------------------------------------------------------------------

D01_LEARNERS = {
    "m0": {"mode": "bundle"},
    "salience": {"mode": "salience"},
    "m1_p0": {"mode": "attentive"},
    "m1_q": {"mode": "attentive", "use_context": True},
    "free_query_q": {"mode": "attentive", "concept_factor": "free", "use_context": True},
}


def run_contextual(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """D0.1: contextual vs static teachers."""
    section, rows = config["d01"], []
    teachers = [("static", 0.0)] + [("contextual", float(s)) for s in section["context_strengths"]]
    for weighting, strength in teachers:
        data = make_frame_teacher(**section["teacher"], weighting=weighting, context_strength=strength or 1.0, seed=seed)
        dim, atoms, rels = data.atomics.shape[1], data.atomics.shape[0], data.roles.shape[0]
        ctx_dim = data.samples["train"]["contexts"].shape[1]
        for name, spec in D01_LEARNERS.items():
            use_context = bool(spec.get("use_context")) and weighting == "contextual"
            torch.manual_seed(seed + 100)
            composer = FrameComposer(
                data.schedule, atoms, rels, dim, operator=section["operator"], mode=spec["mode"],
                concept_factor=spec.get("concept_factor", "induced"),
                key_dimension=int(section["key_dimension"]), context_dimension=ctx_dim if use_context else 0,
            )
            train = data.samples["train"]
            _fit(composer, train["concepts"], train["targets"], train["contexts"] if use_context else None,
                 steps=int(section["steps"]), lr=float(section["learning_rate"]), batch=int(section["batch"]),
                 generator=torch.Generator().manual_seed(seed + 7))
            for split, samples in data.samples.items():
                rows.append({"experiment": "D0.1", "seed": seed, "teacher": weighting, "strength": strength,
                             "learner": name, "split": split, **_evaluate(composer, samples, use_context)})
    return rows


def run_factored(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """D0.3: rank-k* bilinear teacher; free / induced / hybrid concept factors over ranks."""
    section, rows = config["d03"], []
    data = make_frame_teacher(**section["teacher"], weighting="bilinear", seed=seed)
    dim, atoms, rels = data.atomics.shape[1], data.atomics.shape[0], data.roles.shape[0]
    learners = [("m0", "bundle", "induced", 0)] + [
        (f"{factor}_k{rank}", "attentive", factor, rank)
        for factor in ("free", "induced", "hybrid") for rank in section["ranks"]
    ]
    for name, mode, factor, rank in learners:
        torch.manual_seed(seed + 100)
        composer = FrameComposer(data.schedule, atoms, rels, dim, operator=section["operator"], mode=mode,
                                 concept_factor=factor, key_dimension=max(rank, 1))
        train = data.samples["train"]
        _fit(composer, train["concepts"], train["targets"], None, steps=int(section["steps"]),
             lr=float(section["learning_rate"]), batch=int(section["batch"]),
             generator=torch.Generator().manual_seed(seed + 7))
        for split in ("train", "test"):
            rows.append({"experiment": "D0.3", "seed": seed, "learner": name, "factor": factor, "rank": rank,
                         "split": split, **_evaluate(composer, data.samples[split], False)})
    return rows


# -- D0.2 -----------------------------------------------------------------------------------------

def _grow_training(
    composer: FrameComposer, concepts: torch.Tensor, targets: torch.Tensor, policy: str,
    dev_config: DevelopmentalConfig, schedule_config: dict[str, Any], generator: torch.Generator,
    random_split_count: int = 0,
) -> DevelopmentalDictionary | None:
    """Warm-up → growth (screen and split) → consolidation → final phase, minibatch Adam."""
    optimizer = torch.optim.Adam(composer.parameters(), lr=float(schedule_config["learning_rate"]))
    tracker = DevelopmentalDictionary(composer, optimizer, dev_config) if policy not in {"none", "oracle"} else None
    batch, n = int(schedule_config["batch"]), concepts.numel()
    warmup, growth, final = (int(schedule_config[k]) for k in ("warmup_steps", "growth_steps", "final_steps"))
    random_done = 0
    for step in range(warmup + growth + final):
        in_growth = warmup <= step < warmup + growth
        index = torch.randint(n, (min(batch, n),), generator=generator)
        if tracker is not None and in_growth:
            tracker.begin()
        optimizer.zero_grad()
        loss = _cosine_loss(composer(concepts[index]), targets[index])
        loss.backward()
        if tracker is not None and in_growth:
            tracker.observe()
        optimizer.step()
        if tracker is None or not in_growth:
            continue
        if policy in {"m3", "m3_anderson"}:
            tracker.grow()
        elif policy == "coherence_only" and tracker.step % dev_config.screen_every == 0:
            _split_candidates_untested(tracker)
            tracker._screen()
        elif policy == "random" and tracker.step % dev_config.screen_every == 0:
            quota = random_split_count * tracker.step // growth - random_done
            for _ in range(max(0, quota)):
                _random_split(tracker, generator)
                random_done += 1
        if step == warmup + growth - 1 and policy != "random":
            tracker.consolidate()
    return tracker


def _split_candidates_untested(tracker: DevelopmentalDictionary) -> None:
    """Coherence-only policy: split every screened candidate with enough usages, no test."""
    for vector_id, entries in list(tracker.records.items())[:max(0, tracker.budget_left())]:
        usages = torch.cat([u for u, _ in entries]); grads = torch.cat([g for _, g in entries])
        labels, inverse = torch.unique(usages, return_inverse=True)
        if labels.numel() < 2:
            continue
        momenta = grads.new_zeros(labels.numel(), grads.shape[1]).index_add(0, inverse, grads)
        tracker._split(vector_id, {"labels": labels, "momenta": momenta,
                                   "direction": top_between_usage_direction(momenta),
                                   "gain": 0.0, "p_value": 1.0, "contributions": int(usages.numel())})
    tracker.records.clear()


def _random_split(tracker: DevelopmentalDictionary, generator: torch.Generator) -> None:
    """Random policy: split a random vector along a random direction with a random usage partition."""
    schedule = tracker.composer.schedule
    ids = schedule.fillers if tracker.config.target == "atomics" else schedule.relations
    vector_id = int(ids[torch.randint(ids.numel(), (), generator=generator)])
    if tracker.config.target == "atomics":
        usage_of_edge = torch.repeat_interleave(torch.arange(schedule.concept_count), schedule.degrees)
        labels = usage_of_edge[ids == vector_id].unique()
    else:
        labels = (ids == vector_id).nonzero().flatten()
    if labels.numel() < 2:
        return
    dim = tracker.momentum.shape[1]
    direction = F.normalize(torch.randn(dim, generator=generator), dim=0)
    signs = torch.where(torch.rand(labels.numel(), generator=generator) < 0.5, 1.0, -1.0)
    tracker._split(vector_id, {"labels": labels, "momenta": signs[:, None] * direction,
                               "direction": direction, "gain": 0.0, "p_value": 1.0, "contributions": 0})


def _split_quality(tracker: DevelopmentalDictionary | None, planted: torch.Tensor, total: int,
                   original: FrameSchedule, final: FrameSchedule, edge_sense: torch.Tensor,
                   target: str, train_concepts: torch.Tensor) -> dict[str, float]:
    """Split decisions at convergence (after consolidation merges) and raw, plus partition ARI.

    ARI is computed separately over training usages (partitioned by gradient sign) and over
    held-out usages (routed by frame similarity, never observed).
    """
    cards = tracker.cards if tracker else []
    raw_parents = {card["parent"] for card in cards if card["event"] == "split"}
    retired = {card["retired"] for card in cards if card["event"] == "merge"}
    final_ids = final.fillers if target == "atomics" else final.relations
    original_ids = original.fillers if target == "atomics" else original.relations
    # A parent is split at convergence if its original edges still point at more than one id.
    parents = {v for v in raw_parents
               if (final_ids[original_ids == v]).unique().numel() > 1 and v not in retired}
    planted_set = set(planted.tolist())
    def rates(split_set: set[int]) -> tuple[float, float, float]:
        true_pos = len(split_set & planted_set)
        return (true_pos / len(split_set) if split_set else float("nan"),
                true_pos / len(planted_set) if planted_set else float("nan"),
                len(split_set - planted_set) / max(1, total - len(planted_set)))
    precision, recall, false_rate = rates(parents)
    raw_precision, _, raw_false_rate = rates(raw_parents)
    edge_concept = torch.repeat_interleave(torch.arange(original.concept_count), original.degrees)
    is_train = torch.isin(edge_concept, train_concepts)
    aris: dict[str, list[float]] = {"train": [], "heldout": []}
    for vector_id in sorted(parents & planted_set):
        for name, mask in (("train", is_train), ("heldout", ~is_train)):
            edges = ((original_ids == vector_id) & mask).nonzero().flatten()
            if edges.numel() > 1:
                aris[name].append(adjusted_rand_index(edge_sense[edges].tolist(), final_ids[edges].tolist()))
    mean = lambda values: sum(values) / len(values) if values else float("nan")
    return {"splits": len(parents), "raw_splits": len(raw_parents), "precision": precision, "recall": recall,
            "false_split_rate": false_rate, "raw_precision": raw_precision, "raw_false_split_rate": raw_false_rate,
            "ari": mean(aris["train"]), "ari_heldout": mean(aris["heldout"])}


def run_splits(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """D0.2: planted polysemy (atomics) and planted relation sub-types."""
    section, rows = config["d02"], []
    for condition in section["conditions"]:
        target = condition["target"]
        data = make_polysemy_teacher(**section["teacher"], **condition["teacher"], seed=seed)
        dim = data.targets.shape[1]
        atoms = int(data.schedule.fillers.max()) + 1
        rels = int(data.schedule.relations.max()) + 1
        planted = data.polysemous_atomics if target == "atomics" else data.polysemous_relations
        total = atoms if target == "atomics" else rels
        edge_sense = data.edge_atomic_sense if target == "atomics" else data.edge_relation_sense
        m3_splits = 0
        for policy in section["policies"]:
            schedule = data.schedule
            if policy == "oracle":
                schedule = _oracle_schedule(data, target)
            torch.manual_seed(seed + 100)
            composer = FrameComposer(schedule, int(schedule.fillers.max()) + 1, int(schedule.relations.max()) + 1,
                                     dim, operator=section["operator"])
            dev = DevelopmentalConfig(**section["developmental"], target=target, seed=seed,
                                      test="anderson" if policy == "m3_anderson" else "permutation")
            generator = torch.Generator().manual_seed(seed + 7)
            train = data.train_concepts
            tracker = _grow_training(composer, train, data.targets[train], policy, dev, section["schedule"],
                                     generator, random_split_count=m3_splits)
            if policy == "m3":
                m3_splits = sum(card["event"] == "split" for card in tracker.cards)
            with torch.no_grad():
                test_cos = float(F.cosine_similarity(composer(data.test_concepts), data.targets[data.test_concepts]).mean())
                train_cos = float(F.cosine_similarity(composer(train), data.targets[train]).mean())
            quality = _split_quality(tracker, planted, total, data.schedule, composer.schedule, edge_sense, target,
                                     data.train_concepts) \
                if policy != "oracle" else {"splits": 0, "raw_splits": 0, "precision": float("nan"),
                                            "recall": float("nan"), "false_split_rate": float("nan"),
                                            "raw_precision": float("nan"), "raw_false_split_rate": float("nan"),
                                            "ari": 1.0, "ari_heldout": 1.0}
            rows.append({"experiment": "D0.2", "seed": seed, "target": target, "policy": policy,
                         "train_cosine": train_cos, "test_cosine": test_cos,
                         "parameters": sum(p.numel() for p in composer.parameters() if p.requires_grad),
                         **quality})
    return rows


def _oracle_schedule(data, target: str) -> FrameSchedule:
    """True labels: polysemous ids' second sense gets a fresh id."""
    schedule = data.schedule
    relations, fillers = schedule.relations.clone(), schedule.fillers.clone()
    if target == "atomics":
        base = int(fillers.max()) + 1
        second = data.edge_atomic_sense == 1
        remap = {int(a): base + i for i, a in enumerate(data.polysemous_atomics.tolist())}
        fillers[second] = torch.tensor([remap[int(a)] for a in fillers[second].tolist()], dtype=torch.long)
    else:
        base = int(relations.max()) + 1
        second = data.edge_relation_sense == 1
        remap = {int(r): base + i for i, r in enumerate(data.polysemous_relations.tolist())}
        relations[second] = torch.tensor([remap[int(r)] for r in relations[second].tolist()], dtype=torch.long)
    return FrameSchedule(schedule.offsets, relations, fillers)


# -- summary ------------------------------------------------------------------------------------

def _values(rows: list[dict[str, Any]], **match: Any) -> list[float]:
    return [row for row in rows if all(row.get(k) == v for k, v in match.items())]


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    seeds = config["seeds"]
    summary: dict[str, Any] = {"d01": {}, "d03": {}, "d02": {}}
    # D0.1: M1 with context vs M0 on held-out concepts × held-out contexts.
    d01_pass = True
    for strength in [0.0] + [float(s) for s in config["d01"]["context_strengths"]]:
        teacher = "static" if strength == 0.0 else "contextual"
        split = "test"
        gains = []
        for seed in seeds:
            m1 = _values(rows, experiment="D0.1", seed=seed, teacher=teacher, strength=strength, learner="m1_q", split=split)
            m0 = _values(rows, experiment="D0.1", seed=seed, teacher=teacher, strength=strength, learner="m0", split=split)
            gains.append(m1[0]["cosine"] - m0[0]["cosine"])
        ci = mean_confidence_interval(gains)
        if teacher == "static":
            ok = ci["mean"] >= -float(config["d01"]["no_harm_margin"])
        elif strength >= float(config["d01"]["gate_min_strength"]):
            ok = ci["ci_low"] is not None and ci["ci_low"] > 0
        else:
            ok = True
        d01_pass &= ok
        summary["d01"][f"{teacher}_{strength}"] = {"m1_q_minus_m0_test_cosine": ci, "passes": ok}
    summary["d01"]["gate_passed"] = d01_pass
    # D0.3: induced/hybrid transfer, free does not.
    ranks = config["d03"]["ranks"]
    d03 = {}
    for factor in ("free", "induced", "hybrid"):
        for rank in ranks:
            values = [r["cosine"] for r in _values(rows, experiment="D0.3", factor=factor, rank=rank, split="test")]
            d03[f"{factor}_k{rank}"] = sum(values) / len(values)
    m0 = [r["cosine"] for r in _values(rows, experiment="D0.3", learner="m0", split="test")]
    d03["m0"] = sum(m0) / len(m0)
    best_free = max(d03[f"free_k{r}"] for r in ranks)
    best_induced = max(max(d03[f"induced_k{r}"], d03[f"hybrid_k{r}"]) for r in ranks)
    margin = float(config["d03"]["transfer_margin"])
    summary["d03"] = {"test_cosine": d03, "best_free": best_free, "best_induced_or_hybrid": best_induced,
                      "gate_passed": best_induced >= d03["m0"] and best_induced - best_free >= margin}
    # D0.2: M3 precision/recall/ARI/false splits per target.
    d02_pass = True
    for condition in config["d02"]["conditions"]:
        target = condition["target"]
        per_policy = {}
        for policy in config["d02"]["policies"]:
            selected = _values(rows, experiment="D0.2", target=target, policy=policy)
            per_policy[policy] = {key: sum(r[key] for r in selected) / len(selected)
                                  for key in ("precision", "recall", "false_split_rate", "ari", "ari_heldout",
                                              "raw_false_split_rate", "test_cosine", "splits", "raw_splits")}
        m3 = per_policy["m3"]
        gate = config["d02"]["gate"]
        ok = (m3["precision"] >= gate["precision"] and m3["recall"] >= gate["recall"]
              and m3["ari"] >= gate["ari"] and m3["false_split_rate"] <= gate["false_split_rate"])
        d02_pass &= bool(ok)
        summary["d02"][target] = {"policies": per_policy, "passes": bool(ok)}
    summary["d02"]["gate_passed"] = d02_pass
    summary["g1_all_gates_passed"] = d01_pass and summary["d03"]["gate_passed"] and d02_pass
    return summary


def render_report(summary: dict[str, Any]) -> str:
    def ci(value: dict[str, Any]) -> str:
        return f"{value['mean']:+.4f}" + ("" if value["ci_low"] is None else f" [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]")
    lines = ["# Experiment E0 — synthetic identifiability", "", "## D0.1 contextual composition", "",
             "| Teacher | M1(q) − M0, held-out concepts × held-out contexts (cosine, 95% CI) | Passes |", "|---|---|---|"]
    for key, value in summary["d01"].items():
        if key != "gate_passed":
            lines.append(f"| {key} | {ci(value['m1_q_minus_m0_test_cosine'])} | {value['passes']} |")
    lines += ["", f"D0.1 gate: **{'PASS' if summary['d01']['gate_passed'] else 'FAIL'}**.", "",
              "## D0.3 factored mapping (held-out concept cosine)", "", "| Learner | Cosine |", "|---|---:|",
              *[f"| {k} | {v:.4f} |" for k, v in summary["d03"]["test_cosine"].items()], "",
              f"D0.3 gate: **{'PASS' if summary['d03']['gate_passed'] else 'FAIL'}**.", "", "## D0.2 split detection", ""]
    for target, value in summary["d02"].items():
        if target == "gate_passed":
            continue
        lines += [f"### {target}", "",
                  "| Policy | Splits at convergence (raw) | Precision | Recall | ARI train usages | ARI held-out usages | False-split rate (raw) | Held-out cosine |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for policy, m in value["policies"].items():
            lines.append(f"| {policy} | {m['splits']:.1f} ({m['raw_splits']:.1f}) | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['ari']:.3f} | {m['ari_heldout']:.3f} | {m['false_split_rate']:.3f} ({m['raw_false_split_rate']:.3f}) | "
                         f"{m['test_cosine']:.4f} |")
        lines += ["", f"Passes: **{value['passes']}**.", ""]
    lines += [f"D0.2 gate: **{'PASS' if summary['d02']['gate_passed'] else 'FAIL'}**.", "",
              f"**G1 (all E0 gates): {'PASS' if summary['g1_all_gates_passed'] else 'FAIL'}.**", ""]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(config: dict[str, Any], output_dir: Path, *, parts: tuple[str, ...] = ("d01", "d03", "d02")) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir)
    apply_thread_setting(config)
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        if "d01" in parts: rows += run_contextual(config, int(seed))
        if "d03" in parts: rows += run_factored(config, int(seed))
        if "d02" in parts: rows += run_splits(config, int(seed))
    _write_csv(output_dir / "metrics.csv", rows)
    summary = summarize(rows, config) if parts == ("d01", "d03", "d02") else {}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu")
    if summary:
        (output_dir / "report.md").write_text(render_report(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(yaml.safe_load(args.config.read_text()), args.output)
    print(json.dumps({k: summary.get(k) for k in ("g1_all_gates_passed",)}, indent=2))


if __name__ == "__main__":
    main()
