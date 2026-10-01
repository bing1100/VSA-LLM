"""Experiment E2 (D2): mapping × operator frontier for composing frozen-host concept anchors.

Concepts are composed from their ontology frames with a `FrameComposer` and projected to the host
width; the target is the concept's centred frozen-host anchor (see `frame_anchors.py`). Splits are
node-disjoint (test 20%, validation 10%). Two ontology families: WordNet (single-token aligned
synsets) and MeSH (descriptors with scope notes).

Grid: mapping ∈ {binary (static bundle), salience, M2 induced k ∈ {4, 8, 16, 32}, M2 hybrid k = 8}
× operator ∈ {hrr, hrr_identity, diagonal, low_rank_tied (rank 1, parameter-matched to hrr),
orthogonal, random_fixed:hrr, untyped}; three training budgets for data-to-threshold curves;
a derangement-shuffled relation-label control per operator for the primary mapping.
(The contextual mapping is E1; `bounded_residual` is not a composer family and is not run.)

Selection for E4 is made on the **validation** split only; test numbers are reported alongside.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from nltk.corpus import wordnet as wn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.experiments.frame_anchors import (
    FrameRegressionData, build_dataset, centred_targets, node_disjoint_split, regression_metrics, term_anchors,
)
from vsa_embed.ontologies.mesh import build_mesh_ontology
from vsa_embed.ontologies.wordnet import build_wordnet_ontology
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.real_relations import derange_labels
from vsa_embed.statistics import mean_confidence_interval
from vsa_embed.wordnet_relations import aligned_lemma

MAPPINGS = {
    "binary": {"mode": "bundle"}, "salience": {"mode": "salience"},
    "induced_k4": {"mode": "attentive", "key_dimension": 4}, "induced_k8": {"mode": "attentive", "key_dimension": 8},
    "induced_k16": {"mode": "attentive", "key_dimension": 16}, "induced_k32": {"mode": "attentive", "key_dimension": 32},
    "hybrid_k8": {"mode": "attentive", "key_dimension": 8, "concept_factor": "hybrid"},
}


def wordnet_dataset(config: dict[str, Any], host, tokenizer, device) -> FrameRegressionData:
    onto = build_wordnet_ontology(wn, max_atomics=int(config["max_atomics"]))
    candidates = []
    for cid, name in enumerate(onto.concept_names):
        pointer_edges = sum(1 for r, _ in onto.frames[cid] if onto.relation_names[r] not in {"lexname", "pos"})
        if pointer_edges >= 1:
            candidates.append(cid)
    rng = np.random.default_rng(int(config["selection_seed"]))
    rng.shuffle(candidates)
    chosen, glosses, terms = [], [], []
    for cid in candidates:
        synset = wn.synset(onto.concept_names[cid])
        match = aligned_lemma(synset, tokenizer)
        if match is None:
            continue
        chosen.append(cid); glosses.append(synset.definition()); terms.append(match[0])
        if len(chosen) >= int(config["max_concepts"]):
            break
    anchors = term_anchors(host, tokenizer, glosses, terms, device=device)
    return build_dataset(onto, chosen, anchors)


def mesh_dataset(config: dict[str, Any], host, tokenizer, device) -> FrameRegressionData:
    onto = build_mesh_ontology(Path(config["mesh_path"]).expanduser(), max_atomics=int(config["max_atomics"]))
    notes, heads = onto.metadata["scope_notes"], onto.metadata["headings"]
    candidates = [c for c in range(len(onto.concept_names)) if notes[c] and len(onto.frames[c]) >= 3]
    rng = np.random.default_rng(int(config["selection_seed"]))
    rng.shuffle(candidates)
    chosen = sorted(candidates[:int(config["max_concepts"])])
    anchors = term_anchors(host, tokenizer, [notes[c] for c in chosen], [heads[c] for c in chosen], device=device)
    return build_dataset(onto, chosen, anchors)


def deranged_schedule(schedule: FrameSchedule, seed: int) -> FrameSchedule:
    return FrameSchedule(schedule.offsets, derange_labels(schedule.relations, torch.Generator().manual_seed(seed + 991)),
                         schedule.fillers)


def fit_and_eval(data: FrameRegressionData, schedule: FrameSchedule, mapping: str, operator: str, split, budget: float,
                 seed: int, config: dict[str, Any], device) -> dict[str, Any]:
    targets = centred_targets(data.anchors, split["train"]).to(device)
    rng = np.random.default_rng(seed + 17)
    train = np.sort(rng.permutation(split["train"])[:max(1, int(len(split["train"]) * budget))])
    spec = dict(MAPPINGS[mapping])
    torch.manual_seed(seed + 100)
    composer = FrameComposer(schedule, data.atomic_count, data.relation_count, int(config["dimension"]),
                             operator=operator, rank=1, output_dimension=targets.shape[1], **spec).to(device)
    optimizer = torch.optim.Adam([p for p in composer.parameters() if p.requires_grad], lr=float(config["learning_rate"]))
    g = torch.Generator().manual_seed(seed)
    train_t = torch.from_numpy(train)
    for _ in range(int(config["steps"])):
        batch = train_t[torch.randint(len(train_t), (int(config["batch"]),), generator=g)].to(device)
        prediction = composer(batch)
        loss = F.mse_loss(prediction, targets[batch]) + (1 - F.cosine_similarity(prediction, targets[batch])).mean()
        loss = loss + 1e-3 * composer.delta_penalty()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    result = {"parameters": sum(composer.parameter_groups().values()), **{f"params_{k}": v for k, v in composer.parameter_groups().items()}}
    with torch.no_grad():
        for name in ("validation", "test"):
            index = torch.from_numpy(split[name]).to(device)
            metrics = regression_metrics(composer(index).cpu(), targets[index].cpu())
            result.update({f"{name}_{k}": v for k, v in metrics.items()})
    return result


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in {r["family"] for r in rows}:
        full = [r for r in rows if r["family"] == family and r["budget"] == 1.0 and not r["shuffled"]]
        table = defaultdict(lambda: defaultdict(list))
        for r in full:
            for metric in ("validation_mrr", "test_mrr", "test_cosine", "test_variance_explained", "parameters"):
                table[(r["mapping"], r["operator"])][metric].append(r[metric])
        means = {f"{m}|{o}": {k: float(np.mean(v)) for k, v in d.items()} for (m, o), d in table.items()}
        best_key = max(means, key=lambda k: means[k]["validation_mrr"])
        best_mapping, best_operator = best_key.split("|")
        induced = [k for k in means if k.startswith("induced_") and k.endswith(f"|{best_operator}")]
        best_induced = max(induced, key=lambda k: means[k]["validation_mrr"]) if induced else None
        def paired(a: str, b: str, metric: str = "test_mrr"):
            values = []
            for seed in config["seeds"]:
                ra = [r for r in full if f"{r['mapping']}|{r['operator']}" == a and r["seed"] == seed]
                rb = [r for r in full if f"{r['mapping']}|{r['operator']}" == b and r["seed"] == seed]
                if ra and rb:
                    values.append(ra[0][metric] - rb[0][metric])
            return mean_confidence_interval(values) if values else None
        shuffled = {}
        for operator in config["operators"]:
            pairs = []
            for seed in config["seeds"]:
                true = [r for r in full if r["mapping"] == config["primary_mapping"] and r["operator"] == operator and r["seed"] == seed]
                shuf = [r for r in rows if r["family"] == family and r["shuffled"] and r["operator"] == operator and r["seed"] == seed]
                if true and shuf:
                    pairs.append(true[0]["test_mrr"] - shuf[0]["test_mrr"])
            shuffled[operator] = mean_confidence_interval(pairs) if pairs else None
        curves = defaultdict(dict)
        for r in rows:
            if r["family"] == family and not r["shuffled"] and r["mapping"] in (config["primary_mapping"], "binary"):
                curves[f"{r['mapping']}|{r['operator']}"].setdefault(r["budget"], []).append(r["test_mrr"])
        summary[family] = {
            "means": means, "selected_by_validation": {"mapping": best_mapping, "operator": best_operator},
            "best_induced_for_selected_operator": best_induced,
            "induced_vs_salience": paired(best_induced, f"salience|{best_operator}") if best_induced else None,
            "induced_vs_binary": paired(best_induced, f"binary|{best_operator}") if best_induced else None,
            "hybrid_vs_induced_k8": paired(f"hybrid_k8|{best_operator}", f"induced_k8|{best_operator}"),
            "selected_vs_random_fixed": paired(best_key, f"{best_mapping}|random_fixed:hrr"),
            "selected_vs_untyped": paired(best_key, f"{best_mapping}|untyped"),
            "true_minus_shuffled_labels": shuffled,
            "budget_curves": {k: {b: float(np.mean(v)) for b, v in d.items()} for k, d in curves.items()},
        }
        gate_induced = summary[family]["induced_vs_salience"]
        summary[family]["gate_induced_ge_salience"] = bool(gate_induced and gate_induced["mean"] >= 0)
        gate_binary = summary[family]["induced_vs_binary"]
        summary[family]["gate_induced_beats_binary"] = bool(gate_binary and gate_binary["ci_low"] is not None and gate_binary["ci_low"] > 0)
    return summary


def render(summary: dict[str, Any]) -> str:
    ci = lambda v: "n/a" if not v or v.get("ci_low") is None else f"{v['mean']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
    lines = ["# E2 — mapping × operator frontier (frozen-host concept anchors)", ""]
    for family, s in summary.items():
        lines += [f"## {family}", "", "| Mapping | Operator | Val MRR | Test MRR | Test cosine | Test VE | Params |", "|---|---|---:|---:|---:|---:|---:|"]
        for key in sorted(s["means"], key=lambda k: -s["means"][k]["validation_mrr"]):
            m = s["means"][key]; mapping, operator = key.split("|")
            lines.append(f"| {mapping} | {operator} | {m['validation_mrr']:.4f} | {m['test_mrr']:.4f} | {m['test_cosine']:.4f} | "
                         f"{m['test_variance_explained']:.4f} | {m['parameters']:.0f} |")
        sel = s["selected_by_validation"]
        lines += ["", f"Selected on validation: **{sel['mapping']} / {sel['operator']}**; best induced rank for that operator: {s['best_induced_for_selected_operator']}.",
                  f"Induced − salience (test MRR): {ci(s['induced_vs_salience'])}; induced − binary: {ci(s['induced_vs_binary'])}; "
                  f"hybrid − induced k8: {ci(s['hybrid_vs_induced_k8'])}.",
                  f"Selected − random_fixed operator: {ci(s['selected_vs_random_fixed'])}; selected − untyped: {ci(s['selected_vs_untyped'])}.",
                  "True − shuffled relation labels (primary mapping): " + "; ".join(f"{k}: {ci(v)}" for k, v in s["true_minus_shuffled_labels"].items()) + ".",
                  f"Gates: induced ≥ salience **{s['gate_induced_ge_salience']}**; induced > binary **{s['gate_induced_beats_binary']}**.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    git_at_start = prepare_output_dir(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["host"], local_files_only=True)
    host = AutoModelForCausalLM.from_pretrained(config["host"], local_files_only=True)
    datasets = {}
    if "wordnet" in config["families"]:
        datasets["wordnet"] = wordnet_dataset(config, host, tokenizer, device)
    if "mesh" in config["families"]:
        datasets["mesh"] = mesh_dataset(config, host, tokenizer, device)
    del host; torch.cuda.empty_cache()
    rows = []
    metrics_path = args.output / "metrics.jsonl"
    for family, data in datasets.items():
        for seed in config["seeds"]:
            split = node_disjoint_split(len(data.concept_ids), seed=int(seed))
            jobs = [(m, o, b, False) for m in MAPPINGS for o in config["operators"] for b in config["budgets"]]
            jobs += [(config["primary_mapping"], o, 1.0, True) for o in config["operators"]]
            for mapping, operator, budget, shuffled in jobs:
                schedule = deranged_schedule(data.schedule, int(seed)) if shuffled else data.schedule
                result = fit_and_eval(data, schedule, mapping, operator, split, float(budget), int(seed), config, device)
                row = {"family": family, "seed": seed, "mapping": mapping, "operator": operator, "budget": float(budget),
                       "shuffled": shuffled, "concepts": len(data.concept_ids), **result}
                rows.append(row)
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
    summary = summarize(rows, config)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output / "report.md").write_text(render(summary))
    write_run_metadata(args.output, config, git_at_start=git_at_start, device=device,
                       concepts={k: len(v.concept_ids) for k, v in datasets.items()})


if __name__ == "__main__":
    main()
