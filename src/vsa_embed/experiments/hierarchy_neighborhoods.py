"""Experiment 01c.5: complete source-centric WordNet parent neighborhoods.

Configs without `protocol` replay the recorded runs. `protocol: 2` holds out whole
neighbourhoods including their targets (no test parent is a training endpoint, audit F1),
uses mid-rank ties, rank-loss positives by target set and derangement shuffles, requires a
shuffled control for each compared family, and reports confidence intervals. Every protocol
reports query MRR separately for queries whose parents were / were not training endpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from nltk.corpus import wordnet as wn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsa_embed.experiments.wordnet_relations import _build_model, _require_shuffled, _supported
from vsa_embed.provenance import (
    apply_thread_setting, prepare_output_dir, require_clean_tree_for_promotion, write_run_metadata,
)
from vsa_embed.real_relations import (
    HostEdges, derange_labels, fit_host_relations, source_neighborhood_metrics,
)
from vsa_embed.statistics import mean_confidence_interval
from vsa_embed.wordnet_relations import (
    EdgeSplit, RelationEdge, SenseNode, collect_source_neighborhood_graph, contextual_anchors,
    split_neighborhood_partitions, split_source_neighborhoods,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _tensors(
    edges: list[RelationEdge], nodes: list[SenseNode], relations: list[str], anchors: torch.Tensor,
) -> tuple[HostEdges, torch.Tensor, torch.Tensor]:
    node_ids = {node.synset: index for index, node in enumerate(nodes)}
    relation_ids = {relation: index for index, relation in enumerate(relations)}
    sources = torch.tensor([node_ids[edge.source] for edge in edges])
    targets = torch.tensor([node_ids[edge.target] for edge in edges])
    typed = torch.tensor([relation_ids[edge.relation] for edge in edges])
    return HostEdges(anchors[sources], anchors[targets], typed, sources, targets), sources, targets


def _mean(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    selected = [float(row[metric]) for row in rows if row["method"] == method and metric in row]
    return sum(selected) / len(selected) if selected else float("nan")


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill every default the runner uses so `resolved_config.yaml` records them."""
    resolved = copy.deepcopy(config)
    protocol = int(resolved.get("protocol", 1))
    v2 = protocol >= 2
    resolved["protocol"] = protocol
    data = resolved["data"]
    data.setdefault("validation_fraction", 0.0)
    data.setdefault("target_disjoint", v2)
    resolved["host"].setdefault("anchor_truncation", "gloss" if v2 else "right")
    fit = resolved["fit"]
    for key, value in {
        "max_residual_scale": 0.25, "max_log_basis_scale": 0.25, "correction_weight": 0.0,
        "offset_weight": 0.0, "basis_weight": 0.0, "rank_weight": 0.0, "rank_temperature": 0.07,
    }.items():
        fit.setdefault(key, value)
    fit.setdefault("rank_positives", "target_set" if v2 else "source_relation")
    fit.setdefault("shuffle", "derangement" if v2 else "permutation")
    resolved.setdefault("evaluation", {}).setdefault("ties", "mid" if v2 else "optimistic")
    resolved["acceptance"].setdefault("promotion_eligible", False)
    resolved.setdefault("device", "cuda")
    return resolved


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    v2 = int(config.get("protocol", 1)) >= 2
    rows = [row for row in rows if row.get("partition", "test") == "test"]
    methods = [item if isinstance(item, str) else item["family"] for item in config["methods"]]
    metrics = ("query_mrr", "set_recall_at_10", "set_hit_at_10", "set_mass_nll",
               "target_centroid_cosine", "mean_targets_per_query", "query_mrr_seen", "query_mrr_unseen")
    means = {method: {metric: _mean(rows, method, metric) for metric in metrics} for method in methods}
    acceptance = config["acceptance"]
    candidate, control, shuffled = (
        acceptance["candidate"], acceptance["non_hrr_control"], acceptance["shuffled_control"],
    )
    if v2:
        _require_shuffled(methods, [candidate, control])
    by_seed = {(str(row["seed"]), row["method"]): row for row in rows}
    paired = []
    for seed in config["seeds"]:
        c, d, s = (by_seed[(str(seed), method)] for method in (candidate, control, shuffled))
        paired.append({
            "seed": seed,
            "mrr_gain_over_non_hrr": c["query_mrr"] - d["query_mrr"],
            "mrr_gain_over_shuffled": c["query_mrr"] - s["query_mrr"],
            "set_recall_gain_over_non_hrr": c["set_recall_at_10"] - d["set_recall_at_10"],
        })
    required = int(acceptance.get("min_paired_wins", len(config["seeds"])))
    mrr_gain = means[candidate]["query_mrr"] - means[control]["query_mrr"]
    recall_gain = means[candidate]["set_recall_at_10"] - means[control]["set_recall_at_10"]
    shuffled_gain = means[candidate]["query_mrr"] - means[shuffled]["query_mrr"]
    wins = sum(item["mrr_gain_over_non_hrr"] > 0 for item in paired)
    shuffled_wins = sum(item["mrr_gain_over_shuffled"] > 0 for item in paired)
    passed = (
        mrr_gain >= float(acceptance["min_mrr_gain_over_non_hrr"])
        and recall_gain >= float(acceptance["min_recall_gain_over_non_hrr"])
        and shuffled_gain > 0 and wins >= required and shuffled_wins >= required
    )
    control_ci = mean_confidence_interval([item["mrr_gain_over_non_hrr"] for item in paired])
    shuffled_ci = mean_confidence_interval([item["mrr_gain_over_shuffled"] for item in paired])
    if v2:
        n_seeds = len(config["seeds"])
        passed = passed and _supported(control_ci, wins, n_seeds) and _supported(shuffled_ci, shuffled_wins, n_seeds)
    return {
        "experiment_stage": "01c.5-source-neighborhoods", "protocol": int(config.get("protocol", 1)),
        "mean_metrics": means,
        "mrr_gain_over_non_hrr_ci": control_ci, "mrr_gain_over_shuffled_ci": shuffled_ci,
        "proxy_criteria_only": True,
        "candidate": candidate, "non_hrr_control": control, "shuffled_control": shuffled,
        "mrr_gain_over_non_hrr": mrr_gain, "set_recall_gain_over_non_hrr": recall_gain,
        "mrr_gain_over_shuffled": shuffled_gain, "paired_results": paired,
        "paired_non_hrr_wins": wins, "paired_shuffled_wins": shuffled_wins,
        "development_criteria_passed": passed,
        "promotion_eligible": bool(acceptance.get("promotion_eligible", False)),
        "gate_passed": passed and bool(acceptance.get("promotion_eligible", False)),
    }


def render_report(summary: dict[str, Any], nodes: int, edges: int, queries: int) -> str:
    lines = ["# Experiment 01c.5 — complete parent-neighborhood retrieval", "",
             f"Nodes: **{nodes}**; edges: **{edges}**; source queries: **{queries}**.", "",
             f"Protocol: **{summary['protocol']}**.", "",
             "| Method | Query MRR | MRR, parent seen in training | MRR, parent unseen | Set R@10 | Hit@10 | Set NLL | Centroid cosine | Targets/query |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, metrics in summary["mean_metrics"].items():
        lines.append(f"| {method} | {metrics['query_mrr']:.4f} | {metrics['query_mrr_seen']:.4f} | "
                     f"{metrics['query_mrr_unseen']:.4f} | {metrics['set_recall_at_10']:.4f} | "
                     f"{metrics['set_hit_at_10']:.4f} | {metrics['set_mass_nll']:.4f} | "
                     f"{metrics['target_centroid_cosine']:.4f} | {metrics['mean_targets_per_query']:.2f} |")
    ci = lambda value: "n/a" if value["ci_low"] is None else f"[{value['ci_low']:+.4f}, {value['ci_high']:+.4f}]"
    lines += ["", f"HRR MRR gain over matched non-HRR: **{summary['mrr_gain_over_non_hrr']:+.4f}** "
              f"(per-seed 95% CI {ci(summary['mrr_gain_over_non_hrr_ci'])}).",
              f"Set-recall gain: **{summary['set_recall_gain_over_non_hrr']:+.4f}**.",
              f"Gain over shuffled HRR: **{summary['mrr_gain_over_shuffled']:+.4f}** "
              f"(95% CI {ci(summary['mrr_gain_over_shuffled_ci'])}).",
              f"Development criteria (proxy): **{'PASS' if summary['development_criteria_passed'] else 'FAIL'}**.", "",
              "`nan` in the seen column means no held-out parent was a training endpoint (target-disjoint "
              "split); `nan` in the unseen column means every held-out parent was.", ""]
    return "\n".join(lines)


def _split(edges: list[RelationEdge], data: dict[str, Any], seed: int, v2: bool) -> EdgeSplit:
    if not v2:
        train, test = split_source_neighborhoods(edges, test_fraction=float(data["test_fraction"]), seed=seed)
        return EdgeSplit(train, [], test, [])
    return split_neighborhood_partitions(
        edges, test_fraction=float(data["test_fraction"]), seed=seed,
        validation_fraction=float(data["validation_fraction"]),
        target_disjoint=bool(data["target_disjoint"]),
    )


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = resolve_config(config)
    v2 = config["protocol"] >= 2
    git_at_start = prepare_output_dir(output_dir)
    require_clean_tree_for_promotion(bool(config["acceptance"]["promotion_eligible"]))
    apply_thread_setting(config)
    host_config, data_config, fit_config = config["host"], config["data"], config["fit"]
    ties = config["evaluation"]["ties"]
    tokenizer = AutoTokenizer.from_pretrained(host_config["model"], revision=host_config["revision"], local_files_only=True)
    host = AutoModelForCausalLM.from_pretrained(host_config["model"], revision=host_config["revision"], local_files_only=True)
    relations = data_config["relation_types"]
    nodes, edges = collect_source_neighborhood_graph(
        wn, tokenizer, relations,
        max_sources_per_relation=int(data_config["max_sources_per_relation"]),
        min_targets_per_source=int(data_config["min_targets_per_source"]),
        seed=int(data_config["selection_seed"]),
    )
    group_counts = Counter((edge.source, edge.relation) for edge in edges)
    relation_queries = Counter(relation for _, relation in group_counts)
    missing = [r for r in relations if relation_queries[r] < int(data_config["min_sources_per_relation"])]
    if missing: raise RuntimeError(f"insufficient source neighborhoods: {missing}; {dict(relation_queries)}")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    anchors = contextual_anchors(host, tokenizer, nodes, batch_size=int(host_config["batch_size"]),
                                 max_length=int(host_config["max_length"]), device=device,
                                 truncation=host_config["anchor_truncation"])
    rows, split_rows, split_stats, checkpoints = [], [], [], {}
    for seed in config["seeds"]:
        split = _split(edges, data_config, int(seed), v2)
        if not split.train or not split.test: raise RuntimeError(f"empty partition for seed {seed}")
        split_rows.extend({"seed": seed, "edge_index": index, "partition": part}
                          for part, ids in split.partitions().items() for index in ids)
        split_stats.append({"seed": seed, **split.stats})
        raw, source_ids, target_ids = _tensors(edges, nodes, relations, anchors)
        endpoint_ids = torch.cat((source_ids[split.train], target_ids[split.train])).unique()
        centered = F.normalize(anchors - anchors[endpoint_ids].mean(0), dim=-1)
        all_data, source_ids, target_ids = _tensors(edges, nodes, relations, centered)
        train_cpu = all_data.subset(split.train)
        partitions = [("validation", split.validation)] if split.validation else []
        partitions.append(("test", split.test))
        train = HostEdges(*(tensor.to(device) for tensor in (
            train_cpu.sources, train_cpu.targets, train_cpu.relation_ids,
            train_cpu.source_node_ids, train_cpu.target_node_ids,
        )))
        for method_config in config["methods"]:
            method = method_config if isinstance(method_config, str) else method_config["family"]
            shuffled_method = method.startswith("shuffled_")
            family = method.removeprefix("shuffled_") if shuffled_method else method
            torch.manual_seed(int(seed) + 100)
            model = _build_model(family, method_config, config, len(relations), centered.shape[1]).to(device)
            fit_train = train
            if shuffled_method:
                generator = torch.Generator().manual_seed(int(seed) + 991)
                query_keys = torch.stack((train.source_node_ids.cpu(), train.relation_ids.cpu()), dim=1)
                unique_keys, inverse = torch.unique(query_keys, dim=0, return_inverse=True)
                if fit_config["shuffle"] == "derangement":
                    shuffled_query_relations = derange_labels(unique_keys[:, 1], generator)
                else:
                    perm = torch.randperm(len(unique_keys), generator=generator)
                    shuffled_query_relations = unique_keys[perm, 1]
                fit_train = HostEdges(train.sources, train.targets, shuffled_query_relations[inverse].to(device),
                                      train.source_node_ids, train.target_node_ids)
            initial, final = fit_host_relations(
                model, fit_train, steps=int(fit_config["steps"]),
                learning_rate=float(fit_config["learning_rate"]),
                cosine_weight=float(fit_config["cosine_weight"]),
                correction_weight=float(fit_config["correction_weight"]),
                offset_weight=float(fit_config["offset_weight"]),
                basis_weight=float(fit_config["basis_weight"]),
                rank_weight=float(fit_config["rank_weight"]),
                rank_temperature=float(fit_config["rank_temperature"]),
                rank_positives=fit_config["rank_positives"],
                parameter_free_objective=v2,
            )
            for partition, indices in partitions:
                data = all_data.subset(indices)
                with torch.no_grad(): prediction = model(data.sources.to(device), data.relation_ids.to(device)).cpu()
                metrics = source_neighborhood_metrics(
                    prediction, data.source_node_ids, data.target_node_ids,
                    data.relation_ids, centered, temperature=float(fit_config["rank_temperature"]),
                    ties=ties, seen_node_ids=endpoint_ids,
                )
                rows.append({"seed": seed, "method": method, "partition": partition,
                             "initial_loss": initial, "final_loss": final,
                             "parameters": sum(p.numel() for p in model.parameters()), **metrics})
            if seed == config["seeds"][0]: checkpoints[method] = model.cpu().state_dict()
    summary = summarize(rows, config)
    _write_csv(output_dir / "metrics.csv", rows); _write_csv(output_dir / "splits.csv", split_rows)
    _write_csv(output_dir / "split_stats.csv", split_stats)
    _write_csv(output_dir / "nodes.csv", [node.__dict__ for node in nodes])
    _write_csv(output_dir / "edges.csv", [{"index": i, **edge.__dict__} for i, edge in enumerate(edges)])
    torch.save({"schema_version": 1, "models": checkpoints}, output_dir / "relation_models.pt")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device=device, wordnet=wn.get_version(), host=host_config,
                       anchor_sha256=hashlib.sha256(anchors.numpy().tobytes()).hexdigest(),
                       relation_queries=dict(relation_queries),
                       target_count_distribution=dict(Counter(group_counts.values())))
    (output_dir / "report.md").write_text(render_report(summary, len(nodes), len(edges), len(group_counts)))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(yaml.safe_load(args.config.read_text()), args.output), indent=2))


if __name__ == "__main__": main()