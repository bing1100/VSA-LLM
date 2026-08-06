"""Experiment 01b Stage B: heterogeneous WordNet relations in frozen GPT-2 spaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from nltk.corpus import wordnet as wn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsa_embed.real_relations import (
    HostEdges, HostRelationModel, fit_host_relations, host_relation_metrics,
    multi_positive_target_retrieval_metrics, prediction_metrics, target_retrieval_metrics,
)
from vsa_embed.wordnet_relations import (
    RelationEdge, SenseNode, collect_wordnet_graph, contextual_anchors,
    split_edges, static_anchors,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _edge_tensors(edges: list[RelationEdge], nodes: list[SenseNode], relations: list[str], anchors: torch.Tensor) -> tuple[HostEdges, torch.Tensor, torch.Tensor]:
    node_ids = {node.synset: index for index, node in enumerate(nodes)}
    relation_ids = {relation: index for index, relation in enumerate(relations)}
    sources = torch.tensor([node_ids[edge.source] for edge in edges])
    targets = torch.tensor([node_ids[edge.target] for edge in edges])
    typed = torch.tensor([relation_ids[edge.relation] for edge in edges])
    return HostEdges(anchors[sources], anchors[targets], typed, sources, targets), sources, targets


def _center_from_train(anchors: torch.Tensor, source_ids: torch.Tensor, target_ids: torch.Tensor, train: list[int]) -> torch.Tensor:
    endpoint_ids = torch.cat((source_ids[train], target_ids[train])).unique()
    mean = anchors[endpoint_ids].mean(0)
    return F.normalize(anchors - mean, dim=-1)


def _relation_mean(train: HostEdges, test: HostEdges, relation_count: int) -> torch.Tensor:
    global_mean = train.targets.mean(0)
    means = []
    for relation in range(relation_count):
        selected = train.targets[train.relation_ids == relation]
        means.append(selected.mean(0) if selected.numel() else global_mean)
    return F.normalize(torch.stack(means)[test.relation_ids], dim=-1)


def _evaluate_prediction(prediction: torch.Tensor, data: HostEdges, target_ids: torch.Tensor, anchors: torch.Tensor) -> dict[str, float]:
    metrics = {**prediction_metrics(prediction, data.targets),
               **target_retrieval_metrics(prediction, target_ids, anchors)}
    if data.source_node_ids is not None:
        metrics.update(multi_positive_target_retrieval_metrics(
            prediction, data.source_node_ids, target_ids, data.relation_ids, anchors,
        ))
    return metrics


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    host_config, data_config = config["host"], config["data"]
    tokenizer = AutoTokenizer.from_pretrained(host_config["model"], revision=host_config["revision"], local_files_only=True)
    host = AutoModelForCausalLM.from_pretrained(host_config["model"], revision=host_config["revision"], local_files_only=True)
    relations = data_config["relation_types"]
    nodes, edges = collect_wordnet_graph(
        wn, tokenizer, relations, max_edges_per_relation=int(data_config["max_edges_per_relation"]),
        seed=int(data_config["selection_seed"]),
    )
    counts = Counter(edge.relation for edge in edges)
    missing = [relation for relation in relations if counts[relation] < int(data_config["min_edges_per_relation"])]
    if missing: raise RuntimeError(f"insufficient aligned edges for relations: {missing}; counts={dict(counts)}")
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    anchors_by_track: dict[str, torch.Tensor] = {}
    if any(track.startswith("static") for track in config["tracks"]):
        anchors_by_track["static"] = static_anchors(host, nodes)
    if any(track.startswith("contextual") for track in config["tracks"]):
        anchors_by_track["contextual"] = contextual_anchors(
            host, tokenizer, nodes, batch_size=int(host_config["batch_size"]),
            max_length=int(host_config["max_length"]), device=device,
        )
        host = host.cpu()
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    for seed in config["seeds"]:
        for split_mode in config["split_modes"]:
            train_ids, test_ids, discarded = split_edges(
                edges, mode=split_mode, test_fraction=float(data_config["test_fraction"]), seed=int(seed),
            )
            if not train_ids or not test_ids: raise RuntimeError(f"empty {split_mode} partition for seed {seed}")
            split_rows.extend({"seed": seed, "split_mode": split_mode, "edge_index": i, "partition": part}
                              for part, indices in (("train", train_ids), ("test", test_ids), ("discarded", discarded)) for i in indices)
            for track in config["tracks"]:
                base = anchors_by_track[track.split("_")[0]]
                _, source_ids, target_ids = _edge_tensors(edges, nodes, relations, base)
                anchors = _center_from_train(base, source_ids, target_ids, train_ids) if track.endswith("centered") else base
                all_data, source_ids, target_ids = _edge_tensors(edges, nodes, relations, anchors)
                train_cpu, test_cpu = all_data.subset(train_ids), all_data.subset(test_ids)
                train = HostEdges(
                    train_cpu.sources.to(device), train_cpu.targets.to(device), train_cpu.relation_ids.to(device),
                    train_cpu.source_node_ids.to(device), train_cpu.target_node_ids.to(device),
                )
                test = HostEdges(
                    test_cpu.sources.to(device), test_cpu.targets.to(device), test_cpu.relation_ids.to(device),
                    test_cpu.source_node_ids.to(device), test_cpu.target_node_ids.to(device),
                )
                for method_config in config["methods"]:
                    method = method_config if isinstance(method_config, str) else method_config["family"]
                    diagnostics: dict[str, float] = {}
                    if method == "relation_mean":
                        prediction = _relation_mean(train, test, len(relations))
                    else:
                        shuffled_method = method.startswith("shuffled_")
                        family = method.removeprefix("shuffled_") if shuffled_method else method
                        torch.manual_seed(int(seed) + 100)
                        model = HostRelationModel(len(relations), anchors.shape[1], family,
                                                  rank=int(method_config.get("rank", config["fit"]["rank"])) if isinstance(method_config, dict) else int(config["fit"]["rank"]),
                                                  max_residual_scale=float(method_config.get("max_residual_scale", config["fit"].get("max_residual_scale", 0.25))) if isinstance(method_config, dict) else float(config["fit"].get("max_residual_scale", 0.25)),
                                                  max_log_basis_scale=float(method_config.get("max_log_basis_scale", config["fit"].get("max_log_basis_scale", 0.25))) if isinstance(method_config, dict) else float(config["fit"].get("max_log_basis_scale", 0.25))).to(device)
                        fit_train = train
                        if shuffled_method:
                            generator = torch.Generator().manual_seed(int(seed) + 991)
                            shuffled = train.relation_ids.cpu()[torch.randperm(len(train.relation_ids), generator=generator)].to(device)
                            fit_train = HostEdges(
                                train.sources, train.targets, shuffled,
                                train.source_node_ids, train.target_node_ids,
                            )
                        initial, final = fit_host_relations(
                            model, fit_train, steps=int(config["fit"]["steps"]),
                            learning_rate=float(config["fit"]["learning_rate"]),
                            cosine_weight=float(config["fit"]["cosine_weight"]),
                            correction_weight=float(config["fit"].get("correction_weight", 0.0)),
                            offset_weight=float(config["fit"].get("offset_weight", 0.0)),
                            basis_weight=float(config["fit"].get("basis_weight", 0.0)),
                            rank_weight=float(config["fit"].get("rank_weight", 0.0)),
                            rank_temperature=float(config["fit"].get("rank_temperature", 0.07)),
                        )
                        diagnostics = {"initial_loss": initial, "final_loss": final,
                                       "parameters": sum(p.numel() for p in model.parameters())}
                        diagnostics.update(model.diagnostics(test.sources, test.relation_ids))
                        with torch.no_grad(): prediction = model(test.sources, test.relation_ids)
                        relation_metric = host_relation_metrics(model, test)["relation_accuracy"]
                        diagnostics["relation_accuracy"] = relation_metric
                        if seed == config["seeds"][0] and split_mode == config["split_modes"][0]:
                            checkpoints[f"{track}-{method}"] = model.cpu().state_dict()
                    overall = _evaluate_prediction(prediction.cpu(), test_cpu, target_ids[test_ids], anchors)
                    rows.append({"seed": seed, "split_mode": split_mode, "track": track, "method": method,
                                 "relation": "all", "n": len(test_ids), **diagnostics, **overall})
                    for relation_id, relation in enumerate(relations):
                        selector = (test_cpu.relation_ids == relation_id).nonzero().flatten()
                        if selector.numel():
                            metrics = prediction_metrics(prediction.cpu()[selector], test_cpu.targets[selector])
                            # Keep the same candidate universe as the overall split; otherwise tiny
                            # relation strata receive artificially inflated retrieval scores.
                            metrics.update(target_retrieval_metrics(prediction.cpu()[selector],
                                           target_ids[test_ids][selector], anchors, target_ids[test_ids]))
                            metrics.update(multi_positive_target_retrieval_metrics(
                                prediction.cpu()[selector], test_cpu.source_node_ids[selector],
                                target_ids[test_ids][selector], test_cpu.relation_ids[selector], anchors,
                            ))
                            rows.append({"seed": seed, "split_mode": split_mode, "track": track, "method": method,
                                         "relation": relation, "n": int(selector.numel()), **diagnostics, **metrics})
    _write_csv(output_dir / "metrics.csv", rows); _write_csv(output_dir / "splits.csv", split_rows)
    _write_csv(output_dir / "nodes.csv", [node.__dict__ for node in nodes])
    _write_csv(output_dir / "edges.csv", [{"index": i, **edge.__dict__} for i, edge in enumerate(edges)])
    torch.save({"schema_version": 1, "models": checkpoints}, output_dir / "relation_models.pt")
    hashes = {track: hashlib.sha256(value.numpy().tobytes()).hexdigest() for track, value in anchors_by_track.items()}
    summary = summarize(rows, config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "torch": torch.__version__, "wordnet": wn.get_version(),
                "host": host_config, "anchor_sha256": hashes, "relation_counts": dict(counts)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "report.md").write_text(render_report(summary, len(nodes), len(edges), counts))
    return summary


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if config["acceptance"].get("mode") == "taxonomy_retrieval":
        return summarize_taxonomy_retrieval(rows, config)
    if "candidate_methods" in config["acceptance"]:
        return summarize_reconstruction_rescue(rows, config)
    overall = [row for row in rows if row["relation"] == "all"]
    means: dict[str, dict[str, dict[str, float]]] = {}
    for split in config["split_modes"]:
        means[split] = {}
        for track in config["tracks"]:
            means[split][track] = {}
            for method_config in config["methods"]:
                method = method_config if isinstance(method_config, str) else method_config["family"]
                selected = [row for row in overall if row["split_mode"] == split and row["track"] == track and row["method"] == method]
                means[split][track][method] = sum(row["target_mrr"] for row in selected) / len(selected)
    primary_split, primary_track = config["acceptance"]["primary_split"], config["acceptance"]["primary_track"]
    scores = means[primary_split][primary_track]
    structured = [name for name in scores if name in {"hrr", "diagonal", "low_rank", "offset"}]
    controls = [name for name in scores if name not in structured]
    best_structured = max(structured, key=scores.get); best_control = max(controls, key=scores.get)
    gain = scores[best_structured] - scores[best_control]
    primary = [row for row in overall if row["split_mode"] == primary_split and row["track"] == primary_track]
    by_seed = {(str(row["seed"]), row["method"]): row for row in primary}
    matched_control = f"shuffled_{best_structured}"
    paired = []
    for seed in config["seeds"]:
        key = str(seed)
        structured_row = by_seed[(key, best_structured)]
        additive_row = by_seed[(key, "additive")]
        shuffled_row = by_seed.get((key, matched_control))
        paired.append({
            "seed": seed,
            "structured_mrr": structured_row["target_mrr"],
            "additive_mrr": additive_row["target_mrr"],
            "matched_shuffled_mrr": shuffled_row["target_mrr"] if shuffled_row else None,
            "gain_over_additive": structured_row["target_mrr"] - additive_row["target_mrr"],
            "gain_over_matched_shuffled": structured_row["target_mrr"] - shuffled_row["target_mrr"] if shuffled_row else None,
        })
    paired_additive_wins = sum(item["gain_over_additive"] > 0 for item in paired)
    paired_shuffled_wins = sum(item["gain_over_matched_shuffled"] is not None and item["gain_over_matched_shuffled"] > 0 for item in paired)
    structured_rows = [row for row in primary if row["method"] == best_structured]
    additive_rows = [row for row in primary if row["method"] == "additive"]
    cosine_gain = (sum(row["cosine"] for row in structured_rows) - sum(row["cosine"] for row in additive_rows)) / len(structured_rows)
    required_wins = int(config["acceptance"].get("min_paired_wins", len(config["seeds"])))
    retrieval_pass = gain >= float(config["acceptance"]["min_mrr_gain"]) and paired_additive_wins >= required_wins and paired_shuffled_wins >= required_wins
    reconstruction_pass = cosine_gain >= float(config["acceptance"].get("min_cosine_gain", 0.0))
    return {"conditions": len(rows), "mean_target_mrr": means, "best_structured": best_structured,
            "best_control": best_control, "matched_shuffled_control": matched_control,
            "primary_gain": gain, "primary_cosine_gain": cosine_gain,
            "paired_results": paired, "paired_additive_wins": paired_additive_wins,
            "paired_shuffled_wins": paired_shuffled_wins,
            "retrieval_gate_passed": retrieval_pass, "reconstruction_gate_passed": reconstruction_pass,
            "gate_passed": retrieval_pass}


def summarize_reconstruction_rescue(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Summarize 01c as reconstruction gain subject to retrieval non-inferiority."""
    overall = [row for row in rows if row["relation"] == "all"]
    acceptance = config["acceptance"]
    split, track = acceptance["primary_split"], acceptance["primary_track"]
    primary = [row for row in overall if row["split_mode"] == split and row["track"] == track]
    methods = [item if isinstance(item, str) else item["family"] for item in config["methods"]]
    mean_metrics = {
        method: {
            metric: sum(float(row[metric]) for row in primary if row["method"] == method)
                    / len([row for row in primary if row["method"] == method])
            for metric in ("cosine", "mse", "target_mrr", "target_recall_at_10")
        }
        for method in methods
    }
    candidates = [name for name in acceptance["candidate_methods"] if name in mean_metrics]
    if not candidates:
        raise ValueError("01c acceptance.candidate_methods contains no configured method")
    reconstruction_baseline = acceptance.get("reconstruction_baseline", "offset")
    retrieval_baseline = acceptance.get("retrieval_baseline", "hrr")
    min_mrr_delta = float(acceptance.get("min_mrr_delta", 0.0))
    eligible = [name for name in candidates if (
        mean_metrics[name]["target_mrr"] - mean_metrics[retrieval_baseline]["target_mrr"]
    ) >= min_mrr_delta]
    best = max(eligible or candidates, key=lambda name: mean_metrics[name]["cosine"])
    cosine_gain = mean_metrics[best]["cosine"] - mean_metrics[reconstruction_baseline]["cosine"]
    mrr_delta = mean_metrics[best]["target_mrr"] - mean_metrics[retrieval_baseline]["target_mrr"]
    shuffled_name = f"shuffled_{best}"
    shuffled = mean_metrics.get(shuffled_name)
    shuffled_cosine_gain = (
        mean_metrics[best]["cosine"] - shuffled["cosine"] if shuffled is not None else None
    )
    by_seed = {(str(row["seed"]), row["method"]): row for row in primary}
    paired = []
    for seed in config["seeds"]:
        key = str(seed)
        candidate = by_seed[(key, best)]
        reconstruction = by_seed[(key, reconstruction_baseline)]
        retrieval = by_seed[(key, retrieval_baseline)]
        shuffled_row = by_seed.get((key, shuffled_name))
        paired.append({
            "seed": seed,
            "cosine_gain_over_reconstruction_baseline": candidate["cosine"] - reconstruction["cosine"],
            "mrr_delta_from_retrieval_baseline": candidate["target_mrr"] - retrieval["target_mrr"],
            "cosine_gain_over_matched_shuffled": (
                candidate["cosine"] - shuffled_row["cosine"] if shuffled_row else None
            ),
        })
    required_wins = int(acceptance.get("min_paired_wins", len(config["seeds"])))
    reconstruction_wins = sum(item["cosine_gain_over_reconstruction_baseline"] > 0 for item in paired)
    retrieval_noninferior = sum(
        item["mrr_delta_from_retrieval_baseline"] >= min_mrr_delta
        for item in paired
    )
    shuffled_wins = sum(
        item["cosine_gain_over_matched_shuffled"] is not None
        and item["cosine_gain_over_matched_shuffled"] > 0 for item in paired
    )
    require_shuffled = bool(acceptance.get("require_matched_shuffled", True))
    criteria_passed = (
        cosine_gain >= float(acceptance.get("min_cosine_gain", 0.0))
        and mrr_delta >= min_mrr_delta
        and reconstruction_wins >= required_wins
        and retrieval_noninferior >= required_wins
        and (not require_shuffled or (shuffled is not None and shuffled_wins >= required_wins))
    )
    promotion_eligible = bool(acceptance.get("promotion_eligible", True))
    return {
        "experiment_stage": "01c",
        "conditions": len(rows),
        "primary_split": split,
        "primary_track": track,
        "mean_metrics": mean_metrics,
        "best_candidate": best,
        "retrieval_eligible_candidates": eligible,
        "reconstruction_baseline": reconstruction_baseline,
        "retrieval_baseline": retrieval_baseline,
        "matched_shuffled_control": shuffled_name if shuffled is not None else None,
        "cosine_gain_over_reconstruction_baseline": cosine_gain,
        "mrr_delta_from_retrieval_baseline": mrr_delta,
        "cosine_gain_over_matched_shuffled": shuffled_cosine_gain,
        "paired_results": paired,
        "paired_reconstruction_wins": reconstruction_wins,
        "paired_retrieval_noninferior": retrieval_noninferior,
        "paired_shuffled_wins": shuffled_wins,
        "exploratory_criteria_passed": criteria_passed,
        "promotion_eligible": promotion_eligible,
        "gate_passed": criteria_passed and promotion_eligible,
    }


def summarize_taxonomy_retrieval(
    rows: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    """Gate taxonomy HRR on retrieval versus matched non-HRR and shuffled controls."""
    acceptance = config["acceptance"]
    primary = [
        row for row in rows if row["relation"] == "all"
        and row["split_mode"] == acceptance["primary_split"]
        and row["track"] == acceptance["primary_track"]
    ]
    methods = [item if isinstance(item, str) else item["family"] for item in config["methods"]]
    mean_metrics = {
        method: {
            metric: sum(float(row[metric]) for row in primary if row["method"] == method)
                    / len([row for row in primary if row["method"] == method])
            for metric in ("cosine", "mse", "target_mrr", "target_recall_at_10",
                           "distribution_mrr", "distribution_recall_at_10", "mean_positive_targets")
        } for method in methods
    }
    candidate = acceptance["candidate"]
    control = acceptance["non_hrr_control"]
    shuffled = acceptance["shuffled_control"]
    reconstruction = acceptance.get("reconstruction_baseline", "offset")
    by_seed = {(str(row["seed"]), row["method"]): row for row in primary}
    paired = []
    for seed in config["seeds"]:
        c = by_seed[(str(seed), candidate)]
        d = by_seed[(str(seed), control)]
        s = by_seed[(str(seed), shuffled)]
        o = by_seed[(str(seed), reconstruction)]
        paired.append({
            "seed": seed,
            "mrr_gain_over_non_hrr": c["distribution_mrr"] - d["distribution_mrr"],
            "mrr_gain_over_shuffled": c["distribution_mrr"] - s["distribution_mrr"],
            "cosine_delta_from_reconstruction_baseline": c["cosine"] - o["cosine"],
        })
    required = int(acceptance.get("min_paired_wins", len(config["seeds"])))
    mrr_gain = mean_metrics[candidate]["distribution_mrr"] - mean_metrics[control]["distribution_mrr"]
    shuffled_gain = mean_metrics[candidate]["distribution_mrr"] - mean_metrics[shuffled]["distribution_mrr"]
    cosine_delta = mean_metrics[candidate]["cosine"] - mean_metrics[reconstruction]["cosine"]
    control_wins = sum(item["mrr_gain_over_non_hrr"] > 0 for item in paired)
    shuffled_wins = sum(item["mrr_gain_over_shuffled"] > 0 for item in paired)
    locality_wins = sum(
        item["cosine_delta_from_reconstruction_baseline"] >= float(acceptance["min_cosine_delta"])
        for item in paired
    )
    passed = (
        mrr_gain >= float(acceptance["min_mrr_gain_over_non_hrr"])
        and shuffled_gain > 0
        and cosine_delta >= float(acceptance["min_cosine_delta"])
        and control_wins >= required and shuffled_wins >= required and locality_wins >= required
    )
    promotion_eligible = bool(acceptance.get("promotion_eligible", False))
    return {
        "experiment_stage": "01c.4-taxonomy-retrieval",
        "conditions": len(rows), "primary_split": acceptance["primary_split"],
        "primary_track": acceptance["primary_track"], "mean_metrics": mean_metrics,
        "candidate": candidate, "non_hrr_control": control, "shuffled_control": shuffled,
        "mrr_gain_over_non_hrr": mrr_gain, "mrr_gain_over_shuffled": shuffled_gain,
        "cosine_delta_from_reconstruction_baseline": cosine_delta,
        "paired_results": paired, "paired_non_hrr_wins": control_wins,
        "paired_shuffled_wins": shuffled_wins, "paired_locality_wins": locality_wins,
        "development_criteria_passed": passed, "promotion_eligible": promotion_eligible,
        "gate_passed": passed and promotion_eligible,
    }


def render_report(summary: dict[str, Any], nodes: int, edges: int, counts: Counter[str]) -> str:
    if summary.get("experiment_stage") == "01c.4-taxonomy-retrieval":
        return render_taxonomy_retrieval_report(summary, nodes, edges, counts)
    if summary.get("experiment_stage") == "01c":
        return render_reconstruction_rescue_report(summary, nodes, edges, counts)
    lines = ["# Experiment 01b Stage B — heterogeneous WordNet / frozen GPT-2", "",
             f"Nodes: **{nodes}**; edges: **{edges}**; relation counts: `{dict(counts)}`.", ""]
    for split, tracks in summary["mean_target_mrr"].items():
        for track, methods in tracks.items():
            lines += [f"## {split} / {track}", "", "| Method | Target MRR |", "|---|---:|",
                      *[f"| {method} | {score:.4f} |" for method, score in methods.items()], ""]
    lines += ["## Stage-B gate", "", f"Best structured: **{summary['best_structured']}**; best control: **{summary['best_control']}**; "
              f"primary MRR gain: **{summary['primary_gain']:+.4f}**.",
              f"Paired wins over additive: **{summary['paired_additive_wins']}**; over matched shuffled control: **{summary['paired_shuffled_wins']}**.",
              f"Primary cosine gain: **{summary['primary_cosine_gain']:+.4f}**.",
              f"Retrieval gate: **{'PASS' if summary['retrieval_gate_passed'] else 'FAIL'}**; reconstruction gate: **{'PASS' if summary['reconstruction_gate_passed'] else 'FAIL'}**.", "",
              "This tests frozen representation transfer, not vocabulary insertion or generated-answer behavior.", ""]
    return "\n".join(lines)


def render_taxonomy_retrieval_report(
    summary: dict[str, Any], nodes: int, edges: int, counts: Counter[str],
) -> str:
    lines = ["# Experiment 01c.4 — taxonomy distributional retrieval", "",
             f"Nodes: **{nodes}**; edges: **{edges}**; relation counts: `{dict(counts)}`.", "",
             "| Method | Distribution MRR | Distribution R@10 | Exact MRR | Cosine |", "|---|---:|---:|---:|---:|"]
    for method, metrics in summary["mean_metrics"].items():
        lines.append(
            f"| {method} | {metrics['distribution_mrr']:.4f} | "
            f"{metrics['distribution_recall_at_10']:.4f} | {metrics['target_mrr']:.4f} | "
            f"{metrics['cosine']:.4f} |"
        )
    lines += ["", f"Distribution-MRR gain over non-HRR control: **{summary['mrr_gain_over_non_hrr']:+.4f}**.",
              f"Gain over shuffled control: **{summary['mrr_gain_over_shuffled']:+.4f}**.",
              f"Cosine delta from offset: **{summary['cosine_delta_from_reconstruction_baseline']:+.4f}**.",
              f"Development criteria: **{'PASS' if summary['development_criteria_passed'] else 'FAIL'}**.", ""]
    return "\n".join(lines)


def render_reconstruction_rescue_report(
    summary: dict[str, Any], nodes: int, edges: int, counts: Counter[str],
) -> str:
    lines = ["# Experiment 01c — reconstruction rescue", "",
             f"Nodes: **{nodes}**; edges: **{edges}**; relation counts: `{dict(counts)}`.", "",
             f"Primary condition: **{summary['primary_split']} / {summary['primary_track']}**.", "",
             "| Method | Cosine | MSE | Target MRR | Recall@10 |", "|---|---:|---:|---:|---:|"]
    for method, metrics in summary["mean_metrics"].items():
        lines.append(f"| {method} | {metrics['cosine']:.4f} | {metrics['mse']:.5f} | "
                     f"{metrics['target_mrr']:.4f} | {metrics['target_recall_at_10']:.4f} |")
    lines += ["", "## Exploratory joint criterion", "",
              f"Best reconstruction candidate: **{summary['best_candidate']}**.",
              f"Cosine gain over **{summary['reconstruction_baseline']}**: "
              f"**{summary['cosine_gain_over_reconstruction_baseline']:+.4f}**.",
              f"MRR delta from **{summary['retrieval_baseline']}**: "
              f"**{summary['mrr_delta_from_retrieval_baseline']:+.4f}**.",
              f"Exploratory criteria: **{'PASS' if summary['exploratory_criteria_passed'] else 'FAIL'}**; "
              f"promotion eligible: **{summary['promotion_eligible']}**; final gate: "
              f"**{'PASS' if summary['gate_passed'] else 'FAIL'}**.", "",
              "A smoke result is not evidence for promotion. Full 01c requires development/confirmation "
              "separation, multiple seeds, target audits, behavioral locality, and a second ontology.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(yaml.safe_load(args.config.read_text()), args.output), indent=2))


if __name__ == "__main__": main()
