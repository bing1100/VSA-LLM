"""Experiment 01b Stage B / 01c: heterogeneous WordNet relations in frozen GPT-2 spaces.

Configs without a `protocol` key (protocol 1) replay the recorded 2026-07 runs exactly.
`protocol: 2` switches on the corrections from the implementation audit
(`resources/plan-improvement/audit.md`): a validation partition with selection on it only,
named candidates, shared candidate sets and mid-rank ties in every metric, tie-aware relation
accuracy for parametric models only, rank-loss positives by target set, derangement shuffles,
symmetric `attribute` grouping, gloss-only anchor truncation, and confidence intervals.
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

from vsa_embed.provenance import (
    apply_thread_setting, prepare_output_dir, require_clean_tree_for_promotion, write_run_metadata,
)
from vsa_embed.real_relations import (
    HostEdges, HostRelationModel, derange_labels, fit_host_relations, host_relation_metrics,
    multi_positive_target_retrieval_metrics, prediction_metrics, target_retrieval_metrics,
)
from vsa_embed.statistics import mean_confidence_interval
from vsa_embed.wordnet_relations import (
    LEGACY_SYMMETRIC_RELATIONS, SYMMETRIC_RELATIONS, EdgeSplit, RelationEdge, SenseNode,
    collect_wordnet_graph, contextual_anchors, split_edge_partitions, split_edges, static_anchors,
)

STRUCTURED_FAMILIES = {"hrr", "diagonal", "low_rank", "offset"}

DESIGN_ITEMS_01B = (
    "1 beats weighted additive and the strongest equal-budget graph/unstructured baseline on held-out behavioral transfer",
    "2 shifts the adaptation curve left",
    "3 globally shared relations outperform edge-specific memorization on held-out transfer",
    "4 passes recipe/edge/alias leakage audits and reproduces over at least three splits",
    "5 transfers to a second relation or ontology family",
    "6 preserves frozen-host locality",
    "7 emits auditable relation parameters, edge salience, residual allocation, uncertainty and provenance",
)
DESIGN_ITEMS_01C_NOT_EVALUATED = (
    "2 positive relation-residual R²",
    "6 improves more than one relation family",
    "7 reproduces on a second ontology/domain",
    "8 stable across target paraphrases/templates",
    "9 frozen-host behavioral locality",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill every default the runner uses so `resolved_config.yaml` records them."""
    resolved = copy.deepcopy(config)
    protocol = int(resolved.get("protocol", 1))
    v2 = protocol >= 2
    resolved["protocol"] = protocol
    data = resolved["data"]
    # Validation is needed only to select among several candidates; a node-disjoint validation
    # partition costs training data, so it is opt-in.
    data.setdefault("validation_fraction", 0.0)
    data.setdefault("symmetric_relations", list(SYMMETRIC_RELATIONS if v2 else LEGACY_SYMMETRIC_RELATIONS))
    data.setdefault("drop_same_token_edges", False)
    resolved["host"].setdefault("anchor_truncation", "gloss" if v2 else "right")
    fit = resolved["fit"]
    for key, value in {
        "max_residual_scale": 0.25, "max_log_basis_scale": 0.25, "correction_weight": 0.0,
        "offset_weight": 0.0, "basis_weight": 0.0, "rank_weight": 0.0, "rank_temperature": 0.07,
    }.items():
        fit.setdefault(key, value)
    fit.setdefault("rank_positives", "target_set" if v2 else "source_relation")
    fit.setdefault("parameter_free_objective", v2)
    fit.setdefault("shuffle", "derangement" if v2 else "permutation")
    resolved.setdefault("evaluation", {}).setdefault("ties", "mid" if v2 else "optimistic")
    resolved["acceptance"].setdefault("promotion_eligible", False)
    resolved.setdefault("device", "cuda")
    return resolved


def _method_name(method_config: str | dict[str, Any]) -> str:
    return method_config if isinstance(method_config, str) else method_config["family"]


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


def _evaluate_prediction(
    prediction: torch.Tensor, data: HostEdges, target_ids: torch.Tensor, anchors: torch.Tensor, *,
    ties: str,
) -> dict[str, float]:
    metrics = {**prediction_metrics(prediction, data.targets),
               **target_retrieval_metrics(prediction, target_ids, anchors, ties=ties)}
    if data.source_node_ids is not None:
        metrics.update(multi_positive_target_retrieval_metrics(
            prediction, data.source_node_ids, target_ids, data.relation_ids, anchors, ties=ties,
        ))
    return metrics


def _split(
    edges: list[RelationEdge], mode: str, data: dict[str, Any], seed: int,
    token_of: dict[str, int], v2: bool,
) -> EdgeSplit:
    test_fraction = float(data["test_fraction"])
    if not v2:
        train, test, discarded = split_edges(edges, mode=mode, test_fraction=test_fraction, seed=seed)
        return EdgeSplit(train, [], test, discarded)
    base = "edge_disjoint" if mode == "edge_disjoint_size_matched" else mode
    options = {
        "test_fraction": test_fraction, "seed": seed,
        "validation_fraction": float(data["validation_fraction"]),
        "symmetric_relations": data["symmetric_relations"],
    }
    split = split_edge_partitions(
        edges, mode=base, node_groups=token_of if base == "token_disjoint" else None, **options,
    )
    if mode == "edge_disjoint_size_matched":
        # Edge-disjoint training is ~3x larger than node-disjoint training (audit F3); this mode
        # subsamples it to the node-disjoint size so split modes compare at equal data.
        reference = split_edge_partitions(edges, mode="node_disjoint", **options)
        generator = torch.Generator().manual_seed(seed + 2_000_003)
        keep = torch.randperm(len(split.train), generator=generator)[:len(reference.train)].tolist()
        kept = sorted(split.train[i] for i in keep)
        split = EdgeSplit(kept, split.validation, split.test,
                          sorted(set(split.train) - set(kept)))
        split.stats = {"edges": len(edges), **{
            f"{name}_{kind}": (len(ids) if kind == "edges" else len(ids) / len(edges))
            for name, ids in split.partitions().items() for kind in ("edges", "fraction")
        }}
    return split


def _build_model(
    family: str, method_config: str | dict[str, Any], config: dict[str, Any], relation_count: int,
    dimension: int,
) -> HostRelationModel:
    fit = config["fit"]
    options = method_config if isinstance(method_config, dict) else {}
    kwargs = {
        "rank": int(options.get("rank", fit["rank"])),
        "max_residual_scale": float(options.get("max_residual_scale", fit["max_residual_scale"])),
        "max_log_basis_scale": float(options.get("max_log_basis_scale", fit["max_log_basis_scale"])),
    }
    if family == "low_rank_matched":
        # Building the reference model must not consume the global RNG used for initialization.
        with torch.random.fork_rng(devices=[]):
            reference = HostRelationModel(relation_count, dimension, options.get("match", "hrr"), **kwargs)
        kwargs["match_parameters"] = sum(p.numel() for p in reference.parameters())
    return HostRelationModel(relation_count, dimension, family, **kwargs)


def _shuffled_relations(relation_ids: torch.Tensor, seed: int, mode: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 991)
    labels = relation_ids.cpu()
    if mode == "derangement":
        return derange_labels(labels, generator).to(relation_ids.device)
    if mode != "permutation":
        raise ValueError("fit.shuffle must be 'permutation' or 'derangement'")
    return labels[torch.randperm(len(labels), generator=generator)].to(relation_ids.device)


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
    nodes, edges = collect_wordnet_graph(
        wn, tokenizer, relations, max_edges_per_relation=int(data_config["max_edges_per_relation"]),
        seed=int(data_config["selection_seed"]),
        symmetric_relations=data_config["symmetric_relations"],
    )
    token_of = {node.synset: node.token_id for node in nodes}
    same_token = [token_of[edge.source] == token_of[edge.target] for edge in edges]
    dropped_same_token = 0
    if data_config["drop_same_token_edges"]:
        # Same-token edges are solved trivially by the identity on static tracks (audit F11).
        dropped_same_token = sum(same_token)
        edges = [edge for edge, same in zip(edges, same_token) if not same]
        used = {edge.source for edge in edges} | {edge.target for edge in edges}
        nodes = [node for node in nodes if node.synset in used]
        same_token = [False] * len(edges)
    counts = Counter(edge.relation for edge in edges)
    missing = [relation for relation in relations if counts[relation] < int(data_config["min_edges_per_relation"])]
    if missing: raise RuntimeError(f"insufficient aligned edges for relations: {missing}; counts={dict(counts)}")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    anchors_by_track: dict[str, torch.Tensor] = {}
    if any(track.startswith("static") for track in config["tracks"]):
        anchors_by_track["static"] = static_anchors(host, nodes)
    if any(track.startswith("contextual") for track in config["tracks"]):
        anchors_by_track["contextual"] = contextual_anchors(
            host, tokenizer, nodes, batch_size=int(host_config["batch_size"]),
            max_length=int(host_config["max_length"]), device=device,
            truncation=host_config["anchor_truncation"],
        )
        host = host.cpu()
    primary_split = config["acceptance"].get("primary_split")
    checkpoint_split = primary_split if primary_split in config["split_modes"] else config["split_modes"][0]
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    split_stats: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    for seed in config["seeds"]:
        for split_mode in config["split_modes"]:
            split = _split(edges, split_mode, data_config, int(seed), token_of, v2)
            if not split.train or not split.test: raise RuntimeError(f"empty {split_mode} partition for seed {seed}")
            if v2 and float(data_config["validation_fraction"]) > 0 and not split.validation:
                raise RuntimeError(f"empty validation partition for {split_mode}, seed {seed}")
            split_rows.extend({"seed": seed, "split_mode": split_mode, "edge_index": i, "partition": part}
                              for part, indices in split.partitions().items() for i in indices)
            split_stats.append({"seed": seed, "split_mode": split_mode, **split.stats})
            partitions = [("validation", split.validation)] if split.validation else []
            partitions.append(("test", split.test))
            for track in config["tracks"]:
                base = anchors_by_track[track.split("_")[0]]
                _, source_ids, target_ids = _edge_tensors(edges, nodes, relations, base)
                anchors = _center_from_train(base, source_ids, target_ids, split.train) if track.endswith("centered") else base
                all_data, source_ids, target_ids = _edge_tensors(edges, nodes, relations, anchors)
                train_cpu = all_data.subset(split.train)
                train = HostEdges(*(tensor.to(device) for tensor in (
                    train_cpu.sources, train_cpu.targets, train_cpu.relation_ids,
                    train_cpu.source_node_ids, train_cpu.target_node_ids,
                )))
                evaluated = {name: all_data.subset(ids) for name, ids in partitions}
                for method_config in config["methods"]:
                    method = _method_name(method_config)
                    model = None
                    diagnostics: dict[str, float] = {}
                    if method != "relation_mean":
                        shuffled_method = method.startswith("shuffled_")
                        family = method.removeprefix("shuffled_") if shuffled_method else method
                        torch.manual_seed(int(seed) + 100)
                        model = _build_model(family, method_config, config, len(relations), anchors.shape[1]).to(device)
                        fit_train = train
                        if shuffled_method:
                            fit_train = HostEdges(
                                train.sources, train.targets,
                                _shuffled_relations(train.relation_ids, int(seed), fit_config["shuffle"]),
                                train.source_node_ids, train.target_node_ids,
                            )
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
                            parameter_free_objective=bool(fit_config["parameter_free_objective"]),
                        )
                        diagnostics = {"initial_loss": initial, "final_loss": final,
                                       "parameters": sum(p.numel() for p in model.parameters())}
                    for partition, indices in partitions:
                        data_cpu = evaluated[partition]
                        data = HostEdges(*(tensor.to(device) for tensor in (
                            data_cpu.sources, data_cpu.targets, data_cpu.relation_ids,
                            data_cpu.source_node_ids, data_cpu.target_node_ids,
                        )))
                        row_diagnostics = dict(diagnostics)
                        if model is None:
                            prediction = _relation_mean(train, data, len(relations))
                        else:
                            row_diagnostics.update(model.diagnostics(data.sources, data.relation_ids))
                            with torch.no_grad(): prediction = model(data.sources, data.relation_ids)
                            row_diagnostics["relation_accuracy"] = host_relation_metrics(model, data, ties=ties)["relation_accuracy"]
                        partition_targets = target_ids[indices]
                        prediction = prediction.cpu()
                        overall = _evaluate_prediction(prediction, data_cpu, partition_targets, anchors, ties=ties)
                        rows.append({"seed": seed, "split_mode": split_mode, "track": track, "method": method,
                                     "partition": partition, "relation": "all", "n": len(indices),
                                     **row_diagnostics, **overall})
                        for relation_id, relation in enumerate(relations):
                            selector = (data_cpu.relation_ids == relation_id).nonzero().flatten()
                            if not selector.numel():
                                continue
                            metrics = prediction_metrics(prediction[selector], data_cpu.targets[selector])
                            # Keep the same candidate universe as the overall split; otherwise tiny
                            # relation strata receive artificially inflated retrieval scores.
                            metrics.update(target_retrieval_metrics(
                                prediction[selector], partition_targets[selector], anchors,
                                partition_targets, ties=ties,
                            ))
                            metrics.update(multi_positive_target_retrieval_metrics(
                                prediction[selector], data_cpu.source_node_ids[selector],
                                partition_targets[selector], data_cpu.relation_ids[selector], anchors,
                                # Protocol 1 scored this against the relation's own targets only (audit F5b).
                                partition_targets if v2 else None, ties=ties,
                            ))
                            rows.append({"seed": seed, "split_mode": split_mode, "track": track, "method": method,
                                         "partition": partition, "relation": relation,
                                         "n": int(selector.numel()), **row_diagnostics, **metrics})
                    if model is not None and seed == config["seeds"][0] and split_mode == checkpoint_split:
                        checkpoints[f"{track}-{method}"] = model.cpu().state_dict()
    _write_csv(output_dir / "metrics.csv", rows); _write_csv(output_dir / "splits.csv", split_rows)
    _write_csv(output_dir / "split_stats.csv", split_stats)
    _write_csv(output_dir / "nodes.csv", [node.__dict__ for node in nodes])
    _write_csv(output_dir / "edges.csv", [{"index": i, **edge.__dict__, "same_token": same}
                                          for i, (edge, same) in enumerate(zip(edges, same_token))])
    torch.save({"schema_version": 1, "split_mode": checkpoint_split, "models": checkpoints},
               output_dir / "relation_models.pt")
    hashes = {track: hashlib.sha256(value.numpy().tobytes()).hexdigest() for track, value in anchors_by_track.items()}
    summary = summarize(rows, config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device=device, wordnet=wn.get_version(), host=host_config,
                       anchor_sha256=hashes, relation_counts=dict(counts),
                       dropped_same_token_edges=dropped_same_token)
    (output_dir / "report.md").write_text(render_report(summary, len(nodes), len(edges), counts))
    return summary


def _partition_rows(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    """Aggregate rows of one partition; rows without the key predate validation and are test."""
    return [row for row in rows if row["relation"] == "all" and row.get("partition", "test") == partition]


def _method_means(
    rows: list[dict[str, Any]], methods: list[str], metrics: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    means = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if selected:
            means[method] = {metric: sum(float(row[metric]) for row in selected) / len(selected)
                             for metric in metrics}
    return means


def _require_shuffled(methods: list[str], families: list[str]) -> None:
    missing = [f"shuffled_{family}" for family in families if f"shuffled_{family}" not in methods]
    if missing:
        raise ValueError(f"protocol 2 comparisons require a shuffled control for each compared family; missing {missing}")


def _supported(ci: dict[str, Any], wins: int, n: int) -> bool:
    """A paired CI above zero or a win on every predeclared split (design gate wording)."""
    return (ci["ci_low"] is not None and ci["ci_low"] > 0) or wins == n


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if config["acceptance"].get("mode") == "taxonomy_retrieval":
        return summarize_taxonomy_retrieval(rows, config)
    if "candidate_methods" in config["acceptance"]:
        return summarize_reconstruction_rescue(rows, config)
    v2 = int(config.get("protocol", 1)) >= 2
    acceptance = config["acceptance"]
    methods = [_method_name(item) for item in config["methods"]]
    overall = _partition_rows(rows, "test")
    means: dict[str, dict[str, dict[str, float]]] = {}
    for split in config["split_modes"]:
        means[split] = {}
        for track in config["tracks"]:
            means[split][track] = {}
            for method in methods:
                selected = [row for row in overall if row["split_mode"] == split and row["track"] == track and row["method"] == method]
                means[split][track][method] = sum(row["target_mrr"] for row in selected) / len(selected)
    primary_split, primary_track = acceptance["primary_split"], acceptance["primary_track"]
    scores = means[primary_split][primary_track]
    if "candidate" in acceptance:
        best_structured, selection = acceptance["candidate"], "named in config"
    elif v2:
        raise ValueError("protocol 2 requires acceptance.candidate, named before running")
    else:
        structured = [name for name in scores if name in STRUCTURED_FAMILIES]
        best_structured = max(structured, key=scores.get)
        selection = "test split (legacy; optimistic)"
    if v2:
        best_control = acceptance.get("primary_control", "additive")
        _require_shuffled(methods, [best_structured])
    else:
        controls = [name for name in scores if name not in STRUCTURED_FAMILIES]
        best_control = max(controls, key=scores.get)
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
    additive_ci = mean_confidence_interval([item["gain_over_additive"] for item in paired])
    shuffled_gains = [item["gain_over_matched_shuffled"] for item in paired if item["gain_over_matched_shuffled"] is not None]
    shuffled_ci = mean_confidence_interval(shuffled_gains) if shuffled_gains else None
    structured_rows = [row for row in primary if row["method"] == best_structured]
    additive_rows = [row for row in primary if row["method"] == "additive"]
    cosine_gain = (sum(row["cosine"] for row in structured_rows) - sum(row["cosine"] for row in additive_rows)) / len(structured_rows)
    n_seeds = len(config["seeds"])
    required_wins = int(acceptance.get("min_paired_wins", n_seeds))
    retrieval_pass = gain >= float(acceptance["min_mrr_gain"]) and paired_additive_wins >= required_wins and paired_shuffled_wins >= required_wins
    if v2:
        retrieval_pass = (
            retrieval_pass and _supported(additive_ci, paired_additive_wins, n_seeds)
            and shuffled_ci is not None and _supported(shuffled_ci, paired_shuffled_wins, n_seeds)
        )
    reconstruction_pass = cosine_gain >= float(acceptance.get("min_cosine_gain", 0.0))
    promotion_eligible = bool(acceptance.get("promotion_eligible", False))
    return {"conditions": len(rows), "protocol": int(config.get("protocol", 1)),
            "mean_target_mrr": means, "best_structured": best_structured,
            "candidate_selection": selection,
            "best_control": best_control, "matched_shuffled_control": matched_control,
            "primary_gain": gain, "primary_cosine_gain": cosine_gain,
            "gain_over_additive_ci": additive_ci, "gain_over_matched_shuffled_ci": shuffled_ci,
            "paired_results": paired, "paired_additive_wins": paired_additive_wins,
            "paired_shuffled_wins": paired_shuffled_wins,
            "retrieval_gate_passed": retrieval_pass, "reconstruction_gate_passed": reconstruction_pass,
            "proxy_criteria_only": True, "design_items_not_evaluated": list(DESIGN_ITEMS_01B),
            "promotion_eligible": promotion_eligible,
            "gate_passed": retrieval_pass and promotion_eligible}


def summarize_reconstruction_rescue(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Summarize 01c as reconstruction gain subject to retrieval non-inferiority."""
    v2 = int(config.get("protocol", 1)) >= 2
    acceptance = config["acceptance"]
    split, track = acceptance["primary_split"], acceptance["primary_track"]
    metric_names = ("cosine", "mse", "target_mrr", "target_recall_at_10")
    methods = [_method_name(item) for item in config["methods"]]
    def primary_rows(partition: str) -> list[dict[str, Any]]:
        return [row for row in _partition_rows(rows, partition)
                if row["split_mode"] == split and row["track"] == track]
    primary = primary_rows("test")
    mean_metrics = _method_means(primary, methods, metric_names)
    validation_metrics = _method_means(primary_rows("validation"), methods, metric_names)
    candidates = [name for name in acceptance["candidate_methods"] if name in mean_metrics]
    if not candidates:
        raise ValueError("01c acceptance.candidate_methods contains no configured method")
    reconstruction_baseline = acceptance.get("reconstruction_baseline", "offset")
    retrieval_baseline = acceptance.get("retrieval_baseline", "hrr")
    min_mrr_delta = float(acceptance.get("min_mrr_delta", 0.0))
    if v2 and len(candidates) == 1:
        selection_metrics, selection = mean_metrics, "named in config"
    elif v2:
        if not validation_metrics:
            raise ValueError("protocol 2 selects among several 01c candidates on validation; set data.validation_fraction > 0")
        selection_metrics, selection = validation_metrics, "validation partition"
    else:
        selection_metrics, selection = mean_metrics, "test split (legacy; optimistic)"
    eligible = [name for name in candidates if (
        selection_metrics[name]["target_mrr"] - selection_metrics[retrieval_baseline]["target_mrr"]
    ) >= min_mrr_delta]
    best = max(eligible or candidates, key=lambda name: selection_metrics[name]["cosine"])
    non_hrr_control = acceptance.get("non_hrr_control")
    if v2:
        _require_shuffled(methods, [best] + ([non_hrr_control] if non_hrr_control else []))
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
        control_row = by_seed.get((key, non_hrr_control)) if non_hrr_control else None
        paired.append({
            "seed": seed,
            "cosine_gain_over_reconstruction_baseline": candidate["cosine"] - reconstruction["cosine"],
            "mrr_delta_from_retrieval_baseline": candidate["target_mrr"] - retrieval["target_mrr"],
            "cosine_gain_over_matched_shuffled": (
                candidate["cosine"] - shuffled_row["cosine"] if shuffled_row else None
            ),
            "cosine_gain_over_non_hrr_control": (
                candidate["cosine"] - control_row["cosine"] if control_row else None
            ),
        })
    n_seeds = len(config["seeds"])
    required_wins = int(acceptance.get("min_paired_wins", n_seeds))
    reconstruction_wins = sum(item["cosine_gain_over_reconstruction_baseline"] > 0 for item in paired)
    retrieval_noninferior = sum(
        item["mrr_delta_from_retrieval_baseline"] >= min_mrr_delta
        for item in paired
    )
    shuffled_wins = sum(
        item["cosine_gain_over_matched_shuffled"] is not None
        and item["cosine_gain_over_matched_shuffled"] > 0 for item in paired
    )
    cosine_ci = mean_confidence_interval([item["cosine_gain_over_reconstruction_baseline"] for item in paired])
    control_gains = [item["cosine_gain_over_non_hrr_control"] for item in paired
                     if item["cosine_gain_over_non_hrr_control"] is not None]
    control_ci = mean_confidence_interval(control_gains) if control_gains else None
    require_shuffled = bool(acceptance.get("require_matched_shuffled", True))
    criteria_passed = (
        cosine_gain >= float(acceptance.get("min_cosine_gain", 0.0))
        and mrr_delta >= min_mrr_delta
        and reconstruction_wins >= required_wins
        and retrieval_noninferior >= required_wins
        and (not require_shuffled or (shuffled is not None and shuffled_wins >= required_wins))
    )
    if v2:
        control_wins = sum(gain > 0 for gain in control_gains)
        criteria_passed = (
            criteria_passed and _supported(cosine_ci, reconstruction_wins, n_seeds)
            and (control_ci is None or _supported(control_ci, control_wins, n_seeds))
        )
    not_evaluated = list(DESIGN_ITEMS_01C_NOT_EVALUATED)
    if control_ci is None:
        not_evaluated.insert(1, "3 (part) beats the equal-parameter non-HRR control")
    promotion_eligible = bool(acceptance.get("promotion_eligible", False))
    return {
        "experiment_stage": "01c",
        "protocol": int(config.get("protocol", 1)),
        "conditions": len(rows),
        "primary_split": split,
        "primary_track": track,
        "mean_metrics": mean_metrics,
        "validation_metrics": validation_metrics,
        "candidate_selection": selection,
        "best_candidate": best,
        "retrieval_eligible_candidates": eligible,
        "reconstruction_baseline": reconstruction_baseline,
        "retrieval_baseline": retrieval_baseline,
        "non_hrr_control": non_hrr_control,
        "matched_shuffled_control": shuffled_name if shuffled is not None else None,
        "cosine_gain_over_reconstruction_baseline": cosine_gain,
        "cosine_gain_over_reconstruction_baseline_ci": cosine_ci,
        "cosine_gain_over_non_hrr_control_ci": control_ci,
        "mrr_delta_from_retrieval_baseline": mrr_delta,
        "cosine_gain_over_matched_shuffled": shuffled_cosine_gain,
        "paired_results": paired,
        "paired_reconstruction_wins": reconstruction_wins,
        "paired_retrieval_noninferior": retrieval_noninferior,
        "paired_shuffled_wins": shuffled_wins,
        "exploratory_criteria_passed": criteria_passed,
        "proxy_criteria_only": True,
        "design_items_not_evaluated": not_evaluated,
        "promotion_eligible": promotion_eligible,
        "gate_passed": criteria_passed and promotion_eligible,
    }


def summarize_taxonomy_retrieval(
    rows: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    """Gate taxonomy HRR on retrieval versus matched non-HRR and shuffled controls."""
    v2 = int(config.get("protocol", 1)) >= 2
    acceptance = config["acceptance"]
    primary = [
        row for row in _partition_rows(rows, "test")
        if row["split_mode"] == acceptance["primary_split"]
        and row["track"] == acceptance["primary_track"]
    ]
    methods = [_method_name(item) for item in config["methods"]]
    mean_metrics = _method_means(primary, methods, (
        "cosine", "mse", "target_mrr", "target_recall_at_10",
        "distribution_mrr", "distribution_recall_at_10", "mean_positive_targets",
    ))
    candidate = acceptance["candidate"]
    control = acceptance["non_hrr_control"]
    shuffled = acceptance["shuffled_control"]
    if v2:
        _require_shuffled(methods, [candidate, control])
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
    n_seeds = len(config["seeds"])
    required = int(acceptance.get("min_paired_wins", n_seeds))
    mrr_gain = mean_metrics[candidate]["distribution_mrr"] - mean_metrics[control]["distribution_mrr"]
    shuffled_gain = mean_metrics[candidate]["distribution_mrr"] - mean_metrics[shuffled]["distribution_mrr"]
    cosine_delta = mean_metrics[candidate]["cosine"] - mean_metrics[reconstruction]["cosine"]
    control_wins = sum(item["mrr_gain_over_non_hrr"] > 0 for item in paired)
    shuffled_wins = sum(item["mrr_gain_over_shuffled"] > 0 for item in paired)
    locality_wins = sum(
        item["cosine_delta_from_reconstruction_baseline"] >= float(acceptance["min_cosine_delta"])
        for item in paired
    )
    control_ci = mean_confidence_interval([item["mrr_gain_over_non_hrr"] for item in paired])
    shuffled_ci = mean_confidence_interval([item["mrr_gain_over_shuffled"] for item in paired])
    passed = (
        mrr_gain >= float(acceptance["min_mrr_gain_over_non_hrr"])
        and shuffled_gain > 0
        and cosine_delta >= float(acceptance["min_cosine_delta"])
        and control_wins >= required and shuffled_wins >= required and locality_wins >= required
    )
    if v2:
        passed = (passed and _supported(control_ci, control_wins, n_seeds)
                  and _supported(shuffled_ci, shuffled_wins, n_seeds))
    promotion_eligible = bool(acceptance.get("promotion_eligible", False))
    return {
        "experiment_stage": "01c.4-taxonomy-retrieval",
        "protocol": int(config.get("protocol", 1)),
        "conditions": len(rows), "primary_split": acceptance["primary_split"],
        "primary_track": acceptance["primary_track"], "mean_metrics": mean_metrics,
        "candidate": candidate, "non_hrr_control": control, "shuffled_control": shuffled,
        "mrr_gain_over_non_hrr": mrr_gain, "mrr_gain_over_shuffled": shuffled_gain,
        "mrr_gain_over_non_hrr_ci": control_ci, "mrr_gain_over_shuffled_ci": shuffled_ci,
        "cosine_delta_from_reconstruction_baseline": cosine_delta,
        "paired_results": paired, "paired_non_hrr_wins": control_wins,
        "paired_shuffled_wins": shuffled_wins, "paired_locality_wins": locality_wins,
        "development_criteria_passed": passed, "proxy_criteria_only": True,
        "promotion_eligible": promotion_eligible,
        "gate_passed": passed and promotion_eligible,
    }


def _ci_text(ci: dict[str, Any] | None) -> str:
    if not ci or ci.get("ci_low") is None:
        return "n/a"
    return f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]"


def _not_evaluated_lines(summary: dict[str, Any]) -> list[str]:
    items = summary.get("design_items_not_evaluated") or []
    if not items:
        return []
    return ["", "Design-gate items this summarizer does **not** evaluate (the criteria above are proxies):", "",
            *[f"- {item}" for item in items], ""]


def render_report(summary: dict[str, Any], nodes: int, edges: int, counts: Counter[str]) -> str:
    if summary.get("experiment_stage") == "01c.4-taxonomy-retrieval":
        return render_taxonomy_retrieval_report(summary, nodes, edges, counts)
    if summary.get("experiment_stage") == "01c":
        return render_reconstruction_rescue_report(summary, nodes, edges, counts)
    lines = ["# Experiment 01b Stage B — heterogeneous WordNet / frozen GPT-2", "",
             f"Protocol: **{summary['protocol']}**. Nodes: **{nodes}**; edges: **{edges}**; relation counts: `{dict(counts)}`.", ""]
    for split, tracks in summary["mean_target_mrr"].items():
        for track, methods in tracks.items():
            lines += [f"## {split} / {track}", "", "| Method | Target MRR (test) |", "|---|---:|",
                      *[f"| {method} | {score:.4f} |" for method, score in methods.items()], ""]
    lines += ["## Stage-B proxy criteria", "",
              f"Candidate: **{summary['best_structured']}** (selection: {summary['candidate_selection']}); control: **{summary['best_control']}**; "
              f"primary MRR gain: **{summary['primary_gain']:+.4f}**.",
              f"Gain over additive per seed: 95% CI {_ci_text(summary['gain_over_additive_ci'])}; "
              f"over matched shuffled: {_ci_text(summary['gain_over_matched_shuffled_ci'])}.",
              f"Paired wins over additive: **{summary['paired_additive_wins']}**; over matched shuffled control: **{summary['paired_shuffled_wins']}**.",
              f"Primary cosine gain: **{summary['primary_cosine_gain']:+.4f}**.",
              f"Retrieval criteria: **{'PASS' if summary['retrieval_gate_passed'] else 'FAIL'}**; reconstruction criteria: **{'PASS' if summary['reconstruction_gate_passed'] else 'FAIL'}**; "
              f"promotion eligible: **{summary['promotion_eligible']}**.",
              *_not_evaluated_lines(summary),
              "This tests frozen representation transfer, not vocabulary insertion or generated-answer behavior.", ""]
    return "\n".join(lines)


def render_taxonomy_retrieval_report(
    summary: dict[str, Any], nodes: int, edges: int, counts: Counter[str],
) -> str:
    lines = ["# Experiment 01c.4 — taxonomy distributional retrieval", "",
             f"Protocol: **{summary['protocol']}**. Nodes: **{nodes}**; edges: **{edges}**; relation counts: `{dict(counts)}`.", "",
             "| Method | Distribution MRR | Distribution R@10 | Exact MRR | Cosine |", "|---|---:|---:|---:|---:|"]
    for method, metrics in summary["mean_metrics"].items():
        lines.append(
            f"| {method} | {metrics['distribution_mrr']:.4f} | "
            f"{metrics['distribution_recall_at_10']:.4f} | {metrics['target_mrr']:.4f} | "
            f"{metrics['cosine']:.4f} |"
        )
    lines += ["", f"Distribution-MRR gain over non-HRR control: **{summary['mrr_gain_over_non_hrr']:+.4f}** "
              f"(per-seed 95% CI {_ci_text(summary['mrr_gain_over_non_hrr_ci'])}).",
              f"Gain over shuffled control: **{summary['mrr_gain_over_shuffled']:+.4f}** "
              f"(95% CI {_ci_text(summary['mrr_gain_over_shuffled_ci'])}).",
              f"Cosine delta from offset: **{summary['cosine_delta_from_reconstruction_baseline']:+.4f}**.",
              f"Development criteria (proxy): **{'PASS' if summary['development_criteria_passed'] else 'FAIL'}**.", ""]
    return "\n".join(lines)


def render_reconstruction_rescue_report(
    summary: dict[str, Any], nodes: int, edges: int, counts: Counter[str],
) -> str:
    lines = ["# Experiment 01c — reconstruction rescue", "",
             f"Protocol: **{summary['protocol']}**. Nodes: **{nodes}**; edges: **{edges}**; relation counts: `{dict(counts)}`.", "",
             f"Primary condition: **{summary['primary_split']} / {summary['primary_track']}**.", "",
             "| Method | Cosine | MSE | Target MRR | Recall@10 |", "|---|---:|---:|---:|---:|"]
    for method, metrics in summary["mean_metrics"].items():
        lines.append(f"| {method} | {metrics['cosine']:.4f} | {metrics['mse']:.5f} | "
                     f"{metrics['target_mrr']:.4f} | {metrics['target_recall_at_10']:.4f} |")
    lines += ["", "## Exploratory joint criterion (proxy)", "",
              f"Best reconstruction candidate: **{summary['best_candidate']}** (selection: {summary['candidate_selection']}).",
              f"Cosine gain over **{summary['reconstruction_baseline']}**: "
              f"**{summary['cosine_gain_over_reconstruction_baseline']:+.4f}** "
              f"(per-seed 95% CI {_ci_text(summary['cosine_gain_over_reconstruction_baseline_ci'])}).",
              f"MRR delta from **{summary['retrieval_baseline']}**: "
              f"**{summary['mrr_delta_from_retrieval_baseline']:+.4f}**.",
              f"Exploratory criteria: **{'PASS' if summary['exploratory_criteria_passed'] else 'FAIL'}**; "
              f"promotion eligible: **{summary['promotion_eligible']}**; final gate: "
              f"**{'PASS' if summary['gate_passed'] else 'FAIL'}**.",
              *_not_evaluated_lines(summary),
              "A smoke result is not evidence for promotion. Full 01c requires development/confirmation "
              "separation, multiple seeds, target audits, behavioral locality, and a second ontology.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(yaml.safe_load(args.config.read_text()), args.output), indent=2))


if __name__ == "__main__": main()
