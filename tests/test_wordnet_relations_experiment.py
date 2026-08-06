from pathlib import Path

import torch
from torch import nn

from vsa_embed.experiments import wordnet_relations as experiment
from vsa_embed.wordnet_relations import RelationEdge, SenseNode


class FakeHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.randn(20, 8))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


def test_stage_b_runner_writes_auditable_artifacts(tmp_path: Path, monkeypatch) -> None:
    nodes = [SenseNode(f"n{i}", f"word{i}", i, f"definition {i}", "n") for i in range(12)]
    edges = [RelationEdge(f"n{i}", "hypernym" if i % 2 == 0 else "antonym", f"n{(i + 3) % 12}") for i in range(12)]
    monkeypatch.setattr(experiment.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(experiment.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeHost())
    monkeypatch.setattr(experiment, "collect_wordnet_graph", lambda *a, **k: (nodes, edges))
    contextual = torch.nn.functional.normalize(torch.randn(12, 8), dim=-1)
    monkeypatch.setattr(experiment, "contextual_anchors", lambda *a, **k: contextual)
    monkeypatch.setattr(experiment.wn, "get_version", lambda: "fixture")
    config = {
        "host": {"model": "fake", "revision": "fixed", "batch_size": 4, "max_length": 16},
        "data": {"relation_types": ["hypernym", "antonym"], "max_edges_per_relation": 10,
                 "min_edges_per_relation": 5, "selection_seed": 1, "test_fraction": 0.25},
        "tracks": ["static_centered", "contextual_centered"], "split_modes": ["edge_disjoint"],
        "methods": ["additive", "relation_mean", "offset"], "seeds": [3],
        "fit": {"rank": 2, "steps": 20, "learning_rate": 0.05, "cosine_weight": 1.0},
        "acceptance": {"primary_split": "edge_disjoint", "primary_track": "contextual_centered",
                       "min_mrr_gain": -1.0, "min_paired_wins": 0},
        "device": "cpu",
    }
    result = experiment.run(config, tmp_path)
    assert result["gate_passed"]
    for name in ("metrics.csv", "splits.csv", "nodes.csv", "edges.csv", "relation_models.pt",
                 "summary.json", "resolved_config.yaml", "manifest.json", "report.md"):
        assert (tmp_path / name).is_file()


def test_01c_summary_requires_reconstruction_and_retrieval() -> None:
    methods = ["offset", "hrr", "offset_residual_hrr", "shuffled_offset_residual_hrr"]
    rows = []
    values = {
        "offset": (0.40, 0.20), "hrr": (0.30, 0.28),
        "offset_residual_hrr": (0.43, 0.27), "shuffled_offset_residual_hrr": (0.35, 0.19),
    }
    for seed in (1, 2):
        for method, (cosine, mrr) in values.items():
            rows.append({"seed": seed, "split_mode": "node_disjoint", "track": "contextual_centered",
                         "method": method, "relation": "all", "cosine": cosine, "mse": 0.1,
                         "target_mrr": mrr, "target_recall_at_10": 0.5,
                         "distribution_mrr": mrr, "distribution_recall_at_10": 0.5,
                         "mean_positive_targets": 1.5})
    config = {"methods": methods, "seeds": [1, 2], "acceptance": {
        "primary_split": "node_disjoint", "primary_track": "contextual_centered",
        "candidate_methods": ["offset_residual_hrr"], "reconstruction_baseline": "offset",
        "retrieval_baseline": "hrr", "min_cosine_gain": 0.02, "min_mrr_delta": -0.02,
        "min_paired_wins": 2, "require_matched_shuffled": True,
    }}
    result = experiment.summarize(rows, config)
    assert result["gate_passed"]
    assert result["best_candidate"] == "offset_residual_hrr"


def test_01c_selection_filters_by_retrieval_before_cosine() -> None:
    methods = ["offset", "hrr", "high_cosine", "pareto"]
    values = {"offset": (0.40, 0.20), "hrr": (0.30, 0.30),
              "high_cosine": (0.50, 0.20), "pareto": (0.43, 0.29)}
    rows = [{"seed": 1, "split_mode": "node_disjoint", "track": "contextual_centered",
             "method": method, "relation": "all", "cosine": cosine, "mse": 0.1,
             "target_mrr": mrr, "target_recall_at_10": 0.5}
            for method, (cosine, mrr) in values.items()]
    config = {"methods": methods, "seeds": [1], "acceptance": {
        "primary_split": "node_disjoint", "primary_track": "contextual_centered",
        "candidate_methods": ["high_cosine", "pareto"], "reconstruction_baseline": "offset",
        "retrieval_baseline": "hrr", "min_mrr_delta": -0.02,
        "min_paired_wins": 0, "require_matched_shuffled": False,
    }}
    result = experiment.summarize(rows, config)
    assert result["best_candidate"] == "pareto"
    assert result["retrieval_eligible_candidates"] == ["pareto"]


def test_taxonomy_summary_compares_hrr_to_non_hrr_and_shuffled() -> None:
    methods = ["offset", "basis_offset_residual_hrr", "basis_offset_diagonal_control",
               "shuffled_basis_offset_residual_hrr"]
    rows = []
    values = {"offset": (0.40, 0.20), "basis_offset_residual_hrr": (0.39, 0.32),
              "basis_offset_diagonal_control": (0.39, 0.29),
              "shuffled_basis_offset_residual_hrr": (0.25, 0.18)}
    for seed in (1, 2):
        for method, (cosine, mrr) in values.items():
            rows.append({"seed": seed, "split_mode": "node_disjoint", "track": "contextual_centered",
                         "method": method, "relation": "all", "cosine": cosine, "mse": 0.1,
                         "target_mrr": mrr, "target_recall_at_10": 0.5,
                         "distribution_mrr": mrr, "distribution_recall_at_10": 0.5,
                         "mean_positive_targets": 1.5})
    config = {"methods": methods, "seeds": [1, 2], "acceptance": {
        "mode": "taxonomy_retrieval", "primary_split": "node_disjoint",
        "primary_track": "contextual_centered", "candidate": "basis_offset_residual_hrr",
        "non_hrr_control": "basis_offset_diagonal_control",
        "shuffled_control": "shuffled_basis_offset_residual_hrr",
        "reconstruction_baseline": "offset", "min_mrr_gain_over_non_hrr": 0.01,
        "min_cosine_delta": -0.02, "min_paired_wins": 2,
    }}
    result = experiment.summarize(rows, config)
    assert result["development_criteria_passed"]
    assert result["paired_non_hrr_wins"] == 2
