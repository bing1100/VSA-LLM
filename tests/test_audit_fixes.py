"""Regression tests for the implementation-audit fixes (resources/plan-improvement/audit.md)."""

import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from vsa_embed.algebra import UnitaryHRRAlgebra
from vsa_embed.confidence import precision_threshold
from vsa_embed.experiments import wordnet_relations as experiment
from vsa_embed.experiments.hierarchy_neighborhoods import summarize as summarize_hierarchy
from vsa_embed.global_local import make_synthetic_relations
from vsa_embed.provenance import MANIFEST_SCHEMA_VERSION, prepare_output_dir, write_run_metadata
from vsa_embed.real_relations import (
    HostEdges, HostRelationModel, derange_labels, host_relation_metrics, info_nce_rank_loss,
    multi_positive_target_retrieval_metrics, rank_positive_mask, source_neighborhood_metrics,
    target_retrieval_metrics,
)
from vsa_embed.relations import create_relation_transform
from vsa_embed.statistics import (
    mean_confidence_interval, paired_bootstrap_ci, positive_ranks, wilson_interval,
)
from vsa_embed.wordnet_relations import (
    SYMMETRIC_RELATIONS, RelationEdge, edge_group, split_edge_partitions, split_edges,
    split_neighborhood_partitions, split_source_neighborhoods,
)


# F5d: ties and degenerate predictions ------------------------------------------------------

def test_constant_prediction_no_longer_gets_perfect_mrr() -> None:
    anchors = torch.eye(4)
    ids = torch.tensor([0, 1, 2, 3])
    constant = torch.ones(4, 4)
    optimistic = target_retrieval_metrics(constant, ids, anchors, ties="optimistic")
    mid = target_retrieval_metrics(constant, ids, anchors, ties="mid")
    assert optimistic["target_mrr"] == 1.0
    assert mid["target_mrr"] == pytest.approx(1 / 2.5)  # rank 1 + 3 tied negatives / 2


def test_mid_rank_matches_expected_rank_under_random_tie_breaking() -> None:
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.1]])
    positives = torch.tensor([[True, False, False, False]])
    assert float(positive_ranks(scores, positives, ties="mid")[0]) == 2.0


def test_non_finite_predictions_are_rejected() -> None:
    anchors = torch.eye(3)
    with pytest.raises(ValueError, match="NaN"):
        target_retrieval_metrics(torch.full((1, 3), float("nan")), torch.tensor([0]), anchors, ties="mid")


# F5b: shared candidate sets ------------------------------------------------------------------

def test_multi_positive_subset_uses_the_shared_candidate_universe() -> None:
    anchors = torch.eye(6)
    prediction = anchors[[1, 4]] + 0.1
    sources, targets, relations = torch.tensor([0, 3]), torch.tensor([1, 4]), torch.tensor([0, 1])
    universe = torch.tensor([1, 2, 4, 5])
    restricted = multi_positive_target_retrieval_metrics(prediction[:1], sources[:1], targets[:1], relations[:1], anchors)
    shared = multi_positive_target_retrieval_metrics(
        prediction[:1], sources[:1], targets[:1], relations[:1], anchors, universe, ties="mid",
    )
    assert restricted["distribution_mrr"] == 1.0  # a single candidate: trivially rank 1
    assert shared["distribution_mrr"] == 1.0
    worse = multi_positive_target_retrieval_metrics(
        anchors[[5]], sources[:1], targets[:1], relations[:1], anchors, universe, ties="mid",
    )
    assert worse["distribution_mrr"] < 1.0


# F5a: relation accuracy ------------------------------------------------------------------------

def test_relation_accuracy_is_undefined_for_parameter_free_models_and_tie_aware() -> None:
    sources = F.normalize(torch.randn(10, 6), dim=-1)
    data = HostEdges(sources, sources, torch.zeros(10, dtype=torch.long))
    additive = HostRelationModel(3, 6, "additive")
    assert host_relation_metrics(additive, data)["relation_accuracy"] == 1.0  # legacy: share of relation 0
    assert math.isnan(host_relation_metrics(additive, data, ties="mid")["relation_accuracy"])
    offset = HostRelationModel(3, 6, "offset")  # zero offsets: all relations tie at init
    assert host_relation_metrics(offset, data, ties="mid")["relation_accuracy"] == pytest.approx(1 / 3)


# F7: rank-loss positives -------------------------------------------------------------------------

def test_rank_loss_is_near_zero_for_a_perfect_predictor_with_duplicate_targets() -> None:
    anchors = torch.eye(8)
    source_ids = torch.tensor([0, 1, 2, 3])
    target_ids = torch.tensor([5, 5, 5, 6])  # target 5 has multiplicity 3
    data = HostEdges(anchors[source_ids], anchors[target_ids], torch.zeros(4, dtype=torch.long),
                     source_ids, target_ids)
    prediction = anchors[target_ids]
    legacy = info_nce_rank_loss(prediction, data.targets, rank_positive_mask(data, "source_relation"), 0.07)
    fixed = info_nce_rank_loss(prediction, data.targets, rank_positive_mask(data, "target_set"), 0.07)
    assert float(legacy) == pytest.approx(0.75 * math.log(3), abs=1e-4)
    assert float(fixed) < 1e-4


# A1 splits ---------------------------------------------------------------------------------------

def _graph() -> list[RelationEdge]:
    return [RelationEdge(f"n{i}", "hypernym" if i % 3 else "attribute", f"n{(i * 7 + 3) % 40}")
            for i in range(40)]


@pytest.mark.parametrize("mode", ["edge_disjoint", "node_disjoint"])
def test_partitions_without_validation_reproduce_the_legacy_split(mode: str) -> None:
    edges = _graph()
    train, test, discarded = split_edges(edges, mode=mode, test_fraction=0.25, seed=11)
    split = split_edge_partitions(edges, mode=mode, test_fraction=0.25, seed=11)
    assert (sorted(split.train), split.test, split.discarded) == (sorted(train), sorted(test), sorted(discarded))


@pytest.mark.parametrize("mode", ["edge_disjoint", "node_disjoint"])
def test_validation_partition_keeps_the_test_partition_and_is_disjoint(mode: str) -> None:
    edges = _graph()
    base = split_edge_partitions(edges, mode=mode, test_fraction=0.25, seed=11)
    split = split_edge_partitions(edges, mode=mode, test_fraction=0.25, seed=11, validation_fraction=0.2)
    assert split.test == base.test and split.validation
    parts = [set(ids) for ids in split.partitions().values()]
    assert sum(map(len, parts)) == len(edges) and len(set().union(*parts)) == len(edges)
    if mode == "node_disjoint":
        nodes = lambda ids: {node for i in ids for node in (edges[i].source, edges[i].target)}
        assert nodes(split.train).isdisjoint(nodes(split.validation) | nodes(split.test))
    assert split.stats["validation_edges"] == len(split.validation)


def test_token_disjoint_split_keeps_shared_tokens_on_one_side() -> None:
    edges = _graph()
    token_of = {f"n{i}": i // 2 for i in range(40)}  # pairs of nodes share a host token
    split = split_edge_partitions(edges, mode="token_disjoint", test_fraction=0.25, seed=5,
                                  node_groups=token_of)
    tokens = lambda ids: {token_of[node] for i in ids for node in (edges[i].source, edges[i].target)}
    assert tokens(split.train).isdisjoint(tokens(split.test))


def test_symmetric_attribute_edges_share_a_group() -> None:
    forward = RelationEdge("hot.n.01", "attribute", "hot.a.01")
    reverse = RelationEdge("hot.a.01", "attribute", "hot.n.01")
    assert forward.group != reverse.group  # legacy grouping, kept for replay
    assert edge_group(forward, SYMMETRIC_RELATIONS) == edge_group(reverse, SYMMETRIC_RELATIONS)


def test_target_disjoint_neighborhoods_never_train_on_a_test_parent() -> None:
    edges = [RelationEdge(f"child{i}", "hypernym", f"parent{i % 5}") for i in range(40)]
    edges += [RelationEdge(f"child{i}", "hypernym", f"parent{(i + 1) % 5}") for i in range(40)]
    _, legacy_test = split_source_neighborhoods(edges, test_fraction=0.25, seed=3)
    split = split_neighborhood_partitions(edges, test_fraction=0.25, seed=3, validation_fraction=0.1)
    assert split.test == sorted(legacy_test)
    train_nodes = {node for i in split.train for node in (edges[i].source, edges[i].target)}
    test_nodes = {node for i in split.test for node in (edges[i].source, edges[i].target)}
    assert train_nodes.isdisjoint(test_nodes)
    assert split.stats["discarded_edges"] == len(split.discarded) > 0


def test_neighborhood_metrics_report_seen_and_unseen_parents() -> None:
    anchors = torch.eye(6)
    metrics = source_neighborhood_metrics(
        anchors[[2, 4]], torch.tensor([0, 1]), torch.tensor([2, 4]), torch.zeros(2, dtype=torch.long),
        anchors, ties="mid", seen_node_ids=torch.tensor([2]),
    )
    assert metrics["queries_seen"] == 1 and metrics["queries_unseen"] == 1
    assert metrics["query_mrr_seen"] == 1.0 and metrics["query_mrr_unseen"] == 1.0


# A1 selection on validation only ---------------------------------------------------------------

def _rescue_rows(partition: str, values: dict[str, tuple[float, float]]) -> list[dict]:
    return [{"seed": seed, "split_mode": "node_disjoint", "track": "contextual_centered",
             "partition": partition, "method": method, "relation": "all", "cosine": cosine,
             "mse": 0.1, "target_mrr": mrr, "target_recall_at_10": 0.5}
            for seed in (1, 2) for method, (cosine, mrr) in values.items()]


def test_protocol_2_selects_the_rescue_candidate_on_validation_only() -> None:
    methods = ["offset", "hrr", "a", "b", "shuffled_a", "shuffled_b"]
    validation = {"offset": (0.4, 0.2), "hrr": (0.3, 0.3), "a": (0.50, 0.3), "b": (0.45, 0.3),
                  "shuffled_a": (0.3, 0.2), "shuffled_b": (0.3, 0.2)}
    test = {**validation, "a": (0.41, 0.3), "b": (0.60, 0.3)}  # the test split favours b
    config = {"protocol": 2, "methods": methods, "seeds": [1, 2], "acceptance": {
        "primary_split": "node_disjoint", "primary_track": "contextual_centered",
        "candidate_methods": ["a", "b"], "reconstruction_baseline": "offset",
        "retrieval_baseline": "hrr", "min_mrr_delta": -0.02, "min_paired_wins": 0,
    }}
    summary = experiment.summarize(_rescue_rows("validation", validation) + _rescue_rows("test", test), config)
    assert summary["best_candidate"] == "a"
    assert summary["candidate_selection"] == "validation partition"


def test_protocol_2_requires_a_shuffled_control_and_a_named_candidate() -> None:
    rows = [{"seed": 1, "split_mode": "edge_disjoint", "track": "contextual_centered", "partition": "test",
             "method": method, "relation": "all", "target_mrr": 0.3, "cosine": 0.2}
            for method in ("additive", "hrr")]
    config = {"protocol": 2, "methods": ["additive", "hrr"], "seeds": [1],
              "split_modes": ["edge_disjoint"], "tracks": ["contextual_centered"],
              "acceptance": {"primary_split": "edge_disjoint", "primary_track": "contextual_centered",
                             "min_mrr_gain": 0.0}}
    with pytest.raises(ValueError, match="candidate"):
        experiment.summarize(rows, config)
    config["acceptance"]["candidate"] = "hrr"
    with pytest.raises(ValueError, match="shuffled"):
        experiment.summarize(rows, config)


def test_hierarchy_protocol_2_requires_shuffled_controls_for_both_families() -> None:
    methods = ["hrr_family", "control_family", "shuffled_hrr_family"]
    rows = [{"seed": 1, "method": method, "query_mrr": 0.3, "set_recall_at_10": 0.3,
             "set_hit_at_10": 0.5, "set_mass_nll": 1.0, "target_centroid_cosine": 0.1,
             "mean_targets_per_query": 2.0} for method in methods]
    config = {"protocol": 2, "methods": methods, "seeds": [1], "acceptance": {
        "candidate": methods[0], "non_hrr_control": methods[1], "shuffled_control": methods[2],
        "min_mrr_gain_over_non_hrr": 0.0, "min_recall_gain_over_non_hrr": 0.0}}
    with pytest.raises(ValueError, match="shuffled_control_family"):
        summarize_hierarchy(rows, config)


# F8/F9 controls ------------------------------------------------------------------------------

def test_derangement_changes_nearly_every_label_and_keeps_the_multiset() -> None:
    labels = torch.tensor([0] * 50 + [1] * 30 + [2] * 20)
    shuffled = derange_labels(labels, torch.Generator().manual_seed(1))
    assert (shuffled != labels).float().mean() >= 0.95
    assert torch.equal(torch.bincount(shuffled), torch.bincount(labels))


def test_rotated_diagonal_control_uses_its_basis_like_hrr_does() -> None:
    source = F.normalize(torch.randn(16, 12), dim=-1)
    relation_ids = torch.arange(16) % 2
    def output_change(family: str) -> float:
        torch.manual_seed(0)
        model = HostRelationModel(2, 12, family)
        transform = model.residual_transform
        with torch.no_grad():
            transform.residual_scale_logits.fill_(2.0)
            before = transform.components(source, relation_ids)["correction"]
            transform.log_basis_scale.copy_(torch.linspace(-2, 2, 12))
            after = transform.components(source, relation_ids)["correction"]
        return float((after - before).norm())
    assert output_change("basis_offset_diagonal_control") < 1e-5  # the basis cancels (audit F8)
    assert output_change("basis_offset_rotated_diagonal_control") > 1e-3
    assert output_change("basis_offset_residual_hrr") > 1e-3
    count = lambda family: sum(p.numel() for p in HostRelationModel(2, 12, family).parameters())
    assert count("basis_offset_rotated_diagonal_control") == count("basis_offset_residual_hrr")


def test_low_rank_matched_does_not_exceed_the_reference_parameters() -> None:
    reference = sum(p.numel() for p in HostRelationModel(7, 64, "hrr").parameters())
    matched = HostRelationModel(7, 64, "low_rank_matched", match_parameters=reference)
    count = sum(p.numel() for p in matched.parameters())
    assert reference - 7 * 65 < count <= reference + 7  # rank-1 tied: d + 1 per relation
    x = F.normalize(torch.randn(5, 64), dim=-1)
    torch.testing.assert_close(matched(x, torch.zeros(5, dtype=torch.long)), x)
    with pytest.raises(ValueError, match="match_parameters"):
        HostRelationModel(7, 64, "low_rank_matched")


@pytest.mark.parametrize("family", ["hrr_identity", "low_rank_identity", "low_rank_tied"])
def test_identity_initialized_families_start_at_the_identity(family: str) -> None:
    transform = create_relation_transform(family, 3, 32, rank=4)
    x = F.normalize(torch.randn(6, 32), dim=-1)
    y = transform(torch.arange(6) % 3, x).detach()
    assert float(F.cosine_similarity(x, y).min()) > 0.98


# F12 library nits ------------------------------------------------------------------------------

def test_unitary_role_from_bipolar_input_keeps_every_frequency() -> None:
    bipolar = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    role = UnitaryHRRAlgebra.make_role(bipolar)
    magnitude = torch.fft.rfft(role).abs()
    torch.testing.assert_close(magnitude, torch.full_like(magnitude, float(magnitude[0])))


def test_synthetic_relations_do_not_touch_the_global_rng() -> None:
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    make_synthetic_relations(family="hrr", relation_count=2, dimension=8, train_per_relation=4,
                             test_per_relation=2, seed=5)
    torch.testing.assert_close(torch.rand(3), expected)


def test_gloss_truncation_keeps_the_lemma_as_the_last_token() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not cached")
    from vsa_embed.wordnet_relations import SenseNode, _gloss_truncated_prompt
    token_id = tokenizer.encode(" church", add_special_tokens=False)[0]
    node = SenseNode("church.n.01", "church", token_id, "a long gloss " * 60, "n")
    ids = tokenizer.encode(_gloss_truncated_prompt(tokenizer, node, 64), add_special_tokens=False)
    assert len(ids) <= 64 and ids[-1] == token_id


# A3 provenance, A4 statistics ------------------------------------------------------------------

def test_run_directory_is_never_overwritten(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepare_output_dir(run_dir)
    manifest = write_run_metadata(run_dir, {"a": 1}, device="cpu")
    for key in ("git_sha", "git_dirty", "packages", "device", "argv", "config_sha256"):
        assert key in manifest
    assert json.loads((run_dir / "manifest.json").read_text())["schema_version"] == MANIFEST_SCHEMA_VERSION
    with pytest.raises(FileExistsError):
        prepare_output_dir(run_dir)


def test_seed_confidence_interval_reproduces_the_audit_numbers() -> None:
    ci = mean_confidence_interval([0.0162, 0.0168, 0.0251])
    assert ci["ci_low"] == pytest.approx(0.0070, abs=5e-4)
    assert ci["ci_high"] == pytest.approx(0.0317, abs=5e-4)


def test_wilson_interval_and_bootstrap_bracket_the_estimate() -> None:
    low, high = wilson_interval(19, 20)
    assert low < 0.95 < high
    ci = paired_bootstrap_ci([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5], seed=1)
    assert ci["ci_low"] == pytest.approx(0.5) and ci["ci_high"] == pytest.approx(0.5)


def test_wilson_threshold_is_more_conservative_than_the_point_rule() -> None:
    confidence = torch.linspace(1.0, 0.0, 200)
    correct = torch.rand(200, generator=torch.Generator().manual_seed(2)) < torch.linspace(1.0, 0.6, 200)
    _, point_coverage, _ = precision_threshold(confidence, correct, 0.9)
    _, lower_coverage, lower_accuracy = precision_threshold(confidence, correct, 0.9, method="wilson_lower")
    assert lower_coverage < point_coverage
    assert lower_accuracy >= 0.9
