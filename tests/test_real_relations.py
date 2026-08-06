import torch
import pytest

from vsa_embed.real_relations import (
    HostEdges, HostRelationModel, fit_host_relations, multi_positive_target_retrieval_metrics,
    source_neighborhood_metrics, target_retrieval_metrics,
)


def test_target_retrieval_is_exact_for_perfect_predictions() -> None:
    anchors = torch.eye(5)
    ids = torch.tensor([1, 3, 4])
    metrics = target_retrieval_metrics(anchors[ids], ids, anchors)
    assert metrics["target_mrr"] == 1.0
    assert metrics["target_recall_at_10"] == 1.0
    assert metrics["target_candidates"] == 3


def test_target_retrieval_can_use_a_shared_candidate_universe() -> None:
    anchors = torch.eye(5)
    metrics = target_retrieval_metrics(anchors[[1]], torch.tensor([1]), anchors, torch.tensor([0, 1, 2, 3]))
    assert metrics["target_mrr"] == 1.0
    assert metrics["target_candidates"] == 4


def test_multi_positive_retrieval_accepts_any_target_for_source_relation() -> None:
    anchors = torch.eye(5)
    prediction = torch.stack((anchors[2], anchors[2], anchors[4]))
    metrics = multi_positive_target_retrieval_metrics(
        prediction, torch.tensor([0, 0, 1]), torch.tensor([1, 2, 4]),
        torch.tensor([0, 0, 0]), anchors,
    )
    assert metrics["distribution_mrr"] == 1.0
    assert metrics["mean_positive_targets"] == pytest.approx(5 / 3)


def test_source_neighborhood_metrics_score_complete_positive_sets() -> None:
    anchors = torch.eye(6)
    source_ids = torch.tensor([0, 0, 1, 1])
    target_ids = torch.tensor([2, 3, 4, 5])
    prediction = torch.stack((anchors[2] + anchors[3], anchors[2] + anchors[3],
                              anchors[4] + anchors[5], anchors[4] + anchors[5]))
    metrics = source_neighborhood_metrics(
        prediction, source_ids, target_ids, torch.zeros(4, dtype=torch.long), anchors,
    )
    assert metrics["query_mrr"] == 1.0
    assert metrics["set_recall_at_10"] == 1.0
    assert metrics["mean_targets_per_query"] == 2.0


def test_offset_model_learns_relation_specific_transfer() -> None:
    torch.manual_seed(4)
    source = torch.randn(100, 8)
    relation_ids = torch.arange(100) % 2
    offsets = torch.stack((torch.ones(8), -torch.ones(8))) * 0.4
    targets = torch.nn.functional.normalize(source + offsets[relation_ids], dim=-1)
    data = HostEdges(source, targets, relation_ids)
    model = HostRelationModel(2, 8, "offset")
    initial, final = fit_host_relations(model, data, steps=150, learning_rate=0.04, cosine_weight=1.0)
    assert final < initial * 0.05


def test_additive_control_requires_no_optimizer_parameters() -> None:
    source = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    data = HostEdges(source, source.clone(), torch.arange(8) % 2)
    model = HostRelationModel(2, 4, "additive")
    initial, final = fit_host_relations(model, data, steps=5, learning_rate=0.1, cosine_weight=1.0)
    assert abs(initial) < 1e-6 and initial == final


def test_residual_hrr_families_are_identity_initialized() -> None:
    source = torch.nn.functional.normalize(torch.randn(12, 8), dim=-1)
    relation_ids = torch.arange(12) % 3
    for family in (
        "residual_hrr", "offset_residual_hrr", "gated_offset_residual_hrr",
        "basis_offset_residual_hrr",
    ):
        model = HostRelationModel(3, 8, family)
        assert torch.allclose(model(source, relation_ids), source, atol=1e-6)
        diagnostics = model.diagnostics(source, relation_ids)
        assert diagnostics["mean_abs_hrr_coefficient"] == 0.0
        assert diagnostics["mean_hrr_correction_norm"] == 0.0


def test_offset_residual_hrr_recovers_offset_and_structured_teacher() -> None:
    torch.manual_seed(9)
    source = torch.nn.functional.normalize(torch.randn(160, 12), dim=-1)
    relation_ids = torch.arange(160) % 2
    teacher = HostRelationModel(2, 12, "offset_residual_hrr")
    with torch.no_grad():
        assert teacher.residual_transform is not None
        teacher.residual_transform.offset.copy_(torch.randn(2, 12) * 0.15)
        teacher.residual_transform.residual_scale_logits.copy_(torch.tensor([1.5, -1.0]))
        targets = teacher(source, relation_ids)
    model = HostRelationModel(2, 12, "offset_residual_hrr")
    initial, final = fit_host_relations(model, HostEdges(source, targets, relation_ids),
                                        steps=250, learning_rate=0.03, cosine_weight=1.0)
    assert final < initial * 0.1
    assert model.diagnostics(source, relation_ids)["mean_abs_hrr_coefficient"] > 0.01


def test_residual_hrr_correction_is_bounded() -> None:
    source = torch.nn.functional.normalize(torch.randn(20, 16), dim=-1)
    relation_ids = torch.arange(20) % 2
    model = HostRelationModel(2, 16, "offset_residual_hrr", max_residual_scale=0.1)
    assert model.residual_transform is not None
    with torch.no_grad():
        model.residual_transform.residual_scale_logits.fill_(100.0)
    parts = model.residual_transform.components(source, relation_ids)
    assert torch.all(parts["correction"].norm(dim=-1) <= 0.100001)


def test_basis_residual_is_bounded_after_decode() -> None:
    source = torch.nn.functional.normalize(torch.randn(20, 16), dim=-1)
    relation_ids = torch.arange(20) % 2
    model = HostRelationModel(
        2, 16, "basis_offset_residual_hrr", max_residual_scale=0.1,
        max_log_basis_scale=0.2,
    )
    assert model.residual_transform is not None
    with torch.no_grad():
        model.residual_transform.residual_scale_logits.fill_(100.0)
        model.residual_transform.log_basis_scale.copy_(torch.linspace(-100, 100, 16))
    parts = model.residual_transform.components(source, relation_ids)
    assert torch.all(parts["correction"].norm(dim=-1) <= 0.100001)
    effective_log_scale = 0.2 * torch.tanh(model.residual_transform.log_basis_scale)
    assert float(effective_log_scale.detach().abs().max()) <= 0.200001


def test_multi_positive_rank_loss_learns_one_to_many_targets() -> None:
    torch.manual_seed(17)
    sources = torch.nn.functional.normalize(torch.randn(8, 6), dim=-1)
    sources = sources.repeat_interleave(2, dim=0)
    source_ids = torch.arange(8).repeat_interleave(2)
    relation_ids = torch.zeros(16, dtype=torch.long)
    targets = torch.nn.functional.normalize(sources + 0.2 * torch.randn(16, 6), dim=-1)
    data = HostEdges(sources, targets, relation_ids, source_ids, torch.arange(16))
    model = HostRelationModel(1, 6, "offset_residual_hrr")
    initial, final = fit_host_relations(
        model, data, steps=80, learning_rate=0.03, cosine_weight=0.1,
        rank_weight=1.0, rank_temperature=0.1,
    )
    assert final < initial
