import torch

from vsa_embed.global_local import (
    GlobalLocalRelationalModel,
    balanced_prefix,
    fit_global_local,
    make_synthetic_relations,
    relational_metrics,
)


def test_synthetic_split_is_disjoint_balanced_and_nested() -> None:
    train, test, _ = make_synthetic_relations(
        family="hrr", relation_count=3, dimension=8, train_per_relation=12,
        test_per_relation=5, seed=4,
    )
    assert len(train) == 36 and len(test) == 15
    assert [int((train.relation_ids == r).sum()) for r in range(3)] == [12, 12, 12]
    small = balanced_prefix(train, 4, 3)
    large = balanced_prefix(train, 8, 3)
    for field in ("sources", "relation_ids", "edge_features", "targets"):
        small_field, large_field = getattr(small, field), getattr(large, field)
        for relation in range(3):
            assert torch.equal(small_field[small.relation_ids == relation], large_field[large.relation_ids == relation][:4])


def test_model_has_no_edge_or_concept_specific_parameters() -> None:
    model = GlobalLocalRelationalModel(3, 8, family="low_rank", rank=2, residual_dimension=3)
    groups = model.parameter_groups()
    assert groups["edge_specific"] == 0
    assert groups["concept_specific"] == 0
    assert not any("edge_embedding" in name or "concept_embedding" in name for name, _ in model.named_parameters())


def test_non_additive_teachers_are_not_identity_controls() -> None:
    for family in ("hrr", "diagonal", "low_rank"):
        train, _, teacher = make_synthetic_relations(
            family=family, relation_count=3, dimension=8, train_per_relation=8,
            test_per_relation=4, rank=2, seed=12,
        )
        transformed = teacher(train.relation_ids, train.sources)
        assert (transformed - train.sources).square().mean() > 1e-3
        relation_means = torch.stack([
            teacher(torch.full_like(train.relation_ids, relation), train.sources)
            for relation in range(3)
        ])
        assert relation_means.var(dim=0).mean() > 1e-4


def test_aligned_hrr_recovers_heldout_edges() -> None:
    train, test, _ = make_synthetic_relations(
        family="hrr", relation_count=2, dimension=8, train_per_relation=48,
        test_per_relation=32, seed=7,
    )
    torch.manual_seed(107)
    model = GlobalLocalRelationalModel(2, 8, family="hrr")
    before = relational_metrics(model, test)["mse"]
    fit_global_local(model, train, steps=250, learning_rate=0.04)
    after = relational_metrics(model, test)
    assert after["mse"] < before * 0.1
    assert after["weight_correlation"] > 0.95
    assert after["relation_accuracy"] > 0.9


def test_restricted_residual_is_transferable() -> None:
    train, test, _ = make_synthetic_relations(
        family="diagonal", relation_count=2, dimension=8, train_per_relation=64,
        test_per_relation=32, residual_dimension=3, seed=9,
    )
    model = GlobalLocalRelationalModel(2, 8, family="diagonal", residual_dimension=3)
    fit_global_local(model, train, steps=300, learning_rate=0.03)
    assert relational_metrics(model, test)["mse"] < 0.01
