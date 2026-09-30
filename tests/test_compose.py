import pytest
import torch
from torch.nn import functional as F

from vsa_embed.compose import FrameComposer, FrameSchedule, segment_softmax, uniform_bundle


def _schedule() -> FrameSchedule:
    return FrameSchedule.from_frames([
        [(0, 1), (1, 2), (2, 3)],
        [(0, 4), (1, 1)],
        [(2, 0), (0, 5), (1, 3), (2, 2)],
        [(1, 5)],
    ])


def test_bundle_matches_the_loop_reference_and_is_row_normalized() -> None:
    torch.manual_seed(0)
    composer = FrameComposer(_schedule(), 6, 3, 32, operator="hrr")
    ids = torch.tensor([2, 0, 3, 0, 1])
    rows = composer.compose(ids)
    torch.testing.assert_close(rows, uniform_bundle(composer, ids), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rows.norm(dim=-1), torch.ones(5))


def test_large_temperature_recovers_the_uniform_bundle() -> None:
    torch.manual_seed(1)
    composer = FrameComposer(_schedule(), 6, 3, 32, mode="attentive", temperature=1e9)
    ids = torch.arange(4)
    torch.testing.assert_close(composer.compose(ids, return_weights=False), uniform_bundle(composer, ids),
                               atol=1e-6, rtol=1e-5)


def test_segment_softmax_equals_a_loop() -> None:
    scores = torch.randn(9)
    segments = torch.tensor([0, 0, 2, 2, 2, 1, 0, 1, 2])
    result = segment_softmax(scores, segments, 3)
    for segment in range(3):
        mask = segments == segment
        torch.testing.assert_close(result[mask], torch.softmax(scores[mask], 0))


def test_attention_weights_preserve_frame_mass() -> None:
    torch.manual_seed(2)
    composer = FrameComposer(_schedule(), 6, 3, 16, mode="attentive", context_dimension=4)
    ids = torch.tensor([0, 2, 2])
    _, weights, edge_index = composer.compose(ids, torch.randn(3, 4), return_weights=True)
    per_occurrence = [weights[:3].sum(), weights[3:7].sum(), weights[7:].sum()]
    torch.testing.assert_close(torch.stack(per_occurrence), torch.tensor([3.0, 4.0, 4.0]))
    assert edge_index.numel() == 11


def test_context_changes_the_composition() -> None:
    torch.manual_seed(3)
    composer = FrameComposer(_schedule(), 6, 3, 16, mode="attentive", context_dimension=4)
    with torch.no_grad():
        composer.context.weight.mul_(20)
    ids = torch.tensor([2, 2])
    rows = composer.compose(ids, torch.randn(2, 4))
    assert float(F.cosine_similarity(rows[0], rows[1], dim=0).detach()) < 0.999


def test_duplicate_edges_accumulate() -> None:
    torch.manual_seed(4)
    single = FrameSchedule.from_frames([[(0, 0), (1, 1)]])
    doubled = FrameSchedule.from_frames([[(0, 0), (0, 0), (1, 1)]])
    a = FrameComposer(single, 2, 2, 16, operator="untyped")
    b = FrameComposer(doubled, 2, 2, 16, operator="untyped")
    with torch.no_grad():
        b.atomics.copy_(a.atomics)
    atoms = a.atomic_vectors()
    expected = F.normalize(2 * atoms[0] + atoms[1], dim=-1)
    torch.testing.assert_close(b.compose(torch.tensor([0]))[0], expected)
    assert not torch.allclose(a.compose(torch.tensor([0]))[0], expected)


def test_induced_mode_has_no_per_concept_parameters() -> None:
    induced = FrameComposer(_schedule(), 6, 3, 16, mode="attentive", concept_factor="induced")
    free = FrameComposer(_schedule(), 6, 3, 16, mode="attentive", concept_factor="free")
    hybrid = FrameComposer(_schedule(), 6, 3, 16, mode="attentive", concept_factor="hybrid")
    assert induced.parameter_groups()["concept_local"] == 0
    assert free.parameter_groups()["concept_local"] == 4 * 16
    assert hybrid.parameter_groups()["concept_local"] == 4 * 16
    assert float(hybrid.delta_penalty().detach()) == 0.0


def test_held_out_targets_never_influence_training() -> None:
    def train(held_out_target: torch.Tensor) -> dict[str, torch.Tensor]:
        torch.manual_seed(5)
        composer = FrameComposer(_schedule(), 6, 3, 16, mode="attentive")
        targets = torch.randn(4, 16)
        targets[3] = held_out_target
        optimizer = torch.optim.Adam(composer.parameters(), lr=0.05)
        train_ids = torch.tensor([0, 1, 2])
        for _ in range(20):
            optimizer.zero_grad()
            loss = (1 - F.cosine_similarity(composer(train_ids), targets[train_ids])).mean()
            loss.backward(); optimizer.step()
        return {name: value.detach().clone() for name, value in composer.state_dict().items()}
    first, second = train(torch.randn(16)), train(torch.randn(16) * 5)
    for name in first:
        torch.testing.assert_close(first[name], second[name])


def test_random_fixed_operator_is_not_trained_and_projector_maps_width() -> None:
    composer = FrameComposer(_schedule(), 6, 3, 16, operator="random_fixed:hrr", output_dimension=24)
    assert composer.parameter_groups()["relation"] == 0
    assert composer(torch.tensor([0, 1])).shape == (2, 24)


def test_explain_lists_every_edge_with_weights() -> None:
    composer = FrameComposer(_schedule(), 6, 3, 16, mode="attentive")
    explanation = composer.explain(2)
    assert len(explanation) == 4
    assert sum(item["weight"] for item in explanation) == pytest.approx(4.0, abs=1e-5)
    assert {item["filler"] for item in explanation} == {0, 5, 3, 2}


def test_recipes_convert_to_frames() -> None:
    recipes = torch.tensor([[1, -1, 2], [0, 0, -1]])
    schedule = FrameSchedule.from_recipes(recipes)
    assert schedule.degrees.tolist() == [2, 2]
    assert schedule.relations.tolist() == [0, 2, 0, 1]
    assert schedule.fillers.tolist() == [1, 2, 0, 0]


def test_schedule_validation() -> None:
    with pytest.raises(ValueError):
        FrameComposer(_schedule(), 3, 3, 8)  # filler 5 out of range
    composer = FrameComposer(_schedule(), 6, 3, 8)
    with pytest.raises(ValueError):
        composer.compose(torch.tensor([0]).reshape(1, 1))
