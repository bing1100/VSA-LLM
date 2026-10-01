import torch
from torch.nn import functional as F

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.developmental import (
    DevelopmentalConfig, DevelopmentalDictionary, split_gain, split_test, top_between_usage_direction,
)


def test_split_gain_is_nonnegative_and_zero_when_usages_agree() -> None:
    torch.manual_seed(0)
    agreeing = torch.randn(1, 8).abs() * torch.rand(5, 1) + 0.0
    direction = top_between_usage_direction(torch.randn(5, 8))
    assert abs(split_gain(agreeing, F.normalize(torch.ones(8), dim=0))) < 1e-6
    for _ in range(20):
        momenta = torch.randn(6, 8)
        assert split_gain(momenta, F.normalize(torch.randn(8), dim=0)) >= -1e-6
    assert split_gain(torch.randn(5, 8), direction) >= -1e-6


def test_opposed_usages_have_large_gain_along_the_conflict() -> None:
    axis = F.normalize(torch.randn(8), dim=0)
    momenta = torch.cat([axis.expand(4, -1), -axis.expand(4, -1)])
    direction = top_between_usage_direction(momenta)
    assert abs(float(direction @ axis)) > 0.99
    assert split_gain(momenta, direction) > 7.9


def test_permutation_null_rarely_accepts_pure_noise() -> None:
    generator = torch.Generator().manual_seed(1)
    accepted = 0
    trials = 60
    for trial in range(trials):
        usages = torch.randint(0, 6, (60,), generator=generator)
        gradients = torch.randn(60, 16, generator=generator)
        result = split_test(usages, gradients, permutations=99, generator=generator)
        accepted += result["p_value"] < 0.01
    assert accepted / trials <= 0.05


def test_permutation_null_detects_planted_conflict() -> None:
    generator = torch.Generator().manual_seed(2)
    axis = F.normalize(torch.randn(16, generator=generator), dim=0)
    usages = torch.randint(0, 8, (80,), generator=generator)
    sign = torch.where(usages < 4, 1.0, -1.0)
    gradients = sign[:, None] * axis + 0.3 * torch.randn(80, 16, generator=generator)
    result = split_test(usages, gradients, permutations=99, generator=generator)
    assert result["p_value"] < 0.02


def _composer(normalize: bool = False) -> FrameComposer:
    schedule = FrameSchedule.from_frames([[(0, 0), (1, 1)], [(0, 0), (1, 2)], [(1, 0), (0, 2)], [(0, 1)]])
    torch.manual_seed(3)
    return FrameComposer(schedule, 3, 2, 12, operator="hrr", normalize_atomics=normalize)


def test_per_usage_gradients_equal_per_concept_autograd() -> None:
    for normalize in (False, True):
        composer = _composer(normalize)
        targets = torch.randn(4, 12)
        tracker = DevelopmentalDictionary(composer, None, DevelopmentalConfig())
        tracker.candidates = {0}
        tracker.begin()
        ids = torch.tensor([0, 1, 2])
        loss = ((composer(ids) - targets[ids]) ** 2).sum()
        loss.backward()
        [(vector_id, labels, grads)] = tracker._usage_gradients({0})
        for label, grad in zip(labels.tolist(), grads):
            composer.zero_grad()
            single = ((composer(torch.tensor([label])) - targets[[label]]) ** 2).sum()
            single.backward()
            torch.testing.assert_close(grad, composer.atomics.grad[0], atol=1e-5, rtol=1e-4)
        composer.captured.clear()


def test_split_duplicates_optimizer_state_and_rewires_frames() -> None:
    composer = _composer(True)
    optimizer = torch.optim.Adam(composer.parameters(), lr=0.01)
    ids = torch.arange(4)
    loss = composer(ids).sum(); loss.backward(); optimizer.step()
    tracker = DevelopmentalDictionary(composer, optimizer, DevelopmentalConfig(growth_budget=1.0))
    direction = F.normalize(torch.randn(12), dim=0)
    result = {"direction": direction, "labels": torch.tensor([0, 1, 2]),
              "momenta": torch.stack([direction, -direction, direction]), "gain": 2.0, "p_value": 0.001,
              "contributions": 30}
    before = optimizer.state[composer.atomics]["exp_avg"][0].clone()
    card = tracker._split(0, result)
    new_id = card["children"][1]
    assert composer.atomics.shape[0] == 4 and new_id == 3
    torch.testing.assert_close(optimizer.state[composer.atomics]["exp_avg"][3], before)
    assert any(p is composer.atomics for p in optimizer.param_groups[0]["params"])
    fillers = composer.schedule.fillers
    assert int(fillers[2]) == 3                  # concept 1 (negative side) moved to the new child
    assert int(fillers[0]) == 0 and int(fillers[4]) == 0
    assert tracker.momentum.shape[0] == 4


def test_budget_limits_the_number_of_splits() -> None:
    composer = _composer(True)
    tracker = DevelopmentalDictionary(composer, None, DevelopmentalConfig(growth_budget=0.34))
    assert tracker.budget_left() == 1
    tracker.splits = 1
    assert tracker.budget_left() == 0


def test_allocate_and_merge_cards() -> None:
    composer = _composer(True)
    tracker = DevelopmentalDictionary(composer, None, DevelopmentalConfig(merge_cosine=0.9))
    new_id = tracker.allocate(composer.atomics.detach()[1] * 1.0, reason="ontology update")
    assert new_id in tracker.provisional and composer.atomics.shape[0] == 4
    tracker.siblings.append((1, new_id))
    schedule = composer.schedule
    fillers = schedule.fillers.clone(); fillers[1] = new_id
    composer.set_schedule(FrameSchedule(schedule.offsets, schedule.relations, fillers))
    events = tracker.consolidate()
    assert events and events[0]["kept"] == 1 and int(composer.schedule.fillers[1]) == 1
    assert bool(tracker.frozen[new_id])


def test_relation_split_in_attentive_mode_registers_new_keys_with_the_optimizer() -> None:
    schedule = FrameSchedule.from_frames([[(0, 0), (1, 1)], [(0, 2), (1, 0)], [(0, 1)]])
    torch.manual_seed(5)
    composer = FrameComposer(schedule, 3, 2, 12, operator="hrr", mode="attentive")
    optimizer = torch.optim.Adam(composer.parameters(), lr=0.01)
    composer(torch.arange(3)).sum().backward(); optimizer.step()
    for route in ("context", "parent"):
        tracker = DevelopmentalDictionary(composer, optimizer, DevelopmentalConfig(target="relations", route_unobserved=route,
                                                                                    growth_budget=4.0))
        direction = F.normalize(torch.randn(12), dim=0)
        edges = (composer.schedule.relations == 0).nonzero().flatten()
        tracker._split(0, {"direction": direction, "labels": edges, "momenta": torch.stack([direction, -direction, direction][:edges.numel()]),
                           "gain": 1.0, "p_value": 0.001, "contributions": 10})
        params = optimizer.param_groups[0]["params"]
        assert any(p is composer.relation_keys for p in params)
        assert any(p is composer.relation_vectors() for p in params)
        assert composer.relation_keys.shape[0] == composer.relation_count
