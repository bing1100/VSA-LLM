import copy

import pytest
import torch
from torch.nn import functional as F

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.frame_inference import candidate_dictionary, frame_scores, infer_frames
from vsa_embed.learnable_ontology import (
    ORIGIN, EdgeTable, LearnabilityConfig, LearnableOntologyComposer, train_composer,
)
from vsa_embed.ontology_hypotheses import (
    StructuralHypothesis, adjusted_rand_index, generate_hypotheses, partition_scores, property_holds, symmetry,
    transitivity,
)
from vsa_embed.self_test import HeldOutData, bootstrap_lower, edge_contributions, edit_effects

FRAMES = [[(0, 1), (1, 2), (2, 3)], [(0, 4), (1, 1)], [(2, 0), (0, 5), (1, 3), (2, 2)], [(1, 5)]]
ASSERTED = [(i, r, a) for i, frame in enumerate(FRAMES) for r, a in frame]
CANDIDATES = [(0, 1, 5), (1, -1, 3), (3, 2, 0), (2, -1, 1)]
FIXED = LearnabilityConfig(atomics="fixed", relations="fixed", mapping="fixed", frames="fixed")


def _pair(mode: str, candidates=CANDIDATES, max_slots: int = 2, **kwargs):
    torch.manual_seed(0)
    base = FrameComposer(FrameSchedule.from_frames(FRAMES), 6, 3, 32, operator="hrr", mode=mode, **kwargs)
    with torch.no_grad():
        if mode == "salience":
            base.relation_salience.copy_(torch.randn(3))
    table = EdgeTable.build(4, 3, max_slots=max_slots, asserted=ASSERTED, candidates=candidates)
    learnable = LearnableOntologyComposer(table, 6, 32, operator="hrr", mode=mode, learn=FIXED, **kwargs)
    params = dict(learnable.named_parameters())
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            params[name].copy_(parameter)
    return base, learnable


@pytest.mark.parametrize("mode,kwargs", [("bundle", {}), ("salience", {}), ("attentive", {"key_dimension": 8}),
                                         ("attentive", {"key_dimension": 8, "concept_factor": "hybrid"})])
@pytest.mark.parametrize("candidates", [CANDIDATES, []])
def test_fixed_everything_equals_frame_composer(mode, kwargs, candidates) -> None:
    base, learnable = _pair(mode, candidates, **kwargs)
    learnable.add_blank_slot()
    ids = torch.tensor([2, 0, 3, 1, 0])
    torch.testing.assert_close(learnable(ids), base(ids), atol=1e-6, rtol=0)
    rows, weights, edges = learnable.compose(ids, return_weights=True)
    torch.testing.assert_close(rows, base.compose(ids), atol=1e-6, rtol=0)


def test_blank_slots_with_zero_assignment_do_not_change_outputs() -> None:
    _, learnable = _pair("bundle", candidates=[(0, 1, 5), (3, 2, 0)])   # no open edges
    ids = torch.arange(4)
    before = learnable(ids).detach().clone()
    slot = learnable.add_blank_slot()
    with torch.no_grad():
        learnable.slot_roles[slot] = torch.randn(32) * 10
    torch.testing.assert_close(learnable(ids), before)
    # open candidates at mass 0: opening a slot (no initial mass) leaves every row unchanged
    _, opened = _pair("bundle")
    before = opened(ids).detach().clone()
    opened.add_blank_slot()
    opened.add_blank_slot()
    torch.testing.assert_close(opened(ids), before)
    # a slot that no edge may use, with an arbitrary operator, changes nothing even when masses are live
    learn = LearnabilityConfig(atomics="fixed", relations="fixed", mapping="fixed", frames="l2")
    table = EdgeTable.build(4, 3, max_slots=2, asserted=ASSERTED, candidates=CANDIDATES)
    composer = LearnableOntologyComposer(table, 6, 32, learn=learn)
    composer.add_blank_slot(init_mass=0.3)
    composer.options[:, composer.slot_column(1)] = False
    before = composer(ids).detach().clone()
    composer.add_blank_slot()
    with torch.no_grad():
        composer.slot_roles[1] = torch.randn(32)
    torch.testing.assert_close(composer(ids), before)


def test_crystallized_slot_is_unchanged_by_further_training() -> None:
    torch.manual_seed(1)
    table = EdgeTable.build(4, 3, max_slots=2, asserted=ASSERTED, candidates=CANDIDATES + [(0, -1, 2), (3, -1, 4)])
    composer = LearnableOntologyComposer(table, 6, 32, learn=LearnabilityConfig(frame_l1=0.0))
    targets = F.normalize(torch.randn(4, 2, 32), dim=-1)
    slot = composer.add_blank_slot(init_mass=0.5)
    train_composer(composer, torch.arange(4), targets, steps=30, lr=0.05)
    card = composer.crystallize(slot, min_assignment=0.0, min_mass=0.0)
    assert card["edges"] > 0
    role = composer.slot_role_vectors()[slot].detach().clone()
    members = composer.slot_members(slot, hard_only=True).clone()
    masses = composer.edge_masses()[members].detach().clone()
    train_composer(composer, torch.arange(4), targets, steps=40, lr=0.05)
    torch.testing.assert_close(composer.slot_role_vectors()[slot], role)
    assert torch.equal(composer.slot_members(slot, hard_only=True), members)
    assert bool((composer.assignment_fixed[members] == composer.slot_column(slot)).all())
    # with the mapping fixed, crystallized edge masses are frozen too
    composer.learn = LearnabilityConfig(mapping="fixed")
    composer.apply_learnability()
    masses = composer.edge_masses()[members].detach().clone()
    train_composer(composer, torch.arange(4), targets, steps=20, lr=0.05)
    torch.testing.assert_close(composer.edge_masses()[members], masses)
    composer.reopen(slot)
    assert not bool(composer.slot_frozen[slot])
    assert bool((composer.assignment_fixed[members] < 0).all())


def test_add_edges_keeps_existing_compositions_and_marks_the_slot() -> None:
    _, composer = _pair("bundle")
    slot = composer.add_blank_slot()
    ids = torch.arange(4)
    before = composer(ids).detach().clone()
    new = composer.add_edges([3, 1], [composer.slot_column(slot), 2], [2, 0], mass=0.0, origin="rule")
    torch.testing.assert_close(composer(ids), before)
    assert int(composer.assignment_fixed[new[0]]) == composer.slot_column(slot)
    assert int(composer.schedule.relations[new[1]]) == 2 and int(composer.origin[new[1]]) == ORIGIN["rule"]
    assert composer.edge_heads()[new].tolist() == [3, 1]


def test_seed_empty_concepts_and_zero_rows_are_handled() -> None:
    table = EdgeTable.build(3, 2, asserted=[(0, 0, 1)], candidates=[(1, 1, 2), (1, 0, 0)])
    composer = LearnableOntologyComposer(table, 3, 16)
    rows = composer(torch.arange(3)).detach()
    assert float(rows[1].norm()) == 0.0 and float(rows[2].norm()) == 0.0     # concept 2 has no edges at all
    assert composer.seed_empty_concepts(0.1) == 2
    assert float(composer(torch.arange(3)).detach()[1].norm()) > 0.99


def test_edit_effects_match_recomposition() -> None:
    torch.manual_seed(3)
    table = EdgeTable.build(4, 3, asserted=ASSERTED)
    composer = LearnableOntologyComposer(table, 6, 32)
    data = HeldOutData(torch.arange(4), F.normalize(torch.randn(4, 3, 32), dim=-1))
    edge = 4                                                        # an edge of concept 1
    head = int(composer.edge_heads()[edge])
    delta = -edge_contributions(composer, torch.tensor([edge]))
    effect = edit_effects(composer, data, torch.tensor([head]), delta)
    reference = copy.deepcopy(composer)
    reference.pin_masses(torch.tensor([edge]), 0.0)
    before = F.cosine_similarity(composer(torch.tensor([head]))[:, None], data.observations[head][None], dim=-1)
    after = F.cosine_similarity(reference(torch.tensor([head]))[:, None], data.observations[head][None], dim=-1)
    torch.testing.assert_close(effect, after - before, atol=1e-5, rtol=0)
    mean, lower = bootstrap_lower(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), resamples=200)
    assert float(mean) == 2.5 and 1.0 <= float(lower) <= 2.5


def test_frame_inference_recovers_a_planted_frame() -> None:
    torch.manual_seed(4)
    roles = F.normalize(torch.randn(3, 64), dim=-1)
    atomics = F.normalize(torch.randn(40, 64), dim=-1)
    from vsa_embed.algebra import HRRAlgebra
    bind = lambda cols, fills: HRRAlgebra().bind(roles[cols], atomics[fills])   # noqa: E731
    grid_c, grid_f, vectors = candidate_dictionary(bind, range(3), range(40))
    planted = {(0, 5), (1, 17), (2, 30)}
    row = F.normalize(sum(bind(torch.tensor([c]), torch.tensor([a]))[0] for c, a in planted), dim=-1)
    observations = F.normalize(row[None, None] + 0.05 * torch.randn(1, 4, 64) / 8, dim=-1)
    frame = infer_frames(observations, grid_c, grid_f, vectors, max_edges=8, threshold=0.2)[0]
    predicted = set(zip(frame.columns, frame.fillers))
    assert frame_scores(predicted, planted)["f1"] == 1.0


def test_structural_hypotheses_predict_unseen_pairs() -> None:
    symmetric = {(1, 2), (2, 1), (3, 4)}
    hyp = StructuralHypothesis("symmetric")
    positive, negative = hyp.predict(symmetric, {}, heads={1, 2, 3, 4}, tails={1, 2, 3, 4})
    assert positive == {(4, 3)} and not negative
    chain = {("a", "b"), ("b", "c"), ("c", "d")}
    positive, _ = StructuralHypothesis("transitive").predict(chain, {}, heads={"a", "b", "c", "d"}, tails={"a", "b", "c", "d"})
    assert positive == {("a", "c"), ("b", "d")}
    part_of = {(10, 1), (11, 1), (12, 2)}
    has_part = {(1, 10)}
    generated = {h.name for h in generate_hypotheses(has_part | {(1, 11)}, {"part_of": part_of})}
    assert "inverse_of:part_of" in generated
    positive, _ = StructuralHypothesis("inverse_of", ("part_of",)).predict(has_part, {"part_of": part_of}, heads={1, 2},
                                                                         tails={10, 11, 12})
    assert positive == {(1, 11), (2, 12)}
    assert symmetry({(1, 2), (2, 1)}) == 1.0 and transitivity(chain)[1] == 2
    assert property_holds(StructuralHypothesis("inverse_of", ("pof",)), {(1, 10), (1, 11), (2, 12)},
                          {"part_of": part_of}, {"pof": "part_of"})


def test_partition_scores_and_ari() -> None:
    assert adjusted_rand_index([0, 0, 1, 1], [5, 5, 7, 7]) == pytest.approx(1.0)
    gold = ["r", "r", "s", "s", None, None]
    predicted = [1, 1, 2, 2, None, 1]
    pairs = [(i, i) for i in range(6)]
    scores = partition_scores(gold, predicted, pairs, permutations=50)
    assert scores.ari_gold_edges == pytest.approx(1.0)
    assert scores.detection_recall == 1.0 and scores.detection_precision == pytest.approx(0.8)
    assert scores.per_relation["r"]["best_jaccard"] == pytest.approx(2 / 3)
