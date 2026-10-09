"""Unbinding and cleanup primitives (decision 60): an unbind ∘ bind round trip for every composition operator family,
decoding noise that grows with superposition load, hard / type-constrained / soft cleanup, and the learned unitary HRR
family (trainable phases, exactly invertible)."""

import pytest
import torch
from torch.nn import functional as F

from vsa_embed import cleanup as cl
from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.relations import (UNBIND_RIDGE, MAPRelation, UnbindingError, UnitaryHRRRelation, create_composition_operator,
                                 readout_method)

EXACT = ["unitary_hrr", "orthogonal", "low_rank", "low_rank_identity", "low_rank_tied", "translation"]
REGULARIZED = ["diagonal", "map"]


def _perturbed(family: str, relations: int = 4, dimension: int = 64, *, scale: float = 0.3, seed: int = 0):
    torch.manual_seed(seed)
    transform = create_composition_operator(family, relations, dimension, rank=4)
    with torch.no_grad():
        for name, parameter in transform.named_parameters():
            if family == "map" and name == "roles":
                parameter.copy_(1.0 + scale * torch.randn_like(parameter))         # learned roles: no longer bipolar
            elif family in {"diagonal"}:
                parameter.copy_(1.0 + scale * torch.randn_like(parameter))
            else:
                parameter.add_(scale * torch.randn_like(parameter) / dimension**0.5)
    return transform


@pytest.mark.parametrize("family", EXACT)
def test_exact_families_invert_one_binding(family: str) -> None:
    transform = _perturbed(family)
    ids = torch.tensor([0, 1, 2, 3, 1])
    x = torch.randn(5, 64)
    back = transform.unbind(ids, transform(ids, x))
    torch.testing.assert_close(back, x, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize("family", REGULARIZED)
def test_regularized_division_is_nearly_exact(family: str) -> None:
    transform = _perturbed(family, scale=0.2)
    ids = torch.tensor([0, 1, 2, 3])
    x = torch.randn(4, 64)
    back = transform.unbind(ids, transform(ids, x), ridge=1e-8)
    torch.testing.assert_close(back, x, atol=1e-4, rtol=1e-4)
    loose = transform.unbind(ids, transform(ids, x))                                 # default ridge: still close
    assert F.cosine_similarity(loose, x, dim=-1).min() > 0.98


def test_hrr_correlation_is_approximate_and_inverse_is_close_to_exact() -> None:
    transform = create_composition_operator("hrr", 3, 256)
    ids = torch.tensor([0, 1, 2])
    torch.manual_seed(1)
    x = F.normalize(torch.randn(3, 256), dim=-1)
    bound = transform(ids, x)
    correlation = transform.unbind(ids, bound)                                      # primary: the adjoint
    torch.testing.assert_close(correlation, transform.adjoint(ids, bound))
    inverse = transform.unbind(ids, bound, method="inverse", ridge=1e-6)
    cos = lambda a: F.cosine_similarity(a, x, dim=-1)
    assert 0.3 < cos(correlation).min() < 0.95                                     # random Gaussian roles: approximate
    assert cos(inverse).min() > 0.999
    assert cos(transform.unbind(ids, bound, method="inverse")).min() > cos(correlation).max() - 0.05
    with pytest.raises(ValueError):
        transform.unbind(ids, bound, method="transpose")


def test_unitary_roles_have_unit_spectra_and_learnable_phases() -> None:
    transform = UnitaryHRRRelation(3, 32)
    spectrum = torch.fft.rfft(transform.role_vectors())
    torch.testing.assert_close(spectrum.abs(), torch.ones_like(spectrum.abs()), atol=1e-5, rtol=0)
    ids, x = torch.tensor([0, 2]), torch.randn(2, 32)
    loss = transform(ids, x).square().sum() + transform.unbind(ids, x).sum()
    loss.backward()
    assert transform.phases.grad is not None and transform.phases.grad.abs().sum() > 0      # the phases are trained
    with torch.no_grad():
        transform.phases.add_(torch.randn_like(transform.phases))                    # any phases: still exactly invertible
    torch.testing.assert_close(transform.unbind(ids, transform(ids, x)), x, atol=1e-5, rtol=1e-5)
    frozen = create_composition_operator("unitary_hrr", 2, 16)
    assert readout_method(frozen) == "conjugate"


def test_map_is_self_inverse_only_while_bipolar() -> None:
    transform = MAPRelation(2, 16)
    with torch.no_grad():
        transform.roles.copy_(torch.where(torch.rand(2, 16) < 0.5, -1.0, 1.0))
        transform.roles[1, 0] = 0.5                                                 # relation 1 is no longer bipolar
    assert transform.bipolar().tolist() == [True, False]
    x = torch.randn(2, 16)
    ids = torch.tensor([0, 1])
    auto = transform.unbind(ids, transform(ids, x), ridge=1e-9)
    torch.testing.assert_close(auto, x, atol=1e-5, rtol=1e-5)
    self_inverse = transform.unbind(ids, transform(ids, x), method="self")
    torch.testing.assert_close(self_inverse[0], x[0])
    assert not torch.allclose(self_inverse[1], x[1])


def test_additive_has_no_unbinding_but_a_documented_bundle_readout() -> None:
    transform = create_composition_operator("additive", 3, 8)
    ids, x = torch.tensor([0, 1]), torch.randn(2, 8)
    with pytest.raises(UnbindingError, match="bundle"):
        transform.unbind(ids, x)
    assert torch.equal(transform.unbind(ids, x, method="bundle"), x)
    assert readout_method(transform) == "bundle"


def test_translation_bundle_loses_which_filler_went_with_which_relation() -> None:
    transform = create_composition_operator("translation", 2, 64)
    torch.manual_seed(3)
    a, b = F.normalize(torch.randn(2, 64), dim=-1)
    first = transform(torch.tensor([0, 1]), torch.stack([a, b])).sum(0)            # a under r0, b under r1
    second = transform(torch.tensor([0, 1]), torch.stack([b, a])).sum(0)           # roles swapped
    torch.testing.assert_close(first, second)                                       # identical: the binding is lost
    hrr = create_composition_operator("hrr", 2, 64)
    assert not torch.allclose(hrr(torch.tensor([0, 1]), torch.stack([a, b])).sum(0),
                              hrr(torch.tensor([0, 1]), torch.stack([b, a])).sum(0))


def _decode_accuracy(family: str, load: int, *, dimension: int = 256, atoms: int = 512, trials: int = 64, seed: int = 0) -> tuple[float, float]:
    """Top-1 cleanup accuracy and mean cosine to the gold filler after unbinding one role from a bundle of `load` pairs."""
    generator = torch.Generator().manual_seed(seed)
    transform = create_composition_operator(family, load, dimension)
    if family == "orthogonal":                                   # starts at the identity; a trained one is a rotation
        with torch.no_grad():
            transform.raw.copy_(torch.randn(transform.raw.shape, generator=generator))
    dictionary = F.normalize(torch.randn(atoms, dimension, generator=generator), dim=-1)
    fillers = torch.stack([torch.randperm(atoms, generator=generator)[:load] for _ in range(trials)])
    relations = torch.arange(load).expand(trials, load)
    with torch.no_grad():
        bundles = transform(relations.reshape(-1), dictionary[fillers.reshape(-1)]).reshape(trials, load, -1).sum(1)
        probe = transform.unbind(torch.zeros(trials, dtype=torch.long), bundles, method=readout_method(transform))
    index, _ = cl.nearest(probe, dictionary)
    cos = F.cosine_similarity(probe, dictionary[fillers[:, 0]], dim=-1)
    return float((index == fillers[:, 0]).float().mean()), float(cos.mean())


@pytest.mark.parametrize("family", ["unitary_hrr", "hrr", "orthogonal"])
def test_decoding_noise_grows_with_superposition_load(family: str) -> None:
    results = [_decode_accuracy(family, k) for k in (1, 4, 16, 64)]
    accuracies, cosines = zip(*results)
    assert accuracies[0] == 1.0
    assert all(a >= b for a, b in zip(cosines, cosines[1:])) and cosines[0] > 2.5 * cosines[-1]
    assert accuracies[-1] < accuracies[0]


def test_untyped_bundle_readout_cannot_find_the_role() -> None:
    accuracy, _ = _decode_accuracy("additive", 8)
    assert accuracy < 0.4                                                           # ≈ 1/8: any filler of the bag


# -- cleanup ---------------------------------------------------------------------------------------------------------------

def test_relation_candidates_and_type_constrained_nearest() -> None:
    mask = cl.relation_candidates(torch.tensor([0, 0, 1]), torch.tensor([2, 3, 0]), 2, 4)
    assert mask.tolist() == [[False, False, True, True], [True, False, False, False]]
    dictionary = torch.eye(4)
    query = torch.tensor([[1.0, 0.0, 0.2, 0.0]])
    assert cl.nearest(query, dictionary)[0].item() == 0
    assert cl.nearest(query, dictionary, mask[[0]])[0].item() == 2               # constrained to relation 0's fillers


def test_filtered_ranks_count_ties_at_half_weight_and_drop_other_golds() -> None:
    scores = torch.tensor([[0.9, 0.5, 0.5, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.8, 0.9, 0.1]])
    gold = torch.tensor([1, 0, 0])
    ranks = cl.filtered_ranks(scores, gold)
    assert ranks.tolist() == [2.5, 2.5, 3.0]                                         # tie at half weight; all tied → chance
    exclude = torch.zeros(3, 4, dtype=torch.bool)
    exclude[2, 2] = True                                                             # another gold of query 2
    allowed = torch.tensor([[False, True, True, True]] * 3)
    assert cl.filtered_ranks(scores, gold, allowed=allowed, exclude=exclude).tolist() == [1.5, 2.5, 2.0]


def test_chunked_ranks_equal_the_one_shot_ranks() -> None:
    torch.manual_seed(0)
    queries, dictionary = torch.randn(37, 16), torch.randn(50, 16)
    gold = torch.randint(0, 50, (37,))
    rows = torch.rand(3, 50) < 0.5
    which = torch.randint(0, 3, (37,))
    pairs = (torch.tensor([0, 5, 5, 36]), torch.tensor([1, 2, 3, 4]))
    exclude = torch.zeros(37, 50, dtype=torch.bool)
    exclude[pairs] = True
    full = cl.filtered_ranks(cl.cosine_scores(queries, dictionary), gold, allowed=rows[which], exclude=exclude)
    chunked = cl.chunked_filtered_ranks(queries, dictionary, gold, allowed_rows=rows, allowed_index=which,
                                        exclude_pairs=pairs, chunk=8)
    torch.testing.assert_close(chunked, full)


def test_soft_cleanup_approaches_the_hard_one_and_is_differentiable() -> None:
    torch.manual_seed(2)
    dictionary = F.normalize(torch.randn(20, 32), dim=-1)
    queries = dictionary[[3, 7]] + 0.3 * torch.randn(2, 32)
    sharp = cl.SoftCleanup(beta=500.0)
    out, weights = sharp(queries, dictionary, return_weights=True)
    torch.testing.assert_close(out, dictionary[[3, 7]], atol=1e-3, rtol=0)
    assert weights.argmax(-1).tolist() == [3, 7]
    soft = cl.SoftCleanup(beta=4.0, steps=2)
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :5] = True
    masked = soft(queries.requires_grad_(), dictionary, mask)
    masked.sum().backward()
    assert queries.grad is not None and soft.log_beta.grad is not None
    empty = soft(torch.randn(1, 32), dictionary, torch.zeros(1, 20, dtype=torch.bool))
    assert torch.equal(empty, torch.zeros(1, 32))


# -- the composer -------------------------------------------------------------------------------------------------------------

def test_raw_bundle_is_the_unnormalized_composition() -> None:
    schedule = FrameSchedule.from_frames([[(0, 1), (1, 2)], [(1, 0), (2, 3), (0, 2)]])
    composer = FrameComposer(schedule, 4, 3, 32, operator="hrr", mode="attentive", key_dimension=4)
    ids = torch.tensor([0, 1])
    summed, weights, edges, segments = composer.raw_bundle(ids)
    torch.testing.assert_close(F.normalize(summed, dim=-1), composer.compose(ids))
    static, ones, _, _ = composer.raw_bundle(ids, uniform=True)
    assert torch.equal(ones, torch.ones(5)) and segments.tolist() == [0, 0, 1, 1, 1]
    bound = composer.bound_edges(edges)
    torch.testing.assert_close(static[1], bound[2:].sum(0))
    unbound = composer.unbind(torch.tensor([0]), bound[[0]])                         # unbind relation 0 from (0, atomic 1)
    assert F.cosine_similarity(unbound, composer.atomic_vectors()[[1]]).item() > 0.3
    untyped = FrameComposer(schedule, 4, 3, 32, operator="untyped")
    assert torch.equal(untyped.unbind(torch.tensor([1]), static[[0]]), static[[0]])  # the bundle readout


def test_default_ridge_is_documented() -> None:
    assert UNBIND_RIDGE == pytest.approx(1e-2)


# -- decision 61: bounded spectral circulant, block-diagonal unitary, slotted layout ----------------------------------------

def test_spectral_bounded_starts_unitary_and_keeps_its_condition_number_below_four() -> None:
    from vsa_embed.relations import SpectralBoundedRelation
    transform = SpectralBoundedRelation(3, 64)
    torch.testing.assert_close(transform.magnitudes(), torch.ones_like(transform.magnitudes()))
    with torch.no_grad():
        transform.magnitude_logits.copy_(20 * torch.randn_like(transform.magnitude_logits))     # push to the bounds
    magnitudes = transform.magnitudes().detach()
    assert magnitudes.min() >= 0.5 and magnitudes.max() <= 2.0
    assert float((magnitudes.max(-1).values / magnitudes.min(-1).values).max()) <= 4.0 + 1e-5
    role = transform.role_vectors()
    torch.testing.assert_close(torch.fft.rfft(role).abs(), magnitudes, atol=1e-4, rtol=1e-4)      # real roles (conjugate symmetry)
    ids, x = torch.tensor([0, 1, 2]), torch.randn(3, 64)
    bound = transform(ids, x)
    torch.testing.assert_close(transform.unbind(ids, bound, method="exact"), x, atol=1e-4, rtol=1e-4)
    adjoint = transform.unbind(ids, bound)                                              # primary: the adjoint
    torch.testing.assert_close(adjoint, transform.adjoint(ids, bound))
    assert F.cosine_similarity(adjoint, x, dim=-1).min() > 0.5
    (transform(ids, x).square().sum()).backward()
    assert transform.phases.grad is not None and transform.magnitude_logits.grad is not None
    assert readout_method(transform) == "adjoint"


def test_block_unitary_is_orthogonal_invertible_and_non_commutative() -> None:
    from vsa_embed.relations import BlockUnitaryRelation
    transform = BlockUnitaryRelation(3, 64, block=16)
    blocks = transform.matrices()
    eye = torch.eye(16).expand_as(blocks)
    torch.testing.assert_close(blocks @ blocks.transpose(-1, -2), eye, atol=1e-5, rtol=1e-5)
    ids, x = torch.tensor([0, 2, 1]), torch.randn(3, 64)
    torch.testing.assert_close(transform(ids, x).norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(transform.unbind(ids, transform(ids, x)), x, atol=1e-4, rtol=1e-4)
    one, two = torch.tensor([0]), torch.tensor([1])
    path = transform(two, transform(one, x[:1]))                                     # r2 · r1 · f
    reversed_path = transform(one, transform(two, x[:1]))                            # r1 · r2 · f
    assert not torch.allclose(path, reversed_path, atol=1e-3)                        # order is kept
    circulant = create_composition_operator("unitary_hrr", 3, 64)
    torch.testing.assert_close(circulant(two, circulant(one, x[:1])), circulant(one, circulant(two, x[:1])), atol=1e-5, rtol=1e-5)
    assert (BlockUnitaryRelation(2, 256).block, BlockUnitaryRelation(2, 256).blocks) == (16, 16)
    assert BlockUnitaryRelation(2, 60, block=16).block == 15 and BlockUnitaryRelation(2, 31).block == 31   # prime: one block
    with pytest.raises(ValueError):
        BlockUnitaryRelation(2, 60, block=1)


def test_balanced_slots_are_greedy_and_recorded() -> None:
    from vsa_embed.relations import balanced_slots, slot_bounds
    assert balanced_slots([10, 1, 7, 7, 2], 3) == [0, 2, 1, 2, 1]                    # 10 → s0, 7 → s1, 7 → s2, 2 → s1, 1 → s2
    assert slot_bounds(256, 3) == [(0, 86), (86, 171), (171, 256)]
    schedule = FrameSchedule.from_frames([[(0, 1), (0, 2), (1, 3)], [(0, 4), (2, 5)], [(3, 6), (2, 1)]])
    composer = FrameComposer(schedule, 8, 4, 30, operator="slotted_unitary", slots=3)
    assert composer.transform.slot_of.tolist() == [0, 2, 1, 2] and composer.transform.slot_load.tolist() == [3, 2, 2]
    masks = composer.slot_masks()
    assert masks.shape == (3, 30) and masks.sum(-1).tolist() == [10, 10, 10] and composer.slot_of().tolist() == [0, 2, 1, 2]
    assert FrameComposer(schedule, 8, 4, 30, operator="hrr").slot_masks() is None


def test_slotted_bundle_concatenates_slots_and_unbinds_within_them() -> None:
    schedule = FrameSchedule.from_frames([[(0, 1), (1, 2), (2, 3), (3, 4)]])
    composer = FrameComposer(schedule, 6, 4, 33, operator="slotted_unitary", slots=3)
    bound = composer.bound_edges(torch.arange(4))
    slots = composer.slot_of()[schedule.relations]
    masks = composer.slot_masks()
    for edge in range(4):
        assert torch.all(bound[edge][~masks[slots[edge]]] == 0)                     # nothing written outside the slot
    summed, *_ = composer.raw_bundle(torch.tensor([0]), uniform=True)
    for g in range(3):
        alone = bound[slots == g].sum(0)
        torch.testing.assert_close(summed[0][masks[g]], alone[masks[g]])           # each slot holds only its group
    single = composer.unbind(torch.tensor([0]), bound[[0]])
    expected = composer.atomic_vectors()[1] * masks[slots[0]]
    torch.testing.assert_close(single[0], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("family", ["additive", "hrr", "hrr_identity", "map", "diagonal", "low_rank", "low_rank_identity",
                                    "low_rank_tied", "orthogonal", "unitary_hrr", "translation", "spectral_bounded", "block_unitary"])
def test_every_family_moves_and_casts_like_a_module(family: str) -> None:
    transform = create_composition_operator(family, 3, 32, rank=4)
    transform.to("cpu").float()                                     # `nn.Module._apply` must not be shadowed
    ids, x = torch.tensor([0, 2]), torch.randn(2, 32)
    assert transform(ids, x).shape == (2, 32)
    composer = FrameComposer(FrameSchedule.from_frames([[(0, 1), (1, 2)], [(2, 3)]]), 4, 3, 33, operator="slotted_unitary")
    composer.to("cpu").float()
    assert composer.compose(torch.tensor([0, 1])).shape == (2, 33)


def test_the_readout_unbinds_a_fixed_random_operator_and_trains_around_it() -> None:
    """Decision 64 (pre-registration-binding §13.1, arm U5rf): the readout over a fixed random unitary operator
    (`random_fixed:unitary_hrr`, as C5rf) unbinds by the conjugate, recovers a filler exactly, and optimizer steps on the
    trainable parameters move the readout and the atomics but never the operator."""
    from vsa_embed.readout import UnbindingReadout
    torch.manual_seed(0)
    frames = [[(0, 1), (1, 2)], [(2, 3), (0, 4)], [(1, 5)]]
    composer = FrameComposer(FrameSchedule.from_frames(frames), 6, 3, 32, operator="random_fixed:unitary_hrr",
                             mode="attentive", key_dimension=4)
    assert not any(p.requires_grad for p in composer.transform.parameters())
    readout = UnbindingReadout(composer, 16, layer=1, window=4)
    assert readout.method == "conjugate"
    one = composer.raw_bundle(torch.tensor([2]), uniform=True)[0]                  # a one-edge store: exact recovery
    torch.testing.assert_close(composer.unbind(torch.tensor([1]), one)[0], composer.atomic_vectors()[5], atol=1e-5, rtol=1e-5)
    readout.current = {"batch": torch.tensor([0, 1]), "inject": torch.tensor([1, 0]), "entry": torch.tensor([0, 1])}
    hidden = torch.randn(2, 6, 16)
    phases, atomics, query = (composer.transform.phases.detach().clone(), composer.atomics.detach().clone(),
                              readout.query.weight.detach().clone())
    trainable = [p for p in [*composer.parameters(), *readout.parameters()] if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.5)
    for _ in range(2):                                     # the gate starts with w = 0: the second step reaches the query
        optimizer.zero_grad()
        addition = readout(hidden)
        assert addition is not None and addition.shape == hidden.shape
        (addition.square().sum() + (addition * hidden).sum()).backward()
        optimizer.step()
    assert torch.equal(composer.transform.phases, phases)                                 # the operator stays fixed
    assert not torch.equal(composer.atomics, atomics) and not torch.equal(readout.query.weight, query)
