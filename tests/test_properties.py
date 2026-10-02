"""Seeded randomized property tests for the core invariants (task B10).

`hypothesis` is not a dependency, so every property is a loop over seeded random cases: the
failing case index and seed are in the assertion message and each case is reproducible on its
own. The examples in `test_compose.py`, `test_relations.py` and `test_span_channel.py` stay as
the readable specifications; these tests re-check the same statements over random schedules,
shapes, operators and texts.
"""

from __future__ import annotations

import math
import random
import re
from typing import Callable, Iterator

import pytest
import torch
from torch import nn

from vsa_embed.compose import FrameComposer, FrameSchedule, segment_softmax, uniform_bundle
from vsa_embed.relations import RelationTransform, create_relation_transform
from vsa_embed.residual_relations import create_residual_relation
from vsa_embed.span_channel import AliasTable, CausalLinker, SpanChannel, link_batch

COMPOSER_OPERATORS = [
    "hrr", "hrr_identity", "diagonal", "map", "low_rank", "low_rank_identity", "low_rank_tied",
    "orthogonal", "untyped", "random_fixed:hrr", "random_fixed:orthogonal",
]


def cases(count: int, seed: int) -> Iterator[tuple[int, random.Random]]:
    """`count` independent seeded random generators."""
    for index in range(count):
        yield index, random.Random(seed * 10_007 + index)


def random_schedule(rng: random.Random, concepts: int, atoms: int, relations: int,
                    max_degree: int) -> FrameSchedule:
    """Random frames; collisions make some frames contain the same edge more than once."""
    return FrameSchedule.from_frames([
        [(rng.randrange(relations), rng.randrange(atoms)) for _ in range(rng.randint(1, max_degree))]
        for _ in range(concepts)
    ])


# -- FrameComposer: tau -> infinity is the uniform bundle -----------------------------------------


def test_infinite_temperature_equals_the_uniform_bundle_for_random_composers() -> None:
    gap_at_unit_temperature = 0.0
    for index, rng in cases(40, seed=1):
        torch.manual_seed(index)
        concepts, atoms, relations = rng.randint(2, 7), rng.randint(3, 9), rng.randint(1, 4)
        dimension = rng.choice([8, 9, 16, 33])
        factor = rng.choice(["induced", "free", "hybrid"])
        operator = rng.choice(COMPOSER_OPERATORS)
        schedule = random_schedule(rng, concepts, atoms, relations, max_degree=6)
        composer = FrameComposer(schedule, atoms, relations, dimension, operator=operator, mode="attentive",
                                 concept_factor=factor, context_dimension=5, key_dimension=rng.choice([4, 16]))
        with torch.no_grad():   # make the attention scores clearly non-uniform at tau = 1
            for name in ("query", "atomic_key", "context"):
                getattr(composer, name).weight.mul_(8)
            composer.relation_keys.mul_(8)
            if factor == "free":
                composer.free_factor.mul_(8)
            if factor == "hybrid":
                composer.delta.normal_()
        ids = torch.tensor([rng.randrange(concepts) for _ in range(rng.randint(1, 9))])
        context = torch.randn(ids.numel(), 5)
        reference = uniform_bundle(composer, ids)
        with torch.no_grad():
            gap_at_unit_temperature = max(gap_at_unit_temperature,
                                          float((composer.compose(ids, context) - reference).abs().max()))
            composer.log_temperature.fill_(math.log(1e9))
            rows, weights, _ = composer.compose(ids, context, return_weights=True)
        message = f"case {index}: operator={operator} factor={factor} d={dimension}"
        torch.testing.assert_close(rows, reference, atol=1e-5, rtol=1e-4, msg=message)
        torch.testing.assert_close(weights, torch.ones_like(weights), atol=1e-6, rtol=0, msg=message)
        torch.testing.assert_close(rows.norm(dim=-1), torch.ones(ids.numel()), atol=1e-5, rtol=0, msg=message)
    # Not vacuous: at the default temperature attention does move away from the uniform bundle.
    assert gap_at_unit_temperature > 1e-2


def test_attention_preserves_frame_mass_at_every_temperature() -> None:
    for index, rng in cases(25, seed=2):
        torch.manual_seed(index)
        concepts = rng.randint(2, 6)
        schedule = random_schedule(rng, concepts, 8, 3, max_degree=7)
        composer = FrameComposer(schedule, 8, 3, 16, mode="attentive",
                                 temperature=rng.choice([0.05, 1.0, 20.0]), context_dimension=4)
        ids = torch.tensor([rng.randrange(concepts) for _ in range(rng.randint(1, 8))])
        _, weights, _ = composer.compose(ids, torch.randn(ids.numel(), 4), return_weights=True)
        degrees = schedule.degrees[ids].to(weights.dtype)
        segments = torch.repeat_interleave(torch.arange(ids.numel()), schedule.degrees[ids])
        totals = torch.zeros(ids.numel()).index_add(0, segments, weights)
        torch.testing.assert_close(totals, degrees, atol=1e-5, rtol=1e-5, msg=f"case {index}")
        assert bool((weights >= 0).all())


# -- segment softmax equals a per-concept loop ----------------------------------------------------


def test_segment_softmax_equals_a_per_segment_loop_for_random_segments() -> None:
    for index, rng in cases(60, seed=3):
        torch.manual_seed(index)
        count, size = rng.randint(1, 9), rng.randint(1, 40)
        scale = rng.choice([0.1, 1.0, 30.0, 400.0])      # large scores must not overflow
        scores = torch.randn(size) * scale
        segments = torch.tensor([rng.randrange(count) for _ in range(size)])  # unsorted, may leave gaps
        result = segment_softmax(scores, segments, count)
        assert bool(torch.isfinite(result).all()), f"case {index}: non-finite output"
        message = f"case {index}: count={count} size={size} scale={scale}"
        for segment in range(count):
            mask = segments == segment
            if not bool(mask.any()):
                continue
            expected = torch.softmax(scores[mask].double(), 0).float()
            torch.testing.assert_close(result[mask], expected, atol=1e-6, rtol=1e-4, msg=message)
            torch.testing.assert_close(result[mask].sum(), torch.tensor(1.0), atol=1e-5, rtol=0, msg=message)
        shifts = torch.randn(count) * 50
        shifted = segment_softmax(scores + shifts[segments], segments, count)
        torch.testing.assert_close(shifted, result, atol=1e-4, rtol=1e-3, msg=message)   # shift invariance


# -- duplicate destination indices accumulate ------------------------------------------------------


def _reference_row(composer: FrameComposer, concept: int, repeat: dict[int, int]) -> torch.Tensor:
    """`N(Σ_e m_e T(a_e))` with edge `e` listed `repeat.get(e, 1)` times, by explicit loops."""
    schedule, atoms = composer.schedule, composer.atomic_vectors()
    total = torch.zeros(atoms.shape[1])
    for edge in range(int(schedule.offsets[concept]), int(schedule.offsets[concept + 1])):
        relation, filler = schedule.relations[edge:edge + 1], atoms[schedule.fillers[edge:edge + 1]]
        total = total + repeat.get(edge, 1) * composer.transform(relation, filler)[0]
    return torch.nn.functional.normalize(total, dim=-1)


def test_repeated_edges_in_a_frame_accumulate_with_multiplicity() -> None:
    differs = 0
    for index, rng in cases(40, seed=4):
        torch.manual_seed(index)
        atoms, relations = rng.randint(3, 8), rng.randint(1, 3)
        base = [[(rng.randrange(relations), rng.randrange(atoms)) for _ in range(rng.randint(1, 4))]
                for _ in range(rng.randint(1, 4))]
        multiplicity = [[rng.randint(1, 4) for _ in frame] for frame in base]
        expanded = [[edge for edge, times in zip(frame, counts) for _ in range(times)]
                    for frame, counts in zip(base, multiplicity)]
        operator = rng.choice(COMPOSER_OPERATORS)
        single = FrameComposer(FrameSchedule.from_frames(base), atoms, relations, 16, operator=operator)
        repeated = FrameComposer(FrameSchedule.from_frames(expanded), atoms, relations, 16, operator=operator)
        repeated.load_state_dict({k: v for k, v in single.state_dict().items()
                                  if not k.startswith("frame_")}, strict=False)
        ids = torch.arange(len(base))
        rows = repeated.compose(ids)
        offsets, repeat = single.schedule.offsets, {}
        for concept, counts in enumerate(multiplicity):
            for local, times in enumerate(counts):
                repeat[int(offsets[concept]) + local] = times
        for concept in range(len(base)):
            torch.testing.assert_close(rows[concept], _reference_row(single, concept, repeat),
                                       atol=1e-5, rtol=1e-4, msg=f"case {index} concept {concept}")
        differs += int(not torch.allclose(rows, single.compose(ids), atol=1e-4))
    assert differs > 0   # multiplicity really changed some rows; dropping duplicates would be caught


def test_repeated_concept_ids_give_identical_rows_and_accumulated_gradients() -> None:
    for index, rng in cases(30, seed=5):
        torch.manual_seed(index)
        concepts = rng.randint(2, 5)
        schedule = random_schedule(rng, concepts, 6, 3, max_degree=5)
        mode = rng.choice(["bundle", "attentive"])
        composer = FrameComposer(schedule, 6, 3, 16, operator=rng.choice(COMPOSER_OPERATORS), mode=mode)
        target = torch.randn(16)
        concept, times = rng.randrange(concepts), rng.randint(2, 6)
        repeated = torch.full((times,), concept)
        rows = composer.compose(repeated)
        torch.testing.assert_close(rows, rows[:1].expand_as(rows), msg=f"case {index}")
        per_occurrence, _, _ = composer.compose(repeated, return_weights=True)
        torch.testing.assert_close(per_occurrence, rows, atol=1e-6, rtol=1e-5, msg=f"case {index}")
        composer.zero_grad()
        (composer.compose(repeated) @ target).sum().backward()
        many = {n: p.grad.clone() for n, p in composer.named_parameters() if p.grad is not None}
        composer.zero_grad()
        (composer.compose(repeated[:1]) @ target).sum().backward()
        for name, gradient in many.items():
            torch.testing.assert_close(gradient, times * composer.get_parameter(name).grad, atol=1e-5, rtol=1e-4,
                                       msg=f"case {index} parameter {name}")


def test_spans_sharing_an_injection_position_add_up() -> None:
    for index, rng in cases(30, seed=6):
        torch.manual_seed(index)
        entries, d = rng.randint(2, 6), rng.choice([6, 8])
        schedule = random_schedule(rng, entries, 6, 2, max_degree=4)
        channel = SpanChannel(FrameComposer(schedule, 6, 2, 8), d, entry_count=entries)
        nn.init.normal_(channel.gate.weight, std=0.5)    # non-trivial, input-dependent gates
        embeddings = torch.randn(rng.randint(1, 3), rng.randint(3, 8), d)
        count = rng.randint(2, 6)
        batch = torch.tensor([rng.randrange(embeddings.shape[0]) for _ in range(count)])
        position = torch.tensor([rng.randrange(embeddings.shape[1]) for _ in range(count)])
        if rng.random() < 0.5:                         # force at least one collision
            batch[1], position[1] = batch[0], position[0]
        spans = {"batch": batch, "start": position.clone(), "end": position.clone(), "inject": position,
                 "entry": torch.tensor([rng.randrange(entries) for _ in range(count)]),
                 "confidence": torch.rand(count), "length": torch.ones(count, dtype=torch.long)}
        with torch.no_grad():
            combined = channel(embeddings, spans) - embeddings
            expected = torch.zeros_like(embeddings)
            for k in range(count):
                single = {name: value[k:k + 1] for name, value in spans.items()}
                expected += channel(embeddings, single) - embeddings
        torch.testing.assert_close(combined, expected, atol=1e-5, rtol=1e-4, msg=f"case {index}")


# -- relation adjoint: <Tx, y> = <x, T^T y> for every family --------------------------------------


def relation_families() -> list[str]:
    """Every family `create_relation_transform` accepts, read from its own error message."""
    with pytest.raises(ValueError) as error:
        create_relation_transform("__not_a_family__", 1, 4)
    families = re.findall(r"'([a-z_]+)'", str(error.value).split("choose from")[1])
    assert len(families) >= 9
    return families


def _perturbed(transform: nn.Module, rng: random.Random) -> None:
    with torch.no_grad():
        for parameter in transform.parameters():
            parameter.add_(rng.choice([0.1, 0.5]) * torch.randn_like(parameter))


@pytest.mark.parametrize("family", relation_families())
def test_adjoint_identity_holds_for_random_shapes_and_parameters(family: str) -> None:
    for index, rng in cases(25, seed=7):
        torch.manual_seed(index)
        dimension, relations = rng.choice([5, 8, 9, 16, 31]), rng.randint(1, 5)
        rank = rng.randint(1, min(dimension, 6))
        transform = create_relation_transform(family, relations, dimension, rank=rank).double()
        _perturbed(transform, rng)
        shape = rng.choice([(7,), (3, 4), (1,), (2, 1, 3)])
        ids = torch.randint(0, relations, shape)
        x, y = torch.randn(*shape, dimension).double(), torch.randn(*shape, dimension).double()
        left = (transform(ids, x) * y).sum(-1)
        right = (x * transform.adjoint(ids, y)).sum(-1)
        message = f"{family} case {index}: d={dimension} rank={rank} shape={shape}"
        torch.testing.assert_close(left.detach(), right.detach(), atol=1e-9, rtol=1e-9, msg=message)
        # The closed form must also agree with the autograd fallback of the base class.
        fallback = RelationTransform.adjoint(transform, ids, y)
        torch.testing.assert_close(transform.adjoint(ids, y).detach(), fallback.detach(), atol=1e-9, rtol=1e-9,
                                   msg=message)


@pytest.mark.parametrize("family", [
    "residual_hrr", "offset_residual_hrr", "gated_offset_residual_hrr", "basis_offset_residual_hrr",
    "basis_offset_diagonal_control", "basis_offset_rotated_diagonal_control",
])
def test_residual_family_adjoint_is_the_transpose_of_the_jacobian(family: str) -> None:
    for index, rng in cases(10, seed=8):
        torch.manual_seed(index)
        dimension, relations = rng.choice([6, 8, 9, 16]), rng.randint(1, 4)
        transform = create_residual_relation(family, relations, dimension).double()
        _perturbed(transform, rng)
        count = rng.randint(1, 6)
        ids = torch.randint(0, relations, (count,))
        source, direction = torch.randn(count, dimension).double(), torch.randn(count, dimension).double()
        y = torch.randn(count, dimension).double()
        _, jacobian_direction = torch.autograd.functional.jvp(
            lambda z: transform(z, ids), source, direction)
        left = (jacobian_direction * y).sum()
        right = (direction * transform.adjoint_at(source, ids, y)).sum()
        torch.testing.assert_close(left, right, atol=1e-8, rtol=1e-8, msg=f"{family} case {index}")


# -- SpanChannel: tokens after t never change the output at or before t -------------------------


WORDS = ["new", "york", "city", "bank", "banking", "banker", "sea", "of", "the", "ba", "nk", "newyork", "a"]
ALIASES = [("new york", 0), ("new york city", 1), ("york", 2), ("bank", 3), ("bank", 4), ("banking", 5),
           ("sea of", 6), ("city", 7)]
PIECE = re.compile(r"\S+")


def piece_offsets(text: str) -> list[tuple[int, int]]:
    """Toy subword tokenizer: words split into chunks of two characters, spaces are not tokens."""
    offsets = []
    for match in PIECE.finditer(text):
        for i in range(match.start(), match.end(), 2):
            offsets.append((i, min(i + 2, match.end())))
    return offsets


def random_text(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 10)))


def random_tail(rng: random.Random) -> str:
    """Characters that may continue the current word, start new words, or form an alias."""
    pieces = [rng.choice(WORDS + ["x", "rk", "ing", " "]) for _ in range(rng.randint(0, 5))]
    return "".join(piece if rng.random() < 0.5 else " " + piece for piece in pieces)


def make_channel(mode: str, entries: int, rng: random.Random, dimension: int) -> SpanChannel:
    composer = None
    if mode == "compose":
        composer = FrameComposer(random_schedule(rng, entries, 6, 2, max_degree=4), 6, 2, 8)
    channel = SpanChannel(composer, dimension, entry_count=entries, mode=mode)
    nn.init.normal_(channel.gate.weight, std=0.5)
    return channel


def leak_found(boundary: str, mode: str, rng: random.Random, *,
               rewrite_spans: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
               ) -> tuple[bool, int]:
    """Perturb the text after token `t`; report whether the output at positions <= t changed and
    how many links were injected at positions <= t (so a vacuous case can be told apart)."""
    table = AliasTable.from_pairs(ALIASES)
    linker = CausalLinker(table, boundary=boundary, min_subtokens=rng.choice([1, 2]))
    channel = make_channel(mode, len(table.entry_concepts), rng, 6)
    original = random_text(rng)
    offsets_original = piece_offsets(original)
    cut = rng.randrange(len(offsets_original))        # tokens 0..cut are kept
    keep_until = offsets_original[cut][1]
    tail = random_tail(rng)
    if offsets_original[cut][1] - offsets_original[cut][0] < 2 and tail[:1].strip():
        tail = " " + tail                              # a one-character last piece must not be extended
    changed = original[:keep_until] + tail
    offsets_changed = piece_offsets(changed)
    assert offsets_original[:cut + 1] == offsets_changed[:cut + 1]
    embeddings = torch.randn(1, len(offsets_original), 6)
    embeddings_changed = torch.randn(1, len(offsets_changed), 6)
    embeddings_changed[:, :cut + 1] = embeddings[:, :cut + 1]
    outputs, injected = [], 0
    with torch.no_grad():
        for string, offsets, values in ((original, offsets_original, embeddings),
                                        (changed, offsets_changed, embeddings_changed)):
            spans = link_batch(linker, [string], [offsets])
            if rewrite_spans is not None:
                spans = rewrite_spans(spans)
            injected = max(injected, int((spans["inject"] <= cut).sum()))
            outputs.append(channel(values, spans)[:, :cut + 1])
    return not torch.allclose(outputs[0], outputs[1], atol=1e-6), injected


@pytest.mark.parametrize("mode", ["compose", "free", "random"])
@pytest.mark.parametrize("boundary", ["prefix", "next_token"])
def test_perturbing_tokens_after_t_never_changes_the_channel_output_up_to_t(boundary: str, mode: str) -> None:
    exercised = 0
    for index, rng in cases(60, seed=9):
        leaked, injected = leak_found(boundary, mode, rng)
        assert not leaked, f"leak in case {index} ({boundary}, {mode})"
        exercised += injected > 0
    assert exercised > 5   # enough cases actually had a link at or before the cut


@pytest.mark.parametrize("boundary", ["prefix", "next_token"])
def test_links_at_or_before_t_depend_only_on_the_prefix_text(boundary: str) -> None:
    table = AliasTable.from_pairs(ALIASES)
    for index, rng in cases(80, seed=10):
        linker = CausalLinker(table, boundary=boundary, min_subtokens=rng.choice([1, 2, 3]))
        text = random_text(rng)
        offsets = piece_offsets(text)
        full = linker.link(text, offsets)
        cut = rng.randrange(1, len(offsets) + 1)
        prefix = linker.link(text[:offsets[cut - 1][1]], offsets[:cut])
        assert [s for s in full if s.inject_token < cut] == [s for s in prefix if s.inject_token < cut], \
            f"case {index}: {text!r} cut at token {cut}"


def test_the_leak_check_detects_injection_at_the_span_start() -> None:
    """Negative control: injecting at `s` (forbidden for causal LMs) must be caught by the harness."""
    def at_start(spans: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {**spans, "inject": spans["start"]}
    detected = sum(leak_found("prefix", "compose", rng, rewrite_spans=at_start)[0] for _, rng in cases(120, seed=11))
    assert detected > 0
