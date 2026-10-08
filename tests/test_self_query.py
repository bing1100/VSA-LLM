"""The recall tool (E12, decision 62): `vsa_embed.self_query` on toy composers (CPU, random atomics)."""

import numpy as np
import pytest
import torch

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.self_query import (Filler, RecallLine, RecallStore, RecallWriter, relation_phrase, roleless, slot_accuracy,
                                  symbolic_lines)
from vsa_embed.tracks.common import RelationTemplates

R, A, D = 4, 40, 256
# entries: 0 = (0, 10) (1, 11) (2, 12); 1 = (0, 11) (1, 10) (2, 12)  [twins of 0: roles 0 and 1 swapped]
#          2 = (0, 20) (3, 5)       [atom 5 names entry 3]; 3 = (1, 30) (2, 31); 4 = (3, 5) (0, 21)
FRAMES = [[(0, 10), (1, 11), (2, 12)], [(0, 11), (1, 10), (2, 12)], [(0, 20), (3, 5)], [(1, 30), (2, 31)], [(3, 5), (0, 21)]]


def composer(operator: str) -> FrameComposer:
    torch.manual_seed(0)
    return FrameComposer(FrameSchedule.from_frames(FRAMES), A, R, D, operator=operator, mode="bundle").eval()


@pytest.mark.parametrize("operator", ["random_fixed:unitary_hrr", "hrr"])
def test_bound_store_reads_back_each_slot_and_tells_twins_apart(operator) -> None:
    store = RecallStore(composer(operator))
    for entry, frame in enumerate(FRAMES):
        lines = store.decode_slots(store.entry_vectors()[entry], [r for r, _ in frame], cleanup="all")
        assert [line.relation for line in lines] == list(dict.fromkeys(r for r, _ in frame))
        assert slot_accuracy(lines, frame) == {r: 1.0 for r, _ in frame}
        assert all(0 < f.score <= 1 for line in lines for f in line.fillers)
    twin_a = store.decode_role(store.entry_vectors()[0], 0, cleanup="all").fillers[0].atom
    twin_b = store.decode_role(store.entry_vectors()[1], 0, cleanup="all").fillers[0].atom
    assert (twin_a, twin_b) == (10, 11)
    # a frame never stored in the schedule reads back the same way (zero-shot composition from the dictionary); typed
    # cleanup needs each filler to be a candidate of its relation, all-atom cleanup does not
    frame = [(1, 10), (2, 31), (0, 20)]
    assert slot_accuracy(store.decode_slots(store.frame_vector(frame), [r for r, _ in frame]), frame) == {1: 1.0, 2: 1.0, 0: 1.0}
    odd = [(1, 12), (3, 30), (0, 10)]
    assert slot_accuracy(store.decode_slots(store.frame_vector(odd), [r for r, _ in odd]), odd) == {1: 0.0, 3: 0.0, 0: 1.0}
    assert slot_accuracy(store.decode_slots(store.frame_vector(odd), [r for r, _ in odd], cleanup="all"), odd) == {1: 1.0, 3: 1.0, 0: 1.0}


def test_frame_vector_equals_the_static_bundle() -> None:
    store = RecallStore(composer("hrr"))
    for entry, frame in enumerate(FRAMES):
        assert torch.allclose(store.frame_vector(frame), store.entry_vectors()[entry], atol=1e-5)


def test_role_blind_store_returns_the_bundle_readout_and_cannot_tell_twins_apart() -> None:
    store = RecallStore(composer("untyped"))
    assert store.role_blind and store.method == "bundle"
    lines_a = store.decode_slots(store.entry_vectors()[0], [0, 1, 2])
    lines_b = store.decode_slots(store.entry_vectors()[1], [0, 1, 2])
    assert len(lines_a) == 1 and lines_a[0].relation is None
    assert {f.atom for f in lines_a[0].fillers} == {10, 11, 12}
    assert [(f.atom, round(f.score, 6)) for f in lines_a[0].fillers] == [(f.atom, round(f.score, 6)) for f in lines_b[0].fillers]
    # a role query on an untyped store is the type-restricted bundle readout: identical for the twins
    assert store.decode_role(store.entry_vectors()[0], 0).fillers == store.decode_role(store.entry_vectors()[1], 0).fillers


def test_translation_store_subtracts_and_is_role_blind_on_twins() -> None:
    store = RecallStore(composer("translation"))
    assert store.method == "subtract"
    a = store.decode_slots(store.entry_vectors()[0], [0, 1, 2])
    b = store.decode_slots(store.entry_vectors()[1], [0, 1, 2])
    assert torch.allclose(store.entry_vectors()[0], store.entry_vectors()[1], atol=1e-5)    # the same bag of fillers + offsets
    assert [l.fillers[0].atom for l in a] == [l.fillers[0].atom for l in b]


def test_typed_cleanup_restricts_to_the_relation_candidates() -> None:
    store = RecallStore(composer("hrr"))
    scores = store.filler_scores(torch.tensor([0]), store.entry_vectors()[[0]], cleanup="typed")[0]
    allowed = {f for frame in FRAMES for r, f in frame if r == 0}
    assert {int(a) for a in torch.isfinite(scores).nonzero().flatten()} == allowed


def test_chain_follows_the_filler_to_its_own_store() -> None:
    store = RecallStore(composer("random_fixed:unitary_hrr"))
    atom_entry = np.full(A, -1); atom_entry[5] = 3
    lines, meta = store.chain(store.entry_vectors()[2], 3, 2, atom_entry)
    assert meta == {"bridge": 3, "hop1": 5, "hop2": 31}
    assert [l.relation for l in lines] == [3, 2]
    lines, meta = store.chain(store.entry_vectors()[2], 0, 2, atom_entry)          # hop 1 names no concept: the chain stops
    assert meta["bridge"] == -1 and len(lines) == 1


def test_reverse_lookup_ranks_the_holders_first() -> None:
    store = RecallStore(composer("random_fixed:unitary_hrr"))
    top = store.reverse(3, 5, store.entry_vectors(), k=2)
    assert {i for i, _ in top} == {2, 4}                    # both entries hold (3, 5)
    blind = RecallStore(composer("untyped"))
    assert {i for i, _ in blind.reverse(3, 5, blind.entry_vectors(), k=2)} == {2, 4}


def test_presence_thresholds_and_slot_free_decoding() -> None:
    store = RecallStore(composer("random_fixed:unitary_hrr"))
    thresholds = store.presence_thresholds(range(len(FRAMES)))
    assert torch.isfinite(thresholds).all()
    for entry, frame in enumerate(FRAMES):
        lines = store.decode_free(store.entry_vectors()[entry], thresholds)
        assert {l.relation for l in lines} == {r for r, _ in frame}


class Lexicon:
    templates = {"owned_by": RelationTemplates(["{x} is owned by"], "{x} is owned by {y}."),
                 "area": RelationTemplates(["{x} belongs to the"], "{x} belongs to the {y} area.", " {y} area")}
    texts = {"term:Zash Team": "the Zash Team", "area:finance": "finance", "chem:{x}-ol": "{x}-ol"}

    def text(self, atom):
        return self.texts.get(atom)

    def answer(self, relation, text):
        return self.templates[relation].answer.format(y=text)


def test_writer_words_statements_fields_and_role_blind_lines() -> None:
    relations, atoms = ["owned_by", "area", "uses"], ["term:Zash Team", "area:finance", "chem:{x}-ol"]
    writer = RecallWriter(relations, atoms, Lexicon(), home=[0, 1, -1])
    lines = [RecallLine(0, [Filler(0, 0.61)]), RecallLine(1, [Filler(1, 0.5)]), RecallLine(2, [Filler(2, 0.333)])]
    text = writer.render("zorblat", lines)
    assert text.splitlines() == ["recall(zorblat):", "- zorblat is owned by the Zash Team. (0.61)",
                                 "- zorblat belongs to the finance area. (0.50)", "- zorblat uses {x}-ol. (0.33)"]
    fields = RecallWriter(relations, atoms, Lexicon(), style="fields", confidence=False).render("zorblat", lines[:2])
    assert fields.splitlines()[1:] == ["- owned by: the Zash Team", "- area: finance area"]
    blind = writer.render("zorblat", [RecallLine(None, [Filler(1, 0.4), Filler(0, 0.3)])])
    assert blind.splitlines()[1] == "- zorblat is associated with: finance area (0.40), the Zash Team (0.30)"
    assert writer.render("zorblat", []).endswith("(nothing recalled)")
    assert relation_phrase("depends_on") == "depends on"


def test_symbolic_and_roleless_lines() -> None:
    frame = [(0, 10), (1, 11), (0, 12)]
    lines = symbolic_lines(frame)
    assert [(l.relation, [f.atom for f in l.fillers]) for l in lines] == [(0, [10, 12]), (1, [11])]
    assert all(f.score == 1.0 for l in lines for f in l.fillers)
    twin = [(0, 11), (1, 10), (0, 12)]
    a, b = roleless(symbolic_lines(frame)), roleless(symbolic_lines(twin))
    assert [f.atom for f in a[0].fillers] == [f.atom for f in b[0].fillers] == [10, 11, 12]
    assert slot_accuracy(a, frame) == {0: 1.0, 1: 1.0}
