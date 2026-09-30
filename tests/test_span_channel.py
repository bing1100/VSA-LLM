import pytest
import torch

from vsa_embed.compose import FrameComposer
from vsa_embed.span_channel import AliasTable, CausalLinker, SpanChannel, cardinality_report, link_batch


def char_offsets(text: str) -> list[tuple[int, int]]:
    """A toy tokenizer: every 3 characters is one token (spaces included)."""
    return [(i, min(i + 3, len(text))) for i in range(0, len(text), 3)]


def word_piece_offsets(text: str) -> list[tuple[int, int]]:
    """Toy subword tokenizer: split words into chunks of 2 letters; spaces are not tokens."""
    offsets = []
    for match in __import__("re").finditer(r"\S+", text):
        for i in range(match.start(), match.end(), 2):
            offsets.append((i, min(i + 2, match.end())))
    return offsets


def table() -> AliasTable:
    pairs = [("new york", 0), ("new_york_city", 1), ("york", 2), ("bank", 3), ("bank", 4), ("aspirin", 5)]
    return AliasTable.from_pairs(pairs, holdout=[5])


def test_polysemous_aliases_share_a_union_entry_and_holdout_is_removed() -> None:
    t = table()
    assert "aspirin" not in t.alias_to_entry
    assert t.entry_concepts[t.alias_to_entry["bank"]] == (3, 4)
    frames = [[(0, 0)], [(0, 1)], [(1, 2)], [(0, 3), (1, 1)], [(0, 3), (2, 5)], [(0, 9)]]
    schedule = t.entry_schedule(frames)
    bank = t.alias_to_entry["bank"]
    start, end = int(schedule.offsets[bank]), int(schedule.offsets[bank + 1])
    assert end - start == 3  # (0,3) deduplicated
    with_holdout = AliasTable.from_pairs([("aspirin", 5)], holdout=[5], include_holdout=True)
    assert "aspirin" in with_holdout.alias_to_entry


def test_linking_is_prefix_causal() -> None:
    linker = CausalLinker(table(), min_subtokens=1)
    text = "we saw new york city and banking"
    offsets = word_piece_offsets(text)
    full = linker.link(text, offsets)
    for cut in range(1, len(offsets)):
        prefix_text = text[:offsets[cut - 1][1]]
        prefix = linker.link(prefix_text, offsets[:cut])
        assert [s for s in full if s.inject_token < cut] == prefix


def test_longest_backward_match_and_no_right_boundary_in_prefix_mode() -> None:
    linker = CausalLinker(table(), min_subtokens=1)
    text = "new york city bank banking"
    offsets = word_piece_offsets(text)
    spans = linker.link(text, offsets)
    entries = [linker.table.entry_concepts[s.entry] for s in spans]
    assert (0,) in entries and (1,) in entries           # "new york" at york, "new york city" at city
    assert entries.count((3, 4)) == 2                     # "bank" and the prefix of "banking"
    strict = CausalLinker(table(), boundary="next_token", min_subtokens=1).link(text, offsets)
    strict_entries = [linker.table.entry_concepts[s.entry] for s in strict]
    assert strict_entries.count((3, 4)) == 1 and all(s.inject_token == s.end_token + 1 for s in strict)


def test_min_subtokens_filters_short_spans() -> None:
    text = "york and new york"
    offsets = word_piece_offsets(text)
    long_only = CausalLinker(table(), min_subtokens=3).link(text, offsets)
    assert all(s.length >= 3 for s in long_only)
    assert len(CausalLinker(table(), min_subtokens=1).link(text, offsets)) > len(long_only)


def test_channel_is_identity_where_nothing_is_linked_and_respects_causality() -> None:
    torch.manual_seed(0)
    t = table()
    schedule = t.entry_schedule([[(0, 0)], [(0, 1)], [(1, 2)], [(0, 3)], [(1, 3)], [(0, 4)]])
    composer = FrameComposer(schedule, 5, 2, 8)
    channel = SpanChannel(composer, 6, entry_count=len(t.entry_concepts), gate_bias=0.0)
    linker = CausalLinker(t, min_subtokens=1)
    text = "the new york bank"
    offsets = word_piece_offsets(text)
    embeddings = torch.randn(1, len(offsets), 6)
    spans = link_batch(linker, [text], [offsets])
    out = channel(embeddings, spans)
    injected = set(spans["inject"].tolist())
    for position in range(len(offsets)):
        same = torch.allclose(out[0, position], embeddings[0, position])
        assert same == (position not in injected)
    changed_text = "the new york bonk"
    changed = channel(embeddings, link_batch(linker, [changed_text], [word_piece_offsets(changed_text)]))
    last_bank = int(spans["inject"].max())
    torch.testing.assert_close(changed[0, :last_bank], out[0, :last_bank])


@pytest.mark.parametrize("mode", ["free", "random", "hashed"])
def test_control_modes_produce_vectors(mode: str) -> None:
    channel = SpanChannel(None, 6, entry_count=4, mode=mode, hashed_buckets=16, gate_bias=0.0)
    spans = {"batch": torch.tensor([0]), "start": torch.tensor([1]), "end": torch.tensor([2]),
             "inject": torch.tensor([2]), "entry": torch.tensor([3]), "confidence": torch.tensor([1.0]),
             "length": torch.tensor([2])}
    out = channel(torch.zeros(1, 4, 6), spans, input_ids=torch.arange(4)[None])
    assert out[0, 2].abs().sum() > 0 and out[0, :2].abs().sum() == 0


def test_semantic_loss_runs() -> None:
    t = table()
    schedule = t.entry_schedule([[(0, 0)], [(0, 1)], [(1, 2)], [(0, 3)], [(1, 3)], [(0, 4)]])
    channel = SpanChannel(FrameComposer(schedule, 5, 2, 8), 6, entry_count=len(t.entry_concepts), semantic_dimension=6)
    spans = {"batch": torch.tensor([0, 0]), "start": torch.tensor([1, 3]), "end": torch.tensor([2, 4]),
             "inject": torch.tensor([2, 4]), "entry": torch.tensor([0, 1]), "confidence": torch.ones(2),
             "length": torch.tensor([2, 2])}
    loss = channel.semantic_loss(torch.randn(1, 6, 6), spans)
    assert loss.item() > 0


def test_gpt2_offsets_align_and_cardinality_matches_brute_force() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not cached")
    t = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("cat", 2)])
    texts = ["Hydroxychloroquine was studied in New York.", "The cat sat; hydroxychloroquine again."]
    linker = CausalLinker(t, min_subtokens=1)
    for text in texts:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        for span in linker.link(text, encoded["offset_mapping"]):
            start = encoded["offset_mapping"][span.start_token][0]
            end = encoded["offset_mapping"][span.end_token][1]
            assert text[start:end].strip().lower().lstrip() in t.alias_to_entry
    report = {row["min_subtokens"]: row for row in cardinality_report(t, tokenizer, texts, thresholds=(1, 2))}
    assert report[1]["span_occurrences"] == 4 and report[1]["linked_entries"] == 3
    assert report[2]["linked_entries"] == 2  # "cat" is one GPT-2 token


def test_training_view_keeps_entry_ids_and_drops_heldout_aliases() -> None:
    full = AliasTable.from_pairs([("bank", 0), ("bank", 1), ("river", 2), ("aspirin", 3)], holdout=[3],
                                 include_holdout=True)
    train = full.without_holdout()
    assert "aspirin" not in train.alias_to_entry and "aspirin" in full.alias_to_entry
    assert train.alias_to_entry["bank"] == full.alias_to_entry["bank"]
    assert full.heldout_entries() == {full.alias_to_entry["aspirin"]}
