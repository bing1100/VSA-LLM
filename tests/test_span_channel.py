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


def test_dollar_signs_in_text_do_not_break_the_trie() -> None:
    t = AliasTable.from_pairs([("us$", 0), ("cash", 1)])
    linker = CausalLinker(t, min_subtokens=1)
    text = "it cost us$ 5 in $$ cash"
    spans = linker.link(text, word_piece_offsets(text))
    assert {t.entry_concepts[s.entry] for s in spans} == {(0,), (1,)}


def test_skip_empty_frames_injects_nothing_for_edgeless_entries_and_keeps_the_rest() -> None:
    torch.manual_seed(0)
    t = table()
    # concept 2 ("york") has an empty frame; it is entry 3 (entries are unions of concepts)
    schedule = t.entry_schedule([[(0, 0)], [(0, 1)], [], [(0, 3)], [(1, 3)], [(0, 4)]])
    composer = FrameComposer(schedule, 5, 2, 8)
    channel = SpanChannel(composer, 6, entry_count=len(t.entry_concepts), gate_bias=0.0)
    spans = {"batch": torch.tensor([0, 0]), "start": torch.tensor([0, 2]), "end": torch.tensor([1, 3]),
             "inject": torch.tensor([1, 3]), "entry": torch.tensor([3, 0]), "confidence": torch.tensor([1.0, 1.0]),
             "length": torch.tensor([2, 2])}
    assert schedule.degrees[3] == 0 and schedule.degrees[0] > 0
    embeddings = torch.randn(1, 5, 6)
    with pytest.raises(ValueError, match="at least one edge"):
        channel(embeddings, spans)                       # default: unchanged behaviour
    channel.skip_empty_frames = True
    out = channel(embeddings, spans)
    torch.testing.assert_close(out[0, 1], embeddings[0, 1])          # empty frame: no injection
    only = {k: v[1:] for k, v in spans.items()}
    torch.testing.assert_close(out, channel(embeddings, only))        # the other span is exactly as without it
    assert not torch.allclose(out[0, 3], embeddings[0, 3])


# -- HRRBERT's hybrids (decision 65): compose_add (HRRAdd) and compose_cat (HRRCat) ------------------------------------

FRAMES = [[(0, 0), (1, 1)], [(0, 2)], [(1, 3), (0, 4)], [(1, 0)], [(0, 1), (1, 2)], [(1, 4)]]
SPANS = {"batch": torch.tensor([0, 0, 0]), "start": torch.tensor([0, 2, 4]), "end": torch.tensor([1, 3, 5]),
         "inject": torch.tensor([1, 3, 5]), "entry": torch.tensor([0, 2, 5]), "confidence": torch.ones(3),
         "length": torch.tensor([2, 2, 2])}


def hybrid(mode: str, free_dimension: int = 3, frames: list | None = None) -> SpanChannel:
    from vsa_embed.compose import FrameSchedule
    torch.manual_seed(0)
    composer = FrameComposer(FrameSchedule.from_frames(frames or FRAMES), 5, 2, 8, mode="attentive", key_dimension=4)
    return SpanChannel(composer, 6, entry_count=len(frames or FRAMES), mode=mode, free_dimension=free_dimension, gate_bias=0.0)


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_hybrid_rows_shapes_and_parameter_counts() -> None:
    add, cat, same = hybrid("compose_add"), hybrid("compose_cat"), hybrid("compose_add", free_dimension=0)
    composer, gate = _count(add.composer), 2 * 6 + 1 + 1
    assert add.table.weight.shape == (6, 3) and add.free_lift.weight.shape == (8, 3) and add.projector.weight.shape == (6, 8)
    assert _count(add) == composer + 6 * 3 + 3 * 8 + 8 * 6 + gate
    assert cat.free_lift is None and cat.projector.weight.shape == (6, 11)
    assert _count(cat) == composer + 6 * 3 + 11 * 6 + gate
    assert same.free_lift is None and same.table.weight.shape == (6, 8)        # equal widths: summed directly
    for channel in (add, cat, same):
        rows = channel.rows(SPANS)
        assert rows.shape == (3, 6) and torch.isfinite(rows).all()
        out = channel(torch.zeros(1, 7, 6), SPANS)
        assert out[0, [1, 3, 5]].abs().sum(-1).min() > 0 and out[0, [0, 2, 4, 6]].abs().sum() == 0


def test_hybrid_rows_are_the_documented_sum_and_concatenation() -> None:
    add, cat = hybrid("compose_add"), hybrid("compose_cat")
    entries = SPANS["entry"]
    torch.testing.assert_close(add.rows(SPANS), add.projector(add.composer.compose(entries) + add.free_lift(add.table(entries))))
    torch.testing.assert_close(cat.rows(SPANS), cat.projector(torch.cat([cat.table(entries), cat.composer.compose(entries)], -1)))
    big = SpanChannel(add.composer, 6, entry_count=6, mode="compose_add", free_dimension=3)
    assert big.free_lift.weight.norm(dim=0).mean().item() == pytest.approx(1.0, abs=0.4)   # near-isometric lift


@pytest.mark.parametrize("mode", ["compose_add", "compose_cat"])
def test_gradients_reach_the_composed_and_the_free_part(mode: str) -> None:
    channel = hybrid(mode)
    channel(torch.randn(1, 7, 6), SPANS).pow(2).sum().backward()
    assert channel.composer.atomics.grad.abs().sum() > 0 and channel.projector.weight.grad.abs().sum() > 0
    grad = channel.table.weight.grad
    assert grad[[0, 2, 5]].abs().sum(-1).min() > 0 and grad[[1, 3, 4]].abs().sum() == 0   # only the linked entries' rows
    if mode == "compose_add":
        assert channel.free_lift.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["compose_add", "compose_cat"])
def test_held_out_entries_read_the_c2_fallback_and_compose_zero_shot(mode: str) -> None:
    channel = hybrid(mode)
    channel.set_unseen([2, 5])
    fallback = channel.table.weight[[0, 1, 3, 4]].mean(0)
    free = channel._free_rows(torch.tensor([2, 5, 0]))
    torch.testing.assert_close(free[0], fallback)
    torch.testing.assert_close(free[1], fallback)
    torch.testing.assert_close(free[2], channel.table.weight[0])
    rows = channel.rows({"entry": torch.tensor([2, 5])})
    assert not torch.allclose(rows[0], rows[1])                   # same free part: the difference is the composed part
    channel(torch.randn(1, 7, 6), SPANS).pow(2).sum().backward()
    assert channel.table.weight.grad[[2, 5]].abs().sum() == 0     # a held-out row never trains (as C2's)
    with torch.no_grad():
        before = channel.rows({"entry": torch.arange(6)})
        ids = channel.add_entries(1, [[(0, 3), (1, 1)]])           # zero-shot insertion: composed + fallback free row
        assert ids.tolist() == [6] and bool(channel.unseen[6]) and channel.table.weight.shape[0] == 7
        torch.testing.assert_close(channel.rows({"entry": torch.arange(6)}), before)   # existing rows unchanged
        composed = channel.composer.compose(torch.tensor([6]))
        expected = (channel.projector(composed + channel.free_lift(fallback[None])) if mode == "compose_add"
                    else channel.projector(torch.cat([fallback[None], composed], -1)))
        torch.testing.assert_close(channel.rows({"entry": torch.tensor([6])}), expected)


def test_hybrid_skip_empty_frames_keeps_the_free_part() -> None:
    channel = hybrid("compose_cat", frames=[[(0, 0)], [], [(1, 2)]])
    channel.skip_empty_frames = True
    rows = channel.rows({"entry": torch.tensor([1, 0])})
    torch.testing.assert_close(rows[0], channel.projector(torch.cat([channel.table.weight[1], torch.zeros(8)])))
    torch.testing.assert_close(rows[1], channel.rows({"entry": torch.tensor([0])})[0])


def test_existing_modes_keep_their_state_and_rows() -> None:
    from vsa_embed.compose import FrameSchedule
    torch.manual_seed(0)
    composer = FrameComposer(FrameSchedule.from_frames(FRAMES), 5, 2, 8, mode="attentive", key_dimension=4)
    compose = SpanChannel(composer, 6, entry_count=6, gate_bias=0.0)
    assert sorted(compose.state_dict()) == sorted(["projector.weight", "gate.weight", "gate.bias"]
                                                  + [f"composer.{k}" for k in composer.state_dict()])
    torch.testing.assert_close(compose.rows(SPANS), compose.projector(composer.compose(SPANS["entry"])))
    free = SpanChannel(None, 6, entry_count=6, mode="free", free_dimension=3)
    assert sorted(free.state_dict()) == ["free_projector.weight", "gate.bias", "gate.weight", "table.weight", "unseen"]
    free.set_unseen([2])
    expected = free.table.weight[SPANS["entry"]].clone()
    expected[1] = free.table.weight[[0, 1, 3, 4, 5]].mean(0)
    torch.testing.assert_close(free.rows(SPANS), free.free_projector(expected))
    with pytest.raises(ValueError, match="needs a FrameComposer"):
        SpanChannel(None, 6, entry_count=6, mode="compose_add")
