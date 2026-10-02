"""WP-E7 primitives: all-match corpora, streaming discovery, context splits, compute accounting,
batched verification and the first-order utility of formulation §5.2."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

import vsa_embed.authoring as authoring
from vsa_embed.authoring import (
    ComputeLedger, ValidationWindow, authoring_prompt_fewshot, compute_matched_tokens, cut_completion,
    discover_candidates, edge_mass_losses, first_order_edge_utility, pool_proposals, replace_frames, self_consistent,
    span_occurrences, split_contexts, training_flops_per_token, verify_frames,
)
from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.data.corpus import TokenCorpus
from vsa_embed.data.match_corpus import (
    MatchCorpus, add_matches, build_match_corpus, concatenate, documents_for_tokens, select_longest, strings_table, write_view,
)
from vsa_embed.integrations.transformers import ChannelLM
from vsa_embed.span_channel import AliasTable, CausalLinker, SpanChannel


def _tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not cached")


def tiny_lm(dtype=torch.float32):
    config = transformers.GPT2Config(vocab_size=50257, n_positions=128, n_embd=32, n_layer=1, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).to(dtype).eval()


STRINGS = ["new york", "new york city", "york city", "hydroxychloroquine", "ice cream", "cream", "zorblax engine"]
TEXTS = [f"In New York City {i} people ate ice cream and hydroxychloroquine near the zorblax engine of york city."
         for i in range(12)] + ["New  York has a zorblax engine.", "Plain text without aliases."]


@pytest.fixture(scope="module")
def match_root(tmp_path_factory) -> Path:
    _tokenizer()
    root = tmp_path_factory.mktemp("match")
    build_match_corpus(TEXTS, root / "full", tokenizer_name="gpt2", strings=STRINGS, eos_id=50256, max_tokens=10**6,
                       min_subtokens=2, workers=1, batch_texts=4)
    build_match_corpus(TEXTS, root / "part", tokenizer_name="gpt2", strings=STRINGS[:4], eos_id=50256, max_tokens=10**6,
                       min_subtokens=2, workers=1, batch_texts=4)
    return root


def _linker_spans(texts, table: AliasTable, tokenizer, min_subtokens=2):
    out, position = [], 0
    for text in texts:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        for span in CausalLinker(table, min_subtokens=min_subtokens).link(text, encoded["offset_mapping"]):
            out.append((span.start_token + position, span.end_token + position, span.entry, span.length))
        position += len(encoded["input_ids"]) + 1
    return sorted(out)


@pytest.mark.parametrize("active", [[0, 1, 2, 3, 4, 5, 6], [0, 2, 4], [1, 3, 5, 6], [2]])
def test_select_longest_reproduces_the_causal_linker(match_root: Path, active: list[int]) -> None:
    tok = _tokenizer()
    corpus = MatchCorpus.open(match_root / "full")
    entries = np.full(len(STRINGS), -1)
    entries[active] = [10 + k for k in range(len(active))]
    spans = select_longest(corpus.matches, entries)
    got = sorted(zip(spans["start"].tolist(), spans["end"].tolist(), spans["entry"].tolist(), spans["length"].tolist()))
    table = AliasTable({STRINGS[s]: 10 + k for k, s in enumerate(active)}, [(i,) for i in range(10 + len(active))])
    assert got == _linker_spans(TEXTS, table, tok)
    assert got, "the fixture should link something"


def test_add_matches_equals_a_single_scan_and_views_open_as_token_corpora(match_root: Path, tmp_path: Path) -> None:
    ids = add_matches(match_root / "part", STRINGS[2:], workers=1, batch_texts=4)
    part, full = MatchCorpus.open(match_root / "part"), MatchCorpus.open(match_root / "full")
    assert ids["york city"] == 2 and sorted(part.strings) == sorted(STRINGS)
    assert np.array_equal(np.asarray(part.tokens), np.asarray(full.tokens))
    remap = np.asarray([full.strings.index(s) for s in part.strings])
    as_full = sorted(zip(part.matches["token"].tolist(), part.matches["first"].tolist(),
                         remap[part.matches["string"]].tolist(), part.matches["chars"].tolist()))
    assert as_full == sorted(zip(*(full.matches[k].tolist() for k in ("token", "first", "string", "chars"))))
    entries = np.arange(len(full.strings))
    write_view(full, tmp_path / "view", entries, np.ones(len(full.strings)))
    view = TokenCorpus.open(tmp_path / "view")
    assert len(view) == len(full) and view.spans["entry"].size > 0
    assert (tmp_path / "view" / "tokens.bin").is_symlink()


def test_concatenate_keeps_whole_documents_and_shifts_matches(match_root: Path, tmp_path: Path) -> None:
    full = MatchCorpus.open(match_root / "full")
    count = documents_for_tokens(full, 30)
    assert full.documents[count - 1] < 30 <= (full.documents[count] if count < full.documents.size else len(full))
    concatenate([(full, full.documents.size, 1), (full, count, 2)], tmp_path / "mix")
    mix = MatchCorpus.open(tmp_path / "mix")
    head = int(full.documents[count])
    assert len(mix) == len(full) + 2 * head and mix.documents.size == full.documents.size + 2 * count
    assert np.array_equal(np.asarray(mix.tokens[len(full):len(full) + head]), np.asarray(full.tokens[:head]))
    inside = full.matches["token"] < head
    assert mix.matches["token"].size == full.matches["token"].size + 2 * int(inside.sum())


def test_span_occurrences_skip_stopword_edges() -> None:
    keys = {k for k, _, _ in span_occurrences("The zorblax engine of the farm failed.")}
    assert "zorblax engine" in keys and "zorblax" in keys and "farm" in keys
    assert not any(k.startswith("the ") or k.endswith(" of") or k == "the" for k in keys)
    assert "engine of the" not in keys


def test_discover_candidates_ranks_unlinked_frequent_spans() -> None:
    tok = _tokenizer()
    texts = ["We repaired the zorblax engine today and the zorblax engine purred."] * 5 + ["A cat sat on a mat."] * 5
    candidates, stats = discover_candidates(tiny_lm(), tok, texts, torch.device("cpu"), exclude={"cat", "mat"},
                                            min_count=5, min_subtokens=2, max_occurrences=4)
    surfaces = {c.surface: c for c in candidates}
    assert "zorblax engine" in surfaces and "cat" not in surfaces
    assert surfaces["zorblax engine"].count == 10 and len(surfaces["zorblax engine"].occurrences) == 4
    assert all(c.subtokens >= 2 for c in candidates) and stats["forward_tokens"] > 0
    assert candidates == sorted(candidates, key=lambda c: (-c.excess, c.surface))


def test_split_contexts_are_disjoint_and_deterministic() -> None:
    docs = [3, 3, 5, 8, 13, 21, 21, 34, 55]
    authoring_docs, validation_docs = split_contexts(docs, authoring=3, key="zorblax", seed=1)
    assert len(authoring_docs) == 3 and not set(authoring_docs) & set(validation_docs)
    assert set(authoring_docs) | set(validation_docs) == set(docs)
    assert (authoring_docs, validation_docs) == split_contexts(docs, authoring=3, key="zorblax", seed=1)
    few_a, few_v = split_contexts([4, 9], authoring=4, key="x", seed=0)
    assert len(few_a) == 1 and len(few_v) == 1


def test_prompt_pooling_and_self_consistency() -> None:
    prompt = authoring_prompt_fewshot("zorblax engine", "The zorblax engine failed.", ["is_a", "part_of"],
                                      [{"context": "A tabby purred.", "surface": "tabby", "edges": [("is_a", "cat")]}])
    assert prompt.endswith("Concept: zorblax engine\n") and "is_a: cat" in prompt
    assert cut_completion("is_a: machine\npart_of: car\n\nText: other") == "is_a: machine\npart_of: car"
    votes, samples = pool_proposals([["is_a: machine\nis_a: machine", "is_a: engine"], ["is_a: machine\nfoo: bar"]],
                                    ["is_a", "part_of"])
    assert samples == 3 and votes[("is_a", "machine")] == 2
    kept = self_consistent(votes, samples, min_share=0.5)
    assert [(e["relation"], e["filler"]) for e in kept] == [("is_a", "machine")]


def test_compute_matched_tokens_convert_flops_to_training_tokens() -> None:
    ledger = ComputeLedger()
    ledger.add("discovery", parameters=1e6, forward_tokens=1000)
    ledger.add("authoring", parameters=1e6, prompt_tokens=300, generated_tokens=100)
    ledger.add("teacher", parameters=0, prompt_tokens=10**9)
    assert ledger.flops() == pytest.approx(2e6 * 1400)
    frozen = compute_matched_tokens(5000, ledger, parameters=1e6, host_mode="frozen", stages=["discovery", "authoring"])
    assert frozen["extra_tokens"] == 700 and frozen["total_tokens"] == 5700        # 2N·1400 / 4N
    trained = compute_matched_tokens(5000, ledger, parameters=1e6, host_mode="train", stages=["discovery"])
    assert trained["extra_tokens"] == 334                                            # ceil(2N·1000 / 6N)
    assert training_flops_per_token(10, "lora", trainable=2) == 44
    assert ComputeLedger.from_json(ledger.to_json()).flops() == ledger.flops()


def _channel_lm(dtype=torch.float32, frames=None, entries=4):
    schedule = FrameSchedule.from_frames(frames or [[(0, 1)], [(1, 2)], [], []])
    composer = FrameComposer(schedule, 8, 2, 16)
    channel = SpanChannel(composer, 32, entry_count=entries, gate_bias=2.0)
    return ChannelLM(tiny_lm(dtype), channel).to(dtype).eval()


def _windows(tok, candidate: int, count: int, length: int = 24) -> list[ValidationWindow]:
    text = "We bought a zorblax engine for the farm today and it worked very well indeed, said the old farmer."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    windows = []
    for k in range(count):
        piece = np.asarray((ids * 3)[k:k + length], dtype=np.int64)
        empty = {key: np.zeros(0, dtype=np.int64) for key in ("start", "end", "inject", "entry", "length")}
        empty["confidence"] = np.zeros(0, dtype=np.float32)
        windows.append(ValidationWindow(piece, empty, candidate, 6, 8))
    return windows


def test_verify_frames_accepts_greedily_and_restores_the_schedule(monkeypatch) -> None:
    tok = _tokenizer()
    lm = _channel_lm()
    composer = lm.channel.composer
    original = composer.schedule

    def fake_losses(model, windows, entries, device, *, after, batch):
        # the loss falls by 0.5 for every atom-5 edge of the linked entry's frame, rises for atom 6
        out = []
        for w, e in zip(windows, entries):
            frame = [] if e is None else composer.schedule.fillers[composer.schedule.offsets[e]:composer.schedule.offsets[e + 1]].tolist()
            out.append(3.0 - 0.5 * frame.count(5) + 0.5 * frame.count(6) + 0.01 * (w.start % 3))
        return np.asarray(out), len(windows) * 10

    monkeypatch.setattr(authoring, "window_losses", fake_losses)
    windows = {0: _windows(tok, 0, 6), 1: _windows(tok, 1, 2)}
    proposals = {0: [(0, 6), (0, 5), (1, 5)], 1: [(0, 5)]}
    records, forwarded = verify_frames(lm, windows, proposals, {0: 2, 1: 3}, torch.device("cpu"), min_validation=4)
    assert [r["accepted"] for r in records[0]] == [False, True, True]
    assert records[0][2]["mean"] == pytest.approx(0.5)            # scored against the frame accepted so far
    assert records[1][0]["reason"] == "too few validation contexts" and not records[1][0]["accepted"]
    assert composer.schedule.fillers.tolist() == original.fillers.tolist() and forwarded > 0


def test_replace_frames_rebuilds_the_csr() -> None:
    schedule = FrameSchedule.from_frames([[(0, 1)], [(1, 2), (0, 3)], [], [(1, 4)]])
    new = replace_frames(schedule, {2: [(0, 7), (1, 6)], 1: []})
    expected = FrameSchedule.from_frames([[(0, 1)], [], [(0, 7), (1, 6)], [(1, 4)]])
    for key in ("offsets", "relations", "fillers"):
        assert getattr(new, key).tolist() == getattr(expected, key).tolist()


def test_window_losses_change_only_when_the_span_is_linked() -> None:
    tok = _tokenizer()
    lm = _channel_lm(frames=[[(0, 1)], [(1, 2)], [(0, 3)], []])
    windows = _windows(tok, 0, 3)
    unlinked, _ = authoring.window_losses(lm, windows, [None] * 3, torch.device("cpu"))
    linked, _ = authoring.window_losses(lm, windows, [2] * 3, torch.device("cpu"))
    plain, _ = authoring.window_losses(ChannelLM(tiny_lm(), None), windows, [None] * 3, torch.device("cpu"))
    assert np.allclose(unlinked, plain) and not np.allclose(unlinked, linked)


def _float64_loss(hidden, output_weight, labels, *, chunk=2048, output_bias=None, reduction="mean"):
    """The chunked loss without its float32 cast of the logits (finite differences need float64)."""
    logits = torch.nn.functional.linear(hidden[:, :-1], output_weight, output_bias)
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                               ignore_index=-100, reduction="none")
    return losses.view(labels.shape[0], labels.shape[1] - 1) if reduction == "none" else losses.mean()


def test_first_order_utility_equals_the_finite_difference_loss_change(monkeypatch) -> None:
    """U_{j,e} = −∂L/∂β at β = 0 matches (L(0) − L(β)) / β for small β (formulation §5.2)."""
    import vsa_embed.integrations.transformers as integration
    monkeypatch.setattr(integration, "chunked_causal_lm_loss", _float64_loss)
    tok = _tokenizer()
    lm = _channel_lm(torch.float64, frames=[[(0, 1)], [(1, 2)], [(0, 3), (1, 4)], []])
    windows = _windows(tok, 0, 5)
    device = torch.device("cpu")
    utility = first_order_edge_utility(lm, windows, 2, (1, 6), device)
    beta = 1e-5
    base = edge_mass_losses(lm, windows, 2, (1, 6), 0.0, device)
    moved = edge_mass_losses(lm, windows, 2, (1, 6), beta, device)
    finite = (base - moved) / beta
    assert np.allclose(utility["values"], finite, rtol=1e-3, atol=1e-7)
    assert abs(utility["mean"]) > 1e-6
    # β = 0 is the frame as is: the same loss as linking the entry normally
    linked, _ = authoring.window_losses(lm, windows, [2] * 5, device)
    assert np.allclose(base, linked)
