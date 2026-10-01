import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.authoring import (
    Candidate, author, bootstrap_lower, edge_utility, find_candidates, ngram_spans, parse_proposals,
)
from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.integrations.transformers import ChannelLM
from vsa_embed.span_channel import AliasTable, SpanChannel


def _tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not cached")


def tiny_lm():
    config = transformers.GPT2Config(vocab_size=50257, n_positions=128, n_embd=32, n_layer=1, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def test_parse_proposals_keeps_only_closed_vocabulary() -> None:
    text = "hypernym: Drug\n- part_of: body, organ\nfoo: bar\nHypernym : medication;x"
    assert parse_proposals(text, ["hypernym", "part_of"]) == [("hypernym", "drug"), ("part_of", "body"), ("hypernym", "medication")]


def test_ngram_spans_and_candidate_discovery() -> None:
    tok = _tokenizer()
    spans = [s for s, _, _ in ngram_spans("The zorblax engine failed.")]
    assert "zorblax engine" in spans and "The zorblax engine" in spans
    texts = ["The zorblax engine failed again."] * 6 + ["A cat sat."] * 3
    table = AliasTable.from_pairs([("cat", 0)])
    candidates = find_candidates(tiny_lm(), tok, texts, table, torch.device("cpu"), min_count=5)
    surfaces = {c.surface for c in candidates}
    assert "zorblax" in surfaces and "cat" not in surfaces
    assert all(c.subtokens >= 2 for c in candidates)


def test_author_pools_votes_and_resolves_fillers() -> None:
    candidate = Candidate("zorblax engine", [(0, 4, 18)], 3)
    texts = ["The zorblax engine failed."]
    generate = lambda prompt: ["hypernym: machine\npart_of: car", "hypernym: machine", "hypernym: unicorn"]
    atoms = {"machine": 7, "car": 9}
    edges = author(candidate, texts, generate, ["hypernym", "part_of"], atoms.get, contexts=[0], min_share=0.3)
    assert [(e["relation"], e["filler"], e["atom"], e["proposals"]) for e in edges] == [
        ("hypernym", "machine", 7, 2), ("part_of", "car", 9, 1)]


def test_edge_utility_measures_a_real_loss_change() -> None:
    tok = _tokenizer()
    texts = ["We bought a zorblax engine for the farm today and it worked well."] * 4
    table = AliasTable.from_pairs([("zorblax engine", 0)])
    schedule = FrameSchedule.from_frames([[(0, 1)]])
    composer = FrameComposer(schedule, 4, 2, 16)
    channel = SpanChannel(composer, 32, entry_count=1, gate_bias=2.0)
    lm = ChannelLM(tiny_lm(), channel)
    occurrences = [(i, 12, 26) for i in range(4)]
    result = edge_utility(lm, tok, texts, occurrences, 0, [(0, 1)], (1, 2), torch.device("cpu"))
    assert result["n"] == 4 and result["mean"] != 0.0
    assert composer.schedule.fillers.tolist() == [1]        # schedule restored
    assert bootstrap_lower([1.0, 1.0, 1.0]) == pytest.approx(1.0)
