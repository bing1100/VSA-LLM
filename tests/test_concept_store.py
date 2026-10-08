"""The `ConceptStore` facade (methodology M2): read / write / propose / accept over one composer."""

from pathlib import Path

import numpy as np
import pytest
import torch

from vsa_embed import concept_store as cs
from vsa_embed.compose import FrameComposer, FrameSchedule

torch.set_num_threads(1)

RELATIONS = ["owned_by", "owns", "is_a"]
ATOMS = ["term:A", "term:B", "term:C", "term:T1", "term:T2", "type:x", "type:y"]


def toy(erase_c: bool = True) -> cs.ConceptStore:
    """A, B owned by T1; C owned by T2 (its owned_by edge erased when `erase_c`); T1 owns A, B; T2 owns C."""
    frames = [[(0, 3), (2, 5)], [(0, 3), (2, 5)], ([] if erase_c else [(0, 4)]) + [(2, 6)], [(1, 0), (1, 1)], [(1, 2)]]
    torch.manual_seed(0)
    composer = FrameComposer(FrameSchedule.from_frames(frames), len(ATOMS), len(RELATIONS), 64)
    ontology = {"relation_names": RELATIONS, "atomic_names": ATOMS, "entry_count": 5, "concept_names": ["A", "B", "C", "T1", "T2"],
                "entry_concepts": [(i,) for i in range(5)]}
    return cs.ConceptStore(composer, ontology)


def test_rule_closure_proposes_the_inverse_edge() -> None:
    store = toy()
    rules = cs.mined_rules(store, cs.RuleSettings(min_support=1, min_confidence=0.5))
    assert any(r["head"] == 0 and r["kind"] == "inverse_of" and r["args"] == (1,) for r in rules)
    proposals = store.propose(cs.Evidence(entries=[0, 1, 2]), settings=cs.RuleSettings(min_support=1, min_confidence=0.5))
    assert [p.key for p in proposals] == [(2, 0, 4)]                 # C owned_by T2
    assert proposals[0].meta["rule"] == "inverse_of:owns" and proposals[0].score == pytest.approx(1.0)
    assert store.propose(cs.Evidence(entries=[0, 1]), settings=cs.RuleSettings(min_support=1, min_confidence=0.5)) == []
    # a functional relation the entry already fills gets no second filler
    full = toy(erase_c=False)
    assert full.propose(settings=cs.RuleSettings(min_support=1, min_confidence=0.5)) == []


def test_write_written_and_commit_change_only_the_named_frames() -> None:
    store = toy()
    before = store.frame(0)
    with store.written({2: [(0, 4)], 1: None}):
        assert store.frame(2) == [(0, 4)] and store.frame(1) == []
        assert store.frame(0) == before
    assert store.frame(2) == [(2, 6)] and store.frame(1) == before
    assert store.write("C", [("owned_by", "term:T2"), ("is_a", "type:y")]) == 2
    assert store.frame(2) == [(0, 4), (2, 6)]
    assert store.commit([cs.Proposal(0, 1, 2), cs.Proposal(0, 0, 3)]) == 1      # the second edge exists
    assert store.frame(0) == before + [(1, 2)]


def test_read_returns_recalls_in_every_form() -> None:
    store = toy()
    frame = store.read("A")
    assert frame.kind == "frame" and [line.relation for line in frame.lines] == [0, 2]
    assert len(store.read(0, "owned_by", k=2).lines[0].fillers) == 1         # typed clean-up: owned_by's one observed filler
    role = store.read(0, "owned_by", k=2, cleanup="all")
    assert role.kind == "role" and len(role.lines[0].fillers) == 2
    chain = store.read("A", chain=("owned_by", "owns"))
    assert chain.kind == "chain" and chain.meta["hop1"] >= 0
    reverse = store.read(None, reverse=("owned_by", "term:T1"), k=3)
    assert reverse.kind == "reverse" and len(reverse.lines) == 3 and set(reverse.meta["holders"]) <= set(range(5))
    text = store.call("read", term="A")
    assert text.startswith("recall(A):")
    with pytest.raises(KeyError):
        store.call("nothing")


def test_null_proposals_keep_relation_and_type_and_are_false() -> None:
    store = toy(erase_c=False)
    proposals = [cs.Proposal(0, 0, 3), cs.Proposal(1, 2, 5)]
    nulls = cs.null_proposals(store, proposals, seed=1, exclude={(0, 0, 4)})
    edges = store.edges()
    assert len(nulls) == 1                       # A owned_by: the only other team (T2) is excluded; B is_a: type:y
    assert nulls[0].key == (1, 2, 6) and nulls[0].source == "null" and nulls[0].key not in edges


def test_holm_test_accepts_consistent_utilities_and_reports_false_acceptance(monkeypatch) -> None:
    store = toy()
    proposals = [cs.Proposal(2, 0, 4, source="rule_closure"), cs.Proposal(2, 0, 3, source="null"), cs.Proposal(0, 1, 2, source="null")]
    test = cs.HeldOutUtilityTest(lm=None, windows={})
    values = [np.array([0.5, 0.6, 0.55, 0.52]), np.array([0.1, -0.1, 0.05, -0.02]), None]
    monkeypatch.setattr(test, "utilities", lambda s, p: values)
    decisions = store.accept(proposals, test)
    assert [d.accept for d in decisions] == [True, False, False]
    assert decisions[2].extra["reason"] == "too few validation windows"
    assert decisions[0].extra["p_holm"] >= decisions[0].extra["p"]
    far = cs.false_acceptance_rate(decisions)
    assert far["null_proposals"] == 2 and far["accepted"] == 0 and far["rate"] == 0.0


def test_resolve_proposer_accepts_names_paths_and_callables() -> None:
    assert cs.resolve_proposer("rule_closure") is cs.rule_closure
    assert cs.resolve_proposer("vsa_embed.concept_store:rule_closure") is cs.rule_closure
    marker = lambda store, evidence: []                       # noqa: E731
    cs.register_proposer("test-marker", marker)
    assert cs.resolve_proposer("test-marker") is marker and cs.resolve_proposer(marker) is marker
    with pytest.raises(KeyError):
        cs.resolve_proposer("missing")


transformers = pytest.importorskip("transformers")


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


@pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")
def test_heldout_utility_test_runs_on_a_trained_lm(tmp_path: Path) -> None:
    from vsa_embed.data.corpus import TokenCorpus, build_corpus
    from vsa_embed.span_channel import AliasTable
    from vsa_embed.training.lm import load_final, train
    table = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2)])
    texts = [f"Doc {i}: hydroxychloroquine in New York and acetaminophen later, then hydroxychloroquine." for i in range(40)]
    build_corpus(texts, tmp_path / "corpus", tokenizer_name="gpt2", table=table, eos_id=50256, max_tokens=50_000, batch_texts=8, workers=2)
    ontology = {"entry_count": 3, "atomic_count": 5, "relation_count": 2, "offsets": torch.tensor([0, 2, 3, 4]),
                "relations": torch.tensor([0, 1, 0, 1]), "fillers": torch.tensor([0, 1, 2, 3]), "heldout_entries": [],
                "train_frequency": [40, 40, 40], "relation_names": ["r0", "r1"], "atomic_names": [f"a:{i}" for i in range(5)],
                "concept_names": ["h", "n", "a"], "entry_concepts": [(0,), (1,), (2,)]}
    torch.save(ontology, tmp_path / "ontology.pt")
    config = {"seed": 0, "device": "cpu", "model": {"size": "tiny", "seq_len": 32},
              "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 3, "lr": 3e-3, "warmup_tokens": 64},
              "data": {"train": str(tmp_path / "corpus"), "eval": str(tmp_path / "corpus"), "ontology": str(tmp_path / "ontology.pt"),
                       "min_subtokens": 1},
              "eval": {"windows": 2, "batch": 2, "first_tokens": 64}, "channel": {"mode": "compose", "dimension": 16, "key_dimension": 8}}
    train(config, tmp_path / "run")
    model = load_final(tmp_path / "run" / "final.pt")
    store = cs.ConceptStore(model.channel.composer, ontology, channel=model.channel)
    windows = cs.validation_windows(TokenCorpus.open(tmp_path / "corpus"), [0, 1], per_entry=4, length=24, min_subtokens=1)
    assert set(windows) == {0, 1} and all(len(w) == 4 for w in windows.values())
    proposals = [cs.Proposal(0, 0, 4), cs.Proposal(1, 1, 4), cs.Proposal(2, 0, 4)]
    entries_before = model.channel.entry_count
    test = cs.HeldOutUtilityTest(model, windows, min_windows=3)
    decisions = store.accept(proposals, test)
    assert len(decisions) == 3 and decisions[2].extra["reason"] == "too few validation windows"
    assert all(np.isfinite(d.mean) for d in decisions[:2]) and test.forwarded > 0
    assert model.channel.entry_count == entries_before and store.frame(0) == [(0, 0), (1, 1)]
