"""The learn tool (decision 63, TK-L): decomposition over a typed dictionary, proposal sources, the M3 acceptance test and
its null-world calibration (`vsa_embed.learn`)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from vsa_embed import learn as L
from vsa_embed.compose import FrameComposer, FrameSchedule

torch.set_num_threads(1)


# ---------------------------------------------------------------- a planted HRR world (oracle dictionary)


def planted_world(*, seed: int = 0, concepts: int = 240, relations: int = 4, per_relation: int = 60, dimension: int = 128,
                  degree: int = 4, noise: float = 0.5, observations: int = 12):
    """Concepts with `degree` edges over typed fillers (relation r fills from a block of atomics that overlaps the next
    relation's, so a filler can serve two relations); observations are noisy views of the full frame's store. Returns the
    dictionary, the full frames and an observation generator."""
    g = torch.Generator().manual_seed(seed)
    count = relations * per_relation
    atomics = F.normalize(torch.randn(count, dimension, generator=g), dim=-1)
    roles = F.normalize(torch.randn(relations, dimension, generator=g), dim=-1)
    rng = np.random.default_rng(seed)
    frames = {}
    for c in range(concepts):
        frame = set()
        while len(frame) < degree:
            r = int(rng.integers(relations))
            frame.add((r, (r * per_relation + int(rng.integers(3 * per_relation // 2))) % count))
        frames[c] = sorted(frame)
    candidates = L.typed_candidates(frames.values(), relations, atomics.shape[0])
    dictionary = L.Dictionary.from_roles(atomics, roles, candidates)

    def observe(world_frames, generator_seed):
        gen = torch.Generator().manual_seed(generator_seed)
        clean = dictionary.stores([world_frames[c] for c in range(concepts)])
        clean = F.normalize(clean, dim=-1)
        views = clean[:, None, :] + noise * torch.randn(concepts, observations, dimension, generator=gen) / dimension ** 0.5
        return F.normalize(views, dim=-1)

    return dictionary, frames, observe


@pytest.fixture(scope="module")
def world():
    dictionary, frames, observe = planted_world()
    erased_frames, erased = L.erase_edges(frames, frames, fraction=0.25, seed=1)
    real = observe(frames, 11)                    # the data hold every edge
    null = observe(erased_frames, 12)             # planted null: the erased edges are in no observation
    types = L.concept_types(erased_frames)
    return {"dictionary": dictionary, "frames": frames, "erased_frames": erased_frames, "erased": erased, "real": real,
            "null": null, "types": types}


def propose_and_test(w, views, *, correction="holm", null_statistics=None):
    concepts = sorted(w["frames"])
    vectors = {c: views[c, :4].mean(0) for c in concepts}
    observations = {c: views[c, 4:] for c in concepts}
    evidence = L.Evidence(w["dictionary"], w["erased_frames"], vectors=vectors, sources=("decompose",),
                          decompose=L.DecomposeSettings(max_new=2, threshold=0.05))
    proposals = L.propose(evidence)
    test = L.AcceptanceTest(L.VectorUtilityTest(w["dictionary"], w["erased_frames"], observations, types=w["types"]),
                            correction=correction, null_statistics=null_statistics)
    return proposals, L.test_proposals(proposals, test), test


# ---------------------------------------------------------------- the dictionary


def test_dictionary_binds_like_hrr_and_stores_sum_edges(world):
    d = world["dictionary"]
    frame = world["frames"][0]
    bound = d.bound([r for r, _ in frame], [a for _, a in frame])
    rel, fil, units, norms = d.atoms()
    assert units.shape[0] == int(d.candidates.sum())
    assert torch.allclose(d.store(frame), bound.sum(0), atol=1e-6)
    assert torch.allclose(d.stores([frame, []])[0], d.store(frame), atol=1e-6)
    assert float(d.stores([frame, []])[1].norm()) == 0.0
    # unbinding a bound edge recovers its filler best among the relation's typed candidates
    r, a = frame[0]
    unbound = d.unbind(torch.tensor([r]), bound[:1])
    scores = F.normalize(unbound, dim=-1) @ F.normalize(d.atomics, dim=-1).T
    scores[0, ~d.candidates[r]] = -math.inf
    assert int(scores.argmax()) == a


def test_dictionary_from_composer_uses_trained_atomics_and_operator():
    frames = [[(0, 1), (1, 2)], [(1, 0), (0, 3)], [(0, 2)]]
    composer = FrameComposer(FrameSchedule.from_frames(frames), 4, 2, 32, operator="hrr")
    d = L.Dictionary.from_composer(composer, frames[:2])
    assert d.candidates.tolist() == [[False, True, False, True], [True, False, True, False]]
    with torch.no_grad():
        static = composer.raw_bundle(torch.tensor([0]), uniform=True)[0][0]
    assert torch.allclose(d.store(frames[0]), static, atol=1e-5)
    assert d.can_unbind
    untyped = FrameComposer(FrameSchedule.from_frames(frames), 4, 2, 32, operator="untyped")
    assert not L.Dictionary.from_composer(untyped).can_unbind


# ---------------------------------------------------------------- decomposition


@pytest.mark.parametrize("method", L.METHODS)
def test_decompose_recovers_an_erased_edge_from_a_clean_store(world, method):
    d, frames = world["dictionary"], world["frames"]
    hits, total = 0, 0
    for c in range(30):
        full = frames[c]
        base = full[1:]
        out = L.decompose(d.store(full)[None], [base], d, L.DecomposeSettings(method=method, max_new=1, threshold=0.05))
        assert len(out) == 1
        assert all((e["relation"], e["filler"]) not in base for e in out[0])
        hits += int(bool(out[0]) and (out[0][0]["relation"], out[0][0]["filler"]) == full[0])
        total += 1
    assert hits / total >= 0.9


def test_decompose_respects_allowed_relations_and_reports_relative_coefficients(world):
    d, frames = world["dictionary"], world["frames"]
    out = L.decompose(d.store(frames[0])[None], [frames[0][1:]], d, L.DecomposeSettings(max_new=3, relations=(3,)))
    assert all(e["relation"] == 3 for e in out[0])
    full = L.decompose(d.store(frames[1])[None], [frames[1][1:]], d, L.DecomposeSettings(max_new=1))[0]
    assert full and 0.5 < full[0]["relative"] < 2.0          # a bound edge carries the mass of the frame's other edges


def test_crossfit_decoder_is_out_of_fold():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(60, 8, generator=g)
    w = torch.randn(8, 5, generator=g)
    y = x @ w
    oof, predict, info = L.crossfit_decoder(x, y, folds=5, seed=0)
    assert info["oof_cosine"] > 0.95 and torch.allclose(predict(x), y, atol=0.2)
    y2 = y.clone()
    y2[0] += 100.0                                        # a row's own target never fits the map that decodes it
    oof2, _, _ = L.crossfit_decoder(x, y2, folds=5, seed=0)
    assert torch.allclose(oof2[0], oof[0], atol=1e-4)


# ---------------------------------------------------------------- proposal sources and the interface


def test_propose_returns_plain_dicts_without_frame_edges(world):
    proposals, _, _ = propose_and_test(world, world["real"])
    assert proposals and all(set(L.KEYS) <= set(p) for p in proposals)
    assert all(isinstance(p["concept"], int) and isinstance(p["score"], float) for p in proposals)
    present = {(c, r, a) for c, f in world["erased_frames"].items() for r, a in f}
    assert not any(L.edge_key(p) in present for p in proposals)
    assert {p["source"] for p in proposals} == {"decompose"}


def test_rule_closure_proposes_symmetric_edges():
    # concepts 0..9; atomic i names concept i; relation 0 ("similar") is symmetric where observed
    frames = {c: [] for c in range(10)}
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        frames[a].append((0, b)); frames[b].append((0, a))
    frames[8].append((0, 9))                              # 9 lacks the symmetric edge
    for c in range(10):
        frames[c].append((1, 10 + c % 2))                 # a plain-valued relation
    graph = L.RuleGraph.from_atom_concepts(["similar", "kind"], list(range(10)) + [-1, -1])
    proposals, rules = L.rule_closure(frames, graph, settings=L.RuleSettings(min_pca=0.8, chains=False))
    assert any(r["kind"] == "symmetric" for r in rules)
    assert (9, 0, 8) in {L.edge_key(p) for p in proposals}
    assert all(p["source"] == "closure" for p in proposals)


def test_authoring_cards_become_proposals():
    cards = [{"surface": "foo bar", "relation": "uses", "filler": "x", "atom": 3, "proposals": 3, "samples": 4},
             {"surface": "unknown", "relation": "uses", "filler": "x", "atom": 3, "proposals": 4, "samples": 4},
             {"surface": "foo bar", "relation": "nope", "filler": "x", "atom": 3, "proposals": 4, "samples": 4}]
    out = L.authoring_proposals(cards, {"foo bar": 7}.get, {"uses": 2})
    assert [(p["concept"], p["relation"], p["filler"], p["score"], p["source"]) for p in out] == [(7, 2, 3, 0.75, "author")]


# ---------------------------------------------------------------- acceptance and calibration


def test_acceptance_recovers_erased_edges_with_high_precision(world):
    proposals, records, _ = propose_and_test(world, world["real"])
    accepted = [r for r in records if r["accepted"]]
    metrics = L.edge_metrics(accepted, world["erased"], proposals=proposals)
    assert metrics["precision"] >= 0.9 and metrics["recall"] >= 0.5
    assert L.accept(proposals, L.AcceptanceTest(L.VectorUtilityTest(world["dictionary"], world["erased_frames"],
                                                                    {c: world["real"][c, 4:] for c in world["frames"]},
                                                                    types=world["types"])))
    for r in records:
        assert {"utility", "n", "t", "p", "p_adjusted", "accepted", "testable"} <= set(r)


def test_null_world_calibration_controls_false_acceptance(world):
    """Planted-null data: the observations carry only the erased frames, so every proposal is false; the M3 test (head
    control, one-sided Welch t, Holm) must accept at most 5% of them. The decoy constructions (relation labels permuted
    within type, fillers swapped between same-type terms) are false by construction and controlled too."""
    proposals, records, test = propose_and_test(world, world["null"])
    rate = L.false_acceptance(records)
    assert rate["tested"] >= 200 and rate["rate"] <= 0.05
    # decoys on the real data
    real_proposals, _, real_test = propose_and_test(world, world["real"])
    full = {(c, r, a) for c, f in world["frames"].items() for r, a in f}
    for kind in L.NULL_KINDS:
        decoys = L.null_proposals(real_proposals, kind, dictionary=world["dictionary"], frames=world["erased_frames"],
                                  types=world["types"], exclude=full, seed=3)
        assert decoys and not any(L.edge_key(d) in full for d in decoys)
        assert all(d["source"] == f"null:{kind}" and len(d["null_of"]) == 3 for d in decoys)
        decoy_rate = L.false_acceptance(L.test_proposals(decoys, real_test))
        assert decoy_rate["rate"] <= 0.05, (kind, decoy_rate)


def test_uncorrected_test_is_not_calibrated(world):
    """Without the multiplicity correction the same null world accepts far more than 5%: the correction is doing the work."""
    _, records, _ = propose_and_test(world, world["null"], correction="none")
    assert L.false_acceptance(records)["rate"] > 0.05


def test_decoy_rule_never_accepts_when_targets_look_like_decoys(world):
    proposals, records, _ = propose_and_test(world, world["null"])
    null_stats = [r["utility"] for r in records if r["testable"]]
    # targets = a fresh planted-null draw; decoys = this one: nothing is credibly above the decoys
    _, again, _ = propose_and_test(world, world["null"], correction="holm+decoy", null_statistics=null_stats)
    assert L.false_acceptance(again)["rate"] <= 0.05


def test_decision_rules():
    assert L.bh_adjust([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
    assert L.decoy_threshold([5, 4, 3, 2, 1], [1.5], 0.3) == 2           # (1 + 0) / 4 ≤ 0.3 at τ = 2; (1 + 1) / 5 > 0.3 at τ = 1
    assert L.decoy_threshold([5, 4, 3, 2, 1], [4.5], 0.5) == 1           # (1 + 1) / 5 ≤ 0.5 at τ = 1
    assert math.isinf(L.decoy_threshold([1.0], [2.0, 3.0], 0.05))
    assert L.knockoff_threshold([3, 2, 1, -0.5], 0.5) == 1
    records = [{"concept": 0, "relation": 0, "filler": 0, "source": "decompose", "p": 1e-6, "t": 9.0, "utility": 0.3,
                "testable": True},
               {"concept": 1, "relation": 0, "filler": 1, "source": "decompose", "p": 1e-6, "t": 9.0, "utility": 0.01,
                "testable": True},
               {"concept": 2, "relation": 0, "filler": 2, "source": "decompose", "p": 0.5, "t": 0.1, "utility": 0.5,
                "testable": True}]
    out = L.decide([dict(r) for r in records], "holm+decoy", 0.05, threshold=0.1)
    assert [r["accepted"] for r in out] == [True, False, False]
    out = L.decide([dict(r) for r in records], "holm", 0.05)
    assert [r["accepted"] for r in out] == [True, True, False]
    with pytest.raises(ValueError):
        L.AcceptanceTest(None, correction="decoy")


def test_one_sided_statistics():
    t, p = L.one_sided(L.Contrast(np.array([0.2, 0.25, 0.3, 0.22, 0.28])))
    assert t > 5 and p < 0.001
    _, p_low = L.one_sided(L.Contrast(np.array([0.2, 0.25, 0.3, 0.22, 0.28]), np.array([0.2, 0.26, 0.24, 0.25])))
    assert p_low > 0.1
    assert L.one_sided(L.Contrast(np.array([0.1, 0.2])))[1] == 1.0       # fewer than 4 observations: untestable
    assert L.one_sided(None) == (0.0, 1.0)


def test_loss_utility_test_with_a_fake_model():
    frames = {0: [(0, 1)], 1: [(0, 2)]}
    useful = {(0, 0, 3)}

    def losses(trial):
        out = {}
        for c, frame in trial.items():
            gain = sum(0.5 for r, a in frame if (c, r, a) in useful)
            out[c] = np.full(6, 2.0 - gain) + np.linspace(0, 0.01, 6)
        return out

    candidates = torch.zeros(1, 6, dtype=torch.bool)
    candidates[0, 1:6] = True
    test = L.AcceptanceTest(L.LossUtilityTest(losses, frames, candidates=candidates, controls=2), correction="holm")
    proposals = [L.make_proposal(0, 0, 3, 1.0, "decompose"), L.make_proposal(0, 0, 4, 1.0, "decompose"),
                 L.make_proposal(1, 0, 3, 1.0, "decompose")]
    records = L.test_proposals(proposals, test)
    assert [r["accepted"] for r in records] == [True, False, False]
    assert records[0]["utility"] == pytest.approx(0.5)


# ---------------------------------------------------------------- erasure, metrics, placement


def test_erase_edges_is_seeded_and_keeps_an_edge():
    frames = {c: [(0, c), (1, c + 1), (2, c + 2)] for c in range(50)}
    a, ea = L.erase_edges(frames, range(50), fraction=0.9, seed=4)
    b, eb = L.erase_edges(frames, range(50), fraction=0.9, seed=4)
    assert ea == eb and a == b
    assert all(len(f) >= 1 for f in a.values())
    assert all((r, x) not in a[c] for c, r, x in ea)
    _, only = L.erase_edges(frames, range(50), fraction=1.0, relations={2}, seed=0)
    assert {r for _, r, _ in only} == {2}


def test_metrics():
    accepted = [L.make_proposal(0, 0, 1, 1, "decompose"), L.make_proposal(0, 0, 2, 1, "decompose")]
    gold = {(0, 0, 1), (1, 0, 1)}
    m = L.edge_metrics(accepted, gold, proposals=accepted + [L.make_proposal(1, 0, 5, 1, "decompose")])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["false_acceptance_nongold"] == 0.5
    assert L.per_concept_counts(accepted, gold, [0, 1]) == {0: (1, 1, 0), 1: (0, 0, 1)}
    assert L.recall_at_precision([0.9, 0.8, 0.7, 0.6], [True, True, False, True], 0.8) == pytest.approx(2 / 3)
    assert L.recall_at_precision([0.9, 0.8, 0.7, 0.6], [True, True, False, True], 0.75) == pytest.approx(1.0)
    assert L.recall_at_precision([0.9, 0.8, 0.7, 0.6], [False, True, False, True], 0.8) == 0.0
    assert L.roc_auc([0.9, 0.1, 0.5], [True, False, True]) == 1.0
    r = L.ranking_metrics([1, 2, float("nan"), 20])
    assert r["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.05) / 4) and r["hits@1"] == 0.25 and r["hits@10"] == 0.5
    ranks = L.gold_ranks(torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]]), [[2], [0, 1]])
    assert ranks.tolist() == [2.0, 1.0]


def test_placement_scores_rank_the_bound_parent(world):
    d = world["dictionary"]
    relation, parents = 2, list(range(120, 180))
    true = [125, 150, 170]
    vectors = d.bound([relation] * 3, true)
    for method in ("unbind", "correlate"):
        scores = L.placement_scores(vectors, relation, parents, d, method=method)
        ranks = L.gold_ranks(scores, [[parents.index(t)] for t in true])
        assert ranks.tolist() == [1.0, 1.0, 1.0]


def test_permute_within_type_is_a_derangement():
    types = {c: c % 3 for c in range(12)}
    mapping = L.permute_within_type(list(range(12)), types, seed=0)
    assert all(mapping[c] != c and types[mapping[c]] == types[c] for c in range(12))
    assert sorted(mapping.values()) == list(range(12))


def test_concept_store_adapters():
    """The facade contract (TK-E13): a proposer `(store, evidence)` and an acceptance callable `(store, proposals)`."""
    from types import SimpleNamespace

    # the store lacks each frame's last edge, whose filler another frame types under that relation
    full = [[(0, 1), (1, 2), (0, 2)], [(1, 0), (0, 2), (1, 1)], [(0, 0), (1, 1), (0, 3)], [(1, 3), (0, 3), (1, 2)]]
    seen = [f[:-1] for f in full]
    composer = FrameComposer(FrameSchedule.from_frames(seen), 4, 2, 128, operator="hrr")
    store = SimpleNamespace(composer=composer, relation_names=["r0", "r1"], entry_count=len(seen),
                            frame=lambda e: seen[e], atom_entry=np.full(4, -1))
    d = L.Dictionary.from_composer(composer, full)
    vectors = {e: d.store(full[e]) for e in range(len(full))}
    found = L.store_proposer(store, SimpleNamespace(entries=None, extra={"vectors": vectors}),
                             decompose_settings=L.DecomposeSettings(max_new=1, threshold=0.05))
    keys = {L.edge_key(L.as_dict(p)) for p in found}
    assert (0, 0, 2) in keys and all(k[1:] not in set(seen[k[0]]) for k in keys)
    assert L.as_dict(SimpleNamespace(entry=1, relation=0, atom=3, score=0.5, source="x"))["filler"] == 3
    g = torch.Generator().manual_seed(0)
    observations = {e: torch.stack([vectors[e]] * 4) + 0.01 * torch.randn(4, 128, generator=g) for e in vectors}
    test = L.AcceptanceTest(L.VectorUtilityTest(d, {e: seen[e] for e in range(4)}, observations, control="none"),
                            correction="none")
    decisions = L.store_test(test)(store, found)
    assert len(decisions) == len(found) and all(hasattr(x, "accept") and hasattr(x, "key") for x in decisions)
    assert any(x.accept for x in decisions if x.key == (0, 0, 2))
