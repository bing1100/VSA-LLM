"""E10.L runner (decision 63, TK-L): erased-edge recovery on a planted world, the null worlds, the shared pool and its
baselines, outputs, the pooled endpoint report, feature splitting and the placement evaluator (synthetic fixtures)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from vsa_embed import learn as L
from vsa_embed.experiments import e10_learn as E
from vsa_embed.kg_baselines import kge_scores, train_kge

torch.set_num_threads(1)

SMALL_WORLD = {"roots": 3, "branching": [3, 4], "attributes": 16, "wholes": 8, "parts": 20, "groups": 4, "members": 12,
               "similar_pairs_per_family": 1, "regions": 2, "countries_per_region": 2, "cities_per_country": 2}


def small_config(**override) -> dict:
    sets = ["synthetic.seeds=[5]", "baselines=[prior,correlate,lre,amie,transe]", "kge.epochs=20", "kge.dimension=16",
            "kge.negatives=4", "test.controls=4", "label=TEST", f"synthetic.world={json.dumps(SMALL_WORLD)}"]
    sets += [f"{k}={json.dumps(v)}" for k, v in override.items()]
    return E.load_config(None, sets)


@pytest.fixture(scope="module")
def synthetic_run():
    config = small_config(seed=5)
    inputs = E.synthetic_inputs(5, config["synthetic"])
    return config, inputs, E.run_erasure(inputs, config)


def test_run_erasure_structure(synthetic_run):
    config, inputs, out = synthetic_run
    s = out["summary"]
    assert s["rule"] == "holm+decoy" and set(s["decisions"]) == {"holm+decoy", "holm", "bh", "knockoff", "decoy"}
    assert s["gold"] > 0 and 0 < s["typed_coverage"] <= 1
    assert set(s["nulls"]) == {"complete", "relabel", "swap", "permute", "planted"}
    for kind, by_rule in s["nulls"].items():
        assert "knockoff" not in by_rule and by_rule["holm"]["tested"] >= 0
    assert {"decompose", "correlate", "prior", "lre", "amie", "transe"} <= set(s["pool"]["methods"])
    pool = out["pool"]
    assert pool["label"].sum() == s["pool"]["positives"] and len(pool["concept"]) == s["pool"]["rows"]
    real = [r for r in out["records"] if r["world"] == "real"]
    assert real and all(set(r["decisions"]) == set(s["decisions"]) for r in real)
    assert all(r["accepted"] == r["decisions"]["holm+decoy"] for r in real)
    assert set(out["per_concept"]) == set(s["decisions"])


def test_primary_rule_is_calibrated_on_the_planted_null(synthetic_run):
    """The pre-registered rule (Holm and the cross-fitted complete-null decoys) keeps every null world at ≤ 5% false
    acceptance in the planted world with a misfit (noisy-prior) dictionary."""
    _, _, out = synthetic_run
    for kind, by_rule in out["summary"]["nulls"].items():
        if by_rule["holm+decoy"]["tested"]:
            assert by_rule["holm+decoy"]["rate"] <= 0.05, (kind, by_rule["holm+decoy"])


def test_thresholds_are_cross_fitted():
    real = [{"concept": c, "source": "decompose", "utility": float(c % 7), "testable": True} for c in range(200)]
    decoys = [{"concept": c, "source": "decompose", "utility": 0.5, "testable": True} for c in range(200)]
    seed = 3
    thresholds = E.decoy_thresholds(real, decoys, 0.05, seed)
    # changing one half's data cannot move that half's own threshold
    half0 = [r for r in real if E.half_of(r["concept"], seed) == 0]
    bumped = [dict(r, utility=100.0) if E.half_of(r["concept"], seed) == 0 else r for r in real]
    again = E.decoy_thresholds(bumped, decoys, 0.05, seed)
    assert half0 and again[("decompose", 0)] == thresholds[("decompose", 0)]


def test_pool_has_no_curated_distractors(synthetic_run):
    _, inputs, out = synthetic_run
    pool = out["pool"]
    full = {(c, r, a) for c, f in inputs.frames.items() for r, a in f}
    for c, r, a, y in zip(pool["concept"], pool["relation"], pool["filler"], pool["label"]):
        assert bool(y) == ((int(c), int(r), int(a)) in full)


def test_run_synthetic_writes_outputs_and_report_pools_runs(tmp_path):
    config = small_config(seed=5, baselines=["prior", "correlate", "amie", "transe", "rotate", "complex"])
    summary = E.run_synthetic(config, tmp_path / "syn")
    for name in ("summary.json", "proposals.jsonl.gz", "report.md", "resolved_config.yaml", "manifest.json", "pool-s5.npz"):
        assert (tmp_path / "syn" / name).exists(), name
    assert "holm+decoy" in summary["pooled"] and "pool" in summary["pooled"]
    with gzip.open(tmp_path / "syn" / "proposals.jsonl.gz", "rt") as handle:
        worlds = {json.loads(line)["world"] for line in handle}
    assert {"real", "null:planted", "null:complete"} <= worlds
    # one track, written as `erased` runs write them: arms c2 / features / c5full x two "seeds", pooled into the endpoints
    # (amendment 1: c2 and features are one Holm family; c5full is descriptive)
    runs = []
    outs = {}
    for seed in (5, 6):
        cfg = small_config(seed=seed, baselines=["prior", "correlate", "amie", "transe", "rotate", "complex"])
        outs[seed] = E.run_erasure(E.synthetic_inputs(seed, cfg["synthetic"]), cfg)
    for arm in ("c2", "features", "c5full"):
        for seed, out in outs.items():
            folder = tmp_path / f"{arm}-s{seed}"
            folder.mkdir()
            summary = dict(out["summary"], meta={"passive": {"kind": arm}})
            E.write_json(folder / "summary.json", {"schema": E.SCHEMA, "mode": "erased", "label": "TEST", "runs": [summary]})
            np.savez_compressed(folder / "pool.npz", **out["pool"])
            E.write_json(folder / "per_concept.json", {rule: {str(k): v for k, v in t.items()} for rule, t in out["per_concept"].items()})
            runs.append(folder)
    report = E.run_report(runs, tmp_path / "report", resamples=200)
    assert report["rule"] == "holm+decoy" and report["family"] == ["c2", "features"]
    c2, features, c5 = (report["arms"][a] for a in ("c2", "features", "c5full"))
    assert c2["family"] and features["family"] and not c5["family"] and c5["L4a"]["met"] is None
    assert set(c2["L4a"]["null_rates"]) >= {"complete", "planted"} and 0 < c2["L4a"]["p_precision"] <= 1
    # identical data in both family arms: Holm over 2 doubles the smaller p (capped at 1)
    assert c2["L4a"]["p_holm"] == pytest.approx(min(1.0, 2 * c2["L4a"]["p_precision"]))
    assert set(c2["L4b"]["comparisons"]) == {"amie", "transe", "rotate", "complex"}
    for c in c2["L4b"]["comparisons"].values():
        assert c["ci_low"] <= c["mean"] <= c["ci_high"] and c["p_value"] <= c["p_holm"] <= min(1.0, 8 * c["p_value"]) + 1e-12
    assert "p_holm" not in next(iter(c5["L4b"]["comparisons"].values()))
    assert report["L4a_met"] in (True, False) and report["L4b_met"] in (True, False)
    assert (tmp_path / "report" / "report.md").read_text().startswith("# E10.L pooled endpoints")


def test_weighted_recall_matches_unweighted():
    rng = np.random.default_rng(0)
    scores, labels = rng.random(50), rng.random(50) < 0.4
    assert E.weighted_recall_at_precision(scores, labels, np.ones(50), 0.6) == pytest.approx(
        L.recall_at_precision(scores, labels, 0.6))
    doubled = np.r_[scores, scores], np.r_[labels, labels]
    assert E.weighted_recall_at_precision(scores, labels, np.full(50, 2.0), 0.6) == pytest.approx(
        L.recall_at_precision(*doubled, 0.6))


def test_load_features_splits_by_document(tmp_path):
    entries = np.repeat(np.arange(5), 12)
    documents = np.tile(np.arange(12), 5) // 2               # two occurrences per document
    vectors = np.random.default_rng(0).standard_normal((60, 6)).astype(np.float16)
    np.savez(tmp_path / "occurrences.npz", entry=entries, document=documents, position=np.arange(60), middle=vectors,
             final=vectors, meta=json.dumps({"run": "x"}))
    proposal, test, meta = E.load_features(tmp_path / "occurrences.npz", layer="middle", split_seed=1)
    assert meta == {"run": "x"} and set(proposal) == set(test) == set(range(5))
    half = {int(d): L._stable(1, "doc", int(d)) % 2 for d in documents}
    for e in range(5):
        rows = np.flatnonzero(entries == e)
        test_rows = [i for i in rows if half[int(documents[i])] == 1]
        assert test[e].shape[0] == len(test_rows)
        assert np.allclose(proposal[e].numpy(), vectors[[i for i in rows if half[int(documents[i])] == 0]].astype(np.float32).mean(0),
                           atol=1e-3)


def test_placement_items_and_evaluation(tmp_path):
    """The TK-H3L placement format (`experiments/toolkit-learn/README.md`): relation from `gold_relations`, groups from
    `meta`, candidates null = the store's typed candidates; uncovered gold and missing vectors count as not placed."""
    def item(i, gold, relation, *, split="test", t7_group="eval", kind="descriptor", candidates=None):
        return {"id": f"s:{i}", "set": "s", "record": i, "name": f"new {i}", "aliases": [f"alias {i}"], "definition": None,
                "gold_parents": gold, "gold_relations": [[relation, g] for g in gold] + [["pharmacological_action", "P0"]],
                "candidates": candidates, "evidence": {}, "meta": {"split": split, "t7_group": t7_group, "kind": kind}}

    rows = [item("n1", ["P3"], "parent"), item("n2", ["P7"], "mapped_to", t7_group="seen", kind="scr"),
            item("n3", ["Pnone"], "parent", split="dev"), item("n4", ["P2"], "parent", candidates=["P1", "P2", "P9"])]
    (tmp_path / "placement-dev.jsonl").write_text(json.dumps(rows[2]) + "\n")
    (tmp_path / "placement-test.jsonl").write_text("\n".join(json.dumps(r) for r in (rows[0], rows[1], rows[3])) + "\n")
    loaded = E.load_placement_items(tmp_path)
    assert [i["id"] for i in loaded] == ["s:n3", "s:n1", "s:n2", "s:n4"]
    assert [i["relation"] for i in loaded] == ["parent", "parent", "mapped_to", "parent"]
    assert loaded[2]["t7_group"] == "seen" and loaded[2]["kind"] == "scr" and loaded[1]["aliases"] == ["alias n1"]
    assert E.load_placement_items(tmp_path, ["placement-dev.jsonl"])[0]["split"] == "dev"
    assert [t for t, _, _ in E.mention_texts(loaded[1])] == ["We discussed new n1", "We discussed alias n1"]
    g = torch.Generator().manual_seed(0)
    atomics = F.normalize(torch.randn(10, 64, generator=g), dim=-1)
    roles = F.normalize(torch.randn(2, 64, generator=g), dim=-1)
    candidates = torch.zeros(2, 10, dtype=torch.bool)
    candidates[:, :9] = True                                    # atomic 9 is no relation's candidate
    dictionary = L.Dictionary.from_roles(atomics, roles, candidates)
    node_atom = {f"P{i}": i for i in range(10)}
    vectors = {"s:n1": dictionary.bound([0], [3])[0], "s:n2": dictionary.bound([1], [7])[0], "s:n4": dictionary.bound([0], [2])[0]}
    flat = {"text": lambda it, atoms: np.zeros(len(atoms))}
    result = E.placement_eval(loaded, vectors, dictionary, {"parent": 0, "mapped_to": 1}, node_atom, baselines=flat)
    store = result["methods"]["store"]
    assert store["n"] == 4 and store["hits@1"] == pytest.approx(3 / 4) and result["skipped"] == {"gold": 1}
    assert result["coverage"] == pytest.approx(3 / 4) and result["methods"]["text"]["mrr"] < store["mrr"]
    assert [r["candidates"] for r in result["rows"]] == [9, 9, 9, 3]
    groups = E.placement_groups(result["rows"], ["store", "text"],
                                {"primary": {"split": "test", "t7_group": "eval"}, "seen": {"t7_group": "seen"}})
    assert groups["primary"]["n"] == 2 and groups["seen"]["n"] == 1 and groups["primary"]["methods"]["store"]["mrr"] == 1.0


def test_committed_mesh_items_load():
    """The committed MeSH 2025 → 2026 sets read in the evaluator's form (TK-H3L README: 179 primary items, 26 dev)."""
    root = Path(__file__).resolve().parents[1] / "experiments/toolkit-learn/items/mesh-2025-2026-v1"
    if not root.exists():
        pytest.skip("TK-H3L items not present")
    items = E.load_placement_items(root)
    assert len(items) == 179 and sum(i["split"] == "dev" for i in items) == 26
    assert {i["relation"] for i in items} == {"parent", "mapped_to"}
    assert all(i["relation"] == ("mapped_to" if i["kind"] == "scr" else "parent") for i in items)
    assert sum(i["t7_group"] == "eval" for i in items) == 133


def test_content_relations_and_probes():
    frames = {0: [(0, 1), (1, 2)], 1: [(0, 3), (1, 2)], 2: [(0, 4), (1, 2)]}
    assert E.content_relations(frames, [0, 1, 2], 2, 3) == [0]
    erased = {(c, 0, c) for c in range(100)}
    assert E.select_probes(erased, 10, 0) == E.select_probes(erased, 10, 0) and len(E.select_probes(erased, 10, 0)) == 10
    assert E.select_probes(erased, 0, 0) == list(range(100))


def test_train_kge_minibatch_option_keeps_the_full_batch_path():
    g = torch.Generator().manual_seed(0)
    triples = torch.stack([torch.randint(20, (60,), generator=g), torch.randint(3, (60,), generator=g),
                           torch.randint(20, (60,), generator=g)], 1)
    a = train_kge(triples, 20, 3, kind="transe", dimension=8, epochs=5, negatives=4, seed=1)
    b = train_kge(triples, 20, 3, kind="transe", dimension=8, epochs=5, negatives=4, seed=1, batch_size=None)
    c = train_kge(triples, 20, 3, kind="transe", dimension=8, epochs=5, negatives=4, seed=1, batch_size=1000)
    assert torch.equal(a.entity, b.entity) and torch.equal(a.entity, c.entity)
    d = train_kge(triples, 20, 3, kind="rotate", dimension=8, epochs=3, negatives=4, seed=1, batch_size=16)
    assert torch.isfinite(kge_scores(d, triples)).all()


def test_apply_sets_and_config():
    config = E.load_config(None, ["decompose.max_new=3", "test.variants=[bh]", "label=X"])
    assert config["decompose"]["max_new"] == 3 and config["test"]["variants"] == ["bh"] and config["label"] == "X"
    assert config["test"]["correction"] == "holm+decoy" and config["erase"]["fraction"] == 0.2
    root = Path(__file__).resolve().parents[1] / "experiments/e10-self-semantics/e10-learn/configs"
    for path in sorted(root.glob("*.yaml")):
        cfg = E.load_config(path)
        assert cfg["test"]["correction"] == "holm+decoy", path


@pytest.mark.parametrize("kind", ["transe", "rotate", "complex"])
def test_kge_vector_scores_match_id_scores(kind):
    g = torch.Generator().manual_seed(1)
    triples = torch.stack([torch.randint(12, (40,), generator=g), torch.randint(2, (40,), generator=g),
                           torch.randint(12, (40,), generator=g)], 1)
    model = train_kge(triples, 12, 2, kind=kind, dimension=8, epochs=3, negatives=4, seed=0)
    heads = torch.tensor([0, 3])
    tails = [1, 2, 5]
    by_vector = E.kge_vector_scores(model, model.entity.detach()[heads], 1, tails)
    by_id = torch.stack([kge_scores(model, torch.tensor([[int(h), 1, t] for t in tails])) for h in heads])
    assert torch.allclose(by_vector, by_id, atol=1e-5)


def test_entry_means(tmp_path):
    np.savez(tmp_path / "f.npz", entry=np.array([1, 1, 2]), middle=np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 5.0]], np.float16))
    means = E.entry_means(tmp_path / "f.npz", "middle")
    assert means[1].tolist() == [2.0, 1.0] and means[2].tolist() == [5.0, 5.0]
