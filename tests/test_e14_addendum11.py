"""E14 pre-registration addendum 11 (CPU): LG-CF chains of what-if edits (generator, stores, items, `cf-report`) and RD-14's
link predictor, marked recall lines and the `e14_inferred` run on the tiny E9 world."""

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_e9 import _ok, items, world  # noqa: E402,F401  (module fixtures re-used)
from test_e14_reading import heldout  # noqa: E402,F401

from vsa_embed.logic import reference as R  # noqa: E402
from vsa_embed.logic import store as S  # noqa: E402
from vsa_embed.logic import world as W  # noqa: E402
from vsa_embed.experiments import e14_inferred as inf  # noqa: E402
from vsa_embed.experiments import e14_items as items_mod  # noqa: E402
from vsa_embed.experiments import e14_score  # noqa: E402

NAMER = lambda seed=1: W.Namer(seed, forbidden=set())  # noqa: E731


# -- LG-CF ------------------------------------------------------------------------------------------------------------------

def test_cf_edits_change_the_state_and_keep_the_gold_exact() -> None:
    world = W.counterfactual_trees(2, depths=(2, 4), inputs_per_depth=6, length=8, namer=NAMER(62))
    kinds = set()
    for item in world["items"]:
        states = item["states"]
        assert len(states) == 9 and states[0]["edit"] is None
        for prev, st in zip(states, states[1:]):
            e = st["edit"]
            kinds.add(e["kind"])
            assert R.run_tree(st["tree"], st["record"])[0] == st["answer"]
            if e["kind"] == "input":
                assert st["tree"] == prev["tree"] and st["record"][e["attr"]] == e["new"] != prev["record"][e["attr"]]
            else:
                assert st["record"] == prev["record"] and st["tree"] != prev["tree"]
                texts = [W.test_text(n["test"]) for n in W.internal_nodes(prev["tree"])]
                assert texts.count(e["old"]) == 1                                                  # the edited test was unique
    assert kinds == {"input", "retarget", "swap"}
    again = W.counterfactual_trees(2, depths=(2, 4), inputs_per_depth=6, length=8, namer=NAMER(62))
    assert again == world                                                                            # seeded


def test_cf_items_stores_and_determinism(tmp_path) -> None:
    rows, stats = items_mod.cf_trees_set(2, scale=0.04, width=512, namer=NAMER(62))
    assert stats["chains"] == 12 and len(rows) == 12 * len(items_mod.CF_STEPS)                     # 4 depths x max(2, 75 x 0.04)
    for r in rows:
        assert set(r["contexts"]) == set(items_mod.CF_CONDITIONS) and len(r["options"]) == 6 and 0 <= r["gold"] < 6
        assert r["contexts"]["chain-text+exec"].startswith(r["contexts"]["chain-text"]) and "recall(" in r["contexts"]["chain-relay"]
        assert ("Changes so far" in r["contexts"]["chain-text"]) == (r["k"] > 0) and "Changes so far" not in r["contexts"]["chain-relay"]
        assert r["cf"]["exec_sym"] == r["gold"] and 0 <= r["cf"]["exec"] < 6 and len(r["cf"]["edits"]) == r["k"]
        if r["k"] == 0:
            assert r["cf"]["orig"] == r["gold"] == r["cf"]["stale"] and not r["cf"]["changed"]
    assert stats["ceilings"]["executor@8"]["accuracy"] >= 0.8
    d1 = items_mod.write_set(rows, tmp_path / "a.jsonl.gz")
    assert items_mod.write_set(rows, tmp_path / "b.jsonl.gz") == d1
    assert "trees-cf" not in items_mod.BUILDERS
    # The stable tree store reads the same values in every process.
    code = ("from vsa_embed.logic import world as W, store as S\n"
            "w = W.counterfactual_trees(2, depths=(3,), inputs_per_depth=2, length=2, namer=W.Namer(62, forbidden=set()))\n"
            "i = w['items'][1]; st = i['states'][2]\n"
            "print(S.StableTreeStore({'entity': i['entity'], 'attributes': i['attributes'], 'kind': 'tree', 'tree': st['tree'], "
            "'record': st['record']}, width=512, seed=2).execute().text(i['entity']))")
    outs = set()
    for salt in ("1", "2"):
        env = {**__import__("os").environ, "PYTHONHASHSEED": salt, "CUDA_VISIBLE_DEVICES": ""}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip())
    assert len(outs) == 1 and "Conclusion:" in next(iter(outs))


def _fake_cf_predictions(root: Path, host: str, rows, *, text_right, exec_right, seed: int) -> Path:
    """A run folder whose `chain-text` is right with probability `text_right` (falling with k) and the executor conditions with
    probability `exec_right`, the executor followed when right."""
    rng = np.random.default_rng(seed)
    run = root / host.replace("/", "-")
    run.mkdir(parents=True)
    data = run / "predictions.jsonl.gz"
    with gzip.open(data, "wt") as handle:
        for r in rows:
            for cond in items_mod.CF_CONDITIONS:
                p = text_right * (0.9 ** r["k"]) if cond == "chain-text" else exec_right
                right = rng.random() < p
                choice = r["gold"] if right else (r["gold"] + 1) % 6
                handle.write(json.dumps({"id": r["id"], "set": r["set"], "condition": cond, "choice": choice, "gold": r["gold"],
                                         "correct": int(right), "depth": r["depth"], "format": "mc", "k": r["k"], "chain": r["chain"],
                                         "cf": r["cf"], "kind": "chain", "numeric": r["numeric"]}) + "\n")
    (run / "summary.json").write_text(json.dumps({"source": {"host": host}, "predictions": str(data), "smoke": False}))
    return run


def test_cf_report_measures_drift_consistency_and_holm_over_hosts(tmp_path) -> None:
    rows, _ = items_mod.cf_trees_set(3, scale=0.3, width=512, namer=NAMER(63))
    runs = [_fake_cf_predictions(tmp_path, "h/a", rows, text_right=0.9, exec_right=0.99, seed=1),
            _fake_cf_predictions(tmp_path, "h/b", rows, text_right=0.95, exec_right=0.95, seed=2)]
    e14_score.main(["cf-report", "--runs", *map(str, runs), str(tmp_path / "missing"), "--output", str(tmp_path / "rep"), "--resamples", "200"])
    doc = json.loads((tmp_path / "rep" / "report.json").read_text())
    assert doc["missing"] == [str(tmp_path / "missing")] and set(doc["hosts"]) == {"h/a", "h/b"}
    a = doc["hosts"]["h/a"]
    assert a["primary"]["a"] == "chain-relay" and a["primary"]["mean"] > 0 and "p_holm" in a["primary"]
    assert a["accuracy"]["chain-text"]["8"] < a["accuracy"]["chain-text"]["1"]                       # the drift curve
    assert a["consistency"]["chain-relay"] > a["consistency"]["chain-text"] and a["consistency_contrast"]["n"] == a["chains"]
    assert set(a["by_k"]) == {"0", "1", "2", "4", "8"} and {c["subset"] for c in a["by_subset"]} >= {"changed from start", "unchanged"}
    assert a["follow"]["chain-relay"]["follow"] > 0.9 and a["executor_ceiling"]["8"] >= 0.8
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "Holm over hosts" in md and "consistency" in md and "## h/a" in md


# -- RD-14 ------------------------------------------------------------------------------------------------------------------

def _toy_ontology():
    """6 entries over 2 relations: entries 0–4 known, entry 5 held out; `is_a` filler 10 goes with `has_role` filler 20 in every
    known entry that has both, so has_role of a held-out frame {is_a 10} is inferred as 20."""
    frames = [[(0, 10), (1, 20)], [(0, 10), (1, 20)], [(0, 11), (1, 21)], [(0, 10), (1, 20)], [(0, 12), (1, 22)], [(0, 10), (1, 23)]]
    rel, fil, off = [], [], [0]
    for f in frames:
        for r, a in f:
            rel.append(r); fil.append(a)
        off.append(len(rel))
    return {"entry_count": 6, "offsets": off, "relations": rel, "fillers": fil, "atomic_names": [f"a{i}" for i in range(30)],
            "relation_names": ["is_a", "has_role"], "heldout_entries": [5], "synthetic_entries": []}


def test_infer_frames_votes_from_known_entries_and_excludes_heldout() -> None:
    o = _toy_ontology()
    frames, diag = inf.infer_frames({"c5": [(0, 10), (1, 23)], "empty": None}, o, neighbours=5)
    assert frames["empty"] is None
    by_rel = {r: a for r, a in frames["c5"]}
    assert by_rel[1] == 20 and by_rel[0] == 10                       # has_role from is_a 10's neighbours; is_a from has_role 23: marginal 10
    assert diag["is_a"]["precision"] == 1.0 and diag["has_role"]["precision"] == 0.0 and diag["has_role"]["lines"] == 1
    assert diag["all"]["n_edges"] if "n_edges" in diag["all"] else diag["all"]["precision"] == 0.5
    # Without the held-out exclusion the entry's own frame would vote for 23.
    o2 = {**o, "heldout_entries": []}
    assert dict((r, a) for r, a in inf.infer_frames({"c5": [(0, 10), (1, 23)]}, o2)[0]["c5"])[1] == 20   # 3 neighbours with 20 outvote 1 with 23


class _Text:
    def symbolic(self, subject, frame):
        body = [f"- {subject} r{r} a{a}. (1.00)" for r, a in frame]
        return "\n".join([f"recall({subject}):", *body]) if body else f"recall({subject}):\n- (nothing recalled)"


def test_marked_and_mixed_recall_text() -> None:
    t = _Text()
    assert inf.mark_lines("recall(x):\n- x r1 a2. (1.00)\n- x r1 a3. (0.73)") == "recall(x):\n- x r1 a2. (inferred)\n- x r1 a3. (inferred)"
    assert inf.mixed_recall(t, "x", [], [], marked=False) == "recall(x):\n- (nothing recalled)"
    assert inf.mixed_recall(t, "x", [(0, 1)], [(1, 2)], marked=True) == "recall(x):\n- x r0 a1. (1.00)\n- x r1 a2. (inferred)"
    assert inf.mixed_recall(t, "x", [(0, 1)], [], marked=True) == "recall(x):\n- x r0 a1. (1.00)"
    assert inf.mixed_recall(t, "x", [], [(1, 2)], marked=False) == "recall(x):\n- x r1 a2. (1.00)"


@pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")
def test_rd14_recall_and_report_on_the_toy_world(world, heldout, tmp_path) -> None:
    out = tmp_path / "rd14"
    inf.main(["recall", "--run", str(world["runs"]["C0p"]), "--items", str(heldout["items"]), "--output", str(out), "--frames",
              str(heldout["e11"] / "frames.jsonl"), "--style", "prose", "--device", "cpu", "--batch-size", "8", "--resamples", "50", "--smoke"])
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["conditions"]) == set(inf.CONDITIONS) and summary["null"] == "plain"
    assert summary["inference"]["all"]["precision"] is not None and "infer-nn" in summary["readers"]
    assert {(r["a"], r["b"]) for r in summary["items"]["primary"]} <= set(inf.PRIMARY)
    samples = summary["context_samples"]
    assert samples["explicit-rel:minus"].endswith("(nothing recalled)") and "(inferred)" in samples["explicit-rel-marked:infer-nn"]
    assert "(inferred)" not in samples["explicit-rel:infer-nn"] and "(1.00)" in samples["explicit-known:oracle"]
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert any("infer-nn" in (r.get("gold_in") or {}) for r in rows)
    inf.main(["report", "--runs", str(out), str(tmp_path / "missing"), "--output", str(tmp_path / "rep"), "--resamples", "50"])
    doc = json.loads((tmp_path / "rep" / "report.json").read_text())
    g = doc["groups"][0]
    assert {(r["a"], r["b"]) for r in g["primary"]} == set(inf.PRIMARY) and all("p_holm" in r for r in g["primary"])
    assert "split: inferred right" in g and "inference" in g and (tmp_path / "rep" / "report.md").read_text().startswith("# E14 RD-14")
