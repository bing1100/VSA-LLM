"""Claim-C statistics (decision 56): the crossed concept × seed model, the two-way cluster bootstrap, the power formulas,
and `e9_power`'s per-concept loaders, comparisons, power tables and rendering on synthetic run folders."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from vsa_embed.experiments import e9_power as power
from vsa_embed.experiments.e9_report import Group
from vsa_embed.statistics import crossed_components, power_at, required_clusters, two_way_cluster_bootstrap


def test_crossed_components_recovers_an_additive_table() -> None:
    a = np.asarray([0.3, -0.1, 0.0, 0.2, -0.4, 0.1])
    b = np.asarray([0.05, -0.02, 0.01])
    table = 0.07 + a[:, None] + b[None, :]                          # no residual
    c = crossed_components(table)
    assert c["mean"] == pytest.approx(0.07 + a.mean() + b.mean())
    assert c["var_cluster"] == pytest.approx(a.var(ddof=1)) and c["var_seed"] == pytest.approx(b.var(ddof=1))
    assert c["var_residual"] == pytest.approx(0.0, abs=1e-12)
    assert c["se"] == pytest.approx(math.sqrt(a.var(ddof=1) / 6 + b.var(ddof=1) / 3))
    assert c["sd_cluster_mean"] == pytest.approx(a.std(ddof=1)) and c["seed_means"] == pytest.approx((0.07 + a.mean() + b).tolist())
    assert c["ci_low"] < c["mean"] < c["ci_high"] and 0 <= c["p_value"] <= 1


def test_crossed_components_matches_the_one_sample_t_and_large_samples() -> None:
    rng = np.random.default_rng(1)
    one = rng.normal(0.1, 1.0, size=(40, 1))
    c = crossed_components(one)
    t = stats.ttest_1samp(one[:, 0], 0.0)
    assert c["seeds"] == 1 and c["df"] == 39 and c["se"] == pytest.approx(one.std(ddof=1) / math.sqrt(40))
    assert c["p_value"] == pytest.approx(t.pvalue) and c["var_seed"] is None
    big = 0.05 + rng.normal(0, 0.3, (4000, 1)) + rng.normal(0, 0.05, (1, 5)) + rng.normal(0, 0.2, (4000, 5))
    c = crossed_components(big)
    assert c["var_cluster"] == pytest.approx(0.09, rel=0.08) and c["var_residual"] == pytest.approx(0.04, rel=0.05)
    assert c["seeds"] == 5 and 1 < c["df"] < 4000          # the seed component dominates the Satterthwaite df
    no_seed = 0.05 + rng.normal(0, 0.3, (300, 1)) + rng.normal(0, 0.2, (300, 3))
    c = crossed_components(no_seed)
    if c["var_seed"] == 0:                                   # truncated: Var(μ̂) = MS_c / (n S), df = n − 1
        assert c["se"] == pytest.approx(math.sqrt(c["ms_cluster"] / 900)) and c["df"] == pytest.approx(299)
    with pytest.raises(ValueError):
        crossed_components(np.zeros((1, 3)))


def test_two_way_cluster_bootstrap() -> None:
    rng = np.random.default_rng(2)
    table = 0.1 + rng.normal(0, 0.2, (200, 3))
    a = two_way_cluster_bootstrap(table, resamples=500, seed=3)
    assert a == two_way_cluster_bootstrap(table, resamples=500, seed=3)          # deterministic
    assert a["mean"] == pytest.approx(table.mean()) and a["ci_low"] < a["mean"] < a["ci_high"] and a["p_value"] < 0.01
    assert two_way_cluster_bootstrap(np.zeros((5, 2)), resamples=50)["p_value"] == 1.0
    null = two_way_cluster_bootstrap(rng.normal(0, 1, (100, 3)), resamples=500)
    assert null["ci_low"] < 0.0 < null["ci_high"] or null["p_value"] < 0.05


def test_required_clusters_and_power() -> None:
    z = stats.norm.ppf(1 - 0.05 / 8) + stats.norm.ppf(0.8)
    n = required_clusters(0.05, var_cluster=0.04, var_seed=0.0, var_residual=0.03, seeds=3, tests=4)
    assert n == pytest.approx((0.04 + 0.01) * (z / 0.05) ** 2)
    assert power_at(int(math.ceil(n)), 0.05, var_cluster=0.04, var_seed=0.0, var_residual=0.03, seeds=3, tests=4) == pytest.approx(0.8, abs=0.01)
    assert required_clusters(0.05, var_cluster=0.04, var_seed=0.01, var_residual=0.03, seeds=3, tests=4) is None   # seed floor
    assert power_at(700, 0.06, var_cluster=0.04, var_seed=0.0, var_residual=0.03, seeds=3) > power_at(300, 0.06, var_cluster=0.04,
                                                                                                        var_seed=0.0, var_residual=0.03, seeds=3)


# ---------------------------------------------------------------- e9_power on synthetic run folders

def _edit_folder(path: Path, *, effect: float, seed: int, concepts: int = 30, edits: int = 12, random_frame: float = 0.0) -> None:
    """A fake `RUN/edit`: new words with property / entailment / statement items (own source correct with probability
    0.3 + effect, random_frame 0.3 + random_frame) and edits whose after − control log-odds is `effect` per edit."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    rows, resolved = [], {}
    for c in range(concepts):
        cid = f"e9n-{c:04d}"
        resolved[cid] = {"status": "linked" if c != 0 else "unlinked"}
        for source, p in (("own", 0.3 + effect), ("random_frame", 0.3 + random_frame)):
            for k in range(3):
                rows.append({"part": "new", "source": source, "id": f"{cid}-property-r{k}", "concept": cid, "test": "property",
                             "relation": f"r{k}", "correct": float(rng.random() < p), "consistent": int(rng.random() < p),
                             "edge_kind": "resampled" if k else "category"})
                rows.append({"part": "new", "source": source, "id": f"{cid}-statement-r{k}", "concept": cid, "test": "statement",
                             "relation": f"r{k}", "correct": float(rng.random() < p), "margin": 0.0, "gold_loss": 1.0,
                             "edge_kind": "resampled"})
            rows.append({"part": "new", "source": source, "id": f"{cid}-entailment-0", "concept": cid, "test": "entailment",
                         "relation": "corrupted_r0", "correct": float(rng.random() < p + 0.2), "consistent": None,
                         "edge_kind": "category"})
    edit_resolved = {}
    for e in range(edits):
        cid = f"e9e-{e}"
        edit_resolved[cid] = {"status": "linked", "entry_status": "heldout" if e % 2 else "frequent"}
        for test in ("efficacy", "paraphrase"):
            base = rng.normal(0, 0.5)
            for condition, d in (("before", base), ("after", base + effect + rng.normal(0, 0.1)), ("control", base)):
                rows.append({"part": "edit", "condition": condition, "id": f"{cid}-{test}", "d": [d], "d_mean": d,
                             "new_preferred": float(d > 0)})
    (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "summary.json").write_text(json.dumps({"new_words": {"resolved": resolved}, "edits": {"resolved": edit_resolved}}))


def _group(tmp_path: Path) -> Group:
    models = {}
    for model, effect in (("C5", 0.25), ("C0'", 0.0)):
        for seed in (1, 2, 3):
            run = tmp_path / f"{model}-s{seed}"
            _edit_folder(run / "edit", effect=effect, seed=seed + (10 if model == "C5" else 20))
            models.setdefault(model, {})[seed] = SimpleNamespace(path=run)
    return Group("org/host", "full", models)


def test_power_analysis_on_synthetic_runs(tmp_path) -> None:
    group = _group(tmp_path)
    analysis = power.analyse_group(group, zeroshot_folder=None, resamples=200)
    assert analysis["available"] and analysis["seeds"] == [1, 2, 3]
    block = analysis["new_words"]["own − C0' own"]
    prop = block["tests"]["property"]
    assert prop["concepts"] == 29 and prop["items"] == 29 * 3 and prop["items_per_concept"] == 3      # the unlinked concept is out
    assert prop["model"]["mean"] == pytest.approx(0.25, abs=0.08) and prop["model"]["significant"]
    assert {"bootstrap", "item_sd", "item_level_se"} <= set(prop)
    within = analysis["new_words"]["own − random_frame"]["tests"]["property"]["model"]
    assert within["mean"] == pytest.approx(0.25, abs=0.08)
    edits = analysis["edits"]["metrics"]["log_odds"]
    assert set(edits) == {"seen", "heldout"} and edits["heldout"]["efficacy"]["concepts"] == 6
    assert edits["seen"]["efficacy"]["model"]["mean"] == pytest.approx(0.25, abs=0.08)
    plans = power.power_tables(analysis, deltas=[0.05, 0.06], planned_seeds=3, alpha=0.05, power=0.8, targets=[300, 700])
    p = plans["new_words"]["own − C0' own"]["property"]
    assert p["available"] and p["tests"] == 4 and set(p["required"]) == {"0.05", "0.06"}
    assert p["required"]["0.05"] is None or p["required"]["0.05"] >= p["required"]["0.06"]
    assert p["mde"]["700"] < p["mde"]["300"] and p["power_at"]["0.06"]["700"] >= p["power_at"]["0.06"]["300"]
    assert plans["edits"]["log_odds"]["seen"]["efficacy"]["tests"] == 4
    lines = power.render_item_seed(analysis)
    assert any("own − C0' own" in line and "property" in line for line in lines) and any("log_odds" in line for line in lines)
    config = {"runs": ["runs"], "folder": "edit", "zeroshot_folder": None, "deltas": [0.05, 0.06], "planned_seeds": 3,
              "alpha": 0.05, "power": 0.8, "targets": [300, 700]}
    text = power.render_power({"groups": {"host · full": {"analysis": analysis, "power": plans}}}, config)
    assert "n needed Δ=0.05" in text and "### Edits" in text and "MDE@350" in text
    assert power.edit_folder_name("int4", "v2") == "edit-v2-int4" and power.edit_folder_name() == "edit"


def test_paired_table_keeps_concepts_every_seed_has() -> None:
    concepts, seeds, table = power.paired_table({1: {"a": 1.0, "b": 0.0}, 2: {"a": 1.0}}, {1: {"a": 0.5, "b": 0.5}, 2: {"a": 0.0, "b": 0.0}})
    assert concepts == ["a"] and seeds == [1, 2] and table.tolist() == [[0.5, 1.0]]
    concepts, _, table = power.paired_table({1: {"a": 2.0}})
    assert table.tolist() == [[2.0]]
