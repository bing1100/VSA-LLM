"""E15 stage 1d (`experiments/e15-hrr-target/preregistration.md`, "Stage 1d"): the two 500M configs differ from their
stage-1b `-long` bases only in length and named checkpoints, the 500M schedule is as registered (no trainer change), and
`report --stage 1d` computes the pair of curves, the three primaries, the four decision readings and the extrapolated
crossing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("transformers")

from test_e15_stage1b import _write_qa, _write_windows  # noqa: E402
from vsa_embed.experiments import e15_eval  # noqa: E402
from vsa_embed.training import lm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
E15 = REPO / "experiments" / "e15-hrr-target"
CHECKPOINTS = [100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000]
TOKENS_PER_STEP = 1024 * 4 * 16


def test_the_two_xl_configs_differ_from_their_stage1b_long_base_only_by_length_and_checkpoints() -> None:
    folder = E15 / "configs" / "t5"
    names = sorted(p.name for p in folder.glob("*-xl-*.yaml"))
    assert names == ["SmolLM2-360M-full-C5-xl-s1.yaml", "SmolLM2-360M-full-C5WR-xl-s1.yaml"]
    assert e15_eval.STAGE1D_CHECKPOINTS == tuple(CHECKPOINTS)
    for name, arm, base_name in (("SmolLM2-360M-full-C5WR-xl-s1.yaml", "C5WR-xl", "SmolLM2-360M-full-C5WR-long-s1.yaml"),
                                 ("SmolLM2-360M-full-C5-xl-s1.yaml", "C5-xl", "SmolLM2-360M-full-C5-long-s1.yaml")):
        config = yaml.safe_load((folder / name).read_text())
        assert config["e15_base_config"] == base_name
        base = yaml.safe_load((folder / base_name).read_text())
        assert config["e15_stage"] == "1d" and base["e15_stage"] == "1b" and config["e15_arm"] == arm and config["seed"] == base["seed"] == 1
        assert config["experiment"] == f"e15-t5-{name[:-5]}" and config["e15_base_arm"] == base["e15_base_arm"] == "C5"
        for key in ("experiment", "e15_arm", "e15_stage", "e15_base_config"):
            config.pop(key); base.pop(key)
        assert config["train"].pop("save_at_tokens") == CHECKPOINTS
        assert base["train"].pop("save_at_tokens") == [50_000_000, 100_000_000, 150_000_000, 200_000_000]
        assert config["train"]["total_tokens"] == 500_000_000 and base["train"]["total_tokens"] == 200_000_000
        config["train"]["total_tokens"] = base["train"]["total_tokens"]
        assert config == base, name                 # everything else: the 1b recipe (λ_w 0.20, λ_r 0.035, layer, τ) or E9's C5, data, seeds
        resolved = lm.resolve_config(yaml.safe_load((folder / name).read_text()))
        assert resolved["train"]["save_at_tokens"] == CHECKPOINTS and resolved["train"]["warmup_tokens"] == 2_500_000 and resolved["train"]["min_lr_ratio"] == 0.1
    # C5WR-xl keeps the target block exactly; C5-xl has none (probe only).
    wr = yaml.safe_load((folder / "SmolLM2-360M-full-C5WR-xl-s1.yaml").read_text())["channel"]
    assert wr["target"] == {"layer": "third", "tau": 0.1, "detach_target": True, "write_weight": 0.2, "read_weight": 0.035}
    assert "target" not in yaml.safe_load((folder / "SmolLM2-360M-full-C5-xl-s1.yaml").read_text())["channel"]
    # The schedule at 500M (pre-registration §18.1): 7,629 steps, warm-up 38, the cosine over 7,591; the named checkpoints
    # at the first step reaching each count; the 500M count beyond the last step (written at the end as the final state).
    total = 500_000_000 // TOKENS_PER_STEP
    assert total == 7629 and total * TOKENS_PER_STEP == 499_974_144 < 500_000_000 and 2_500_000 // TOKENS_PER_STEP == 38 and total - 38 == 7591
    steps = {t: -(-t // TOKENS_PER_STEP) for t in CHECKPOINTS}
    assert [steps[t] for t in CHECKPOINTS] == [1526, 3052, 4578, 6104, 7630]
    assert [steps[t] * TOKENS_PER_STEP for t in CHECKPOINTS[:-1]] == [100_007_936, 200_015_872, 300_023_808, 400_031_744]
    assert steps[500_000_000] > total                                                  # never reached: the hard link to final.pt
    lr = lambda step: lm._lr(step, total, 38, 1.0, 0.1)  # noqa: E731
    assert 0.91 < lr(1526) < 0.93 and 0.68 < lr(3052) < 0.70 and 0.40 < lr(4578) < 0.42 and 0.18 < lr(6104) < 0.20
    assert lr(total - 1) == pytest.approx(0.1, abs=1e-6)                                # the floor at the end
    assert lm._lr(3051, 3051, 38, 1.0, 0.1) == pytest.approx(0.1, abs=1e-6)             # 1b's 200M was at the floor (S7)
    assert lm.eval_token_schedule(2_500_000, total * TOKENS_PER_STEP) == [2_500_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000, 80_000_000,
                                                                          160_000_000, 320_000_000, 499_974_144]


# -- the stage-1d report -----------------------------------------------------------------------------------------------

def _write_read(folder: Path, per_concept: dict[str, list[float]], key: str = "trained", chance: float = 1 / 3493) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    groups = {name: {"concepts": len(v), "linked_share": 1.0, key: {"accuracy": float(np.mean(v)), "concept_accuracy": float(np.mean(v)),
                                                                     "per_concept": v, "edges": 3 * len(v)}} for name, v in per_concept.items()}
    (folder / "read.json").write_text(json.dumps({"layer": 11, "atoms": 3493, "chance": chance, "trained_map": key == "trained", "groups": groups}))


def _noisy(level: float, n: int, rng: np.random.Generator, sd: float = 0.05) -> list[float]:
    return [float(min(1.0, max(0.0, level + rng.normal(0, sd)))) for _ in range(n)]


def test_report_stage_1d_computes_the_curves_the_primaries_and_the_four_decision_readings(tmp_path: Path) -> None:
    splits_path = E15 / "splits" / "t5-v1" / "splits.json"
    splits = e15_eval.load_splits(splits_path)
    concepts = [r["concept"] for r in splits["splits"]["compositional"]]
    strict = [r["concept"] for r in splits["splits"]["compositional_strict"]]
    unseen = [r["concept"] for r in splits["splits"]["unseen_filler"] if r["kind"] == "new"]
    rng = np.random.default_rng(0)
    host, root = "SmolLM2-360M", tmp_path
    runs, base, qa, rd = root / "runs", root / "base", root / "qa", root / "read"
    compose = ("none", "implicit:oracle", "explicit-rel:oracle")
    n = len(concepts)
    new_words = sorted(set(concepts) | set(unseen))                 # the QA items cover every new word; P1 keeps the compositional ones

    def write_curve(levels: dict[int, float], share_implicit: float, c5xl_implicit: float = 0.26) -> None:
        for t, level in levels.items():
            acc = {"compositional": _noisy(level, n, rng), "compositional_strict": [level] * len(strict), "unseen_filler": [level - 0.05] * len(unseen),
                   "seen": [0.95] * 300}
            _write_read(rd / f"{host}-full-C5WR-xl-s1" / f"ckpt-{t}", acc)
            _write_read(rd / f"{host}-full-C5-xl-s1" / f"ckpt-{t}", {"compositional": [0.3] * n, "unseen_filler": [0.4] * len(unseen), "seen": [0.95] * 300}, key="probe")
            _write_qa(qa / f"{host}-full-C5WR-xl-s1" / f"ckpt-{t}", new_words, {"implicit:oracle": share_implicit, "none": 0.2, "explicit-rel:oracle": 0.82}, compose, rng)
            _write_qa(qa / f"{host}-full-C5-xl-s1" / f"ckpt-{t}", new_words, {"implicit:oracle": c5xl_implicit, "none": 0.19, "explicit-rel:oracle": 0.82}, compose, rng)

    # (1) Reads at 0.82 and the objective's share is ≈ +0.15: seeds 2–3 and stage 2.
    write_curve({100_000_000: 0.60, 200_000_000: 0.72, 300_000_000: 0.77, 400_000_000: 0.80, 500_000_000: 0.82}, share_implicit=0.41)
    _write_qa(qa / f"{host}-full-C5-s1", new_words, {"implicit:oracle": 0.26, "none": 0.19, "explicit-rel:oracle": 0.84}, compose, rng)
    _write_read(rd / f"{host}-full-C5WR-long-s1" / "ckpt-200000000", {"compositional": [0.76] * n})      # stage 1b's 200M (consistency)
    _write_windows(runs / f"{host}-full-C5WR-xl-s1", 0.0); _write_windows(runs / f"{host}-full-C5-xl-s1", 0.05); _write_windows(base / f"{host}-full-C5-s1", 0.3)
    out_dir = root / "report"
    e15_eval.main(["report", "--stage", "1d", "--runs-root", str(runs), "--baseline-root", str(base), "--qa-root", str(qa), "--read-root", str(rd),
                   "--splits", str(splits_path), "--output", str(out_dir), "--host", host, "--resamples", "200"])
    out = json.loads((out_dir / "report.json").read_text())
    assert out["stage"] == "1d" and out["seeds"] == [1] and out["checkpoints"] == CHECKPOINTS and out["missing"] == []
    assert out["implicit_condition"] == {"C5": "implicit:oracle", "C5-xl": "implicit:oracle"}
    assert [r["name"][:2] for r in out["primary"]] == ["P1", "P2", "P3"] and all("p_holm" in r for r in out["primary"])
    p1, p2, p3 = out["primary"]
    assert p1["checkpoint"] == 500_000_000 and p1["n"] == 5 * n and 0.10 < p1["mean"] < 0.20
    assert 0.80 <= p2["mean_value"] < 0.85 and p2["n"] == n
    assert p3["from"] == 400_000_000 and p3["to"] == 500_000_000 and p3["still_rising"] and 0.01 < p3["mean"] < 0.03
    assert "seeds 2–3 and stage 2" in out["decision"] and out["extrapolated_crossing_tokens"] is None
    assert list(out["curve"]) == [str(t) for t in CHECKPOINTS]
    row = out["curve"]["500000000"]
    assert 0.10 < row["share"]["property"]["mean"] < 0.20 and 0.10 < row["vs_C5"]["property"]["mean"] < 0.20
    assert row["read"]["C5-xl"]["compositional/probe"]["mean_value"] == pytest.approx(0.3)
    assert row["means"]["C5-xl"]["property"]["implicit:oracle"] is not None and "compositional/trained" in row["read"]["C5WR-xl"]
    c = out["consistency_200M_vs_stage1b"]
    assert c["checkpoint"] == 200_000_000 and c["n"] == n and c["b_mean"] == pytest.approx(0.76) and c["a_mean"] == pytest.approx(0.72, abs=0.02)
    assert set(out["losses"]) == {"C5WR-xl − C5-xl (matched length)", "C5WR-xl − C5 (50M)", "C5-xl − C5 (length alone)"}
    assert out["losses"]["C5WR-xl − C5-xl (matched length)"]["after_heldout"]["relative"] < 0 < out["losses"]["C5-xl − C5 (length alone)"]["all"]["relative"] * -1
    names = [r["name"] for r in out["secondary"]]
    assert any("C5WR-xl − C5 (50M; continuity" in s for s in names) and any("C5-xl − C5 (length alone)" in s for s in names)
    assert any("strict compositional" in s for s in names) and any("unseen-filler new words" in s for s in names)
    assert any("trained W) − C5-xl (probe)" in s for s in names) and any("probe read accuracy, compositional, C5-xl at 500,000,000" in s for s in names)
    assert sum("implicit paraphrase" in s or "implicit entailment" in s for s in names) == 2
    text = (out_dir / "report.md").read_text()
    assert "## Read accuracy over the checkpoints" in text and "## Implicit-route QA over the checkpoints" in text
    assert "Decision rule reading" in text and "Consistency check" in text and "still rising" in text

    ns = lambda name: argparse.Namespace(runs_root=runs, baseline_root=base, qa_root=qa, read_root=rd, splits=splits_path, output=root / name,  # noqa: E731
                                         host=host, resamples=200, seed=0, checkpoints=None, seeds=None)
    # (2) Reads at 0.82 but the share is ≈ +0.04: small matched-length gain, no further length stages.
    write_curve({100_000_000: 0.60, 200_000_000: 0.72, 300_000_000: 0.77, 400_000_000: 0.80, 500_000_000: 0.82}, share_implicit=0.30)
    small = e15_eval.run_report_1d(ns("small"))
    assert "use gain is small" in small["decision"] and "no further length stages" in small["decision"] and small["primary"][0]["mean"] < 0.10
    # (3) Saturated below the bar: the 400 → 500M increment's CI includes 0.
    write_curve({100_000_000: 0.60, 200_000_000: 0.72, 300_000_000: 0.76, 400_000_000: 0.77, 500_000_000: 0.77}, share_implicit=0.41)
    at400 = json.loads((rd / f"{host}-full-C5WR-xl-s1" / "ckpt-400000000" / "read.json").read_text())["groups"]["compositional"]["trained"]["per_concept"]
    _write_read(rd / f"{host}-full-C5WR-xl-s1" / "ckpt-500000000", {"compositional": [float(min(1.0, v + rng.normal(0, 0.005))) for v in at400],
                                                                     "compositional_strict": [0.77] * len(strict), "unseen_filler": [0.72] * len(unseen), "seen": [0.95] * 300})
    flat = e15_eval.run_report_1d(ns("flat"))
    assert not flat["increment"]["still_rising"] and "saturated below the bar; closed" in flat["decision"]
    # (4) Still rising below the bar: the extrapolated crossing is reported and the reading says stop (no 1e).
    write_curve({100_000_000: 0.50, 200_000_000: 0.60, 300_000_000: 0.64, 400_000_000: 0.67, 500_000_000: 0.70}, share_implicit=0.41)
    rising = e15_eval.run_report_1d(ns("rising"))
    assert rising["increment"]["still_rising"] and "no stage 1e" in rising["decision"] and "extrapolated crossing" in rising["decision"]
    level, inc = rising["primary"][1]["mean_value"], rising["increment"]["mean"]
    assert rising["extrapolated_crossing_tokens"] == pytest.approx(500e6 + (0.8 - level) / inc * 100e6) and 700e6 < rising["extrapolated_crossing_tokens"] < 1_100e6
    assert "Extrapolated crossing" in (root / "rising" / "report.md").read_text()
    # The helper itself: at the bar or not rising → None.
    assert e15_eval.extrapolated_crossing(0.80, 0.01, step_tokens=100_000_000, last_tokens=500_000_000) is None
    assert e15_eval.extrapolated_crossing(0.70, 0.0, step_tokens=100_000_000, last_tokens=500_000_000) is None
    assert e15_eval.extrapolated_crossing(0.70, 0.05, step_tokens=100_000_000, last_tokens=500_000_000) == pytest.approx(700e6)
    # Missing folders are listed, not fatal: a run without its 500M evaluations reads as incomplete.
    (rd / f"{host}-full-C5WR-xl-s1" / "ckpt-500000000" / "read.json").unlink()
    partial = e15_eval.run_report_1d(ns("partial"))
    assert partial["decision"] == "incomplete" and any("ckpt-500000000" in m for m in partial["missing"])
    # `--checkpoints` / `--seeds` parsing: absent → the registered defaults; 1b's default is unchanged.
    assert e15_eval.parse_checkpoints(None, e15_eval.STAGE1D_CHECKPOINTS) == CHECKPOINTS and e15_eval.parse_checkpoints("", e15_eval.STAGE1B_CHECKPOINTS) == [50_000_000, 100_000_000, 150_000_000, 200_000_000]
    assert e15_eval.parse_checkpoints("200000000,100000000", ()) == [100_000_000, 200_000_000]
    assert e15_eval.parse_seeds(None) == (1,) and e15_eval.parse_seeds("1,2,3") == (1, 2, 3)
