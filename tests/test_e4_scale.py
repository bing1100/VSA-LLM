"""Decision 65 scaling report (`e4_scale`) on synthetic run folders whose data multipliers are planted."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from vsa_embed.convergence import common_targets, tokens_to_loss
from vsa_embed.experiments import e4_report, e4_scale
from vsa_embed.training.lm import MODEL_SIZES, build_model, resolve_config, save_window_losses

STRATA = ["all", "unlinked", "inside", "after", "after_heldout", "after_rare_seen", "after_len3plus"]
WINDOWS = 64
PER_STEP = 32_768


def _loss(tokens: np.ndarray, k: float) -> np.ndarray:
    return 2.0 + 2.0 * (k * tokens / 1e7) ** -0.4


def make_run(root: Path, size: str, condition: str, seed: int, k: dict[str, float], *, total: int = 500_000_000,
             first: int = 5_000_000, speed: float = 100_000.0, channel: int = 0, noise: float = 0.002) -> Path:
    """A finished run whose per-window losses follow `L(t) = 2 + 2·(k·t/1e7)^-0.4` per stratum (k from `k`, default 1),
    plus a window difficulty shared by every condition, size and evaluation and a little independent noise."""
    path = root / f"{size}-{condition}-s{seed}"
    path.mkdir(parents=True)
    final = (total // PER_STEP) * PER_STEP
    tokens = [first * 2**i for i in range(20) if first * 2**i < final] + [final]
    shared = np.random.default_rng(7)
    counts = {s: shared.poisson(20, WINDOWS).astype(np.int32) for s in STRATA}
    counts["after_rare_seen"] = (shared.poisson(2, WINDOWS) * (shared.random(WINDOWS) < 0.6)).astype(np.int32)
    difficulty = shared.normal(0, 0.3, (len(STRATA), WINDOWS))
    rng = np.random.default_rng(abs(hash((size, condition, seed))) % 2**32)
    rows, sink = [], {}
    for t in [0, *tokens]:
        sink_t = {}
        for i, stratum in enumerate(STRATA):
            level = 10.8 if t == 0 else float(_loss(np.asarray(t, dtype=float), k.get(stratum, 1.0)))
            per_token = level + difficulty[i] + rng.normal(0, noise, WINDOWS)
            sums = per_token * counts[stratum]
            sink_t[stratum] = ([sums], [counts[stratum]])
            total_count = counts[stratum].sum()
            rows.append({"type": "eval", "step": t // PER_STEP, "tokens": t, "stratum": stratum,
                         "loss": float(sums.sum() / total_count) if total_count else float("nan"), "stratum_tokens": int(total_count)})
        save_window_losses(path / "eval_windows.npz", t, list(range(0, WINDOWS * 1024, 1024)), sink_t)
    rows += [{"type": "train", "step": s, "tokens": s * PER_STEP, "loss": 4.0, "lr": 1e-3, "tokens_per_s": speed,
              "train_tokens_per_s": speed} for s in range(20, 200, 20)]
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = {"seed": seed, "experiment": f"e4-scale-v1-{size}-{condition}-s{seed}",
              "model": {"size": size, "seq_len": 1024, "vocab_size": 50257},
              "train": {"micro_batch": 32, "grad_accum": 1, "total_tokens": total},
              "data": {"eval": "/data/eval", "ontology": "/data/ontology.pt", "min_subtokens": 2},
              "eval": {"windows": WINDOWS}, "channel": {"mode": "none"}}
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    host = {"20M": 30_339_456, "50M": 51_475_968, "125M": 124_439_808}[size]
    (path / "manifest.json").write_text(json.dumps({"parameters": host + channel, "channel_parameters": channel}))
    return path


SIZES = ("20M", "50M", "125M")
# Planted multipliers: C5's k on `all` shrinks with size (1.30 → 1.20 → 1.10), on after_len3plus it grows (1.2 → 1.3 → 1.4).
C5_ALL = {"20M": 1.30, "50M": 1.20, "125M": 1.10}
C5_LONG = {"20M": 1.20, "50M": 1.30, "125M": 1.40}


def _k(condition: str, size: str) -> dict[str, float]:
    if condition == "C0":
        return {}
    if condition == "C2":
        return {"all": 1 + (C5_ALL[size] - 1) / 2, "after_len3plus": 1 + (C5_LONG[size] - 1) / 2}
    bonus = 1.05 if condition == "HRRAdd" else 1.0
    return {"all": C5_ALL[size] * bonus, "after_len3plus": C5_LONG[size] * bonus, "after": 1.15, "inside": 1.1}


@pytest.fixture(scope="module")
def scale_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("scale")
    for size in SIZES:
        for condition in ("C0", "C2", "C5", "HRRAdd", "HRRCat") + (("C5sh",) if size == "50M" else ()):
            k = {} if condition == "C5sh" else _k(condition, size)
            make_run(root / "runs", size, condition, 1, k, speed=100_000.0 if condition == "C0" else 98_000.0,
                     channel=0 if condition == "C0" else 2_200_000)
    for seed, shift in ((1, -0.02), (2, 0.0), (3, 0.02)):        # reference cohort: ln k of C5 varies by ±0.02 over seeds
        make_run(root / "reference", "50M", "C0", seed, {}, total=100_000_000, first=10_000_000)
        make_run(root / "reference", "50M", "C5", seed, {s: 1.2 * math.exp(shift) for s in STRATA}, total=100_000_000,
                 first=10_000_000)
        make_run(root / "reference", "50M", "C2", seed, {s: 1.1 for s in STRATA}, total=100_000_000, first=10_000_000)
    return root


@pytest.fixture(scope="module")
def analysis(scale_root: Path) -> tuple[dict, dict]:
    summary, e4, _ = e4_scale.analyze_scale(e4_report.discover([scale_root / "runs"]),
                                            reference_runs=e4_report.discover([scale_root / "reference"]),
                                            resamples=2000, k_resamples=300)
    return summary, e4


def test_vectorized_tokens_to_loss_matches_the_scalar_definition() -> None:
    rng = np.random.default_rng(0)
    tokens = np.array([1e7, 2e7, 4e7, 8e7, 1.6e8])
    losses = np.sort(rng.normal(4.0, 0.5, (200, 5)), axis=1)[:, ::-1] + rng.normal(0, 0.05, (200, 5))  # some non-monotone
    targets = rng.normal(4.0, 0.6, 200)
    got = e4_scale.tokens_to_loss_rows(tokens, losses, targets)
    for row, target, value in zip(losses, targets, got):
        expected = tokens_to_loss(list(zip(tokens, row)), float(target))
        assert (expected is None and np.isnan(value)) or value == pytest.approx(expected, rel=1e-12)
    curves = [list(zip(tokens, row)) for row in losses[:2]]
    target = e4_scale.loss_target(losses[None, :2, 0], losses[None, :2, -1], 0.1)[0]
    assert target == pytest.approx(common_targets([curves[0]], [curves[1]])[0])


def test_multiplier_recovers_a_planted_data_multiplier_and_the_slope_math() -> None:
    tokens = 5e6 * 2.0 ** np.arange(8)
    k = e4_scale.multiplier_rows(tokens, _loss(tokens, 1.0)[None], _loss(tokens, 1.25)[None], 0.1)[0]
    assert k == pytest.approx(1.25, rel=0.02)
    x = np.log([10.0, 25.0, 85.0])
    assert e4_scale.ols_slope(x, 0.3 - 0.05 * x)[0] == pytest.approx(-0.05)
    centred = x - x.mean()
    assert e4_scale.slope_weight(x) == pytest.approx(1 / math.sqrt((centred**2).sum()))
    draws = np.random.default_rng(1).normal(0.0, 0.01, 4000)
    low, high = e4_scale.inflated_interval(0.0, draws, 0.0)
    assert high == pytest.approx(1.96 * 0.01, rel=0.05) and low == pytest.approx(-high)
    assert e4_scale.inflated_interval(0.0, draws, 0.02)[1] == pytest.approx(1.96 * math.sqrt(0.01**2 + 0.02**2), rel=0.05)
    assert e4_scale.inflated_interval(0.0, draws, None) is None
    for size in ("tiny", "20M", "50M"):
        model = build_model(resolve_config({"model": {"size": size}, "device": "cpu"}))
        embedding = (50257 + 1024) * MODEL_SIZES[size]["n_embd"]
        assert e4_scale.non_embedding_parameters(size) == sum(p.numel() for p in model.parameters()) - embedding


def test_size_trend_recovers_planted_multipliers_and_slopes(analysis) -> None:
    summary, _ = analysis
    assert summary["sizes"] == list(SIZES) and summary["shared_windows"] and summary["single_seed"]
    trend = summary["trend"]["C5|C0"]
    x = np.log([e4_scale.non_embedding_parameters(s) for s in SIZES])
    for stratum, planted in (("all", C5_ALL), ("after_len3plus", C5_LONG)):
        entry = trend[stratum]
        for size in SIZES:
            k = entry["sizes"][size]["k"]
            assert k["k"] == pytest.approx(planted[size], rel=0.03) and k["ci_low"] < k["k"] < k["ci_high"]
            assert k["seed_ci"]["ci_high"] - k["seed_ci"]["ci_low"] > k["ci_high"] - k["ci_low"]   # seed variance added
        expected = e4_scale.ols_slope(x, np.log([planted[s] for s in SIZES]))[0]
        slope = entry["trend"]["k"]
        assert slope["slope"] == pytest.approx(expected, abs=0.01) and slope["seed_ci"]["ci_low"] < slope["slope"] < slope["seed_ci"]["ci_high"]
    assert trend["all"]["trend"]["k"]["seed_ci"]["ci_high"] < 0 < trend["after_len3plus"]["trend"]["k"]["seed_ci"]["ci_low"]
    unlinked = trend["unlinked"]["sizes"]["50M"]["k"]
    assert unlinked["ci_low"] < 1.0 < unlinked["ci_high"]                          # nothing planted off the spans
    assert trend["all"]["sizes"]["125M"]["paired"]["significant"] and trend["all"]["sizes"]["125M"]["paired"]["relative"] < 0


def test_seed_reference_measures_the_planted_seed_spread(analysis) -> None:
    reference = analysis[0]["seed_reference"]
    c5 = reference["strata"]["all"]["C5|C0"]
    assert c5["seeds"] == [1, 2, 3] and c5["k_log_sd"] == pytest.approx(0.02, rel=0.15)
    assert reference["strata"]["all"]["C2|C0"]["k_log_sd"] < 0.005                 # no planted seed spread for C2
    assert analysis[0]["trend"]["HRRAdd|C0"]["all"]["sizes"]["20M"]["k"]["seed_sd"] == c5["k_log_sd"]   # composed → C5's


def test_compute_adjustment_share_hybrids_and_specificity(analysis) -> None:
    summary, _ = analysis
    row = summary["trend"]["C5|C0"]["all"]["sizes"]["50M"]
    assert row["compute"]["wall_clock"]["factor"] == pytest.approx(0.98)
    assert row["compute"]["flops"]["factor"] == pytest.approx(51_475_968 / (51_475_968 + 2_200_000))
    assert row["compute"]["wall_clock"]["k"] == pytest.approx(row["k"]["k"] * 0.98)
    share = summary["c2_share"]["all"]["50M"]
    assert 0.2 < share["share"] < 0.8 and share["ci_low"] < share["share"] < share["ci_high"]
    hybrid = summary["trend"]["HRRAdd|C5"]["all"]["sizes"]["125M"]
    assert hybrid["k"]["k"] == pytest.approx(1.05, rel=0.03) and hybrid["paired"]["relative"] < 0
    specificity = summary["trend"]["C5|C5sh"]["all"]["sizes"]
    assert list(specificity) == ["50M"] and specificity["50M"]["k"]["k"] == pytest.approx(1.2, rel=0.03)


def test_escalation_and_phase2_rules(analysis) -> None:
    summary, _ = analysis
    rules = summary["escalation"]["C5|C0"]
    assert rules["all"]["20M"]["smaller_model_multiplier"] is None
    assert not rules["all"]["125M"]["size_trend_ok"] and rules["after_len3plus"]["125M"]["size_trend_ok"]
    assert rules["after_len3plus"]["125M"]["rule_iii"] and rules["after_len3plus"]["125M"]["rule_ii"]   # C2 beaten
    assert not rules["all"]["125M"]["projection_ok"]                              # one seed: no projection interval
    phase2 = summary["phase2"]
    assert not phase2["strata"]["all"]["pass"] and phase2["strata"]["all"]["shrinkage_not_established"] is False
    assert phase2["strata"]["after_len3plus"]["pass"] and phase2["request_phase2"]
    assert phase2["hybrids"]["HRRAdd"]["forward"] and not phase2["hybrids"]["HRRCat"]["forward"]


def test_cli_writes_both_reports_and_refuses_to_overwrite(scale_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    args = ["--runs", str(scale_root / "runs"), "--output", str(out), "--seed-reference", str(scale_root / "reference"),
            "--resamples", "2000", "--k-resamples", "200", "--title", "SMOKE scale test", "--no-figures"]
    e4_scale.main(args)
    for name in ("report.md", "summary.json", "e4-report.md", "e4-summary.json", "manifest.json", "resolved_config.yaml"):
        assert (out / name).is_file()
    report = (out / "report.md").read_text()
    assert report.startswith("# SMOKE scale test") and "Request phase 2: True" in report and "⟨" in report
    assert "## Seed-variance reference" in report and "| after_len3plus | C5 | " in report and "| all | C5 vs C0 | [1, 2, 3] |" in report
    assert "| unlinked | n.s. | n.s. | n.s. |" in report                        # no C2 share without a C5 gap
    colours = {e4_report.condition_style(c)["color"] for c in ("C0", "C2", "C5", "HRRAdd", "HRRCat")}
    assert len(colours) == 5 and e4_report.condition_style("C5sh")["color"] == e4_report.condition_style("C5")["color"]
    assert json.loads((out / "manifest.json").read_text())["request_phase2"] is True
    with pytest.raises(FileExistsError):
        e4_scale.main(args)
    e4_scale.main([*args, "--overwrite"])
