"""E4 report on synthetic run folders whose effects (and so verdicts and CI signs) are known."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from vsa_embed.experiments.e4_report import analyze, discover, main, probe_difference
from vsa_embed.training.lm import eval_token_schedule, save_window_losses

STRATA = ["all", "inside", "after", "after_heldout", "after_rare", "after_mid", "after_frequent",
          "after_len1", "after_len2", "after_len3plus", "unlinked"]
WINDOWS = 120
PER_STEP = 1024 * 32 * 8
# Per-token loss shift (negative = better) and the curve speed-up on linked strata; channel parameters.
EFFECTS = {
    "C0": dict(linked=0.0, heldout=0.0, rare=0.0, unlinked=0.0, speed=1.0, channel=0),
    "C1": dict(linked=-0.02, heldout=-0.02, rare=-0.02, unlinked=0.0, speed=1.05, channel=300_000),
    "C2": dict(linked=-0.04, heldout=-0.01, rare=-0.04, unlinked=0.0, speed=1.2, channel=600_000),
    "C3": dict(linked=-0.02, heldout=-0.015, rare=-0.02, unlinked=0.06, speed=1.05, channel=550_000),
    "C5": dict(linked=-0.10, heldout=-0.10, rare=-0.06, unlinked=0.0005, speed=1.8, channel=580_000),
}


def _counts() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    counts = {s: rng.poisson(25, WINDOWS) for s in STRATA}
    counts["after_heldout"] = rng.poisson(12, WINDOWS) * (rng.random(WINDOWS) < 0.4)
    counts["after_len1"] = np.zeros(WINDOWS, dtype=int)
    counts["unlinked"] = rng.poisson(700, WINDOWS); counts["all"] = counts["unlinked"] + 300
    return {s: c.astype(np.int32) for s, c in counts.items()}


def _shift(condition: str, stratum: str) -> float:
    e = EFFECTS[condition]
    return {"unlinked": e["unlinked"], "all": 0.2 * e["linked"] + e["unlinked"], "after_heldout": e["heldout"],
            "after_rare": e["rare"]}.get(stratum, e["linked"])


def make_run(root: Path, condition: str, seed: int, *, size: str = "50M", total: int = 100_000_000, windows: bool = True,
             label: str | None = None, train_extra: dict | None = None, probes: dict | None = None, complete: bool = True,
             parameters: float = 5e7, pretrained: str | None = None, host_mode: str = "lora") -> Path:
    label, effect = label or condition, condition.rstrip("'")         # C0' (continued pretraining) acts as C0
    path = root / f"{size}-{label}-s{seed}"
    path.mkdir(parents=True)
    final_tokens = (total // PER_STEP) * PER_STEP
    counts = _counts()
    difficulty = np.random.default_rng(7).normal(size=(len(STRATA), WINDOWS))      # shared text: pairs windows
    rng = np.random.default_rng(1000 * seed + sorted(EFFECTS).index(effect))
    offset = np.random.default_rng(seed).normal(0, 0.002)                          # seed-level, shared by conditions
    sums, final = {}, {}
    for i, stratum in enumerate(STRATA):
        per_token = 3.0 + 0.3 * difficulty[i] + _shift(effect, stratum) + offset + rng.normal(0, 0.01, WINDOWS)
        sums[stratum] = per_token * counts[stratum]
        total_count = counts[stratum].sum()
        final[stratum] = float(sums[stratum].sum() / total_count) if total_count else float("nan")
    rows = []
    for tokens in [0, *eval_token_schedule(10_000_000, final_tokens)]:
        for stratum in STRATA:
            speed = 1.0 if stratum == "unlinked" else EFFECTS[effect]["speed"]
            loss = 10.8 if tokens == 0 else final[stratum] + ((speed * tokens / 1e7) ** -0.4 - (speed * final_tokens / 1e7) ** -0.4)
            if math.isnan(final[stratum]):
                loss = float("nan")
            rows.append({"type": "eval", "step": tokens // PER_STEP, "tokens": tokens, "stratum": stratum, "loss": loss,
                         "stratum_tokens": int(counts[stratum].sum())})
    rows += [{"type": "train", "step": s, "tokens": s * PER_STEP, "loss": 4.0, "lr": 1e-3,
              "tokens_per_s": 70_000 / (1 + 0.1 * (effect != "C0"))} for s in range(20, 200, 20)]
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    model = {"size": size, "seq_len": 1024, "vocab_size": 50257, **({"pretrained": pretrained, "host_mode": host_mode} if pretrained else {})}
    config = {"seed": seed, "experiment": f"e4-test-{size}-{label}-s{seed}", "model": model,
              "train": {"micro_batch": 32, "grad_accum": 8, "total_tokens": total, **(train_extra or {})},
              "data": {"eval": "/data/eval", "ontology": "/data/ontology.pt", "min_subtokens": 2},
              "eval": {"windows": WINDOWS}, "channel": {"mode": "none"}}
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    if windows:
        save_window_losses(path / "eval_windows.npz", final_tokens, list(range(0, WINDOWS * 1024, 1024)),
                           {s: ([sums[s]], [counts[s]]) for s in STRATA})
    if complete:
        channel = EFFECTS[effect]["channel"]
        (path / "manifest.json").write_text(json.dumps({"parameters": parameters + channel, "channel_parameters": channel}))
    if probes is not None:
        (path / "probes.json").write_text(json.dumps({"probes": probes}))
    return path


def _probes(condition: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    gold = np.random.default_rng(11).normal(size=300)
    good = condition == "C5"
    wic = (rng.random(400) < (0.75 if good else 0.55)).astype(float)
    card = np.stack([gold + rng.normal(0, 0.7 if good else 2.0, 300), gold], 1)
    wsd = (np.random.default_rng(99 + seed).random(400) < 0.6).astype(float)       # identical for both → no change
    return {"wic_linear": {"score": float(wic.mean()), "per_example": wic.tolist()},
            "card660": {"score": 0.0, "statistic": "spearman", "per_example": card.tolist()},
            "wsd_all": {"score": float(wsd.mean()), "per_example": wsd.tolist()}}


@pytest.fixture(scope="module")
def runs_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("e4runs")
    for condition in ("C0", "C1", "C2", "C3", "C5"):
        for seed in (1, 2, 3):
            make_run(root / "pilot", condition, seed, probes=_probes(condition, seed) if condition in ("C0", "C5") else None)
            make_run(root / "big", condition, seed, size="125M", total=500_000_000, windows=False, parameters=1.25e8)
    make_run(root / "pilot", "C5", 1, label="C5@resume", train_extra={"stop_after_steps": 100})
    make_run(root / "pilot", "C1", 4, complete=False)
    for condition in ("C0", "C2", "C5"):
        make_run(root / "short", condition, 1, total=50_000_000)
    for condition in ("C0'", "C2", "C5"):
        for seed in (1, 2):
            make_run(root / "cpt", condition, seed, total=25_000_000, parameters=1.35e8, pretrained="HuggingFaceTB/SmolLM2-135M")
    return root


@pytest.fixture(scope="module")
def summary(runs_root: Path) -> dict:
    return analyze(discover([runs_root]), candidates=["C5", "C3"], resamples=1000)[0]


def _cohort(summary: dict, prefix: str) -> dict:
    (label,) = [k for k in summary["cohorts"] if k.startswith(prefix)]
    return summary["cohorts"][label]


def test_runs_are_grouped_into_cohorts_with_resume_checks_and_incomplete_runs(summary: dict) -> None:
    assert sorted(summary["cohorts"]) == ["125M · 500M tokens", "50M · 100M tokens", "50M · 50M tokens",
                                          "HuggingFaceTB/SmolLM2-135M/lora · 25M tokens"]
    pilot = _cohort(summary, "50M · 100M")
    assert pilot["conditions"] == ["C0", "C1", "C2", "C3", "C5"] and pilot["seeds"]["C1"] == [1, 2, 3]
    assert [r["complete"] for r in summary["runs"] if r["seed"] == 4] == [False]
    (check,) = pilot["resume_checks"]
    assert check["verdict"] == "identical" and check["twin"].endswith("50M-C5-s1")
    assert "after_len1" not in pilot["paired"]["C0"]["C5"]        # no targets: no comparison


def test_paired_differences_have_the_planted_signs(summary: dict) -> None:
    pilot = _cohort(summary, "50M · 100M")
    held = pilot["paired"]["C0"]["C5"]["after_heldout"]
    assert held["method"] == "window bootstrap" and held["ci_high"] < 0 and held["significant"]
    assert held["delta"] == pytest.approx(-0.10, abs=0.01) and held["relative"] == pytest.approx(-0.10 / 3.0, abs=0.01)
    assert held["holm_p"] >= held["p_value"] and held["holm_family"] > 1
    null = pilot["paired"]["C0"]["C1"]["unlinked"]
    assert null["ci_low"] < 0 < null["ci_high"] and not null["significant"]
    worse = pilot["paired"]["C0"]["C3"]["unlinked"]
    assert worse["relative"] == pytest.approx(0.02, abs=0.002) and worse["significant"]
    final = pilot["final_loss"]["after_heldout"]["conditions"]
    assert final["C5"]["mean"] < final["C2"]["mean"] < final["C0"]["mean"] and final["C5"]["n"] == 3


def test_gate_items_pass_for_the_planted_winner_and_fail_for_the_boundary_detector(summary: dict) -> None:
    pilot = _cohort(summary, "50M · 100M")
    gate = pilot["gate"]["C5"]
    assert [gate["items"][i]["verdict"] for i in "1234"] == ["pass"] * 4 and gate["overall"] == "pass"
    assert gate["items"]["4"]["families_improved"] == ["rare_word", "wic"]
    assert pilot["probes"]["C5"]["wsd_all"]["delta"] == 0.0 and not pilot["probes"]["C5"]["wsd_all"]["significant"]
    c3 = pilot["gate"]["C3"]["items"]
    assert c3["2"]["verdict"] == "fail" and c3["3"]["verdict"] == "fail" and c3["4"]["verdict"] == "not available"
    assert pilot["gate"]["C3"]["overall"] == "fail"
    assert any("125M × 500M" in flag for flag in pilot["exploratory"])


def test_convergence_multipliers_projections_and_escalation(summary: dict) -> None:
    pilot = _cohort(summary, "50M · 100M")
    comparison = pilot["convergence"]["after_heldout"]["C5"]
    lowest = comparison["data_multiplier"][min(comparison["data_multiplier"], key=float)]
    assert lowest["mean"] > 1.8 and lowest["ci_low"] > 1.1
    assert comparison["projected_loss_gaps"]["2.5e+09"]["mean"] > 0 and comparison["k_curve"]
    assert pilot["fits"]["after_heldout"]["C5"]["beta"] == pytest.approx(0.4, abs=0.05)
    assert pilot["escalation"]["C5"]["after_heldout"]["escalate"]
    assert not pilot["escalation"]["C3"]["after_heldout"]["rule_ii"]
    big = _cohort(summary, "125M")
    assert big["escalation"]["C5"]["after_heldout"]["smaller_model_multiplier"] is not None
    assert pilot["throughput"]["C5"]["overhead"] == pytest.approx(0.1)
    assert pilot["throughput"]["C5"]["channel_bytes_fp32"] == 4 * 580_000


def test_single_seed_and_seed_level_fallbacks(summary: dict) -> None:
    short = _cohort(summary, "50M · 50M")
    assert short["single_seed"]
    held = short["paired"]["C0"]["C5"]["after_heldout"]
    assert held["method"] == "window bootstrap" and held["ci_high"] < 0 and held["seed_ci"]["ci_low"] is None
    assert short["final_loss"]["all"]["conditions"]["C0"]["ci_low"] is None
    big = _cohort(summary, "125M")
    assert big["exploratory"] == [] and not big["single_seed"]
    held = big["paired"]["C2"]["C5"]["after_heldout"]
    assert held["method"] == "seed t" and held["ci_high"] < 0
    gate = big["gate"]["C5"]
    assert [gate["items"][i]["verdict"] for i in "123"] == ["pass"] * 3 and gate["overall"] == "incomplete"


def test_continued_pretraining_cohort_uses_its_own_baseline_and_size_chain(summary: dict) -> None:
    cpt = _cohort(summary, "HuggingFaceTB")
    assert cpt["baseline"] == "C0'" and cpt["references"] == ["C0'", "C2"]
    assert cpt["paired"]["C0'"]["C5"]["after_heldout"]["ci_high"] < 0
    assert cpt["gate"]["C5"]["items"]["3"]["verdict"] == "pass" and cpt["gate"]["C5"]["items"]["2"]["verdict"] == "not available"
    assert cpt["escalation"]["C5"]["after_heldout"]["smaller_model_multiplier"] is None   # not chained to from-scratch 50M


def test_an_evaluation_only_baseline_gets_paired_gaps_but_no_multiplier(tmp_path: Path) -> None:
    host = "HuggingFaceTB/SmolLM2-360M"
    for condition in ("C0'", "C5"):
        for seed in (1, 2):
            make_run(tmp_path, condition, seed, total=100_000_000, pretrained=host, host_mode="frozen",
                     train_extra={"eval_only": True} if condition == "C0'" else None)
    (cohort,) = analyze(discover([tmp_path]), resamples=1000)[0]["cohorts"].values()
    assert cohort["baseline"] == "C0'" and any("evaluation-only" in w for w in cohort["warnings"])
    assert cohort["paired"]["C0'"]["C5"]["after_heldout"]["ci_high"] < 0
    assert cohort["convergence"] == {} and cohort["escalation"] == {}
    assert "C0'" not in cohort["fits"].get("all", {}) and "C5" in cohort["fits"]["all"]


def test_too_few_resamples_for_holm_are_flagged(runs_root: Path) -> None:
    cohort = analyze(discover([runs_root / "short"]), resamples=100)[0]["cohorts"]["50M · 50M tokens"]
    assert any("cannot reach Holm" in w for w in cohort["warnings"])


def test_probe_difference_without_examples_uses_seed_intervals() -> None:
    cand = {s: {"wic": {"score": 0.70 + 0.01 * s, "family": "wic"}} for s in (1, 2, 3)}
    base = {s: {"wic": {"score": 0.60 + 0.012 * s, "family": "wic"}} for s in (1, 2, 3)}
    result = probe_difference(cand, base, "wic", resamples=100, seed=0)
    assert result["method"] == "seed t" and result["ci_low"] > 0
    lower = {s: {"perplexity": {"score": 10.0, "family": "domain", "higher_is_better": False}} for s in (1,)}
    higher = {s: {"perplexity": {"score": 12.0, "family": "domain", "higher_is_better": False}} for s in (1,)}
    assert probe_difference(lower, higher, "perplexity", resamples=100, seed=0)["delta"] == pytest.approx(2.0)


def test_cli_writes_the_run_folder_contract(runs_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    main(["--runs", str(runs_root / "pilot"), str(runs_root / "short"), "--output", str(out), "--resamples", "2000",
          "--references", "C3", "--title", "S0 pilot test"])
    for name in ("report.md", "summary.json", "manifest.json", "resolved_config.yaml"):
        assert (out / name).is_file()
    report = (out / "report.md").read_text()
    assert report.startswith("# S0 pilot test") and "Relative difference vs C3" in report and "single seed" in report
    assert "Kill-and-resume check" in report and "| C5 | pass | pass | pass | pass | **pass** |" in report
    assert sorted(p.name for p in (out / "figures").iterdir()) == [
        "k-50m-100m-tokens.png", "k-50m-50m-tokens.png", "loss-50m-100m-tokens.png", "loss-50m-50m-tokens.png"]
    with pytest.raises(FileExistsError):
        main(["--runs", str(runs_root / "short"), "--output", str(out)])


def test_cpt_folder_name_c0p_is_the_c0_prime_baseline(tmp_path) -> None:
    from vsa_embed.experiments import e4_report
    assert e4_report.NAME.search("SmolLM2-135M-lora-C0p-s1")["condition"] == "C0p"
    source = Path(e4_report.__file__).read_text()
    assert "C0p" in source and "\"C0'\" if condition == \"C0p\"" in source
