"""Convergence-speed evidence and the escalation rule (experiments.md §0.13, task B14).

Inputs are evaluation curves `[(tokens, loss), ...]` per run and stratum (from the trainer's
`metrics.jsonl`). Outputs:

- tokens-to-threshold `D(L)` (log-linear interpolation between checkpoints);
- the data multiplier `k(L) = D_baseline(L) / D_condition(L)` per seed, with a t-interval;
- `L(D) = E + B·D^(−β)` fits and the projected loss gap at a larger budget;
- the pre-registered escalation verdict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .statistics import mean_confidence_interval

Curve = list[tuple[float, float]]


def load_evaluations(run_dir: Path) -> dict[str, dict[float, dict[str, float]]]:
    """Stratum → training tokens → `{"loss", "stratum_tokens"}` from the trainer's `metrics.jsonl`.

    An evaluation repeated after a resume keeps its last value. Rows without `stratum_tokens`
    (trainer before the fix) logged the stratum's target count under `tokens`; their training
    tokens are recovered as `step × tokens_per_step` from `resolved_config.yaml`."""
    per_step = None
    evaluations: dict[str, dict[float, dict[str, float]]] = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if row.get("type") != "eval":
            continue
        tokens, count = row["tokens"], row.get("stratum_tokens")
        if "stratum_tokens" not in row:
            if per_step is None:
                config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
                per_step = config["model"]["seq_len"] * config["train"]["micro_batch"] * config["train"]["grad_accum"]
            tokens, count = row["step"] * per_step, row["tokens"]
        evaluations.setdefault(row["stratum"], {})[float(tokens)] = {"loss": float(row["loss"]), "stratum_tokens": count}
    return evaluations


def load_curves(run_dir: Path) -> dict[str, Curve]:
    """Stratum → sorted (tokens, loss) points, excluding the step-0 evaluation (see `load_evaluations`)."""
    curves = {stratum: sorted((t, v["loss"]) for t, v in points.items() if t > 0 and math.isfinite(v["loss"]))
              for stratum, points in load_evaluations(run_dir).items()}
    return {k: v for k, v in curves.items() if v}


def tokens_to_loss(curve: Curve, target: float) -> float | None:
    """First token count at which the curve reaches `target`, interpolating linearly in log-tokens."""
    previous = None
    for tokens, loss in curve:
        if loss <= target:
            if previous is None:
                return tokens
            t0, l0 = previous
            if l0 == loss:
                return tokens
            fraction = (l0 - target) / (l0 - loss)
            return float(math.exp(math.log(t0) + fraction * (math.log(tokens) - math.log(t0))))
        previous = (tokens, loss)
    return None


def data_multiplier(baseline: Curve, condition: Curve, target: float) -> float | None:
    """`k = D_baseline(L) / D_condition(L)`; > 1 means the condition needs fewer tokens."""
    base, cond = tokens_to_loss(baseline, target), tokens_to_loss(condition, target)
    if base is None or cond is None:
        return None
    return base / cond


def common_targets(baselines: Sequence[Curve], conditions: Sequence[Curve], count: int = 3) -> list[float]:
    """Loss levels every curve reaches: between the worst final loss and the best first point."""
    finals = [c[-1][1] for c in (*baselines, *conditions)]
    firsts = [c[0][1] for c in (*baselines, *conditions)]
    high, low = min(firsts), max(finals)
    if not high > low:
        return []
    return [float(low + (high - low) * f) for f in np.linspace(0.1, 0.6, count)]


def fit_power_law(curve: Curve) -> dict[str, float] | None:
    """Least-squares `L(D) = E + B·D^(−β)` over the curve (needs ≥ 4 points)."""
    if len(curve) < 4:
        return None
    from scipy.optimize import curve_fit
    tokens = np.asarray([t for t, _ in curve], dtype=np.float64)
    losses = np.asarray([l for _, l in curve], dtype=np.float64)
    scale = tokens[0]
    x = tokens / scale
    def model(d, e, b, beta):
        return e + b * np.power(d, -beta)
    guess = (max(0.0, losses.min() * 0.9), max(1e-6, losses[0] - losses.min() * 0.9), 0.3)
    try:
        params, _ = curve_fit(model, x, losses, p0=guess, bounds=([0.0, 0.0, 1e-3], [losses.min(), np.inf, 3.0]),
                              maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    e, b, beta = (float(v) for v in params)
    residual = float(np.sqrt(np.mean((model(x, *params) - losses) ** 2)))
    return {"E": e, "B": b, "beta": beta, "scale_tokens": float(scale), "rmse": residual}


def project(fit: dict[str, float], tokens: float) -> float:
    return fit["E"] + fit["B"] * (tokens / fit["scale_tokens"]) ** (-fit["beta"])


def compare(baseline_runs: Sequence[Curve], condition_runs: Sequence[Curve], *, projection_tokens: float,
            targets: Sequence[float] | None = None) -> dict[str, Any]:
    """Paired (by seed order) comparison of a condition against a baseline on one stratum."""
    if len(baseline_runs) != len(condition_runs) or not baseline_runs:
        raise ValueError("baseline and condition need the same number of seeds (paired)")
    targets = list(targets) if targets is not None else common_targets(baseline_runs, condition_runs)
    multipliers: dict[str, Any] = {}
    for target in targets:
        values = [data_multiplier(b, c, target) for b, c in zip(baseline_runs, condition_runs)]
        values = [v for v in values if v is not None]
        multipliers[f"{target:.4f}"] = mean_confidence_interval(values) if values else None
    final_gaps = [b[-1][1] - c[-1][1] for b, c in zip(baseline_runs, condition_runs)]
    projected = []
    for b, c in zip(baseline_runs, condition_runs):
        fb, fc = fit_power_law(b), fit_power_law(c)
        if fb and fc:
            projected.append(project(fb, projection_tokens) - project(fc, projection_tokens))
    return {
        "targets": targets, "data_multiplier": multipliers,
        "final_loss_gap": mean_confidence_interval(final_gaps),  # > 0: condition has lower loss
        "projected_loss_gap": mean_confidence_interval(projected) if projected else None,
        "projection_tokens": projection_tokens,
    }


def _lowest_multiplier_ci(comparison: dict[str, Any]) -> tuple[float | None, float | None]:
    """The multiplier at the hardest (lowest) common loss target, which is the pre-registered one."""
    items = [(float(k), v) for k, v in comparison["data_multiplier"].items() if v is not None]
    if not items:
        return None, None
    target, ci = min(items)
    return ci["mean"], ci["ci_low"]


def escalation_verdict(
    versus_baseline: dict[str, Any], versus_controls: dict[str, dict[str, Any]], *,
    smaller_model_multiplier: float | None = None, min_multiplier: float = 1.1,
) -> dict[str, Any]:
    """Pre-registered rule: (i) k's lower CI bound > `min_multiplier` at the largest committed
    budget; (ii) the condition's final loss beats each control (C1, C2, C1h) with a CI excluding
    zero; (iii) the projected gap's CI excludes zero, or k does not shrink from the smaller model."""
    k_mean, k_low = _lowest_multiplier_ci(versus_baseline)
    rule_i = k_low is not None and k_low > min_multiplier
    control_results = {}
    for name, comparison in versus_controls.items():
        gap = comparison["final_loss_gap"]
        control_results[name] = gap["ci_low"] is not None and gap["ci_low"] > 0
    rule_ii = bool(control_results) and all(control_results.values())
    projected = versus_baseline.get("projected_loss_gap")
    projection_ok = projected is not None and projected["ci_low"] is not None and projected["ci_low"] > 0
    trend_ok = smaller_model_multiplier is not None and k_mean is not None and k_mean >= smaller_model_multiplier
    rule_iii = projection_ok or trend_ok
    return {"multiplier_mean": k_mean, "multiplier_ci_low": k_low, "rule_i": rule_i,
            "controls_beaten": control_results, "rule_ii": rule_ii,
            "projection_ok": projection_ok, "size_trend_ok": trend_ok, "rule_iii": rule_iii,
            "escalate": rule_i and rule_ii and rule_iii}
