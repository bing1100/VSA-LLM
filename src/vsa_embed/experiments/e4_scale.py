"""Decision 65, from-scratch scaling screen (`scale-v1`): the E4 analysis per size plus the size trend of the multiplier.

    python -m vsa_embed.experiments.e4_scale --runs experiments/e4-small-lm/runs/scale-v1 \\
        --output experiments/e4-small-lm/analysis/scale-v1 --seed-reference experiments/e4-small-lm/runs/opscreen \\
        [--resamples 10000] [--k-resamples 2000] [--projection-tokens 2.5e9 7e9] [--overwrite]

Pre-registration: `experiments/e4-small-lm/preregistration-scale-v1.md`. Each model size is an `e4_report` cohort, and the
full E4 analysis runs on it unchanged (`e4_report.analyze`, baseline C0, references C2, C5sh and C5: final losses, paired
window-bootstrap differences with Holm, `k(L)` at common loss targets, power-law fits, projected gaps, the escalation
rule chained over sizes), written to `e4-report.md` / `e4-summary.json` with its figures. On top, in `report.md` /
`summary.json`:

- **k with single-seed CIs.** `k = D_ref(L) / D_cond(L)` at the lowest common loss target (`common_targets`' first, the
  one the escalation rule reads; `loss_target` fraction 0.1) and `k_end` at the worse final loss of the pair (fraction
  0), with a percentile cluster bootstrap over evaluation windows: each resample reweights the windows, rebuilds both
  curves from the per-window loss sums of every evaluation (`eval_windows.npz`) and recomputes the target and k
  (`multiplier_rows`, a vectorized `convergence.tokens_to_loss`). Runs of a condition with several seeds are pooled per
  window (sums over the seeds both conditions share).
- **Size trend** per stratum (`TREND_STRATA`) and pair (`PAIRS`): k, `k_end` and the final relative gap (from the E4
  analysis, Holm-starred) at every size, and the OLS slope of log k on log N (N = non-embedding parameters,
  `non_embedding_parameters`), bootstrapped with the same window weights at every size (the evaluation windows are
  shared; sizes with other windows get their own weights). With `--seed-reference` (the opscreen's 3 seeds at
  50M × 100M) every k and slope also gets a CI that adds the reference's seed variance of log k in quadrature
  (`seed_reference`, `inflated_interval`; assumed size- and budget-independent; C2 → C2 vs C0, the composed
  conditions → C5 vs C0, comparisons between channel conditions → the reference's C5 variants vs C5, pooled).
- **Compute-adjusted k** (vs C0): × the measured tokens/s ratio (wall clock) and × the 6N ratio of total parameters
  (FLOPs with the channel counted as dense compute, an upper bound on its cost: its tables are read at linked spans only).
- **C2's share** of C5's final gap `(L_C0 − L_C2) / (L_C0 − L_C5)` (window bootstrap); HRRAdd − C5, HRRCat − C5 and
  C5 − C5sh (final relative difference and k against C5 / C5sh).
- **Escalation rule** (experiments.md §0.13) on the primary strata with single-seed substitutes
  (`convergence.escalation_verdict`): (i) the window (seed-inflated when available) CI of k, (ii) C2 beaten in the
  window-bootstrap final difference, (iii) the projected gap's seed CI (none with one seed) or k not shrinking from the
  next-smaller size; and the pre-registered phase-2 decision (`phase2_verdict`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..convergence import escalation_verdict
from ..provenance import git_state, prepare_output_dir, write_run_metadata
from ..statistics import _cluster_weights
from . import e4_report
from .e4_report import Run

TREND_STRATA = ("all", "unlinked", "after", "after_len3plus", "after_rare_seen", "after_heldout", "inside")
PRIMARY_STRATA = ("all", "after_len3plus")
BASELINE = "C0"
PAIRS = (("C5", "C0"), ("C2", "C0"), ("HRRAdd", "C0"), ("HRRCat", "C0"), ("C5sh", "C0"),
         ("HRRAdd", "C5"), ("HRRCat", "C5"), ("C5", "C5sh"))        # (condition, reference)
FRACTIONS = {"k": 0.1, "k_end": 0.0}
Z = 1.959964


# ---------------------------------------------------------------- per-window curves


@dataclass
class WindowCurves:
    """Per-window loss sums and target counts of every evaluation after the start (strata × evaluations × windows)."""
    tokens: np.ndarray
    strata: list[str]
    sums: np.ndarray
    counts: np.ndarray
    starts: np.ndarray

    def stratum(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        if name not in self.strata:
            return None
        i = self.strata.index(name)
        return self.sums[i], self.counts[i]


def load_window_curves(run_dir: Path) -> WindowCurves | None:
    from ..training.lm import load_window_losses
    if not (Path(run_dir) / "eval_windows.npz").exists():
        return None
    data = load_window_losses(Path(run_dir) / "eval_windows.npz")
    tokens = [t for t in data["evals"] if t > 0]
    if not tokens:
        return None
    return WindowCurves(np.asarray(tokens, dtype=np.float64), list(data["strata"]),
                        np.stack([data["evals"][t][0] for t in tokens], 1).astype(np.float64),
                        np.stack([data["evals"][t][1] for t in tokens], 1).astype(np.float64), np.asarray(data["starts"]))


def pool_seeds(curves: Sequence[WindowCurves]) -> WindowCurves | None:
    """Sum per-window sums and counts over seeds (same windows, evaluations and strata required)."""
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    first = curves[0]
    for c in curves[1:]:
        if not (np.array_equal(c.tokens, first.tokens) and c.strata == first.strata and np.array_equal(c.starts, first.starts)):
            return None
    return WindowCurves(first.tokens, first.strata, sum(c.sums for c in curves), sum(c.counts for c in curves), first.starts)


def resampled_curves(sums: np.ndarray, counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Loss curves (rows × evaluations) under window weights (rows × windows): `Σ_w a_w s_wt / Σ_w a_w n_wt`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ sums.T) / (weights @ counts.T)


def loss_target(first: np.ndarray, final: np.ndarray, fraction: float) -> np.ndarray:
    """Per row, the loss `convergence.common_targets` places at `fraction` of the way from the worst final loss to the
    best first point of a set of curves (`first`, `final`: rows × curves); NaN where no loss is reached by all of them.
    Fraction 0.1 is `common_targets`' first (lowest) target; fraction 0 is the worst final loss itself."""
    high, low = np.min(first, axis=1), np.max(final, axis=1)
    with np.errstate(invalid="ignore"):
        return np.where(high > low, low + (high - low) * fraction, np.nan)


def tokens_to_loss_rows(tokens: np.ndarray, losses: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """`convergence.tokens_to_loss` for many curves at once: per row of `losses` (rows × evaluations at `tokens`), the
    first token count at which the curve reaches its row's target, interpolated linearly in log-tokens from the previous
    evaluation (the first evaluation itself if it is already there); NaN when the curve never reaches it."""
    with np.errstate(invalid="ignore"):
        below = losses <= targets[:, None]
    rows = np.arange(losses.shape[0])
    first = below.argmax(1)
    previous = np.maximum(first - 1, 0)
    at, before = losses[rows, first], losses[rows, previous]
    log_tokens = np.log(tokens)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = (before - targets) / (before - at)
        interpolated = np.exp(log_tokens[previous] + fraction * (log_tokens[first] - log_tokens[previous]))
    value = np.where((first == 0) | (before == at), tokens[first], interpolated)
    return np.where(below.any(1), value, np.nan)


def multiplier_rows(tokens: np.ndarray, reference: np.ndarray, condition: np.ndarray, fraction: float) -> np.ndarray:
    """`k = D_ref(L) / D_cond(L)` per row of two curve matrices (rows × evaluations), the target recomputed per row."""
    target = loss_target(np.stack([reference[:, 0], condition[:, 0]], 1), np.stack([reference[:, -1], condition[:, -1]], 1),
                         fraction)
    return tokens_to_loss_rows(tokens, reference, target) / tokens_to_loss_rows(tokens, condition, target)


def _interval(draws: np.ndarray) -> dict[str, Any]:
    valid = draws[np.isfinite(draws)]
    if valid.size < 2:
        return {"ci_low": None, "ci_high": None, "valid_resamples": int(valid.size)}
    return {"ci_low": float(np.quantile(valid, 0.025)), "ci_high": float(np.quantile(valid, 0.975)),
            "valid_resamples": int(valid.size)}


def pair_statistics(reference: WindowCurves, condition: WindowCurves, stratum: str,
                    weights: np.ndarray) -> dict[str, Any] | None:
    """k, `k_end` (point and window-bootstrap draws) and the final losses of one stratum for a pair of runs."""
    a, b = reference.stratum(stratum), condition.stratum(stratum)
    if a is None or b is None or not np.array_equal(reference.tokens, condition.tokens) or a[1][-1].sum() == 0:
        return None
    ones = np.ones((1, a[0].shape[1]))
    point_ref, point_cond = resampled_curves(*a, ones), resampled_curves(*b, ones)
    draw_ref, draw_cond = resampled_curves(*a, weights), resampled_curves(*b, weights)
    result: dict[str, Any] = {"final_reference": float(point_ref[0, -1]), "final_condition": float(point_cond[0, -1]),
                              "final_draws": (draw_ref[:, -1], draw_cond[:, -1])}
    for name, fraction in FRACTIONS.items():
        point = float(multiplier_rows(reference.tokens, point_ref, point_cond, fraction)[0])
        draws = multiplier_rows(reference.tokens, draw_ref, draw_cond, fraction)
        with np.errstate(invalid="ignore"):
            logs = np.log(draws[np.isfinite(draws) & (draws > 0)])
        result[name] = {"k": point if math.isfinite(point) else None, **_interval(draws),
                        "log_sd": float(np.std(logs, ddof=1)) if logs.size > 1 else None, "draws": draws}
    return result


# ---------------------------------------------------------------- size trend


def non_embedding_parameters(size: str) -> int:
    """Parameters of a from-scratch host outside its token and position embeddings (the tied head is the token table):
    `12·L·d² + 13·L·d` per transformer stack plus the final layer norm `2d`."""
    from ..training.lm import MODEL_SIZES
    layers, width = MODEL_SIZES[size]["n_layer"], MODEL_SIZES[size]["n_embd"]
    return 12 * layers * width**2 + 13 * layers * width + 2 * width


def ols_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares slope of each row of `y` (rows × points) on `x`; NaN for a row with a non-finite value."""
    centred = x - x.mean()
    weights = centred / (centred**2).sum()
    return (np.atleast_2d(y) * weights).sum(-1)


def slope_weight(x: np.ndarray) -> float:
    """√Σc_i² of the OLS slope `Σ c_i y_i`: the factor on an independent per-point SD in the slope's SD."""
    centred = x - x.mean()
    return float(np.sqrt(((centred / (centred**2).sum()) ** 2).sum()))


def inflated_interval(point: float | None, draws: np.ndarray | None, extra_sd: float | None) -> tuple[float, float] | None:
    """Normal interval `point ± z·√(var(draws) + extra_sd²)`: window-bootstrap variance plus an independent seed SD."""
    if point is None or extra_sd is None or draws is None:
        return None
    valid = draws[np.isfinite(draws)]
    if valid.size < 2:
        return None
    half = Z * math.sqrt(float(np.var(valid, ddof=1)) + extra_sd**2)
    return point - half, point + half


def size_trend(by_size: dict[str, dict[str, Any]], sizes: Sequence[str], seed_sd: float | None, key: str) -> dict[str, Any] | None:
    """Slope of log k (`key`: k or k_end) on log N over the sizes with a k, its window-bootstrap percentile CI (draws aligned
    by resample index) and, with a seed SD of log k, the seed-inflated CI."""
    present = [s for s in sizes if by_size.get(s) and by_size[s][key]["k"] and by_size[s][key]["k"] > 0]
    if len(present) < 2:
        return None
    x = np.log([non_embedding_parameters(s) for s in present])
    point = float(ols_slope(x, np.log([by_size[s][key]["k"] for s in present]))[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = ols_slope(x, np.log(np.stack([by_size[s][key]["draws"] for s in present], 1)))
    inflated = inflated_interval(point, draws, None if seed_sd is None else seed_sd * slope_weight(x))
    return {"sizes": present, "slope": point, **_interval(draws),
            "seed_ci": None if inflated is None else {"ci_low": inflated[0], "ci_high": inflated[1]},
            "draws": draws}


# ---------------------------------------------------------------- seed reference


def _proxy(condition: str, reference: str) -> str:
    if reference == BASELINE:
        return f"{'C2' if condition == 'C2' else 'C5'}|{BASELINE}"
    return "C5@*|C5"


def seed_reference(runs: Sequence[Run], strata: Sequence[str] = TREND_STRATA, *, resamples: int = 2000,
                   seed: int = 0) -> dict[str, Any]:
    """Seed SDs of log k (both targets) and of the relative final gap from a multi-seed cohort (the opscreen), per
    stratum for C2 vs C0, C5 vs C0 and (pooled) every C5 variant `C5@…` vs C5; with saved windows also the window-
    bootstrap SD of log k for the cohort's first seed, to compare evaluation noise with seed noise."""
    grid: dict[str, dict[int, Run]] = {}
    for run in runs:
        if run.complete and not run.resume_check:
            grid.setdefault(run.condition, {})[run.seed] = run
    pairs = [(c, BASELINE) for c in ("C2", "C5") if c in grid] + [(c, "C5") for c in grid if c.startswith("C5@")]
    result: dict[str, Any] = {"conditions": {c: sorted(r) for c, r in grid.items()}, "strata": {}}
    loaded: dict[Path, WindowCurves | None] = {}

    def windows(run: Run) -> WindowCurves | None:
        if run.path not in loaded:
            loaded[run.path] = load_window_curves(run.path)
        return loaded[run.path]

    for stratum in strata:
        per_pair: dict[str, Any] = {}
        for condition, reference in pairs:
            if reference not in grid:
                continue
            seeds = sorted(set(grid[condition]) & set(grid[reference]))
            values: dict[str, list[float]] = {"k": [], "k_end": [], "gap": []}
            window_sd = None
            for s in seeds:
                a, b = windows(grid[reference][s]), windows(grid[condition][s])
                if a is None or b is None:
                    continue
                stats = pair_statistics(a, b, stratum, _cluster_weights(a.sums.shape[2], resamples, seed))
                if stats is None:
                    continue
                for key in FRACTIONS:
                    if stats[key]["k"]:
                        values[key].append(math.log(stats[key]["k"]))
                values["gap"].append(stats["final_condition"] / stats["final_reference"] - 1)
                if window_sd is None:
                    window_sd = stats["k"]["log_sd"]
            if len(values["k"]) >= 2:
                per_pair[f"{condition}|{reference}"] = {
                    "seeds": seeds, **{f"{key}_log_sd": float(np.std(v, ddof=1)) if len(v) >= 2 else None for key, v in
                                       (("k", values["k"]), ("k_end", values["k_end"]))},
                    "gap_sd": float(np.std(values["gap"], ddof=1)), "k_mean": float(np.exp(np.mean(values["k"]))),
                    "window_log_sd_first_seed": window_sd}
        variants = [v for k, v in per_pair.items() if k.startswith("C5@")]
        if variants:
            per_pair["C5@*|C5"] = {key: float(np.sqrt(np.mean([v[key] ** 2 for v in variants])))
                                   if all(v.get(key) is not None for v in variants) else None
                                   for key in ("k_log_sd", "k_end_log_sd", "gap_sd")} | {"pooled": len(variants)}
        result["strata"][stratum] = per_pair
    return result


def _seed_sd(reference: dict[str, Any] | None, stratum: str, condition: str, ref: str, key: str) -> float | None:
    if not reference:
        return None
    entry = reference["strata"].get(stratum, {}).get(_proxy(condition, ref))
    return None if entry is None else entry.get(f"{key}_log_sd")


# ---------------------------------------------------------------- analysis


def _windows_by_condition(grid: dict[str, dict[int, Run]], condition: str, seeds: Sequence[int]) -> WindowCurves | None:
    return pool_seeds([load_window_curves(grid[condition][s].path) for s in seeds])


def _paired(cohort: dict[str, Any], condition: str, reference: str, stratum: str) -> dict[str, Any] | None:
    return cohort["paired"].get(reference, {}).get(condition, {}).get(stratum)


def analyze_scale(runs: Sequence[Run], *, reference_runs: Sequence[Run] = (), resamples: int = 10_000,
                  k_resamples: int = 2000, seed: int = 0,
                  projection_tokens: Sequence[float] = (2.5e9, 7e9)) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """`(scale summary, e4 summary, e4 grids)`; see the module docstring."""
    from ..training.lm import MODEL_SIZES
    e4, grids = e4_report.analyze(runs, baseline=BASELINE, references=("C5sh", "C5"), candidates=("C5", "HRRAdd", "HRRCat"),
                                  projection_tokens=projection_tokens, resamples=resamples, seed=seed)
    warnings: list[str] = []
    labels: dict[str, str] = {}
    for label, cohort in sorted(e4["cohorts"].items(), key=lambda item: -item[1]["budget_tokens"]):
        if cohort["model"] not in MODEL_SIZES:
            warnings.append(f"cohort {label}: not a from-scratch size, left out of the size trend")
        elif cohort["model"] in labels:
            warnings.append(f"cohort {label}: a second cohort of size {cohort['model']} (kept {labels[cohort['model']]})")
        else:
            labels[cohort["model"]] = label
    sizes = sorted(labels, key=non_embedding_parameters)
    reference = seed_reference(reference_runs, resamples=k_resamples, seed=seed) if reference_runs else None
    # One weight matrix per distinct window set (shared windows: the same resample at every size).
    window_sets: dict[bytes, np.ndarray] = {}
    pairs: dict[str, dict[str, dict[str, Any]]] = {}        # "cond|ref" → stratum → size → statistics
    shares: dict[str, dict[str, Any]] = {}
    for size in sizes:
        grid = grids[labels[size]]
        cache: dict[tuple[str, tuple[int, ...]], WindowCurves | None] = {}
        for condition, ref in PAIRS:
            if condition not in grid or ref not in grid:
                continue
            seeds = tuple(sorted(set(grid[condition]) & set(grid[ref])))
            for name in (condition, ref):
                if (name, seeds) not in cache:
                    cache[(name, seeds)] = _windows_by_condition(grid, name, seeds)
            a, b = cache[(ref, seeds)], cache[(condition, seeds)]
            if a is None or b is None:
                warnings.append(f"{size} {condition} vs {ref}: no per-window losses of every evaluation; no k CIs")
                continue
            key = a.starts.tobytes()
            if key not in window_sets:
                window_sets[key] = _cluster_weights(a.sums.shape[2], k_resamples, seed + len(window_sets))
            for stratum in TREND_STRATA:
                stats = pair_statistics(a, b, stratum, window_sets[key])
                if stats is not None:
                    pairs.setdefault(f"{condition}|{ref}", {}).setdefault(stratum, {})[size] = stats
        for stratum in TREND_STRATA:            # C2's share of C5's final gap, same resamples
            c2, c5 = (pairs.get(f"{c}|{BASELINE}", {}).get(stratum, {}).get(size) for c in ("C2", "C5"))
            if c2 and c5:
                with np.errstate(invalid="ignore", divide="ignore"):
                    point = (c2["final_reference"] - c2["final_condition"]) / (c5["final_reference"] - c5["final_condition"])
                    draws = (c2["final_draws"][0] - c2["final_draws"][1]) / (c5["final_draws"][0] - c5["final_draws"][1])
                gap = _paired(e4["cohorts"][labels[size]], "C5", BASELINE, stratum) or {}
                shares.setdefault(stratum, {})[size] = {"share": float(point), **_interval(draws),
                                                        "c5_gap_significant": bool(gap.get("significant") and gap["delta"] < 0)}
    if len(window_sets) > 1:
        warnings.append("sizes were evaluated on different windows: their bootstrap draws are independent")
    trend: dict[str, dict[str, Any]] = {}
    for pair, by_stratum in pairs.items():
        condition, ref = pair.split("|")
        for stratum, by_size in by_stratum.items():
            entry: dict[str, Any] = {"sizes": {}, "trend": {}}
            for size, stats in by_size.items():
                cohort = e4["cohorts"][labels[size]]
                row: dict[str, Any] = {"final_reference": stats["final_reference"], "final_condition": stats["final_condition"],
                                       "paired": _strip(_paired(cohort, condition, ref, stratum))}
                for key in FRACTIONS:
                    k = stats[key]
                    sd = _seed_sd(reference, stratum, condition, ref, key)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        inflated = (inflated_interval(math.log(k["k"]), np.log(k["draws"]), sd)
                                    if k["k"] and k["k"] > 0 and sd is not None else None)
                    row[key] = {**{n: v for n, v in k.items() if n != "draws"}, "seed_sd": sd,
                                "seed_ci": None if inflated is None else {"ci_low": math.exp(inflated[0]), "ci_high": math.exp(inflated[1])}}
                if ref == BASELINE:
                    row["compute"] = _compute_adjusted(grids[labels[size]], condition, row["k"])
                    comparison = cohort["convergence"].get(stratum, {}).get(condition) or {}
                    row["projected_gaps"] = {t: (v or {}).get("mean") for t, v in comparison.get("projected_loss_gaps", {}).items()}
                entry["sizes"][size] = row
            for key in FRACTIONS:
                trend_ = size_trend(by_size, sizes, _seed_sd(reference, stratum, condition, ref, key), key)
                entry["trend"][key] = None if trend_ is None else {n: v for n, v in trend_.items() if n != "draws"}
            trend.setdefault(pair, {})[stratum] = entry
    rules = escalation_primary(trend, e4, labels, sizes)
    summary = {"sizes": sizes, "cohorts": labels, "non_embedding_parameters": {s: non_embedding_parameters(s) for s in sizes},
               "k_resamples": k_resamples, "resamples": resamples, "shared_windows": len(window_sets) <= 1,
               "trend": trend, "c2_share": shares, "seed_reference": reference, "escalation": rules,
               "phase2": phase2_verdict(trend, e4, labels, sizes, rules), "warnings": warnings,
               "single_seed": all(e4["cohorts"][labels[s]]["single_seed"] for s in sizes) if sizes else None}
    return summary, e4, grids


def _strip(paired: dict[str, Any] | None) -> dict[str, Any] | None:
    if paired is None:
        return None
    return {k: paired.get(k) for k in ("delta", "ci_low", "ci_high", "relative", "relative_ci_low", "relative_ci_high",
                                       "holm_p", "significant", "method")}


def _compute_adjusted(grid: dict[str, dict[int, Run]], condition: str, k: dict[str, Any]) -> dict[str, Any]:
    """k in wall-clock (× tokens/s ratio, measured) and FLOPs (× 6N ratio of total parameters, channel counted dense)."""
    def mean(name: str, value) -> float | None:
        values = [value(r) for r in grid[name].values() if value(r)]
        return float(np.mean(values)) if values else None
    speed = {n: mean(n, lambda r: r.tokens_per_s) for n in (BASELINE, condition)}
    params = {n: mean(n, lambda r: (r.manifest or {}).get("parameters")) for n in (BASELINE, condition)}
    wall = speed[condition] / speed[BASELINE] if speed[condition] and speed[BASELINE] else None
    flops = params[BASELINE] / params[condition] if params[condition] and params[BASELINE] else None
    scaled = {}
    for name, factor in (("wall_clock", wall), ("flops", flops)):
        scaled[name] = None if factor is None or not k.get("k") else {
            "factor": factor, "k": k["k"] * factor, "ci_low": None if k.get("ci_low") is None else k["ci_low"] * factor,
            "ci_high": None if k.get("ci_high") is None else k["ci_high"] * factor}
    return {"tokens_per_s": speed, "parameters": params, **scaled}


def escalation_primary(trend: dict[str, Any], e4: dict[str, Any], labels: dict[str, str], sizes: Sequence[str]) -> dict[str, Any]:
    """`convergence.escalation_verdict` per size on the primary strata for the conditions vs C0, with single-seed
    substitutes: k's interval is the seed-inflated window CI (else the window CI), the C2 comparison is the window-
    bootstrap final difference, the projection keeps the E4 analysis' seed interval (none with one seed), and the size
    trend compares with the next-smaller size's k."""
    result: dict[str, Any] = {}
    for pair, by_stratum in trend.items():
        condition, ref = pair.split("|")
        if ref != BASELINE or condition == "C2":
            continue
        for stratum in PRIMARY_STRATA:
            entry = by_stratum.get(stratum)
            previous = None
            for size in sizes:
                row = (entry or {}).get("sizes", {}).get(size)
                if row is None:
                    continue
                k = row["k"]
                interval = k["seed_ci"] or {"ci_low": k["ci_low"], "ci_high": k["ci_high"]}
                cohort = e4["cohorts"][labels[size]]
                comparison = cohort["convergence"].get(stratum, {}).get(condition) or {}
                versus = {"data_multiplier": {"0": {"mean": k["k"], "ci_low": interval["ci_low"]}},
                          "projected_loss_gap": comparison.get("projected_loss_gap")}
                controls = {}
                c2 = _paired(cohort, condition, "C2", stratum)
                if c2 is not None and c2.get("ci_low") is not None:
                    controls["C2"] = {"final_loss_gap": {"mean": -c2["delta"], "ci_low": -c2["ci_high"], "ci_high": -c2["ci_low"]}}
                verdict = escalation_verdict(versus, controls, smaller_model_multiplier=previous)
                verdict.update(smaller_model_multiplier=previous, k_interval="seed-inflated" if k["seed_ci"] else "window")
                result.setdefault(pair, {}).setdefault(stratum, {})[size] = verdict
                previous = k["k"]
    return result


def phase2_verdict(trend: dict[str, Any], e4: dict[str, Any], labels: dict[str, str], sizes: Sequence[str],
                   rules: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered phase-2 decision (preregistration-scale-v1.md §6): per primary stratum, C5 beats C0 at the
    largest size (window-bootstrap final difference below 0, Holm), rule (iii) holds there, and shrinkage is not
    established (the slope of log k on log N has a CI — seed-inflated when available — whose upper bound is ≥ 0);
    phase 2 is requested if a primary stratum passes all three. A hybrid goes forward if it beats C5 (final difference,
    Holm) on a primary stratum at ≥ 2 of the sizes."""
    if not sizes:
        return {"request_phase2": False, "detail": "no sizes"}
    largest = sizes[-1]
    strata: dict[str, Any] = {}
    for stratum in PRIMARY_STRATA:
        entry = trend.get(f"C5|{BASELINE}", {}).get(stratum)
        paired = _paired(e4["cohorts"][labels[largest]], "C5", BASELINE, stratum)
        beats = bool(paired and paired.get("significant") and paired["delta"] < 0)
        rule = rules.get(f"C5|{BASELINE}", {}).get(stratum, {}).get(largest)
        slope = (entry or {}).get("trend", {}).get("k")
        interval = slope and (slope["seed_ci"] or ({"ci_low": slope["ci_low"], "ci_high": slope["ci_high"]}
                                                   if slope["ci_high"] is not None else None))
        not_shrinking = None if not interval else interval["ci_high"] >= 0
        strata[stratum] = {"c5_beats_c0_at_largest": beats, "rule_iii_at_largest": None if rule is None else rule["rule_iii"],
                           "shrinkage_not_established": not_shrinking, "slope": None if not slope else slope["slope"],
                           "slope_interval": interval, "slope_interval_kind": None if not slope else
                           ("seed-inflated" if slope["seed_ci"] else "window"),
                           "pass": bool(beats and rule and rule["rule_iii"] and not_shrinking)}
    hybrids = {}
    for name in ("HRRAdd", "HRRCat"):
        wins = {stratum: [s for s in sizes if (p := _paired(e4["cohorts"][labels[s]], name, "C5", stratum))
                          and p.get("significant") and p["delta"] < 0] for stratum in PRIMARY_STRATA}
        hybrids[name] = {"sizes_beating_c5": wins, "forward": any(len(v) >= 2 for v in wins.values())}
    return {"largest_size": largest, "complete_sizes": len(sizes), "strata": strata, "hybrids": hybrids,
            "request_phase2": any(v["pass"] for v in strata.values())}


# ---------------------------------------------------------------- rendering


def _k_cell(k: dict[str, Any] | None, *, seed: bool = False) -> str:
    if not k or k.get("k") is None:
        return "–"
    text = f"{k['k']:.3f}"
    if k.get("ci_low") is not None:
        text += f" [{k['ci_low']:.3f}, {k['ci_high']:.3f}]"
    if seed and k.get("seed_ci"):
        text += f" ⟨{k['seed_ci']['ci_low']:.3f}, {k['seed_ci']['ci_high']:.3f}⟩"
    return text


def _slope_cell(slope: dict[str, Any] | None) -> str:
    if not slope:
        return "–"
    text = f"{slope['slope']:+.4f}"
    if slope.get("ci_low") is not None:
        text += f" [{slope['ci_low']:+.4f}, {slope['ci_high']:+.4f}]"
    if slope.get("seed_ci"):
        text += f" ⟨{slope['seed_ci']['ci_low']:+.4f}, {slope['seed_ci']['ci_high']:+.4f}⟩"
    return text


def _gap_cell(paired: dict[str, Any] | None) -> str:
    return e4_report._relative_cell(paired) if paired else "–"


def render(summary: dict[str, Any], e4: dict[str, Any], *, title: str) -> str:
    sizes = summary["sizes"]
    lines = [f"# {title}", ""]
    complete = sum(r["complete"] for r in e4["runs"])
    counts = ", ".join(f"{s} {summary['non_embedding_parameters'][s]:,}" for s in sizes)
    lines += [f"{len(e4['runs'])} runs found, {complete} complete; sizes {', '.join(sizes) or 'none'} "
              f"(non-embedding parameters: {counts or 'n/a'}). "
              "Pre-registration: `experiments/e4-small-lm/preregistration-scale-v1.md`. The per-size E4 analysis "
              "(final losses, every paired comparison, fits, gate items, figures) is in [e4-report.md](e4-report.md).", ""]
    if summary["single_seed"]:
        lines += ["> **Single seed per size (phase 1, exploratory).** `[a, b]` are 95% percentile bootstraps over evaluation "
                  f"windows ({summary['k_resamples']} resamples, shared across sizes: "
                  f"{'yes' if summary['shared_windows'] else 'no'}): evaluation noise only. `⟨a, b⟩` add the seed variance "
                  "of the reference cohort (`--seed-reference`) in quadrature" + ("" if summary["seed_reference"] else
                  " — none given, so no seed-inflated intervals") + ". Final gaps are the E4 analysis' window-bootstrap "
                  "relative differences (`*` = Holm-significant within the size).", ""]
    lines += [f"- warning: {w}" for w in summary["warnings"]] + ([""] if summary["warnings"] else [])
    phase2 = summary["phase2"]
    lines += ["## Phase-2 decision (pre-registered rule, computed mechanically)", ""]
    if "strata" in phase2:
        lines += [f"Largest size {phase2['largest_size']} ({phase2['complete_sizes']} sizes complete). "
                  f"**Request phase 2: {phase2['request_phase2']}.**", "",
                  "| Stratum | C5 beats C0 at the largest size | rule (iii) there | slope of log k on log N [interval] | "
                  "shrinkage not established | pass |", "|---|---|---|---|---|---|"]
        for stratum, v in phase2["strata"].items():
            interval = v["slope_interval"]
            slope = "–" if v["slope"] is None else f"{v['slope']:+.4f}" + (
                "" if not interval else f" [{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}] ({v['slope_interval_kind']})")
            lines.append(f"| {stratum} | {v['c5_beats_c0_at_largest']} | {v['rule_iii_at_largest']} | {slope} | "
                         f"{v['shrinkage_not_established']} | **{v['pass']}** |")
        lines += ["", "Hybrids (forward to phase 2 if they beat C5 on a primary stratum at ≥ 2 sizes): " + "; ".join(
            f"{name} **{v['forward']}** (sizes beating C5 — " + " · ".join(
                f"{s}: {' '.join(z) or 'none'}" for s, z in v["sizes_beating_c5"].items()) + ")"
            for name, v in phase2["hybrids"].items()) + ".", ""]
    lines += ["## Size trend vs C0", "",
              "k = D_C0(L) / D_cond(L) at the lowest common loss target (the escalation rule's), `k_end` at the worse final "
              "loss of the pair; slope = d ln k / d ln N over the sizes. Final gap: (cond − C0) / C0 at the end (negative = "
              "lower loss).", "",
              "| Stratum | Condition | " + " | ".join(f"k {s}" for s in sizes) + " | slope (k) | " +
              " | ".join(f"k_end {s}" for s in sizes) + " | slope (k_end) | " + " | ".join(f"gap {s}" for s in sizes) + " |",
              "|---|---|" + "---|" * (3 * len(sizes) + 2)]
    for stratum in TREND_STRATA:
        for pair in (f"{c}|{BASELINE}" for c, r in PAIRS if r == BASELINE):
            entry = summary["trend"].get(pair, {}).get(stratum)
            if not entry:
                continue
            row = entry["sizes"]
            cells = [_k_cell(row.get(s, {}).get("k"), seed=True) for s in sizes] + [_slope_cell(entry["trend"]["k"])]
            cells += [_k_cell(row.get(s, {}).get("k_end"), seed=True) for s in sizes] + [_slope_cell(entry["trend"]["k_end"])]
            cells += [_gap_cell(row.get(s, {}).get("paired")) for s in sizes]
            lines.append(f"| {stratum} | {pair.split('|')[0]} | " + " | ".join(cells) + " |")
    lines += ["", "### Compute-adjusted k (vs C0, lowest common target)", "",
              "Wall clock: k × (tokens/s of the condition / tokens/s of C0), measured medians. FLOPs: k × 6N_C0 / 6N_cond "
              "with every channel parameter counted as dense compute (an upper bound on the channel's cost).", "",
              "| Stratum | Condition | Size | k | tokens/s ratio | k wall clock | 6N ratio | k FLOPs (upper-bound cost) | "
              "projected gap @ " + " / ".join(e4_report._tokens_label(t) for t in e4["projection_tokens"]) + " (1 seed) |",
              "|---|---|---|---|---:|---|---:|---|---|"]
    for stratum in PRIMARY_STRATA:
        for pair in (f"{c}|{BASELINE}" for c, r in PAIRS if r == BASELINE):
            for size in sizes:
                row = summary["trend"].get(pair, {}).get(stratum, {}).get("sizes", {}).get(size)
                if not row:
                    continue
                wall, flops = row["compute"]["wall_clock"], row["compute"]["flops"]
                factors = ["n/a" if not v else f"{v['factor']:.3f}" for v in (wall, flops)]
                gaps = " / ".join("n/a" if v is None else f"{v:+.4f}" for v in row["projected_gaps"].values()) or "n/a"
                lines.append(f"| {stratum} | {pair.split('|')[0]} | {size} | {_k_cell(row['k'])} | {factors[0]} | "
                             f"{_k_cell(wall)} | {factors[1]} | {_k_cell(flops)} | {gaps} |")
    lines += ["", "### C2's share of C5's final gap", "", "(L_C0 − L_C2) / (L_C0 − L_C5) at the end; window bootstrap; shown "
              "only where C5's own gap is Holm-significant (n.s.: the ratio's denominator is indistinguishable from 0).", "",
              "| Stratum | " + " | ".join(sizes) + " |", "|---|" + "---|" * len(sizes)]

    def share_cell(v: dict[str, Any] | None) -> str:
        if not v:
            return "–"
        if not v["c5_gap_significant"]:
            return "n.s."
        return f"{v['share']:.2f}" + ("" if v["ci_low"] is None else f" [{v['ci_low']:.2f}, {v['ci_high']:.2f}]")

    for stratum in TREND_STRATA:
        by_size = summary["c2_share"].get(stratum, {})
        if by_size:
            lines.append(f"| {stratum} | " + " | ".join(share_cell(by_size.get(s)) for s in sizes) + " |")
    lines += ["", "## Hybrids vs C5 and specificity (C5 vs C5sh)", "",
              "k against C5 (or C5sh) as the reference, and the final relative difference (condition − reference) / reference.", "",
              "| Comparison | Stratum | " + " | ".join(f"k {s}" for s in sizes) + " | slope (k) | " +
              " | ".join(f"gap {s}" for s in sizes) + " |", "|---|---|" + "---|" * (2 * len(sizes) + 1)]
    for condition, ref in PAIRS:
        if ref == BASELINE:
            continue
        for stratum in TREND_STRATA:
            entry = summary["trend"].get(f"{condition}|{ref}", {}).get(stratum)
            if entry:
                row = entry["sizes"]
                lines.append(f"| {condition} − {ref} | {stratum} | " + " | ".join(_k_cell(row.get(s, {}).get("k"), seed=True) for s in sizes)
                             + f" | {_slope_cell(entry['trend']['k'])} | " + " | ".join(_gap_cell(row.get(s, {}).get("paired")) for s in sizes) + " |")
    lines += ["", "## Escalation rule on the primary strata (single-seed substitutes)", "",
              "(i) k's interval low > 1.1 (seed-inflated when a reference is given, else window); (ii) C2 beaten (window-"
              "bootstrap final difference); (iii) projected-gap seed CI excludes 0 (not estimable with one seed) or k ≥ the "
              "next-smaller size's k.", "",
              "| Condition | Stratum | Size | k | interval low | (i) | (ii) C2 | (iii) projection / size trend | smaller k | escalate |",
              "|---|---|---|---:|---:|---|---|---|---:|---|"]
    for pair, by_stratum in summary["escalation"].items():
        for stratum, by_size in by_stratum.items():
            for size, v in by_size.items():
                controls = ", ".join(f"{'beaten' if ok else 'not beaten'}" for ok in v["controls_beaten"].values()) or "n/a"
                lines.append(f"| {pair.split('|')[0]} | {stratum} | {size} | {e4_report._number(v['multiplier_mean'])} | "
                             f"{e4_report._number(v['multiplier_ci_low'])} ({v['k_interval']}) | {v['rule_i']} | {controls} | "
                             f"{v['projection_ok']} / {v['size_trend_ok']} | {e4_report._number(v['smaller_model_multiplier'])} | "
                             f"**{v['escalate']}** |")
    reference = summary["seed_reference"]
    if reference:
        lines += ["", "## Seed-variance reference", "",
                  "Seed SD over the reference cohort's seeds (" + "; ".join(f"{c} {s}" for c, s in reference["conditions"].items())
                  + ") of ln k (lowest common target, worse final loss) and of the relative final gap, against the window-"
                  "bootstrap SD of ln k for its first seed. The scale-v1 seed-inflated intervals use C2 vs C0 for C2, C5 vs C0 "
                  "for the composed conditions and the pooled C5 variants vs C5 between channel conditions.", "",
                  "| Stratum | Pair | seeds | k (mean) | SD ln k | SD ln k_end | SD gap | window SD ln k (first seed) |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for stratum, per_pair in reference["strata"].items():
            for pair, v in per_pair.items():
                seeds = v["seeds"] if "seeds" in v else f"pooled over {v['pooled']} variants"
                lines.append(f"| {stratum} | {pair.replace('|', ' vs ')} | {seeds} | {e4_report._number(v.get('k_mean'))} | "
                             f"{e4_report._number(v.get('k_log_sd'), 4)} | {e4_report._number(v.get('k_end_log_sd'), 4)} | "
                             f"{e4_report._number(v.get('gap_sd'), 5)} | {e4_report._number(v.get('window_log_sd_first_seed'), 4)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def write_report(runs_dirs: Sequence[Path], output: Path, *, seed_reference_dirs: Sequence[Path] = (), resamples: int = 10_000,
                 k_resamples: int = 2000, seed: int = 0, projection_tokens: Sequence[float] = (2.5e9, 7e9),
                 title: str = "E4 scaling screen, phase 1 (decision 65)", figures: bool = True,
                 overwrite: bool = False) -> dict[str, Any]:
    config = {"runs": [str(p) for p in runs_dirs], "seed_reference": [str(p) for p in seed_reference_dirs],
              "resamples": resamples, "k_resamples": k_resamples, "seed": seed, "projection_tokens": list(projection_tokens),
              "title": title}
    if overwrite and output.exists():
        git_at_start = git_state()
    else:
        git_at_start = prepare_output_dir(output)
    runs = e4_report.discover(runs_dirs)
    if not runs:
        raise FileNotFoundError(f"no run folders (metrics.jsonl) under {', '.join(map(str, runs_dirs))}")
    reference = e4_report.discover(seed_reference_dirs) if seed_reference_dirs else []
    summary, e4, grids = analyze_scale(runs, reference_runs=reference, resamples=resamples, k_resamples=k_resamples, seed=seed,
                                       projection_tokens=projection_tokens)
    plots: dict[str, dict[str, str]] = {}
    if figures:
        (output / "figures").mkdir(exist_ok=True)
        for i, (label, cohort) in enumerate(e4["cohorts"].items()):
            slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower() or f"cohort-{i}"
            plots[label] = e4_report.plot_cohort(label, cohort, grids[label], output, slug)
    (output / "e4-summary.json").write_text(json.dumps(e4, indent=2, default=e4_report._json_default) + "\n")
    (output / "e4-report.md").write_text(e4_report.render(e4, figures=plots, title=f"{title}: per-size E4 analysis"))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=e4_report._json_default) + "\n")
    (output / "report.md").write_text(render(summary, e4, title=title))
    write_run_metadata(output, config, git_at_start=git_at_start, runs=len(runs), complete_runs=sum(r.complete for r in runs),
                       sizes=summary["sizes"], request_phase2=summary["phase2"].get("request_phase2"))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-reference", type=Path, nargs="*", default=[],
                        help="run folders of a multi-seed cohort (the opscreen) whose seed variance widens the single-seed CIs")
    parser.add_argument("--resamples", type=int, default=10_000, help="window resamples of the E4 paired differences")
    parser.add_argument("--k-resamples", type=int, default=2000, help="window resamples of k, k_end, slopes and shares")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection-tokens", type=float, nargs="+", default=[2.5e9, 7e9])
    parser.add_argument("--title", default="E4 scaling screen, phase 1 (decision 65)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="write into an existing output folder (a re-run)")
    args = parser.parse_args(argv)
    summary = write_report(args.runs, args.output, seed_reference_dirs=args.seed_reference, resamples=args.resamples,
                           k_resamples=args.k_resamples, seed=args.seed, projection_tokens=args.projection_tokens,
                           title=args.title, figures=not args.no_figures, overwrite=args.overwrite)
    print(json.dumps({"sizes": summary["sizes"], "request_phase2": summary["phase2"].get("request_phase2")}))


if __name__ == "__main__":
    main()
