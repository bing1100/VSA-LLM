"""Confidence intervals and rank helpers shared by experiment summaries."""

from __future__ import annotations

import functools
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

# Two-sided 97.5% Student-t quantiles; beyond 30 degrees of freedom the normal value is used.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
    18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def mean_confidence_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    """95% Student-t interval for the mean of a few paired differences (e.g. seeds)."""
    data = [float(value) for value in values]
    n = len(data)
    if n == 0:
        raise ValueError("values must be non-empty")
    mean = sum(data) / n
    if n == 1:
        return {"mean": mean, "ci_low": None, "ci_high": None, "n": 1}
    variance = sum((value - mean) ** 2 for value in data) / (n - 1)
    half_width = _T975.get(n - 1, 1.96) * math.sqrt(variance / n)
    return {"mean": mean, "ci_low": mean - half_width, "ci_high": mean + half_width, "n": n}


def paired_bootstrap_ci(
    candidate: Sequence[float] | Tensor, control: Sequence[float] | Tensor, *,
    resamples: int = 2000, seed: int = 0,
) -> dict[str, float | int]:
    """Percentile 95% interval of the mean paired difference, resampling examples."""
    a = torch.as_tensor(candidate, dtype=torch.float64)
    b = torch.as_tensor(control, dtype=torch.float64)
    if a.shape != b.shape or a.ndim != 1 or a.numel() == 0:
        raise ValueError("candidate and control must be equal-length non-empty vectors")
    differences = a - b
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, differences.numel(), (resamples, differences.numel()), generator=generator)
    means = differences[index].mean(1)
    return {
        "mean": float(differences.mean()),
        "ci_low": float(torch.quantile(means, 0.025)),
        "ci_high": float(torch.quantile(means, 0.975)),
        "n": int(differences.numel()),
    }


@functools.lru_cache(maxsize=8)
def _cluster_weights(clusters: int, resamples: int, seed: int) -> np.ndarray:
    """Multiplicity of each cluster in each bootstrap resample (resamples × clusters)."""
    return np.random.default_rng(seed).multinomial(clusters, np.full(clusters, 1.0 / clusters), size=resamples).astype(np.float64)


def paired_ratio_bootstrap(
    differences: Sequence[float] | np.ndarray, denominators: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray | None = None, *, resamples: int = 2000, seed: int = 0,
) -> dict[str, float | int | None]:
    """Cluster bootstrap of a token-weighted paired difference `Σ d / Σ n`.

    Each cluster (e.g. an evaluation window, carrying every seed) contributes its summed paired
    loss difference `d` and its target count `n`; resampling clusters keeps text shared across
    seeds together. With `baseline` (the reference's summed loss per cluster) the relative
    difference `Σ d / Σ baseline` is bootstrapped from the same resamples. `p_value` is the
    two-sided percentile-bootstrap p for a zero difference. The same `(clusters, resamples,
    seed)` reuse the same resamples (common random numbers across strata and conditions).
    """
    d = np.asarray(differences, dtype=np.float64)
    n = np.asarray(denominators, dtype=np.float64)
    if d.shape != n.shape or d.ndim != 1 or d.size == 0:
        raise ValueError("differences and denominators must be equal-length non-empty vectors")
    if n.sum() <= 0:
        raise ValueError("denominators sum to zero")
    weights = _cluster_weights(d.size, resamples, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (weights @ d) / (weights @ n)
        draws = draws[np.isfinite(draws)]
        result: dict[str, float | int | None] = {
            "mean": float(d.sum() / n.sum()),
            "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p_value": float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (draws.size + 1))),
            "clusters": int(d.size), "nonempty_clusters": int((n > 0).sum()), "resamples": int(draws.size),
        }
        if baseline is not None:
            b = np.asarray(baseline, dtype=np.float64)
            relative = (weights @ d) / (weights @ b)
            relative = relative[np.isfinite(relative)]
            result.update(relative=float(d.sum() / b.sum()), relative_ci_low=float(np.quantile(relative, 0.025)),
                          relative_ci_high=float(np.quantile(relative, 0.975)))
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [1.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def crossed_components(values: Sequence[Sequence[float]] | np.ndarray) -> dict[str, float | int | None]:
    """Crossed random-effects model `y[c, s] = μ + a_c + b_s + e_cs` of a complete clusters × seeds table (one value per
    cell, e.g. a concept's mean paired difference in one training seed; items × seeds in the claim-C analysis).

    ANOVA estimators of the variance components (equal to REML for a balanced table when none is truncated at 0):
    `var_cluster` = (MS_c − MS_e) / S, `var_seed` = (MS_s − MS_e) / n, `var_residual` = MS_e, truncated at 0. The
    mean's variance `Var(μ̂) = var_cluster / n + var_seed / S + var_residual / (n S)` is a linear combination of the
    mean squares, whose Satterthwaite degrees of freedom give a t interval and a two-sided p. One seed: the ordinary
    one-sample t (the seed and residual components are not identifiable). `sd_cluster_mean` is the SD of the
    clusters' seed-averaged values (`sqrt(var_cluster + var_residual / S)`), the per-cluster SD that a seed-averaged
    paired test uses."""
    from scipy import stats
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 1 or not np.isfinite(y).all():
        raise ValueError("values must be a finite clusters × seeds table with at least 2 clusters")
    n, s = y.shape
    mu = float(y.mean())
    rows, cols = y.mean(1), y.mean(0)
    ms_c = s * float(((rows - mu) ** 2).sum()) / (n - 1)
    out: dict[str, float | int | None] = {"mean": mu, "clusters": n, "seeds": s}
    if s == 1:
        terms = [(ms_c, n - 1)]
        out.update(var_cluster=None, var_seed=None, var_residual=None, var_total=ms_c, ms_cluster=ms_c, ms_seed=None,
                   ms_residual=None, sd_cluster_mean=math.sqrt(ms_c), seed_means=[float(c) for c in cols])
        variance = ms_c / n
    else:
        ms_s = n * float(((cols - mu) ** 2).sum()) / (s - 1)
        ms_e = float(((y - rows[:, None] - cols[None, :] + mu) ** 2).sum()) / ((n - 1) * (s - 1))
        var_c, var_s, var_e = max(0.0, (ms_c - ms_e) / s), max(0.0, (ms_s - ms_e) / n), ms_e
        # Var(μ̂) from the (truncated) components, written as Σ ± MS / (n S) for the Satterthwaite df.
        terms = [(ms_c if ms_c > ms_e else 0.0, n - 1), (ms_s if ms_s > ms_e else 0.0, s - 1)]
        residual_sign = (ms_c > ms_e) + (ms_s > ms_e) - 1                   # MS_e enters with this coefficient
        terms.append((residual_sign * ms_e, (n - 1) * (s - 1)))
        variance = var_c / n + var_s / s + var_e / (n * s)
        out.update(var_cluster=var_c, var_seed=var_s, var_residual=var_e, var_total=var_c + var_s + var_e, ms_cluster=ms_c,
                   ms_seed=ms_s, ms_residual=ms_e, sd_cluster_mean=math.sqrt(var_c + var_e / s),
                   seed_means=[float(c) for c in cols])
    se = math.sqrt(max(variance, 0.0))
    numerator = sum(t for t, _ in terms) ** 2
    denominator = sum(t * t / d for t, d in terms if d > 0 and t != 0)
    df = float(numerator / denominator) if denominator > 0 else float("inf")
    if se == 0:
        out.update(se=0.0, df=df, ci_low=mu, ci_high=mu, p_value=1.0 if mu == 0 else 0.0)
        return out
    q = float(stats.t.ppf(0.975, df)) if math.isfinite(df) else 1.959963984540054
    t_value = mu / se
    p = float(2 * stats.t.sf(abs(t_value), df)) if math.isfinite(df) else float(2 * stats.norm.sf(abs(t_value)))
    out.update(se=se, df=df, t=t_value, ci_low=mu - q * se, ci_high=mu + q * se, p_value=p)
    return out


def two_way_cluster_bootstrap(values: Sequence[Sequence[float]] | np.ndarray, *, resamples: int = 2000,
                              seed: int = 0) -> dict[str, float | int]:
    """Percentile interval and two-sided p of the mean of a clusters × seeds table, resampling clusters (rows) and seeds
    (columns) independently with replacement (the pigeonhole bootstrap, Owen 2007: conservative, its variance counts
    the residual component twice). One seed: an ordinary cluster bootstrap."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 1 or y.shape[1] < 1 or not np.isfinite(y).all():
        raise ValueError("values must be a finite clusters × seeds table")
    n, s = y.shape
    rng = np.random.default_rng(seed)
    row_weights = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float64)       # resamples × n
    col_weights = rng.multinomial(s, np.full(s, 1.0 / s), size=resamples).astype(np.float64)       # resamples × s
    draws = np.einsum("rn,ns,rs->r", row_weights, y, col_weights) / (n * s)
    p = float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (resamples + 1)))
    if np.all(y == 0):
        p = 1.0
    return {"mean": float(y.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p_value": p, "clusters": n, "seeds": s, "resamples": int(resamples)}


def required_clusters(delta: float, *, var_cluster: float, var_seed: float, var_residual: float, seeds: int,
                      alpha: float = 0.05, power: float = 0.8, tests: int = 1) -> float | None:
    """Clusters (items) needed for a two-sided level-`alpha / tests` test of the mean (the Bonferroni bound of Holm over
    `tests`, normal approximation) to detect `delta` with `power` when `seeds` seeds are pooled:
    `n = (var_cluster + var_residual / S) / ((Δ / (z_{1−α/2m} + z_power))² − var_seed / S)`. None when the seed
    component alone exceeds the budget (more items cannot reach the power; more seeds are needed)."""
    from scipy import stats
    z = float(stats.norm.ppf(1 - alpha / (2 * tests)) + stats.norm.ppf(power))
    budget = (abs(delta) / z) ** 2 - var_seed / seeds
    if budget <= 0:
        return None
    return (var_cluster + var_residual / seeds) / budget


def power_at(n: int, delta: float, *, var_cluster: float, var_seed: float, var_residual: float, seeds: int,
             alpha: float = 0.05, tests: int = 1) -> float:
    """Power of the same test with `n` clusters and `seeds` seeds (normal approximation)."""
    from scipy import stats
    se = math.sqrt(var_cluster / n + var_seed / seeds + var_residual / (n * seeds))
    if se == 0:
        return 1.0 if delta else 0.0
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha / (2 * tests)) - abs(delta) / se))


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be in [0, total]")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def positive_ranks(scores: Tensor, positives: Tensor, *, ties: str = "mid") -> Tensor:
    """Rank of the best-scoring positive per row among positives and negatives.

    `ties="optimistic"` reproduces the historical `1 + #(score > positive)` rule, which gives
    a constant or tied prediction rank 1. `ties="mid"` counts negatives tied with the best
    positive at half weight, the expected rank under random tie-breaking.
    """
    if scores.shape != positives.shape or scores.ndim != 2:
        raise ValueError("scores and positives must be matching 2-D tensors")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("retrieval scores contain NaN or infinite values")
    if not bool(positives.any(1).all()):
        raise ValueError("every query needs at least one positive candidate")
    best = scores.masked_fill(~positives, float("-inf")).max(1).values
    if ties == "optimistic":
        return 1 + (scores > best[:, None]).sum(1).float()
    if ties != "mid":
        raise ValueError("ties must be 'mid' or 'optimistic'")
    negatives = ~positives
    above = ((scores > best[:, None]) & negatives).sum(1).float()
    tied = ((scores == best[:, None]) & negatives).sum(1).float()
    return 1 + above + 0.5 * tied
