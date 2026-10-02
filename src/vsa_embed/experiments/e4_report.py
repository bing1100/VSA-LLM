"""E4 analysis report (WP-E4R): stratified losses, paired statistics, gate items, convergence.

    python -m vsa_embed.experiments.e4_report --runs <dir> [<dir> ...] --output <report dir>
        [--baseline C0] [--references C3 ...] [--candidates C5 C6] [--projection-tokens 2.5e9 7e9]

Every folder under `--runs` holding `metrics.jsonl` is a run. Runs are grouped into cohorts (same
model, token budget, evaluation windows, eval corpus, ontology and `ℓ_min`); inside a cohort a run
is a (condition, seed), parsed from the config's `experiment` (`e4-<stage>-<size>-<condition>-s<seed>`)
or the folder name (`<size>-<condition>-s<seed>`). Runs without `manifest.json` are listed as
incomplete and skipped. Runs with `train.stop_after_steps` are kill-and-resume checks: they are
compared with their uninterrupted twin (condition before any `@`, same seed) instead of entering
the grid.

Statistics (experiments.md §0.8). Final loss per stratum: mean over seeds with a seed-level t
interval. Paired differences (condition − reference; negative = lower loss) at the final
evaluation are token-weighted, with a cluster bootstrap over evaluation windows (each window
carries every seed; needs `eval_windows.npz` from `eval.save_window_losses`), else a seed-level t
interval. Holm correction over the condition grid (every condition × reference pair) within each
stratum; "significant" = Holm-adjusted p < 0.05.

Gate items (experiments.md E4 gate), per candidate (default C5, C6; otherwise every non-control
condition): (1) held-out stratum significantly better than C2, and seen-rare stratum
non-inferior to C2 (relative upper bound ≤ `match_margin`) with fewer channel parameters;
(2) significantly better than C1 on every linked stratum with targets; (3) locality: unlinked
loss not worse than the baseline by more than `locality_margin` (relative upper bound);
(4) at least two of {WiC, rare-word similarity, WSD, domain task} improve over the baseline
(Holm across the candidate's probes). The gate is defined at 125M × 500M tokens with 3 seeds;
any other cohort is labelled exploratory.

Convergence (§0.13, `convergence.py`): data multipliers `k(L)` vs the baseline per stratum with
seed CIs, `L(D) = E + B·D^(−β)` fits, projected gaps (baseline − condition) at
`--projection-tokens`, and the escalation rule on the held-out and rare strata against the
controls C1, C2, C1h, with the size trend taken from the next-smaller cohort.

`probes.json` (optional, per run; written by the probe harness)::

    {"probes": {"<name>": {"family": "wic" | "rare_word" | "wsd" | "domain" | ..., "score": float,
                            "higher_is_better": true, "statistic": "mean" | "spearman",
                            "per_example": [float, ...] | [[prediction, gold], ...]}}}

A bare number is a score; without `family` it is inferred from the name. With per-example values
in both runs, a probe difference gets a paired bootstrap over examples (pooled over seeds), else
a seed-level t interval.
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
import yaml

from ..convergence import (Curve, common_targets, compare, data_multiplier, escalation_verdict, fit_power_law,
                           load_evaluations, project, tokens_to_loss)
from ..provenance import prepare_output_dir, write_run_metadata
from ..statistics import holm_adjust, mean_confidence_interval, paired_ratio_bootstrap

STRATA = ("all", "unlinked", "inside", "after", "after_heldout", "after_rare", "after_mid", "after_frequent",
          "after_len1", "after_len2", "after_len3plus", "after_rare_seen", "after_unseen")
LINKED = STRATA[2:]
KEY_STRATA = ("all", "unlinked", "after", "after_heldout", "after_rare", "after_rare_seen", "inside")
CONTROLS = frozenset({"C0", "C0'", "C1", "C1s", "C1h", "C2", "C2f"})
ESCALATION_CONTROLS = ("C1", "C2", "C1h")
ESCALATION_STRATA = ("after_heldout", "after_rare_seen")   # runs without it fall back to `after_rare`
PROBE_FAMILIES = {"wic": "WiC", "rare_word": "rare-word similarity", "wsd": "WSD", "domain": "domain task"}
NAME = re.compile(r"(?:^|-)(?P<size>[^-]+)-(?P<condition>[^-]+)-s(?P<seed>\d+)$")
ALPHA = 0.05


@dataclass
class Run:
    path: Path
    config: dict[str, Any]
    condition: str
    seed: int
    model: str
    cohort: tuple
    manifest: dict[str, Any] | None
    curves: dict[str, Curve]
    final: dict[str, dict[str, float]]                         # stratum → {"loss", "stratum_tokens"}
    final_tokens: float
    windows: dict[str, tuple[np.ndarray, np.ndarray]] | None   # stratum → (loss sums, counts) per window
    starts: np.ndarray | None
    tokens_per_s: float | None
    probes: dict[str, dict[str, Any]] | None

    @property
    def complete(self) -> bool:
        return self.manifest is not None and bool(self.final)

    @property
    def resume_check(self) -> bool:
        return bool(self.config.get("train", {}).get("stop_after_steps"))

    @property
    def eval_only(self) -> bool:
        """C0' on a frozen host: evaluated once, the same loss at every point (a flat curve)."""
        return bool(self.config.get("train", {}).get("eval_only") or (self.manifest or {}).get("eval_only"))


# ---------------------------------------------------------------- loading


def _probe_family(name: str, entry: dict[str, Any]) -> str:
    if entry.get("family"):
        return str(entry["family"])
    lower = name.lower()
    for family, keys in (("wic", ("wic",)), ("rare_word", ("card", "rare", "rw")),
                         ("wsd", ("wsd", "raganato", "semeval", "senseval")), ("domain", ("domain",))):
        if any(k in lower for k in keys):
            return family
    return "other"


def load_probes(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    raw = raw.get("probes", raw)
    probes = {}
    for name, entry in raw.items():
        entry = {"score": float(entry)} if isinstance(entry, (int, float)) else dict(entry)
        if entry.get("score") is None:
            continue
        entry["family"] = _probe_family(name, entry)
        probes[name] = entry
    return probes


def load_run(path: Path) -> Run:
    from ..training.lm import load_window_losses
    config_path = path / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    match = NAME.search(str(config.get("experiment", ""))) or NAME.search(path.name)
    model_cfg, data_cfg = config.get("model", {}), config.get("data", {})
    model = (f"{model_cfg['pretrained']}/{model_cfg.get('host_mode', 'train')}" if model_cfg.get("pretrained")
             else str(model_cfg.get("size") or (match["size"] if match else "?")))
    budget = int(config.get("train", {}).get("total_tokens", 0))    # configured; the trainer rounds down to whole steps
    cohort = (model, budget, int(config.get("eval", {}).get("windows", 0)), int(model_cfg.get("seq_len", 0)),
              str(data_cfg.get("eval")), str(data_cfg.get("ontology")), int(data_cfg.get("min_subtokens", 0)))
    manifest = json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else None
    evaluations = load_evaluations(path)
    final_tokens = max((t for points in evaluations.values() for t in points), default=0.0)
    final = {s: points[final_tokens] for s, points in evaluations.items() if final_tokens in points} if final_tokens else {}
    curves = {s: sorted((t, v["loss"]) for t, v in points.items() if t > 0 and math.isfinite(v["loss"]))
              for s, points in evaluations.items()}
    windows = starts = None
    if (path / "eval_windows.npz").exists():
        data = load_window_losses(path / "eval_windows.npz")
        if data["evals"] and max(data["evals"]) == final_tokens:     # windows of the final evaluation only
            sums, counts = data["evals"][max(data["evals"])]
            windows = {s: (sums[i], counts[i]) for i, s in enumerate(data["strata"])}
            starts = data["starts"]
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line.strip()]
    speeds = [r["tokens_per_s"] for r in rows if r.get("type") == "train" and r.get("tokens_per_s")]
    condition = match["condition"] if match else path.name
    condition = "C0'" if condition == "C0p" else condition         # C0' is written C0p in CPT file names
    return Run(path=path, config=config, condition=condition,
               seed=int(match["seed"]) if match else int(config.get("seed", 0)), model=model, cohort=cohort,
               manifest=manifest, curves={s: c for s, c in curves.items() if c}, final=final, final_tokens=final_tokens,
               windows=windows, starts=starts, tokens_per_s=float(np.median(speeds)) if speeds else None,
               probes=load_probes(path / "probes.json"))


def discover(roots: Sequence[Path]) -> list[Run]:
    paths = sorted({metrics.parent.resolve() for root in roots for metrics in Path(root).rglob("metrics.jsonl")})
    return [load_run(p) for p in paths]


def condition_order(name: str) -> tuple[int, str]:
    match = re.match(r"C(\d+)(.*)", name)
    return (int(match[1]), match[2]) if match else (999, name)


def _tokens_label(tokens: float) -> str:
    return f"{tokens / 1e9:.3g}B" if tokens >= 1e9 else f"{tokens / 1e6:.3g}M"


def cohort_labels(keys: Sequence[tuple]) -> dict[tuple, str]:
    labels = {}
    for key in keys:
        model, budget, _, _, _, _, min_subtokens = key
        label = f"{model} · {_tokens_label(budget)} tokens" + (f" · ℓmin {min_subtokens}" if min_subtokens not in (0, 2) else "")
        labels[key] = label
    counts: dict[str, int] = {}
    for key, label in list(labels.items()):
        if list(labels.values()).count(label) > 1:
            counts[label] = counts.get(label, 0) + 1
            labels[key] = f"{label} · #{counts[label]}"
    return labels


# ---------------------------------------------------------------- statistics


def _seed_summary(values: dict[int, float]) -> dict[str, Any] | None:
    finite = {s: v for s, v in values.items() if v is not None and math.isfinite(v)}
    if not finite:
        return None
    return {**mean_confidence_interval(list(finite.values())), "seeds": finite}


def _t_p_value(values: Sequence[float]) -> float | None:
    if len(values) < 2 or float(np.std(values)) == 0.0:
        return None
    from scipy.stats import ttest_1samp
    return float(ttest_1samp(values, 0.0).pvalue)


def paired_difference(condition: dict[int, Run], reference: dict[int, Run], stratum: str, *,
                      resamples: int, seed: int) -> dict[str, Any] | None:
    """Condition − reference at the final evaluation, paired by seed and (when saved) by window."""
    seeds = sorted(s for s in set(condition) & set(reference)
                   if stratum in condition[s].final and stratum in reference[s].final
                   and math.isfinite(condition[s].final[stratum]["loss"]) and math.isfinite(reference[s].final[stratum]["loss"]))
    if not seeds:
        return None
    per_seed = {s: condition[s].final[stratum]["loss"] - reference[s].final[stratum]["loss"] for s in seeds}
    reference_mean = float(np.mean([reference[s].final[stratum]["loss"] for s in seeds]))
    result: dict[str, Any] = {"seeds": seeds, "per_seed": per_seed, "seed_ci": mean_confidence_interval(list(per_seed.values())),
                              "single_seed": len(seeds) == 1}
    paired_windows = all(condition[s].windows and reference[s].windows and stratum in condition[s].windows
                         and stratum in reference[s].windows and condition[s].starts is not None
                         and reference[s].starts is not None and np.array_equal(condition[s].starts, reference[s].starts)
                         and np.array_equal(condition[s].windows[stratum][1], reference[s].windows[stratum][1])
                         for s in seeds)
    if paired_windows:
        d = sum(condition[s].windows[stratum][0] - reference[s].windows[stratum][0] for s in seeds)
        n = sum(condition[s].windows[stratum][1].astype(np.float64) for s in seeds)
        b = sum(reference[s].windows[stratum][0] for s in seeds)
        if n.sum() > 0:
            boot = paired_ratio_bootstrap(d, n, b, resamples=resamples, seed=seed)
            result.update(method="window bootstrap", delta=boot["mean"], ci_low=boot["ci_low"], ci_high=boot["ci_high"],
                          p_value=boot["p_value"], relative=boot["relative"], relative_ci_low=boot["relative_ci_low"],
                          relative_ci_high=boot["relative_ci_high"], windows=boot["nonempty_clusters"], tokens=int(n.sum()))
            return result
    if all(condition[s].windows and reference[s].windows for s in seeds):
        result["note"] = "per-window losses do not pair (different evaluation windows or counts)"
    ci = result["seed_ci"]
    scale = 1.0 / reference_mean if reference_mean else float("nan")
    result.update(method="seed t" if len(seeds) > 1 else "single seed, no windows", delta=ci["mean"], ci_low=ci["ci_low"],
                  ci_high=ci["ci_high"], p_value=_t_p_value(list(per_seed.values())), relative=ci["mean"] * scale,
                  relative_ci_low=None if ci["ci_low"] is None else ci["ci_low"] * scale,
                  relative_ci_high=None if ci["ci_high"] is None else ci["ci_high"] * scale)
    return result


def _spearman_rows(prediction: np.ndarray, gold: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    a, b = rankdata(prediction, axis=-1), rankdata(gold, axis=-1)
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


def probe_difference(condition: dict[int, dict[str, Any]], baseline: dict[int, dict[str, Any]], name: str, *,
                     resamples: int, seed: int) -> dict[str, Any] | None:
    """Candidate − baseline probe score (sign-adjusted so that positive = better)."""
    seeds = sorted(s for s in set(condition) & set(baseline) if name in condition[s] and name in baseline[s])
    if not seeds:
        return None
    entries = [(condition[s][name], baseline[s][name]) for s in seeds]
    sign = 1.0 if entries[0][0].get("higher_is_better", True) else -1.0
    statistic = entries[0][0].get("statistic", "mean")
    per_seed = {s: sign * (c["score"] - b["score"]) for s, (c, b) in zip(seeds, entries)}
    result: dict[str, Any] = {"seeds": seeds, "family": entries[0][0]["family"], "per_seed": per_seed,
                              "candidate": float(np.mean([c["score"] for c, _ in entries])),
                              "baseline": float(np.mean([b["score"] for _, b in entries]))}
    examples = [(np.asarray(c.get("per_example", []), dtype=np.float64), np.asarray(b.get("per_example", []), dtype=np.float64))
                for c, b in entries]
    sizes = {x.shape for pair in examples for x in pair}
    if len(sizes) == 1 and examples[0][0].shape and examples[0][0].shape[0] > 1:
        count = examples[0][0].shape[0]
        rng = np.random.default_rng(seed)
        draws = []
        for start in range(0, resamples, 250):
            index = rng.integers(0, count, (min(250, resamples - start), count))
            if statistic == "spearman":
                diff = [_spearman_rows(c[index, 0], c[index, 1]) - _spearman_rows(b[index, 0], b[index, 1]) for c, b in examples]
            else:
                diff = [(c - b)[index].mean(-1) for c, b in examples]
            draws.append(sign * np.mean(diff, axis=0))
        draws = np.concatenate(draws); draws = draws[np.isfinite(draws)]
        if statistic == "spearman":
            scores = [(float(_spearman_rows(c[:, 0], c[:, 1])), float(_spearman_rows(b[:, 0], b[:, 1]))) for c, b in examples]
        else:
            scores = [(float(c.mean()), float(b.mean())) for c, b in examples]
        point = np.mean([c - b for c, b in scores])
        result.update(candidate=float(np.mean([c for c, _ in scores])), baseline=float(np.mean([b for _, b in scores])))
        result.update(method="example bootstrap", delta=float(sign * point), ci_low=float(np.quantile(draws, 0.025)),
                      ci_high=float(np.quantile(draws, 0.975)),
                      p_value=float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (draws.size + 1))))
        return result
    ci = mean_confidence_interval(list(per_seed.values()))
    result.update(method="seed t" if len(seeds) > 1 else "single seed, no examples", delta=ci["mean"], ci_low=ci["ci_low"],
                  ci_high=ci["ci_high"], p_value=_t_p_value(list(per_seed.values())))
    return result


# ---------------------------------------------------------------- analysis


def _improved(comparison: dict[str, Any] | None) -> bool:
    return bool(comparison and comparison.get("significant") and comparison["delta"] < 0)


def _mean_manifest(runs: dict[int, Run], key: str) -> float | None:
    values = [r.manifest.get(key) for r in runs.values() if r.manifest and r.manifest.get(key) is not None]
    return float(np.mean(values)) if values else None


def gate_items(candidate: str, grid: dict[str, dict[int, Run]], paired: dict[str, dict[str, dict[str, Any]]],
               final_loss: dict[str, dict[str, Any]], probes: dict[str, Any] | None, *, baseline: str,
               match_margin: float, locality_margin: float) -> dict[str, Any]:
    def get(reference: str, stratum: str) -> dict[str, Any] | None:
        return paired.get(reference, {}).get(candidate, {}).get(stratum)

    def unavailable(why: str) -> dict[str, Any]:
        return {"verdict": "not available", "detail": why}

    items: dict[str, Any] = {}
    if "C2" not in grid:
        items["1"] = unavailable("C2 not in this cohort")
    else:
        held, rare = get("C2", "after_heldout"), get("C2", "after_rare_seen") or get("C2", "after_rare")
        own, free = _mean_manifest(grid[candidate], "channel_parameters"), _mean_manifest(grid["C2"], "channel_parameters")
        if held is None or rare is None:
            items["1"] = unavailable("held-out or rare stratum missing")
        else:
            better = _improved(held)
            matched = rare.get("relative_ci_high") is not None and rare["relative_ci_high"] <= match_margin
            fewer = own is not None and free is not None and own < free
            items["1"] = {"verdict": "pass" if better and matched and fewer else "fail",
                          "heldout_better": better, "rare_noninferior": matched, "fewer_channel_parameters": fewer,
                          "channel_parameters": {"candidate": own, "C2": free},
                          "detail": f"held-out Δ {_fmt_ci(held, 'delta')}; rare relative upper bound "
                                    f"{_pct(rare.get('relative_ci_high'))} (margin {_pct(match_margin)}); "
                                    f"channel parameters {_count(own)} vs C2 {_count(free)}"}
    if "C1" not in grid:
        items["2"] = unavailable("C1 not in this cohort")
    else:
        strata = [s for s in LINKED if final_loss.get(s, {}).get("tokens") and get("C1", s) is not None]
        failed = [s for s in strata if not _improved(get("C1", s))]
        items["2"] = ({"verdict": "pass" if not failed else "fail", "strata": strata, "not_improved": failed,
                       "detail": "better than C1 on " + (f"all {len(strata)} linked strata" if not failed
                                                         else f"{len(strata) - len(failed)}/{len(strata)}; not on " + ", ".join(failed))}
                      if strata else unavailable("no linked strata"))
    local = get(baseline, "unlinked")
    if local is None or local.get("relative_ci_high") is None:
        items["3"] = unavailable(f"no unlinked comparison with a CI against {baseline}")
    else:
        ok = local["relative_ci_high"] <= locality_margin
        items["3"] = {"verdict": "pass" if ok else "fail", "relative": local["relative"],
                      "relative_ci_high": local["relative_ci_high"],
                      "detail": f"unlinked {_pct(local['relative'])} vs {baseline}, upper bound {_pct(local['relative_ci_high'])} "
                                f"(margin {_pct(locality_margin)})"}
    if not probes:
        items["4"] = unavailable("no probes.json for the candidate and the baseline")
    else:
        families = sorted({p["family"] for p in probes.values()} & set(PROBE_FAMILIES))
        improved = sorted({p["family"] for p in probes.values() if p.get("significant") and p["delta"] > 0} & set(PROBE_FAMILIES))
        items["4"] = {"verdict": "pass" if len(improved) >= 2 else "fail", "families_tested": families,
                      "families_improved": improved,
                      "detail": f"{len(improved)} of {len(families)} probe families improve "
                                f"({', '.join(PROBE_FAMILIES[f] for f in improved) or 'none'})"
                                + (f"; missing: {', '.join(PROBE_FAMILIES[f] for f in PROBE_FAMILIES if f not in families)}"
                                   if len(families) < len(PROBE_FAMILIES) else "")}
    verdicts = [item["verdict"] for item in items.values()]
    overall = "fail" if "fail" in verdicts else "pass" if all(v == "pass" for v in verdicts) else "incomplete"
    return {"items": items, "overall": overall}


def _k_curve(baselines: list[Curve], conditions: list[Curve], count: int = 8) -> list[dict[str, float]]:
    points = []
    for target in common_targets(baselines, conditions, count=count):
        values = [(tokens_to_loss(b, target), data_multiplier(b, c, target)) for b, c in zip(baselines, conditions)]
        values = [(t, k) for t, k in values if t is not None and k is not None]
        if values:
            ks = [k for _, k in values]
            points.append({"target": target, "baseline_tokens": float(np.mean([t for t, _ in values])),
                           "k_mean": float(np.mean(ks)), "k_min": float(min(ks)), "k_max": float(max(ks)), "n": len(ks)})
    return points


def _projected_gap(baselines: list[Curve], conditions: list[Curve], tokens: float) -> dict[str, Any] | None:
    gaps = []
    for b, c in zip(baselines, conditions):
        fb, fc = fit_power_law(b), fit_power_law(c)
        if fb and fc:
            gaps.append(project(fb, tokens) - project(fc, tokens))
    return mean_confidence_interval(gaps) if gaps else None


def _curves(runs: dict[int, Run], seeds: Sequence[int], stratum: str) -> list[Curve]:
    return [runs[s].curves[stratum] for s in seeds]


def _curve_seeds(a: dict[int, Run], b: dict[int, Run], stratum: str) -> list[int]:
    return sorted(s for s in set(a) & set(b) if len(a[s].curves.get(stratum, [])) >= 2 and len(b[s].curves.get(stratum, [])) >= 2)


def convergence(grid: dict[str, dict[int, Run]], strata: Sequence[str], *, baseline: str,
                projection_tokens: Sequence[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per stratum and condition: `convergence.compare` vs the baseline, projected gaps and `k(L)` curve; plus fits.
    An evaluation-only baseline (flat curve) has no `k(L)` or projection, so only fits are returned."""
    result: dict[str, Any] = {}; fits: dict[str, Any] = {}
    flat_baseline = baseline in grid and all(r.eval_only for r in grid[baseline].values())
    for stratum in strata:
        for condition, runs in grid.items():
            per_seed = {s: fit_power_law(r.curves[stratum]) for s, r in runs.items()
                        if len(r.curves.get(stratum, [])) >= 4 and not r.eval_only}
            per_seed = {s: f for s, f in per_seed.items() if f}
            if per_seed:
                fits.setdefault(stratum, {})[condition] = {
                    "per_seed": per_seed, **{k: float(np.mean([f[k] for f in per_seed.values()])) for k in ("E", "B", "beta", "rmse")},
                    "projected_loss": {f"{t:.3g}": float(np.mean([project(f, t) for f in per_seed.values()])) for t in projection_tokens}}
            if condition == baseline or baseline not in grid or flat_baseline:
                continue
            seeds = _curve_seeds(grid[baseline], runs, stratum)
            if not seeds:
                continue
            base, cond = _curves(grid[baseline], seeds, stratum), _curves(runs, seeds, stratum)
            comparison = compare(base, cond, projection_tokens=projection_tokens[0])
            comparison["projected_loss_gaps"] = {f"{t:.3g}": _projected_gap(base, cond, t) for t in projection_tokens}
            comparison["k_curve"] = _k_curve(base, cond)
            comparison["seeds"] = seeds
            result.setdefault(stratum, {})[condition] = comparison
    return result, fits


def escalation(grid: dict[str, dict[int, Run]], versus_baseline: dict[str, Any], candidates: Sequence[str], *,
               projection_tokens: float, smaller: dict[tuple[str, str], float]) -> dict[str, Any]:
    verdicts: dict[str, Any] = {}
    for candidate in candidates:
        strata = [s if s in versus_baseline or s != "after_rare_seen" else "after_rare" for s in ESCALATION_STRATA]
        for stratum in strata:
            comparison = versus_baseline.get(stratum, {}).get(candidate)
            if comparison is None:
                continue
            controls = {}
            for control in ESCALATION_CONTROLS:
                if control in grid and control != candidate:
                    seeds = _curve_seeds(grid[control], grid[candidate], stratum)
                    if seeds:
                        controls[control] = compare(_curves(grid[control], seeds, stratum), _curves(grid[candidate], seeds, stratum),
                                                    projection_tokens=projection_tokens)
            verdict = escalation_verdict(comparison, controls, smaller_model_multiplier=smaller.get((candidate, stratum)))
            verdict["smaller_model_multiplier"] = smaller.get((candidate, stratum))
            verdicts.setdefault(candidate, {})[stratum] = verdict
    return verdicts


def resume_checks(checks: Sequence[Run], grid: dict[str, dict[int, Run]]) -> list[dict[str, Any]]:
    results = []
    for run in checks:
        twin = grid.get(run.condition.split("@")[0], {}).get(run.seed)
        if twin is None:
            results.append({"run": str(run.path), "twin": None, "verdict": "no uninterrupted twin"}); continue
        final = [abs(run.final[s]["loss"] - twin.final[s]["loss"]) for s in set(run.final) & set(twin.final)
                 if math.isfinite(run.final[s]["loss"]) and math.isfinite(twin.final[s]["loss"])]
        curve = [abs(a - b) for s in set(run.curves) & set(twin.curves)
                 for (ta, a), (tb, b) in zip(run.curves[s], twin.curves[s]) if ta == tb]
        window = ([float(np.abs(run.windows[s][0] - twin.windows[s][0]).max()) for s in set(run.windows) & set(twin.windows)]
                  if run.windows and twin.windows else [])
        worst = max(final + curve + window, default=float("nan"))
        results.append({"run": str(run.path), "twin": str(twin.path), "max_final_loss_diff": max(final, default=None),
                        "max_curve_diff": max(curve, default=None), "max_window_sum_diff": max(window, default=None),
                        "verdict": "identical" if worst == 0 else f"max |Δ| {worst:.3g}"})
    return results


def analyze_cohort(label: str, runs: Sequence[Run], *, baseline: str, references: Sequence[str], candidates: Sequence[str],
                   projection_tokens: Sequence[float], resamples: int, seed: int, match_margin: float,
                   locality_margin: float) -> tuple[dict[str, Any], dict[str, dict[int, Run]]]:
    warnings: list[str] = []
    grid: dict[str, dict[int, Run]] = {}
    checks = []
    for run in sorted(runs, key=lambda r: str(r.path)):
        if not run.complete:
            continue
        if run.resume_check:
            checks.append(run); continue
        if run.seed in grid.setdefault(run.condition, {}):
            warnings.append(f"duplicate {run.condition} seed {run.seed}: kept {grid[run.condition][run.seed].path}, skipped {run.path}")
            continue
        grid[run.condition][run.seed] = run
    grid = {c: grid[c] for c in sorted(grid, key=condition_order) if grid[c]}
    conditions = list(grid)
    if baseline not in grid and f"{baseline}'" in grid:      # continued pretraining names its baseline C0'
        baseline = f"{baseline}'"
    present = {s for runs_ in grid.values() for r in runs_.values() for s in r.final}
    strata = [s for s in STRATA if s in present] + sorted(present - set(STRATA))
    final_loss: dict[str, dict[str, Any]] = {}
    for stratum in strata:
        counts = [r.final[stratum].get("stratum_tokens") for runs_ in grid.values() for r in runs_.values() if stratum in r.final]
        final_loss[stratum] = {"tokens": int(counts[0]) if counts and counts[0] is not None else None,
                               "conditions": {c: _seed_summary({s: r.final[stratum]["loss"] for s, r in grid[c].items()
                                                                if stratum in r.final}) for c in conditions}}
    refs = [r for r in dict.fromkeys([baseline, "C1", "C2", *references]) if r in grid]
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for i, ref in enumerate(refs):
        for condition in conditions:
            if condition == ref or condition in refs[:i]:
                continue
            for stratum in strata:
                comparison = paired_difference(grid[condition], grid[ref], stratum, resamples=resamples, seed=seed)
                if comparison is not None:
                    paired.setdefault(ref, {}).setdefault(condition, {})[stratum] = comparison
    for stratum in strata:   # Holm over the condition grid within each stratum
        family = [c[stratum] for by_ref in paired.values() for c in by_ref.values()
                  if stratum in c and c[stratum].get("p_value") is not None]
        for comparison, adjusted in zip(family, holm_adjust([c["p_value"] for c in family])):
            comparison.update(holm_p=adjusted, holm_family=len(family), significant=adjusted < ALPHA)
    largest = max((c[s]["holm_family"] for by_ref in paired.values() for c in by_ref.values() for s in c
                   if c[s].get("method") == "window bootstrap"), default=0)
    if largest and largest * 2 / (resamples + 1) >= ALPHA:
        warnings.append(f"{resamples} bootstrap resamples cannot reach Holm-adjusted p < {ALPHA} over {largest} comparisons "
                        f"(p floor 2/(B+1)); use --resamples ≥ {math.ceil(2 * largest / ALPHA)}")
    seeds_per_condition = {c: sorted(r) for c, r in grid.items()}
    min_seeds = min((len(s) for s in seeds_per_condition.values()), default=0)
    gate_candidates = [c for c in candidates if c in grid] or [c for c in conditions if c not in CONTROLS and c != baseline]
    probe_results: dict[str, Any] = {}
    for candidate in gate_candidates:
        if baseline not in grid:
            continue
        cand = {s: r.probes for s, r in grid[candidate].items() if r.probes}
        base = {s: r.probes for s, r in grid[baseline].items() if r.probes}
        names = sorted({n for p in cand.values() for n in p} & {n for p in base.values() for n in p})
        results = {n: probe_difference(cand, base, n, resamples=resamples, seed=seed) for n in names}
        results = {n: r for n, r in results.items() if r is not None}
        tested = [r for r in results.values() if r.get("p_value") is not None]
        for comparison, adjusted in zip(tested, holm_adjust([r["p_value"] for r in tested])):
            comparison.update(holm_p=adjusted, significant=adjusted < ALPHA)
        probe_results[candidate] = results
    model, budget = runs[0].cohort[0], runs[0].cohort[1]
    exploratory = []
    if model != "125M" or abs(budget - 500_000_000) > 0.01 * 500_000_000:
        exploratory.append("the pre-registered gate is defined at 125M × 500M tokens")
    if 1 < min_seeds < 3:
        exploratory.append(f"{min_seeds} seeds (the gate needs 3)")
    gate = {c: gate_items(c, grid, paired, final_loss, probe_results.get(c), baseline=baseline,
                          match_margin=match_margin, locality_margin=locality_margin) for c in gate_candidates}
    converge, fits = convergence(grid, strata, baseline=baseline, projection_tokens=projection_tokens)
    if baseline in grid and all(r.eval_only for r in grid[baseline].values()):
        warnings.append(f"{baseline} is evaluation-only (frozen host, flat curve): k(L), projections and the escalation "
                        f"rule against it are undefined; use the paired differences")
    throughput = {}
    base_speed = None
    if baseline in grid:
        speeds = [r.tokens_per_s for r in grid[baseline].values() if r.tokens_per_s]
        base_speed = float(np.mean(speeds)) if speeds else None
    for condition, runs_ in grid.items():
        speeds = [r.tokens_per_s for r in runs_.values() if r.tokens_per_s]
        speed = float(np.mean(speeds)) if speeds else None
        channel = _mean_manifest(runs_, "channel_parameters")
        throughput[condition] = {"tokens_per_s": speed, "overhead": base_speed / speed - 1 if speed and base_speed else None,
                                 "parameters": _mean_manifest(runs_, "parameters"), "channel_parameters": channel,
                                 "channel_bytes_fp32": None if channel is None else 4 * channel,
                                 "channel_bytes_fp16": None if channel is None else 2 * channel}
    summary = {"label": label, "baseline": baseline, "model": model, "budget_tokens": budget, "eval_windows": runs[0].cohort[2],
               "min_subtokens": runs[0].cohort[6], "conditions": conditions, "seeds": seeds_per_condition,
               "single_seed": min_seeds == 1, "exploratory": exploratory, "references": refs, "strata": strata,
               "final_loss": final_loss, "paired": paired, "gate": gate, "probes": probe_results,
               "convergence": converge, "fits": fits, "throughput": throughput,
               "resume_checks": resume_checks(checks, grid), "warnings": warnings,
               "escalation_candidates": [c for c in conditions if c not in CONTROLS and c != baseline]}
    return summary, grid


def analyze(runs: Sequence[Run], *, baseline: str = "C0", references: Sequence[str] = (), candidates: Sequence[str] = ("C5", "C6"),
            projection_tokens: Sequence[float] = (2.5e9, 7e9), resamples: int = 10_000, seed: int = 0,
            match_margin: float = 0.005, locality_margin: float = 0.005) -> tuple[dict[str, Any], dict[str, dict[str, dict[int, Run]]]]:
    by_key: dict[tuple, list[Run]] = {}
    for run in runs:
        by_key.setdefault(run.cohort, []).append(run)
    labels = cohort_labels(list(by_key))
    cohorts, grids = {}, {}
    for key, members in by_key.items():
        if not any(r.complete for r in members):
            continue
        summary, grid = analyze_cohort(labels[key], members, baseline=baseline, references=references, candidates=candidates,
                                       projection_tokens=projection_tokens, resamples=resamples, seed=seed,
                                       match_margin=match_margin, locality_margin=locality_margin)
        cohorts[labels[key]], grids[labels[key]] = summary, grid
    # Escalation rule, smallest model first so the size trend can use the next-smaller cohort's multiplier
    # (from-scratch and pretrained-host cohorts form separate size chains).
    sizes = {label: (s["throughput"].get(s["baseline"], {}).get("parameters") or 0.0) for label, s in cohorts.items()}
    for pretrained in (False, True):
        previous: dict[tuple[str, str], float] = {}
        last_size = None
        for label in sorted((l for l, s in cohorts.items() if ("/" in s["model"]) == pretrained), key=lambda l: sizes[l]):
            summary = cohorts[label]
            smaller = previous if last_size is not None and sizes[label] > last_size else {}
            summary["escalation"] = escalation(grids[label], summary["convergence"], summary["escalation_candidates"],
                                               projection_tokens=projection_tokens[0], smaller=smaller)
            current = {(c, s): v["multiplier_mean"] for c, by in summary["escalation"].items() for s, v in by.items()
                       if v["multiplier_mean"] is not None}
            if current:
                previous, last_size = current, sizes[label]
    run_rows = [{"path": str(r.path), "condition": r.condition, "seed": r.seed, "model": r.model, "cohort": labels[r.cohort],
                 "complete": r.complete, "resume_check": r.resume_check, "final_tokens": r.final_tokens,
                 "eval_windows": r.windows is not None, "probes": r.probes is not None, "tokens_per_s": r.tokens_per_s,
                 "parameters": (r.manifest or {}).get("parameters"), "channel_parameters": (r.manifest or {}).get("channel_parameters")}
                for r in runs]
    return {"runs": run_rows, "baseline": baseline, "projection_tokens": list(projection_tokens), "resamples": resamples,
            "match_margin": match_margin, "locality_margin": locality_margin, "cohorts": cohorts}, grids


# ---------------------------------------------------------------- rendering


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value * 100:+.{digits}f}%"


def _count(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_ci(v: dict[str, Any] | None, key: str = "mean", digits: int = 4) -> str:
    if not v or v.get(key) is None:
        return "n/a"
    if v.get("ci_low") is None:
        return f"{v[key]:+.{digits}f}"
    return f"{v[key]:+.{digits}f} [{v['ci_low']:+.{digits}f}, {v['ci_high']:+.{digits}f}]"


def _loss_cell(v: dict[str, Any] | None) -> str:
    if not v:
        return "–"
    if v.get("ci_low") is None:
        return f"{v['mean']:.4f}" + (" (1 seed)" if v["n"] == 1 else "")
    return f"{v['mean']:.4f} ± {(v['ci_high'] - v['ci_low']) / 2:.4f}"


def _relative_cell(v: dict[str, Any] | None) -> str:
    if not v:
        return "–"
    text = _pct(v.get("relative"))
    if v.get("relative_ci_low") is not None:
        text += f" [{v['relative_ci_low'] * 100:+.2f}, {v['relative_ci_high'] * 100:+.2f}]"
    return text + ("*" if v.get("significant") else "")


def _tokens_cell(t: float | None) -> str:
    return "n/a" if t is None else _tokens_label(t)


def render(summary: dict[str, Any], *, figures: dict[str, dict[str, str]], title: str) -> str:
    lines = [f"# {title}", ""]
    runs = summary["runs"]
    complete = [r for r in runs if r["complete"]]
    lines += [f"{len(runs)} runs found, {len(complete)} complete. Baseline **{summary['baseline']}** (a cohort without it "
              f"uses {summary['baseline']}'). Paired differences are "
              "condition − reference at the final evaluation (negative = lower loss), token-weighted; CIs are 95% cluster "
              f"bootstraps over evaluation windows ({summary['resamples']} resamples; each window carries every seed) when "
              "`eval_windows.npz` exists, else seed-level t intervals. `*` = significant after Holm correction over the "
              "condition grid within the stratum. Final-loss cells are mean ± half-width of the seed-level 95% t interval.", ""]
    incomplete = [r for r in runs if not r["complete"]]
    if incomplete:
        lines += ["Incomplete runs (skipped): " + ", ".join(f"`{Path(r['path']).name}`" for r in incomplete) + ".", ""]
    lines += ["## Runs", "", "| Cohort | Condition | Seed | Status | Final tokens | Windows saved | Probes | Parameters | Channel | tokens/s |",
              "|---|---|---:|---|---:|---|---|---:|---:|---:|"]
    for r in sorted(runs, key=lambda r: (r["cohort"], condition_order(r["condition"]), r["seed"])):
        status = "resume check" if r["resume_check"] else "complete" if r["complete"] else "incomplete"
        lines.append(f"| {r['cohort']} | {r['condition']} | {r['seed']} | {status} | {_tokens_cell(r['final_tokens'] or None)} | "
                     f"{'yes' if r['eval_windows'] else 'no'} | {'yes' if r['probes'] else 'no'} | {_count(r['parameters'])} | "
                     f"{_count(r['channel_parameters'])} | {_count(r['tokens_per_s'])} |")
    for label, c in summary["cohorts"].items():
        lines += ["", f"## {label}", ""]
        flags = list(c["exploratory"])
        if c["single_seed"]:
            flags.insert(0, "**single seed**: CIs are over evaluation windows only and say nothing about seed variance "
                            "(the gate needs 3 seeds)")
        if flags:
            lines += ["> Exploratory: " + "; ".join(flags) + ". Gate verdicts below are computed mechanically, not as a gate.", ""]
        lines += [f"Conditions and seeds: " + "; ".join(f"{k} {v}" for k, v in c["seeds"].items()) + ".", ""]
        lines += [f"- warning: {warning}" for warning in c["warnings"]] + ([""] if c["warnings"] else [])
        conditions = c["conditions"]
        lines += ["### Final loss per stratum", "", "| Stratum | Targets | " + " | ".join(conditions) + " |",
                  "|---|---:|" + "---:|" * len(conditions)]
        for stratum in c["strata"]:
            row = c["final_loss"][stratum]
            lines.append(f"| {stratum} | {_count(row['tokens'])} | " + " | ".join(_loss_cell(row["conditions"].get(k)) for k in conditions) + " |")
        for ref, by_condition in c["paired"].items():
            others = [k for k in conditions if k in by_condition]
            lines += ["", f"### Relative difference vs {ref} (condition − {ref}) / {ref}", "",
                      "| Stratum | " + " | ".join(others) + " |", "|---|" + "---:|" * len(others)]
            for stratum in c["strata"]:
                lines.append(f"| {stratum} | " + " | ".join(_relative_cell(by_condition[k].get(stratum)) for k in others) + " |")
            methods = sorted({v["method"] for by in by_condition.values() for v in by.values()})
            lines += ["", f"Method: {', '.join(methods)}."]
        lines += ["", "### E4 gate items", ""]
        if c["gate"]:
            lines += ["| Candidate | 1 held-out vs C2 | 2 linked vs C1 | 3 locality | 4 probes | Overall |", "|---|---|---|---|---|---|"]
            for candidate, g in c["gate"].items():
                lines.append(f"| {candidate} | " + " | ".join(g["items"][i]["verdict"] for i in "1234") + f" | **{g['overall']}** |")
            lines.append("")
            for candidate, g in c["gate"].items():
                for i in "1234":
                    lines.append(f"- {candidate} item {i}: {g['items'][i]['detail']}")
        else:
            lines.append("No candidate condition in this cohort.")
        for candidate, probes in c["probes"].items():
            if probes:
                lines += ["", f"Probes, {candidate} − {c['baseline']} (sign-adjusted, positive = better):", "",
                          "| Probe | Family | Candidate | Baseline | Δ [95% CI] | Method | Holm p |", "|---|---|---:|---:|---|---|---:|"]
                for name, p in probes.items():
                    holm = "n/a" if p.get("holm_p") is None else f"{p['holm_p']:.3g}" + ("*" if p.get("significant") else "")
                    lines.append(f"| {name} | {p['family']} | {p['candidate']:.4f} | {p['baseline']:.4f} | {_fmt_ci(p, 'delta')} | "
                                 f"{p['method']} | {holm} |")
        lines += ["", "### Convergence", "", f"Data multiplier `k = D_{c['baseline']}(L) / D_cond(L)` at the lowest common loss "
                  "target (the one the escalation rule uses), with the seed-level CI; gaps are baseline − condition (positive = "
                  "condition better), projected with per-seed power-law fits.", "",
                  "| Stratum | Condition | Target L | k [95% CI] | Final gap | " + " | ".join(
                      f"Projected gap @ {_tokens_label(t)}" for t in summary["projection_tokens"]) + " |",
                  "|---|---|---:|---|---|" + "---|" * len(summary["projection_tokens"])]
        for stratum, by_condition in c["convergence"].items():
            if stratum not in KEY_STRATA:
                continue
            for condition, comp in by_condition.items():
                items = [(float(k), v) for k, v in comp["data_multiplier"].items() if v is not None]
                target, k = min(items) if items else (None, None)
                gaps = " | ".join(_fmt_ci(comp["projected_loss_gaps"].get(f"{t:.3g}")) for t in summary["projection_tokens"])
                lines.append(f"| {stratum} | {condition} | {'n/a' if target is None else f'{target:.4f}'} | {_fmt_ci(k, digits=3)} | "
                             f"{_fmt_ci(comp['final_loss_gap'])} | {gaps} |")
        if c["fits"]:
            lines += ["", "Power-law fits `L(D) = E + B·(D/D₀)^(−β)` (mean over seeds):", "",
                      "| Stratum | Condition | E | B | β | RMSE |", "|---|---|---:|---:|---:|---:|"]
            for stratum in KEY_STRATA:
                for condition, f in c["fits"].get(stratum, {}).items():
                    lines.append(f"| {stratum} | {condition} | {f['E']:.4f} | {f['B']:.4f} | {f['beta']:.3f} | {f['rmse']:.4f} |")
        lines += ["", "Escalation rule (§0.13; rule ii against C1/C2/C1h, rule iii projection or size trend):", ""]
        if c.get("escalation"):
            lines += ["| Candidate | Stratum | k mean | k CI low | (i) k_low > 1.1 | (ii) controls | (iii) projection / size trend | Escalate |",
                      "|---|---|---:|---:|---|---|---|---|"]
            for candidate, by_stratum in c["escalation"].items():
                for stratum, v in by_stratum.items():
                    controls = ", ".join(f"{k} {'beaten' if ok else 'not beaten'}" for k, ok in v["controls_beaten"].items()) or "none"
                    lines.append(f"| {candidate} | {stratum} | {_number(v['multiplier_mean'])} | {_number(v['multiplier_ci_low'])} | "
                                 f"{v['rule_i']} | {controls} | {v['projection_ok']} / {v['size_trend_ok']} | **{v['escalate']}** |")
        else:
            lines.append("No candidate with held-out or rare-stratum curves against the baseline.")
        lines += ["", "### Throughput and parameters", "",
                  f"| Condition | tokens/s | Overhead vs {c['baseline']} | Parameters | Channel parameters | Channel bytes fp32 / fp16 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for condition, t in c["throughput"].items():
            speed = "n/a" if t["tokens_per_s"] is None else f"{t['tokens_per_s']:,.0f}"
            channel_bytes = "n/a" if t["channel_bytes_fp32"] is None else (
                f"{t['channel_bytes_fp32'] / 2**20:.1f} / {t['channel_bytes_fp16'] / 2**20:.1f} MiB")
            lines.append(f"| {condition} | {speed} | {_pct(t['overhead'], 1)} | {_count(t['parameters'])} | "
                         f"{_count(t['channel_parameters'])} | {channel_bytes} |")
        lines += ["", "Throughput is the median logged training rate per run (evaluation intervals included in a few log windows); "
                  "the GPU is shared, so treat overheads as indicative."]
        if c["resume_checks"]:
            lines += ["", "### Kill-and-resume check", ""]
            for check in c["resume_checks"]:
                lines.append(f"- `{Path(check['run']).name}` vs `{Path(check['twin']).name if check['twin'] else '—'}`: {check['verdict']}")
        if figures.get(label):
            lines += ["", "### Figures", ""]
            for caption, path in figures[label].items():
                lines += [f"![{caption}]({path})", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures

_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_BASELINE_INK, _MUTED, _GRID = "#52514e", "#8a8984", "#e6e5e0"


def condition_style(name: str) -> dict[str, Any]:
    """Colour follows the condition family (C1…C8 → fixed slots; baselines neutral); variants are dashed."""
    base = name.split("@")[0]
    match = re.match(r"C(\d+)", base)
    if base in ("C0", "C0'") or (match and match[1] == "0"):
        colour = _BASELINE_INK
    elif match and 1 <= int(match[1]) <= 8:
        colour = _PALETTE[int(match[1]) - 1]
    else:
        colour = _MUTED
    variant = base != (f"C{match[1]}" if match else base) or "@" in name or base == "C0'"
    return {"color": colour, "linestyle": "--" if variant else "-"}


def _token_tick(value: float, _: Any) -> str:
    if value <= 0:
        return ""
    mantissa = value / 10 ** math.floor(math.log10(value))
    return _tokens_label(value) if abs(mantissa - round(mantissa)) < 1e-6 and round(mantissa) in (1, 2, 5) else ""


def _axes_style(ax, *, token_axis: bool = True) -> None:
    if token_axis:
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(_token_tick)); ax.xaxis.set_minor_formatter(FuncFormatter(_token_tick))
    ax.grid(True, color=_GRID, linewidth=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors="#52514e", labelsize=8)


def plot_cohort(label: str, cohort: dict[str, Any], grid: dict[str, dict[int, Run]], out: Path, slug: str) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = {}
    strata = [s for s in KEY_STRATA if any(s in r.curves for runs in grid.values() for r in runs.values())]
    if strata:
        cols = 3; rows = math.ceil(len(strata) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
        for ax, stratum in zip(axes.flat, strata):
            for condition, runs in grid.items():
                curves = [r.curves[stratum] for r in runs.values() if stratum in r.curves]
                if not curves:
                    continue
                tokens = sorted(set.intersection(*(set(t for t, _ in c) for c in curves)))
                values = np.array([[dict(c)[t] for t in tokens] for c in curves])
                style = condition_style(condition)
                ax.plot(tokens, values.mean(0), label=condition, linewidth=1.6, marker="o", markersize=3, **style)
                if len(curves) > 1:
                    ax.fill_between(tokens, values.min(0), values.max(0), color=style["color"], alpha=0.15, linewidth=0)
            ax.set_xscale("log"); ax.set_title(stratum, fontsize=10, color="#0b0b0b", loc="left")
            ax.set_xlabel("training tokens", fontsize=8, color="#52514e"); ax.set_ylabel("loss (nats/token)", fontsize=8, color="#52514e")
            _axes_style(ax)
        for ax in list(axes.flat)[len(strata):]:
            ax.set_visible(False)
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=min(len(names), 8), frameon=False, fontsize=9)
        fig.suptitle(f"Loss vs tokens — {label} (mean over seeds; band = seed range)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        name = f"figures/loss-{slug}.png"; fig.savefig(out / name, dpi=150); plt.close(fig)
        figures["Loss vs tokens per stratum"] = name
    k_strata = [s for s in KEY_STRATA if cohort["convergence"].get(s)]
    if k_strata:
        cols = 3; rows = math.ceil(len(k_strata) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
        for ax, stratum in zip(axes.flat, k_strata):
            for condition, comp in cohort["convergence"][stratum].items():
                points = comp["k_curve"]
                if not points:
                    continue
                x = [p["baseline_tokens"] for p in points]; y = [p["k_mean"] for p in points]
                err = [[p["k_mean"] - p["k_min"] for p in points], [p["k_max"] - p["k_mean"] for p in points]]
                style = condition_style(condition)
                ax.errorbar(x, y, yerr=err, label=condition, linewidth=1.6, marker="o", markersize=3, capsize=2, **style)
            ax.axhline(1.0, color=_MUTED, linewidth=0.8); ax.axhline(1.1, color=_MUTED, linewidth=0.8, linestyle=":")
            ax.set_xscale("log"); ax.set_title(stratum, fontsize=10, color="#0b0b0b", loc="left")
            ax.set_xlabel(f"tokens {cohort.get('baseline', 'C0')} needs to reach L", fontsize=8, color="#52514e")
            ax.set_ylabel("data multiplier k", fontsize=8, color="#52514e")
            _axes_style(ax)
        for ax in list(axes.flat)[len(k_strata):]:
            ax.set_visible(False)
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=min(max(1, len(names)), 8), frameon=False, fontsize=9)
        fig.suptitle(f"Data multiplier k(L) vs baseline — {label} (bars = seed range; dotted = 1.1)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        name = f"figures/k-{slug}.png"; fig.savefig(out / name, dpi=150); plt.close(fig)
        figures["Data multiplier k(L)"] = name
    return figures


# ---------------------------------------------------------------- CLI


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def write_report(runs_dirs: Sequence[Path], output: Path, *, baseline: str = "C0", references: Sequence[str] = (),
                 candidates: Sequence[str] = ("C5", "C6"), projection_tokens: Sequence[float] = (2.5e9, 7e9),
                 resamples: int = 10_000, seed: int = 0, match_margin: float = 0.005, locality_margin: float = 0.005,
                 title: str = "E4 small-LM report", figures: bool = True) -> dict[str, Any]:
    config = {"runs": [str(p) for p in runs_dirs], "baseline": baseline, "references": list(references),
              "candidates": list(candidates), "projection_tokens": list(projection_tokens), "resamples": resamples,
              "seed": seed, "match_margin": match_margin, "locality_margin": locality_margin, "title": title}
    git_at_start = prepare_output_dir(output)
    runs = discover(runs_dirs)
    if not runs:
        raise FileNotFoundError(f"no run folders (metrics.jsonl) under {', '.join(map(str, runs_dirs))}")
    summary, grids = analyze(runs, baseline=baseline, references=references, candidates=candidates,
                             projection_tokens=projection_tokens, resamples=resamples, seed=seed,
                             match_margin=match_margin, locality_margin=locality_margin)
    plots: dict[str, dict[str, str]] = {}
    if figures:
        (output / "figures").mkdir(exist_ok=True)
        for i, (label, cohort) in enumerate(summary["cohorts"].items()):
            slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower() or f"cohort-{i}"
            plots[label] = plot_cohort(label, cohort, grids[label], output, slug)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    (output / "report.md").write_text(render(summary, figures=plots, title=title))
    write_run_metadata(output, config, git_at_start=git_at_start, runs=len(runs),
                       complete_runs=sum(r.complete for r in runs), cohorts=list(summary["cohorts"]))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="C0")
    parser.add_argument("--references", nargs="*", default=[], help="extra reference conditions besides the baseline, C1 and C2")
    parser.add_argument("--candidates", nargs="*", default=["C5", "C6"])
    parser.add_argument("--projection-tokens", type=float, nargs="+", default=[2.5e9, 7e9])
    parser.add_argument("--resamples", type=int, default=10_000); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match-margin", type=float, default=0.005); parser.add_argument("--locality-margin", type=float, default=0.005)
    parser.add_argument("--title", default="E4 small-LM report"); parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    summary = write_report(args.runs, args.output, baseline=args.baseline, references=args.references, candidates=args.candidates,
                           projection_tokens=args.projection_tokens, resamples=args.resamples, seed=args.seed,
                           match_margin=args.match_margin, locality_margin=args.locality_margin, title=args.title,
                           figures=not args.no_figures)
    print(json.dumps({label: {c: g["overall"] for c, g in cohort["gate"].items()} for label, cohort in summary["cohorts"].items()}))


if __name__ == "__main__":
    main()
