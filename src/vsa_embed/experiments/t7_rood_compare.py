"""T7-ROOD against T7 v1: the pre-registered paired comparison (`experiments/t7-new-vocabulary/preregistration-rood.md`
§4, P2 and P3; decision 63, holdout H1).

Both tracks evaluate on the same `eval-pubmed` corpus with the same 4,096 windows and the same held-out entries
(`t7_rood`: identical tokens, spans and holdout), so their final evaluations pair **by window**. Per track, the window
sums of a stratum are pooled over the common seeds (P0, one evaluation-only run, stands for every seed; `e9_report`'s
rule), and the relative difference of two models is `Σ (candidate − reference) / Σ reference`, as `e9_report`'s
dimension 1 reports it.

- **P2** (the ordering): Δ = (C5 − C0′)/C0′ on T7-ROOD − (C5 − C0′)/C0′ on T7 v1, on `after_heldout`. Predicted ≤ 0
  (the gain on T7-ROOD at least T7 v1's). One 95% cluster bootstrap over windows (10,000 resamples; the same resampled
  windows for both tracks): *confirmed* if the CI's upper end is < 0, *contradicted* if its lower end is > 0, otherwise
  the point estimate is read against 0.
- **P3** (the mechanism): Δ = (C0′ − P0)/P0 on T7-ROOD − the same on T7 v1, on `after_heldout`. Predicted > 0 (C0′
  improves less on T7-ROOD, where it never reads the held-out names).
- Context: each track's own relative differences (C5 − C0′, C0′ − P0) with their bootstrap CIs, on the same resamples.

The comparison refuses runs whose windows do not pair (other starts or other target counts) and stages without a common
seed of C5 and C0′. Output (run-folder contract): `summary.json`, `report.md`, `resolved_config.yaml`, `manifest.json`.

    python -m vsa_embed.experiments.t7_rood_compare --rood experiments/e9-retrofit/runs/t7rood \\
        --reference experiments/e9-retrofit/runs/t7 --output experiments/e9-retrofit/report/t7rood-vs-t7 \\
        [--host SmolLM2-360M] [--strata after_heldout ...] [--resamples 10000] [--seed 0] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..statistics import _cluster_weights
from .e5_common import finish_output, json_ready, start_output, write_json

STRATUM = "after_heldout"
RESULT_FILES = ("summary.json", "report.md", "resolved_config.yaml", "manifest.json")
# (candidate, reference, predicted sign of Δ = ROOD − v1): P2 and P3 of the pre-registration.
PREDICTIONS = {"P2": ("C5", "C0'", -1), "P3": ("C0'", "P0", 1)}


def stage_group(runs_root: Path, host: str) -> Any:
    """The `e9_report` group (host × training mode) of a stage folder whose host name ends with `host`."""
    from .e4_report import discover
    from .e9_report import build_groups
    groups = [g for g in build_groups(discover([Path(runs_root)])) if g.host.split("/")[-1] == host]
    if len(groups) != 1:
        raise ValueError(f"{runs_root}: {len(groups)} finished run groups for host {host} (need exactly one)")
    return groups[0]


def pooled(group: Any, model: str, seeds: Sequence[int], stratum: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(loss sums, target counts, window starts) of `model` on `stratum` at the final evaluation, summed over `seeds`."""
    runs = group.paired(model, seeds)
    if len(runs) != len(seeds):
        raise ValueError(f"{model}: runs for seeds {sorted(runs)} of {list(seeds)}")
    sums = counts = starts = None
    for seed, run in sorted(runs.items()):
        if not run.windows or stratum not in run.windows or run.starts is None:
            raise ValueError(f"{run.path}: no per-window losses of {stratum} at the final evaluation")
        s, n = run.windows[stratum]
        if starts is not None and not np.array_equal(starts, run.starts):
            raise ValueError(f"{run.path}: other evaluation windows than seed {min(runs)}")
        starts = run.starts
        sums = s.astype(np.float64) if sums is None else sums + s
        counts = n.astype(np.float64) if counts is None else counts + n
    return sums, counts, starts


def _interval(draws: np.ndarray, point: float) -> dict[str, float]:
    draws = draws[np.isfinite(draws)]
    return {"mean": float(point), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p_value": float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (draws.size + 1))),
            "resamples": int(draws.size)}


def relative_difference_of_differences(rood: tuple[np.ndarray, np.ndarray], reference: tuple[np.ndarray, np.ndarray], *,
                                       resamples: int, seed: int) -> dict[str, Any]:
    """Each pair is (candidate window sums, reference-model window sums) of one track on the same windows. Returns each
    track's relative difference Σ (c − r) / Σ r and their difference (T7-ROOD − T7 v1), with 95% cluster bootstraps over
    windows drawn once (the same resampled windows for both tracks)."""
    (c1, r1), (c2, r2) = rood, reference
    if not (c1.shape == r1.shape == c2.shape == r2.shape) or c1.ndim != 1:
        raise ValueError("the window sums of both tracks must be equal-length vectors")
    weights = _cluster_weights(c1.size, resamples, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (weights @ (c1 - r1)) / (weights @ r1)
        b = (weights @ (c2 - r2)) / (weights @ r2)
    point_a, point_b = float((c1 - r1).sum() / r1.sum()), float((c2 - r2).sum() / r2.sum())
    return {"rood": _interval(a, point_a), "reference": _interval(b, point_b), "difference": _interval(a - b, point_a - point_b),
            "windows": int(c1.size)}


def reading(result: dict[str, Any], sign: int) -> str:
    """The pre-registered reading of Δ against its predicted sign (−1: Δ ≤ 0 predicted; +1: Δ > 0 predicted)."""
    low, high, mean = result["ci_low"], result["ci_high"], result["mean"]
    if sign < 0:
        return "confirmed" if high < 0 else "contradicted" if low > 0 else ("holds (point estimate ≤ 0)" if mean <= 0 else
                                                                            "does not hold (point estimate > 0)")
    return "confirmed" if low > 0 else "contradicted" if high < 0 else ("holds (point estimate > 0)" if mean > 0 else
                                                                        "does not hold (point estimate ≤ 0)")


def compare(rood_root: Path, reference_root: Path, *, host: str = "SmolLM2-360M", strata: Sequence[str] = (STRATUM,),
            resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """P2 and P3 (and each track's own differences) per stratum; see the module docstring."""
    groups = {"rood": stage_group(rood_root, host), "reference": stage_group(reference_root, host)}
    seeds = sorted(set.intersection(*(set(g.seeds(m)) for g in groups.values() for m in ("C5", "C0'"))))
    if not seeds:
        raise ValueError("no seed has C5 and C0′ on both tracks")
    out: dict[str, Any] = {"host": host, "seeds": seeds, "rood": str(rood_root), "reference": str(reference_root),
                           "resamples": resamples, "seed": seed, "strata": {},
                           "flags": ["single seed: CIs cover evaluation windows only"] if len(seeds) == 1 else []}
    for stratum in strata:
        block: dict[str, Any] = {}
        sums: dict[tuple[str, str], np.ndarray] = {}
        counts: dict[str, np.ndarray] = {}
        for track, group in groups.items():
            starts = None
            for model in ("P0", "C0'", "C5"):
                if model == "P0" and not group.seeds("P0"):
                    continue
                s, n, st = pooled(group, model, seeds, stratum)
                if starts is not None and (not np.array_equal(starts, st) or not np.array_equal(counts[track], n)):
                    raise ValueError(f"{track}: {model}'s windows do not pair with C0′'s")
                starts, counts[track] = st, n
                sums[(track, model)] = s
            block.setdefault("windows", {})[track] = {"windows": int(starts.size), "tokens": int(counts[track].sum())}
            out.setdefault("starts", {})[track] = starts
        if not np.array_equal(out["starts"]["rood"], out["starts"]["reference"]) or \
                not np.array_equal(counts["rood"], counts["reference"]):
            raise ValueError(f"{stratum}: the two tracks' evaluation windows do not pair (other starts or target counts)")
        for name, (candidate, reference, sign) in PREDICTIONS.items():
            if any((track, model) not in sums for track in groups for model in (candidate, reference)):
                block[name] = {"available": False, "reason": f"{candidate} or {reference} missing"}
                continue
            result = relative_difference_of_differences(
                (sums[("rood", candidate)], sums[("rood", reference)]), (sums[("reference", candidate)], sums[("reference", reference)]),
                resamples=resamples, seed=seed)
            block[name] = {"available": True, "contrast": f"({candidate} − {reference}) / {reference}",
                           "predicted": "Δ ≤ 0" if sign < 0 else "Δ > 0", **result,
                           "reading": reading(result["difference"], sign)}
        out["strata"][stratum] = block
    out.pop("starts", None)
    return out


def render(summary: dict[str, Any], title: str) -> str:
    pct = lambda r: f"{r['mean'] * 100:+.2f}% [{r['ci_low'] * 100:+.2f}, {r['ci_high'] * 100:+.2f}]"  # noqa: E731
    lines = [f"# {title}", "",
             f"Host {summary['host']}; seeds {', '.join(map(str, summary['seeds']))} (P0 stands for every seed); "
             f"T7-ROOD runs `{summary['rood']}`, T7 v1 runs `{summary['reference']}`. Relative differences Σ (candidate − reference) "
             f"/ Σ reference over the final-evaluation windows, pooled over seeds; 95% cluster bootstraps over windows "
             f"({summary['resamples']:,} resamples, the same windows for both tracks). Pre-registration: "
             "`experiments/t7-new-vocabulary/preregistration-rood.md` §4.", ""]
    lines += [f"**Flag:** {f}." for f in summary["flags"]]
    for stratum, block in summary["strata"].items():
        windows = block["windows"]["rood"]
        lines += ["", f"## {stratum} ({windows['windows']:,} windows, {windows['tokens']:,} target tokens per track and seed pool)", "",
                  "| prediction | contrast | T7-ROOD | T7 v1 | Δ = ROOD − v1 | p | predicted | reading |", "|---|---|---|---|---|---:|---|---|"]
        for name in PREDICTIONS:
            r = block[name]
            if not r.get("available"):
                lines.append(f"| {name} | — | — | — | — | — | — | not available: {r.get('reason')} |")
                continue
            lines.append(f"| {name} | {r['contrast']} | {pct(r['rood'])} | {pct(r['reference'])} | {pct(r['difference'])} | "
                         f"{r['difference']['p_value']:.4f} | {r['predicted']} | {r['reading']} |")
    lines += ["", "P2 is the pre-registered ordering (the gain on T7-ROOD at least T7 v1's); P3 is its mechanism check (C0′ "
              "improves less after held-out names it never read). Δ values are differences of relative differences, in percentage "
              "points of the reference model's loss."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rood", type=Path, required=True, help="the T7-ROOD stage's run folder (e.g. runs/t7rood)")
    parser.add_argument("--reference", type=Path, required=True, help="the T7 v1 stage's run folder (e.g. runs/t7)")
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--host", default="SmolLM2-360M")
    parser.add_argument("--strata", nargs="+", default=[STRATUM]); parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--title", default="T7-ROOD against T7 v1 — the pre-registered paired comparison (P2, P3)")
    args = parser.parse_args(argv)
    config = {"experiment": "t7rood-vs-t7", "rood": str(args.rood), "reference": str(args.reference), "host": args.host,
              "strata": list(args.strata), "resamples": args.resamples, "seed": args.seed}
    if args.overwrite:
        for name in RESULT_FILES:
            if (args.output / name).is_file():
                (args.output / name).unlink()
    git_at_start = start_output(args.output, config)
    started = time.monotonic()
    summary = compare(args.rood, args.reference, host=args.host, strata=args.strata, resamples=args.resamples, seed=args.seed)
    write_json(args.output / "summary.json", json_ready(summary))
    (args.output / "report.md").write_text(render(summary, args.title))
    finish_output(args.output, config, git_at_start=git_at_start, device="cpu", seconds=round(time.monotonic() - started, 1))
    print(json.dumps({s: {n: b[n].get("reading") for n in PREDICTIONS} for s, b in summary["strata"].items()}, indent=2))


if __name__ == "__main__":
    main()
