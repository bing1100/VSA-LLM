"""Experiment 00 milestone 2: correlation, degree skew, and calibration."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F

from vsa_embed.algebra import create_algebra
from vsa_embed.atomics import correlated_hypervectors, random_hypervectors
from vsa_embed.confidence import expected_calibration_error, fit_logistic_calibrator, precision_threshold
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.statistics import wilson_interval


@dataclass(frozen=True)
class Condition:
    algebra: str
    dimension: int
    correlation: float
    degree_profile: str
    noise_std: float


def sample_degree(profile: dict[str, Any], generator: torch.Generator) -> int:
    kind = profile["kind"]
    low, high = int(profile.get("min", 2)), int(profile.get("max", 128))
    if kind == "fixed":
        degree = int(profile["value"])
    elif kind == "poisson":
        degree = int(torch.poisson(torch.tensor(float(profile["mean"])), generator=generator).item())
    elif kind == "power_law":
        alpha = float(profile.get("alpha", 2.5))
        if alpha <= 1:
            raise ValueError("power-law alpha must exceed 1")
        u = torch.rand((), generator=generator).clamp_max(1 - 1e-7)
        degree = int(low * (1 - u).pow(-1 / (alpha - 1)).item())
    else:
        raise ValueError(f"unknown degree profile kind: {kind}")
    return max(low, min(high, degree))


def _conditions(config: dict[str, Any]) -> list[Condition]:
    keys = ("algebras", "dimensions", "correlations", "degree_profiles", "noise_std")
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    unknown_algebras = set(config["algebras"]) - {"real_hrr", "unitary_hrr", "map"}
    if unknown_algebras:
        raise ValueError(f"unsupported algebras: {sorted(unknown_algebras)}")
    return [Condition(*values) for values in itertools.product(*(config[key] for key in keys))]


def simulate_seed(
    condition: Condition,
    profile: dict[str, Any],
    *,
    seed: int,
    split: str,
    candidate_count: int,
    memories: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    algebra = create_algebra(condition.algebra)
    candidates = correlated_hypervectors(candidate_count, condition.dimension, condition.correlation, generator=generator)
    rows: list[dict[str, Any]] = []
    for memory_id in range(memories):
        degree = sample_degree(profile, generator)
        if degree > candidate_count:
            raise ValueError("sampled degree exceeds candidate_count")
        targets = torch.randperm(candidate_count, generator=generator)[:degree]
        fillers = candidates[targets]
        if condition.algebra == "map":
            roles = random_hypervectors(degree, condition.dimension, distribution="bipolar", generator=generator) * condition.dimension**0.5
        else:
            roles = random_hypervectors(degree, condition.dimension, unitary=condition.algebra == "unitary_hrr", generator=generator)
        memory = algebra.bundle(algebra.bind(roles, fillers))
        queries = algebra.unbind(memory.expand(degree, -1), roles)
        if condition.noise_std:
            queries += condition.noise_std * torch.randn(queries.shape, generator=generator) / condition.dimension**0.5
        similarities = F.normalize(queries, dim=-1) @ F.normalize(candidates, dim=-1).T
        predictions = similarities.argmax(dim=1)
        # Confidence must be available without knowing the target: use the
        # predicted top-1 minus top-2 similarity, never the oracle true margin.
        top2 = similarities.topk(2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]
        ranks = similarities.argsort(dim=1, descending=True).eq(targets[:, None]).nonzero()[:, 1] + 1
        for query_id in range(degree):
            rows.append({
                **asdict(condition), "split": split, "seed": seed, "memory_id": memory_id,
                "query_id": query_id, "degree": degree, "target": int(targets[query_id]),
                "prediction": int(predictions[query_id]), "correct": int(predictions[query_id] == targets[query_id]),
                "rank": int(ranks[query_id]), "margin": float(margins[query_id]),
            })
    return rows


def _tensor(rows: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.tensor([row[key] for row in rows], dtype=torch.float64)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    calibration_seeds = list(config["calibration_seeds"])
    evaluation_seeds = list(config["evaluation_seeds"])
    if set(calibration_seeds) & set(evaluation_seeds):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    prepare_output_dir(output_dir)
    config = {"threshold_method": "point", **config}
    profiles = config["degree_profile_definitions"]
    all_rows: list[dict[str, Any]] = []
    summaries, models = [], {}
    for condition in _conditions(config):
        if condition.degree_profile not in profiles:
            raise ValueError(f"missing definition for profile {condition.degree_profile!r}")
        profile = profiles[condition.degree_profile]
        train = sum((simulate_seed(condition, profile, seed=seed, split="calibration", candidate_count=config["candidate_count"], memories=config["memories_per_seed"]) for seed in calibration_seeds), [])
        test = sum((simulate_seed(condition, profile, seed=seed, split="evaluation", candidate_count=config["candidate_count"], memories=config["memories_per_seed"]) for seed in evaluation_seeds), [])
        calibrator = fit_logistic_calibrator(_tensor(train, "margin"), _tensor(train, "correct"))
        train_conf = calibrator.predict(_tensor(train, "margin"))
        test_conf = calibrator.predict(_tensor(test, "margin"))
        threshold, calibration_coverage, calibration_accuracy = precision_threshold(
            train_conf, _tensor(train, "correct").bool(), config["target_accepted_accuracy"],
            method=config["threshold_method"],
        )
        accepted = test_conf >= threshold
        test_correct = _tensor(test, "correct").bool()
        coverage = accepted.float().mean().item()
        selective_accuracy = test_correct[accepted].float().mean().item() if accepted.any() else 0.0
        accepted_n, accepted_correct = int(accepted.sum()), int(test_correct[accepted].sum())
        selective_ci = wilson_interval(accepted_correct, accepted_n) if accepted_n else (0.0, 0.0)
        key = "|".join(map(str, asdict(condition).values()))
        models[key] = {**asdict(condition), **calibrator.to_dict(), "threshold": threshold}
        for row, confidence, is_accepted in zip(test, test_conf, accepted):
            row["confidence"] = float(confidence); row["accepted"] = int(is_accepted)
        for row, confidence in zip(train, train_conf):
            row["confidence"] = float(confidence); row["accepted"] = int(confidence >= threshold)
        all_rows.extend(train + test)
        summaries.append({
            **asdict(condition), "calibration_n": len(train), "evaluation_n": len(test),
            "top1": test_correct.float().mean().item(), "mrr": (1 / _tensor(test, "rank")).mean().item(),
            "ece": expected_calibration_error(test_conf, test_correct),
            "brier": torch.mean((test_conf - test_correct.float()) ** 2).item(), "threshold": threshold,
            "calibration_coverage": calibration_coverage, "calibration_selective_accuracy": calibration_accuracy,
            "evaluation_coverage": coverage, "evaluation_selective_accuracy": selective_accuracy,
            "evaluation_accepted_n": accepted_n,
            "evaluation_selective_accuracy_ci_low": selective_ci[0],
            "evaluation_selective_accuracy_ci_high": selective_ci[1],
        })
    _write_csv(output_dir / "observations.csv", all_rows)
    _write_csv(output_dir / "summary.csv", summaries)
    artifact = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__, "models": models,
    }
    (output_dir / "capacity_model.json").write_text(json.dumps(artifact, indent=2) + "\n")
    write_run_metadata(output_dir, config, device="cpu")
    _write_report(summaries, output_dir / "report.md", config)
    return {"conditions": len(summaries), "observations": len(all_rows)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _write_report(rows: list[dict[str, Any]], path: Path, config: dict[str, Any]) -> None:
    lines = ["# Realistic capacity and calibration", "", "| Algebra | D | Corr. | Degree | Top-1 | MRR | ECE | Brier | Coverage | Selective acc. |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['algebra']} | {r['dimension']} | {r['correlation']:.2f} | {r['degree_profile']} | {r['top1']:.3f} | {r['mrr']:.3f} | {r['ece']:.3f} | {r['brier']:.3f} | {r['evaluation_coverage']:.3f} | {r['evaluation_selective_accuracy']:.3f} |")
    passing = sum(r["ece"] <= config["max_ece"] for r in rows)
    selective = sum(
        r["evaluation_coverage"] > 0 and r["evaluation_selective_accuracy"] >= config["target_accepted_accuracy"]
        for r in rows
    )
    both = sum(
        r["ece"] <= config["max_ece"] and r["evaluation_coverage"] > 0
        and r["evaluation_selective_accuracy"] >= config["target_accepted_accuracy"]
        for r in rows
    )
    confident = sum(
        r["evaluation_coverage"] > 0
        and r["evaluation_selective_accuracy_ci_low"] >= config["target_accepted_accuracy"]
        for r in rows
    )
    promote = both == len(rows) and confident == len(rows)
    cells: dict[tuple[Any, ...], dict[str, float]] = {}
    for r in rows:
        cell = (r["dimension"], r["correlation"], r["degree_profile"], r["noise_std"])
        cells.setdefault(cell, {})[r["algebra"]] = r["top1"]
    winners = {max(scores, key=scores.get) for scores in cells.values() if len(scores) > 1}
    dominance = (
        f"- {next(iter(winners))} has the best top-1 in every condition cell."
        if len(winners) == 1 else
        f"- No algebra has the best top-1 in every condition cell (cell winners: {sorted(winners)})."
    )

    def grouped_finding(field: str, label: str, metrics: tuple[str, ...]) -> str:
        values = sorted({r[field] for r in rows}, key=str)
        parts = []
        for value in values:
            selected = [r for r in rows if r[field] == value]
            rendered = ", ".join(
                f"{metric}={sum(r[metric] for r in selected) / len(selected):.3f}"
                for metric in metrics
            )
            parts.append(f"{value}: {rendered}")
        return f"- {label}: " + "; ".join(parts) + "."

    lines += [
        "", "## Gate decision", "",
        f"- ECE gate (`≤ {config['max_ece']:.2f}`): **{passing}/{len(rows)} conditions pass**.",
        f"- Held-out selective accuracy (`≥ {config['target_accepted_accuracy']:.2f}`, nonzero coverage): **{selective}/{len(rows)} pass**.",
        f"- Both calibration gates: **{both}/{len(rows)} pass**.",
        f"- Selective accuracy with the Wilson 95% lower bound at or above target: **{confident}/{len(rows)}**.",
        f"- Threshold rule: `{config.get('threshold_method', 'point')}`.",
        f"- **Decision (computed): {'promote' if promote else 'do not promote'} a universal default backend or confidence policy.**",
        "", "## Aggregate findings", "",
        grouped_finding("dimension", "Dimension", ("top1", "ece")),
        grouped_finding("correlation", "Candidate correlation", ("top1", "evaluation_coverage")),
        grouped_finding("degree_profile", "Degree profile", ("top1", "ece", "evaluation_selective_accuracy")),
        dominance,
        "", "Calibration and evaluation use disjoint seeds. Confidence is based only on the observable top-1/top-2 similarity gap.", "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(yaml.safe_load(args.config.read_text()), args.output)
    print(f"Completed {result['conditions']} conditions and {result['observations']} observations")


if __name__ == "__main__":
    main()