"""Experiment 00: empirical VSA retrieval-capacity sweeps."""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from vsa_embed.algebra import create_algebra
from vsa_embed.atomics import random_hypervectors
from vsa_embed.metrics import retrieval_metrics
from vsa_embed.provenance import prepare_output_dir, write_run_metadata


DTYPES = {"float32": torch.float32, "float64": torch.float64}
EXECUTABLE_KEYS = {
    "experiment", "seeds", "algebras", "dimensions", "bundle_sizes", "noise_std",
    "dtypes", "candidate_count", "query_count", "primary_metric", "constraints", "artifacts",
}
DESIGN_ONLY_KEYS = {"path_depths", "correlations", "sharding", "cleanup"}


@dataclass(frozen=True)
class Trial:
    algebra: str
    dimension: int
    bundle_size: int
    noise_std: float
    dtype: str
    seed: int


def _trial_grid(config: dict[str, Any]) -> list[Trial]:
    deferred = sorted(DESIGN_ONLY_KEYS & config.keys())
    if deferred:
        raise ValueError(
            f"Design-only axes are not implemented by this milestone: {deferred}. "
            "Use the executable smoke config; axes must never be silently ignored."
        )
    unknown = sorted(config.keys() - EXECUTABLE_KEYS - DESIGN_ONLY_KEYS)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    keys = ("algebras", "dimensions", "bundle_sizes", "noise_std", "dtypes", "seeds")
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    trials = [Trial(*values) for values in itertools.product(*(config[key] for key in keys))]
    unsupported = sorted({t.dtype for t in trials} - DTYPES.keys())
    if unsupported:
        raise ValueError(f"CPU capacity runner supports dtypes {sorted(DTYPES)}; got {unsupported}")
    return trials


def run_trial(trial: Trial, *, candidate_count: int, query_count: int, device: str) -> dict[str, Any]:
    if trial.bundle_size > candidate_count:
        raise ValueError("bundle_size cannot exceed candidate_count")
    generator = torch.Generator(device=device).manual_seed(trial.seed)
    dtype = DTYPES[trial.dtype]
    algebra = create_algebra(trial.algebra)
    unitary_roles = trial.algebra == "unitary_hrr"
    candidates = random_hypervectors(candidate_count, trial.dimension, generator=generator, device=device, dtype=dtype)
    # MAP requires self-inverse ±1 roles. Normalized ±1/sqrt(D) roles would
    # shrink unbound signals by D and make additive-noise comparisons invalid.
    if trial.algebra == "map":
        roles = random_hypervectors(
            trial.bundle_size, trial.dimension, distribution="bipolar",
            generator=generator, device=device, dtype=dtype,
        ) * (trial.dimension**0.5)
    else:
        roles = random_hypervectors(
            trial.bundle_size, trial.dimension, unitary=unitary_roles,
            generator=generator, device=device, dtype=dtype,
        )
    fillers = candidates[: trial.bundle_size]

    started = time.perf_counter()
    pairs = algebra.bind(roles, fillers)
    memory = algebra.bundle(pairs, unit=True)
    target_indices = torch.arange(query_count, device=device) % trial.bundle_size
    queries = algebra.unbind(memory.expand(query_count, -1), roles[target_indices])
    if trial.noise_std:
        noise = torch.randn(queries.shape, generator=generator, device=device, dtype=dtype)
        queries = queries + trial.noise_std * noise / (trial.dimension**0.5)
    elapsed = time.perf_counter() - started
    metrics = retrieval_metrics(queries, candidates, target_indices)
    return {
        **asdict(trial),
        "candidate_count": candidate_count,
        "query_count": query_count,
        **metrics,
        "latency_ms": elapsed * 1000,
        "bytes_memory": memory.numel() * memory.element_size(),
        "bytes_atomics": (roles.numel() + candidates.numel()) * candidates.element_size(),
    }


def run(config: dict[str, Any], output_dir: Path, *, device: str = "cpu") -> list[dict[str, Any]]:
    prepare_output_dir(output_dir)
    trials = _trial_grid(config)
    candidate_count = int(config.get("candidate_count", 2048))
    query_count = int(config.get("query_count", 32))
    results = [run_trial(t, candidate_count=candidate_count, query_count=query_count, device=device) for t in trials]
    fields = list(results[0]) if results else []
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    write_run_metadata(output_dir, config, device=device, trials=len(results))
    _write_report(results, output_dir / "report.md")
    return results


def _write_report(results: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[tuple[str, int, int, float], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["algebra"], row["dimension"], row["bundle_size"], row["noise_std"]), []).append(row)
    summary = []
    for key, rows in grouped.items():
        summary.append((*key, sum(r["top1"] for r in rows) / len(rows), sum(r["mrr"] for r in rows) / len(rows)))
    summary.sort(key=lambda x: (x[0], x[1], x[2]))
    lines = ["# Capacity experiment report", "", "| Algebra | Dimension | Bundle | Noise σ | Mean top-1 | Mean MRR |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {a} | {d} | {b} | {noise:g} | {top1:.3f} | {mrr:.3f} |" for a, d, b, noise, top1, mrr in summary]
    lines += ["", "Values average configured seeds. Noise is kept explicit to avoid hiding robustness failures.", ""]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    results = run(config, args.output, device=args.device)
    print(f"Completed {len(results)} trials; artifacts written to {args.output}")


if __name__ == "__main__":
    main()
