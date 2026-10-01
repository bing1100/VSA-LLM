"""E4 run configs and job queueing (D4.0, D4.7, D4.8, D4.9, D4.1).

`python -m vsa_embed.experiments.e4_plan --stage shakeout|baselines|lmin|screen|core [--queue]` writes
one YAML per (size, condition, seed) under `experiments/e4-small-lm/configs/<stage>/` and, with
`--queue`, adds a job per config to the local GPU queue (`.jobs/`). Conditions (experiments.md E4.1):

C0 none · C1 random fixed span vectors · C1s shuffled frames (matched params) · C1h hashed n-gram
memory (matched params) · C2 free table (capacity-matched width) · C2f free table (full width) ·
C3 static VSA bundle · C3t relation-type-only bundle · C4 + factored attentive mapping · C5 + P1
context · C6 + M3 growth · C7 best + semantic head · C8 operator ablation of the best condition.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path("experiments/e4-small-lm")

BASE = {
    "model": {"vocab_size": 50257, "seq_len": 1024},
    "data": {"min_subtokens": 2, "seed": 1234},
    "eval": {"windows": 1024, "batch": 32, "first_tokens": 10_000_000},
    "device": "cuda",
}
SIZES = {
    "50M": {"model": {"size": "50M"}, "train": {"micro_batch": 32, "grad_accum": 8, "total_tokens": 300_000_000,
                                                 "lr": 1.0e-3, "warmup_tokens": 10_000_000}},
    "125M": {"model": {"size": "125M"}, "train": {"micro_batch": 16, "grad_accum": 16, "total_tokens": 500_000_000,
                                                   "lr": 6.0e-4, "warmup_tokens": 10_000_000}},
}


def conditions(operator: str, key_dimension: int, channel_dimension: int, matched_free_dim: int,
               matched_buckets: int, developmental: dict[str, Any]) -> dict[str, dict[str, Any]]:
    compose = {"mode": "compose", "operator": operator, "dimension": channel_dimension}
    return {
        "C0": {"channel": {"mode": "none"}},
        "C1": {"channel": {"mode": "random"}},
        "C1s": {"channel": {**compose, "composition": "bundle", "frames": "shuffled"}},
        "C1h": {"channel": {"mode": "hashed", "hashed_buckets": matched_buckets}},
        "C2": {"channel": {"mode": "free", "free_dimension": matched_free_dim}},
        "C2f": {"channel": {"mode": "free"}},
        "C3": {"channel": {**compose, "composition": "bundle"}},
        "C3t": {"channel": {**compose, "composition": "bundle", "frames": "relation_only", "operator": "untyped"}},
        "C4": {"channel": {**compose, "composition": "attentive", "key_dimension": key_dimension}},
        "C5": {"channel": {**compose, "composition": "attentive", "key_dimension": key_dimension, "context_window": 8}},
        "C6": {"channel": {**compose, "composition": "attentive", "key_dimension": key_dimension, "context_window": 8,
                           "developmental": developmental}},
    }


def matched_sizes(ontology_path: Path, model_dimension: int, channel_dimension: int) -> tuple[int, int]:
    """Free-table width and hashed buckets whose parameters match the composition channel's
    dictionary plus projector (the parameters a C3 channel adds)."""
    onto = torch.load(ontology_path, weights_only=False)
    channel = (onto["atomic_count"] + onto["relation_count"]) * channel_dimension + channel_dimension * model_dimension
    free_dim = max(4, round((channel - 0) / (onto["entry_count"] + model_dimension)))
    buckets = max(256, round(channel / model_dimension))
    return free_dim, buckets


def write_stage(stage: str, *, data_root: Path, sizes: list[str], names: list[str], seeds: list[int], operator: str,
                key_dimension: int, developmental: dict[str, Any], overrides: dict[str, Any] | None = None) -> list[Path]:
    out = ROOT / "configs" / stage
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for size in sizes:
        model_dim = {"50M": 512, "125M": 768}[size]
        free_dim, buckets = matched_sizes(data_root / "ontology.pt", model_dim, 256)
        table = conditions(operator, key_dimension, 256, free_dim, buckets, developmental)
        for name in names:
            spec = table[name.split("@")[0]]
            for seed in seeds:
                config = copy.deepcopy(BASE)
                for part in (SIZES[size], spec, overrides or {}):
                    for key, value in part.items():
                        config.setdefault(key, {})
                        if isinstance(value, dict):
                            config[key].update(copy.deepcopy(value))
                        else:
                            config[key] = value
                config["seed"] = seed
                config["data"].update(train=str(data_root / "train"), eval=str(data_root / "eval"),
                                      ontology=str(data_root / "ontology.pt"))
                config["experiment"] = f"e4-{stage}-{size}-{name}-s{seed}"
                path = out / f"{size}-{name}-s{seed}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                paths.append(path)
    return paths


def queue_jobs(paths: list[Path], stage: str, priority: int) -> None:
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    for path in paths:
        name = f"{stage}-{path.stem}"
        output = ROOT / "runs" / stage / path.stem
        try:
            add(DEFAULT_DIR, ["python", "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(output)],
                name=name, priority=priority, min_free_gb=20, env={"PYTHONPATH": "src"})
        except FileExistsError:
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("~/data/vsa-llm/c3/wordnet-gpt2-v1").expanduser())
    parser.add_argument("--sizes", nargs="+", default=["50M"])
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--operator", default="hrr"); parser.add_argument("--key-dimension", type=int, default=8)
    parser.add_argument("--developmental", default="{}")
    parser.add_argument("--overrides", default="{}", help="JSON merged into every config (e.g. shorter budgets)")
    parser.add_argument("--queue", action="store_true"); parser.add_argument("--priority", type=int, default=50)
    args = parser.parse_args(argv)
    paths = write_stage(args.stage, data_root=args.data_root, sizes=args.sizes, names=args.conditions, seeds=args.seeds,
                        operator=args.operator, key_dimension=args.key_dimension,
                        developmental=json.loads(args.developmental), overrides=json.loads(args.overrides))
    print("\n".join(str(p) for p in paths))
    if args.queue:
        queue_jobs(paths, args.stage, args.priority)


if __name__ == "__main__":
    main()
