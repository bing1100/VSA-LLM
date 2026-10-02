"""E4 run configs and job queueing (S0 pilot, D4.0, D4.7, D4.8, D4.9, D4.1).

`python -m vsa_embed.experiments.e4_plan --stage <stage> --conditions … [--queue]` writes one YAML per
(size, condition, seed) under `experiments/e4-small-lm/configs/<stage>/` and, with `--queue`, adds a
job per config to the local GPU queue (`.jobs/`). Without `--conditions` the stage's preset in
`STAGES` is used (`--sizes`, `--seeds`, `--overrides`, `--priority` still override it):

- `pilot` — S0 sanity pilot (execution.md, not a gate, never pooled with gate runs): 50M × 100M
  tokens (warmup 5M), evaluations at 10/20/40/80/100M on 512 windows with per-window losses,
  checkpoints every 5 min, C0, C1, C2, C3, C5 × seeds 1, 2; priority 10.
- `shakeout` — D4.0: C0, C1, C2, C5 × seeds 1, 2 at 50M × 200M tokens (checkpoints every 10 min),
  plus a kill-and-resume check (`C5@resume`, seed 1: stops after half the steps, a second job
  resumes it); priority 50.
- `baselines` — D4.8: C0 at 50M × 300M and 125M × 500M tokens (the committed recipe), seeds 1–3;
  priority 90 (lowest, backfill).

A condition name `X@label` uses condition X's channel under its own label. Configs carrying
`train.stop_after_steps` get a second, `--resume` job. Conditions (experiments.md E4.1):

C0 none · C1 random fixed span vectors · C1s shuffled frames (matched params) · C1h hashed n-gram
memory (matched params) · C2 free table (capacity-matched width) · C2f free table (full width) ·
C3 static VSA bundle · C3t relation-type-only bundle · C4 + factored attentive mapping · C5 + P1
context · C6 + M3 growth · C7 best + semantic head · C8 operator ablation of the best condition.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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
# Short runs checkpoint more often than the 30-minute default (resume is bit-exact, so results do not change).
_SHAKEOUT = {"train": {"total_tokens": 200_000_000, "checkpoint_minutes": 10}, "eval": {"save_window_losses": True}}
STAGES: dict[str, dict[str, Any]] = {
    "pilot": {"priority": 10, "blocks": [
        {"sizes": ["50M"], "conditions": ["C0", "C1", "C2", "C3", "C5"], "seeds": [1, 2],
         "overrides": {"train": {"total_tokens": 100_000_000, "warmup_tokens": 5_000_000, "checkpoint_minutes": 5},
                       "eval": {"first_tokens": 10_000_000, "windows": 512, "save_window_losses": True}}}]},
    "shakeout": {"priority": 50, "blocks": [
        {"sizes": ["50M"], "conditions": ["C0", "C1", "C2", "C5"], "seeds": [1, 2], "overrides": _SHAKEOUT},
        # 200M tokens = 762 steps of 262,144 tokens; stop after 381, then resume (bit-exactness check).
        {"sizes": ["50M"], "conditions": ["C5@resume"], "seeds": [1],
         "overrides": {"train": {**_SHAKEOUT["train"], "stop_after_steps": 381}, "eval": _SHAKEOUT["eval"]}}]},
    "baselines": {"priority": 90, "blocks": [
        {"sizes": ["50M", "125M"], "conditions": ["C0"], "seeds": [1, 2, 3], "overrides": {"eval": {"save_window_losses": True}}}]},
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


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        merged[key] = _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else copy.deepcopy(value)
    return merged


def stage_blocks(stage: str, *, conditions: list[str] | None = None, sizes: list[str] | None = None,
                 seeds: list[int] | None = None, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Blocks of (sizes, conditions, seeds, overrides): the CLI's when `conditions` is given, else the preset's."""
    if conditions:
        return [{"sizes": sizes or ["50M"], "conditions": conditions, "seeds": seeds or [1, 2, 3], "overrides": overrides or {}}]
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} has no preset; pass --conditions (presets: {', '.join(STAGES)})")
    return [{"sizes": sizes or block["sizes"], "conditions": block["conditions"], "seeds": seeds or block["seeds"],
             "overrides": _merge(block["overrides"], overrides or {})} for block in STAGES[stage]["blocks"]]


def queue_jobs(paths: list[Path], stage: str, priority: int, *, queue_dir: Path | None = None) -> None:
    """One job per config; a config with `train.stop_after_steps` also gets a `--resume` job (run after it)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    for path in paths:
        name = f"{stage}-{path.stem}"
        output = ROOT / "runs" / stage / path.stem
        command = [sys.executable, "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(output)]
        jobs = [(name, command)]
        if (yaml.safe_load(path.read_text()).get("train") or {}).get("stop_after_steps"):
            jobs.append((f"{name}-resume", [*command, "--resume"]))
        for job_name, job_command in jobs:
            try:
                add(queue_dir or DEFAULT_DIR, job_command, name=job_name, priority=priority, min_free_gb=20, env={"PYTHONPATH": "src"})
            except FileExistsError:
                pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("~/data/vsa-llm/c3/wordnet-gpt2-v1").expanduser())
    parser.add_argument("--sizes", nargs="+", default=None, help="default: the preset's, else 50M")
    parser.add_argument("--conditions", nargs="+", default=None, help="default: the stage preset")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the preset's, else 1 2 3")
    parser.add_argument("--operator", default="hrr"); parser.add_argument("--key-dimension", type=int, default=8)
    parser.add_argument("--developmental", default="{}")
    parser.add_argument("--overrides", default="{}", help="JSON merged into every config (e.g. shorter budgets)")
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--priority", type=int, default=None, help="default: the preset's, else 50")
    args = parser.parse_args(argv)
    blocks = stage_blocks(args.stage, conditions=args.conditions, sizes=args.sizes, seeds=args.seeds,
                          overrides=json.loads(args.overrides))
    paths = []
    for block in blocks:
        paths += write_stage(args.stage, data_root=args.data_root, sizes=block["sizes"], names=block["conditions"],
                             seeds=block["seeds"], operator=args.operator, key_dimension=args.key_dimension,
                             developmental=json.loads(args.developmental), overrides=block["overrides"])
    for size in sorted({s for block in blocks for s in block["sizes"]}):
        free_dim, buckets = matched_sizes(args.data_root / "ontology.pt", {"50M": 512, "125M": 768}[size], 256)
        print(f"# {size}: C2 free_dimension {free_dim}, C1h hashed_buckets {buckets}", file=sys.stderr)
    print("\n".join(str(p) for p in paths))
    if args.queue:
        preset_priority = 50 if args.conditions else STAGES[args.stage]["priority"]
        queue_jobs(paths, args.stage, args.priority if args.priority is not None else preset_priority)


if __name__ == "__main__":
    main()
