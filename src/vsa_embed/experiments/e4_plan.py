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
- `recipe` — D4.0 recipe sweep: C0 at 50M × 100M tokens, tokens/step {65k, 131k, 262k} × lr {1e-3, 2e-3,
  4e-3}, seed 1; priority 25. Its best setting is passed to `shakeout`/`baselines` with `--overrides`.
- `baselines` — D4.8: C0 at 50M × 300M and 125M × 500M tokens (the committed recipe), seeds 1–3;
  priority 90 (lowest, backfill).
- `scale-v1` — decision 65, from-scratch scaling screen phase 1 (`experiments/e4-small-lm/preregistration-scale-v1.md`):
  20M / 50M / 125M × 500M tokens, seed 1, the opscreen's corpus, ontology and frozen recipe (32,768 tokens/step; peak
  lr ∝ 1/width anchored at 50M's 2e-3, `SCALE_LR`; warmup 10M; evaluations at 5M·2^k and the end with window losses);
  C0, C2, C5, HRRAdd, HRRCat at every size and C5sh at 50M; priority 54.3 (the queue script
  `experiments/e4-small-lm/queue-commands-scale-v1.sh` adds the same jobs in the pre-registered order).

A condition name `X@label` uses condition X's channel under its own label. Configs carrying
`train.stop_after_steps` get a second, `--resume` job. Conditions (experiments.md E4.1):

C0 none · C1 random fixed span vectors · C1s shuffled frames (matched params) · C1h hashed n-gram
memory (matched params) · C2 free table (capacity-matched width) · C2f free table (full width) ·
C3 static VSA bundle · C3t relation-type-only bundle · C4 + factored attentive mapping · C5 + P1
context · C6 + M3 growth · C7 best + semantic head · C8 operator ablation of the best condition ·
C5sh C5 on shuffled frames (each entry reads another entry's frame) · HRRAdd / HRRCat HRRBERT's hybrids (decision
65): C5's composer at half its width plus a free per-entry row, summed (lifted) or concatenated before the projector,
matched to C5's channel parameters (`matched_hybrid_widths`).
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
    # D4.0 recipe sweep (added after the S0 pilot: at 262k tokens/step a 100M-token run is only 381 optimizer
    # steps and the loss is still ≈ 5.6): C0 at 50M × 100M tokens over tokens/step {65k, 131k, 262k} × peak lr
    # {1e-3, 2e-3, 4e-3}; the best final loss fixes the recipe that `shakeout` and `baselines` then use.
    "recipe": {"priority": 25, "blocks": [
        {"sizes": ["50M"], "conditions": [f"C0@b{accum * 32}k_lr{lr:g}"], "seeds": [1],
         "overrides": {"train": {"total_tokens": 100_000_000, "warmup_tokens": 5_000_000, "checkpoint_minutes": 5,
                                 "micro_batch": 32, "grad_accum": accum, "lr": lr},
                       "eval": {"first_tokens": 10_000_000, "windows": 512, "save_window_losses": True}}}
        for accum in (2, 4, 8) for lr in (1e-3, 2e-3, 4e-3)]},
    "baselines": {"priority": 90, "blocks": [
        {"sizes": ["50M", "125M"], "conditions": ["C0"], "seeds": [1, 2, 3], "overrides": {"eval": {"save_window_losses": True}}}]},
}
# Decision 65 (scale-v1): the frozen 50M recipe (gates.md D4.0: 32,768 tokens/step, peak lr 2e-3, cosine to 0.1×) at
# every size. Tokens/step stay fixed (125M accumulates 2 micro-batches of 16, its measured memory limit); the peak lr
# scales as 1/width from 50M (the plan's 50M → 125M ratio was 0.6, as GPT-3's and Pythia's; 1/width gives 0.67) and the
# warmup is the plan's 10M tokens for the committed 300M/500M budgets at every size.
SCALE_LR = {"20M": 2.667e-3, "50M": 2.0e-3, "125M": 1.333e-3}
SCALE_BATCH = {"20M": (32, 1), "50M": (32, 1), "125M": (16, 2)}
SCALE_CONDITIONS = {"20M": ["C0", "C5", "C2", "HRRAdd", "HRRCat"], "50M": ["C0", "C5", "C2", "HRRAdd", "HRRCat", "C5sh"],
                    "125M": ["C0", "C5", "C2", "HRRAdd", "HRRCat"]}      # queue order within a size: C0 first
STAGES["scale-v1"] = {"priority": 54.3, "blocks": [
    {"sizes": [size], "conditions": names, "seeds": [1], "overrides": {
        "train": {"total_tokens": 500_000_000, "warmup_tokens": 10_000_000, "checkpoint_minutes": 10, "lr": SCALE_LR[size],
                  "micro_batch": SCALE_BATCH[size][0], "grad_accum": SCALE_BATCH[size][1]},
        "eval": {"first_tokens": 5_000_000, "save_window_losses": True}}}
    for size, names in SCALE_CONDITIONS.items()]}


def conditions(operator: str, key_dimension: int, channel_dimension: int, matched_free_dim: int,
               matched_buckets: int, developmental: dict[str, Any],
               hybrid: dict[str, dict[str, int]] | None = None) -> dict[str, dict[str, Any]]:
    """Channel settings per condition; `hybrid` (`matched_hybrid_widths`) adds HRRAdd / HRRCat at their matched widths."""
    compose = {"mode": "compose", "operator": operator, "dimension": channel_dimension}
    table = {
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
    c5 = table["C5"]["channel"]
    table["C5sh"] = {"channel": {**c5, "frames": "shuffled"}}
    for name, widths in (hybrid or {}).items():
        table[name] = {"channel": {**c5, **widths}}
    return table


def matched_sizes(ontology_path: Path, model_dimension: int, channel_dimension: int) -> tuple[int, int]:
    """Free-table width and hashed buckets whose parameters match the composition channel's
    dictionary plus projector (the parameters a C3 channel adds)."""
    onto = torch.load(ontology_path, weights_only=False)
    channel = (onto["atomic_count"] + onto["relation_count"]) * channel_dimension + channel_dimension * model_dimension
    free_dim = max(4, round((channel - 0) / (onto["entry_count"] + model_dimension)))
    buckets = max(256, round(channel / model_dimension))
    return free_dim, buckets


HYBRIDS = {"HRRAdd": "compose_add", "HRRCat": "compose_cat"}


def model_width(size: str) -> int:
    from ..training.lm import MODEL_SIZES
    return int(MODEL_SIZES[size]["n_embd"])


def channel_parameter_count(channel: dict[str, Any], ontology: dict[str, Any], model_dimension: int) -> int:
    """Parameters of the span channel (not the P1 context) that a config's `channel` section builds at this width."""
    from ..training.lm import build_channel, resolve_config
    built, _ = build_channel(resolve_config({"channel": copy.deepcopy(channel), "device": "cpu"}), ontology, model_dimension)
    return sum(p.numel() for p in built.parameters()) if built is not None else 0


def matched_hybrid_widths(ontology: dict[str, Any], model_dimension: int, channel_dimension: int, key_dimension: int,
                          operator: str = "hrr") -> dict[str, dict[str, Any]]:
    """HRRAdd / HRRCat widths matched to C5's channel parameters (decision 65).

    The composer keeps C5's attentive form at half C5's width and the free per-entry table takes the rest of C5's budget:
    its width is the integer that brings the channel's parameter count closest to C5's (the count is linear in it), so
    each part holds about half the budget. HRRBERT's equal widths (d/2 ‖ d/2) would leave both parts ≈ 20-dimensional
    here (the free table spans all 101,500 entries). Returns {name: {"channel": settings, "parameters", "c5_parameters"}}."""
    c5 = conditions(operator, key_dimension, channel_dimension, 1, 1, {})["C5"]["channel"]
    target = channel_parameter_count(c5, ontology, model_dimension)
    result = {}
    for name, mode in HYBRIDS.items():
        base = {**c5, "mode": mode, "dimension": channel_dimension // 2}
        one, two = (channel_parameter_count({**base, "free_dimension": w}, ontology, model_dimension) for w in (1, 2))
        width = max(1, round((target - (one - (two - one))) / (two - one)))
        settings = {**base, "free_dimension": width}
        result[name] = {"channel": settings, "parameters": channel_parameter_count(settings, ontology, model_dimension),
                        "c5_parameters": target}
    return result


def write_stage(stage: str, *, data_root: Path, sizes: list[str], names: list[str], seeds: list[int], operator: str,
                key_dimension: int, developmental: dict[str, Any], overrides: dict[str, Any] | None = None) -> list[Path]:
    out = ROOT / "configs" / stage
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    ontology = None
    for size in sizes:
        model_dim = model_width(size)
        free_dim, buckets = matched_sizes(data_root / "ontology.pt", model_dim, 256)
        hybrid = None
        if any(name.split("@")[0] in HYBRIDS for name in names):
            ontology = ontology or torch.load(data_root / "ontology.pt", weights_only=False)
            hybrid = {name: {k: v for k, v in record["channel"].items() if k in ("mode", "dimension", "free_dimension")}
                      for name, record in matched_hybrid_widths(ontology, model_dim, 256, key_dimension, operator).items()}
        table = conditions(operator, key_dimension, 256, free_dim, buckets, developmental, hybrid)
        for name in names:
            spec = table[name.split("@")[0]]
            for seed in seeds:
                config = copy.deepcopy(BASE)
                for part in (SIZES.get(size, {"model": {"size": size}}), spec, overrides or {}):
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


def queue_jobs(paths: list[Path], stage: str, priority: int | float, *, queue_dir: Path | None = None) -> None:
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


def _priority(text: str) -> int | float:
    from vsa_embed.jobqueue import _priority as parse
    return parse(text)


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
    parser.add_argument("--priority", type=_priority, default=None, help="default: the preset's, else 50")
    args = parser.parse_args(argv)
    blocks = stage_blocks(args.stage, conditions=args.conditions, sizes=args.sizes, seeds=args.seeds,
                          overrides=json.loads(args.overrides))
    paths = []
    for block in blocks:
        paths += write_stage(args.stage, data_root=args.data_root, sizes=block["sizes"], names=block["conditions"],
                             seeds=block["seeds"], operator=args.operator, key_dimension=args.key_dimension,
                             developmental=json.loads(args.developmental), overrides=block["overrides"])
    for size in sorted({s for block in blocks for s in block["sizes"]}):
        free_dim, buckets = matched_sizes(args.data_root / "ontology.pt", model_width(size), 256)
        print(f"# {size}: C2 free_dimension {free_dim}, C1h hashed_buckets {buckets}", file=sys.stderr)
        if any(name.split("@")[0] in HYBRIDS for block in blocks if size in block["sizes"] for name in block["conditions"]):
            ontology = torch.load(args.data_root / "ontology.pt", weights_only=False)
            for name, record in matched_hybrid_widths(ontology, model_width(size), 256, args.key_dimension, args.operator).items():
                channel = record["channel"]
                print(f"# {size}: {name} ({channel['mode']}) composed width {channel['dimension']}, free width "
                      f"{channel['free_dimension']}: {record['parameters']:,} channel parameters vs C5 {record['c5_parameters']:,} "
                      f"({record['parameters'] / record['c5_parameters'] - 1:+.3%})", file=sys.stderr)
    print("\n".join(str(p) for p in paths))
    if args.queue:
        preset_priority = 50 if args.conditions else STAGES[args.stage]["priority"]
        queue_jobs(paths, args.stage, args.priority if args.priority is not None else preset_priority)


if __name__ == "__main__":
    main()
