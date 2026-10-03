"""E9 run configs and job chaining: retrofit × quantization on pretrained hosts (execution.md, E9).

    python -m vsa_embed.experiments.e9_plan [--stage main] [--hosts SmolLM2-360M SmolLM2-135M]
        [--host-mode train|lora] [--lora-rank 64] [--host-lr X] [--gate-bias G] [--tokens 50000000]
        [--seeds 1] [--models P0 C0p C2 C5] [--no-evals] [--queue] [--priority 22]

writes one YAML per (host, model, seed) under `experiments/e9-retrofit/configs/<stage>/` (stem
`<host>-<mode>-<model>-s<seed>`; mode `full` for a fully trained host, `lora`, or `frozen` for P0) and,
with `--queue`, adds the jobs to the local GPU queue (`.jobs/`, through `vsa_embed.jobqueue`, run with
this interpreter). Runs go to `experiments/e9-retrofit/runs/<stage>/<stem>/`.

Models (one fixed test set, the host corpus's evaluation strata with per-window losses):

- `P0` — the original host, evaluation only (frozen host, no channel, `train.eval_only`; the same
  evaluation windows and schedule as the trained runs, so it pairs with them by window);
- `C0p` (C0′) — continued training without the channel: the same tokens, the same trainable host
  parameters (`--host-mode train`: the whole host at `--host-lr`, default 3e-5; `lora`: LoRA of rank
  `--lora-rank`, default 64, at `--host-lr`, default 2e-4);
- `C2` — the same plus a capacity-matched free per-concept table (`e4_plan.matched_sizes` on the host
  ontology: the width whose parameters equal the C3 dictionary plus projector);
- `C5` — the same plus the attentive VSA channel (WordNet ontology, C3 holdout; hrr, 256 dimensions,
  key 8, P1 context window 8).

C2 and C5 share `--gate-bias` (the trainer's default, −2, unless the engagement check picked another).
The recipe follows the engagement check (`experiments/e4-small-lm/configs/e9-check/`): 64 sequences of
1024 tokens per step, channel lr 1e-3, warmup 2.5M tokens, evaluations at 2.5M, 5M, 10M, … tokens on
≥ 1,024 windows with `eval.save_window_losses`, checkpoints every 10 minutes.

Job chaining (the queue runs the lowest priority number first, so everything queued after a training
job at priority P runs once the stage's training is done): per run (P0 included) at P + 1 — channel
probes at bf16 and INT4 (`RUN/probes.json`, `RUN/probes-int4.json`), E5.4 zero-shot on the contamination-
free synthetic items at bf16 and INT4 (`RUN/zeroshot`, `RUN/zeroshot-int4`, structure-only sources) and
the ontology-editing evaluation at bf16 and INT4 (`RUN/edit`, `RUN/edit-int4`); at P + 2 — `e4_quant` over
the batch's runs (bf16 / INT8 / INT4 × channel FP16 (A) / quantized (B), into
`experiments/e9-retrofit/quant/<stage>/`, resumable); at P + 3 — the R9 report
(`experiments/e9-retrofit/report/<stage>/`). Retries of evaluation jobs replace their partial outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from vsa_embed.experiments.cpt_plan import HOSTS as CPT_HOSTS
from vsa_embed.experiments.e4_plan import conditions, matched_sizes
from vsa_embed.experiments.host_corpus import default_paths

ROOT = Path("experiments/e9-retrofit")
E9_HOSTS = ("SmolLM2-360M", "SmolLM2-135M")
MODELS = ("P0", "C0p", "C2", "C5")
MODE_LABELS = {"train": "full", "lora": "lora", "frozen": "frozen"}
HOST_LR = {"train": 3.0e-5, "lora": 2.0e-4}
# Micro-batches: SmolLM2-360M as in the engagement check (full and LoRA-64 at 4); SmolLM2-135M at twice
# that; P0 (frozen, evaluation only) as the C4 frozen-host measurement.
MICRO_BATCH = {"SmolLM2-360M": {"train": 4, "lora": 4, "frozen": 8}, "SmolLM2-135M": {"train": 8, "lora": 8, "frozen": 16}}
BASE = {
    "model": {"size": "pretrained", "seq_len": 1024, "gradient_checkpointing": False},
    "train": {"lr": 1.0e-3, "min_lr_ratio": 0.1, "warmup_tokens": 2_500_000, "weight_decay": 0.1, "log_every": 10,
              "checkpoint_minutes": 10},
    "data": {"min_subtokens": 2, "seed": 1234},
    "eval": {"windows": 1024, "first_tokens": 2_500_000, "save_window_losses": True},
    "device": "cuda",
}
MIN_WINDOWS = 1024
ZEROSHOT_ITEMS = Path("experiments/e5-explainability/items/c3-synthetic-smollm2-v1")
NEW_ITEMS = ROOT / "items" / "new-words-smollm2-v1"
EDIT_ITEMS = ROOT / "items" / "edits-smollm2-v1"
ZEROSHOT_SOURCES = "own,none,random,mean_row,surface_mean,graph_projection"     # structure-only (E5.4)
PRIORITY = 22


def _merge(config: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            _merge(config[key], value)
        else:
            config[key] = copy.deepcopy(value)


def model_spec(model: str, *, free_dimension: int, gate_bias: float) -> dict[str, Any]:
    table = conditions("hrr", 8, 256, free_dimension, 256, {})
    if model in {"P0", "C0p"}:
        return {"channel": {"mode": "none"}}
    if model not in {"C2", "C5"}:
        raise ValueError(f"unknown E9 model {model!r}; choose from {', '.join(MODELS)}")
    spec = copy.deepcopy(table[model])
    spec["channel"]["gate_bias"] = float(gate_bias)
    return spec


def run_config(*, stage: str, host: str, mode: str, model: str, seed: int, data_root: Path, tokens: int, lora_rank: int,
               host_lr: float | None, gate_bias: float, free_dimension: int, sequences_per_step: int = 64,
               windows: int = MIN_WINDOWS, overrides: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """(file stem, config) of one run; P0 ignores the host mode (frozen, evaluation only)."""
    if windows < MIN_WINDOWS:
        raise ValueError(f"E9 evaluates on ≥ {MIN_WINDOWS} windows (got {windows})")
    if mode not in HOST_LR:
        raise ValueError("host mode must be train or lora")
    host_mode = "frozen" if model == "P0" else mode
    micro = MICRO_BATCH[host][host_mode]
    if sequences_per_step % micro:
        raise ValueError(f"{sequences_per_step} sequences per step is not a multiple of micro-batch {micro}")
    settings = CPT_HOSTS[host]
    config = copy.deepcopy(BASE)
    _merge(config, {
        "model": {"pretrained": settings["pretrained"], "host_mode": host_mode, "vocab_size": settings["vocab_size"],
                  "lora_rank": int(lora_rank)},
        "train": {"micro_batch": micro, "grad_accum": sequences_per_step // micro, "total_tokens": int(tokens),
                  # trainable-only checkpoints: the frozen host is reloaded from the hub cache (a fully trained
                  # host is saved whole either way, as in the engagement check)
                  "save_trainable_only": host_mode != "train"},
        "eval": {"windows": int(windows), "batch": settings["eval_batch"]},
    })
    _merge(config, model_spec(model, free_dimension=free_dimension, gate_bias=gate_bias))
    if model == "P0":
        config["train"]["eval_only"] = True
    else:
        config["train"]["host_lr"] = float(host_lr if host_lr is not None else HOST_LR[mode])
    _merge(config, overrides or {})
    config["seed"] = int(seed)
    config["data"].update(train=str(data_root / "train"), eval=str(data_root / "eval"), ontology=str(data_root / "ontology.pt"))
    stem = f"{host}-{MODE_LABELS[host_mode]}-{model}-s{seed}"
    config["experiment"] = f"e9-{stage}-{stem}"
    return stem, config


def write_stage(stage: str, *, hosts: list[str], models: list[str], seeds: list[int], mode: str = "train",
                lora_rank: int = 64, host_lr: float | None = None, gate_bias: float = -2.0, tokens: int = 50_000_000,
                sequences_per_step: int = 64, windows: int = MIN_WINDOWS, data_root: Path | None = None,
                counts_ontology: Path | None = None, overrides: dict[str, Any] | None = None, root: Path = ROOT) -> list[Path]:
    """Write the stage's configs; P0 is written once per host (seed 1: it has no training randomness)."""
    out = root / "configs" / stage
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for host in hosts:
        if host not in MICRO_BATCH:
            raise ValueError(f"E9 hosts are {', '.join(MICRO_BATCH)}")
        corpus_root = data_root or default_paths(CPT_HOSTS[host]["corpus"])[0]
        counts = corpus_root / "ontology.pt" if (corpus_root / "ontology.pt").exists() else counts_ontology
        if counts is None:
            raise FileNotFoundError(f"no ontology under {corpus_root} (pass counts_ontology)")
        free_dimension, _ = matched_sizes(counts, CPT_HOSTS[host]["width"], 256)
        for model in models:
            for seed in ([1] if model == "P0" else seeds):
                stem, config = run_config(stage=stage, host=host, mode=mode, model=model, seed=seed, data_root=corpus_root,
                                          tokens=tokens, lora_rank=lora_rank, host_lr=host_lr, gate_bias=gate_bias,
                                          free_dimension=free_dimension, sequences_per_step=sequences_per_step,
                                          windows=windows, overrides=overrides)
                path = out / f"{stem}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                paths.append(path)
    return paths


def evaluation_jobs(run_dir: Path, *, python: str = sys.executable, zeroshot_items: Path = ZEROSHOT_ITEMS,
                    new_items: Path = NEW_ITEMS, edit_items: Path = EDIT_ITEMS,
                    int4_probes: str = "all") -> list[tuple[str, list[str], list[str]]]:
    """(suffix, command, retry arguments) of the per-run evaluations at bf16 and INT4 (variant A).
    `int4_probes` restricts the INT4 probe run to a comma-separated subset (it pairs with the bf16 run
    on the tables both have)."""
    run = str(run_dir)
    jobs = []
    for suffix, quantize in (("", []), ("-int4", ["--quantize", "int4"])):
        subset = ["--probes", int4_probes] if suffix and int4_probes != "all" else []
        jobs.append((f"probes{suffix}", [python, "-m", "vsa_embed.evaluation.channel_probes", "--run", run, *quantize, *subset,
                                         "--output", str(run_dir / f"probes{suffix}.json")], ["--overwrite"]))
        jobs.append((f"zeroshot{suffix}", [python, "-m", "vsa_embed.experiments.e5_zeroshot", "evaluate", "--run", run,
                                           "--items", str(zeroshot_items), "--sources", ZEROSHOT_SOURCES, *quantize,
                                           "--output", str(run_dir / f"zeroshot{suffix}")], ["--overwrite"]))
        jobs.append((f"edit{suffix}", [python, "-m", "vsa_embed.experiments.e9_ontology_edit", "evaluate", "--run", run,
                                       "--new-items", str(new_items), "--edit-items", str(edit_items), *quantize,
                                       "--output", str(run_dir / f"edit{suffix}")], ["--overwrite"]))
    return jobs


def quant_command(stage: str, run_dirs: list[Path], *, python: str = sys.executable, root: Path = ROOT) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e4_quant", "--runs", *map(str, run_dirs), "--output", str(root / "quant" / stage),
            "--bits", "8", "4", "--variants", "A", "B", "--baseline", "C0'", "--references", "C2", "--resume",
            "--title", f"E9 post-training quantization ({stage})"]


def report_command(stage: str, *, python: str = sys.executable, root: Path = ROOT) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_report", "--runs", str(root / "runs" / stage), "--quant",
            str(root / "quant" / stage), "--output", str(root / "report" / stage), "--overwrite"]


def queue_jobs(paths: list[Path], stage: str, priority: int = PRIORITY, *, evals: bool = True, root: Path = ROOT,
               queue_dir: Path | None = None, int4_probes: str = "all") -> list[str]:
    """Training jobs at `priority`; per-run evaluations at +1, `e4_quant` over the runs at +2, the R9
    report at +3. Names are idempotent: a job that exists is left alone (P0, shared by seed batches)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    queue = queue_dir or DEFAULT_DIR
    env = {"PYTHONPATH": "src"}
    queued: list[str] = []

    def submit(name: str, command: list[str], level: int, *, min_free_gb: float, resume_args: list[str] | None = None) -> None:
        try:
            add(queue, command, name=name, priority=level, min_free_gb=min_free_gb, env=env, resume_args=resume_args)
            queued.append(name)
        except FileExistsError:
            pass

    run_dirs = []
    for path in paths:
        run_dir = root / "runs" / stage / path.stem
        run_dirs.append(run_dir)
        submit(f"{stage}-{path.stem}", [sys.executable, "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(run_dir)],
               priority, min_free_gb=20)
        if evals:
            for suffix, command, retry in evaluation_jobs(run_dir, int4_probes=int4_probes):
                submit(f"{stage}-{path.stem}-{suffix}", command, priority + 1, min_free_gb=5, resume_args=retry)
    if evals and run_dirs:
        seeds = sorted({int(p.stem.rsplit("-s", 1)[1]) for p in paths if "-P0-" not in p.stem}) or [1]
        batch = f"s{'-'.join(map(str, seeds))}"
        submit(f"{stage}-quant-{batch}", quant_command(stage, run_dirs, root=root), priority + 2, min_free_gb=5, resume_args=[])
        submit(f"{stage}-report-{batch}", report_command(stage, root=root), priority + 3, min_free_gb=1, resume_args=[])
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="main")
    parser.add_argument("--hosts", nargs="+", default=list(E9_HOSTS), choices=list(E9_HOSTS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--host-mode", default="train", choices=sorted(HOST_LR), help="train = full fine-tuning; lora")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--host-lr", type=float, default=None, help="default 3e-5 (train) / 2e-4 (lora)")
    parser.add_argument("--gate-bias", type=float, default=-2.0, help="initial gate bias of C2 and C5")
    parser.add_argument("--tokens", type=int, default=50_000_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--sequences-per-step", type=int, default=64)
    parser.add_argument("--windows", type=int, default=MIN_WINDOWS)
    parser.add_argument("--data-root", type=Path, default=None, help="host corpus root (default: host_corpus's SmolLM2 path)")
    parser.add_argument("--overrides", default="{}", help="JSON merged into every config")
    parser.add_argument("--no-evals", action="store_true", help="queue training only (no chained evaluations)")
    parser.add_argument("--int4-probes", default="all",
                        help="probe subset of the INT4 probe jobs (e.g. lambada,wic,card660,rare_words,bless,hyperlex; WSD is the slowest)")
    parser.add_argument("--queue", action="store_true"); parser.add_argument("--priority", type=int, default=PRIORITY)
    args = parser.parse_args(argv)
    paths = write_stage(args.stage, hosts=args.hosts, models=args.models, seeds=args.seeds, mode=args.host_mode,
                        lora_rank=args.lora_rank, host_lr=args.host_lr, gate_bias=args.gate_bias, tokens=args.tokens,
                        sequences_per_step=args.sequences_per_step, windows=args.windows, data_root=args.data_root,
                        overrides=json.loads(args.overrides))
    print("\n".join(str(p) for p in paths))
    if args.queue:
        queued = queue_jobs(paths, args.stage, args.priority, evals=not args.no_evals, int4_probes=args.int4_probes)
        print(f"queued {len(queued)} job(s)")


if __name__ == "__main__":
    main()
