"""E9 run configs and job chaining: retrofit × quantization on pretrained hosts (execution.md, E9).

    python -m vsa_embed.experiments.e9_plan --track t5|t4|t1|wordnet [--stage NAME] [--hosts SmolLM2-360M SmolLM2-135M]
        [--host-mode train|lora] [--lora-rank 64] [--host-lr X] [--channel-lr 1e-3] [--gate-bias 0] [--tokens 50000000]
        [--seeds 1] [--models P0 C0p C2 C5] [--no-evals] [--int4-probes SUBSET] [--queue] [--priority 22]

writes one YAML per (host, model, seed) under `experiments/e9-retrofit/configs/<stage>/` (stem
`<host>-<mode>-<model>-s<seed>`; mode `full` for a fully trained host, `lora`, or `frozen` for P0; the stage
defaults to the track name) and, with `--queue`, adds the jobs to the local GPU queue (`.jobs/`, through
`vsa_embed.jobqueue`, run with this interpreter from the repository root). Runs go to
`experiments/e9-retrofit/runs/<stage>/<stem>/`.

**Tracks** (`e9_tracks`; orchestrator decision after the engagement check: SmolLM2 already models general
WordNet words, so the primary corpora are vocabularies that are new or rare for the host): `t5` enterprise
glossary (contamination-free invented terms; run first), `t4` chemistry (ChEBI names), `t1` T1-open (MeSH on
PubMed; strata on `eval-pubmed` with 2,048 windows), `wordnet` (C3 general corpus; secondary, the negative
control). Each points `data.train` / `data.eval` / `data.ontology` at its SmolLM2 corpus.

**Recipe** (engagement check, 2026-10-02): full fine-tuning (`--host-mode train`, host lr 3e-5; `lora`: rank
`--lora-rank`, host lr 2e-4), channel lr `--channel-lr` 1e-3, gate bias 0 for C2 and C5, 50M tokens, 64
sequences of 1024 tokens per step, warmup 2.5M tokens, evaluations at 2.5M, 5M, 10M, … on ≥ 1,024 windows
(the track's count if larger) with `eval.save_window_losses`, checkpoints every 10 minutes.

Models (one fixed test set per track):

- `P0` — the original host, evaluation only (frozen host, no channel, `train.eval_only`; the same windows);
- `C0p` (C0′) — continued training without the channel: the same tokens and trainable host parameters;
- `C2` — the same plus a capacity-matched free per-concept table (`e4_plan.matched_sizes` on the track ontology);
- `C5` — the same plus the attentive VSA channel (hrr, 256 dimensions, key 8, P1 context window 8).

Job chaining (the queue runs the lowest priority number first, so everything queued after training at
priority P runs once the stage's training is done): per run (P0 included) at P + 1 — channel probes at bf16
and INT4 (`RUN/probes.json`, `RUN/probes-int4.json`), the track's zero-shot items at bf16 and INT4
(`RUN/zeroshot`, `RUN/zeroshot-int4`: E5.4 synthetic items for WordNet, the WP-C7 `zeroshot_property` /
`zeroshot_entailment` items for T5/T4, none for T1) and the ontology-editing evaluation at bf16 and INT4
(`RUN/edit`, `RUN/edit-int4`: `experiments/e9-retrofit/items/{new-words,edits}-<track>-smollm2-v1`); at P + 2 —
`e4_quant` over the batch's runs on the track's evaluation corpus (`experiments/e9-retrofit/quant/<stage>/`)
and, for tracks with a general-text corpus, on `eval-general` (`quant-general/<stage>/`); at P + 3 — the R9
report (`experiments/e9-retrofit/report/<stage>/`). Track runs read the evaluation alias table written by
`e9_tracks.ensure_alias_table` at queue time. Retries of evaluation jobs replace their partial outputs.
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
from vsa_embed.experiments.e9_tracks import TRACKS, TrackSpec, ensure_alias_table, track_spec

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
GATE_BIAS = 0.0
ITEMS = ROOT / "items"
ZEROSHOT_ITEMS = Path("experiments/e5-explainability/items/c3-synthetic-smollm2-v1")
ZEROSHOT_SOURCES = "own,none,random,mean_row,surface_mean,graph_projection"     # structure-only (E5.4)
PRIORITY = 22


def dimension3_items(track: str) -> tuple[Path, Path]:
    """(new-word items, edit items) of a track (the WordNet ones keep their original names)."""
    if track == "wordnet":
        return ITEMS / "new-words-smollm2-v1", ITEMS / "edits-smollm2-v1"
    return ITEMS / f"new-words-{track}-smollm2-v1", ITEMS / f"edits-{track}-smollm2-v1"


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
               windows: int = MIN_WINDOWS, channel_lr: float = 1.0e-3, eval_split: str = "eval",
               overrides: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
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
                  "lr": float(channel_lr),
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
    config["data"].update(train=str(data_root / "train"), eval=str(data_root / eval_split), ontology=str(data_root / "ontology.pt"))
    stem = f"{host}-{MODE_LABELS[host_mode]}-{model}-s{seed}"
    config["experiment"] = f"e9-{stage}-{stem}"
    return stem, config


def write_stage(stage: str, *, hosts: list[str], models: list[str], seeds: list[int], track: str = "t5", mode: str = "train",
                lora_rank: int = 64, host_lr: float | None = None, gate_bias: float = GATE_BIAS, tokens: int = 50_000_000,
                sequences_per_step: int = 64, windows: int | None = None, channel_lr: float = 1.0e-3,
                data_root: Path | None = None, counts_ontology: Path | None = None, overrides: dict[str, Any] | None = None,
                root: Path = ROOT) -> list[Path]:
    """Write the stage's configs; P0 is written once per host (seed 1: it has no training randomness)."""
    spec = track_spec(track)
    windows = spec.windows if windows is None else int(windows)
    if windows < max(MIN_WINDOWS, spec.windows):
        raise ValueError(f"E9 evaluates {spec.name} on ≥ {max(MIN_WINDOWS, spec.windows)} windows (got {windows})")
    out = root / "configs" / stage
    out.mkdir(parents=True, exist_ok=True)
    corpus_root = Path(data_root) if data_root else spec.data_root
    counts = corpus_root / "ontology.pt" if (corpus_root / "ontology.pt").exists() else counts_ontology
    if counts is None:
        raise FileNotFoundError(f"no ontology under {corpus_root} (pass counts_ontology)")
    paths = []
    for host in hosts:
        if host not in MICRO_BATCH:
            raise ValueError(f"E9 hosts are {', '.join(MICRO_BATCH)}")
        free_dimension, _ = matched_sizes(counts, CPT_HOSTS[host]["width"], 256)
        for model in models:
            for seed in ([1] if model == "P0" else seeds):
                stem, config = run_config(stage=stage, host=host, mode=mode, model=model, seed=seed, data_root=corpus_root,
                                          tokens=tokens, lora_rank=lora_rank, host_lr=host_lr, gate_bias=gate_bias,
                                          free_dimension=free_dimension, sequences_per_step=sequences_per_step,
                                          windows=windows, channel_lr=channel_lr,
                                          eval_split=spec.eval_split, overrides=overrides)
                config["e9_track"] = spec.name
                path = out / f"{stem}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                paths.append(path)
    return paths


def evaluation_jobs(run_dir: Path, spec: TrackSpec, *, python: str = sys.executable, alias_table: Path | None = None,
                    int4_probes: str = "all") -> list[tuple[str, list[str], list[str]]]:
    """(suffix, command, retry arguments) of the per-run evaluations at bf16 and INT4 (variant A).
    `int4_probes` restricts the INT4 probe run to a comma-separated subset (it pairs with the bf16 run on
    the tables both have)."""
    run = str(run_dir)
    table = ["--alias-table", str(alias_table)] if alias_table else []
    new_items, edit_items = dimension3_items(spec.name)
    jobs = []
    for suffix, quantize in (("", []), ("-int4", ["--quantize", "int4"])):
        subset = ["--probes", int4_probes] if suffix and int4_probes != "all" else []
        jobs.append((f"probes{suffix}", [python, "-m", "vsa_embed.evaluation.channel_probes", "--run", run, *table, *quantize,
                                         *subset, "--output", str(run_dir / f"probes{suffix}.json")], ["--overwrite"]))
        if spec.name == "wordnet":
            jobs.append((f"zeroshot{suffix}", [python, "-m", "vsa_embed.experiments.e5_zeroshot", "evaluate", "--run", run,
                                               "--items", str(ZEROSHOT_ITEMS), "--sources", ZEROSHOT_SOURCES, *quantize,
                                               "--output", str(run_dir / f"zeroshot{suffix}")], ["--overwrite"]))
        elif spec.zeroshot_items is not None:
            jobs.append((f"zeroshot{suffix}", [python, "-m", "vsa_embed.experiments.e9_tracks", "zeroshot", "--run", run,
                                               "--track", spec.name, *table, *quantize,
                                               "--output", str(run_dir / f"zeroshot{suffix}")], ["--overwrite"]))
        jobs.append((f"edit{suffix}", [python, "-m", "vsa_embed.experiments.e9_ontology_edit", "evaluate", "--run", run,
                                       "--new-items", str(new_items), "--edit-items", str(edit_items), *table, *quantize,
                                       "--output", str(run_dir / f"edit{suffix}")], ["--overwrite"]))
    return jobs


def quant_command(stage: str, run_dirs: list[Path], *, python: str = sys.executable, root: Path = ROOT,
                  eval_corpus: Path | None = None) -> list[str]:
    folder = "quant-general" if eval_corpus else "quant"
    return [python, "-m", "vsa_embed.experiments.e4_quant", "--runs", *map(str, run_dirs), "--output", str(root / folder / stage),
            "--bits", "8", "4", "--variants", "A", "B", "--baseline", "C0'", "--references", "C2", "--resume",
            *(["--eval-corpus", str(eval_corpus)] if eval_corpus else []),
            "--title", f"E9 post-training quantization ({stage}{', general text' if eval_corpus else ''})"]


def report_command(stage: str, *, python: str = sys.executable, root: Path = ROOT, general: bool = False) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_report", "--runs", str(root / "runs" / stage), "--quant",
            str(root / "quant" / stage), *(["--quant-general", str(root / "quant-general" / stage)] if general else []),
            "--output", str(root / "report" / stage), "--overwrite"]


def queue_jobs(paths: list[Path], stage: str, priority: int = PRIORITY, *, track: str = "t5", evals: bool = True,
               root: Path = ROOT, queue_dir: Path | None = None, int4_probes: str = "all",
               alias_table: Path | None = None) -> list[str]:
    """Training jobs at `priority`; per-run evaluations at +1, `e4_quant` over the runs at +2, the R9
    report at +3. Names are idempotent: a job that exists is left alone (P0, shared by seed batches).
    Track runs get the evaluation alias table (written here once if `alias_table` is not given)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    spec = track_spec(track)
    queue = queue_dir or DEFAULT_DIR
    env = {"PYTHONPATH": "src"}
    queued: list[str] = []
    if evals and alias_table is None:
        alias_table = ensure_alias_table(spec)

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
            for suffix, command, retry in evaluation_jobs(run_dir, spec, alias_table=alias_table, int4_probes=int4_probes):
                submit(f"{stage}-{path.stem}-{suffix}", command, priority + 1, min_free_gb=5, resume_args=retry)
    if evals and run_dirs:
        seeds = sorted({int(p.stem.rsplit("-s", 1)[1]) for p in paths if "-P0-" not in p.stem}) or [1]
        batch = f"s{'-'.join(map(str, seeds))}"
        submit(f"{stage}-quant-{batch}", quant_command(stage, run_dirs, root=root), priority + 2, min_free_gb=5, resume_args=[])
        if spec.general_corpus is not None:
            submit(f"{stage}-quant-general-{batch}", quant_command(stage, run_dirs, root=root, eval_corpus=spec.general_corpus),
                   priority + 2, min_free_gb=5, resume_args=[])
        submit(f"{stage}-report-{batch}", report_command(stage, root=root, general=spec.general_corpus is not None),
               priority + 3, min_free_gb=1, resume_args=[])
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="t5", choices=sorted(TRACKS), help="corpus, ontology and items (default t5)")
    parser.add_argument("--stage", default=None, help="run-folder stage name (default: the track)")
    parser.add_argument("--hosts", nargs="+", default=list(E9_HOSTS), choices=list(E9_HOSTS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--host-mode", default="train", choices=sorted(HOST_LR), help="train = full fine-tuning; lora")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--host-lr", type=float, default=None, help="default 3e-5 (train) / 2e-4 (lora)")
    parser.add_argument("--channel-lr", type=float, default=1.0e-3, help="train.lr (the channel; the host takes --host-lr)")
    parser.add_argument("--gate-bias", type=float, default=GATE_BIAS, help="initial gate bias of C2 and C5")
    parser.add_argument("--tokens", type=int, default=50_000_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--sequences-per-step", type=int, default=64)
    parser.add_argument("--windows", type=int, default=None, help="evaluation windows (default and minimum: the track's)")
    parser.add_argument("--data-root", type=Path, default=None, help="override the track's corpus root")
    parser.add_argument("--overrides", default="{}", help="JSON merged into every config")
    parser.add_argument("--no-evals", action="store_true", help="queue training only (no chained evaluations)")
    parser.add_argument("--int4-probes", default="all",
                        help="probe subset of the INT4 probe jobs (e.g. lambada,wic,card660,rare_words,bless,hyperlex; WSD is the slowest)")
    parser.add_argument("--queue", action="store_true"); parser.add_argument("--priority", type=int, default=PRIORITY)
    args = parser.parse_args(argv)
    stage = args.stage or args.track
    paths = write_stage(stage, hosts=args.hosts, models=args.models, seeds=args.seeds, track=args.track, mode=args.host_mode,
                        lora_rank=args.lora_rank, host_lr=args.host_lr, gate_bias=args.gate_bias, tokens=args.tokens,
                        sequences_per_step=args.sequences_per_step, windows=args.windows, channel_lr=args.channel_lr,
                        data_root=args.data_root, overrides=json.loads(args.overrides))
    print("\n".join(str(p) for p in paths))
    if args.queue:
        queued = queue_jobs(paths, stage, args.priority, track=args.track, evals=not args.no_evals, int4_probes=args.int4_probes)
        print(f"queued {len(queued)} job(s)")


if __name__ == "__main__":
    main()
