"""E9 run configs and job chaining: retrofit × quantization on pretrained hosts (execution.md, E9).

    python -m vsa_embed.experiments.e9_plan --track t5|t4|t1|wordnet [--stage NAME] [--hosts SmolLM2-360M SmolLM2-135M]
        [--host-mode train|lora] [--lora-rank 64] [--host-lr X] [--channel-lr 1e-3] [--gate-bias 0] [--tokens 50000000]
        [--seeds 1] [--models P0 C0p C2 C5] [--no-evals] [--int4-probes SUBSET] [--queue] [--priority 22]
    python -m vsa_embed.experiments.e9_plan --track t5 --hosts Qwen3-1.7B-Base Qwen3-0.6B-Base --host-mode lora
        --lora-rank 64 [--memory-report PATH] [--micro-batch N] [--channel-scale auto|on|off] [--queue]
    python -m vsa_embed.experiments.e9_plan --track t5 --hosts Qwen3.5-2B-Base Qwen3.5-0.8B-Base --host-mode lora
        --lora-rank 64 [--queue]

writes one YAML per (host, model, seed) under `experiments/e9-retrofit/configs/<stage>/` (stem
`<host>-<mode>-<model>-s<seed>`; mode `full` for a fully trained host, `lora`, or `frozen` for P0; the stage
defaults to the track name, `<track>-qwen3` for Qwen3 hosts) and, with `--queue`, adds the jobs to the local GPU
queue (`.jobs/`, through `vsa_embed.jobqueue`, run with this interpreter from the repository root). Runs go to
`experiments/e9-retrofit/runs/<stage>/<stem>/`.

**Tracks** (`e9_tracks`; orchestrator decision after the engagement check: SmolLM2 already models general
WordNet words, so the primary corpora are vocabularies that are new or rare for the host): `t5` enterprise
glossary (contamination-free invented terms; run first), `t4` chemistry (ChEBI names), `t1` T1-open (MeSH on
PubMed; strata on `eval-pubmed` with 2,048 windows), `wordnet` (C3 general corpus; secondary, the negative
control). Each points `data.train` / `data.eval` / `data.ontology` at its corpus for the hosts' tokenizer family
(`TrackSpec.for_family`: SmolLM2, or the Qwen3 relink with the same ontology, alias table and holdout).

**Recipe** (engagement check, 2026-10-02): full fine-tuning (`--host-mode train`, host lr 3e-5; `lora`: rank
`--lora-rank`, host lr 2e-4), channel lr `--channel-lr` 1e-3, gate bias 0 for C2 and C5, 50M tokens, 64
sequences of 1024 tokens per step (65,536 tokens), warmup 2.5M tokens, evaluations at 2.5M, 5M, 10M, … on
≥ 1,024 windows (the track's count if larger) with `eval.save_window_losses`, checkpoints every 10 minutes.

**Qwen3 hosts** (WP-Qwen: Qwen3-0.6B/1.7B/4B-Base, LoRA only — the default host mode for them): the same
recipe with LoRA r = `--lora-rank` (64) at host lr 2e-4; C2 and C5 scale the channel's injection to the host
(`channel.scale_to_host`, open decision 1; `--channel-scale off` disables it, `on` applies it to SmolLM2 too);
Qwen3-4B with non-reentrant gradient checkpointing. Micro-batch, gradient checkpointing, host dtype and
evaluation batch per host come from the memory probe's `recommendation.json` (`e9_memory`; default path
`experiments/e9-retrofit/memory/qwen3-v1/recommendation.json`, used when present) or `--micro-batch`, else
from the provisional `cpt_plan.HOSTS` values; the per-run evaluations get smaller batches
(`EVAL_JOB_BATCH`: 151,936-entry logits) and INT4 probes without WSD by default (open decision 41); default
priority 26.

**Qwen3.5 hosts** (WP-Qwen35: Qwen3.5-2B-Base, Qwen3.5-0.8B-Base; family `qwen3_5`, stage `<track>-qwen35`, default
priority 52): the Qwen3 recipe with LoRA on every projection of the hybrid layers (`model.lora_targets:
linear_attention`) and loss chunks of 1,024 rows (`cpt_plan.HOSTS`); corpora and items of the Qwen3.5 tokenizer
(`e9_tracks.QWEN35_ROOTS`, `items/{new-words,edits}-<track>-qwen3_5-v1`); memory settings from
`experiments/e9-retrofit/memory/qwen3_5-v1/recommendation.json` when present. Every job of these hosts (training,
evaluations, `e4_quant`, report) runs with the host's `python` — the separate environment `~/venvs/vsa-qwen35`
(transformers 5.18, flash-linear-attention, causal-conv1d) — and every other host's with the pinned interpreter
(`stage_python`; also when planned from the Qwen3.5 environment); one interpreter per stage.

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
(`RUN/edit`, `RUN/edit-int4`: `experiments/e9-retrofit/items/{new-words,edits}-<track>-<family>-v1`); at P + 2 —
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
from vsa_embed.experiments.cpt_plan import host_python, pinned_python
from vsa_embed.experiments.e4_plan import conditions, matched_sizes
from vsa_embed.experiments.e9_tracks import TRACKS, TrackSpec, ensure_alias_table, track_spec

ROOT = Path("experiments/e9-retrofit")
E9_HOSTS = ("SmolLM2-360M", "SmolLM2-135M")
QWEN3_HOSTS = ("Qwen3-1.7B-Base", "Qwen3-0.6B-Base", "Qwen3-4B-Base")
QWEN35_HOSTS = ("Qwen3.5-2B-Base", "Qwen3.5-0.8B-Base")
ALL_HOSTS = E9_HOSTS + QWEN3_HOSTS + QWEN35_HOSTS
MODELS = ("P0", "C0p", "C2", "C5")
MODE_LABELS = {"train": "full", "lora": "lora", "frozen": "frozen"}
HOST_LR = {"train": 3.0e-5, "lora": 2.0e-4}
# Micro-batches: SmolLM2-360M as in the engagement check (full and LoRA-64 at 4); SmolLM2-135M at twice
# that; P0 (frozen, evaluation only) as the C4 frozen-host measurement. Qwen3 (LoRA only): the provisional
# `cpt_plan.HOSTS` values, replaced by the memory probe's recommendation when it exists.
MICRO_BATCH = {"SmolLM2-360M": {"train": 4, "lora": 4, "frozen": 8}, "SmolLM2-135M": {"train": 8, "lora": 8, "frozen": 16},
               **{host: dict(CPT_HOSTS[host]["micro_batch"]) for host in QWEN3_HOSTS + QWEN35_HOSTS}}
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
QWEN_PRIORITY = 26
QWEN35_PRIORITY = 52
INT4_PROBES_NO_WSD = "lambada,wic,card660,rare_words,bless,hyperlex"           # open decision 41
MEMORY_REPORT = ROOT / "memory" / "qwen3-v1" / "recommendation.json"
QWEN35_MEMORY_REPORT = ROOT / "memory" / "qwen3_5-v1" / "recommendation.json"
# Per host tokenizer family: default host mode, queue priority, INT4 probe subset, channel keys of C2/C5, stage suffix,
# default memory-probe report.
FAMILIES: dict[str, dict[str, Any]] = {
    "smollm2": {"mode": "train", "priority": PRIORITY, "int4_probes": "all", "channel": {}, "suffix": "",
                "memory_report": MEMORY_REPORT},
    "qwen3": {"mode": "lora", "priority": QWEN_PRIORITY, "int4_probes": INT4_PROBES_NO_WSD,
              "channel": {"scale_to_host": True}, "suffix": "-qwen3", "memory_report": MEMORY_REPORT},
    # Qwen3.5 (WP-Qwen35): the Qwen3 recipe; jobs run with the Qwen3.5 environment's interpreter (`cpt_plan.host_python`)
    "qwen3_5": {"mode": "lora", "priority": QWEN35_PRIORITY, "int4_probes": INT4_PROBES_NO_WSD,
                "channel": {"scale_to_host": True}, "suffix": "-qwen35", "memory_report": QWEN35_MEMORY_REPORT},
}
# Batch of the per-run probe / zero-shot / editing jobs (their prompts produce full 151,936- or 248,320-entry logits).
EVAL_JOB_BATCH = {"Qwen3-0.6B-Base": 16, "Qwen3-1.7B-Base": 8, "Qwen3-4B-Base": 4, "Qwen3.5-0.8B-Base": 16, "Qwen3.5-2B-Base": 8}
# Throughput model for estimates before the memory probe: tokens/s ≈ R / (6 N), with R calibrated on the
# engagement check's SmolLM2-360M LoRA-64 run (12.4k tokens/s at 362M parameters ⇒ R ≈ 27 TFLOP/s); gradient
# checkpointing adds a forward pass (× 3/4).
EFFECTIVE_FLOPS = 6 * 361.8e6 * 12_400
HOST_PARAMETERS = {"SmolLM2-135M": 134.5e6, "SmolLM2-360M": 361.8e6, "Qwen3-0.6B-Base": 596.0e6, "Qwen3-1.7B-Base": 2.032e9,
                   "Qwen3-4B-Base": 4.022e9, "Qwen3.5-0.8B-Base": 752.4e6, "Qwen3.5-2B-Base": 1.8818e9}   # Qwen3.5: text weights


def stage_python(hosts: list[str]) -> str:
    """The interpreter of a stage's jobs: the hosts' `python` (the Qwen3.5 environment) or the pinned one (`cpt_plan.pinned_python`);
    one per stage."""
    pythons = {host_python(h) for h in hosts}
    if len(pythons) != 1:
        raise ValueError(f"the hosts of one stage must run with one interpreter (got {sorted(pythons)})")
    return pythons.pop()


def host_family(host: str) -> str:
    """Tokenizer family of a host (`cpt_plan.HOSTS[host]["corpus"]`): smollm2 or qwen3."""
    family = CPT_HOSTS[host]["corpus"]
    if family not in FAMILIES:
        raise ValueError(f"{host} ({family}) is not an E9 host family; choose from {', '.join(ALL_HOSTS)}")
    return family


def stage_family(hosts: list[str]) -> str:
    families = {host_family(h) for h in hosts}
    if len(families) != 1:
        raise ValueError("one E9 stage per host tokenizer family (SmolLM2 and Qwen3 runs evaluate on different corpora)")
    return families.pop()


def dimension3_items(track: str, family: str = "smollm2") -> tuple[Path, Path]:
    """(new-word items, edit items) of a track for a host tokenizer family (the WordNet ones keep their names)."""
    if track == "wordnet":
        return ITEMS / f"new-words-{family}-v1", ITEMS / f"edits-{family}-v1"
    return ITEMS / f"new-words-{track}-{family}-v1", ITEMS / f"edits-{track}-{family}-v1"


def zeroshot_items(family: str = "smollm2") -> Path:
    """E5.4 synthetic items of the WordNet track (built per host tokenizer)."""
    return ZEROSHOT_ITEMS if family == "smollm2" else ZEROSHOT_ITEMS.with_name(f"c3-synthetic-{family}-v1")


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


def load_memory_report(path: Path | None) -> dict[str, Any] | None:
    """The memory probe's `recommendation.json` (`e9_memory`): host → host mode → setting; None if absent."""
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())["hosts"]


def host_plan(host: str, mode: str, *, memory: dict[str, Any] | None = None, micro_batch: int | None = None) -> dict[str, Any]:
    """Micro-batches (per host mode), host-specific model keys and evaluation batch of `host` for training in
    `mode`: the tables (`MICRO_BATCH`, `cpt_plan.HOSTS`), then the memory probe's recommendation for (host, mode),
    then `micro_batch` (an explicit override)."""
    micro = dict(MICRO_BATCH[host])
    if mode not in micro:
        raise ValueError(f"{host} runs with --host-mode {' or '.join(m for m in micro if m != 'frozen')} (not {mode})")
    settings = CPT_HOSTS[host]
    plan = {"micro": micro, "model": copy.deepcopy(settings.get("model", {})), "eval_batch": int(settings["eval_batch"]),
            "source": "table"}
    found = (memory or {}).get(host, {}).get(mode)
    if found:
        micro[mode] = int(found["micro_batch"])
        if found.get("gradient_checkpointing"):
            plan["model"].update(gradient_checkpointing=True, checkpoint_use_reentrant=False)
        else:
            plan["model"].pop("gradient_checkpointing", None); plan["model"].pop("checkpoint_use_reentrant", None)
        if found.get("host_dtype", "float32") != "float32":
            plan["model"]["host_dtype"] = found["host_dtype"]
        plan["eval_batch"] = int(found.get("eval_batch") or plan["eval_batch"])
        plan["source"] = "memory probe"
        plan["tokens_per_s"] = found.get("tokens_per_s")
    if micro_batch:
        micro[mode] = int(micro_batch)
        plan["source"] = "override"
        plan.pop("tokens_per_s", None)
    return plan


def estimate_hours(host: str, tokens: int, *, tokens_per_s: float | None = None, checkpointing: bool = False) -> float:
    """GPU hours of one training run: measured tokens/s if given, else the 6N throughput model."""
    if not tokens_per_s:
        tokens_per_s = EFFECTIVE_FLOPS / (6 * HOST_PARAMETERS[host]) * (0.75 if checkpointing else 1.0)
    return tokens / tokens_per_s / 3600


def run_config(*, stage: str, host: str, mode: str, model: str, seed: int, data_root: Path, tokens: int, lora_rank: int,
               host_lr: float | None, gate_bias: float, free_dimension: int, sequences_per_step: int = 64,
               windows: int = MIN_WINDOWS, channel_lr: float = 1.0e-3, eval_split: str = "eval",
               overrides: dict[str, Any] | None = None, plan: dict[str, Any] | None = None,
               channel_extra: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """(file stem, config) of one run; P0 ignores the host mode (frozen, evaluation only). `plan` (`host_plan`)
    sets micro-batches, host model keys and evaluation batch (default: the tables); `channel_extra` is merged
    into the channel of C2 and C5 (Qwen3: `scale_to_host`)."""
    if windows < MIN_WINDOWS:
        raise ValueError(f"E9 evaluates on ≥ {MIN_WINDOWS} windows (got {windows})")
    if mode not in HOST_LR:
        raise ValueError("host mode must be train or lora")
    host_mode = "frozen" if model == "P0" else mode
    plan = plan or host_plan(host, mode)
    micro = plan["micro"][host_mode]
    if sequences_per_step % micro:
        raise ValueError(f"{sequences_per_step} sequences per step is not a multiple of micro-batch {micro}")
    settings = CPT_HOSTS[host]
    config = copy.deepcopy(BASE)
    _merge(config, {
        "model": {"pretrained": settings["pretrained"], "host_mode": host_mode, "vocab_size": settings["vocab_size"],
                  "lora_rank": int(lora_rank), **plan["model"]},
        "train": {"micro_batch": micro, "grad_accum": sequences_per_step // micro, "total_tokens": int(tokens),
                  "lr": float(channel_lr),
                  # trainable-only checkpoints: the frozen host is reloaded from the hub cache (a fully trained
                  # host is saved whole either way, as in the engagement check)
                  "save_trainable_only": host_mode != "train"},
        "eval": {"windows": int(windows), "batch": plan["eval_batch"]},
    })
    _merge(config, model_spec(model, free_dimension=free_dimension, gate_bias=gate_bias))
    if model in {"C2", "C5"} and channel_extra:
        _merge(config, {"channel": channel_extra})
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


def write_stage(stage: str, *, hosts: list[str], models: list[str], seeds: list[int], track: str = "t5", mode: str | None = None,
                lora_rank: int = 64, host_lr: float | None = None, gate_bias: float = GATE_BIAS, tokens: int = 50_000_000,
                sequences_per_step: int = 64, windows: int | None = None, channel_lr: float = 1.0e-3,
                data_root: Path | None = None, counts_ontology: Path | None = None, overrides: dict[str, Any] | None = None,
                root: Path = ROOT, memory: dict[str, Any] | None = None, micro_batch: int | None = None,
                channel_scale: str = "auto") -> list[Path]:
    """Write the stage's configs; P0 is written once per host (seed 1: it has no training randomness). All hosts
    of a stage share one tokenizer family; `mode` defaults to the family's (SmolLM2 train, Qwen3 lora);
    `channel_scale` auto | on | off sets `channel.scale_to_host` of C2/C5 (auto: on for Qwen3, off for SmolLM2)."""
    family = stage_family(hosts)
    mode = mode or FAMILIES[family]["mode"]
    spec = track_spec(track, family)
    windows = spec.windows if windows is None else int(windows)
    if windows < max(MIN_WINDOWS, spec.windows):
        raise ValueError(f"E9 evaluates {spec.name} on ≥ {max(MIN_WINDOWS, spec.windows)} windows (got {windows})")
    if channel_scale not in {"auto", "on", "off"}:
        raise ValueError("channel_scale must be auto, on or off")
    channel_extra = ({"scale_to_host": True} if channel_scale == "on" else {} if channel_scale == "off"
                     else FAMILIES[family]["channel"])
    out = root / "configs" / stage
    out.mkdir(parents=True, exist_ok=True)
    corpus_root = Path(data_root) if data_root else spec.data_root
    counts = corpus_root / "ontology.pt" if (corpus_root / "ontology.pt").exists() else counts_ontology
    if counts is None:
        raise FileNotFoundError(f"no ontology under {corpus_root} (build the track's {family} corpus, or pass counts_ontology)")
    paths = []
    for host in hosts:
        free_dimension, _ = matched_sizes(counts, CPT_HOSTS[host]["width"], 256)
        plan = host_plan(host, mode, memory=memory, micro_batch=micro_batch)
        for model in models:
            for seed in ([1] if model == "P0" else seeds):
                stem, config = run_config(stage=stage, host=host, mode=mode, model=model, seed=seed, data_root=corpus_root,
                                          tokens=tokens, lora_rank=lora_rank, host_lr=host_lr, gate_bias=gate_bias,
                                          free_dimension=free_dimension, sequences_per_step=sequences_per_step,
                                          windows=windows, channel_lr=channel_lr,
                                          eval_split=spec.eval_split, overrides=overrides, plan=plan, channel_extra=channel_extra)
                config["e9_track"] = spec.name
                if family != "smollm2":            # new key only where it applies (SmolLM2 configs as before)
                    config["e9_family"] = family
                path = out / f"{stem}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                paths.append(path)
    return paths


def evaluation_jobs(run_dir: Path, spec: TrackSpec, *, python: str = sys.executable, alias_table: Path | None = None,
                    int4_probes: str = "all", batch_size: int | None = None) -> list[tuple[str, list[str], list[str]]]:
    """(suffix, command, retry arguments) of the per-run evaluations at bf16 and INT4 (variant A).
    `int4_probes` restricts the INT4 probe run to a comma-separated subset (it pairs with the bf16 run on
    the tables both have); `batch_size` (large-vocabulary hosts) is passed to every evaluation."""
    run = str(run_dir)
    table = ["--alias-table", str(alias_table)] if alias_table else []
    batch = ["--batch-size", str(int(batch_size))] if batch_size else []
    new_items, edit_items = dimension3_items(spec.name, spec.family)
    jobs = []
    for suffix, quantize in (("", []), ("-int4", ["--quantize", "int4"])):
        subset = ["--probes", int4_probes] if suffix and int4_probes != "all" else []
        jobs.append((f"probes{suffix}", [python, "-m", "vsa_embed.evaluation.channel_probes", "--run", run, *table, *quantize,
                                         *subset, *batch, "--output", str(run_dir / f"probes{suffix}.json")], ["--overwrite"]))
        if spec.name == "wordnet":
            jobs.append((f"zeroshot{suffix}", [python, "-m", "vsa_embed.experiments.e5_zeroshot", "evaluate", "--run", run,
                                               "--items", str(zeroshot_items(spec.family)), "--sources", ZEROSHOT_SOURCES,
                                               *quantize, *batch, "--output", str(run_dir / f"zeroshot{suffix}")], ["--overwrite"]))
        elif spec.zeroshot_items is not None:
            jobs.append((f"zeroshot{suffix}", [python, "-m", "vsa_embed.experiments.e9_tracks", "zeroshot", "--run", run,
                                               "--track", spec.name, *table, *quantize, *batch,
                                               "--output", str(run_dir / f"zeroshot{suffix}")], ["--overwrite"]))
        jobs.append((f"edit{suffix}", [python, "-m", "vsa_embed.experiments.e9_ontology_edit", "evaluate", "--run", run,
                                       "--new-items", str(new_items), "--edit-items", str(edit_items), *table, *quantize, *batch,
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


def _config_host(config: dict[str, Any]) -> str | None:
    pretrained = config.get("model", {}).get("pretrained")
    return next((name for name, settings in CPT_HOSTS.items() if settings["pretrained"] == pretrained), None)


def queue_jobs(paths: list[Path], stage: str, priority: int | None = None, *, track: str = "t5", evals: bool = True,
               root: Path = ROOT, queue_dir: Path | None = None, int4_probes: str | None = None,
               alias_table: Path | None = None) -> list[str]:
    """Training jobs at `priority` (default: the hosts' family's, 22 SmolLM2 / 26 Qwen3); per-run evaluations
    at +1, `e4_quant` over the runs at +2, the R9 report at +3. Names are idempotent: a job that exists is left
    alone (P0, shared by seed batches). Track runs get the evaluation alias table (written here once if
    `alias_table` is not given). The tokenizer family (corpora, items) is read from the configs, and every job
    runs with the hosts' interpreter (`stage_python`: the Qwen3.5 environment for Qwen3.5 hosts, else the pinned one)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    configs = {path: yaml.safe_load(Path(path).read_text()) for path in paths}
    families = {c.get("e9_family", "smollm2") for c in configs.values()} or {"smollm2"}
    if len(families) != 1:
        raise ValueError("one E9 stage per host tokenizer family")
    family = families.pop()
    hosts = [h for h in (_config_host(c) for c in configs.values()) if h is not None]
    python = stage_python(hosts) if hosts else pinned_python()
    spec = track_spec(track, family)
    priority = FAMILIES[family]["priority"] if priority is None else int(priority)
    int4_probes = FAMILIES[family]["int4_probes"] if int4_probes is None else int4_probes
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
        submit(f"{stage}-{path.stem}", [python, "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(run_dir)],
               priority, min_free_gb=20)
        if evals:
            batch = EVAL_JOB_BATCH.get(_config_host(configs[path]) or "")
            for suffix, command, retry in evaluation_jobs(run_dir, spec, python=python, alias_table=alias_table,
                                                          int4_probes=int4_probes, batch_size=batch):
                submit(f"{stage}-{path.stem}-{suffix}", command, priority + 1, min_free_gb=5, resume_args=retry)
    if evals and run_dirs:
        seeds = sorted({int(p.stem.rsplit("-s", 1)[1]) for p in paths if "-P0-" not in p.stem}) or [1]
        batch = f"s{'-'.join(map(str, seeds))}"
        submit(f"{stage}-quant-{batch}", quant_command(stage, run_dirs, python=python, root=root), priority + 2, min_free_gb=5,
               resume_args=[])
        if spec.general_corpus is not None:
            submit(f"{stage}-quant-general-{batch}",
                   quant_command(stage, run_dirs, python=python, root=root, eval_corpus=spec.general_corpus),
                   priority + 2, min_free_gb=5, resume_args=[])
        submit(f"{stage}-report-{batch}", report_command(stage, python=python, root=root, general=spec.general_corpus is not None),
               priority + 3, min_free_gb=1, resume_args=[])
    return queued


def describe_plan(paths: list[Path], plans: dict[str, dict[str, Any]]) -> list[str]:
    """One line per host: micro-batch, checkpointing, host dtype, their source (table, memory probe, override) and
    GPU hours of the trained runs (measured tokens/s from the probe, else the 6N model)."""
    lines = []
    by_host: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        config = yaml.safe_load(Path(path).read_text())
        by_host.setdefault(_config_host(config) or "?", []).append(config)
    for host, configs in by_host.items():
        trained = [c for c in configs if not c["train"].get("eval_only")]
        if not trained or host not in HOST_PARAMETERS:
            continue
        c, plan = trained[0], plans.get(host, {})
        checkpointing = bool(c["model"].get("gradient_checkpointing"))
        hours = estimate_hours(host, int(c["train"]["total_tokens"]), tokens_per_s=plan.get("tokens_per_s"), checkpointing=checkpointing)
        basis = "measured tokens/s" if plan.get("tokens_per_s") else "6N model"
        lines.append(f"{host}: micro-batch {c['train']['micro_batch']} × {c['train']['grad_accum']}, checkpointing {checkpointing}, "
                     f"host dtype {c['model'].get('host_dtype', 'float32')}, eval batch {c['eval']['batch']} "
                     f"({plan.get('source', 'table')}); ≈ {hours:.1f} GPU-h per trained run ({basis}), {len(trained)} trained "
                     f"run(s) ≈ {hours * len(trained):.1f} GPU-h before evaluations")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="t5", choices=sorted(TRACKS), help="corpus, ontology and items (default t5)")
    parser.add_argument("--stage", default=None, help="run-folder stage name (default: the track; <track>-qwen3 / <track>-qwen35 for Qwen3 / Qwen3.5 hosts)")
    parser.add_argument("--hosts", nargs="+", default=list(E9_HOSTS), choices=list(ALL_HOSTS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--host-mode", default=None, choices=sorted(HOST_LR),
                        help="train = full fine-tuning; lora (default: train for SmolLM2, lora for Qwen3 and Qwen3.5)")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--host-lr", type=float, default=None, help="default 3e-5 (train) / 2e-4 (lora)")
    parser.add_argument("--channel-lr", type=float, default=1.0e-3, help="train.lr (the channel; the host takes --host-lr)")
    parser.add_argument("--gate-bias", type=float, default=GATE_BIAS, help="initial gate bias of C2 and C5")
    parser.add_argument("--channel-scale", default="auto", choices=["auto", "on", "off"],
                        help="channel.scale_to_host on C2/C5 (auto: on for Qwen3, off for SmolLM2)")
    parser.add_argument("--tokens", type=int, default=50_000_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--sequences-per-step", type=int, default=64)
    parser.add_argument("--windows", type=int, default=None, help="evaluation windows (default and minimum: the track's)")
    parser.add_argument("--data-root", type=Path, default=None, help="override the track's corpus root")
    parser.add_argument("--memory-report", type=Path, default=None,
                        help="the memory probe's recommendation.json (used if it exists; default: the family's, "
                             f"{MEMORY_REPORT} for SmolLM2/Qwen3, {QWEN35_MEMORY_REPORT} for Qwen3.5)")
    parser.add_argument("--micro-batch", type=int, default=None, help="override the micro-batch of the trained runs")
    parser.add_argument("--overrides", default="{}", help="JSON merged into every config")
    parser.add_argument("--no-evals", action="store_true", help="queue training only (no chained evaluations)")
    parser.add_argument("--int4-probes", default=None,
                        help="probe subset of the INT4 probe jobs (default: all for SmolLM2; "
                             f"{INT4_PROBES_NO_WSD} for Qwen3 and Qwen3.5 — WSD is the slowest)")
    parser.add_argument("--queue", action="store_true"); parser.add_argument("--priority", type=int, default=None,
                                                                             help="default 22 (SmolLM2) / 26 (Qwen3) / 52 (Qwen3.5)")
    args = parser.parse_args(argv)
    family = stage_family(args.hosts)
    stage = args.stage or f"{args.track}{FAMILIES[family]['suffix']}"
    memory = load_memory_report(args.memory_report or FAMILIES[family]["memory_report"])
    paths = write_stage(stage, hosts=args.hosts, models=args.models, seeds=args.seeds, track=args.track, mode=args.host_mode,
                        lora_rank=args.lora_rank, host_lr=args.host_lr, gate_bias=args.gate_bias, tokens=args.tokens,
                        sequences_per_step=args.sequences_per_step, windows=args.windows, channel_lr=args.channel_lr,
                        data_root=args.data_root, overrides=json.loads(args.overrides), memory=memory,
                        micro_batch=args.micro_batch, channel_scale=args.channel_scale)
    print("\n".join(str(p) for p in paths))
    if family != "smollm2" or memory:
        mode = args.host_mode or FAMILIES[family]["mode"]
        plans = {h: host_plan(h, mode, memory=memory, micro_batch=args.micro_batch) for h in args.hosts}
        print("\n".join(describe_plan(paths, plans)))
    if args.queue:
        queued = queue_jobs(paths, stage, args.priority, track=args.track, evals=not args.no_evals, int4_probes=args.int4_probes)
        print(f"queued {len(queued)} job(s)")


if __name__ == "__main__":
    main()
