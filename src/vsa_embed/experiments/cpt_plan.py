"""Continued-pretraining (CPT) run configs and job queueing (S0 pilot, D4.4; experiments.md E4.6).

`python -m vsa_embed.experiments.cpt_plan --stage cpt-pilot|cpt [--best-condition C5] [--queue]` writes
one YAML per (host, host mode, condition, seed) under `experiments/e4-small-lm/configs/<stage>/` and,
with `--queue`, adds a job per config to the local GPU queue (`.jobs/`); runs go to
`experiments/e4-small-lm/runs/<stage>/<config stem>/`.

Stages:

- `cpt-pilot` (S0): SmolLM2-135M, LoRA r = 16, 25M tokens, {C0', C2, C5}, seed 1, priority 11.
- `cpt` (D4.4): SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B with a frozen host × {C0', C1, C2, best} ×
  seeds {1, 2, 3}, 100M tokens; plus the LoRA check on SmolLM2-135M (same grid with LoRA r = 16).
  `best` is the best channel condition of the 50M screen, fixed at G3 (`--best-condition`); without
  it the `best` configs are not written. After the LoRA check, `--large-host-mode lora` regenerates
  the larger hosts with LoRA if it won.

C0' is continued pretraining without the channel on the same tokens with the same LoRA if any
(E4.6): with LoRA it trains the adapters only; on a frozen host nothing is trainable, so C0' is the
host itself, evaluated once and logged at every evaluation point (`train.eval_only`).

Channel conditions are e4_plan's (C1 random fixed vectors; C2 free table with capacity-matched width;
C3–C6 composition), sized for the host's width. Every host uses sequence 1024 and a fixed number of
sequences per optimizer step, reached by gradient accumulation over the micro-batch measured for that
host in C4 (experiments.md §Compute). Data come from the per-host corpora of `host_corpus`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from vsa_embed.experiments.e4_plan import conditions, matched_sizes
from vsa_embed.experiments.host_corpus import default_paths

ROOT = Path("experiments/e4-small-lm")
C3_ONTOLOGY = Path("~/data/vsa-llm/c3/wordnet-gpt2-v1/ontology.pt").expanduser()

# Micro-batches from the C4 memory/throughput table (experiments/b6-host-memory/runs/3090-v1): frozen
# hosts as measured; LoRA measured on SmolLM2-135M at 8 (9.5 GiB), halved for the larger hosts.
HOSTS: dict[str, dict[str, Any]] = {
    "SmolLM2-135M": {"pretrained": "HuggingFaceTB/SmolLM2-135M", "corpus": "smollm2", "width": 576, "vocab_size": 49152,
                     "micro_batch": {"frozen": 16, "lora": 8}, "eval_batch": 32},
    "SmolLM2-360M": {"pretrained": "HuggingFaceTB/SmolLM2-360M", "corpus": "smollm2", "width": 960, "vocab_size": 49152,
                     "micro_batch": {"frozen": 8, "lora": 4}, "eval_batch": 16},
    "Qwen2.5-0.5B": {"pretrained": "Qwen/Qwen2.5-0.5B", "corpus": "qwen2.5", "width": 896, "vocab_size": 151936,
                     "micro_batch": {"frozen": 4, "lora": 2}, "eval_batch": 8},
    # Qwen3 base hosts (E9 on modern hosts; the three share one tokenizer, fingerprint a63080b4…, which is Qwen2.5's
    # byte-level BPE plus 4 added tokens, so they get their own corpora). Tied 151,936-row embeddings (the chunked LM
    # loss never builds full logits); LoRA targets q/k/v/o/gate/up/down_proj (`add_lora` finds 7 per layer). Micro-
    # batches are provisional (memory estimates at sequence 1024, fp32 host, bf16 autocast) until the E9 memory
    # probe (`e9_memory`) measures them; Qwen3-4B uses gradient checkpointing (non-reentrant, so a LoRA C0′ trains).
    "Qwen3-0.6B-Base": {"pretrained": "Qwen/Qwen3-0.6B-Base", "corpus": "qwen3", "width": 1024, "vocab_size": 151936,
                        "micro_batch": {"frozen": 8, "lora": 4}, "eval_batch": 8},
    "Qwen3-1.7B-Base": {"pretrained": "Qwen/Qwen3-1.7B-Base", "corpus": "qwen3", "width": 2048, "vocab_size": 151936,
                        "micro_batch": {"frozen": 4, "lora": 2}, "eval_batch": 4},
    "Qwen3-4B-Base": {"pretrained": "Qwen/Qwen3-4B-Base", "corpus": "qwen3", "width": 2560, "vocab_size": 151936,
                      "micro_batch": {"frozen": 2, "lora": 1}, "eval_batch": 2,
                      "model": {"gradient_checkpointing": True, "checkpoint_use_reentrant": False}},
}
BASE = {
    "model": {"size": "pretrained", "seq_len": 1024, "lora_rank": 16, "gradient_checkpointing": False},
    "train": {"lr": 1.0e-3, "min_lr_ratio": 0.1, "warmup_tokens": 2_500_000, "weight_decay": 0.1,
              "log_every": 10, "checkpoint_minutes": 10, "save_trainable_only": True},
    "data": {"min_subtokens": 2, "seed": 1234},
    "eval": {"windows": 1024, "first_tokens": 2_500_000},
    "device": "cuda",
}
LORA_HOST_LR = 2.0e-4      # LoRA adapters; the channel keeps train.lr
STAGES: dict[str, dict[str, Any]] = {
    "cpt-pilot": {"hosts": ["SmolLM2-135M"], "modes": ["lora"], "conditions": ["C0p", "C2", "C5"], "seeds": [1],
                  "total_tokens": 25_000_000, "priority": 11, "lora_check": False},
    "cpt": {"hosts": ["SmolLM2-135M", "SmolLM2-360M", "Qwen2.5-0.5B"], "modes": ["frozen"],
            "conditions": ["C0p", "C1", "C2", "best"], "seeds": [1, 2, 3], "total_tokens": 100_000_000,
            "priority": 100, "lora_check": True},
}


def condition_name(name: str) -> str:
    """C0' is written C0p in file and job names."""
    return "C0p" if name in {"C0'", "C0p"} else name


def channel_spec(name: str, *, best: str | None, table: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    """(label, config fragment) of a condition; None for `best` before G3."""
    name = condition_name(name)
    if name == "C0p":
        return "C0p", {"channel": {"mode": "none"}}
    if name == "best":
        if best is None:
            return None
        if best not in table:
            raise ValueError(f"unknown best condition {best!r}; choose one of {sorted(table)}")
        return f"best-{best}", table[best]
    if name not in table:
        raise ValueError(f"unknown condition {name!r}")
    return name, table[name]


def _merge(config: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if isinstance(value, dict):
            config.setdefault(key, {})
            config[key].update(copy.deepcopy(value))
        else:
            config[key] = value


def run_config(*, stage: str, host: str, mode: str, label: str, spec: dict[str, Any], seed: int, data_root: Path,
               total_tokens: int, sequences_per_step: int, save_window_losses: bool,
               overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = HOSTS[host]
    micro = settings["micro_batch"][mode]
    if sequences_per_step % micro:
        raise ValueError(f"{sequences_per_step} sequences per step is not a multiple of micro-batch {micro}")
    config = copy.deepcopy(BASE)
    _merge(config, {
        "model": {"pretrained": settings["pretrained"], "host_mode": mode, "vocab_size": settings["vocab_size"],
                  **settings.get("model", {})},      # host-specific model keys (Qwen3-4B: gradient checkpointing)
        "train": {"micro_batch": micro, "grad_accum": sequences_per_step // micro, "total_tokens": total_tokens},
        "eval": {"batch": settings["eval_batch"]},
    })
    if save_window_losses:
        config["eval"]["save_window_losses"] = True   # honoured by trainers that implement it
    if mode == "lora":
        config["train"]["host_lr"] = LORA_HOST_LR
    _merge(config, spec)
    if mode == "frozen" and config["channel"]["mode"] == "none":
        config["train"]["eval_only"] = True             # C0' on a frozen host: nothing to train
    _merge(config, overrides or {})
    config["seed"] = seed
    config["data"].update(train=str(data_root / "train"), eval=str(data_root / "eval"), ontology=str(data_root / "ontology.pt"))
    config["experiment"] = f"e4-{stage}-{host}-{mode}-{label}-s{seed}"
    return config


def write_stage(stage: str, *, hosts: list[str], modes: dict[str, list[str]], names: list[str], seeds: list[int],
                total_tokens: int, best: str | None = None, operator: str = "hrr", key_dimension: int = 8,
                developmental: dict[str, Any] | None = None, sequences_per_step: int = 128,
                save_window_losses: bool = True, data_roots: dict[str, Path] | None = None,
                counts_ontology: Path | None = None, overrides: dict[str, Any] | None = None,
                root: Path = ROOT) -> list[Path]:
    """Write the configs of one stage; `modes` maps host → host modes; returns the config paths.

    The capacity-matched C2 width and C1h buckets need the ontology's entry/atomic/relation counts,
    which every host shares with C3 (same entry ids): the host ontology is used if it exists,
    otherwise `counts_ontology` (default: C3's)."""
    out = root / "configs" / stage
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for host in hosts:
        corpus = HOSTS[host]["corpus"]
        data_root = (data_roots or {}).get(corpus, default_paths(corpus)[0])
        counts = data_root / "ontology.pt" if (data_root / "ontology.pt").exists() else (counts_ontology or C3_ONTOLOGY)
        free_dim, buckets = matched_sizes(counts, HOSTS[host]["width"], 256)
        table = conditions(operator, key_dimension, 256, free_dim, buckets, developmental or {})
        for mode in modes[host]:
            for name in names:
                found = channel_spec(name, best=best, table=table)
                if found is None:
                    continue
                label, spec = found
                for seed in seeds:
                    config = run_config(stage=stage, host=host, mode=mode, label=label, spec=spec, seed=seed,
                                        data_root=data_root, total_tokens=total_tokens,
                                        sequences_per_step=sequences_per_step, save_window_losses=save_window_losses,
                                        overrides=overrides)
                    path = out / f"{host}-{mode}-{label}-s{seed}.yaml"
                    path.write_text(yaml.safe_dump(config, sort_keys=False))
                    paths.append(path)
    return paths


def stage_modes(stage: str, hosts: list[str], *, large_host_mode: str = "frozen", lora_check: bool | None = None) -> dict[str, list[str]]:
    """Host modes per host: the stage's modes (the larger hosts take `large_host_mode` in `cpt`),
    plus LoRA on SmolLM2-135M for the LoRA-vs-frozen check."""
    spec = STAGES[stage]
    modes = {}
    for host in hosts:
        base = list(spec["modes"]) if host == "SmolLM2-135M" or stage != "cpt" else [large_host_mode]
        if (spec["lora_check"] if lora_check is None else lora_check) and host == "SmolLM2-135M" and "lora" not in base:
            base.append("lora")
        modes[host] = base
    return modes


def queue_jobs(paths: list[Path], stage: str, priority: int, *, root: Path = ROOT) -> list[str]:
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    queued = []
    for path in paths:
        name = f"{stage}-{path.stem}"
        output = root / "runs" / stage / path.stem
        try:
            add(DEFAULT_DIR, [sys.executable, "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(output)],
                name=name, priority=priority, min_free_gb=20, env={"PYTHONPATH": "src"})
            queued.append(name)
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--hosts", nargs="+", default=None, choices=sorted(HOSTS))
    parser.add_argument("--conditions", nargs="+", default=None, help="C0' (or C0p), C1, C2, C3…C6, best")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--best-condition", default=None, help="best channel condition from G3 (e.g. C5)")
    parser.add_argument("--large-host-mode", default="frozen", choices=["frozen", "lora"],
                        help="host mode of SmolLM2-360M and Qwen2.5-0.5B in `cpt` (after the LoRA check)")
    parser.add_argument("--no-lora-check", action="store_true", help="omit the SmolLM2-135M LoRA runs of `cpt`")
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--sequences-per-step", type=int, default=128, help="sequences of 1024 tokens per optimizer step")
    parser.add_argument("--operator", default="hrr"); parser.add_argument("--key-dimension", type=int, default=8)
    parser.add_argument("--developmental", default="{}")
    parser.add_argument("--no-window-losses", action="store_true", help="do not set eval.save_window_losses")
    parser.add_argument("--smollm2-root", type=Path, default=None); parser.add_argument("--qwen-root", type=Path, default=None)
    parser.add_argument("--qwen3-root", type=Path, default=None)
    parser.add_argument("--overrides", default="{}", help="JSON merged into every config")
    parser.add_argument("--queue", action="store_true"); parser.add_argument("--priority", type=int, default=None)
    args = parser.parse_args(argv)
    spec = STAGES[args.stage]
    hosts = args.hosts or spec["hosts"]
    names = [condition_name(n) for n in (args.conditions or spec["conditions"])]
    if "best" in names and args.best_condition is None:
        print("note: `best` is fixed at G3; its configs are written once --best-condition is given")
    roots = {k: v for k, v in (("smollm2", args.smollm2_root), ("qwen2.5", args.qwen_root), ("qwen3", args.qwen3_root))
             if v is not None}
    paths = write_stage(args.stage, hosts=hosts,
                        modes=stage_modes(args.stage, hosts, large_host_mode=args.large_host_mode,
                                          lora_check=False if args.no_lora_check else None),
                        names=names, seeds=args.seeds or spec["seeds"], total_tokens=args.total_tokens or spec["total_tokens"],
                        best=args.best_condition, operator=args.operator, key_dimension=args.key_dimension,
                        developmental=json.loads(args.developmental), sequences_per_step=args.sequences_per_step,
                        save_window_losses=not args.no_window_losses, data_roots=roots, overrides=json.loads(args.overrides))
    print("\n".join(str(p) for p in paths))
    if args.queue:
        queued = queue_jobs(paths, args.stage, args.priority if args.priority is not None else spec["priority"])
        print(f"queued {len(queued)} job(s)")


if __name__ == "__main__":
    main()
