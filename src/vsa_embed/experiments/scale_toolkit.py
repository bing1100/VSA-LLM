"""From-scratch scaling screen, phase 1b (decision 65, author 2026-10-10): the toolkit arms at 50M × 500M tokens, seed 1.

Phase 1a (another package, stage `scale-v1`, priority 54.3) trains C0, C2, C5, HRRAdd, HRRCat and C5sh at 20M / 50M / 125M
on 500M tokens of the C3 corpus (WordNet-linked FineWeb-Edu). Phase 1b adds two arms at 50M, same corpus and recipe, so each
reads against phase 1a's 50M C5 and C0 (pre-registration `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md`):

- **C5dev** (passive learning): C5 plus the developmental dictionary (`vsa_embed.developmental`, trainer key
  `channel.developmental`): every `screen_every` steps the least coherent active atomics are screened, their per-usage
  gradients recorded, and a split is made where the between-usage gain beats a permutation null; near-identical siblings
  are merged back (`consolidate_every`, after `merge_min_age` steps). Settings: `DEVELOPMENTAL` (rationale below).
- **C5teach** (learning from reading, Claude as teacher): C5 on a teacher-built ontology (`vsa_embed.experiments.c3_teacher`)
  — same entries, relations, atomics and holdout as C3; only the edges differ. `C5teach` (recommended) is the hybrid:
  teacher frames for the 5,900 held-out and 5,900 matched trained entries, WordNet elsewhere, plus the reference strata of
  the two sets (phase 1a's C5 / C0 are rescored on them: `C5@teachref`, `C0@teachref`, evaluation only); `C5teachF` is the
  all-teacher ontology. The configs point at teacher ontologies that do not exist until the author approves the spend;
  their queue lines stay commented out.

**Base config.** Phase 1a's `configs/scale-v1/50M-C5-s1.yaml` when it exists (copied, channel untouched); otherwise derived
from `configs/opscreen/50M-C5-s1.yaml` (the frozen D4.0 recipe: 32,768 tokens / step, peak lr 2e-3, warmup 5M, cosine to
0.1×) with 500M tokens and log-spaced evaluations 5M, 10M, …, 320M, 500M (`eval.first_tokens` 5M) on the same 1,024
windows with per-window losses. `base_config` reports which source was used.

**C5dev settings** (`DEVELOPMENTAL`; 500M tokens = 15,258 optimizer steps): the G1 default variant (`route_unobserved:
parent`, `sync_parent`: held-out and other unobserved usages keep the unsplit parent; gates.md, 2026-09-30) on atomics;
E3's WordNet screen (activity above the 25th percentile, coherence < 0.6, min 4 usages / 20 contributions, merge cosine
0.98, ε 0.05) with the LM-scale changes: a screen every 250 steps (≈ 8.2M tokens; 61 rounds, the first after warmup), at
most 16 candidates and 4 splits per round, a stricter test (999 permutations, p < 0.002: ≈ 980 tests over the run, so
≈ 2 false splits expected under the global null instead of ≈ 10 at p < 0.01), a growth budget of 5% of the 8,192 atomics
(≤ 409 splits, ≤ 818 new rows with parent fallback), a 1,000-step cooldown, and consolidation every 1,000 steps for pairs
at least 1,000 steps old (the trainer never consolidated before decision 65; E0 showed the screen alone over-splits).
The overhead is measured by `overhead` (CPU SMOKE at the 50M channel's real size: real C3 batches, the real composer, the
tracker forced to its maximum work: `max_candidates` candidates every round, parent sync timed with the budget used up).

    python -m vsa_embed.experiments.scale_toolkit configs     # configs/scale-v1-toolkit/50M-{C5dev,C5teach,C5teachF,…}-s1.yaml
    python -m vsa_embed.experiments.scale_toolkit overhead --steps 760 --output experiments/e4-small-lm/scale-v1-toolkit/smoke
    python -m vsa_embed.experiments.scale_toolkit smoke --output <scratch>     # tiny C5 / C5dev through the trainer (SMOKE)
    python -m vsa_embed.experiments.scale_toolkit commands > experiments/e4-small-lm/queue-commands-scale-v1-toolkit.sh
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .c3_teacher import C3_ROOT, STORE

STAGE = "scale-v1-toolkit"
ROOT = Path("experiments/e4-small-lm")
SCALE_V1_C5 = ROOT / "configs" / "scale-v1" / "50M-C5-s1.yaml"
OPSCREEN_C5 = ROOT / "configs" / "opscreen" / "50M-C5-s1.yaml"
DERIVED = {"train": {"total_tokens": 500_000_000, "warmup_tokens": 5_000_000},
           "eval": {"first_tokens": 5_000_000, "save_window_losses": True}}
DEVELOPMENTAL: dict[str, Any] = {
    "target": "atomics", "beta": 0.9, "screen_every": 250, "activity_percentile": 0.25, "coherence_threshold": 0.6,
    "max_candidates": 16, "min_usages": 4, "min_contributions": 20, "max_splits_per_round": 4, "growth_budget": 0.05,
    "cooldown": 1000, "epsilon": 0.05, "permutations": 999, "p_value": 0.002, "test": "permutation",
    "route_unobserved": "parent", "sync_parent": True, "merge_cosine": 0.98, "consolidate_every": 1000,
    "merge_min_age": 1000, "freeze_activity": 0.0, "seed": 1,
}
TEACHER_ONTOLOGY = {"full": STORE / "ontology-full.pt", "hybrid": STORE / "ontology-hybrid.pt"}
TEACHER_REFERENCE = STORE / "reference-hybrid-1024x1024.npz"
PHASE_1A_RUNS = ROOT / "runs" / "scale-v1"          # e4_plan's run folder convention (runs/<stage>/<config stem>)
PRIORITY = {"C5dev": 54.301, "C5teach": 54.302, "C5teachF": 54.302, "C5@teachref": 54.302, "C0@teachref": 54.302}
# Opscreen 50M C5 (seeds 1–2): median optimizer-step throughput 67.6k tokens / s, mean wall clock with evaluations and
# checkpoints 66.6k (scale-v1 quotes ≈ 68k: 500M tokens ≈ 2.05 h).
TOKENS_PER_S_50M = 66_600
DEV_OVERHEAD = 0.048          # `overhead` SMOKE (2026-10-10): tracker time as a fraction of the 50M step, forced upper bound


def base_config(root: Path = ROOT, condition: str = "C5") -> tuple[dict[str, Any], str]:
    """Phase 1a's 50M config of `condition` (seed 1) if present, else the opscreen one with the 500M-token schedule."""
    scale = Path(root) / "configs" / "scale-v1" / f"50M-{condition}-s1.yaml"
    if scale.exists():
        return yaml.safe_load(scale.read_text()), str(scale)
    opscreen = Path(root) / "configs" / "opscreen" / f"50M-{condition}-s1.yaml"
    config = yaml.safe_load(opscreen.read_text())
    for key, part in DERIVED.items():
        config.setdefault(key, {}).update(copy.deepcopy(part))
    return config, f"{opscreen} + {json.dumps(DERIVED)}"


def toolkit_configs(base: dict[str, Any], *, c0: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """From the base C5 config (seed 1), changing only what each arm changes:

    - `C5dev`: `channel.developmental` = `DEVELOPMENTAL`;
    - `C5teach` (recommended, hybrid): the hybrid teacher ontology (teacher frames for the 5,900 held-out and 5,900 matched
      trained entries, WordNet elsewhere) and the reference strata of the two teacher entry sets;
    - `C5teachF` (full): the all-teacher ontology;
    - `C5@teachref` / `C0@teachref`: evaluation-only rescoring of phase 1a's finished 50M C5 / C0 (seed 1) on the hybrid
      reference strata (`train.eval_only`, `train.init_from` their `final.pt`), the comparators of C5teach's matched stratum.
    """
    if base["channel"].get("mode") != "compose" or base["channel"].get("developmental"):
        raise ValueError("the base must be a C5 (compose) config without growth")
    out = {}
    dev = copy.deepcopy(base)
    dev["channel"]["developmental"] = copy.deepcopy(DEVELOPMENTAL)
    out["C5dev"] = dev
    teach = copy.deepcopy(base)
    teach["data"]["ontology"] = str(TEACHER_ONTOLOGY["hybrid"])
    teach["eval"]["reference_strata"] = str(TEACHER_REFERENCE)
    out["C5teach"] = teach
    full = copy.deepcopy(base)
    full["data"]["ontology"] = str(TEACHER_ONTOLOGY["full"])
    out["C5teachF"] = full
    for name, source in (("C5@teachref", base), ("C0@teachref", c0)):
        if source is None:
            continue
        rescore = copy.deepcopy(source)
        rescore["train"].update(eval_only=True, init_from=str(PHASE_1A_RUNS / f"50M-{name.split('@')[0]}-s1" / "final.pt"))
        rescore["eval"]["reference_strata"] = str(TEACHER_REFERENCE)
        out[name] = rescore
    for name, config in out.items():
        config["seed"] = 1
        config["experiment"] = f"e4-{STAGE}-50M-{name}-s1"
    return out


def write_configs(root: Path = ROOT) -> tuple[list[Path], dict[str, str]]:
    base, source = base_config(root, "C5")
    c0, c0_source = base_config(root, "C0")
    out = Path(root) / "configs" / STAGE
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, config in toolkit_configs(base, c0=c0).items():
        path = out / f"50M-{name}-s1.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append(path)
    return paths, {"C5": source, "C0": c0_source}


# -- developmental overhead (CPU SMOKE) -----------------------------------------------------------------------------------

def developmental_overhead(*, steps: int, settings: dict[str, Any] | None = None, root: Path = C3_ROOT, width: int = 512,
                           batch: int = 32, length: int = 1024, seed: int = 1, force: bool = True,
                           threads: int | None = None) -> dict[str, Any]:
    """Time the tracker at the 50M channel's real size on CPU: the real C3 ontology and composer (256-d HRR, attentive, P1
    context), real training batches (32 × 1024, ℓ ≥ 2 spans, held-out entries masked), a stand-in host (a frozen random
    embedding table of width 512 and a next-token dot-product loss: the tracker's work depends on the spans and the
    dictionary, not on the host). `force` sets the coherence threshold above 1, so every round screens `max_candidates`
    candidates: an upper bound on the tracker's work. Returns per-step and per-round seconds and the cards."""
    import numpy as np
    import torch

    from ..data.corpus import TokenCorpus, sample_batch
    from ..developmental import DevelopmentalConfig, DevelopmentalDictionary
    from ..training.lm import build_channel, resolve_config
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    settings = dict(settings or DEVELOPMENTAL)
    if force:
        settings["coherence_threshold"] = 1.01
    base, _ = base_config()
    config = resolve_config({**copy.deepcopy(base), "device": "cpu"})
    ontology = torch.load(Path(root) / "ontology.pt", weights_only=False)
    corpus = TokenCorpus.open(Path(root) / "train")
    mask = np.ones(int(ontology["entry_count"]), dtype=bool)
    mask[list(ontology["heldout_entries"])] = False
    channel, context = build_channel(config, ontology, width)
    table = torch.nn.Embedding(int(config["model"]["vocab_size"]), width).requires_grad_(False)
    parameters = [*channel.parameters(), *(context.parameters() if context is not None else [])]
    optimizer = torch.optim.AdamW(parameters, lr=2e-3)
    tracker = DevelopmentalDictionary(channel.composer, optimizer, DevelopmentalConfig(**settings))
    timing: dict[str, list[float]] = {"sample": [], "forward_backward": [], "observe": [], "grow": [], "begin": []}
    rounds, usages = [], []
    for step in range(steps):
        t0 = time.perf_counter()
        ids, spans = sample_batch(corpus, seed=1234, step=step, micro_step=0, batch=batch, length=length, min_subtokens=2,
                                  entry_mask=mask)
        t1 = time.perf_counter()
        tracker.begin()
        t2 = time.perf_counter()
        embeddings = table(ids)
        pooled = context(embeddings) if context is not None else None
        ctx = pooled[spans["batch"], spans["inject"]] if pooled is not None and spans["entry"].numel() else None
        out = channel(embeddings, spans, input_ids=ids, context=ctx)
        loss = -(out[:, :-1] * embeddings[:, 1:]).sum(-1).mean()
        loss.backward()
        t3 = time.perf_counter()
        tracker.observe()
        t4 = time.perf_counter()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        t5 = time.perf_counter()
        records = {v: sum(int(u.numel()) for u, _ in entries) for v, entries in tracker.records.items()}
        cards = tracker.grow()
        t6 = time.perf_counter()
        timing["sample"].append(t1 - t0); timing["begin"].append(t2 - t1); timing["forward_backward"].append(t3 - t2)
        timing["observe"].append(t4 - t3); timing["grow"].append(t6 - t5)
        if tracker.step % settings["screen_every"] == 0:
            rounds.append({"step": tracker.step, "seconds": t6 - t5, "tested": len(records),
                           "contributions": sorted(records.values(), reverse=True)[:5], "events": len(cards),
                           "splits": sum(c["event"] == "split" for c in cards)})
            usages.append(records)
    per_step = {k: float(np.mean(v)) for k, v in timing.items()}
    screen = int(settings["screen_every"])
    tested_rounds = [r for r in rounds if r["tested"]]
    observe_steps = [o for i, o in enumerate(timing["observe"]) if i >= screen]      # steps with candidates recorded
    round_seconds = float(np.mean([r["seconds"] for r in tested_rounds])) if tested_rounds else None
    # Parent sync runs every step over every split so far: time it with the growth budget used up (stand-in links).
    budget = int(settings["growth_budget"] * tracker.initial_count)
    links, tracker.parent_links = tracker.parent_links, [(i, i + 1, i + 2, 3.0, 2.0) for i in range(0, 3 * budget, 3)]
    started = time.perf_counter()
    for _ in range(20):
        tracker.sync_parents()
    sync_seconds = (time.perf_counter() - started) / 20
    tracker.parent_links = links
    tracker_per_step = (float(np.mean(observe_steps or timing["observe"])) + per_step["begin"] + (round_seconds or 0.0) / screen
                        + (sync_seconds if settings.get("route_unobserved") == "parent" and settings.get("sync_parent", True) else 0.0))
    gpu_step = 32_768 / 67_600
    return {"smoke": True, "label": "SMOKE (CPU, stand-in host; tracker timing only)", "steps": steps, "settings": settings,
            "forced": force, "threads": torch.get_num_threads(), "per_step_seconds": per_step,
            "observe_seconds_with_candidates": float(np.mean(observe_steps)) if observe_steps else None,
            "round_seconds": round_seconds, "rounds": rounds, "parent_sync_seconds_at_budget": sync_seconds,
            "tracker_seconds_per_step": tracker_per_step,
            "gpu_step_seconds_50M": gpu_step, "overhead_fraction_upper_bound": tracker_per_step / gpu_step,
            "splits": tracker.splits, "atomics_after": int(channel.composer.atomics.shape[0]),
            "cards": [{k: v for k, v in c.items() if k != "direction"} for c in tracker.cards]}


SMOKE_OVERRIDES = {"model": {"size": "tiny", "seq_len": 256}, "device": "cpu",
                   "train": {"micro_batch": 4, "grad_accum": 1, "total_tokens": 256 * 4 * 120, "warmup_tokens": 256 * 4 * 10,
                             "checkpoint_minutes": 60, "log_every": 10},
                   "eval": {"windows": 32, "batch": 8, "first_tokens": 256 * 4 * 30}}
SMOKE_DEVELOPMENTAL = {"screen_every": 10, "cooldown": 20, "consolidate_every": 20, "merge_min_age": 20,
                       "permutations": 199, "p_value": 0.01, "growth_budget": 0.01}


def smoke(output: Path) -> dict[str, Any]:
    """CPU SMOKE of C5 and C5dev through the trainer at tiny size on the real C3 corpus (the C5dev config with the screen
    scaled to a 120-step run); wall times, the developmental cards and the final strata. Not a result."""
    from ..training.lm import train
    base, _ = base_config()
    configs = toolkit_configs(base)
    out = {}
    for name, config in (("C5", copy.deepcopy(base)), ("C5dev", configs["C5dev"])):
        for key, part in SMOKE_OVERRIDES.items():
            if isinstance(part, dict):
                config.setdefault(key, {}).update(copy.deepcopy(part))
            else:
                config[key] = part
        if name == "C5dev":
            config["channel"]["developmental"].update(SMOKE_DEVELOPMENTAL)
        config["experiment"] = f"SMOKE-{STAGE}-tiny-{name}"
        run_dir = Path(output) / f"tiny-{name}"
        started = time.monotonic()
        result = train(config, run_dir)
        seconds = time.monotonic() - started
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        final = max(r["tokens"] for r in rows if r["type"] == "eval")
        cards = json.loads((run_dir / "cards.json").read_text()) if (run_dir / "cards.json").exists() else []
        (run_dir / "final.pt").unlink(missing_ok=True)
        out[name] = {"steps": result["steps"], "seconds": round(seconds, 1),
                     "final": {r["stratum"]: round(r["loss"], 4) for r in rows if r["type"] == "eval" and r["tokens"] == final},
                     "cards": {e: sum(c["event"] == e for c in cards) for e in ("split", "merge", "freeze")}}
    summary = {"smoke": True, "label": "SMOKE (CPU, tiny model, 120 steps; plumbing only, not a result)", "runs": out}
    (Path(output) / "trainer-smoke.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# -- queue script ---------------------------------------------------------------------------------------------------------

HEADER = """#!/usr/bin/env bash
# From-scratch scaling screen, phase 1b (decision 65, author 2026-10-10): the toolkit arms at 50M x 500M tokens, seed 1
# (pre-registration experiments/e4-small-lm/preregistration-scale-v1-toolkit.md; printed by
# `python -m vsa_embed.experiments.scale_toolkit commands`).
# NOT EXECUTED by the agent that wrote it. Run from the repository root of the main checkout after merging this branch:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python bash <this file>
# Priorities: phase 1a (stage scale-v1) is at 54.3; C5dev at 54.301 runs after phase 1a's 50M C5. C5teach (54.302) is
# commented out until the teacher ontology is built (c3_teacher teach + ontology, Claude spend approved by the author).
# GPU-h: 500M tokens / {tokens:,} tokens/s (opscreen 50M C5, mean wall clock with evaluations and checkpoints) = {c5:.2f} h;
# C5dev x (1 + {overhead:.2f}) for the developmental tracker (CPU SMOKE upper bound, scale-v1-toolkit/smoke/overhead.json).
set -euo pipefail
: "${{PY:?set PY to the pinned interpreter}}"
"""


def _train_line(name: str, hours: float, min_free_gb: int = 20) -> str:
    config = ROOT / "configs" / STAGE / f"50M-{name}-s1.yaml"
    output = ROOT / "runs" / STAGE / f"50M-{name}-s1"
    return (f"PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-{STAGE}-50M-{name}-s1 --priority {PRIORITY[name]} "
            f"--min-free-gb {min_free_gb} -- $PY -m vsa_embed.training.lm --config {config} --output {output}   # ≈ {hours:.2f} GPU-h")


def queue_lines(*, overhead: float = DEV_OVERHEAD, teacher: dict[str, Any] | None = None) -> list[str]:
    """The queue script: C5dev live; the C5teach block commented out (teacher build first, by hand, after approval)."""
    c5 = 500_000_000 / TOKENS_PER_S_50M / 3600
    teacher = teacher or {}
    lines = [HEADER.format(tokens=TOKENS_PER_S_50M, c5=c5, overhead=overhead), "# ---- C5dev (passive learning)",
             _train_line("C5dev", c5 * (1 + overhead)), "",
             "# ---- C5teach (learning from reading; Claude as teacher). NOT QUEUED: the teacher ontology does not exist yet.",
             "# Recommended: the hybrid variant (teacher frames for the 5,900 held-out + 5,900 matched trained entries, WordNet",
             f"# elsewhere; ≈ ${teacher.get('hybrid_usd', float('nan')):.0f} and ≈ {teacher.get('hybrid_hours', float('nan')):.1f} h of "
             "teacher time at 4 workers, projected from the pilot).",
             "# After the author approves the spend, by hand (not a queue job: it calls `claude -p` and spends money; resumable, capped):",
             "#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher teach --scope hybrid --max-usd <approved cap incl. the pilot's spend>",
             "#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher ontology --scope hybrid",
             "#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher reference",
             f"# {_train_line('C5teach', c5)}",
             "# Rescoring of phase 1a's finished 50M C5 / C0 (seed 1) on the hybrid reference strata (evaluation only; ≈ 0.02 GPU-h each):",
             f"# {_train_line('C5@teachref', 0.02, 5)}",
             f"# {_train_line('C0@teachref', 0.02, 5)}",
             f"# Alternative, full variant (all 117,659 synsets; ≈ ${teacher.get('full_usd', float('nan')):.0f}, "
             f"≈ {teacher.get('full_hours', float('nan')):.1f} h): teach / ontology --scope full, then",
             f"# {_train_line('C5teachF', c5)}"]
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("configs", help="write the C5dev, C5teach (hybrid), C5teachF (full) and teachref configs")
    over = sub.add_parser("overhead", help="CPU SMOKE timing of the developmental tracker at the 50M channel's size")
    over.add_argument("--steps", type=int, default=520); over.add_argument("--output", type=Path, required=True)
    over.add_argument("--no-force", action="store_true"); over.add_argument("--threads", type=int, default=None)
    over.add_argument("--screen-every", type=int, default=None)
    tiny = sub.add_parser("smoke", help="CPU SMOKE of C5 / C5dev through the trainer at tiny size")
    tiny.add_argument("--output", type=Path, required=True)
    commands = sub.add_parser("commands", help="print the queue script (never queues)")
    commands.add_argument("--overhead", type=float, default=None, help="default: the measured value in the smoke folder")
    args = parser.parse_args(argv)
    if args.command == "configs":
        paths, sources = write_configs()
        print(json.dumps({"configs": [str(p) for p in paths], "base": sources}, indent=1))
    elif args.command == "overhead":
        settings = dict(DEVELOPMENTAL, **({"screen_every": args.screen_every} if args.screen_every else {}))
        result = developmental_overhead(steps=args.steps, settings=settings, force=not args.no_force, threads=args.threads)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "overhead.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("per_step_seconds", "round_seconds", "tracker_seconds_per_step",
                                                 "overhead_fraction_upper_bound", "splits")}, indent=1))
    elif args.command == "smoke":
        print(json.dumps(smoke(args.output), indent=1))
    else:
        overhead = args.overhead
        if overhead is None:
            path = ROOT / STAGE / "smoke" / "overhead.json"
            overhead = json.loads(path.read_text())["overhead_fraction_upper_bound"] if path.exists() else DEV_OVERHEAD
        projection = ROOT / STAGE / "teacher-pilot" / "projection.json"
        teacher = {}
        if projection.exists():
            p = json.loads(projection.read_text())
            teacher = {"hybrid_usd": p["hybrid"]["usd"], "hybrid_hours": p["hybrid"]["hours"], "full_usd": p["full"]["usd"],
                       "full_hours": p["full"]["hours"]}
        print("\n".join(queue_lines(overhead=overhead, teacher=teacher)))


if __name__ == "__main__":
    main()
