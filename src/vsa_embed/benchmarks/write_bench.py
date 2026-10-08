"""Toolkit *write* benchmarks on existing E9 checkpoints (decision 63, WP TK-B1; pre-registration
`experiments/toolkit-bench/preregistration.md`): COMPS-WUGS and ALCUNA on the WordNet-track runs.

One GPU job per run × set (`vsa_embed.benchmarks.ranking evaluate`), each run with the conditions its channel
supports (`conditions_for`), then one CPU-lane report job (`ranking report` with the pre-registered contrasts of
`contrast_spec`, Holm over the primaries W1–W2). Nothing is queued by `plan`; `queue` adds the jobs through the job
queue's Python API (fractional priorities: GPU jobs 54.498, the report 54.4981, whose name ends in `-report` for the
CPU lane).

    python -m vsa_embed.benchmarks.write_bench smoke --runs RUNS_ROOT [--limit 40] [--threads 4]   (CPU, SMOKE-labelled)
    python -m vsa_embed.benchmarks.write_bench cost                    (smoke/cost.json from the smoke folders)
    python -m vsa_embed.benchmarks.write_bench plan [--write experiments/toolkit-bench/queue-commands.sh]
    python -m vsa_embed.benchmarks.write_bench contrasts --output experiments/toolkit-bench/contrasts.json
    python -m vsa_embed.benchmarks.write_bench queue [--dry-run]        (adds the jobs; run by the author)
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Sequence

ROOT = Path("experiments/toolkit-bench")
ITEMS = ROOT / "items"
RUNS = Path("experiments/e9-retrofit/runs/wordnet")
HOSTS = ("SmolLM2-360M", "SmolLM2-135M")
MODELS = ("C5", "C2", "C0p", "P0")
SETS = {"comps-wugs": ITEMS / "comps-wugs-wordnet-v1" / "items.jsonl.gz", "alcuna": ITEMS / "alcuna-wordnet-v1" / "items.jsonl.gz"}
GPU_PRIORITY, REPORT_PRIORITY = 54.498, 54.4981
CONDITIONS = {
    "C5": ["none", "channel-off", "store:oracle", "store:linker", "store:typeprior", "store:random", "store:oracle:parent",
           "frame-in-context:oracle", "definition-in-context", "store:oracle+definition-in-context"],
    "C2": ["none", "channel-off", "store:oracle", "store:random", "frame-in-context:oracle", "definition-in-context"],
    "C0p": ["none", "frame-in-context:oracle", "definition-in-context"],
    "P0": ["none", "frame-in-context:oracle", "definition-in-context"],
}
# ALCUNA definitions are long property lists: the linker would score ≈ 10² single-edge candidates over ≈ 10² tokens per
# entity, and no pre-registered contrast uses it there (preregistration §11.1).
SET_DROP = {"alcuna": frozenset({"store:linker", "store:oracle:parent"})}   # ALCUNA items have no parent-entry frames
BATCH, TOKEN_BUDGET = 256, 32768            # GPU jobs: texts per batch (short COMPS texts) under a token budget (long prompts)
# GPU-hour model (`plan`, `job_hours`): per item, the padded forward tokens, unique scored texts and CPU-side seconds
# (tokenizing, linking, prompts, readers' bookkeeping) measured in the CPU smoke (SmolLM2-135M seed 1, 40 items per set
# spread over the file; experiments/toolkit-bench/smoke/cost.json), with an assumed idle RTX 3090:
# - forward throughput 360M ≈ 39k tokens/s (3× the measured 13.0k training tokens/s of the E9 WordNet runs), 135M ≈ 80k
#   tokens/s (execution.md, E10 reflection);
# - a fixed cost per batch (kernel launches, ~30 layers) of 50 / 35 ms;
# × SAFETY, + 2 min loading.
GPU_TOKENS_PER_S = {"SmolLM2-360M": 39_000.0, "SmolLM2-135M": 80_000.0}
GPU_BATCH_SECONDS = {"SmolLM2-360M": 0.05, "SmolLM2-135M": 0.035}
SAFETY = 1.5


def conditions_for(model: str, set_name: str) -> list[str]:
    return [c for c in CONDITIONS[model] if c not in SET_DROP.get(set_name, ())]


def run_dir(host: str, model: str, seed: int = 1, runs: Path = RUNS) -> Path:
    mode = "frozen" if model == "P0" else "full"
    return runs / f"{host}-{mode}-{model}-s{seed}"


def output_dir(run: Path, set_name: str) -> Path:
    return run / f"toolkit-bench-{set_name}"


def label(host: str, model: str) -> str:
    return f"{host}-{model}"


def contrast_spec(hosts: Sequence[str] = HOSTS) -> dict[str, list[dict[str, Any]]]:
    """The pre-registered contrasts (preregistration §6): primaries W1–W2 (Holm), then the secondaries."""
    big, small = label(hosts[0], "C5"), label(hosts[-1], "C5")

    def c(name: str, a: str, ca: str, b: str, cb: str, set_name: str, **where: Any) -> dict[str, Any]:
        return {"name": name, "set": set_name, "where": where, "a": {"model": a, "condition": ca}, "b": {"model": b, "condition": cb}}

    primary = [c("W1", big, "store:oracle", big, "none", "comps-wugs"), c("W2", big, "store:linker", big, "none", "comps-wugs")]
    secondary = [
        c("S1a ALCUNA-MC oracle store", big, "store:oracle", big, "none", "alcuna-mc"),
        c("S1b ALCUNA-Bool oracle store", big, "store:oracle", big, "none", "alcuna-bool"),
        c("S2 content control (oracle − random)", big, "store:oracle", big, "store:random", "comps-wugs"),
        c("S3 random frame − none", big, "store:random", big, "none", "comps-wugs"),
        c("S4 channel off − none", big, "channel-off", big, "none", "comps-wugs"),
        c("S5 exact-mapped pairs (oracle − none)", big, "store:oracle", big, "none", "comps-wugs", exact_both=True),
        c("S6 parent's own frame − none", big, "store:oracle:parent", big, "none", "comps-wugs"),
        c("S7 typeprior − none", big, "store:typeprior", big, "none", "comps-wugs"),
        c("S8 linker − typeprior", big, "store:linker", big, "store:typeprior", "comps-wugs"),
        c("S9 definition in context − none", big, "definition-in-context", big, "none", "comps-wugs"),
        c("S10 frame in context − store", big, "frame-in-context:oracle", big, "store:oracle", "comps-wugs"),
        c("S11 store + definition − definition", big, "store:oracle+definition-in-context", big, "definition-in-context", "comps-wugs"),
        c("S12 C5 store − C2 store (composition vs free table)", big, "store:oracle", label(hosts[0], "C2"), "store:oracle", "comps-wugs"),
        c("S13 C5 definition − C0′ definition", big, "definition-in-context", label(hosts[0], "C0p"), "definition-in-context", "comps-wugs"),
        c("S14 135M replication (oracle − none)", small, "store:oracle", small, "none", "comps-wugs"),
        c("S15 135M replication (linker − none)", small, "store:linker", small, "none", "comps-wugs"),
    ]
    for kind in ("taxonomic", "overlap", "co-occurrence", "random"):
        secondary.append(c(f"S16 {kind} negatives (oracle − none)", big, "store:oracle", big, "none", "comps-wugs",
                           negative_sample_type=kind))
    return {"primary": primary, "secondary": secondary}


def load_smoke(path: Path = ROOT / "smoke" / "cost.json") -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def job_hours(host: str, model: str, set_name: str, items: int, smoke: dict[str, dict[str, Any]]) -> float | None:
    """Idle-GPU hours of one job from the smoke's per-item forward tokens and CPU-side seconds (per condition)."""
    entry = smoke.get(f"{set_name}|{model}")
    if not entry:
        return None
    tokens = entry["padded_tokens_per_item"] * items
    batches = max(entry.get("unique_requests_per_item", 0.0) * items / BATCH, tokens / TOKEN_BUDGET)
    gpu_seconds = tokens / GPU_TOKENS_PER_S[host] + batches * GPU_BATCH_SECONDS[host]
    cpu_seconds = entry["cpu_seconds_per_item"] * items
    return SAFETY * (gpu_seconds + cpu_seconds + 120.0) / 3600.0          # + model loading


def plan_jobs(*, python: str = "$PY", hosts: Sequence[str] = HOSTS, models: Sequence[str] = MODELS, seeds: Sequence[int] = (1,),
              runs: Path = RUNS, smoke: dict[str, dict[str, Any]] | None = None, item_counts: dict[str, int] | None = None
              ) -> list[dict[str, Any]]:
    smoke = load_smoke() if smoke is None else smoke
    counts = item_counts or {name: json.loads((path.parent / "manifest.json").read_text())["counts"]["items"]
                             for name, path in SETS.items() if (path.parent / "manifest.json").exists()}
    jobs, outputs = [], []
    for host in hosts:
        for model in models:
            for seed in seeds:
                run = run_dir(host, model, seed, runs)
                for set_name, items in SETS.items():
                    out = output_dir(run, set_name)
                    outputs.append(out)
                    command = [python, "-m", "vsa_embed.benchmarks.ranking", "evaluate", "--run", str(run), "--items", str(items),
                               "--conditions", ",".join(conditions_for(model, set_name)), "--batch-size", str(BATCH),
                               "--token-budget", str(TOKEN_BUDGET), "--max-length", "2048", "--output", str(out)]
                    jobs.append({"name": f"tk-bench-{set_name}-{host}-{model}-s{seed}", "priority": GPU_PRIORITY, "min_free_gb": 6,
                                 "lane": "gpu", "hours": job_hours(host, model, set_name, counts.get(set_name, 0), smoke),
                                 "command": command})
    jobs.append({"name": "tk-bench-write-report", "priority": REPORT_PRIORITY, "min_free_gb": 1, "lane": "cpu", "hours": 0.0,
                 "command": [python, "-m", "vsa_embed.benchmarks.ranking", "report", "--inputs", *map(str, outputs),
                             "--contrasts", str(ROOT / "contrasts.json"), "--output", str(ROOT / "report")]})
    return jobs


def render_commands(jobs: Sequence[dict[str, Any]]) -> str:
    lines = ["#!/usr/bin/env bash",
             "# Toolkit write benchmarks (decision 63, TK-B1) on the E9 WordNet-track runs: COMPS-WUGS and ALCUNA.",
             "# Pre-registration: experiments/toolkit-bench/preregistration.md. NOT QUEUED by the agent.",
             "# The job queue's CLI takes integer priorities only, so the jobs are added through its Python API:",
             "#   PYTHONPATH=src $PY -m vsa_embed.benchmarks.write_bench queue        (idempotent; --dry-run prints)",
             "# GPU jobs at priority 54.498 (gpu lane), the report at 54.4981 (name ends in -report: cpu lane).",
             "# Hours: idle-GPU estimates from the timed CPU smoke (write_bench.job_hours; experiments/toolkit-bench/smoke/cost.json).",
             "set -euo pipefail", "cd \"$(dirname \"$0\")/../..\"", "PY=${PY:-$HOME/anaconda3/envs/vsa-repro/bin/python}",
             "PYTHONPATH=src $PY -m vsa_embed.benchmarks.write_bench queue", "", "# Jobs added by the line above:"]
    total = 0.0
    for job in jobs:
        hours = job.get("hours")
        total += hours or 0.0
        lines.append(f"#   {job['name']}  priority {job['priority']}  lane {job['lane']}  ≈ {hours:.2f} GPU-h" if hours is not None
                     else f"#   {job['name']}  priority {job['priority']}  lane {job['lane']}")
        lines.append("#     " + " ".join(shlex.quote(c) if c != "$PY" else c for c in job["command"]))
    lines.append(f"# Total ≈ {total:.1f} GPU-h (idle RTX 3090, estimate; seed 1 only — the WordNet track has no seeds 2–3).")
    return "\n".join(lines) + "\n"


def queue(jobs: Sequence[dict[str, Any]], *, queue_dir: Path | None = None, dry_run: bool = False) -> list[str]:
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    added = []
    for job in jobs:
        if dry_run:
            print(job["name"], job["priority"], " ".join(job["command"]))
            continue
        try:
            add(queue_dir or DEFAULT_DIR, job["command"], name=job["name"], priority=job["priority"],
                min_free_gb=job["min_free_gb"], env={"PYTHONPATH": "src"}, resume_args=[], lane=job["lane"])
            added.append(job["name"])
        except FileExistsError:
            pass
    return added


def smoke(*, runs: Path, host: str = "SmolLM2-135M", models: Sequence[str] = MODELS, limit: int = 40, threads: int = 4,
          output: Path = ROOT / "smoke") -> dict[str, dict[str, Any]]:
    """CPU smoke (labelled SMOKE): the first `limit` items of each set on each model of `host`, seed 1; writes the run
    folders and `cost.json` (per item and condition: padded forward tokens and CPU-side seconds) for `job_hours`."""
    from . import ranking
    for model in models:
        for set_name, items in SETS.items():
            out = Path(output) / f"{set_name}-{host}-{model}-s1"
            ranking.main(["evaluate", "--run", str(run_dir(host, model, 1, runs)), "--items", str(items), "--limit", str(limit), "--spread",
                          "--conditions", ",".join(conditions_for(model, set_name)), "--device", "cpu", "--threads", str(threads),
                          "--batch-size", "32", "--resamples", "200", "--smoke", "--output", str(out), "--overwrite"])
    return smoke_cost(output)


def smoke_cost(output: Path = ROOT / "smoke") -> dict[str, dict[str, Any]]:
    """`cost.json` from the smoke run folders: per (set, model), per item over all its conditions, the padded forward
    tokens, unique scored texts, CPU-side seconds (evaluation seconds minus the CPU forward) and CPU forward seconds."""
    cost: dict[str, dict[str, Any]] = {}
    for folder in sorted(Path(output).glob("*-s1")):
        if not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        set_name = next(s for s in SETS if folder.name.startswith(s + "-"))
        model = folder.name.rsplit("-", 2)[-2]
        stats, n = summary["scorer"], summary["summary"]["items"]
        cost[f"{set_name}|{model}"] = {
            "items": n, "conditions": summary["conditions"], "padded_tokens_per_item": stats.get("padded_tokens", 0) / n,
            "forward_tokens_per_item": stats.get("forward_tokens", 0) / n,
            "unique_requests_per_item": stats.get("unique_requests", 0) / n,
            "cpu_seconds_per_item": max(0.0, summary["seconds"] - stats.get("model_seconds", 0.0)) / n,
            "cpu_forward_seconds_per_item": stats.get("model_seconds", 0.0) / n, "wall_seconds": summary["seconds"],
            "host": summary["source"].get("size"), "smoke": bool(summary.get("smoke"))}
    (Path(output) / "cost.json").write_text(json.dumps(cost, indent=2) + "\n")
    return cost


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke"); s.add_argument("--runs", type=Path, default=RUNS); s.add_argument("--limit", type=int, default=40)
    s.add_argument("--threads", type=int, default=4); s.add_argument("--models", nargs="+", default=list(MODELS))
    s.add_argument("--output", type=Path, default=ROOT / "smoke")
    k = sub.add_parser("cost", help="recompute smoke/cost.json from the smoke folders"); k.add_argument("--output", type=Path, default=ROOT / "smoke")
    p = sub.add_parser("plan"); p.add_argument("--write", type=Path, default=None)
    c = sub.add_parser("contrasts"); c.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("queue"); q.add_argument("--dry-run", action="store_true"); q.add_argument("--queue-dir", type=Path, default=None)
    q.add_argument("--python", default=None, help="interpreter in the job commands (default: this one)")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(smoke(runs=args.runs, models=args.models, limit=args.limit, threads=args.threads, output=args.output), indent=1))
        return
    if args.command == "cost":
        print(json.dumps(smoke_cost(args.output), indent=1))
        return
    if args.command == "contrasts":
        args.output.write_text(json.dumps(contrast_spec(), indent=2) + "\n")
        return
    if args.command == "plan":
        text = render_commands(plan_jobs())
        if args.write:
            args.write.write_text(text)
            args.write.chmod(0o755)
        print(text)
        return
    import sys
    added = queue(plan_jobs(python=args.python or sys.executable), queue_dir=args.queue_dir, dry_run=args.dry_run)
    print(f"added {len(added)} jobs")


if __name__ == "__main__":
    main()
