"""E7 run plan (WP-E7): GPU jobs per stage with GPU-hour estimates, queued through `vsa_embed.jobqueue`.

    python -m vsa_embed.experiments.e7_plan --run experiments/e7-self-authoring/runs/general-v1 --stage d71|d72|d73 [--queue]

Stages (priority 60; jobs of a stage are queued in dependency order, the queue runs one at a time):

- `d71` — D7.1 on the GPU: discovery under the three hosts (consumer first), authoring by the three
  hosts (the first runs `link` if it has not run), OLLM-style direct prompting by the three hosts.
  CPU / LLM steps the orchestrator runs afterwards are listed under `then` (Hearst, random frames,
  the Claude teacher, quality, judge items, judge).
- `d72` — D7.2: EntiGraph-style generation, round-0 channel × 3 seeds, verification (self, teacher) ×
  3 seeds, then 7 conditions × 3 seeds. Refuses to queue while the self or teacher proposals are
  missing (`--no-teacher` drops the teacher condition).
- `d73` — D7.3: the 50M cross-authoring runs (none, curated, authored × 3 seeds); `cross-prepare`
  (CPU, builds the GPT-2 corpora) must have run.

Estimates use the C4 throughputs (experiments.md §Compute) — frozen-host training tokens/s; a forward
pass is taken as 2× faster than frozen training (4N vs 2N FLOPs per token); batched sampling rates are
assumptions (`GENERATION`) — and the planned token counts; the plan file records them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vsa_embed.experiments.e7_authoring import HOSTS, _json, load_track
from vsa_embed.experiments.e7_round import CONDITIONS, CROSS_CONDITIONS, round_settings

PRIORITY = 60
TRAIN = {"SmolLM2-135M": 36_000, "SmolLM2-360M": 18_500, "Qwen2.5-0.5B": 16_000, "50M": 70_000}   # tokens/s (C4)
GENERATION = {"SmolLM2-135M": 2_500, "SmolLM2-360M": 1_500, "Qwen2.5-0.5B": 1_200}               # generated tokens/s (assumed)
PROMPT_TOKENS = 450          # few-shot authoring prompt incl. a 400-character context
DIRECT_PROMPT_TOKENS = 350


def _forward(host: str) -> float:
    return 2.0 * TRAIN[host]


def estimates(run: Path, *, teacher: bool = True) -> dict[str, dict[str, float]]:
    """GPU-hours per job kind from the planned token counts."""
    track = load_track(run)
    settings = track["settings"]
    rounds = round_settings(run)
    corpora, authoring, direct = settings["corpora"], settings["authoring"], settings["direct"]
    read, test = float(corpora["read_tokens"]), float(corpora["test_tokens"])
    candidates = float(authoring["candidates"])
    out: dict[str, dict[str, float]] = {"discover": {}, "author": {}, "direct": {}}
    for host in HOSTS:
        out["discover"][host] = 1.33 * read / _forward(host) / 3600
        prompts = candidates * float(authoring["contexts"])
        samples = prompts * float(authoring["samples"])
        out["author"][host] = (samples * PROMPT_TOKENS / _forward(host) + samples * 40 / GENERATION[host]) / 3600
        docs = float(direct["documents"])
        out["direct"][host] = (docs * DIRECT_PROMPT_TOKENS / _forward(host) + docs * direct["max_new_tokens"] / GENERATION[host]) / 3600
    host = rounds["host"]
    evals = 3 * test / _forward(host)
    verification = rounds["verification"]
    verify_tokens = candidates * float(authoring["max_validation"]) * 5 * float(verification["window"])
    discovery_tokens = 1.33 * read
    authoring_tokens = candidates * float(authoring["contexts"]) * float(authoring["samples"]) * (PROMPT_TOKENS + 40)
    extra = (discovery_tokens + authoring_tokens + verify_tokens) / 2          # 2N per token / 4N per training token
    entigraph = rounds["entigraph"]
    out["entigraph"] = {"generate": candidates * entigraph["per_entity"] * entigraph["max_new_tokens"] / GENERATION[entigraph["writer"]] / 3600}
    out["base"] = {"per_seed": (float(rounds["base_tokens"]) / TRAIN[host] + evals) / 3600}
    out["verify"] = {"per_seed_author": verify_tokens / _forward(host) / 3600}
    round_tokens = float(rounds["round_tokens"])
    out["train"] = {c: ((round_tokens + (extra if c in ("cm", "entigraph") else 0)) / TRAIN[host] + evals) / 3600 for c in CONDITIONS}
    cross = rounds["cross"]
    out["cross"] = {"per_run": (float(cross["train_tokens"]) / TRAIN["50M"] + 7 * 4e6 / (2 * TRAIN["50M"])) / 3600}
    out["compute_matched_extra_tokens"] = {"estimate": extra}
    return out


def stage_jobs(run: Path, stage: str, *, seeds: list[int], teacher: bool = True) -> tuple[list[dict[str, Any]], list[str]]:
    """(jobs, follow-up commands) of one stage; a job is {name, command, gpu_hours}."""
    track = load_track(run)
    est = estimates(run, teacher=teacher)
    py = [sys.executable, "-m"]
    authoring = py + ["vsa_embed.experiments.e7_authoring"]
    rounds = py + ["vsa_embed.experiments.e7_round"]
    tag = Path(run).name
    consumer = track["consumer"]
    hosts = [consumer] + [h for h in HOSTS if h != consumer]
    jobs: list[dict[str, Any]] = []
    then: list[str] = []
    if stage == "d71":
        for host in hosts:
            jobs.append({"name": f"e7-{tag}-discover-{host}", "gpu_hours": est["discover"][host],
                         "command": authoring + ["discover", "--run", str(run), "--host", host]})
        for host in hosts:
            jobs.append({"name": f"e7-{tag}-author-{host}", "gpu_hours": est["author"][host],
                         "command": authoring + ["author", "--run", str(run), "--author", host]})
        for host in hosts:
            jobs.append({"name": f"e7-{tag}-direct-{host}", "gpu_hours": est["direct"][host],
                         "command": authoring + ["author", "--run", str(run), "--author", f"direct-{host}"]})
        base = f"PYTHONPATH=src {sys.executable} -m vsa_embed.experiments.e7_authoring"
        then = [f"{base} link --run {run}   # CPU; run after the {consumer} discovery job, before the author jobs",
                f"{base} author --run {run} --author hearst   # CPU",
                f"{base} author --run {run} --author random   # CPU, after the {consumer} author job",
                f"{base} author --run {run} --author teacher  # claude -p (≈ candidates / 8 calls)",
                f"{base} quality --run {run}", f"{base} judge-items --run {run}",
                f"{base} judge --run {run} --runner cli   # claude -p, ≥ 3 calls per item"]
    elif stage == "d72":
        missing = [p for p in (f"proposals/{consumer}.json",) + (("proposals/teacher.json",) if teacher else ())
                   if not (Path(run) / p).exists()]
        if missing:
            raise FileNotFoundError(f"D7.2 needs {missing} first (D7.1); pass --no-teacher to drop the teacher condition")
        conditions = [c for c in CONDITIONS if teacher or c != "teacher"]
        jobs.append({"name": f"e7-{tag}-entigraph", "gpu_hours": est["entigraph"]["generate"],
                     "command": rounds + ["entigraph", "--run", str(run)]})
        for seed in seeds:
            jobs.append({"name": f"e7-{tag}-base-s{seed}", "gpu_hours": est["base"]["per_seed"],
                         "command": rounds + ["base", "--run", str(run), "--seed", str(seed)]})
        for seed in seeds:
            for author in [consumer] + (["teacher"] if teacher else []):
                jobs.append({"name": f"e7-{tag}-verify-{author}-s{seed}", "gpu_hours": est["verify"]["per_seed_author"],
                             "command": rounds + ["verify", "--run", str(run), "--seed", str(seed), "--author", author]})
        for seed in seeds:
            for condition in conditions:
                jobs.append({"name": f"e7-{tag}-round1-{condition}-s{seed}", "gpu_hours": est["train"][condition],
                             "command": rounds + ["train", "--run", str(run), "--seed", str(seed), "--condition", condition]})
        then = [f"PYTHONPATH=src {sys.executable} -m vsa_embed.experiments.e7_report --run {run}"]
    elif stage == "d73":
        if not (Path(track["data_root"]).expanduser() / "cross" / "prepared.json").exists():
            raise FileNotFoundError(f"run `python -m vsa_embed.experiments.e7_round cross-prepare --run {run}` (CPU) first")
        for seed in seeds:
            for condition in CROSS_CONDITIONS:
                jobs.append({"name": f"e7-{tag}-cross-{condition}-s{seed}", "gpu_hours": est["cross"]["per_run"],
                             "command": rounds + ["cross-train", "--run", str(run), "--seed", str(seed), "--condition", condition]})
        then = [f"PYTHONPATH=src {sys.executable} -m vsa_embed.experiments.e7_report --run {run}"]
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return jobs, then


def queue(jobs: list[dict[str, Any]], *, priority: int = PRIORITY, queue_dir: Path | None = None) -> list[str]:
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    queued = []
    for job in jobs:
        try:
            add(queue_dir or DEFAULT_DIR, job["command"], name=job["name"], priority=priority, min_free_gb=20,
                env={"PYTHONPATH": "src"})
            queued.append(job["name"])
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("experiments/e7-self-authoring/runs/general-v1"))
    parser.add_argument("--stage", required=True, choices=["d71", "d72", "d73"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no-teacher", action="store_true")
    parser.add_argument("--queue", action="store_true"); parser.add_argument("--priority", type=int, default=PRIORITY)
    args = parser.parse_args(argv)
    if args.no_teacher and args.stage == "d72":
        from vsa_embed.experiments.e7_round import update_round_settings
        update_round_settings(args.run, {"teacher_condition": False})
    jobs, then = stage_jobs(args.run, args.stage, seeds=args.seeds, teacher=not args.no_teacher)
    total = sum(j["gpu_hours"] for j in jobs)
    plan = {"stage": args.stage, "jobs": jobs, "gpu_hours": total, "then": then, "priority": args.priority,
            "assumptions": {"train_tokens_per_s": TRAIN, "generation_tokens_per_s": GENERATION}}
    _json(Path(args.run) / "plan" / f"{args.stage}.json", plan)
    for job in jobs:
        print(f"{job['gpu_hours']:6.2f} h  {job['name']}")
    print(f"total ≈ {total:.1f} GPU-h ({len(jobs)} jobs)")
    for line in then:
        print("then:", line)
    if args.queue:
        print(f"queued {len(queue(jobs, priority=args.priority))} job(s) at priority {args.priority}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
