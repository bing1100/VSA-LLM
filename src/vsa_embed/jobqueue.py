"""Local GPU job queue (B11): one GPU job at a time, priority-ordered, resumable after interruption.

Jobs are JSON files under the queue directory (default `.jobs/`, git-ignored). `run` executes the
highest-priority pending job, streams its output to a log file, and records the outcome. A job
left in `running` by a dead runner (crash, reboot) is re-queued; its `resume_args` are appended
when it runs again, so trainers continue from their last checkpoint. A job starts only if the
disk has at least `min_free_gb` free.

Lanes (opt-in): a job is in the `cpu` lane if its `lane` field says so or its name matches
`--cpu-pattern`, else in the `gpu` lane. `run --lane gpu` skips CPU-lane jobs, so CPU-only jobs
(reports) no longer hold the GPU; `run --lane cpu` starts a CPU-lane job only once every job ahead of
it in priority order is done (its inputs). The default `--lane all` is the original single runner.

    vsa-queue add --name e4-50m-c0-s1 --priority 10 -- python -m vsa_embed.training.lm --config … --output …
    vsa-queue status
    vsa-queue run [--once] [--lane all|gpu|cpu] [--cpu-pattern REGEX]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(".jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, job: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, indent=2) + "\n")
    temporary.replace(path)


def add(queue: Path, command: list[str], *, name: str, priority: int = 100, cwd: str | None = None,
        resume_args: list[str] | None = None, min_free_gb: float = 10.0, env: dict[str, str] | None = None,
        lane: str | None = None) -> Path:
    queue.mkdir(parents=True, exist_ok=True)
    path = queue / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"job {name!r} already exists")
    _write(path, {"name": name, "command": command, "priority": priority, "cwd": cwd or os.getcwd(),
                  "resume_args": resume_args if resume_args is not None else ["--resume"], "env": env or {},
                  "min_free_gb": min_free_gb, "status": "pending", "attempts": 0, "created": _now(),
                  "log": str(queue / f"{name}.log"), **({"lane": lane} if lane else {})})
    return path


def jobs(queue: Path) -> list[dict[str, Any]]:
    return sorted((json.loads(p.read_text()) for p in queue.glob("*.json")), key=lambda j: (j["priority"], j["created"]))


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def lane_of(job: dict[str, Any], cpu_pattern: str | None = None) -> str:
    if job.get("lane"):
        return job["lane"]
    return "cpu" if cpu_pattern and re.search(cpu_pattern, job["name"]) else "gpu"


def _candidates(queue: Path, lane: str, cpu_pattern: str | None) -> list[dict[str, Any]]:
    everything = jobs(queue)
    pending = [j for j in everything if j["status"] == "pending"
               and (lane == "all" or lane_of(j, cpu_pattern) == lane)]
    if lane == "cpu":  # inputs first: every job ahead in priority order must be done
        pending = [j for j in pending
                   if all(o["status"] == "done" for o in everything if o["priority"] < j["priority"])]
    return pending


def recover(queue: Path) -> list[str]:
    """Re-queue jobs marked running whose process is gone."""
    recovered = []
    for job in jobs(queue):
        if job["status"] == "running" and not _alive(job.get("pid")):
            job.update(status="pending", interrupted=True, pid=None)
            _write(queue / f"{job['name']}.json", job); recovered.append(job["name"])
    return recovered


def run_next(queue: Path, *, lane: str = "all", cpu_pattern: str | None = None) -> dict[str, Any] | None:
    recover(queue)
    pending = _candidates(queue, lane, cpu_pattern)
    if not pending:
        return None
    job = pending[0]
    path = queue / f"{job['name']}.json"
    free_gb = shutil.disk_usage(job["cwd"]).free / 2**30
    if free_gb < job["min_free_gb"]:
        job.update(status="blocked", reason=f"only {free_gb:.1f} GB free"); _write(path, job)
        return job
    command = list(job["command"]) + (list(job["resume_args"]) if job.get("interrupted") or job["attempts"] else [])
    job.update(status="running", started=_now(), attempts=job["attempts"] + 1)
    with open(job["log"], "a") as log:
        log.write(f"\n=== attempt {job['attempts']} {job['started']}: {' '.join(command)}\n"); log.flush()
        process = subprocess.Popen(command, cwd=job["cwd"], stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, **job["env"]}, start_new_session=True)
        job["pid"] = process.pid; _write(path, job)
        code = process.wait()
    job = json.loads(path.read_text())
    job.update(status="done" if code == 0 else "failed", returncode=code, finished=_now(), pid=None)
    _write(path, job)
    return job


def run(queue: Path, *, once: bool = False, poll_seconds: float = 30.0, lane: str = "all",
        cpu_pattern: str | None = None) -> None:
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        job = run_next(queue, lane=lane, cpu_pattern=cpu_pattern)
        if once:
            return
        if job is None:
            time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path, default=DEFAULT_DIR)
    sub = parser.add_subparsers(dest="action", required=True)
    add_parser = sub.add_parser("add")
    add_parser.add_argument("--name", required=True); add_parser.add_argument("--priority", type=int, default=100)
    add_parser.add_argument("--min-free-gb", type=float, default=10.0)
    add_parser.add_argument("--no-resume", action="store_true", help="do not append --resume on retries")
    add_parser.add_argument("--lane", choices=("gpu", "cpu"), default=None)
    add_parser.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    run_parser = sub.add_parser("run"); run_parser.add_argument("--once", action="store_true")
    run_parser.add_argument("--lane", choices=("all", "gpu", "cpu"), default="all")
    run_parser.add_argument("--cpu-pattern", default=None, help="regex: matching job names are CPU-lane jobs")
    retry_parser = sub.add_parser("retry"); retry_parser.add_argument("name")
    args = parser.parse_args(argv)
    if args.action == "add":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        print(add(args.queue, command, name=args.name, priority=args.priority, min_free_gb=args.min_free_gb,
                  resume_args=[] if args.no_resume else None, lane=args.lane))
    elif args.action == "status":
        for job in jobs(args.queue):
            print(f"{job['status']:8} p{job['priority']:<4} {job['name']:40} attempts={job['attempts']} "
                  f"{job.get('returncode', '')}")
    elif args.action == "retry":
        path = args.queue / f"{args.name}.json"
        job = json.loads(path.read_text()); job.update(status="pending", interrupted=True); _write(path, job)
    else:
        run(args.queue, once=args.once, lane=args.lane, cpu_pattern=args.cpu_pattern)


if __name__ == "__main__":
    main(sys.argv[1:])
