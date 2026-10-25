#!/usr/bin/env python
"""Deterministic part of the prelim-results skill: what the queue is doing, what landed, what is evaluable early.

Prints markdown. Reads only: the job queue (.jobs/*.json), report folders, run folders' metrics.jsonl, and `df`.
Run from the repository root with the pinned env:

    PYTHONPATH=src python .claude/skills/prelim-results/scripts/prelim_status.py --since 36

Options:
    --since HOURS   reports and run folders modified within the window count as "new" (default 36)
    --block REGEX   restrict the per-block table to job names matching REGEX
    --json PATH     also write the raw data as JSON (for a follow-up pass)
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "experiments"

# Job-name prefixes → block label. Order matters: first match wins. Extend when a new experiment is queued.
BLOCKS = [
    (r"^e15-t5-.*-(xl|long)", "E15 stage 1b/1d (long runs)"),
    (r"^e15-t5-.*(RD|FV)", "E15 stage 1c (aligned objectives)"),
    (r"^e15-", "E15 stage 1 (HRR as target)"),
    (r"^e14-(rd4e)", "E14 RD-4e routing"),
    (r"^e14-(treetext|rd13|lgcf|rd14)", "E14 addenda 9–11"),
    (r"^e14-", "E14 other"),
    (r"^e13-t5", "E13 cycle, T5"),
    (r"^e13-t7", "E13 cycle, T7-ROOD"),
    (r"^t7rood", "T7-ROOD"),
    (r"^t8", "T8 Wikidata"),
    (r"^t2-", "T2 developer tools"),
    (r"C0L|bb1", "B-B1 side-module control"),
    (r"^t1c", "T1c clinical (aggregates only)"),
    (r"^e5s1", "E5 on 125M"),
    (r"^e7-", "E7 self-authoring"),
    (r"^learn", "E10 learn tool"),
    (r"edit-v2|-int4$|rome|memit|alphaedit", "E9 editing baselines"),
    (r"quant|-pq", "E9 quantization"),
    (r"incontext", "E9 in-context glossary"),
    (r"qwen3", "E9 Qwen3 hosts"),
    (r"^t[457]-", "E9 core tracks (T4/T5/T7)"),
]
STATUSES = ("done", "running", "pending")


def jobs() -> list[dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from vsa_embed import jobqueue  # type: ignore

    return jobqueue.jobs(jobqueue.DEFAULT_DIR)


def block_of(name: str) -> str:
    for pat, label in BLOCKS:
        if re.search(pat, name):
            return label
    return "other"


def queue_table(J: list[dict], only: str | None) -> tuple[str, dict]:
    by: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for j in J:
        if only and not re.search(only, j["name"]):
            continue
        by[block_of(j["name"])][j["status"]] += 1
    lines = ["| block | done | running | pending | other |", "|---|---:|---:|---:|---:|"]
    data = {}
    for label, c in sorted(by.items(), key=lambda kv: (-kv[1]["running"], -kv[1]["pending"], kv[0])):
        other = sum(v for k, v in c.items() if k not in STATUSES)
        lines.append(f"| {label} | {c['done']} | {c['running']} | {c['pending']} | {other} |")
        data[label] = dict(c)
    return "\n".join(lines), data


def running_and_failed(J: list[dict]) -> str:
    out = []
    run = [j for j in J if j["status"] == "running"]
    out.append("Running: " + (", ".join(f"`{j['name']}` (since {(j.get('started') or '')[5:16]})" for j in run) or "nothing"))
    bad = [j for j in J if j["status"] not in STATUSES]
    out.append("Failed / other status: " + (", ".join(f"`{j['name']}` ({j['status']})" for j in bad[:12]) or "none"))
    pend = sorted([j for j in J if j["status"] == "pending"], key=lambda j: float(j["priority"]))
    out.append("Next in line: " + ", ".join(f"`{j['name']}` @{j['priority']}" for j in pend[:4]))
    return "\n".join(out)


def new_reports(since_h: float) -> list[tuple[str, str]]:
    cutoff = time.time() - since_h * 3600
    found = []
    for p in EXPERIMENTS.glob("*/report/**/report.md"):
        if p.stat().st_mtime >= cutoff:
            found.append((time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)), str(p.relative_to(ROOT))))
    for p in EXPERIMENTS.glob("*/analysis/**/report.md"):
        if p.stat().st_mtime >= cutoff:
            found.append((time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)), str(p.relative_to(ROOT))))
    return sorted(found)


def final_eval(run: Path) -> dict | None:
    """Last evaluation row group of a run's metrics.jsonl: {stratum: loss}, or None."""
    m = run / "metrics.jsonl"
    if not m.exists():
        return None
    rows = []
    with m.open() as f:
        for line in f:
            if line.strip():
                try:
                    x = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if x.get("type") == "eval":
                    rows.append(x)
    if not rows:
        return None
    last = max(x["step"] for x in rows)
    return {x["stratum"]: x["loss"] for x in rows if x["step"] == last}


def finished_runs_without_report(since_h: float, J: list[dict]) -> list[str]:
    """Run folders with a final.pt newer than the window whose block has no `-report` job done yet."""
    cutoff = time.time() - since_h * 3600
    done_reports = {j["name"] for j in J if j["status"] == "done" and "-report" in j["name"]}
    out = []
    for fp in EXPERIMENTS.glob("*/runs/**/final.pt"):
        if fp.stat().st_mtime < cutoff:
            continue
        run = fp.parent
        # skip runs whose experiment already has a report newer than the run: those are read in §2, not here
        exp = EXPERIMENTS / fp.relative_to(EXPERIMENTS).parts[0]
        reports = list((exp / "report").glob("**/report.md")) + list((exp / "analysis").glob("**/report.md"))
        if any(r.stat().st_mtime > fp.stat().st_mtime for r in reports):
            continue
        rel = str(run.relative_to(ROOT))
        ev = final_eval(run)
        strata = ", ".join(f"{k} {v:.4f}" for k, v in ev.items() if k in ("after_heldout", "after", "all")) if ev else "no eval rows"
        out.append(f"- `{rel}` — final eval: {strata}")
    return out[:40]


def disk() -> str:
    du = shutil.disk_usage("/")
    return f"{du.free / 1e9:.0f} GB free of {du.total / 1e9:.0f} GB ({100 * du.used / du.total:.0f}% used)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=36)
    ap.add_argument("--block", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    J = jobs()
    table, data = queue_table(J, a.block)
    reports = new_reports(a.since)
    runs = finished_runs_without_report(a.since, J)
    print(f"# Preliminary status, {time.strftime('%Y-%m-%d %H:%M')} (window {a.since:g} h)\n")
    print(f"Queue: {collections.Counter(j['status'] for j in J)}; disk {disk()}\n")
    print(running_and_failed(J) + "\n")
    print("## Jobs by block\n\n" + table + "\n")
    print(f"## Reports landed in the window ({len(reports)})\n")
    print("\n".join(f"- {t}  `{p}`" for t, p in reports) or "- none")
    print(f"\n## Finished runs in the window (candidates for an early read before their report)\n")
    print("\n".join(runs) or "- none")
    if a.json:
        Path(a.json).write_text(json.dumps({"queue": data, "reports": reports, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
