"""Paper-readiness audit (WP-PQ2): manifests, compute ledger, holdouts, audits — read-only.

Scans the run folders of every experiment, the local GPU job queue and the frozen holdouts, and
writes a machine-readable record (`--output`) plus Markdown tables on stdout. It never writes into
a run folder or into `.jobs/`.

    python scripts/paper_readiness.py --main /home/bhux/workplace/VSA-LLM --output manuscript/tables/paper_readiness.json

- **Manifests.** Every folder under `experiments/<exp>/` that holds a run artefact (`manifest.json`,
  `metrics.jsonl`, `summary.json`, `results.json`, `resolved_config.yaml`) is a *run folder*. Its
  `manifest.json` is read as written by `vsa_embed.provenance` (`git_sha`, `git_dirty`,
  `git_state_recorded_at`, `schema_version`); item and data manifests (a string `schema`, no git
  state) are listed separately as data cards. Two trees are scanned: `--repo` (this checkout, i.e.
  what is committed on its branch) and `--main` (the main checkout on disk, incl. runs still in
  progress and not yet committed). "Committed" is answered with `git ls-tree` on `--refs`
  (default `HEAD` and `main`) of `--repo`, so no git command runs inside the main checkout.
- **Compute ledger.** `.jobs/*.json` of the main checkout (`vsa_embed.jobqueue`): hours =
  `finished − started` of the last attempt (earlier attempts of interrupted jobs are not timed by
  the queue; they are counted). The queue runs one job at a time, so queue hours are the hours the
  GPU slot was held; jobs whose command hides the GPU (`CUDA_VISIBLE_DEVICES=`) or runs a CPU-only
  module (reports, plans) are counted as CPU-in-queue. Jobs are grouped into blocks by name.
- **Holdouts.** Each track's frozen holdout is recomputed from `holdout_concepts.txt`
  (`tracks.common.names_sha256`: sha256 of the sorted names joined by newlines) and compared with
  every recorded value (run summary, `items/holdout.json`, `expected_holdout_sha256` in configs).
- **Audits.** Contamination and leakage fields of item manifests and track summaries,
  pre-registration files, committed configs, and the reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ARTEFACTS = ("manifest.json", "metrics.jsonl", "summary.json", "results.json", "resolved_config.yaml")
SKIP_DIRS = {"cache", "exchange", "__pycache__", "figures", ".ipynb_checkpoints"}
# Frozen holdouts named in the plan (gates.md, execution.md, R reports): track → (run folder, pinned prefix).
HOLDOUTS = {
    "C3 general (WordNet × FineWeb-Edu)": ("experiments/c3-general-corpus/runs/v2", "7f2462ed"),
    "T1-open clinical (MeSH × PubMed)": ("experiments/t1-open-clinical/runs/v1", "1c477afd"),
    "T2 developer tools": ("experiments/t2-developer-tools/runs/v1", "75ee2217"),
    "T3 product": ("experiments/t3-product-catalogues/runs/v1", None),
    "T4 chemistry": ("experiments/t4-chemistry/runs/v1", "b58e504f"),
    "T5 enterprise glossary": ("experiments/t5-enterprise-glossary/runs/v1", "e7313dce"),
    "T6 legal": ("experiments/t6-legal-regulatory/runs/v1", None),
}
PREREGISTRATIONS = ("experiments/e4-small-lm/preregistration.md", "experiments/e5-explainability/preregistration.md")
CPU_MODULES = {"vsa_embed.experiments.e9_report", "vsa_embed.experiments.e4_report", "vsa_embed.experiments.e4_plan",
               "vsa_embed.experiments.e9_plan", "vsa_embed.experiments.e10_report"}
# Job-name prefix → block (first match wins; checked in order).
BLOCKS = (
    (r"^probes-pilot-", "S0 pilot (50M from scratch): probes"),
    (r"^pilot-50M-", "S0 pilot (50M from scratch): training"),
    (r"^probes-cpt-pilot-", "S0 pilot (SmolLM2-135M CPT): probes"),
    (r"^cpt-pilot-", "S0 pilot (SmolLM2-135M CPT): training"),
    (r"^d2-e2", "E2 mapping × operator frontier"),
    (r"^d1-e1", "E1 contextual composition"),
    (r"^d3-e3", "E3 developmental WordNet"),
    (r"^recipe-", "D4.0 recipe sweep"),
    (r"^e9-check-", "E9 engagement check"),
    (r"^e9-qwen3-memory-probe", "E9 Qwen3 memory probe"),
    (r"^e9-qwen35-memory-probe", "E9 Qwen3.5 memory probe"),
    (r"^e10", "E10"),
    (r"^e7-", "E7 self-authoring"),
    (r"^t5-qwen3-4b-", "E9 T5 Qwen3-4B"),
    (r"^t5-qwen35-", "E9 T5 Qwen3.5"),
    (r"^t5-qwen3-.*-s1(-|$)|^t5-qwen3-(quant|quant-general|report)-s1$", "E9 T5 Qwen3 (1.7B, 0.6B) seed 1"),
    (r"^t5-qwen3-.*-s2(-|$)|^t5-qwen3-(quant|quant-general|report)-s2$", "E9 T5 Qwen3 (1.7B, 0.6B) seed 2"),
    (r"^t5-SmolLM2-.*-s1(-|$)|^t5-(quant|quant-general|report)-s1$", "E9 T5 SmolLM2 seed 1"),
    (r"^t5-SmolLM2-.*-s[23](-|$)|^t5-(quant|quant-general|report)-s2-3$", "E9 T5 SmolLM2 seeds 2–3"),
    (r"^t5-SmolLM2-.*-P0-", "E9 T5 SmolLM2 seed 1"),
    (r"^t4-qwen3", "E9 T4 Qwen3"),
    (r"^t4-", "E9 T4 SmolLM2"),
    (r"^t1-", "E9 T1-open SmolLM2"),
    (r"^wordnet-", "E9 WordNet control SmolLM2"),
    (r"^pq1-|^e9-pq1|-pq1-", "E9 WP-PQ1 controls"),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def git_tree(repo: Path, ref: str, prefix: str = "experiments") -> set[str] | None:
    """Paths committed at `ref` under `prefix` (None if the ref does not exist)."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ref, "--", prefix],
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.splitlines())


def git_ref_sha(repo: Path, ref: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", ref], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_folders(root: Path, experiment: Path) -> list[Path]:
    """Folders under `experiment` holding a run artefact (`ARTEFACTS`), outermost first."""
    found = []
    for dirpath, dirnames, filenames in os.walk(experiment):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        if any(name in filenames for name in ARTEFACTS):
            found.append(Path(dirpath))
    return found


def classify(folder: Path, manifest: dict[str, Any] | None) -> str:
    parts = folder.parts
    if manifest is not None and isinstance(manifest.get("schema"), str) and "git_sha" not in manifest:
        return "data card"
    if "items" in parts:
        return "data card"
    if any(p in {"report", "analysis", "quant", "quant-general"} for p in parts):
        return "analysis"
    if any(p in {"edit", "edit-int4", "zeroshot", "zeroshot-int4"} for p in parts[-1:]):
        return "evaluation"
    if "memory" in parts or "env" in parts:
        return "probe"
    return "run"


def scan_tree(base: Path, committed: dict[str, set[str] | None]) -> dict[str, Any]:
    """Per experiment: run folders with their manifest state and commit status."""
    out: dict[str, Any] = {}
    experiments = base / "experiments"
    if not experiments.is_dir():
        return out
    for experiment in sorted(p for p in experiments.iterdir() if p.is_dir()):
        rows = []
        for folder in run_folders(base, experiment):
            rel = folder.relative_to(base).as_posix()
            manifest_path = folder / "manifest.json"
            manifest = _read_json(manifest_path) if manifest_path.exists() else None
            kind = classify(folder.relative_to(base), manifest)
            row: dict[str, Any] = {"path": rel, "kind": kind, "manifest": manifest is not None,
                                   "resolved_config": (folder / "resolved_config.yaml").exists(),
                                   "metrics": (folder / "metrics.jsonl").exists()}
            if manifest is not None and kind != "data card":
                row.update(git_sha=(manifest.get("git_sha") or None), git_dirty=manifest.get("git_dirty"),
                           git_state_recorded_at=manifest.get("git_state_recorded_at"),
                           schema_version=manifest.get("schema_version"), created_at=manifest.get("created_at"),
                           device=manifest.get("device"))
                if row["git_sha"] is None:
                    row["provenance"] = "no git state (pre-provenance manifest)"
                elif row["git_dirty"] is True:
                    row["provenance"] = "dirty tree at run start"
                elif row["git_dirty"] is False:
                    row["provenance"] = "clean"
                else:
                    row["provenance"] = "git state unknown"
            elif manifest is None:
                row["provenance"] = "manifest missing"
            else:
                row["provenance"] = "data card (no git state)"
                row["schema"] = manifest.get("schema")
            for ref, tree in committed.items():
                row[f"committed@{ref}"] = None if tree is None else f"{rel}/manifest.json" in tree or any(
                    f"{rel}/{name}" in tree for name in ARTEFACTS)
            rows.append(row)
        if rows:
            out[experiment.name] = rows
    return out


def summarize_tree(scan: dict[str, Any], refs: list[str]) -> dict[str, Any]:
    summary = {}
    for experiment, rows in scan.items():
        provenance = Counter(r["provenance"] for r in rows)
        entry = {"folders": len(rows), "kinds": dict(Counter(r["kind"] for r in rows)), "provenance": dict(provenance),
                 "git_shas": sorted({r["git_sha"][:7] for r in rows if r.get("git_sha")}),
                 "missing_resolved_config": sum(1 for r in rows if r["kind"] in {"run", "evaluation", "analysis"}
                                                and not r["resolved_config"]),
                 "dirty": [r["path"] for r in rows if r["provenance"] == "dirty tree at run start"],
                 "manifest_missing": [r["path"] for r in rows if r["provenance"] == "manifest missing"],
                 "no_git_state": [r["path"] for r in rows if r["provenance"] == "no git state (pre-provenance manifest)"]}
        for ref in refs:
            key = f"committed@{ref}"
            values = [r.get(key) for r in rows]
            if any(v is not None for v in values):
                entry[f"uncommitted@{ref}"] = [r["path"] for r in rows if r.get(key) is False]
        summary[experiment] = entry
    return summary


# ------------------------------------------------------------------------------------------- jobs


def _module(command: list[str]) -> str:
    return next((c for c in command if c.startswith("vsa_embed")), command[0] if command else "?")


def job_device(job: dict[str, Any]) -> str:
    command = [str(c) for c in job.get("command", [])]
    env = {**{k: str(v) for k, v in (job.get("env") or {}).items()}}
    hidden = env.get("CUDA_VISIBLE_DEVICES") == "" or any(c == "CUDA_VISIBLE_DEVICES=" for c in command)
    if hidden or _module(command) in CPU_MODULES:
        return "cpu-in-queue"
    return "gpu"


def job_block(name: str) -> str:
    for pattern, block in BLOCKS:
        if re.search(pattern, name):
            return block
    return "other: " + re.split(r"-(?:SmolLM2|Qwen|50M|125M)", name)[0]


def _hours(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600


def ledger(jobs_dir: Path) -> dict[str, Any]:
    jobs = []
    for path in sorted(jobs_dir.glob("*.json")):
        job = _read_json(path)
        if not isinstance(job, dict) or "name" not in job:
            continue
        hours = _hours(job.get("started"), job.get("finished"))
        running = job.get("status") == "running"
        jobs.append({"name": job["name"], "status": job.get("status"), "priority": job.get("priority"),
                     "module": _module(job.get("command", [])), "device": job_device(job), "block": job_block(job["name"]),
                     "attempts": job.get("attempts", 0), "interrupted": bool(job.get("interrupted")),
                     "started": job.get("started"), "finished": job.get("finished"), "hours": hours,
                     "hours_so_far": _hours(job.get("started"), _now().isoformat()) if running else None})
    blocks: dict[str, dict[str, Any]] = defaultdict(lambda: {"done": 0, "running": 0, "pending": 0, "failed": 0, "other": 0,
                                                              "gpu_hours": 0.0, "cpu_in_queue_hours": 0.0,
                                                              "running_hours_so_far": 0.0, "priorities": set(),
                                                              "retried": 0})
    for job in jobs:
        b = blocks[job["block"]]
        status = job["status"] if job["status"] in {"done", "running", "pending", "failed"} else "other"
        b[status] += 1
        b["priorities"].add(job["priority"])
        if job["attempts"] and job["attempts"] > 1 or job["interrupted"]:
            b["retried"] += 1
        if job["hours"] is not None:
            b["gpu_hours" if job["device"] == "gpu" else "cpu_in_queue_hours"] += job["hours"]
        if job["hours_so_far"] is not None:
            b["running_hours_so_far"] += job["hours_so_far"]
    blocks_out = {k: {**v, "priorities": sorted(p for p in v["priorities"] if p is not None),
                      "gpu_hours": round(v["gpu_hours"], 3), "cpu_in_queue_hours": round(v["cpu_in_queue_hours"], 3),
                      "running_hours_so_far": round(v["running_hours_so_far"], 3)} for k, v in blocks.items()}
    totals = {"jobs": len(jobs), "status": dict(Counter(j["status"] for j in jobs)),
              "gpu_hours_done": round(sum(j["hours"] or 0 for j in jobs if j["device"] == "gpu"), 3),
              "cpu_in_queue_hours_done": round(sum(j["hours"] or 0 for j in jobs if j["device"] != "gpu"), 3),
              "running": [{"name": j["name"], "hours_so_far": round(j["hours_so_far"] or 0, 3)} for j in jobs if j["status"] == "running"],
              "retried_or_interrupted": [j["name"] for j in jobs if (j["attempts"] or 0) > 1 or j["interrupted"]],
              "first_start": min((j["started"] for j in jobs if j["started"]), default=None),
              "last_finish": max((j["finished"] for j in jobs if j["finished"]), default=None)}
    return {"totals": totals, "blocks": dict(sorted(blocks_out.items(), key=lambda kv: min(kv[1]["priorities"] or [999]))),
            "jobs": jobs}


# ------------------------------------------------------------------------------------------- holdouts and audits


def names_sha256(names: list[str]) -> str:
    """As `vsa_embed.tracks.common.names_sha256`."""
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def _walk_values(value: Any, path: str = "") -> list[tuple[str, Any]]:
    out = []
    if isinstance(value, dict):
        for k, v in value.items():
            out += _walk_values(v, f"{path}.{k}" if path else str(k))
    elif not isinstance(value, list):
        out.append((path, value))
    return out


def holdouts(base: Path) -> dict[str, Any]:
    out = {}
    for track, (run, pinned) in HOLDOUTS.items():
        folder = base / run
        names_file = folder / "holdout_concepts.txt"
        entry: dict[str, Any] = {"run": run, "pinned_prefix": pinned, "recorded": []}
        if names_file.exists():
            names = [line for line in names_file.read_text().splitlines() if line.strip()]
            entry.update(recomputed=names_sha256(names), concepts=len(names))
        experiment = folder.parents[1]
        candidates = [folder / "summary.json", experiment / "items" / "holdout.json"]
        for candidate in candidates:
            data = _read_json(candidate) if candidate.exists() else None
            for key, value in _walk_values(data or {}):
                if "holdout" in key and "sha256" in key and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                    entry["recorded"].append({"source": f"{candidate.relative_to(base).as_posix()}:{key}", "sha256": value})
            if data and isinstance(data.get("holdout"), dict) and isinstance(data["holdout"].get("sha256"), str):
                entry["recorded"].append({"source": f"{candidate.relative_to(base).as_posix()}:holdout.sha256",
                                          "sha256": data["holdout"]["sha256"]})
        for config in sorted(experiment.glob("*.yaml")):
            for line in config.read_text().splitlines():
                match = re.search(r"expected_holdout_sha256:\s*([0-9a-f]{64})", line)
                if match:
                    entry["recorded"].append({"source": config.relative_to(base).as_posix(), "sha256": match.group(1)})
        values = {r["sha256"] for r in entry["recorded"]}
        entry["consistent"] = bool(entry.get("recomputed")) and all(v == entry["recomputed"] for v in values)
        entry["pin_matches"] = None if pinned is None or not entry.get("recomputed") else entry["recomputed"].startswith(pinned)
        out[track] = entry
    return out


def audits(base: Path, committed: dict[str, set[str] | None]) -> dict[str, Any]:
    contamination = []
    for manifest in sorted((base / "experiments").glob("*/items/*/manifest.json")) + sorted(
            (base / "experiments").glob("*/items/manifest.json")):
        data = _read_json(manifest) or {}
        checks = data.get("checks") if isinstance(data.get("checks"), dict) else {}
        contamination.append({"items": manifest.parent.relative_to(base).as_posix(), "schema": data.get("schema"),
                              "contamination_free": data.get("contamination_free"),
                              "checked_against": checks.get("checked_against"),
                              "contamination_chars": checks.get("contamination_chars"),
                              "reserved_clashes": checks.get("reserved_clashes"),
                              "rejected_candidates": checks.get("rejected_candidates")})
    leakage = []
    for summary in sorted((base / "experiments").glob("*/runs/*/summary.json")) + sorted(
            (base / "experiments").glob("c6-devtools-benchmark/*/summary.json")):
        data = _read_json(summary) or {}
        found = [(k, v) for k, v in _walk_values(data) if re.search(r"leak|contamin", k)]
        if found:
            leakage.append({"summary": summary.relative_to(base).as_posix(), "fields": dict(found)})
    configs: dict[str, Any] = {}
    for experiment in sorted(p for p in (base / "experiments").iterdir() if p.is_dir()):
        yamls = sorted(p.relative_to(base).as_posix() for p in experiment.rglob("*.yaml")
                       if "runs" not in p.relative_to(experiment).parts[:1] and not any(s in p.parts for s in SKIP_DIRS))
        if not yamls:
            continue
        configs[experiment.name] = {"configs": len(yamls)}
        for ref, tree in committed.items():
            if tree is not None:
                configs[experiment.name][f"uncommitted@{ref}"] = [y for y in yamls if y not in tree]
    reports = sorted(p.name for p in (base / "reports").glob("*.md")) if (base / "reports").is_dir() else []
    return {"contamination_items": contamination, "leakage_fields": leakage,
            "preregistration": {p: (base / p).exists() for p in PREREGISTRATIONS}, "configs": configs, "reports": reports}


# ------------------------------------------------------------------------------------------- rendering


def render(record: dict[str, Any]) -> str:
    lines = ["## Manifests (main checkout on disk)", "",
             "| Experiment | folders | clean | dirty | no git state | manifest missing | data cards | uncommitted@main |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for experiment, s in record["main_summary"].items():
        p = s["provenance"]
        lines.append(f"| {experiment} | {s['folders']} | {p.get('clean', 0)} | {p.get('dirty tree at run start', 0)} | "
                     f"{p.get('no git state (pre-provenance manifest)', 0)} | {p.get('manifest missing', 0)} | "
                     f"{p.get('data card (no git state)', 0)} | {len(s.get('uncommitted@main', []))} |")
    lines += ["", "Dirty-tree manifests:", ""]
    lines += [f"- `{path}`" for s in record["main_summary"].values() for path in s["dirty"]] or ["- none"]
    lines += ["", "Run folders without a manifest:", ""]
    lines += [f"- `{path}`" for s in record["main_summary"].values() for path in s["manifest_missing"]] or ["- none"]
    lines += ["", "## Compute ledger (.jobs, last attempt per job)", "",
              "| Block | priorities | done | running | pending | failed | GPU-h | CPU-in-queue h | running h so far |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for block, b in record["ledger"]["blocks"].items():
        lines.append(f"| {block} | {','.join(map(str, b['priorities']))} | {b['done']} | {b['running']} | {b['pending']} | "
                     f"{b['failed']} | {b['gpu_hours']:.2f} | {b['cpu_in_queue_hours']:.2f} | {b['running_hours_so_far']:.2f} |")
    t = record["ledger"]["totals"]
    lines += ["", f"Totals: {t['jobs']} jobs {t['status']}; GPU-h done {t['gpu_hours_done']:.2f}, CPU-in-queue h "
                  f"{t['cpu_in_queue_hours_done']:.2f}; running {t['running']}; retried/interrupted {t['retried_or_interrupted']}.",
              "", "## Holdouts", "", "| Track | concepts | recomputed | pinned | recorded values agree |", "|---|---:|---|---|---|"]
    for track, h in record["holdouts"].items():
        lines.append(f"| {track} | {h.get('concepts', 'n/a')} | `{(h.get('recomputed') or 'n/a')[:8]}…` | "
                     f"{h['pinned_prefix'] or '—'} ({'match' if h['pin_matches'] else 'n/a' if h['pin_matches'] is None else 'MISMATCH'}) | "
                     f"{'yes' if h['consistent'] else 'NO'} ({len(h['recorded'])} records) |")
    a = record["audits"]
    lines += ["", "## Pre-registration", ""] + [f"- `{p}`: {'present' if ok else 'not found'}" for p, ok in a["preregistration"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="checkout whose git refs answer 'committed?' (default: this script's repository)")
    parser.add_argument("--main", type=Path, default=None, help="main checkout on disk (runs in progress, .jobs/)")
    parser.add_argument("--jobs", type=Path, default=None, help="job queue folder (default: <main>/.jobs)")
    parser.add_argument("--refs", nargs="+", default=["HEAD", "main"])
    parser.add_argument("--output", type=Path, default=None, help="machine-readable record (JSON)")
    args = parser.parse_args(argv)
    main_root = args.main or args.repo
    committed = {ref: git_tree(args.repo, ref) for ref in args.refs}
    repo_scan = scan_tree(args.repo, committed)
    main_scan = scan_tree(main_root, committed) if main_root.resolve() != args.repo.resolve() else repo_scan
    jobs_dir = args.jobs or main_root / ".jobs"
    record = {"generated_at": _now().isoformat(timespec="seconds"), "repo": str(args.repo), "main": str(main_root),
              "refs": {ref: git_ref_sha(args.repo, ref) for ref in args.refs},
              "repo_summary": summarize_tree(repo_scan, args.refs), "main_summary": summarize_tree(main_scan, args.refs),
              "main_folders": main_scan,
              "ledger": ledger(jobs_dir) if jobs_dir.is_dir() else {"totals": {}, "blocks": {}, "jobs": []},
              "holdouts": holdouts(main_root), "audits": audits(main_root, committed)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1, default=str) + "\n")
    sys.stdout.write(render(record))


if __name__ == "__main__":
    main()
