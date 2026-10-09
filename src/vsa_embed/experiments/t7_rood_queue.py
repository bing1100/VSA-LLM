"""T7-ROOD follow-up jobs (decision 63, WP TK-H1): the queue commands are printed, never submitted.

- **Paired comparison** (`t7_rood_compare`; preregistration-rood.md §4, P2 and P3): T7-ROOD against T7 v1 on SmolLM2-360M,
  seeds 1–3, as the CPU-lane job `t7rood-vs-t7-report` at 54.4974 (after both stages' runs: T7 at 51–54, T7-ROOD at
  54.497–54.4973). The Qwen3-1.7B seed-1 secondary is `t7rood-vs-t7-qwen3-report` at 54.4978, after both Qwen3 blocks
  (54.4975, 54.4977).
- **E11 read-to-learn on T7-ROOD** (54.4979): the T7-ROOD held-out records read from their MeSH SCR notes
  (`experiments/e11-read-to-learn/items/t7rood-heldout-smollm2-v1`, built as T7's `t7-heldout-smollm2-v1`) by the
  SmolLM2-360M C5 runs (every reader; frames, gradient, windows, locality) and the C0′ runs (the gradient text route and the
  windows), seeds 1–3, as E11 planned T7-H. The gradient lr is chosen on the T5 dev words by a job of this block
  (`dev-t7rood/`), so the block does not wait for E11's own dev job at 62. Then the pooled E11 report (`report-t7rood`).
- **E12 recall on T7-ROOD** (54.49795): T7 has no role-swap twins and no two-hop items, so the Q1-style test reads the WP-UB
  understanding items' reverse family and the relation families (`e12_self_query.RELATION_FAMILIES`: paraphrase, negation,
  affordance; the anchor's whole-frame recall in context) on every T7-ROOD SmolLM2-360M run with the core conditions
  (`none`, the recall, `symbolic`: no `definition`, which has no writer for MeSH frames), then the stage report
  (`e12-self-query/report/t7rood-understanding`).

GPU-h are idle-GPU estimates: E11 from the T7-H plan (C5 ≈ 0.60, C0′ ≈ 0.40, dev ≈ 0.04); E12 from the phase-A twin timing
(≈ 1.5 s per 100 scored texts at 360M) over ≈ 600 anchors × ≈ 6 items × 2 templates × ≈ 3 candidates × 3 conditions.

    python -m vsa_embed.experiments.t7_rood_queue > experiments/t7-new-vocabulary/queue-commands-rood-followups.sh
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

E9 = Path("experiments/e9-retrofit")
E11 = Path("experiments/e11-read-to-learn")
E12 = Path("experiments/e12-self-query")
ALIAS_TABLE = Path("~/data/vsa-llm/e9/alias-tables/t7rood.json").expanduser()
COMPARE_PRIORITY, COMPARE_QWEN_PRIORITY = 54.4974, 54.4978
E11_PRIORITY, E12_PRIORITY = 54.4979, 54.49795
HOST, SEEDS = "SmolLM2-360M", (1, 2, 3)
E11_ITEMS = E11 / "items" / "t7rood-heldout-smollm2-v1"
E11_DEV = E11 / "dev-t7rood" / HOST
E11_C5_READERS = "oracle,stated,typeprior,pattern,linker,linker-all,linker-joint,host,random,none"   # T7-H's (no teacher)
E11_HOURS = {"C5": 0.60, "C0p": 0.40, "dev": 0.04}
UNDERSTANDING = E9 / "items" / "understanding-t7rood-smollm2-v1"
E12_FAMILIES = ("reverse", "paraphrase", "negation", "affordance")
E12_SECONDS_PER_100_TEXTS = 1.5


def compare_jobs(python: str = "$PY") -> list[dict[str, Any]]:
    module = [python, "-m", "vsa_embed.experiments.t7_rood_compare"]
    return [{"name": "t7rood-vs-t7-report", "priority": COMPARE_PRIORITY, "min_free_gb": 1, "hours": 0.0, "block": "compare",
             "command": [*module, "--rood", str(E9 / "runs" / "t7rood"), "--reference", str(E9 / "runs" / "t7"), "--host", HOST,
                         "--output", str(E9 / "report" / "t7rood-vs-t7"), "--overwrite"]},
            {"name": "t7rood-vs-t7-qwen3-report", "priority": COMPARE_QWEN_PRIORITY, "min_free_gb": 1, "hours": 0.0, "block": "compare",
             "command": [*module, "--rood", str(E9 / "runs" / "t7rood-qwen3"), "--reference", str(E9 / "runs" / "t7-qwen3"),
                         "--host", "Qwen3-1.7B-Base", "--output", str(E9 / "report" / "t7rood-vs-t7-qwen3"), "--overwrite"]}]


def e11_jobs(python: str = "$PY") -> list[dict[str, Any]]:
    module = [python, "-m", "vsa_embed.experiments.e11_read_to_learn"]
    jobs = [{"name": f"e11-gradient-dev-{HOST}-t7rood", "priority": E11_PRIORITY, "min_free_gb": 10, "hours": E11_HOURS["dev"], "block": "e11",
             "command": [*module, "gradient-dev", "--run", str(E9 / "runs" / "t5" / f"{HOST}-full-C0p-s1"), "--items",
                         str(E11 / "items" / "t5-dev-smollm2-v1"), "--output", str(E11_DEV)]}]
    outputs = []
    for model in ("C5", "C0p"):
        methods, readers = ("frames,gradient,windows,locality", E11_C5_READERS) if model == "C5" else ("gradient,windows", "none")
        for seed in SEEDS:
            run = E9 / "runs" / "t7rood" / f"{HOST}-full-{model}-s{seed}"
            output = run / "e11-t7rood-heldout"
            outputs.append(str(output))
            jobs.append({"name": f"e11-t7rood-heldout-{HOST}-{model}-s{seed}", "priority": E11_PRIORITY, "min_free_gb": 10,
                         "hours": E11_HOURS[model], "block": "e11",
                         "command": [*module, "evaluate", "--run", str(run), "--items", str(E11_ITEMS), "--methods", methods,
                                     "--readers", readers, "--styles", "scr", "--alias-table", str(ALIAS_TABLE),
                                     "--gradient-lr-from", str(E11_DEV / "gradient_lr.json"), "--window-gradient",
                                     "--max-windows", "2048", "--output", str(output)]})
    jobs.append({"name": "e11-t7rood-report", "priority": E11_PRIORITY, "min_free_gb": 1, "hours": 0.0, "block": "e11",
                 "command": [*module, "report", "--runs", *outputs, "--output", str(E11 / "report-t7rood")]})
    return jobs


def e12_hours(items: Path = UNDERSTANDING, *, conditions: int = 3) -> float:
    """Idle-GPU hours of one E12 job on the understanding set (module docstring)."""
    from .e12_self_query import SET_LIMIT, load_item_set
    item_set = load_item_set(items, limit=SET_LIMIT["understanding"], families=E12_FAMILIES)
    texts = sum(len(p.templates) * len(p.candidates) for p in item_set.prompts) * conditions
    return texts / 100 * E12_SECONDS_PER_100_TEXTS / 3600


def e12_jobs(python: str = "$PY", *, root: Path = E9) -> list[dict[str, Any]]:
    from .e12_self_query import queue_stage
    # the decision-63 models only: decision 64's C5sh / C6d arms (configs in the same stage) are loss controls, not E12 subjects
    planned = queue_stage("t7rood", UNDERSTANDING, priority=0, models=["P0", "C0p", "C2", "C5"], hosts=[HOST], root=root,
                          python=python, dry_run=True, core=True, families=list(E12_FAMILIES))
    hours = e12_hours() if (UNDERSTANDING / "manifest.json").exists() else float("nan")
    jobs = [{**j, "priority": E12_PRIORITY, "min_free_gb": 5, "hours": hours * (len(j["command"][j["command"].index("--conditions") + 1]
                                                                                     .split(",")) / 3), "block": "e12"} for j in planned]
    jobs.append({"name": "e12-t7rood-understanding-report", "priority": E12_PRIORITY, "min_free_gb": 1, "hours": 0.0, "block": "e12",
                 "command": [python, "-m", "vsa_embed.experiments.e12_report", "--runs", str(root / "runs" / "t7rood"), "--output",
                             str(E12 / "report" / "t7rood-understanding"), "--understanding", UNDERSTANDING.name, "--hosts", HOST,
                             "--title", "E12 — recall on T7-ROOD (WP-UB reverse and relation families)", "--overwrite"]})
    return jobs


def all_jobs(python: str = "$PY") -> list[dict[str, Any]]:
    return compare_jobs(python) + e11_jobs(python) + e12_jobs(python)


def shell(command: list[str]) -> str:
    """A command line for the script: every argument shell-quoted except the interpreter placeholder (`$PY`)."""
    return " ".join(arg if arg.startswith("$") else shlex.quote(arg) for arg in command)


def render(jobs: list[dict[str, Any]]) -> str:
    lines = ["#!/usr/bin/env bash",
             "# T7-ROOD follow-ups (decision 63, WP TK-H1): paired comparison against T7 v1, E11 read-to-learn and E12 recall on",
             "# T7-ROOD. Printed by `python -m vsa_embed.experiments.t7_rood_queue`; NOT EXECUTED by the agent that wrote them.",
             "# Run from the repository root of the main checkout after merging, in this order (equal priorities run first come,",
             "# first served; a CPU-lane `-report` job starts only once every job ahead of it is done):",
             "#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python",
             "set -euo pipefail", ': "${PY:?set PY to the pinned interpreter}"', ""]
    for block, title in (("compare", "paired comparison (preregistration-rood.md §4: P2, P3)"),
                         ("e11", f"E11 read-to-learn on T7-ROOD at {E11_PRIORITY}"), ("e12", f"E12 recall on T7-ROOD at {E12_PRIORITY}")):
        chosen = [j for j in jobs if j["block"] == block]
        total = sum(j["hours"] for j in chosen if j["hours"] == j["hours"])
        lines.append(f"# --- {title}: {len(chosen)} job(s), ≈ {total:.1f} GPU-h (idle-GPU estimate)")
        for job in chosen:
            lines.append(f"PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name {job['name']} --priority {job['priority']} "
                         f"--min-free-gb {job['min_free_gb']} --no-resume -- {shell(job['command'])}   # ≈ {job['hours']:.2f} GPU-h")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default="$PY", help="the interpreter written into the commands")
    args = parser.parse_args(argv)
    print(render(all_jobs(args.python)))


if __name__ == "__main__":
    main()
