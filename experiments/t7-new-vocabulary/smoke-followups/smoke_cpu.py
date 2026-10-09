"""SMOKE (not a result): the T7-ROOD follow-up pipelines end to end on CPU (decision 63, TK-H1 follow-ups).

A tiny T7-ROOD stage is trained on CPU (SmolLM2-135M, LoRA, P0 / C0′ / C5, seed 1, a few hundred tokens of the real
`rood-v1` corpora, 16 evaluation windows of 256 tokens), then every follow-up job runs on it with small limits:
E11 read-to-learn on `t7rood-heldout-smollm2-v1`, E12 recall on `understanding-t7rood-smollm2-v1` (reverse and relation
families), the E11 and E12 reports, and the paired comparison against a copy of the same stage (Δ must be 0 exactly). The
checkpoints stay in the scratch folder; only the small result files are copied here.

    PYTHONPATH=src OMP_NUM_THREADS=4 python experiments/t7-new-vocabulary/smoke-followups/smoke_cpu.py --scratch DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ITEMS_E11 = Path("experiments/e11-read-to-learn/items/t7rood-heldout-smollm2-v1")
ITEMS_E12 = Path("experiments/e9-retrofit/items/understanding-t7rood-smollm2-v1")
ALIAS = Path("~/data/vsa-llm/e9/alias-tables/t7rood.json").expanduser()
SMOKE = {"model": {"seq_len": 256}, "train": {"total_tokens": 4096, "micro_batch": 2, "grad_accum": 1, "warmup_tokens": 1024,
                                               "checkpoint_minutes": 600}, "eval": {"windows": 16, "first_tokens": 4096, "batch": 4},
         "device": "cpu"}


def run(command: list[str], log: list[dict]) -> None:
    started = time.monotonic()
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    log.append({"command": command[2:6], "seconds": round(time.monotonic() - started, 1)})


def merge(config: dict, part: dict) -> None:
    for key, value in part.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merge(config[key], value)
        else:
            config[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    from vsa_embed.experiments import e9_plan
    py, root, log = sys.executable, args.scratch / "e9", []
    stage = root / "runs" / "t7rood"
    paths = e9_plan.write_stage("t7rood", hosts=["SmolLM2-135M"], models=["P0", "C0p", "C5"], seeds=[1], track="t7rood", mode="lora",
                                root=root)
    for path in paths:
        config = yaml.safe_load(path.read_text())
        merge(config, SMOKE)
        config["experiment"] += "-SMOKE"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        run([py, "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(stage / path.stem)], log)
    c5, c0 = stage / "SmolLM2-135M-lora-C5-s1", stage / "SmolLM2-135M-lora-C0p-s1"
    common = ["--items", str(ITEMS_E11), "--styles", "scr", "--alias-table", str(ALIAS), "--limit", "3", "--max-windows", "16",
              "--gradient-lr", "1e-4", "--window-gradient", "--device", "cpu", "--batch-size", "8"]
    run([py, "-m", "vsa_embed.experiments.e11_read_to_learn", "evaluate", "--run", str(c5), "--methods", "frames,gradient,windows,locality",
         "--readers", "oracle,linker,random,none", *common, "--output", str(c5 / "e11-t7rood-heldout")], log)
    run([py, "-m", "vsa_embed.experiments.e11_read_to_learn", "evaluate", "--run", str(c0), "--methods", "gradient,windows",
         "--readers", "none", *common, "--output", str(c0 / "e11-t7rood-heldout")], log)
    run([py, "-m", "vsa_embed.experiments.e11_read_to_learn", "report", "--runs", str(c5 / "e11-t7rood-heldout"),
         str(c0 / "e11-t7rood-heldout"), "--output", str(args.scratch / "e11-report")], log)
    families = "reverse,paraphrase,negation,affordance"
    for model_dir, conditions in ((c5, "none,recall:own,symbolic"), (c0, "none,recall:C5,symbolic")):
        run([py, "-m", "vsa_embed.experiments.e12_self_query", "evaluate", "--run", str(model_dir), "--items", str(ITEMS_E12),
             "--conditions", conditions, "--families", families, "--limit", "2", "--device", "cpu", "--batch-size", "8",
             "--label", "SMOKE", "--overwrite"], log)
    run([py, "-m", "vsa_embed.experiments.e12_report", "--runs", str(stage), "--output", str(args.scratch / "e12-report"),
         "--understanding", ITEMS_E12.name, "--label", "SMOKE", "--overwrite"], log)
    reference = root / "runs" / "t7"
    if reference.exists():
        shutil.rmtree(reference)
    shutil.copytree(stage, reference, ignore=shutil.ignore_patterns("*.pt"))
    run([py, "-m", "vsa_embed.experiments.t7_rood_compare", "--rood", str(stage), "--reference", str(reference), "--host",
         "SmolLM2-135M", "--resamples", "200", "--output", str(args.scratch / "compare"), "--overwrite",
         "--title", "SMOKE (not a result): T7-ROOD against a copy of itself"], log)
    # the small result files, labelled SMOKE
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    copies = {"e11-C5-summary.json": c5 / "e11-t7rood-heldout" / "summary.json", "e11-report.md": args.scratch / "e11-report" / "report.md",
              "e12-C5-summary.json": c5 / f"self-query-{ITEMS_E12.name}" / "summary.json",
              "e12-C5-report.md": c5 / f"self-query-{ITEMS_E12.name}" / "report.md", "e12-report.md": args.scratch / "e12-report" / "report.md",
              "compare-summary.json": args.scratch / "compare" / "summary.json", "compare-report.md": args.scratch / "compare" / "report.md"}
    for name, source in copies.items():
        if source.exists():
            shutil.copy(source, out / name)
    (out / "timings.json").write_text(json.dumps({"label": "SMOKE (not a result)", "steps": log}, indent=2) + "\n")


if __name__ == "__main__":
    main()
