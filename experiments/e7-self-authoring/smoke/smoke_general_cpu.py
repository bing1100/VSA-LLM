"""WP-E7 smoke run on a tiny slice of the real data (CPU): D7.0 → D7.1 → D7.2 → D7.3 → R6."""
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
from vsa_embed.experiments import e7_authoring, e7_plan, e7_report, e7_round


def main():
    global SCRATCH, STEPS, RUN, DATA, CPU, timings
    SCRATCH = Path(sys.argv[1])
    STEPS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["all"]
    RUN, DATA = SCRATCH / "run", SCRATCH / "data"
    CPU = torch.device("cpu")
    timings = json.loads((SCRATCH / "timings.json").read_text()) if (SCRATCH / "timings.json").exists() else {}


    def step(name, fn):
        if "all" not in STEPS and name not in STEPS:
            return None
        started = time.monotonic()
        result = fn()
        timings[name] = round(time.monotonic() - started, 1)
        (SCRATCH / "timings.json").write_text(json.dumps(timings, indent=1))
        print(f"[{name}] {timings[name]} s", flush=True)
        return result


    config = {"c3_run": "experiments/c3-general-corpus/runs/v2", "host_root": "~/data/vsa-llm/c3/wordnet-smollm2-v1",
              "data_root": str(DATA),
              "corpora": {"general_tokens": 100_000, "replay_tokens": 20_000, "read_tokens": 40_000, "test_tokens": 8_000, "workers": 4},
              "discovery": {"min_count": 4, "keep": 2000, "max_occurrences": 8, "batch": 4},
              "authoring": {"candidates": 12, "contexts": 2, "samples": 2, "max_validation": 6, "batch": 8},
              "direct": {"documents": 3, "batch": 3},
              "teacher": {"limit": 12, "batch": 6}}
    step("prepare", lambda: e7_authoring.prepare("general", e7_authoring._merge(e7_authoring.DEFAULTS, config), RUN))
    for host in ("SmolLM2-360M", "SmolLM2-135M", "Qwen2.5-0.5B"):
        step(f"discover-{host}", lambda h=host: e7_authoring.discover(RUN, h, device=CPU,
                                                                     overrides=None if h == "SmolLM2-360M" else {"limit_texts": 12}))
    step("link", lambda: e7_authoring.link(RUN))
    for host in ("SmolLM2-360M", "SmolLM2-135M", "Qwen2.5-0.5B"):
        step(f"author-{host}", lambda h=host: e7_authoring.author(RUN, h, device=CPU))
    step("author-direct", lambda: e7_authoring.author(RUN, "direct-SmolLM2-135M", device=CPU))
    step("author-hearst", lambda: e7_authoring.author(RUN, "hearst", device=CPU))
    step("author-random", lambda: e7_authoring.author(RUN, "random", device=CPU))
    step("author-teacher", lambda: e7_authoring.author(RUN, "teacher", device=CPU))
    step("quality", lambda: e7_authoring.quality(RUN))
    step("judge-items", lambda: e7_authoring.judge_items(RUN, per_author=4, calibration=6))
    ROUND = {"host": "SmolLM2-135M", "seq_len": 256, "base_tokens": 256 * 2 * 6, "round_tokens": 256 * 2 * 4,
             "sequences_per_step": 16, "verification": {"window": 96, "batch": 8, "min_validation": 2, "resamples": 500},
             "entigraph": {"writer": "SmolLM2-135M", "per_entity": 1, "max_new_tokens": 32, "batch": 8},
             "notes": {"writer": "SmolLM2-135M", "max_new_tokens": 32, "batch": 8},
             "spa": {"writer": "SmolLM2-135M", "documents": 6, "max_new_tokens": 48, "batch": 6},
             "config_overrides": {"train": {"micro_batch": 2}, "eval": {"batch": 8}, "device": "cpu"},
             "cross": {"train_tokens": 200_000, "run_tokens": 2048, "workers": 4, "seeds": [1],
                       "config_overrides": {"train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 2048, "warmup_tokens": 512},
                                            "eval": {"batch": 8, "first_tokens": 1024}, "device": "cpu"}}}
    step("round-settings", lambda: e7_round.update_round_settings(RUN, ROUND))
    step("base", lambda: e7_round.train_base(RUN, 1))
    step("verify-self", lambda: e7_round.verify(RUN, 1, "SmolLM2-360M", device=CPU))
    step("verify-teacher", lambda: e7_round.verify(RUN, 1, "teacher", device=CPU))
    step("entigraph", lambda: e7_round.entigraph(RUN, device=CPU))
    step("spa", lambda: e7_round.spa(RUN, device=CPU))
    step("notes", lambda: e7_round.notes(RUN, device=CPU))
    step("verify-notes", lambda: e7_round.verify_notes(RUN, 1, device=CPU))
    for condition in e7_round.CONDITIONS + e7_round.OPTIONAL_CONDITIONS:
        step(f"train-{condition}", lambda c=condition: e7_round.train_condition(RUN, 1, c))
    step("cross-prepare", lambda: e7_round.cross_prepare(RUN))
    for condition in e7_round.CROSS_CONDITIONS:
        step(f"cross-{condition}", lambda c=condition: e7_round.cross_train(RUN, 1, c))
    step("report", lambda: e7_report.write_report(RUN, resamples=2000))
    print(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
