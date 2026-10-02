"""WP-E7 supplementary smoke (CPU): held-out-utility verification of real host frames and notes on real text."""
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
from vsa_embed.experiments import e7_authoring, e7_round


def main():
    scratch = Path(sys.argv[1])
    run, cpu = scratch / "run", torch.device("cpu")
    timings = {}

    def step(name, fn):
        started = time.monotonic()
        result = fn()
        timings[name] = round(time.monotonic() - started, 1)
        print(f"[{name}] {timings[name]} s", flush=True)
        return result

    config = {"c3_run": "experiments/c3-general-corpus/runs/v2", "host_root": "~/data/vsa-llm/c3/wordnet-smollm2-v1",
              "data_root": str(scratch / "data"),
              "corpora": {"general_tokens": 100_000, "replay_tokens": 20_000, "read_tokens": 250_000, "test_tokens": 8_000, "workers": 4},
              "discovery": {"min_count": 4, "min_documents": 3, "keep": 2000, "max_occurrences": 6, "batch": 4, "limit_texts": 60},
              "authoring": {"candidates": 40, "contexts": 2, "samples": 2, "max_validation": 6, "batch": 8}}
    step("prepare", lambda: e7_authoring.prepare("general", e7_authoring._merge(e7_authoring.DEFAULTS, config), run))
    step("discover", lambda: e7_authoring.discover(run, "SmolLM2-360M", device=cpu))
    aset = step("link", lambda: e7_authoring.link(run))
    print("validation occurrences", sum(len(c["validation"]) for c in aset["candidates"]),
          "candidates with ≥ 2", sum(len(c["validation"]) >= 2 for c in aset["candidates"]))
    step("author", lambda: e7_authoring.author(run, "SmolLM2-360M", device=cpu))
    e7_round.update_round_settings(run, {
        "host": "SmolLM2-135M", "seq_len": 256, "base_tokens": 256 * 2 * 40, "sequences_per_step": 16,
        "verification": {"window": 96, "batch": 8, "min_validation": 2, "resamples": 1000},
        "notes": {"writer": "SmolLM2-135M", "max_new_tokens": 32, "batch": 8},
        "config_overrides": {"train": {"micro_batch": 2}, "eval": {"batch": 8}, "device": "cpu"}})
    step("base", lambda: e7_round.train_base(run, 1))
    verified = step("verify", lambda: e7_round.verify(run, 1, "SmolLM2-360M", device=cpu))
    step("notes", lambda: e7_round.notes(run, device=cpu))
    notes = step("verify-notes", lambda: e7_round.verify_notes(run, 1, device=cpu))
    print(json.dumps(verified["summary"]), json.dumps({k: notes[k] for k in ("notes", "kept", "tokens", "median_accepted_utility")}))
    cards = [json.loads(line) for line in (run / "round1" / "s1" / "cards-SmolLM2-360M.jsonl").read_text().splitlines()]
    for card in sorted(cards, key=lambda c: -c["utility_mean"])[:8]:
        print(f"  {card['surface']!r} {card['relation']} {card['filler']!r} U={card['utility_mean']:+.4f} low={card['utility_low']:+.4f} "
              f"n={card['n']} accepted={card['accepted']}")
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
