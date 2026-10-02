"""WP-E7 dev-tools smoke (CPU): prepare → discover (SmolLM2-135M) → link → authors (host, hearst, teacher) → quality."""
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
from vsa_embed.experiments import e7_authoring


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

    e7_authoring.CONSUMER = "SmolLM2-135M"
    config = e7_authoring._merge(e7_authoring.DEFAULTS, {
        "devtools_root": "experiments/c6-devtools-benchmark/v1", "data_root": str(scratch / "data"), "limit_documents": 300,
        "discovery": {"min_count": 3, "keep": 500, "max_occurrences": 4, "batch": 4},
        "authoring": {"candidates": 30, "contexts": 2, "samples": 2, "batch": 8},
        "teacher": {"limit": 6, "batch": 6}})
    step("prepare", lambda: e7_authoring.prepare("devtools", config, run))
    step("discover", lambda: e7_authoring.discover(run, "SmolLM2-135M", device=cpu))
    step("link", lambda: e7_authoring.link(run))
    step("author-host", lambda: e7_authoring.author(run, "SmolLM2-135M", device=cpu))
    step("author-hearst", lambda: e7_authoring.author(run, "hearst", device=cpu))
    if "--teacher" in sys.argv:
        step("author-teacher", lambda: e7_authoring.author(run, "teacher", device=cpu))
    quality = step("quality", lambda: e7_authoring.quality(run))
    print(json.dumps({k: {m: v[m] for m in ("edges", "precision", "recall", "f1")} for k, v in quality["authors"].items()}, indent=1, default=str))
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
