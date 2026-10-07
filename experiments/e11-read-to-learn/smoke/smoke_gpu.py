"""E11 GPU smoke test (labelled SMOKE): hard 4 GB cap, waits for ≥ 4.5 GB free, two short evaluations."""
import json
import subprocess
import sys
import time

import torch

CAP_GB = 4.0
RUNS = "/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs"
TABLES = "/home/bhux/data/vsa-llm/e9/alias-tables"
OUT = "experiments/e11-read-to-learn/smoke"


def free_gb() -> float:
    text = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.split(",")
    return (float(text[1]) - float(text[0])) / 1024


deadline = time.time() + 3 * 3600
while free_gb() < CAP_GB + 0.5:
    if time.time() > deadline:
        sys.exit("GPU never had enough free memory")
    time.sleep(20)
print(f"free {free_gb():.1f} GB at {time.strftime('%H:%M:%S')}", flush=True)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
torch.cuda.set_per_process_memory_fraction(CAP_GB / total, 0)

from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402

jobs = {
    "t5-new-SmolLM2-360M-C5-s1": ["evaluate", "--run", f"{RUNS}/t5/SmolLM2-360M-full-C5-s1",
                                  "--items", "experiments/e11-read-to-learn/items/t5-new-smollm2-v1", "--limit", "20",
                                  "--styles", "prose", "--alias-table", f"{TABLES}/t5.json",
                                  "--methods", "frames,persistence,context,gradient,locality",
                                  "--readers", "oracle,stated,typeprior,pattern,linker,linker-all,linker-joint,host,random,none",
                                  "--gradient-lr", "1e-3", "--gradient-optimizer", "sgd", "--gradient-backup", "cpu",
                                  "--gradient-factors", "1.0", "--batch-size", "16", "--resamples", "1000"],
    "t4-heldout-SmolLM2-360M-C5-s1": ["evaluate", "--run", f"{RUNS}/t4/SmolLM2-360M-full-C5-s1",
                                      "--items", "experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1", "--limit", "40",
                                      "--alias-table", f"{TABLES}/t4.json",
                                      "--methods", "frames,persistence,context,windows,locality",
                                      "--readers", "oracle,stated,typeprior,pattern,linker,linker-all,linker-joint,random,none",
                                      "--max-windows", "48", "--batch-size", "16", "--resamples", "1000"],
}
timings = {}
for name, argv in jobs.items():
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    e11.main(argv + ["--output", f"{OUT}/{name}", "--device", "cuda", "--smoke", "--overwrite"])
    timings[name] = {"seconds": round(time.time() - started, 1),
                     "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    print(name, timings[name], flush=True)
json.dump(timings, open(f"{OUT}/timings.json", "w"), indent=2)
