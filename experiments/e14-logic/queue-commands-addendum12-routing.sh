#!/usr/bin/env bash
# E14 addendum 12 (preregistration-addendum-12-routing.md): RD-4e, routing that generalises — paraphrase-augmented few-shot
# router, abstention, top-2, the size rule, whole-frame recall re-measured without prompt truncation (--max-length 1280).
# SmolLM2-360M C5 seeds 1–3 on T4 held-out and T5 new words; the CPU-lane report at PRIO_REPORT (above PRIO_GPU, so it waits
# for its inputs). Prepared 2026-10-21 at the author's "do these steps" for the routing block. Qwen3-1.7B C0′ on T5 is not
# queued (it would add ≥ 1.5 GPU-h on its own; see the addendum's cost section).
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_GPU=${PRIO_GPU:-54.49599975}; PRIO_REPORT=${PRIO_REPORT:-54.49599988}
T4_ITEMS=experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1
T5_ITEMS=experiments/e11-read-to-learn/items/t5-new-smollm2-v1
T4_ALIASES=/home/bhux/data/vsa-llm/e9/alias-tables/t4.json
T5_ALIASES=/home/bhux/data/vsa-llm/e9/alias-tables/t5.json
COMMON="--family rd4e --null plain --max-length 1280 --route-scoring prefix --calibration-style chebi"
RUNS=()
for s in 1 2 3; do
  T4_RUN=experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s$s
  T5_RUN=experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s$s
  CAL="--calibration-items $T4_ITEMS --calibration-frames $T4_RUN/e11-t4-heldout/frames.jsonl"   # T4 training-template calibration
  OUT4=experiments/e14-logic/runs/rd4e/t4-heldout/SmolLM2-360M-C5-s$s
  OUT5=experiments/e14-logic/runs/rd4e/t5-new/SmolLM2-360M-C5-s$s
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd4e-t4-heldout-SmolLM2-360M-C5-s$s --priority $PRIO_GPU --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e14_reading recall --run $T4_RUN --items $T4_ITEMS --alias-table $T4_ALIASES $COMMON $CAL --output $OUT4
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd4e-t5-new-SmolLM2-360M-C5-s$s --priority $PRIO_GPU --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e14_reading recall --run $T5_RUN --items $T5_ITEMS --alias-table $T5_ALIASES $COMMON $CAL --output $OUT5
  RUNS+=("$OUT4" "$OUT5")
done
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd4e-report --priority $PRIO_REPORT --no-resume -- \
  $PY -m vsa_embed.experiments.e14_reading recall-report --family rd4e --runs "${RUNS[@]}" --output experiments/e14-logic/report/rd4e
