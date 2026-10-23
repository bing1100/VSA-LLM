#!/usr/bin/env bash
# E15 — HRR as a training target, stage 1c (T5): aligned objectives (route distillation C5RD / C2RD, fact verbalisation
# C5FV / C2FV); pre-registration experiments/e15-hrr-target/preregistration.md, section "Stage 1c (registered 2026-10-23,
# before any run of it)". Also queues the stage-1b addition of the same date (§15): the matched-length loss control
# `C5-long` (E9's C5 recipe at 200M tokens, seed 1) with its read and QA evaluation at 200M.
# Prepared 2026-10-23 at the author's approval ("Queue these two").
#
# Priorities (all before the pending T7-ROOD block at 54.497):
#   54.49696  C5-long training (with stage 1b's training jobs; runs after the one still running)      --min-free-gb 20, default resume
#   54.49697  C5-long read + QA at checkpoint-200000000.pt (with stage 1b's evaluations, before its report at 54.49698)
#   54.49699  8 stage-1c training jobs: C5RD s1–3, C5FV s1–3, C2RD s1, C2FV s1 (50M tokens each)
#   54.496992 their 16 evaluations: implicit-route QA (`e14_reading recall --family e15 --null plain`) and read accuracy
#             (`e15_eval read`; probe only, these arms train no W)                                      --no-resume
#   54.496995 the CPU-lane report `e15-t5-stage1c-report` (`e15_eval report --stage 1c`)                --no-resume
# Ordering guarantee (jobqueue): jobs are sorted by (priority, created) and a runner takes the first pending job of its
# lane only when no job of the lane is live, so every 54.49699 training job finishes (or fails) before any 54.496992
# evaluation starts; the CPU lane (names matching `-report`) starts the report only once every job of lower priority is
# done. The optional C5shRD seed 1 of the pre-registration is NOT queued (no config written); add it on request.
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_LONG_TRAIN=${PRIO_LONG_TRAIN:-54.49696}; PRIO_LONG_EVAL=${PRIO_LONG_EVAL:-54.49697}
PRIO_TRAIN=${PRIO_TRAIN:-54.49699}; PRIO_EVAL=${PRIO_EVAL:-54.496992}; PRIO_REPORT=${PRIO_REPORT:-54.496995}
HOST=SmolLM2-360M
E15=experiments/e15-hrr-target
E9=experiments/e9-retrofit/runs/t5
ITEMS=experiments/e11-read-to-learn/items/t5-new-smollm2-v1
FRAMES=$E9/$HOST-full-C5-s1/e11-t5-new/frames.jsonl
ALIAS=/home/bhux/data/vsa-llm/e9/alias-tables/t5.json
SPLITS=$E15/splits/t5-v1/splits.json
add() { PYTHONPATH=src $PY -m vsa_embed.jobqueue add "$@"; }

# (A) stage 1b addition: the matched-length control C5-long (200M tokens, named checkpoints as the 1b arms).
name=$HOST-full-C5-long-s1
add --name e15-t5-$name --priority $PRIO_LONG_TRAIN --min-free-gb 20 -- \
  $PY -m vsa_embed.training.lm --config $E15/configs/t5/$name.yaml --output $E15/runs/t5/$name
add --name e15-t5-$name-ckpt200000000-read --priority $PRIO_LONG_EVAL --min-free-gb 10 --no-resume -- \
  $PY -m vsa_embed.experiments.e15_eval read --run $E15/runs/t5/$name --checkpoint checkpoint-200000000.pt --splits $SPLITS \
  --new-items $ITEMS --alias-table $ALIAS --output $E15/read/t5/$name/ckpt-200000000
add --name e15-t5-$name-ckpt200000000-qa --priority $PRIO_LONG_EVAL --min-free-gb 10 --no-resume -- \
  $PY -m vsa_embed.experiments.e14_reading recall --run $E15/runs/t5/$name --checkpoint checkpoint-200000000.pt --items $ITEMS \
  --frames $FRAMES --alias-table $ALIAS --family e15 --null plain --output $E15/qa/t5/$name/ckpt-200000000

# (B) stage 1c: training.
ARMS="C5RD-s1 C5RD-s2 C5RD-s3 C5FV-s1 C5FV-s2 C5FV-s3 C2RD-s1 C2FV-s1"
for arm in $ARMS; do
  name=$HOST-full-$arm
  add --name e15-t5-$name --priority $PRIO_TRAIN --min-free-gb 20 -- \
    $PY -m vsa_embed.training.lm --config $E15/configs/t5/$name.yaml --output $E15/runs/t5/$name
done

# (B) stage 1c: evaluations, as stage 1's (§7).
for arm in $ARMS; do
  name=$HOST-full-$arm
  add --name e15-t5-$name-qa --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e14_reading recall --run $E15/runs/t5/$name --items $ITEMS --frames $FRAMES --alias-table $ALIAS \
    --family e15 --null plain --output $E15/qa/t5/$name
  add --name e15-t5-$name-read --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e15_eval read --run $E15/runs/t5/$name --splits $SPLITS --new-items $ITEMS --alias-table $ALIAS \
    --output $E15/read/t5/$name
done

add --name e15-t5-stage1c-report --priority $PRIO_REPORT --no-resume -- \
  $PY -m vsa_embed.experiments.e15_eval report --stage 1c --runs-root $E15/runs/t5 --baseline-root $E9 --qa-root $E15/qa/t5 \
  --read-root $E15/read/t5 --splits $SPLITS --host $HOST --output $E15/report/t5-stage1c
