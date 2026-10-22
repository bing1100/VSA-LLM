#!/usr/bin/env bash
# E15 — HRR as a training target, stage 1b (T5): longer training of the HRR-as-target arms; pre-registration
# experiments/e15-hrr-target/preregistration.md, section "Stage 1b (registered 2026-10-22, before any run of it)".
# Prepared 2026-10-22 at the author's approval ("Register and queue 1b").
# 5 training jobs at 54.49696 (after stage 1's report at 54.49695, before the pending T7-ROOD block at 54.497): C5WR-long
# s1–3, C2R-long s1, C5WR-long-lr0.10 s1, each 200M tokens with named checkpoints at 50M/100M/150M/200M
# (`train.save_at_tokens`). Evaluations at 54.49697 (`--no-resume`): read accuracy (`e15_eval read`) and implicit-route QA
# (`e14_reading recall --family e15`) at every checkpoint for the three C5WR-long seeds; read at 200M for C2R-long and
# lr0.10, QA at 200M for C2R-long; outputs under {read,qa}/t5/<run>/ckpt-<tokens>/. The CPU-lane report
# `e15-t5-stage1b-report` at 54.49698. Training jobs keep the queue's default resume (--resume on retry) and
# --min-free-gb 20, as stage 1's. Ordering guarantee (jobqueue): jobs are sorted by (priority, created) and a runner
# takes the first pending job of its lane only when no job of the lane is live, so every 54.49696 training job finishes
# (or fails) before any 54.49697 evaluation starts; the CPU lane starts the report only once every job of lower
# priority is done.
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_TRAIN=${PRIO_TRAIN:-54.49696}; PRIO_EVAL=${PRIO_EVAL:-54.49697}; PRIO_REPORT=${PRIO_REPORT:-54.49698}
HOST=SmolLM2-360M
E15=experiments/e15-hrr-target
E9=experiments/e9-retrofit/runs/t5
ITEMS=experiments/e11-read-to-learn/items/t5-new-smollm2-v1
FRAMES=$E9/$HOST-full-C5-s1/e11-t5-new/frames.jsonl
ALIAS=/home/bhux/data/vsa-llm/e9/alias-tables/t5.json
SPLITS=$E15/splits/t5-v1/splits.json
CKPTS="50000000 100000000 150000000 200000000"
add() { PYTHONPATH=src $PY -m vsa_embed.jobqueue add "$@"; }

for name in $HOST-full-C5WR-long-s1 $HOST-full-C5WR-long-s2 $HOST-full-C5WR-long-s3 $HOST-full-C2R-long-s1 $HOST-full-C5WR-long-lr0.10-s1; do
  add --name e15-t5-$name --priority $PRIO_TRAIN --min-free-gb 20 -- \
    $PY -m vsa_embed.training.lm --config $E15/configs/t5/$name.yaml --output $E15/runs/t5/$name
done

qa() {   # run name, tokens
  add --name e15-t5-$1-ckpt$2-qa --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e14_reading recall --run $E15/runs/t5/$1 --checkpoint checkpoint-$2.pt --items $ITEMS \
    --frames $FRAMES --alias-table $ALIAS --family e15 --null plain --output $E15/qa/t5/$1/ckpt-$2
}
read_acc() {   # run name, tokens
  add --name e15-t5-$1-ckpt$2-read --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e15_eval read --run $E15/runs/t5/$1 --checkpoint checkpoint-$2.pt --splits $SPLITS \
    --new-items $ITEMS --alias-table $ALIAS --output $E15/read/t5/$1/ckpt-$2
}

for s in 1 2 3; do
  for t in $CKPTS; do
    read_acc $HOST-full-C5WR-long-s$s $t
    qa $HOST-full-C5WR-long-s$s $t
  done
done
read_acc $HOST-full-C2R-long-s1 200000000
qa $HOST-full-C2R-long-s1 200000000
read_acc $HOST-full-C5WR-long-lr0.10-s1 200000000

add --name e15-t5-stage1b-report --priority $PRIO_REPORT --no-resume -- \
  $PY -m vsa_embed.experiments.e15_eval report --stage 1b --runs-root $E15/runs/t5 --baseline-root $E9 --qa-root $E15/qa/t5 \
  --read-root $E15/read/t5 --splits $SPLITS --host $HOST --output $E15/report/t5-stage1b
