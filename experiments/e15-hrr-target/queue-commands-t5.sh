#!/usr/bin/env bash
# E15 — HRR as a training target, stage 1 (T5): pre-registration experiments/e15-hrr-target/preregistration.md.
# Prepared 2026-10-22 at the author's approval ("Run and queue this experiment with high priority").
# 12 training jobs at 54.4969 (ahead of the pending T7-ROOD block at 54.497), their QA (E14 recall, family e15) and
# read-accuracy evaluations at 54.49692 (also for the E9 baselines C5, C2, C5sh, whose runs are reused), the CPU-lane
# report at 54.49695. Training jobs keep the queue's default resume (--resume on retry: the trainer continues from its
# checkpoint and a finished run returns at once); evaluation and report jobs use --no-resume, as E14's.
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_TRAIN=${PRIO_TRAIN:-54.4969}; PRIO_EVAL=${PRIO_EVAL:-54.49692}; PRIO_REPORT=${PRIO_REPORT:-54.49695}
HOST=SmolLM2-360M
E15=experiments/e15-hrr-target
E9=experiments/e9-retrofit/runs/t5
ITEMS=experiments/e11-read-to-learn/items/t5-new-smollm2-v1
FRAMES=$E9/$HOST-full-C5-s1/e11-t5-new/frames.jsonl
ALIAS=/home/bhux/data/vsa-llm/e9/alias-tables/t5.json
SPLITS=$E15/splits/t5-v1/splits.json
add() { PYTHONPATH=src $PY -m vsa_embed.jobqueue add "$@"; }

for arm in C5WR C5R C2R C5shWR; do
  for s in 1 2 3; do
    name=$HOST-full-$arm-s$s
    add --name e15-t5-$name --priority $PRIO_TRAIN --min-free-gb 20 -- \
      $PY -m vsa_embed.training.lm --config $E15/configs/t5/$name.yaml --output $E15/runs/t5/$name
  done
done

for arm in C5WR C5R C2R C5shWR C5 C2 C5sh; do
  case $arm in C5|C2|C5sh) root=$E9 ;; *) root=$E15/runs/t5 ;; esac
  for s in 1 2 3; do
    name=$HOST-full-$arm-s$s
    add --name e15-t5-$name-qa --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
      $PY -m vsa_embed.experiments.e14_reading recall --run $root/$name --items $ITEMS --frames $FRAMES --alias-table $ALIAS \
      --family e15 --null plain --output $E15/qa/t5/$name
    add --name e15-t5-$name-read --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
      $PY -m vsa_embed.experiments.e15_eval read --run $root/$name --splits $SPLITS --new-items $ITEMS --alias-table $ALIAS \
      --output $E15/read/t5/$name
  done
done

add --name e15-t5-report --priority $PRIO_REPORT --no-resume -- \
  $PY -m vsa_embed.experiments.e15_eval report --runs-root $E15/runs/t5 --baseline-root $E9 --qa-root $E15/qa/t5 \
  --read-root $E15/read/t5 --splits $SPLITS --host $HOST --output $E15/report/t5
