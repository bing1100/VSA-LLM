#!/usr/bin/env bash
# E15 — HRR as a training target, stage 1e (T5): "read, then inject"; pre-registration
# experiments/e15-hrr-target/preregistration.md, section "Stage 1e (registered 2026-10-25, before any run of it): read, then inject".
# Prepared 2026-10-25 at the author's approval ("Use 1 - implement and test").
#
# 6 training jobs at 54.4969985 (after stage 1d's report at 54.496998, before the T7 Qwen3 block at 54.4975): C5WRI s1–3
# (stage 1's C5 + WR recipe, λ_w 0.20, λ_r 0.035, layer 11, plus the injection path: decoded fillers → P → gate σ(b), b = −4,
# into the input of layer 12), C5WRI-nogate s1 (gate fixed open), C2RI s1 (free table + read loss + injection, the
# memorisation control), C5shWRI s1 (shuffled facts + WR + injection). Evaluations at 54.4969986 (`--no-resume`,
# `--min-free-gb 10`): implicit-route QA (`e14_reading recall --family e15 --null plain`), read accuracy (`e15_eval read`)
# for every run, and the gate-closed QA (`--gate-closed`, output `<run>/gate-closed/`) for the three C5WRI seeds → 15 jobs.
# The CPU-lane report `e15-t5-stage1e-report` at 54.4969987. Training jobs keep the queue's default resume (--resume on
# retry) and --min-free-gb 20, as stage 1's. Ordering guarantee (jobqueue): jobs are sorted by (priority, created) and a
# runner takes the first pending job of its lane only when no job of the lane is live, so every 54.4969985 training job
# finishes (or fails) before any 54.4969986 evaluation starts; the CPU lane (names matching `-report`) starts the report
# only once every job of lower priority is done. Baselines (E9's C5; stage 1's C5WR, C2R, C5shWR) are reused: nothing is re-run.
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_TRAIN=${PRIO_TRAIN:-54.4969985}; PRIO_EVAL=${PRIO_EVAL:-54.4969986}; PRIO_REPORT=${PRIO_REPORT:-54.4969987}
HOST=SmolLM2-360M
E15=experiments/e15-hrr-target
E9=experiments/e9-retrofit/runs/t5
ITEMS=experiments/e11-read-to-learn/items/t5-new-smollm2-v1
FRAMES=$E9/$HOST-full-C5-s1/e11-t5-new/frames.jsonl
ALIAS=/home/bhux/data/vsa-llm/e9/alias-tables/t5.json
SPLITS=$E15/splits/t5-v1/splits.json
RUNS="C5WRI-s1 C5WRI-s2 C5WRI-s3 C5WRI-nogate-s1 C2RI-s1 C5shWRI-s1"
add() { PYTHONPATH=src $PY -m vsa_embed.jobqueue add "$@"; }

for run in $RUNS; do
  name=$HOST-full-$run
  add --name e15-t5-$name --priority $PRIO_TRAIN --min-free-gb 20 -- \
    $PY -m vsa_embed.training.lm --config $E15/configs/t5/$name.yaml --output $E15/runs/t5/$name
done

for run in $RUNS; do
  name=$HOST-full-$run
  add --name e15-t5-$name-qa --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e14_reading recall --run $E15/runs/t5/$name --items $ITEMS --frames $FRAMES --alias-table $ALIAS \
    --family e15 --null plain --output $E15/qa/t5/$name
  add --name e15-t5-$name-read --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e15_eval read --run $E15/runs/t5/$name --splits $SPLITS --new-items $ITEMS --alias-table $ALIAS \
    --output $E15/read/t5/$name
done

# The injection ablation on the trained model (primary 3): the three C5WRI seeds with the gate forced closed at evaluation.
for s in 1 2 3; do
  name=$HOST-full-C5WRI-s$s
  add --name e15-t5-$name-qa-gateclosed --priority $PRIO_EVAL --min-free-gb 10 --no-resume -- \
    $PY -m vsa_embed.experiments.e14_reading recall --run $E15/runs/t5/$name --items $ITEMS --frames $FRAMES --alias-table $ALIAS \
    --family e15 --null plain --gate-closed --output $E15/qa/t5/$name/gate-closed
done

add --name e15-t5-stage1e-report --priority $PRIO_REPORT --no-resume -- \
  $PY -m vsa_embed.experiments.e15_eval report --stage 1e --runs-root $E15/runs/t5 --baseline-root $E9 --qa-root $E15/qa/t5 \
  --read-root $E15/read/t5 --splits $SPLITS --host $HOST --output $E15/report/t5-stage1e
