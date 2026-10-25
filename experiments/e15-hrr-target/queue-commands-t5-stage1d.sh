#!/usr/bin/env bash
# E15 — HRR as a training target, stage 1d (T5): 500M tokens, seed 1 of the matched-length pair; pre-registration
# experiments/e15-hrr-target/preregistration.md, section "Stage 1d (registered 2026-10-25, before any run of it): 500M tokens".
# Prepared 2026-10-25 at the author's approval ("sounds good - register the queue"), after stage 1b's rule read
# "0.6–0.8 and still rising → register a longer stage before any verdict" (read 0.765 at 200M, increment +0.010, CI > 0).
#
# 2 training jobs at 54.496996 (after stage 1c's report at 54.496995, before T7-ROOD's remaining jobs at 54.497 and E13 at
# 54.4985): C5WR-xl s1 (stage 1b's C5 + WR recipe, λ_w 0.20, λ_r 0.035, layer 11) and C5-xl s1 (E9's C5 recipe, the
# matched-length control), each 500M tokens with named checkpoints at 100M/200M/300M/400M/500M (`train.save_at_tokens`;
# the 500M file is the final state). Evaluations at 54.496997 (`--no-resume`): read accuracy (`e15_eval read`; probe only
# for C5-xl) and implicit-route QA (`e14_reading recall --family e15`) at every checkpoint of both runs → 20 jobs, outputs
# under {read,qa}/t5/<run>/ckpt-<tokens>/. The CPU-lane report `e15-t5-stage1d-report` at 54.496998. Training jobs keep
# the queue's default resume (--resume on retry) and --min-free-gb 20, as stage 1b's. Ordering guarantee (jobqueue): jobs
# are sorted by (priority, created) and a runner takes the first pending job of its lane only when no job of the lane is
# live, so both 54.496996 training jobs finish (or fail) before any 54.496997 evaluation starts; the CPU lane (names
# matching `-report`) starts the report only once every job of lower priority is done. Seeds 2–3 (the conditional
# follow-up of §18.1) are NOT queued by this script; add them on the author's go-ahead with SEEDS="2 3".
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_TRAIN=${PRIO_TRAIN:-54.496996}; PRIO_EVAL=${PRIO_EVAL:-54.496997}; PRIO_REPORT=${PRIO_REPORT:-54.496998}
SEEDS=${SEEDS:-1}
HOST=SmolLM2-360M
E15=experiments/e15-hrr-target
E9=experiments/e9-retrofit/runs/t5
ITEMS=experiments/e11-read-to-learn/items/t5-new-smollm2-v1
FRAMES=$E9/$HOST-full-C5-s1/e11-t5-new/frames.jsonl
ALIAS=/home/bhux/data/vsa-llm/e9/alias-tables/t5.json
SPLITS=$E15/splits/t5-v1/splits.json
CKPTS="100000000 200000000 300000000 400000000 500000000"
add() { PYTHONPATH=src $PY -m vsa_embed.jobqueue add "$@"; }

for s in $SEEDS; do
  for arm in C5WR-xl C5-xl; do
    name=$HOST-full-$arm-s$s
    add --name e15-t5-$name --priority $PRIO_TRAIN --min-free-gb 20 -- \
      $PY -m vsa_embed.training.lm --config $E15/configs/t5/$name.yaml --output $E15/runs/t5/$name
  done
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

for s in $SEEDS; do
  for arm in C5WR-xl C5-xl; do
    for t in $CKPTS; do
      read_acc $HOST-full-$arm-s$s $t
      qa $HOST-full-$arm-s$s $t
    done
  done
done

# The report pools the seeds named here (seed 1 at registration); when seeds 2–3 are queued later, re-add it with --seeds 1,2,3.
add --name e15-t5-stage1d-report --priority $PRIO_REPORT --no-resume -- \
  $PY -m vsa_embed.experiments.e15_eval report --stage 1d --seeds $(echo $SEEDS | tr ' ' ',') --runs-root $E15/runs/t5 --baseline-root $E9 \
  --qa-root $E15/qa/t5 --read-root $E15/read/t5 --splits $SPLITS --host $HOST --output $E15/report/t5-stage1d
