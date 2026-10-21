#!/usr/bin/env bash
# E14 addendum 11 (pre-registration-addendum-11-cheap.md): LG-CF chains of what-if edits over decision trees (3 frozen hosts)
# and RD-14 inferred facts under selective recall (SmolLM2-360M C5, T4 held-out, seeds 1-3). Prepared 2026-10-20 at the
# author's "Let's also do some of the cheaper tests." GPU jobs at PRIO_GPU; the two CPU-lane reports at PRIO_REPORT (above
# PRIO_GPU, so they wait for their inputs).
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_GPU=${PRIO_GPU:-54.4959997}; PRIO_REPORT=${PRIO_REPORT:-54.49599987}
ITEMS=/home/bhux/data/vsa-llm/e14/items
T4_ITEMS=experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1
T4_ALIASES=/home/bhux/data/vsa-llm/e9/alias-tables/t4.json

# LG-CF: the trees-cf set (built 2026-10-20 by `e14_items counterfactual`; manifest experiments/e14-logic/items/manifest-cf-s1.json,
# rows sha256 4cba3f98...; file sha256 7943555f...). Estimates from LG-D's measured durations x 1.17 sequences x ~2 tokens.
LGCF_RUNS=""
for pair in H-360:SmolLM2-360M:0.14 H-q06:Qwen3-0.6B:0.19 H-q17:Qwen3-1.7B:0.39; do
  IFS=: read -r host name hours <<< "$pair"
  out=experiments/e14-logic/runs/lgcf/$name
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-lgcf-$name --priority "$PRIO_GPU" --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_score score --host $host --items $ITEMS --sets trees-cf --output $out --data /home/bhux/data/vsa-llm/e14/runs/lgcf/$name   # ~$hours GPU-h
  LGCF_RUNS="$LGCF_RUNS $out"
done
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-lgcf-report --priority "$PRIO_REPORT" --lane cpu --no-resume -- $PY -m vsa_embed.experiments.e14_score cf-report --runs $LGCF_RUNS --output experiments/e14-logic/report/lgcf

# RD-14: RD-4b's items, frames (each run's own E11 oracle frames) and scoring; ~0.07 GPU-h per seed (RD-4b's T4 jobs: 0.042-0.047 h).
RD14_RUNS=""
for s in 1 2 3; do
  out=experiments/e14-logic/runs/rd14/t4-heldout/SmolLM2-360M-C5-s$s
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd14-t4-heldout-SmolLM2-360M-C5-s$s --priority "$PRIO_GPU" --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_inferred recall --run experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s$s --items $T4_ITEMS --alias-table $T4_ALIASES --output $out
  RD14_RUNS="$RD14_RUNS $out"
done
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd14-report --priority "$PRIO_REPORT" --lane cpu --no-resume -- $PY -m vsa_embed.experiments.e14_inferred report --runs $RD14_RUNS --output experiments/e14-logic/report/rd14
