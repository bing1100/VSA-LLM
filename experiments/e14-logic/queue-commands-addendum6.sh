#!/usr/bin/env bash
# E14 addendum 6 (pre-registration §17): RD-12c — the RD-12 cycle's read frames under selective explicit recall.
# Queued 2026-10-20 by workplace-d2 at the author's go-ahead ("queue up any other experiments worth doing").
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
R=""
for s in 1 2 3; do
  out=experiments/e14-logic/runs/rd12c/t4-heldout/SmolLM2-360M-C5-s$s; R="$R $out"
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd12c-t4-heldout-SmolLM2-360M-C5-s$s --priority 54.4959984 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_reading recall --run experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s$s --items experiments/e14-logic/items/t4-heldout-rd12 --frames experiments/e14-logic/items/t4-heldout-rd12/frames.jsonl --alias-table /home/bhux/data/vsa-llm/e9/alias-tables/t4.json --family rd12c --null plain --output $out   # ~0.1 GPU-h
done
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd12c-report --priority 54.4959989 --no-resume -- $PY -m vsa_embed.experiments.e14_reading recall-report --family rd12c --runs $R --output experiments/e14-logic/report/rd12c
