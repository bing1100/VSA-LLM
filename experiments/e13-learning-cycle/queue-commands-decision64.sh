#!/usr/bin/env bash
# E13 pre-registration amendment 2 (decision 64, author 2026-10-09): queue commands.
# NOT EXECUTED by the agent that wrote them (nothing was queued, cancelled or changed in .jobs/). Printed by
# `python -m vsa_embed.experiments.e13_cycle plan --config experiments/e13-learning-cycle/{t5,t7-rood}.yaml` (the lines of
# the jobs amendment 2 adds; the other 138 lines are byte-identical to queue-commands.sh, i.e. to the queued jobs).
# Run from the repository root of the main checkout AFTER merging this branch (the new trainer configs
# configs/{t5,t7-rood}/round2/Qwen3-1.7B-Base-random-s1.yaml and the `context` subcommand must exist there):
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
#
# Checked on 2026-10-09 before writing: all 138 e13-* jobs in .jobs/ are `pending` (none done / running; no e13 log, no
# experiments/e13-learning-cycle/runs/), so amendment 2 covers every E13 result.
#
# CANCEL: (none)
#   No queued command line changes: every queued job's command and priority is reproduced exactly by the amended plan
#   (tests/test_e13_amendment2.py checks it against queue-commands.sh). What changes is read at run time from files these
#   commands already name:
#   - e13-t7-rood-*-s*-reason (4 jobs): t7-rood.yaml now lists the `definition` condition (`reason.definition_source:
#     read_set`, the read set's SCR notes); ≈ +0.05 GPU-h each (the relation items of the sampled anchors with a note).
#   - e13-{t5,t7-rood}-*-reason (8 jobs): the code records the prompt tokens each condition adds per item (`added_tokens`).
#   - e13-t5-report, e13-t7-rood-report: the code computes L1's gate (read − random; Holm over six tests), the definition
#     comparators with their token costs and L5's cost criterion. Both reports stay last in their band (54.4989 / 54.49979).
#   These need this branch merged before the first E13 reason job starts (54.4988 / 54.49975, after stages 0, 4 and 5).
#   A reason job that ran before the merge must be re-run (cancel it, re-add its line from queue-commands.sh).
#
# ADD (10 jobs, ≈ 3.6 GPU-h: T5 ≈ 1.41, T7-ROOD ≈ 2.23), each at a level no job used on 2026-10-09:
#   54.49865 T5 stage 4 (after 54.4986, before stage 5 at 54.4987): Qwen3-1.7B-Base `random` (L1's gate on the secondary host)
#   54.49885 T5 `context` (after reason at 54.4988, before the report at 54.4989): the noread arm's step-0 model without and
#            with the round-2 definitions in context (L1's definition comparator), SmolLM2-360M s1–s3, Qwen3-1.7B-Base s1
#   54.49967 T7-ROOD stage 4 (after 54.49965, before stage 5 at 54.4997): Qwen3-1.7B-Base `random`
#   54.49977 T7-ROOD `context` (after reason at 54.49975, before the report at 54.49979)
# GPU-h: `random` as the other Qwen3 round-2 arms (the measured E9 tokens/s plus evaluation passes over 1,024 / 2,356
# windows); `context` ≈ 2.5 evaluation passes over the round-2 windows plus loading (`e13_cycle.context_hours`).
set -euo pipefail
: "${PY:?set PY to the pinned interpreter}"

# ---------------------------------------------------------------- T5 (54.49865, 54.49885)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t5-Qwen3-1.7B-Base-random-s1 --priority 54.49865 --min-free-gb 20 -- $PY -m vsa_embed.experiments.e13_cycle round2 --config experiments/e13-learning-cycle/configs/t5/round2/Qwen3-1.7B-Base-random-s1.yaml --output experiments/e13-learning-cycle/runs/t5/round2/Qwen3-1.7B-Base-random-s1   # ≈ 1.18 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t5-SmolLM2-360M-s1-context --priority 54.49885 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t5.yaml --arm experiments/e13-learning-cycle/configs/t5/round2/SmolLM2-360M-noread-s1.yaml --output experiments/e13-learning-cycle/runs/t5/cycle/SmolLM2-360M-s1/context   # ≈ 0.04 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t5-SmolLM2-360M-s2-context --priority 54.49885 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t5.yaml --arm experiments/e13-learning-cycle/configs/t5/round2/SmolLM2-360M-noread-s2.yaml --output experiments/e13-learning-cycle/runs/t5/cycle/SmolLM2-360M-s2/context   # ≈ 0.04 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t5-SmolLM2-360M-s3-context --priority 54.49885 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t5.yaml --arm experiments/e13-learning-cycle/configs/t5/round2/SmolLM2-360M-noread-s3.yaml --output experiments/e13-learning-cycle/runs/t5/cycle/SmolLM2-360M-s3/context   # ≈ 0.04 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t5-Qwen3-1.7B-Base-s1-context --priority 54.49885 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t5.yaml --arm experiments/e13-learning-cycle/configs/t5/round2/Qwen3-1.7B-Base-noread-s1.yaml --output experiments/e13-learning-cycle/runs/t5/cycle/Qwen3-1.7B-Base-s1/context   # ≈ 0.11 GPU-h

# ---------------------------------------------------------------- T7-ROOD (54.49967, 54.49977)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t7-rood-Qwen3-1.7B-Base-random-s1 --priority 54.49967 --min-free-gb 20 -- $PY -m vsa_embed.experiments.e13_cycle round2 --config experiments/e13-learning-cycle/configs/t7-rood/round2/Qwen3-1.7B-Base-random-s1.yaml --output experiments/e13-learning-cycle/runs/t7-rood/round2/Qwen3-1.7B-Base-random-s1   # ≈ 1.81 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t7-rood-SmolLM2-360M-s1-context --priority 54.49977 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t7-rood.yaml --arm experiments/e13-learning-cycle/configs/t7-rood/round2/SmolLM2-360M-noread-s1.yaml --output experiments/e13-learning-cycle/runs/t7-rood/cycle/SmolLM2-360M-s1/context   # ≈ 0.06 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t7-rood-SmolLM2-360M-s2-context --priority 54.49977 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t7-rood.yaml --arm experiments/e13-learning-cycle/configs/t7-rood/round2/SmolLM2-360M-noread-s2.yaml --output experiments/e13-learning-cycle/runs/t7-rood/cycle/SmolLM2-360M-s2/context   # ≈ 0.06 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t7-rood-SmolLM2-360M-s3-context --priority 54.49977 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t7-rood.yaml --arm experiments/e13-learning-cycle/configs/t7-rood/round2/SmolLM2-360M-noread-s3.yaml --output experiments/e13-learning-cycle/runs/t7-rood/cycle/SmolLM2-360M-s3/context   # ≈ 0.06 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e13-t7-rood-Qwen3-1.7B-Base-s1-context --priority 54.49977 --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e13_cycle context --config experiments/e13-learning-cycle/t7-rood.yaml --arm experiments/e13-learning-cycle/configs/t7-rood/round2/Qwen3-1.7B-Base-noread-s1.yaml --output experiments/e13-learning-cycle/runs/t7-rood/cycle/Qwen3-1.7B-Base-s1/context   # ≈ 0.24 GPU-h
