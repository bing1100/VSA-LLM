#!/usr/bin/env bash
# T7-ROOD (decision 63, holdout H1; WP TK-H1): E9 queue commands. NOT EXECUTED by the agent that wrote them.
# Run from the repository root of the main checkout after merging this branch:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
# Pre-registration: experiments/t7-new-vocabulary/preregistration-rood.md (committed before any run). Data, items and
# configs are built (ROOD.md); `--dry-run` instead of `--queue` lists every job with its estimate and queues nothing.
# The configs these commands write are the committed ones (experiments/e9-retrofit/configs/{t7rood,t7-qwen3,t7rood-qwen3}/;
# e9_plan rewrites them identically).
#
# Priorities (fractional slots, e9_plan --level-step 0.0001: training P, per-run evaluations P+0.0001, e4_quant P+0.0002,
# report P+0.0003). They sit after the binding step 3 (54.494–54.495) and TK-Q's T4 Qwen3 block (54.496–54.4963), and
# before the 54.5 Qwen3.5 memory probe. The two Qwen3 slots overlap by design (54.4977–54.4978): equal priorities run
# first come, first served, so T7's Qwen3 evaluations and quant interleave with T7-ROOD's Qwen3 training.
#
# GPU-h (dry runs of 2026-10-08):
#   1. T7-ROOD, SmolLM2-360M, seeds 1–3, P0 C0′ C2 C5: 53 jobs ≈ 15.4 GPU-h (training and P0 ≈ 11.3; measured 360M costs
#      plus T7's 3 extra 1,024-window passes at each of 7 evaluation points). The same as T7's block.
#   2. T7, Qwen3-1.7B-Base LoRA r64, seed 1: 23 jobs ≈ 27.5 GPU-h by the 6N throughput model (training ≈ 19.1). With the
#      memory probe's measured throughput (3.2 GPU-h per trained run) ≈ 18 GPU-h, plus ≈ 1 GPU-h per trained run for the
#      4,096-window evaluations at 7 points (not in either estimate; scaled from SmolLM2-360M by parameters): ≈ 21 GPU-h.
#   3. T7-ROOD, Qwen3-1.7B-Base LoRA r64, seed 1: 23 jobs, the same as 2 (≈ 27.5 by the 6N model, ≈ 21 realistic).
#   Total ≈ 15.4 + 2 × 21 ≈ 57 GPU-h (≈ 70 GPU-h by the 6N model).
set -euo pipefail
: "${PY:?set PY to the pinned interpreter}"

# 1. T7-ROOD on SmolLM2-360M, seeds 1–3 (the primary block of the pre-registration), stage t7rood
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t7rood --hosts SmolLM2-360M --seeds 1 2 3 \
  --priority 54.497 --level-step 0.0001 --queue

# 2. T7 on Qwen3-1.7B-Base (LoRA r 64, as the T4/T5 Qwen3 configs), seed 1, stage t7-qwen3
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t7 --hosts Qwen3-1.7B-Base --host-mode lora --lora-rank 64 --seeds 1 \
  --priority 54.4975 --level-step 0.0001 --queue

# 3. T7-ROOD on Qwen3-1.7B-Base, seed 1, stage t7rood-qwen3
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t7rood --hosts Qwen3-1.7B-Base --host-mode lora --lora-rank 64 --seeds 1 \
  --priority 54.4977 --level-step 0.0001 --queue
