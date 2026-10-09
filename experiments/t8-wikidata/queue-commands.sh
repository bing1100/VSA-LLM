#!/usr/bin/env bash
# T8 Wikidata-framed track (decision 63, WP TK-B2): E9 queue commands. NOT EXECUTED by the agent that wrote them.
# Run from the repository root of the main checkout after merging this branch:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python bash experiments/t8-wikidata/queue-commands.sh
# Pre-registration: experiments/t8-wikidata/preregistration.md (committed before any run). Prerequisites (built, CPU):
# the corpora under ~/data/vsa-llm/tracks/t8-wikidata/v1 (+ hosts/qwen3), the alias table
# ~/data/vsa-llm/e9/alias-tables/t8.json and the dimension-3 items experiments/e9-retrofit/items/{new-words,edits}-t8-*-v1.
#
# Priorities (decision 63 slot, lower runs first): SmolLM2 at 54.499 with --level-step 0.0001 (training 54.499, per-run
# evaluations 54.4991, e4_quant 54.4992, R9 report 54.4993); Qwen3-1.7B at 54.4992 (54.4992 / .4993 / .4994 / .4995).
# GPU-h from `e9_plan --dry-run` (measured SmolLM2-360M minutes per 50M tokens plus the 8,192-window evaluations; Qwen3 by
# the 6N model, which the memory probe's measured 3.2 h per trained run brings to ≈ 13 h with the 8,192-window passes):
#   1. SmolLM2-360M P0, C0', C2, C5 × seeds 1–3        53 jobs  ≈ 17.1 GPU-h (training and P0 ≈ 13.0)
#   2. SmolLM2-360M WP-PQ1 arms C5ut, C5tr × seeds 1–3  19 jobs  ≈  9.5 GPU-h (training ≈ 8.7)
#   3. Qwen3-1.7B-Base LoRA r64 P0, C0', C5 seed 1      18 jobs  ≈ 18.9 GPU-h (training and P0 ≈ 12.8)
#                                                      total ≈ 45.5 GPU-h
# The base batch (1) reports to report/t8, the arm batch (2) to report/t8-pq, Qwen3 (3) to report/t8-qwen3.
set -euo pipefail
: "${PY:?set PY to the pinned interpreter}"

# 1. SmolLM2-360M base batch (primary: C5 − C0′ on after_heldout of eval-entities, seeds 1–3)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t8 --hosts SmolLM2-360M --models P0 C0p C2 C5 --seeds 1 2 3 \
  --priority 54.499 --level-step 0.0001 --queue

# 2. SmolLM2-360M WP-PQ1 operator controls (rule 7: a binding claim needs C5 better than both)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t8 --hosts SmolLM2-360M --models C5ut C5tr --seeds 1 2 3 \
  --priority 54.499 --level-step 0.0001 --queue

# 3. Qwen3-1.7B-Base (LoRA r64; seeds 2–3 only if seed 1 shows a clear effect, M6)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t8 --hosts Qwen3-1.7B-Base --host-mode lora --lora-rank 64 \
  --models P0 C0p C5 --seeds 1 --priority 54.4992 --level-step 0.0001 --queue

# Optional, not part of the pre-registered block (author decision): WP-UB understanding items on T8
# (experiments/e9-retrofit/items/understanding-t8-smollm2-v1, built), evaluated on the finished SmolLM2 runs:
#   PYTHONPATH=src $PY -m vsa_embed.experiments.e9_understanding queue --stage t8 \
#     --items experiments/e9-retrofit/items/understanding-t8-smollm2-v1 --priority 56 --dry-run
