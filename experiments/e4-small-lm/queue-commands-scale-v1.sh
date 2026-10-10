#!/usr/bin/env bash
# E4 from-scratch scaling screen, phase 1 (decision 65, author 2026-10-10; experiments/e4-small-lm/preregistration-scale-v1.md).
# NOT EXECUTED by the agent that wrote it: nothing was added to .jobs. Run from the main checkout's root after merging (the
# configs under experiments/e4-small-lm/configs/scale-v1/ come with this commit; `e4_plan --stage scale-v1` rewrites them):
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python bash experiments/e4-small-lm/queue-commands-scale-v1.sh
# 16 training jobs at priority 54.3; equal priorities run in creation order, so the order below is the run order: 20M, then
# 50M, then 125M, C0 first within a size. The trainer is resumable (retries append the default --resume; checkpoints every
# 10 min). The report (CPU lane, name ending -report) at 54.31 starts once every job ahead of it is done.
# GPU-h: the opscreen's measured 50M throughput on this RTX 3090 (C0 69.1k, C2 69.0k, C5 67.7k tokens/s; hybrids and C5sh at
# C5's) scaled by 6N (total parameters 30.3M / 51.5M / 124.4M), 499,974,144 tokens plus nine evaluations. Total ≈ 43.1 GPU-h;
# at the C4-measured 125M throughput (36k tokens/s) the 125M block is ≈ 19.4 h instead of 24.8 h (total ≈ 37.7 h).
# Disk: ≈ 4.5 GB of final.pt in all; a 125M run holds ≤ 1.5 GB of checkpoint while running (deleted when it finishes).
set -euo pipefail
: "${PY:?set PY to the pinned interpreter}"

# --- 20M block (lr 2.667e-3, micro-batch 32 × 1) ≈ 6.0 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-20M-C0-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/20M-C0-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/20M-C0-s1   # ≈ 1.19 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-20M-C5-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/20M-C5-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/20M-C5-s1   # ≈ 1.22 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-20M-C2-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/20M-C2-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/20M-C2-s1   # ≈ 1.19 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-20M-HRRAdd-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/20M-HRRAdd-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/20M-HRRAdd-s1   # ≈ 1.22 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-20M-HRRCat-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/20M-HRRCat-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/20M-HRRCat-s1   # ≈ 1.22 GPU-h

# --- 50M block (lr 2e-3, micro-batch 32 × 1: the opscreen recipe at 500M tokens) ≈ 12.3 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-50M-C0-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/50M-C0-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/50M-C0-s1   # ≈ 2.02 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-50M-C5-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/50M-C5-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/50M-C5-s1   # ≈ 2.06 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-50M-C2-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/50M-C2-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/50M-C2-s1   # ≈ 2.02 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-50M-HRRAdd-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/50M-HRRAdd-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/50M-HRRAdd-s1   # ≈ 2.06 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-50M-HRRCat-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/50M-HRRCat-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/50M-HRRCat-s1   # ≈ 2.06 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-50M-C5sh-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/50M-C5sh-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/50M-C5sh-s1   # ≈ 2.06 GPU-h

# --- 125M block (lr 1.333e-3, micro-batch 16 × 2) ≈ 24.8 GPU-h (≈ 19.4 at the measured 36k tokens/s)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-125M-C0-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/125M-C0-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/125M-C0-s1   # ≈ 4.89 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-125M-C5-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/125M-C5-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/125M-C5-s1   # ≈ 4.99 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-125M-C2-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/125M-C2-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/125M-C2-s1   # ≈ 4.89 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-125M-HRRAdd-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/125M-HRRAdd-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/125M-HRRAdd-s1   # ≈ 4.99 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name scale-v1-125M-HRRCat-s1 --priority 54.3 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1/125M-HRRCat-s1.yaml --output experiments/e4-small-lm/runs/scale-v1/125M-HRRCat-s1   # ≈ 4.99 GPU-h

# --- report (CPU lane; per-size E4 analysis, size trend, phase-2 verdict; the opscreen's seeds 1–3 as the seed reference)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-report --priority 54.31 --lane cpu --min-free-gb 1 --no-resume -- $PY -m vsa_embed.experiments.e4_scale --runs experiments/e4-small-lm/runs/scale-v1 --output experiments/e4-small-lm/analysis/scale-v1 --seed-reference experiments/e4-small-lm/runs/opscreen --overwrite   # ≈ 0.00 GPU-h (≈ 5 CPU-min)
