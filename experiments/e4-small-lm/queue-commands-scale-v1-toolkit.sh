#!/usr/bin/env bash
# From-scratch scaling screen, phase 1b (decision 65, author 2026-10-10): the toolkit arms at 50M x 500M tokens, seed 1
# (pre-registration experiments/e4-small-lm/preregistration-scale-v1-toolkit.md; printed by
# `python -m vsa_embed.experiments.scale_toolkit commands`).
# NOT EXECUTED by the agent that wrote it. Run from the repository root of the main checkout after merging this branch:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python bash <this file>
# Priorities: phase 1a (stage scale-v1) is at 54.3; C5dev at 54.301 runs after phase 1a's 50M C5. C5teach (54.302) is
# commented out until the teacher ontology is built (c3_teacher teach + ontology, Claude spend approved by the author).
# GPU-h: 500M tokens / 66,600 tokens/s (opscreen 50M C5, mean wall clock with evaluations and checkpoints) = 2.09 h;
# C5dev x (1 + 0.05) for the developmental tracker (CPU SMOKE upper bound, scale-v1-toolkit/smoke/overhead.json).
set -euo pipefail
: "${PY:?set PY to the pinned interpreter}"

# ---- C5dev (passive learning)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C5dev-s1 --priority 54.301 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C5dev-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5dev-s1   # ≈ 2.19 GPU-h

# ---- C5teach (learning from reading; Claude as teacher). NOT QUEUED: the teacher ontology does not exist yet.
# Recommended: the hybrid variant (teacher frames for the 5,900 held-out + 5,900 matched trained entries, WordNet
# elsewhere; ≈ $73 and ≈ 1.1 h of teacher time at 4 workers, projected from the pilot).
# After the author approves the spend, by hand (not a queue job: it calls `claude -p` and spends money; resumable, capped):
#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher teach --scope hybrid --max-usd <approved cap incl. the pilot's spend>
#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher ontology --scope hybrid
#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher reference
# PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C5teach-s1 --priority 54.302 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C5teach-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5teach-s1   # ≈ 2.09 GPU-h
# Rescoring of phase 1a's finished 50M C5 / C0 (seed 1) on the hybrid reference strata (evaluation only; ≈ 0.02 GPU-h each):
# PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C5@teachref-s1 --priority 54.302 --min-free-gb 5 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C5@teachref-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5@teachref-s1   # ≈ 0.02 GPU-h
# PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C0@teachref-s1 --priority 54.302 --min-free-gb 5 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C0@teachref-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C0@teachref-s1   # ≈ 0.02 GPU-h
# Alternative, full variant (all 117,659 synsets; ≈ $342, ≈ 5.3 h): teach / ontology --scope full, then
# PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C5teachF-s1 --priority 54.302 --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C5teachF-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5teachF-s1   # ≈ 2.09 GPU-h

# Report (added 2026-10-10 when queued): 50M C0 / C5 (phase 1a) with C5dev, CPU lane after every training below 54.311.
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-report --priority 54.311 --min-free-gb 1 --no-resume -- $PY -m vsa_embed.experiments.e4_report --runs experiments/e4-small-lm/runs/scale-v1/50M-C0-s1 experiments/e4-small-lm/runs/scale-v1/50M-C5-s1 experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5dev-s1 --output experiments/e4-small-lm/analysis/scale-v1-toolkit --title "Scale-v1 toolkit arms (50M x 500M, seed 1): C5dev vs C5 and C0"
