#!/usr/bin/env bash
# E14 addendum 8 (pre-registration §19): X7b (one trained T4 store 768 wide against the three-store ensemble) and LG-D2
# (decision trees on noisy or partial records). Prepared 2026-10-20 for the author's approval of the tier-F follow-ups
# ("approve all … run them and queue"); NOT run by its author. X8b and LG-P3b are CPU studies that were run directly.
#
# Set the two priorities before running (the script refuses the placeholders):
#   PRIO_GPU     the GPU jobs: the C5w768 training (about 1.2 GPU-h) and the four LG-D2 scoring jobs (about 0.56 GPU-h);
#   PRIO_REPORT  the CPU-lane jobs (names end in -report): the X7b decode and the LG-D2 report. Use a number above
#                PRIO_GPU: a CPU-lane job starts only when every job ahead of it in priority order is done.
# 7 jobs: 5 GPU (about 1.75 GPU-h in all) and 2 CPU-lane. Estimates from measured .jobs durations: T4 C5 360M training
# 1.12-1.13 h; tier-P LG-D scoring 0.03/0.06/0.08/0.17 h, times 1.64 (LG-D2 has 1.64x the prompt tokens).
set -euo pipefail
PRIO_GPU="${PRIO_GPU:-PRIO_GPU}"
PRIO_REPORT="${PRIO_REPORT:-PRIO_REPORT}"
if [[ "$PRIO_GPU" == PRIO_GPU || "$PRIO_REPORT" == PRIO_REPORT ]]; then
  echo "set PRIO_GPU and PRIO_REPORT (e.g. PRIO_GPU=54.49599xx PRIO_REPORT=54.49599yy $0)" >&2
  exit 1
fi
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}

# X7b — training: C5 s1's resolved recipe with channel.dimension 768 (1.12-1.13 GPU-h for the 256-wide C5 runs; resumable).
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-x7b-train-SmolLM2-360M-C5w768-s1 --priority "$PRIO_GPU" --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e14-logic/configs/x7b/SmolLM2-360M-full-C5w768-s1.yaml --output experiments/e14-logic/runs/x7b/SmolLM2-360M-full-C5w768-s1   # ~1.2 GPU-h
# X7b — decode (CPU lane): X7's slots for s1-s3, ensemble-3, C5w768, the untrained init-256 / init-768 controls (and E9's C5w512 s1-s3 if they exist by then).
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-x7b-report --priority "$PRIO_REPORT" --lane cpu --no-resume -- $PY -m vsa_embed.experiments.e14_concat x7b --wide experiments/e14-logic/runs/x7b/SmolLM2-360M-full-C5w768-s1 --output experiments/e14-logic/runs/lga-concat-x7b   # CPU, a few minutes

# LG-D2 — the trees-noisy set (built 2026-10-20 by `e14_items noisy`; manifest experiments/e14-logic/items/manifest-noisy-s1.json).
R=""
for h in "H-135 SmolLM2-135M 0.05" "H-360 SmolLM2-360M 0.10" "H-q06 Qwen3-0.6B 0.14" "H-q17 Qwen3-1.7B 0.27"; do
  set -- $h
  out=experiments/e14-logic/runs/lgd2/$2; R="$R $out"
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-lgd2-$2 --priority "$PRIO_GPU" --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_score score --host $1 --items /home/bhux/data/vsa-llm/e14/items --sets trees-noisy --output $out --data /home/bhux/data/vsa-llm/e14/runs/lgd2/$2   # ~$3 GPU-h
done
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-lgd2-report --priority "$PRIO_REPORT" --lane cpu --no-resume -- $PY -m vsa_embed.experiments.e14_score noisy-report --runs $R --output experiments/e14-logic/report/lgd2
