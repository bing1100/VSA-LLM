#!/usr/bin/env bash
# E9 B-B1, the bit-matched unstructured side module (author, 2026-10-20: "approve all … run them and queue"). NOT EXECUTED by
# the agent that wrote it: nothing was added to .jobs. Run from the repository root of the main checkout:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python PRIO_BB1=<slot> PRIO_BB1_REPORT=<slot> \
#     bash experiments/e9-retrofit/queue-commands-bb1.sh
# Pre-registration: experiments/e9-retrofit/preregistration-addendum-2026-10-20-bb1.md (commit it before running this script).
# Ownership: R9, claims.md and draft.md belong to workplace-9a; these jobs write only runs/<stage>/SmolLM2-360M-full-C0L-s*,
# report/<stage>-bb1 and report/bb1, and read the existing rescorings and e4_quant windows of P0 / C0′ / C2 / C5.
#
# Priorities: set PRIO_BB1 (the three stages share it; e9_plan --level-step 0.0001: training at PRIO_BB1, the rescoring at
# +0.0001, the per-stage `-bb1-` report at +0.0003) and PRIO_BB1_REPORT (the cross-stage report; give it a value above
# PRIO_BB1 + 0.0003). Every report job name contains "-report", so the CPU lane runs it, after every job ahead of it.
#
# Jobs and GPU-h (addendum §8, from measured durations in .jobs/*.json): 22 jobs ≈ 12.9 GPU-h. Training: 9 runs, measured C0′ /
# C2 / C5 360M times (T5 66.6, T4 67.1, T7 75.9 min) × 1.05 assumed adapter overhead ≈ 11.0. Rescoring: 9 runs × 9 variants,
# as the C5 controls jobs (T5 7.2, T4 7.5, T7 23.9 min) ≈ 1.9. The reports run on the CPU lane. `--dry-run` instead of
# `--queue` lists the jobs (7 per stage).
set -euo pipefail
: "${PY:?set PY to the pinned interpreter (/home/bhux/anaconda3/envs/vsa-repro/bin/python)}"
PRIO_BB1="${PRIO_BB1:?set PRIO_BB1 to the queue slot of the B-B1 block}"
PRIO_BB1_REPORT="${PRIO_BB1_REPORT:?set PRIO_BB1_REPORT to a slot after PRIO_BB1 + 0.0003 (cross-stage report, CPU lane)}"

# Prerequisites (checked, not built): the P0 / C0′ / C2 / C5 rescorings of T5, T4 and T7 at 360M, seeds 1–3 (all present on
# 2026-10-20), and the stages' e4_quant windows (variant B of C2 / C5).
for stage in t5 t4 t7; do
  for model in frozen-P0-s1 full-C0p-s1 full-C0p-s2 full-C0p-s3 full-C2-s1 full-C2-s2 full-C2-s3 full-C5-s1 full-C5-s2 full-C5-s3; do
    test -f "experiments/e9-retrofit/runs/$stage/SmolLM2-360M-$model/rescore/windows.npz" \
      || { echo "missing rescoring: $stage SmolLM2-360M-$model" >&2; exit 1; }
  done
done

# 1–3. C0L (C0′ + LoRA side module matched to C5's channel parameters; addendum §2) on T5, T4 and T7, SmolLM2-360M, seeds 1–3:
#      training + its claim-B rescoring (e9_rescore profile `side`) + the stage's `-bb1-` report (e9_bb1). The configs are
#      the ones the dry run of 2026-10-20 wrote (experiments/e9-retrofit/configs/{t5,t4,t7}/SmolLM2-360M-full-C0L-s{1,2,3}.yaml;
#      e9_plan rewrites them identically: rank patterns 2+4 / 4+11 / 2+1).
for track in t5 t4 t7; do
  PYTHONPATH=src "$PY" -m vsa_embed.experiments.e9_plan --track "$track" --hosts SmolLM2-360M --models C0L --seeds 1 2 3 \
    --priority "$PRIO_BB1" --level-step 0.0001 --queue
done

# 4. The cross-stage B-B1 report (the primary family's Holm adjustment over T5, T4 and T7; addendum §5, §7), CPU lane.
PYTHONPATH=src "$PY" -m vsa_embed.jobqueue add --name bb1-report-t5-t4-t7 --priority "$PRIO_BB1_REPORT" --min-free-gb 1 --no-resume -- \
  env PYTHONPATH=src "$PY" -m vsa_embed.experiments.e9_bb1 report --stages t5 t4 t7 --output experiments/e9-retrofit/report/bb1
