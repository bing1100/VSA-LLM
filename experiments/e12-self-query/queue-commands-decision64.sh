#!/usr/bin/env bash
# E12 decision 64 (amendment 16.5, 2026-10-09): 3b's C0′-host and C5rf-store arms; 3c's text competitors to K1 (K1b).
# NOT QUEUED — this script has not been executed. Written 2026-10-09 with the code and amendment 16.5, before any 3b / 3c run.
# Run from the main checkout after merging this branch.
#
# Queue state when written (main checkout `.jobs`, 2026-10-09): every 3b / 3c job pending, none started — 3b 18 GPU jobs at
# 54.4497, 3c 10 GPU jobs at 54.4498, reports t5-report-e12-3b / t5-report-e12-3c at 54.44985 (CPU lane).
#
# CANCEL: none. Every queued 3b / 3c command line stays valid under amendment 16.5's code:
#   - 3b (t5-SmolLM2-360M-full-C5{,ut}-s{1,2,3}-self-query-traces-*): unchanged; the new arms are new jobs below.
#   - 3c evaluate (t5-SmolLM2-360M-full-C5{,ut}-s{1,2,3}-self-query-critique): the text competitors are scored by default
#     (`--competitors auto`: on for the C5 store, off for C5ut's role-blind store); same command lines.
#   - 3c model loops (t5-qwen3-…-self-query-critique-loop) and both reports: unchanged; the reports discover the new arms and
#     compute K1b and the decision-64 contrasts.
#   Exception: a C5 critique job that finished before this branch was merged has no `symbolic` / `definition` scores in its
#   items.jsonl.gz. Re-run it (same command line): PYTHONPATH=src "$PY" -m vsa_embed.jobqueue retry <job name>.
#
# ADD (GPU lane; free slots in the 3b band, ahead of 3c's 54.4498 and the reports' 54.44985):
#   54.44971  C0′ host (no channel; its tool reads C5's store of the same seed): T, base × seeds 1–3  ≈ 2.2 GPU-h (6 jobs)
#   54.44972  C5rf (its own fixed-random store): T, I (no agentic episodes), L × seeds 1–3          ≈ 2.7 GPU-h (9 jobs)
#   GPU-h per job (upper bounds; amendment 16.3's rates — training 0.031 GPU-h per M tokens with padding, tests ≈ 7 min,
#   agentic episodes ≈ 10 min): T 0.45, base 0.28, I 0.17, L 0.28 → ≈ 4.9 GPU-h for the 15 jobs.
#   3c: no new job; the competitors add ≈ 0.8 GPU-h to the three queued C5 critique jobs (≈ 25 → ≈ 41 min each).
#   Reports: no new job (t5-report-e12-3b / -3c at 54.44985 start after every job ahead of them).
# Total new GPU work ≈ 5.7 GPU-h (4.9 in new jobs + 0.8 inside queued ones).
# Jobs are idempotent by name; each passes --overwrite and --no-resume (a retried job rewrites its partial output).
# The same jobs: `python -m vsa_embed.experiments.e12_traces commands --set decision64`.
set -euo pipefail
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
I=experiments/e9-retrofit/items
R=experiments/e9-retrofit/runs
TRAIN=experiments/e12-self-query/items/traces-t5-smollm2-v1
add() { PYTHONPATH=src "$PY" -m vsa_embed.jobqueue add "$@"; }
TESTS=(--train-items "$TRAIN" --twins "$I/role-twins-t5-smollm2-v1" --new-words "$I/new-words-t5-smollm2-v2"
       --understanding "$I/understanding-t5-smollm2-v1" --batch-size 24 --max-length 1024)

# ---- 3b, the C0′ host reading C5's store of its seed: arms T and base, seeds 1-3: priority 54.44971 ----
for s in 1 2 3; do
  for arm in T base; do
    add --name "t5-SmolLM2-360M-full-C0p-s$s-self-query-traces-$arm" --priority 54.44971 --min-free-gb 5 --no-resume -- \
      "$PY" -m vsa_embed.experiments.e12_traces run --run "$R/t5/SmolLM2-360M-full-C0p-s$s" --arm "$arm" \
      --store "$R/t5/SmolLM2-360M-full-C5-s$s" "${TESTS[@]}" --overwrite
  done
done

# ---- 3b, C5rf with its own fixed-random store: arms T, I (no agentic episodes) and L, seeds 1-3: priority 54.44972 ----
for s in 1 2 3; do
  for arm in T I L; do
    extra=()
    if [[ "$arm" == I ]]; then extra=(--agent-pairs 0); fi
    add --name "t5-SmolLM2-360M-full-C5rf-s$s-self-query-traces-$arm" --priority 54.44972 --min-free-gb 5 --no-resume -- \
      "$PY" -m vsa_embed.experiments.e12_traces run --run "$R/t5/SmolLM2-360M-full-C5rf-s$s" --arm "$arm" "${TESTS[@]}" \
      ${extra[@]+"${extra[@]}"} --overwrite
  done
done
