#!/usr/bin/env bash
# E12 3b (learning from tool-using traces, pre-registration §12) and 3c (calibrated self-critique, §13) — queue commands.
# QUEUED 2026-10-08 (amendment 16.4, `10c4fa3`); do not run again (`jobqueue add` refuses an existing name). Written the same
# day with the harnesses (`e12_traces`, `e12_critique`) and amendment 16.3, before any 3b / 3c run.
# Queue state 2026-10-09 (main checkout `.jobs`): all 30 jobs pending, none started — 3b 18 GPU jobs at 54.4497, 3c 10 GPU
# jobs at 54.4498 (the base-host model loop of 16.4 included), reports t5-report-e12-3b / -3c at 54.44985.
# Decision 64 (amendment 16.5, 2026-10-09) adds 3b's C0′-host and C5rf-store arms and 3c's text competitors without changing
# any command line below: see `queue-commands-decision64.sh`.
#
# Priorities (author): 3b GPU jobs 54.4497, 3c GPU jobs 54.4498, both reports 54.44985 (CPU lane; names contain "-report",
# so a report starts only after every job ahead of it in priority order is done).
# GPU-h (upper bounds; amendments 16.3 and 16.4): 3b ≈ 5.2 (18 jobs: training ≈ 1.6, tests ≈ 2.1, agentic episodes ≈ 1.5),
# 3c ≈ 4.0 (10 jobs: C5 × 3 ≈ 1.25, C5ut × 3 ≈ 0.7, model loop × 2 ≈ 1.0, base-host model loop × 2 ≈ 1.0); reports 0.
# Amendment 16.5 adds ≈ 0.8 GPU-h to the three C5 critique jobs (the text competitors, `--competitors auto`).
# Jobs are idempotent by name; each passes --overwrite and --no-resume (a retried job rewrites its partial output).
set -euo pipefail
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
I=experiments/e9-retrofit/items
R=experiments/e9-retrofit/runs
TRAIN=experiments/e12-self-query/items/traces-t5-smollm2-v1
add() { PYTHONPATH=src "$PY" -m vsa_embed.jobqueue add "$@"; }

# ---- 3b: LoRA arms on the T5 SmolLM2-360M C5 runs (T, I, S, L, base) and I-ut on C5ut, seeds 1-3: priority 54.4497 ----
for s in 1 2 3; do
  for arm in T I S L base; do
    add --name "t5-SmolLM2-360M-full-C5-s$s-self-query-traces-$arm" --priority 54.4497 --min-free-gb 5 --no-resume -- \
      "$PY" -m vsa_embed.experiments.e12_traces run --run "$R/t5/SmolLM2-360M-full-C5-s$s" --arm "$arm" --train-items "$TRAIN" \
      --twins "$I/role-twins-t5-smollm2-v1" --new-words "$I/new-words-t5-smollm2-v2" --understanding "$I/understanding-t5-smollm2-v1" \
      --batch-size 24 --max-length 1024 --overwrite
  done
  # I-ut: arm I on C5ut (the twins' stores are identical there); no agentic episodes (its tool reads a role-blind store)
  add --name "t5-SmolLM2-360M-full-C5ut-s$s-self-query-traces-I" --priority 54.4497 --min-free-gb 5 --no-resume -- \
    "$PY" -m vsa_embed.experiments.e12_traces run --run "$R/t5/SmolLM2-360M-full-C5ut-s$s" --arm I --train-items "$TRAIN" \
    --twins "$I/role-twins-t5-smollm2-v1" --new-words "$I/new-words-t5-smollm2-v2" --understanding "$I/understanding-t5-smollm2-v1" \
    --batch-size 24 --max-length 1024 --agent-pairs 0 --overwrite
done

# ---- 3c: critique inputs on C5 (twins, new words, held-out terms) and C5ut (twins, new words), seeds 1-3: priority 54.4498 ----
for s in 1 2 3; do
  add --name "t5-SmolLM2-360M-full-C5-s$s-self-query-critique" --priority 54.4498 --min-free-gb 5 --no-resume -- \
    "$PY" -m vsa_embed.experiments.e12_critique evaluate --run "$R/t5/SmolLM2-360M-full-C5-s$s" --sets twins,new,heldout \
    --batch-size 24 --max-length 1024 --overwrite
  add --name "t5-SmolLM2-360M-full-C5ut-s$s-self-query-critique" --priority 54.4498 --min-free-gb 5 --no-resume -- \
    "$PY" -m vsa_embed.experiments.e12_critique evaluate --run "$R/t5/SmolLM2-360M-full-C5ut-s$s" --sets twins,new \
    --batch-size 24 --max-length 1024 --overwrite
done
# the model loop (secondary): Qwen3-1.7B-Base C5, seeds 1-2, in bf16 (the output head in vocabulary slices)
for s in 1 2; do
  add --name "t5-qwen3-Qwen3-1.7B-Base-lora-C5-s$s-self-query-critique-loop" --priority 54.4498 --min-free-gb 5 --no-resume -- \
    "$PY" -m vsa_embed.experiments.e12_critique loop --run "$R/t5-qwen3/Qwen3-1.7B-Base-lora-C5-s$s" --items 300 \
    --host-dtype bfloat16 --batch-size 4 --max-length 4096 --overwrite
done
# Secondary (amendment 16.4, 2026-10-08): the model loop on the untouched base host reading the C5 store, as 3a's
# amendment 16.2 did for the agent (≈ +1.0 GPU-h).
for s in 1 2; do
  add --name "t5-qwen3-Qwen3-1.7B-Base-frozen-P0-C5store-s$s-self-query-critique-loop" --priority 54.4498 --min-free-gb 5 --no-resume -- \
    "$PY" -m vsa_embed.experiments.e12_critique loop --run "$R/t5-qwen3/Qwen3-1.7B-Base-frozen-P0-s1" \
    --store "$R/t5-qwen3/Qwen3-1.7B-Base-lora-C5-s$s" --output "experiments/e12-self-query/critique-loop/qwen3-1.7b-base-C5store-s$s" \
    --items 300 --host-dtype bfloat16 --batch-size 4 --max-length 4096 --overwrite
done

# ---- reports: priority 54.44985, CPU lane ----
add --name t5-report-e12-3b --priority 54.44985 --lane cpu --min-free-gb 0 --no-resume -- \
  "$PY" -m vsa_embed.experiments.e12_traces report --runs "$R/t5" --hosts SmolLM2-360M \
  --output experiments/e12-self-query/report/t5-3b --overwrite --title "E12 3b — learning from tool-using traces (T5 SmolLM2-360M)"
add --name t5-report-e12-3c --priority 54.44985 --lane cpu --min-free-gb 0 --no-resume -- \
  "$PY" -m vsa_embed.experiments.e12_critique report --runs "$R/t5" --hosts SmolLM2-360M --loop-runs "$R/t5-qwen3" \
  --output experiments/e12-self-query/report/t5-3c --overwrite --title "E12 3c — calibrated self-critique (T5 SmolLM2-360M)"
