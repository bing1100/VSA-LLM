#!/usr/bin/env bash
# E5 amendment 1 (experiments/e5-explainability/preregistration-amendment-1-scale-v1.md): E5.1–E5.5 retargeted to the
# scale-v1 125M checkpoints (C5 and C0 seeds 1–3, C2 seed 1, HRRAdd/HRRCat seed 1 exploratory).
# Written 2026-10-20 at the author's request ("run them and queue"). NOT executed by its author: set the two priorities
# below, then run it from the repository root.
#
# Lanes: job names containing "-report" run on the CPU lane (runner `--cpu-pattern=-report`, CUDA hidden): E5.5 (CPU
# only by design), the E5.3 judge grading (no GPU; claude -p) and the R5 report. Every other job runs on the GPU lane.
# A CPU-lane job starts only when every job with a lower priority number is done, so PRIO_REPORT must be > PRIO_GPU.
# Within one priority, jobs run in the order added: the grade jobs follow the pre-registered budget order
# (rare neighbours -> edges -> held-out neighbours) and the report runs last.
#
# Jobs: 31 GPU-lane (E5.1 x5, E5.2 x5, E5.4 x18, E5.3 builders x3) + 13 CPU-lane (E5.5 x9, E5.3 grades x3, report) = 44.
# GPU estimate: ~3.3 GPU-h (~4.3 with a 30% margin); CPU lane ~1.7 h. Derivation in the amendment §A.7 (measured
# analogues in .jobs: 125M training 241-247 min per 500M tokens; full probe suite at 135M 2.0-2.2 min; E9 zero-shot jobs
# 0.3-1.4 min; plus the pre-registration's planning figures for E5.1 and E5.4).
# Judge: hard cap USD 25 for all of E5.3, cumulative over the shared ledger <judge-cache>/spend-ledger.jsonl; a study whose
# projected cost does not fit the remaining budget is not started ("not run (budget)", exit 0, budget-refused.json).
set -euo pipefail

PRIO_GPU="${PRIO_GPU:-PRIO_GPU}"            # placeholder: GPU-lane priority for every E5 GPU job
PRIO_REPORT="${PRIO_REPORT:-PRIO_REPORT}"   # placeholder: CPU-lane priority for every "-report" job (must be > PRIO_GPU)

for value in "$PRIO_GPU" "$PRIO_REPORT"; do
  [[ "$value" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "set PRIO_GPU and PRIO_REPORT to numbers (got '$value')" >&2; exit 1; }
done
awk -v g="$PRIO_GPU" -v r="$PRIO_REPORT" 'BEGIN { exit !(r > g) }' || { echo "PRIO_REPORT must be > PRIO_GPU" >&2; exit 1; }
[[ -f src/vsa_embed/jobqueue.py ]] || { echo "run from the repository root" >&2; exit 1; }

PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
R=experiments/e4-small-lm/runs/scale-v1
O=experiments/e5-explainability/runs/scale-v1
ITEMS=experiments/e5-explainability/items
JUDGE_USD=25

for run in 125M-C0-s1 125M-C0-s2 125M-C0-s3 125M-C5-s1 125M-C5-s2 125M-C5-s3 125M-C2-s1 125M-HRRAdd-s1 125M-HRRCat-s1; do
  [[ -f "$R/$run/final.pt" ]] || { echo "missing checkpoint $R/$run/final.pt" >&2; exit 1; }
done

gpu() { local name=$1; shift; PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name "$name" --priority "$PRIO_GPU" --no-resume -- "$@"; }
cpu() { local name=$1; shift; PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name "$name" --priority "$PRIO_REPORT" --no-resume -- "$@"; }

COMPOSED="125M-C5-s1 125M-C5-s2 125M-C5-s3 125M-HRRAdd-s1 125M-HRRCat-s1"
ALL="125M-C5-s1 125M-C5-s2 125M-C5-s3 125M-C0-s1 125M-C0-s2 125M-C0-s3 125M-C2-s1 125M-HRRAdd-s1 125M-HRRCat-s1"

# -- GPU lane ------------------------------------------------------------------------------------------------------------
# E5.1 faithfulness (1,024 windows, k 1 2 4, 2 random draws, KL); probe-level ablation (wic, wsd, k = 2) on C5 only.
for run in $COMPOSED; do
  probes=(); [[ $run == 125M-C5-* ]] && probes=(--probes wic,wsd --probe-k 2)
  gpu "e5s1-faith-$run" $PY -m vsa_embed.experiments.e5_faithfulness --run "$R/$run" --output "$O/faithfulness/$run" "${probes[@]}"
done
# E5.2 sense alignment (SemCor; WordNet frames rebuilt and checked against the entry schedule).
for run in $COMPOSED; do
  gpu "e5s1-senses-$run" $PY -m vsa_embed.experiments.e5_senses --run "$R/$run" --output "$O/senses/$run"
done
# E5.4 zero-shot insertion, both GPT-2 scenarios (all sources on channel runs; C0 runs `own` only by construction).
for run in $ALL; do
  gpu "e5s1-zs-c3h-$run" $PY -m vsa_embed.experiments.e5_zeroshot evaluate --run "$R/$run" \
    --items "$ITEMS/c3-heldout-gpt2-v1" --output "$O/zeroshot-c3h/$run"
  gpu "e5s1-zs-c3s-$run" $PY -m vsa_embed.experiments.e5_zeroshot evaluate --run "$R/$run" \
    --items "$ITEMS/c3-synthetic-gpt2-v1" --output "$O/zeroshot-c3s/$run"
done
# E5.3 item builders (pre-registered systems: C5 s1 = reference, C0 s1, C2 s1; edges from C5 s1). M3 cards dropped.
gpu e5s1-rating-build-neighbours-rare $PY -m vsa_embed.experiments.e5_rating build neighbours \
  --runs "C5=$R/125M-C5-s1" "C0=$R/125M-C0-s1" "C2=$R/125M-C2-s1" --set rare --space state --output "$O/rating/items-neighbours-rare"
gpu e5s1-rating-build-edges $PY -m vsa_embed.experiments.e5_rating build edges \
  --run "$R/125M-C5-s1" --output "$O/rating/items-edges"
gpu e5s1-rating-build-neighbours-heldout $PY -m vsa_embed.experiments.e5_rating build neighbours \
  --runs "C5=$R/125M-C5-s1" "C0=$R/125M-C0-s1" "C2=$R/125M-C2-s1" --set heldout --space state --output "$O/rating/items-neighbours-heldout"

# -- CPU lane ("-report" in every name) ----------------------------------------------------------------------------------
# E5.5 frequency disentanglement (CPU only); the first job computes the training-stream token counts and saves them.
for run in $ALL; do
  cpu "e5s1-freq-$run-report" $PY -m vsa_embed.experiments.e5_frequency --run "$R/$run" --output "$O/frequency/$run" \
    --token-counts "$O/gpt2-train-counts.npy" --device cpu
done
# E5.3 grading, LLM judge (claude-opus-5-5 via claude -p): one shared cache and ledger, hard cap USD 25 over all three,
# in the pre-registered budget order. Non-clinical text only (WordNet aliases, FineWeb-Edu sentences).
# SKIP_JUDGE=1 queues everything except the three Claude-judge grade jobs (their spend waits for the author; see the
# 2026-10-20 note in the amendment). Re-run the script without it, or add only these jobs, once the author confirms.
if [[ "${SKIP_JUDGE:-0}" == 1 ]]; then JUDGE_STUDIES=""; else JUDGE_STUDIES="neighbours-rare edges neighbours-heldout"; fi
for study in $JUDGE_STUDIES; do
  # exit 4 = cut by the cap: labelled INCOMPLETE in its output, resumable from the cache. The wrapper records it and exits 0,
  # so that one capped study cannot fail the queue job and hold back every later CPU-lane report. Other exits pass through.
  cpu "e5s1-rating-grade-$study-report" bash -c '"$@"; c=$?; if [ "$c" -eq 4 ]; then echo "INCOMPLETE (judge cap): recorded, not a queue failure"; exit 0; fi; exit "$c"' _ \
    $PY -m vsa_embed.experiments.e5_rating grade --items "$O/rating/items-$study" \
    --output "$O/rating/graded-$study" --cache "$O/rating/judge-cache" --max-usd "$JUDGE_USD" --max-new-calls 500
done
# R5 for this amendment (runs last on the CPU lane).
cpu e5s1-r5-report $PY -m vsa_embed.experiments.e5_report --inputs "$O" --output experiments/e5-explainability/report/scale-v1
