#!/usr/bin/env bash
# Decision 64 (author, 2026-10-09), the rest after the coordinator's 0cd0c1c: exact queue changes. NOT EXECUTED by the
# agent that wrote it (nothing was added to or removed from .jobs). Run from the main checkout's root after merging:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python bash experiments/e9-retrofit/queue-commands-decision64-rest.sh
# Each part first lists the existing jobs it replaces (`# CANCEL:`). Move those jobs to .jobs/cancelled, as for T8's
# C5ut / C5tr seeds 2–3, BEFORE running that part's adds: the replacements reuse the names, and `jobqueue add` refuses an
# existing name. Cancel a job only while it is still pending. If one has already run, keep it and use the part's
# FALLBACK line.
# Equal priorities run in creation order, so each part re-adds jobs in dependency order.
# GPU-h are idle-GPU estimates from measured costs (sources per part).
# Pre-registration amendments (all dated 2026-10-09, written before any affected run):
#   part 1 — experiments/t7-new-vocabulary/preregistration-rood.md §8, experiments/t8-wikidata/preregistration.md §8.1,
#            experiments/t1c-clinical/rood/preregistration.md §15.3, resources/plan-improvement/execution.md (T7 section);
#   part 2 — experiments/t1c-clinical/rood/preregistration.md §15.3;
#   part 3 — experiments/e9-retrofit/preregistration-binding.md §12.5 / §13.1;
#   part 4 — experiments/e10-self-semantics/e10-learn/preregistration.md §12 (amendment 2).
# Total new GPU work ≈ 5.9 GPU-h (part 2 ≈ 1.1 over the replaced jobs; part 3 ≈ 4.8); part 4 ≈ 3.7 CPU-h.
set -euo pipefail
: "${PY:?set PY to the pinned interpreter}"
Q="env PYTHONPATH=src $PY -m vsa_embed.jobqueue add"

# ======================================================================================================================
# Part 1 — E9 controls on T7, T7-ROOD, T8 and T1c-ROOD: amendments only.
# C5sh and C6d × seeds 1–3 are already queued by the coordinator (0cd0c1c: T7 51, T7-ROOD 54.497, T8 54.499, T1c-ROOD
# 54.49981 --no-evals), and T8's C5ut / C5tr seeds 2–3 are already in .jobs/cancelled. Nothing to cancel or add.
# The rules are read in the existing arm-batch reports: report/t7-pq, report/t7rood-pq, report/t8-pq, and the T1c-ROOD
# stage report (t1crood-e9-report).
# ======================================================================================================================
# CANCEL: (none)

# ======================================================================================================================
# Part 2 — T1c-ROOD coding head gains composed_head_untyped (secondary specificity condition; §15.3).
# rood.yaml now lists it as its 8th default condition. The P0 jobs below name the conditions explicitly anyway (the same
# list as the new default, so the run folder keeps its name), so a job started before the merge fails loudly instead of
# silently training 7 heads.
# The P0 analysis is re-added after the P0 trainings, because a job at the same priority runs in creation order.
# The composed_c5-only jobs (t1crood-train-P0-360M-c5dict-s*), the encodes, the rescoring and every other analysis are
# unchanged; the analyses pick the condition up without a command change.
# GPU-h: decision 63's 0.75 / 0.85 per seed for 7 / 8 conditions, + ≈ 0.12 for the untyped composed head (a composed
# condition costs about one-seventh of a 7-condition seed). Replaced 9 jobs ≈ 8.3 GPU-h, of which ≈ 1.1 is new.
# ======================================================================================================================
# CANCEL: t1crood-train-P0-360M-s1 t1crood-train-P0-360M-s2 t1crood-train-P0-360M-s3 t1crood-analyze-P0-360M
# CANCEL: t1crood-train-C0p-ROOD-360M-s1 t1crood-train-C0p-ROOD-360M-s2 t1crood-train-C0p-ROOD-360M-s3
# CANCEL: t1crood-train-C5-ROOD-360M-s1 t1crood-train-C5-ROOD-360M-s2 t1crood-train-C5-ROOD-360M-s3
ROOD="--config experiments/t1c-clinical/rood/rood.yaml"
M="$PY -m vsa_embed.experiments.t1c_icd_frequency"
EIGHT="free composed_head transe title random gram composed_free composed_head_untyped"
NINE="free composed_head composed_c5 transe title random gram composed_free composed_head_untyped"
C5RUN=experiments/e9-retrofit/runs/t1c-rood/SmolLM2-360M-full-C5-s1
for s in 1 2 3; do
  $Q --name t1crood-train-P0-360M-s$s --priority 54.4998 --min-free-gb 10 --no-resume -- \
    $M train $ROOD --encoder P0-360M --seed $s --conditions $EIGHT   # 8 conditions + free_mean / free_zero ≈ 0.87 GPU-h
done
$Q --name t1crood-analyze-P0-360M --priority 54.4998 --min-free-gb 2 --no-resume -- \
  $PY -m vsa_embed.experiments.t1c_rood analyze $ROOD --encoder P0-360M --seeds 1 2 3 --device cuda   # R1 + binding ≈ 0.12 GPU-h
for enc in C0p-ROOD-360M C5-ROOD-360M; do for s in 1 2 3; do
  $Q --name t1crood-train-$enc-s$s --priority 54.49982 --min-free-gb 10 --no-resume -- \
    $M train $ROOD --encoder $enc --seed $s --conditions $NINE --c5-run $C5RUN   # ≈ 0.97 GPU-h
done; done
# FALLBACK, only for a head job that had already FINISHED (do not cancel it then): train the untyped head alone into the
# same heads folder. Its seed is name-keyed and the batch order depends on the seed only, so the result equals the joint
# job's (tested). ≈ 0.15 GPU-h each. Example for P0 seed 1 (C0p/C5-ROOD: same, with their --encoder, at 54.49982):
# $Q --name t1crood-train-P0-360M-untyped-s1 --priority 54.4998 --min-free-gb 10 --no-resume -- \
#   $M train $ROOD --encoder P0-360M --seed 1 --conditions composed_head_untyped

# ======================================================================================================================
# Part 3 — readout arm U5rf (a fixed random unitary operator, as C5rf), T5 SmolLM2-360M × seeds 1–3, in the readout
# arms' band as found in .jobs (not the audit's 54.4 / 55 / 57): training 54.491, evaluations 54.492 (the readout chain).
# The step-2 binding report (t5-report-binding-step2, 54.493) and the R9 readout batch report
# (t5-report-s1-2-3-readout-SmolLM2-360M, 54.4935) already exist and read every run of the stage. e9_plan skips the
# existing report name, so it adds 30 jobs.
# GPU-h (e9_plan --dry-run): 3 trainings × 1.28 + 27 evaluations ≈ 1.0 → ≈ 4.8 GPU-h. The configs are committed
# (experiments/e9-retrofit/configs/t5/SmolLM2-360M-full-U5rf-s{1,2,3}.yaml).
# ======================================================================================================================
# CANCEL: (none)
env PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t5 --hosts SmolLM2-360M --models U5rf --seeds 1 2 3 \
  --priority 54.491 --level-step 0.001 --queue
# OPTIONAL (not part of §13.1's endpoint): U5rf in E12 (twins core + F1, as the other readout arms, ≈ 0.8 GPU-h) and in
# step 3 (CPU lane, minutes). The float priorities match those arms' jobs.
# for s in 1 2 3; do R=experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-U5rf-s$s; N=t5-SmolLM2-360M-full-U5rf-s$s
#   $Q --name $N-self-query-role-twins-t5-smollm2-v1 --priority 54.4927 --min-free-gb 5 --no-resume -- $PY -m vsa_embed.experiments.e12_self_query \
#     evaluate --run $R --items experiments/e9-retrofit/items/role-twins-t5-smollm2-v1 --overwrite --batch-size 24 --max-length 768 \
#     --conditions none,recall:own,symbolic,roleless:own
#   $Q --name $N-self-query-faithfulness --priority 54.4927 --min-free-gb 5 --no-resume -- $PY -m vsa_embed.experiments.e12_faithfulness \
#     evaluate --run $R --overwrite --batch-size 32
#   $Q --name $N-binding-chain --priority 54.4925 --lane cpu --min-free-gb 1 --no-resume -- $PY -m vsa_embed.experiments.e9_binding_chain \
#     chain --run $R --overwrite --items experiments/e9-retrofit/items/understanding-t5-smollm2-v1
# done

# ======================================================================================================================
# Part 4 — E10.L amendment 2: the fixed-random-operator store (C5rf) on T4 / T5.
# Every erased arm (c2, features, c5full) × seeds 1–3 is re-run with store = the C5rf run of the same seed. Every other
# input is identical, and the evidence comes from the existing extract jobs (no GPU).
# Then, per track, a pooled report of the C5rf runs and the paired learned-vs-fixed store report.
# Placed in E10.L's free slots: erased runs 54.49957, reports 54.49958, all on the CPU lane (behind the C5-store runs at
# 54.49955).
# CPU-h (`e10_learn queue --rf-store --dry-run`, scaled from the CPU smokes as §9): 18 erased runs ≈ 3.4 + 4 reports ≈ 0.3
# → ≈ 3.7 CPU-h, 0 GPU-h. 22 jobs.
# ======================================================================================================================
# CANCEL: (none)
env PYTHONPATH=src $PY -m vsa_embed.experiments.e10_learn queue --rf-store
