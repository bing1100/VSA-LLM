#!/usr/bin/env bash
# From-scratch scaling screen, phase 1b: arm C5teach (hybrid; learning from reading, Claude as teacher) at 50M x 500M
# tokens, seed 1, with the rescoring of phase 1a's 50M C5 / C0 on the teacher reference strata and the two reports.
# Pre-registration experiments/e4-small-lm/preregistration-scale-v1-toolkit.md (§§1-4, 6); status note
# experiments/e4-small-lm/preregistration-scale-v1-toolkit-status-2026-10-20.md (author's approval of the hybrid teacher spend,
# 2026-10-20: "approve about USD 73 in teacher calls" = §6's hybrid projection; C5teachF, about USD 342, is NOT approved and
# is not in this script).
#
# NOT EXECUTED by the agent that wrote it. Run from the repository root of the main checkout:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python PRIO_TRAIN=<p> PRIO_REPORT=<q> bash <this file>
#
# Same recipe, size, seed and job lines as the commented C5teach block of queue-commands-scale-v1-toolkit.sh (C5dev's
# lines): configs configs/scale-v1-toolkit/50M-{C5teach,C5@teachref,C0@teachref}-s1.yaml, identical to phase 1a's 50M C5 /
# C0 configs except the arm keys (data.ontology + eval.reference_strata; eval_only + init_from + eval.reference_strata).
#
# Prerequisites (by hand, before this script; not queue jobs: `teach` calls `claude -p` and spends money; resumable,
# cached, HARD cap cumulative over the store's ledger.jsonl = USD 2.409 before 2026-10-20 + USD 80 approved cap):
#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher teach --scope hybrid --max-usd 82.40
#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher ontology --scope hybrid      # -> $STORE/ontology-hybrid.pt (frozen)
#   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher reference                   # -> $STORE/reference-hybrid-1024x1024.npz
#                                                                                       #    (built 2026-10-20, sha256 9399d453...)
# The guard below refuses to queue unless the ontology exists, is the hybrid one (11,800 teacher entries, no unanswered
# synset) and was built from the current answers.jsonl (sha256 in its `teacher` record), and the reference exists.
#
# Priorities: placeholders, set the band. Lower numbers run first. PRIO_TRAIN = the three GPU-lane jobs (training + two
# evaluation-only rescorings; same priority, run in the order added). PRIO_REPORT = the two CPU-lane reports (names end in
# -report); the CPU lane starts a job only when EVERY job with a lower number is done, so PRIO_REPORT > PRIO_TRAIN.
#
# GPU-h (measured, .jobs/*.json started -> finished): C5teach = phase 1a's scale-v1-50M-C5-s1, 2.06 h wall clock (50M C0 /
# C2 / C5sh / HRRAdd / HRRCat: 2.01-2.06 h; C5dev, with the developmental tracker, 3.93 h). Rescoring: eval-only on 1,024 x
# 1,024 windows, no from-scratch analogue measured; SmolLM2-135M frozen eval-only on 1,024 x 1,024 took 0.004 h, the
# pre-registration budgets 0.02 h each. Total ~ 2.06 + 2 x 0.02 = 2.10 GPU-h; reports CPU, ~0.005 h each.
set -euo pipefail
: "${PY:?set PY to the pinned interpreter}"

PRIO_TRAIN="${PRIO_TRAIN:-}"        # <-- set: C5teach training and the two @teachref rescorings (GPU lane)
PRIO_REPORT="${PRIO_REPORT:-}"      # <-- set: the two reports (CPU lane); must be greater than PRIO_TRAIN
: "${PRIO_TRAIN:?set PRIO_TRAIN (queue band for the GPU jobs)}"
: "${PRIO_REPORT:?set PRIO_REPORT (queue band for the reports, after PRIO_TRAIN)}"
awk -v t="$PRIO_TRAIN" -v r="$PRIO_REPORT" 'BEGIN { exit !(r + 0 > t + 0) }' || { echo "PRIO_REPORT must be > PRIO_TRAIN" >&2; exit 1; }

# ---- guard: the teacher ontology is built and frozen before any C5teach job (pre-registration §1)
PYTHONPATH=src "$PY" - <<'EOF'
import hashlib, sys
from pathlib import Path
import torch
from vsa_embed.experiments.c3_teacher import STORE
from vsa_embed.experiments.scale_toolkit import TEACHER_ONTOLOGY, TEACHER_REFERENCE
problems = []
path = TEACHER_ONTOLOGY["hybrid"]
if not path.exists():
    sys.exit(f"{path} does not exist: build the teacher ontology first (c3_teacher teach / ontology --scope hybrid)")
teacher = torch.load(path, weights_only=False).get("teacher") or {}
if teacher.get("scope") != "hybrid":
    problems.append(f"scope {teacher.get('scope')!r}")
if teacher.get("teacher_entries") != 11_800:
    problems.append(f"{teacher.get('teacher_entries')} teacher entries (expected 5,900 held-out + 5,900 matched trained)")
if (teacher.get("parse") or {}).get("unanswered", 0):
    problems.append(f"{teacher['parse']['unanswered']} unanswered synsets")
if teacher.get("answers_sha256") != hashlib.sha256((STORE / "answers.jsonl").read_bytes()).hexdigest():
    problems.append("answers.jsonl changed after the ontology was built")
if not Path(TEACHER_REFERENCE).exists():
    problems.append(f"{TEACHER_REFERENCE} missing (c3_teacher reference)")
if problems:
    sys.exit("teacher ontology not ready: " + "; ".join(problems))
print(f"teacher ontology ok: {path} ({teacher['teacher_entries']} entries, {teacher['edges']} edges, "
      f"parse rate {teacher['parse']['parse_rate']:.3f}, answers sha256 {teacher['answers_sha256'][:12]})")
EOF

# ---- C5teach (hybrid): teacher frames for the 5,900 held-out + 5,900 matched trained entries, WordNet elsewhere
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C5teach-s1 --priority "$PRIO_TRAIN" --min-free-gb 20 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C5teach-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5teach-s1   # ≈ 2.06 GPU-h

# ---- Rescoring of phase 1a's finished 50M C5 / C0 (seed 1) on the hybrid reference strata (evaluation only)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C5@teachref-s1 --priority "$PRIO_TRAIN" --min-free-gb 5 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C5@teachref-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5@teachref-s1   # ≤ 0.02 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-50M-C0@teachref-s1 --priority "$PRIO_TRAIN" --min-free-gb 5 -- $PY -m vsa_embed.training.lm --config experiments/e4-small-lm/configs/scale-v1-toolkit/50M-C0@teachref-s1.yaml --output experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C0@teachref-s1   # ≤ 0.02 GPU-h

# ---- Reports (CPU lane), C5teach pooled into C5's cohort (--pool-ontologies, pre-registration §2)
# (1) the pre-registered comparisons on the trainer's strata: C5teach - C5, C5teach - C0, C5dev - C5, C5dev - C0 (C5dev
#     re-read with C5 as a reference: the paired C5dev - C5 table the first toolkit report lacked). e4_report's Holm family
#     per stratum is every comparison in the grid, here 5 (the four + C5 - C0): at least as conservative as the registered 4.
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-c5teach-report --priority "$PRIO_REPORT" --min-free-gb 1 --no-resume -- $PY -m vsa_embed.experiments.e4_report --runs experiments/e4-small-lm/runs/scale-v1/50M-C0-s1 experiments/e4-small-lm/runs/scale-v1/50M-C5-s1 experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5dev-s1 experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5teach-s1 --references C5 --pool-ontologies --output experiments/e4-small-lm/analysis/scale-v1-toolkit-c5teach --title "Scale-v1 toolkit arms (50M x 500M, seed 1): C5teach (hybrid) and C5dev vs C5 and C0"
# (2) the teacher reference strata (ref_after_teacher_heldout = after_heldout; ref_after_teacher_trained = the matched seen
#     entries): C5teach against phase 1a's C5 / C0 rescored on the same masks (C5@teachref, C0@teachref; flat eval-only
#     curves, so k(L) is undefined there and the report says so; read the paired differences).
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e4-scale-v1-toolkit-c5teach-teachref-report --priority "$PRIO_REPORT" --min-free-gb 1 --no-resume -- $PY -m vsa_embed.experiments.e4_report --runs experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C0@teachref-s1 experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5@teachref-s1 experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5teach-s1 --baseline C0@teachref --references C5@teachref --pool-ontologies --output experiments/e4-small-lm/analysis/scale-v1-toolkit-c5teach-teachref --title "Scale-v1 toolkit C5teach (hybrid, 50M x 500M, seed 1): teacher reference strata vs C5 and C0 rescored"
