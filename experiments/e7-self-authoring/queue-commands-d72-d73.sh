#!/usr/bin/env bash
# E7 after D7.1: the D7.1 CPU and Claude follow-ups, the G3 gate, then queueing D7.2 and D7.3 (author, 2026-10-20:
# "approve all … run them and queue"). Prepared 2026-10-20; NOT run by the agent that wrote it.
# Pre-registration: experiments/e7-self-authoring/preregistration-2026-10-20-reschedule.md.
#
# Steps, in order. Each step checks its own inputs and refuses to go on without them:
#   followups  after the 9 D7.1 GPU jobs are done: Hearst, random frames, Claude teacher (capped), quality, judge items,
#              then the G3 check (CPU only; the teacher reads FineWeb-Edu excerpts, an open licence)
#   judge      Claude judge of non-gold edges (capped; informational for G3; sequential, may take many hours)
#   d72        G3 check (stage d72) -> PASS: write the pre-registered D7.2 settings to round.json, queue D7.2 at PRIO_D72
#   d73        after the D7.2 jobs verify-<consumer>-s1 and round1-self-s1 are done: G3 check (stage d73) -> PASS:
#              cross-prepare (CPU, builds the GPT-2 corpora), then queue D7.3 at PRIO_D73
#   report     after D7.2 (and D7.3): the R6 report
# DRY_RUN=1 prints the commands of a step instead of running them; the G3 check still runs, since it only reads.
#
# Usage:  experiments/e7-self-authoring/queue-commands-d72-d73.sh followups|judge|d72|d73|report
set -euo pipefail

# ---- placeholders and caps (set before use) -------------------------------------------------------------------------
PRIO_D72="${PRIO_D72:-__SET_PRIO_D72__}"       # queue priority for D7.2 (integer or fractional band)
PRIO_D73="${PRIO_D73:-__SET_PRIO_D73__}"       # queue priority for D7.3
MAX_USD_TEACHER="${MAX_USD_TEACHER:-15}"       # hard cap, Claude teacher (≈125 calls; estimate USD 8–13)
MAX_USD_JUDGE="${MAX_USD_JUDGE:-50}"           # hard cap, Claude judge (≤ 390 items × 3 calls; estimate ≈ USD 46)
DRY_RUN="${DRY_RUN:-0}"

REPO=/home/bhux/workplace/VSA-LLM
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
RUN=experiments/e7-self-authoring/runs/general-v1
GATE="$RUN/gate"
cd "$REPO"
export PYTHONPATH=src

say() { printf '\n== %s\n' "$*"; }
run() {                                         # run a command (or print it under DRY_RUN=1)
    printf '+ %s\n' "$*"
    if [[ "$DRY_RUN" != 1 ]]; then "$@"; fi
}
cpu() { CUDA_VISIBLE_DEVICES= run "$@"; }       # CPU steps never touch the (busy) GPU
need_priority() {
    if [[ "$1" == __SET_* ]]; then echo "set $2 (top of this script or the environment) first" >&2; exit 64; fi
}
jobs_done() {                                   # all named jobs exist in .jobs/ with status done
    "$PY" - "$@" <<'EOF'
import json, sys
from vsa_embed import jobqueue
bad = []
for name in sys.argv[1:]:
    path = jobqueue.DEFAULT_DIR / f"{name}.json"
    status = json.loads(path.read_text())["status"] if path.exists() else "missing"
    if status != "done":
        bad.append(f"{name}: {status}")
if bad:
    print("not done yet:\n  " + "\n  ".join(bad), file=sys.stderr)
    sys.exit(1)
EOF
}
g3() {                                          # G3 check of a stage; exit status 0 PASS, 1 FAIL, 2 INCOMPLETE
    CUDA_VISIBLE_DEVICES= "$PY" -m vsa_embed.experiments.e7_g3 --run "$RUN" --stage "$1"
}
gate_field() { "$PY" -c "import json,sys; v=json.load(open('$GATE/g3-$1.json')).get('$2'); print(' '.join(v) if isinstance(v, list) else v)"; }

D71_JOBS=()
for kind in discover author direct; do
    for host in SmolLM2-360M SmolLM2-135M Qwen2.5-0.5B; do D71_JOBS+=("e7-general-v1-$kind-$host"); done
done
A="$PY -m vsa_embed.experiments.e7_authoring"

case "${1:-}" in
followups)
    say "D7.1 follow-ups (CPU; Claude teacher capped at USD $MAX_USD_TEACHER)"
    jobs_done "${D71_JOBS[@]}"
    cpu $A author --run "$RUN" --author hearst --device cpu
    cpu $A author --run "$RUN" --author random --device cpu
    cpu $A author --run "$RUN" --author teacher --device cpu --max-usd "$MAX_USD_TEACHER"
    cpu $A quality --run "$RUN"
    cpu $A judge-items --run "$RUN"
    say "G3 (stage d72); the judge is informational and may run later (step 'judge')"
    g3 d72 || true
    ;;
judge)
    say "D7.1 Claude judge (capped at USD $MAX_USD_JUDGE)"
    test -f "$RUN/judge/items.jsonl" || { echo "run the 'followups' step first" >&2; exit 1; }
    cpu $A judge --run "$RUN" --runner cli --max-usd "$MAX_USD_JUDGE"
    g3 d72 || true
    ;;
d72)
    need_priority "$PRIO_D72" PRIO_D72
    say "G3 (stage d72)"
    status=0; g3 d72 || status=$?
    if [[ $status -ne 0 ]]; then echo "G3 did not pass (exit $status): D7.2 is not queued" >&2; exit 1; fi
    mode=$(gate_field d72 mode); teacher=$(gate_field d72 teacher_arm); conditions=$(gate_field d72 conditions)
    say "G3 PASS, mode $mode, teacher arm $teacher; conditions: $conditions"
    # Pre-registered D7.2/D7.3 settings (§3.1): LoRA host (decision 30), C5 with HRR (no operator beat C5 in decision 54's
    # screen), the E9 engagement check's LoRA arm (rank 64, gate bias 0; decision 36), and the frozen 50M recipe for
    # E7.3 (gates.md D4.0: 32,768 tokens/step, peak lr 2e-3, cosine to 0.1x; warmup 10M as in scale-v1).
    SETTINGS='{"host_mode": "lora", "channel_condition": "C5", "operator": "hrr",
               "config_overrides": {"model": {"lora_rank": 64}, "channel": {"gate_bias": 0.0}},
               "cross": {"config_overrides": {"train": {"micro_batch": 32, "grad_accum": 1, "lr": 0.002,
                                                         "warmup_tokens": 10000000}}}}'
    printf '+ update round.json with %s\n' "$(echo "$SETTINGS" | tr -s ' \n' ' ')"
    if [[ "$DRY_RUN" != 1 ]]; then
        SETTINGS="$SETTINGS" "$PY" -c "import json, os; from vsa_embed.experiments.e7_round import update_round_settings; \
print(json.dumps(update_round_settings('$RUN', json.loads(os.environ['SETTINGS'])), indent=1))"
    fi
    args=(--run "$RUN" --stage d72 --priority "$PRIO_D72" --conditions $conditions)
    [[ "$teacher" == True ]] || args+=(--no-teacher)
    run "$PY" -m vsa_embed.experiments.e7_plan "${args[@]}" --queue
    ;;
d73)
    need_priority "$PRIO_D73" PRIO_D73
    jobs_done e7-general-v1-verify-SmolLM2-360M-s1 e7-general-v1-round1-self-s1
    say "G3 (stage d73)"
    status=0; g3 d73 || status=$?
    if [[ $status -ne 0 ]]; then echo "G3 (d73) did not pass (exit $status): D7.3 is not queued" >&2; exit 1; fi
    say "cross-prepare (CPU: GPT-2 corpora of 300M tokens, curated and authored)"
    cpu "$PY" -m vsa_embed.experiments.e7_round cross-prepare --run "$RUN" --device cpu
    run "$PY" -m vsa_embed.experiments.e7_plan --run "$RUN" --stage d73 --priority "$PRIO_D73" --queue
    ;;
report)
    cpu "$PY" -m vsa_embed.experiments.e7_report --run "$RUN"
    ;;
*)
    sed -n '2,19p' "$0"; exit 64
    ;;
esac
