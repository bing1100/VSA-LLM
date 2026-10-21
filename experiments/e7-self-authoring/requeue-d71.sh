#!/usr/bin/env bash
# E7 D7.1: re-prioritise the 9 pending D7.1 jobs (author, 2026-10-20: "approve all … run them and queue").
# Prepared 2026-10-20; NOT run by the agent that wrote it. Pre-registration:
# experiments/e7-self-authoring/preregistration-2026-10-20-reschedule.md (§1).
#
# - Only `priority` changes. The 9 commands were smoke-tested on CPU against today's code on 2026-10-20 (discover,
#   author with link, direct; tiny limits, scratch copies of the run folder and match corpora) and are current, so
#   `command` is left as it is.
# - Only jobs whose status is "pending" are touched; any other status is reported and skipped.
# - The queue's own writer is used (`vsa_embed.jobqueue._write`: write to .tmp, then an atomic replace). Each file is
#   read again just before the write and skipped if it changed in between (e.g. a runner started it).
# - The 9 jobs keep their 2 Oct `created` stamps. They therefore run in plan order (discover SmolLM2-360M first: `link`
#   inside the first author job needs it) and ahead of any later job at the same priority.
#
# Usage:  PRIO_D71=54.4965 experiments/e7-self-authoring/requeue-d71.sh        # or edit the placeholder below
#         DRY_RUN=1 PRIO_D71=54.4965 experiments/e7-self-authoring/requeue-d71.sh   # print only
set -euo pipefail

PRIO_D71="${PRIO_D71:-__SET_PRIO_D71__}"   # placeholder: queue priority for D7.1 (integer or fractional band)
DRY_RUN="${DRY_RUN:-0}"
REPO=/home/bhux/workplace/VSA-LLM
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python

if [[ "$PRIO_D71" == __SET_* ]]; then
    echo "set PRIO_D71 (top of this script or the environment) to the queue priority for D7.1" >&2
    exit 64
fi

cd "$REPO"
PRIO_D71="$PRIO_D71" DRY_RUN="$DRY_RUN" PYTHONPATH=src "$PY" - <<'EOF'
import json
import os
import sys

from vsa_embed import jobqueue

priority = jobqueue._priority(os.environ["PRIO_D71"])
dry = os.environ.get("DRY_RUN") == "1"
hosts = ("SmolLM2-360M", "SmolLM2-135M", "Qwen2.5-0.5B")
names = [f"e7-general-v1-{kind}-{host}" for kind in ("discover", "author", "direct") for host in hosts]
queue = jobqueue.DEFAULT_DIR
changed, skipped = [], []
for name in names:
    path = queue / f"{name}.json"
    if not path.exists():
        skipped.append((name, "no job file")); continue
    text = path.read_text()
    job = json.loads(text)
    if job["status"] != "pending":
        skipped.append((name, f"status {job['status']}")); continue
    old = job["priority"]
    job["priority"] = priority
    if dry:
        changed.append((name, old)); continue
    if path.read_text() != text:                     # changed since it was read (a runner picked it up)
        skipped.append((name, "changed while being edited; re-run")); continue
    jobqueue._write(path, job)
    changed.append((name, old))
for name, old in changed:
    print(f"{'would set' if dry else 'set'} {name}: priority {old} -> {priority}")
for name, why in skipped:
    print(f"skipped {name}: {why}")
pending = [j for j in jobqueue.jobs(queue) if j["status"] in ("pending", "blocked")]
ahead = sum(1 for j in pending if j["priority"] < priority)
print(f"{len(changed)} job(s) {'to re-prioritise' if dry else 're-prioritised'}; {ahead} pending/blocked job(s) rank ahead "
      f"of priority {priority}")
sys.exit(0 if len(changed) + sum(why.startswith("status") for _, why in skipped) == len(names) else 1)
EOF
