#!/usr/bin/env bash
# T2 developer tools in the E9 retrofit (author, 2026-10-20: "approve all … run them and queue"). NOT EXECUTED by the agent
# that wrote it: nothing was added to .jobs. Run from the repository root of the main checkout:
#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python PRIO_T2=<slot> \
#     bash experiments/t2-developer-tools/queue-commands-e9.sh
# Pre-registration: experiments/e9-retrofit/preregistration-addendum-2026-10-20-t2.md (commit it before running this script).
#
# Priorities: set PRIO_T2 below (or in the environment) to the free slot the coordinator picks. e9_plan --level-step 0.0001
# then places training at PRIO_T2, per-run evaluations and the claim-B rescoring at +0.0001, e4_quant at +0.0002 and the R9
# report at +0.0003. The report job is named `t2-report-s1-2-3`; its name contains "-report", so the CPU lane runs it.
#
# Jobs and GPU-h (addendum §7, from measured T4 SmolLM2-360M durations in .jobs/*.json, scaled to T2's 3,072 windows and item
# counts): 83 jobs ≈ 23 GPU-h (training 9 × ≈ 73 min ≈ 10.9; evaluations, rescoring and e4_quant ≈ 12.3; the report on the CPU
# lane). `--dry-run` instead of `--queue` lists every job and prints e9_plan's flat estimate (≈ 18.8 GPU-h).
set -euo pipefail
: "${PY:?set PY to the pinned interpreter (/home/bhux/anaconda3/envs/vsa-repro/bin/python)}"
PRIO_T2="${PRIO_T2:?set PRIO_T2 to the queue slot of the T2 block (a free fractional slot)}"

# 0. Prerequisites (CPU, ≈ 1 min, ≤ 2.3 GB RSS; idempotent). The evaluation alias table ~/data/vsa-llm/e9/alias-tables/t2.json
#    is already written (2026-10-20); e9_plan --queue rewrites nothing if it exists. The dimension-3 items are built here with
#    the standard builder and the addendum's seeds, then checked against the hashes pinned in addendum §3.
ITEMS=experiments/e9-retrofit/items
for kind in new edits; do
  folder=$ITEMS/$([ "$kind" = new ] && echo new-words || echo edits)-t2-smollm2-v1
  if [ ! -f "$folder/manifest.json" ]; then
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src "$PY" -m vsa_embed.experiments.e9_tracks items --track t2 --kind "$kind" --output "$folder"
  fi
done
sha256sum --check --strict - <<'EOF'
51ad2bc69f78023ef172a069ec32d2c11fcd7e967c66513051075000ff98c0ad  experiments/e9-retrofit/items/new-words-t2-smollm2-v1/items.jsonl
2bec1f0111e4ef49cd3d5a7a19aa34134f291d27191bcd98691d28890dd953d2  experiments/e9-retrofit/items/new-words-t2-smollm2-v1/concepts.jsonl
0f96d856950027ff0b1f83502f49dd5c6e6627245811b2f85a46763ffa8e1f23  experiments/e9-retrofit/items/edits-t2-smollm2-v1/items.jsonl
51dc4c1ebd7200a1227ab8823d3f6b06d8484dc959b64f09170e8caf7558ee15  experiments/e9-retrofit/items/edits-t2-smollm2-v1/concepts.jsonl
EOF

# 1. T2 on SmolLM2-360M, the T4 core set P0 / C0′ / C2 / C5 × seeds 1–3, the standard evaluation chain (probes, WP-C7
#    zero-shot, dimension 3, at bf16 and INT4), the claim-B rescoring of every model, e4_quant (eval, eval-general), R9 report.
#    Stage t2; configs experiments/e9-retrofit/configs/t2/ (written by the dry run of 2026-10-20; e9_plan rewrites them
#    identically).
PYTHONPATH=src "$PY" -m vsa_embed.experiments.e9_plan --track t2 --hosts SmolLM2-360M --models P0 C0p C2 C5 --seeds 1 2 3 \
  --rescore all --priority "$PRIO_T2" --level-step 0.0001 --queue
