#!/usr/bin/env bash
# E14 addendum 9 (experiments/e14-logic/preregistration-addendum-9-treetext.md): decision trees read from prose by the frozen
# E14 hosts, one GPU job per host and one CPU-lane report. Prepared 2026-10-20 at the author's approval ("Do the tree test and
# see whether it is worth queuing more experiments for"). Estimated 0.15–0.5 GPU-h in total (see the addendum's cost section).
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_GPU=${PRIO_GPU:-54.4959997}; PRIO_REPORT=${PRIO_REPORT:-54.49599985}
ITEMS=experiments/e14-logic/items/trees-text-v1
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-treetext-SmolLM2-360M --priority $PRIO_GPU --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_treetext read --host H-360 --items $ITEMS --batch-size 8 --output experiments/e14-logic/runs/treetext/SmolLM2-360M
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-treetext-Qwen3-0.6B --priority $PRIO_GPU --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_treetext read --host H-q06 --items $ITEMS --batch-size 8 --output experiments/e14-logic/runs/treetext/Qwen3-0.6B
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-treetext-Qwen3-1.7B --priority $PRIO_GPU --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_treetext read --host H-q17 --items $ITEMS --batch-size 8 --output experiments/e14-logic/runs/treetext/Qwen3-1.7B
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-treetext-report --priority $PRIO_REPORT --no-resume -- $PY -m vsa_embed.experiments.e14_treetext report --items $ITEMS --runs experiments/e14-logic/runs/treetext/SmolLM2-360M experiments/e14-logic/runs/treetext/Qwen3-0.6B experiments/e14-logic/runs/treetext/Qwen3-1.7B --output experiments/e14-logic/report/treetext
