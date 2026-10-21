#!/usr/bin/env bash
# E14 addendum 10 (`preregistration-addendum-10-retrieve.md`): RD-13, retrieve-then-read on T4 held-out concepts — Wikipedia
# text read by E11's CPU readers into the store (selective recall, the lm router) against the same text in the prompt (RAG)
# and the ChEBI definition. Queued 2026-10-20 by workplace at the author's go-ahead ("Retrieval tool over a Wikipedia dump
# could be cool"). After the §19 tier-F jobs (54.4959991–54.4959994); before the toolkit band (54.496).
# Inputs (CPU, built before queueing): experiments/e14-logic/items/t4-heldout-wiki/ (`e14_retrieve build`; the retrieved
# text itself is git-ignored and rebuilt from the cache ~/data/vsa-llm/wikipedia/t4-heldout-v1/).
set -euo pipefail
PY=${PY:-/home/bhux/anaconda3/envs/vsa-repro/bin/python}
PRIO_GPU=${PRIO_GPU:-54.4959997}; PRIO_REPORT=${PRIO_REPORT:-54.49599986}
WIKI=experiments/e14-logic/items/t4-heldout-wiki
ITEMS=experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1
ALIAS=/home/bhux/data/vsa-llm/e9/alias-tables/t4.json
for s in 1 2 3; do
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd13-t4-heldout-SmolLM2-360M-C5-s$s --priority $PRIO_GPU --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_reading recall --run experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s$s --items $ITEMS --alias-table $ALIAS --family rd13 --null plain --retrieved $WIKI --output experiments/e14-logic/runs/rd13/t4-heldout/SmolLM2-360M-C5-s$s   # ~0.1-0.2 GPU-h
done
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd13-t4-heldout-Qwen3-1.7B-Base-C0p-s1 --priority $PRIO_GPU --min-free-gb 10 --no-resume -- $PY -m vsa_embed.experiments.e14_reading recall --run experiments/e9-retrofit/runs/t4-qwen3/Qwen3-1.7B-Base-lora-C0p-s1 --frames experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1/e11-t4-heldout/frames.jsonl --items $ITEMS --alias-table $ALIAS --family rd13 --null plain --retrieved $WIKI --batch-size 16 --output experiments/e14-logic/runs/rd13/t4-heldout/Qwen3-1.7B-Base-C0p-s1   # ~0.3-0.5 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e14-rd13-report --priority $PRIO_REPORT --no-resume -- $PY -m vsa_embed.experiments.e14_reading recall-report --family rd13 --retrieved $WIKI --runs experiments/e14-logic/runs/rd13/t4-heldout/SmolLM2-360M-C5-s1 experiments/e14-logic/runs/rd13/t4-heldout/SmolLM2-360M-C5-s2 experiments/e14-logic/runs/rd13/t4-heldout/SmolLM2-360M-C5-s3 experiments/e14-logic/runs/rd13/t4-heldout/Qwen3-1.7B-Base-C0p-s1 --output experiments/e14-logic/report/rd13
