# E12 faithfulness of decoding (F1) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Store: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1` (hrr, unbinding `correlation`). Channel route, no recall in context. τ = 0.05 nats.

## New words (property items)

100 terms; 489 of 499 edges with an item are decoded (decode accuracy over every edge: 0.981).

| measure (decoded edges; mean over terms) | value |
|---|---:|
| comprehensiveness net share (τ = 0.05) | +0.362 |
| moved beyond the matched control (share) | 0.610 |
| moved the other way (share) | 0.248 |
| mean gap (nats) | +0.3706 |
| mean Δ removing the edge (nats) | -0.3953 |
| sufficiency net share | +0.268 |
| specificity (share of edges) | 0.233 |

Sensitivity (comprehensiveness net share by τ): τ = 0: +0.363, τ = 0.02: +0.354, τ = 0.05: +0.362, τ = 0.1: +0.347
