# E12 faithfulness of decoding (F1) — C5ut seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Store: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5ut-s1` (additive, unbinding `bundle`). Channel route, no recall in context. τ = 0.05 nats.

## New words (property items)

100 terms; 483 of 499 edges with an item are decoded (decode accuracy over every edge: 0.966).

| measure (decoded edges; mean over terms) | value |
|---|---:|
| comprehensiveness net share (τ = 0.05) | +0.445 |
| moved beyond the matched control (share) | 0.668 |
| moved the other way (share) | 0.223 |
| mean gap (nats) | +0.5381 |
| mean Δ removing the edge (nats) | -0.5453 |
| sufficiency net share | +0.300 |
| specificity (share of edges) | 0.230 |

Sensitivity (comprehensiveness net share by τ): τ = 0: +0.435, τ = 0.02: +0.452, τ = 0.05: +0.445, τ = 0.1: +0.423
