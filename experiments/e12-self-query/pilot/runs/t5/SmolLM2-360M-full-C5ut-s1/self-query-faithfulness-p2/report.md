# E12 faithfulness of decoding (F1) — C5ut seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Store: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5ut-s1` (additive, unbinding `bundle`). Channel route, no recall in context. τ = 0.05 nats.

## New words (property items)

100 terms; 491 of 511 edges with an item are decoded (decode accuracy over every edge: 0.959).

| measure (decoded edges; mean over terms) | value |
|---|---:|
| comprehensiveness net share (τ = 0.05) | +0.410 |
| moved beyond the matched control (share) | 0.648 |
| moved the other way (share) | 0.238 |
| mean gap (nats) | +0.5528 |
| mean Δ removing the edge (nats) | -0.5587 |
| sufficiency net share | +0.348 |
| specificity (share of edges) | 0.240 |

Sensitivity (comprehensiveness net share by τ): τ = 0: +0.424, τ = 0.02: +0.428, τ = 0.05: +0.410, τ = 0.1: +0.402
