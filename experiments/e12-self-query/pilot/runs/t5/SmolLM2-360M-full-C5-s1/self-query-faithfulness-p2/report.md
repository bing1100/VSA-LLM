# E12 faithfulness of decoding (F1) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Store: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1` (hrr, unbinding `correlation`). Channel route, no recall in context. τ = 0.05 nats.

## New words (property items)

100 terms; 500 of 511 edges with an item are decoded (decode accuracy over every edge: 0.975).

| measure (decoded edges; mean over terms) | value |
|---|---:|
| comprehensiveness net share (τ = 0.05) | +0.356 |
| moved beyond the matched control (share) | 0.600 |
| moved the other way (share) | 0.244 |
| mean gap (nats) | +0.3908 |
| mean Δ removing the edge (nats) | -0.4091 |
| sufficiency net share | +0.299 |
| specificity (share of edges) | 0.230 |

Sensitivity (comprehensiveness net share by τ): τ = 0: +0.383, τ = 0.02: +0.384, τ = 0.05: +0.356, τ = 0.1: +0.318
