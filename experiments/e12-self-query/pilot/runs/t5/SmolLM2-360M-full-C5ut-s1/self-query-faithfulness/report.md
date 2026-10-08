# E12 faithfulness of decoding (F1) — C5ut seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Store: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5ut-s1` (additive, unbinding `bundle`). Channel route, no recall in context. τ = 0.05 nats.

## New words (property items)

100 terms; 496 of 519 edges with an item are decoded (decode accuracy over every edge: 0.962).

| measure (decoded edges; mean over terms) | value |
|---|---:|
| comprehensiveness net share (τ = 0.05) | +0.322 |
| moved beyond the matched control (share) | 0.603 |
| moved the other way (share) | 0.280 |
| mean gap (nats) | +0.5228 |
| mean Δ removing the edge (nats) | -0.5510 |
| sufficiency net share | +0.248 |
| specificity (share of edges) | 0.247 |

Sensitivity (comprehensiveness net share by τ): τ = 0: +0.355, τ = 0.02: +0.326, τ = 0.05: +0.322, τ = 0.1: +0.323

## Twins (role specificity)

300 pairs: role-specificity net share +0.008, mean RS +0.0012 nats (> 0: the removed filler drops more under its own role); twin contrast own 0.502, swapped frames 0.498.
