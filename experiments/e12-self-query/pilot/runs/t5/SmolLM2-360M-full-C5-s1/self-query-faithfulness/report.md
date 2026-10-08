# E12 faithfulness of decoding (F1) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Store: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1` (hrr, unbinding `correlation`). Channel route, no recall in context. τ = 0.05 nats.

## New words (property items)

100 terms; 508 of 519 edges with an item are decoded (decode accuracy over every edge: 0.976).

| measure (decoded edges; mean over terms) | value |
|---|---:|
| comprehensiveness net share (τ = 0.05) | +0.308 |
| moved beyond the matched control (share) | 0.587 |
| moved the other way (share) | 0.280 |
| mean gap (nats) | +0.3641 |
| mean Δ removing the edge (nats) | -0.3752 |
| sufficiency net share | +0.232 |
| specificity (share of edges) | 0.201 |

Sensitivity (comprehensiveness net share by τ): τ = 0: +0.345, τ = 0.02: +0.324, τ = 0.05: +0.308, τ = 0.1: +0.300

## Twins (role specificity)

300 pairs: role-specificity net share -0.036, mean RS -0.0214 nats (> 0: the removed filler drops more under its own role); twin contrast own 0.504, swapped frames 0.501.
