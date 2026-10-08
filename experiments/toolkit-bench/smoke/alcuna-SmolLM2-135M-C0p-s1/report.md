**SMOKE TEST — not a result.**

# Ranking benchmark: items.jsonl.gz on e9-wordnet-SmolLM2-135M-full-C0p-s1

- run: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C0p-s1` (C0p, seed 1, channel none)
- items: `experiments/toolkit-bench/items/alcuna-wordnet-v1/items.jsonl.gz` (40 items; chance 0.350)
- conditions: none, frame-in-context:oracle, definition-in-context
- new surfaces: 0 (shadowed existing aliases: 0); frames unresolvable in this run: 0

## all

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.3000 [+0.1500, +0.4506] | 0.2750 | 0.3000 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.3250 [+0.1750, +0.5000] | 0.2500 | 0.3000 | 25.0 | 1.000 |
| definition-in-context | +0.5500 [+0.3750, +0.7000] | 0.4000 | 0.5000 | 463.4 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| frame-in-context:oracle − none | +0.0250 [-0.0500, +0.1250] | 0.7960 | 40 |
| definition-in-context − none | +0.2500 [+0.0494, +0.4500] | 0.0299 | 40 |

## alcuna-bool

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.4375 [+0.1875, +0.6266] | 0.3125 | 0.4375 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.3750 [+0.1875, +0.5625] | 0.3125 | 0.3750 | 25.7 | 1.000 |
| definition-in-context | +0.5000 [+0.2500, +0.7500] | 0.3750 | 0.5000 | 403.6 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| frame-in-context:oracle − none | -0.0625 [-0.1875, +0.0000] | 0.7562 | 16 |
| definition-in-context − none | +0.0625 [-0.3750, +0.5000] | 0.8856 | 16 |

## alcuna-mc

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.2083 [+0.0833, +0.3750] | 0.2500 | 0.2083 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.2917 [+0.1250, +0.4583] | 0.2083 | 0.2500 | 24.5 | 1.000 |
| definition-in-context | +0.5833 [+0.3333, +0.7510] | 0.4167 | 0.5000 | 503.3 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| frame-in-context:oracle − none | +0.0833 [+0.0000, +0.2083] | 0.3085 | 24 |
| definition-in-context − none | +0.3750 [+0.2073, +0.5417] | 0.0100 | 24 |

## Readers (frames against the oracle frame)

| Reader | Terms | Frames | Precision | Recall | Cost per term |
|---|---|---|---|---|---|
| oracle | 40 | 40 | — | — | — |

Scorer: 384 unique of 384 requests, 73572 forward tokens, 129.1 s, truncated 4.
