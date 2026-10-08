**SMOKE TEST — not a result.**

# Ranking benchmark: items.jsonl.gz on e9-wordnet-SmolLM2-135M-full-C5-s1

- run: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C5-s1` (C5, seed 1, channel compose)
- items: `experiments/toolkit-bench/items/alcuna-wordnet-v1/items.jsonl.gz` (40 items; chance 0.350)
- conditions: none, channel-off, store:oracle, store:typeprior, store:random, frame-in-context:oracle, definition-in-context, store:oracle+definition-in-context
- new surfaces: 40 (shadowed existing aliases: 2); frames unresolvable in this run: 0

## all

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.3000 [+0.1500, +0.4506] | 0.2750 | 0.3000 | 0.0 | 1.000 |
| channel-off | +0.3000 [+0.1500, +0.4506] | 0.2750 | 0.3000 | 0.0 | 0.000 |
| store:oracle | +0.3000 [+0.1500, +0.4506] | 0.2750 | 0.3000 | 0.0 | 1.000 |
| store:typeprior | +0.3000 [+0.1500, +0.4506] | 0.2750 | 0.3000 | 0.0 | 1.000 |
| store:random | +0.2750 [+0.1494, +0.4250] | 0.2750 | 0.2750 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.3250 [+0.1750, +0.5000] | 0.2500 | 0.2750 | 25.0 | 1.000 |
| definition-in-context | +0.5500 [+0.3750, +0.7000] | 0.4000 | 0.5000 | 463.4 | 1.000 |
| store:oracle+definition-in-context | +0.5750 [+0.4250, +0.7006] | 0.4000 | 0.5250 | 463.4 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| channel-off − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 40 |
| store:oracle − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 40 |
| store:typeprior − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 40 |
| store:random − none | -0.0250 [-0.0756, +0.0000] | 0.7960 | 40 |
| frame-in-context:oracle − none | +0.0250 [-0.0500, +0.1250] | 0.7960 | 40 |
| definition-in-context − none | +0.2500 [+0.0494, +0.4500] | 0.0299 | 40 |
| store:oracle+definition-in-context − none | +0.2750 [+0.0744, +0.4750] | 0.0299 | 40 |

## alcuna-bool

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.4375 [+0.1875, +0.6266] | 0.3125 | 0.4375 | 0.0 | 1.000 |
| channel-off | +0.4375 [+0.1875, +0.6266] | 0.3125 | 0.4375 | 0.0 | 0.000 |
| store:oracle | +0.4375 [+0.1875, +0.6266] | 0.3125 | 0.4375 | 0.0 | 1.000 |
| store:typeprior | +0.4375 [+0.1875, +0.6266] | 0.3125 | 0.4375 | 0.0 | 1.000 |
| store:random | +0.4375 [+0.1875, +0.6266] | 0.3125 | 0.4375 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.3750 [+0.1875, +0.5625] | 0.3125 | 0.3750 | 25.7 | 1.000 |
| definition-in-context | +0.5000 [+0.2500, +0.7500] | 0.3750 | 0.5000 | 403.6 | 1.000 |
| store:oracle+definition-in-context | +0.5625 [+0.3125, +0.8125] | 0.3750 | 0.5625 | 403.6 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| channel-off − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 16 |
| store:oracle − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 16 |
| store:typeprior − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 16 |
| store:random − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 16 |
| frame-in-context:oracle − none | -0.0625 [-0.1875, +0.0000] | 0.7562 | 16 |
| definition-in-context − none | +0.0625 [-0.3750, +0.5000] | 0.8856 | 16 |
| store:oracle+definition-in-context − none | +0.1250 [-0.3141, +0.5625] | 0.6866 | 16 |

## alcuna-mc

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.2083 [+0.0833, +0.3750] | 0.2500 | 0.2083 | 0.0 | 1.000 |
| channel-off | +0.2083 [+0.0833, +0.3750] | 0.2500 | 0.2083 | 0.0 | 0.000 |
| store:oracle | +0.2083 [+0.0833, +0.3750] | 0.2500 | 0.2083 | 0.0 | 1.000 |
| store:typeprior | +0.2083 [+0.0833, +0.3750] | 0.2500 | 0.2083 | 0.0 | 1.000 |
| store:random | +0.1667 [+0.0417, +0.2927] | 0.2500 | 0.1667 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.2917 [+0.1250, +0.4583] | 0.2083 | 0.2083 | 24.5 | 1.000 |
| definition-in-context | +0.5833 [+0.3333, +0.7510] | 0.4167 | 0.5000 | 503.3 | 1.000 |
| store:oracle+definition-in-context | +0.5833 [+0.3333, +0.7510] | 0.4167 | 0.5000 | 503.3 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| channel-off − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 24 |
| store:oracle − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 24 |
| store:typeprior − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 24 |
| store:random − none | -0.0417 [-0.1250, +0.0000] | 0.7164 | 24 |
| frame-in-context:oracle − none | +0.0833 [+0.0000, +0.2083] | 0.3085 | 24 |
| definition-in-context − none | +0.3750 [+0.2073, +0.5417] | 0.0100 | 24 |
| store:oracle+definition-in-context − none | +0.3750 [+0.2073, +0.5417] | 0.0100 | 24 |

## Readers (frames against the oracle frame)

| Reader | Terms | Frames | Precision | Recall | Cost per term |
|---|---|---|---|---|---|
| oracle | 40 | 40 | — | — | — |
| typeprior | 40 | 40 | 0.001 | 0.017 | — |
| random | 40 | 40 | — | — | — |

Scorer: 1024 unique of 1024 requests, 150258 forward tokens, 187.2 s, truncated 8.
