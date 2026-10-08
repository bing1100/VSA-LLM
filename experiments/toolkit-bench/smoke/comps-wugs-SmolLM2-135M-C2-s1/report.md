**SMOKE TEST — not a result.**

# Ranking benchmark: items.jsonl.gz on e9-wordnet-SmolLM2-135M-full-C2-s1

- run: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C2-s1` (C2, seed 1, channel free)
- items: `experiments/toolkit-bench/items/comps-wugs-wordnet-v1/items.jsonl.gz` (40 items; chance 0.500)
- conditions: none, channel-off, store:oracle, store:random, frame-in-context:oracle, definition-in-context
- new surfaces: 4 (shadowed existing aliases: 0); frames unresolvable in this run: 0

## all

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.5000 [+0.5000, +0.5000] | 0.5000 | 0.5000 | 0.0 | 1.000 |
| channel-off | +0.5000 [+0.5000, +0.5000] | 0.5000 | 0.5000 | 0.0 | 0.000 |
| store:oracle | +0.5000 [+0.5000, +0.5000] | 0.5000 | 0.5000 | 0.0 | 1.000 |
| store:random | +0.5000 [+0.5000, +0.5000] | 0.5000 | 0.5000 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.7000 [+0.5500, +0.8256] | 0.7000 | 0.7000 | 17.9 | 1.000 |
| definition-in-context | +0.6750 [+0.5250, +0.8250] | 0.6750 | 0.6750 | 8.6 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| channel-off − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 40 |
| store:oracle − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 40 |
| store:random − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 40 |
| frame-in-context:oracle − none | +0.2000 [+0.0250, +0.3250] | 0.0199 | 40 |
| definition-in-context − none | +0.1750 [+0.0500, +0.3250] | 0.0199 | 40 |

## Readers (frames against the oracle frame)

| Reader | Terms | Frames | Precision | Recall | Cost per term |
|---|---|---|---|---|---|
| oracle | 77 | 77 | — | — | — |
| random | 77 | 77 | — | — | — |

Scorer: 400 unique of 480 requests, 5782 forward tokens, 5.2 s, truncated 0.
