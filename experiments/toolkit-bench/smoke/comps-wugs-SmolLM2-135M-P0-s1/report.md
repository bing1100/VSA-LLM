**SMOKE TEST — not a result.**

# Ranking benchmark: items.jsonl.gz on e9-wordnet-SmolLM2-135M-frozen-P0-s1

- run: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-frozen-P0-s1` (P0, seed 1, channel none)
- items: `experiments/toolkit-bench/items/comps-wugs-wordnet-v1/items.jsonl.gz` (40 items; chance 0.500)
- conditions: none, frame-in-context:oracle, definition-in-context
- new surfaces: 0 (shadowed existing aliases: 0); frames unresolvable in this run: 0

## all

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.5000 [+0.5000, +0.5000] | 0.5000 | 0.5000 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.7250 [+0.5750, +0.8506] | 0.7250 | 0.7250 | 17.9 | 1.000 |
| definition-in-context | +0.7250 [+0.6000, +0.8500] | 0.7250 | 0.7250 | 8.6 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| frame-in-context:oracle − none | +0.2250 [+0.0744, +0.3500] | 0.0100 | 40 |
| definition-in-context − none | +0.2250 [+0.0994, +0.3500] | 0.0199 | 40 |

## Readers (frames against the oracle frame)

| Reader | Terms | Frames | Precision | Recall | Cost per term |
|---|---|---|---|---|---|
| oracle | 77 | 77 | — | — | — |

Scorer: 200 unique of 240 requests, 3872 forward tokens, 10.5 s, truncated 0.
