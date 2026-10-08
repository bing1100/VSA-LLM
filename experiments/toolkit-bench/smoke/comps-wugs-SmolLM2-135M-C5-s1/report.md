**SMOKE TEST — not a result.**

# Ranking benchmark: items.jsonl.gz on e9-wordnet-SmolLM2-135M-full-C5-s1

- run: `/home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C5-s1` (C5, seed 1, channel compose)
- items: `experiments/toolkit-bench/items/comps-wugs-wordnet-v1/items.jsonl.gz` (40 items; chance 0.500)
- conditions: none, channel-off, store:oracle, store:linker, store:typeprior, store:random, store:oracle:parent, frame-in-context:oracle, definition-in-context, store:oracle+definition-in-context
- new surfaces: 4 (shadowed existing aliases: 0); frames unresolvable in this run: 0

## all

| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |
|---|---|---|---|---|---|
| none | +0.5000 [+0.5000, +0.5000] | 0.5000 | 0.5000 | 0.0 | 1.000 |
| channel-off | +0.5000 [+0.5000, +0.5000] | 0.5000 | 0.5000 | 0.0 | 0.000 |
| store:oracle | +0.5250 [+0.3994, +0.6750] | 0.5250 | 0.5250 | 0.0 | 1.000 |
| store:linker | +0.4125 [+0.2750, +0.5378] | 0.4125 | 0.4125 | 0.0 | 1.000 |
| store:typeprior | +0.5375 [+0.4250, +0.6878] | 0.5375 | 0.5375 | 0.0 | 1.000 |
| store:random | +0.5500 [+0.4000, +0.6750] | 0.5500 | 0.5500 | 0.0 | 1.000 |
| store:oracle:parent | +0.5000 [+0.3500, +0.6500] | 0.5000 | 0.5000 | 0.0 | 1.000 |
| frame-in-context:oracle | +0.7000 [+0.5500, +0.8256] | 0.7000 | 0.7000 | 17.9 | 1.000 |
| definition-in-context | +0.6750 [+0.5250, +0.8250] | 0.6750 | 0.6750 | 8.6 | 1.000 |
| store:oracle+definition-in-context | +0.6750 [+0.5250, +0.8250] | 0.6750 | 0.6750 | 8.6 | 1.000 |

| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |
|---|---|---|---|
| channel-off − none | +0.0000 [+0.0000, +0.0000] | 1.0000 | 40 |
| store:oracle − none | +0.0250 [-0.1256, +0.1756] | 0.8259 | 40 |
| store:linker − none | -0.0875 [-0.2128, +0.0378] | 0.2488 | 40 |
| store:typeprior − none | +0.0375 [-0.0875, +0.1625] | 0.7363 | 40 |
| store:random − none | +0.0500 [-0.1000, +0.1500] | 0.5672 | 40 |
| store:oracle:parent − none | +0.0000 [-0.1256, +0.1500] | 1.0000 | 40 |
| frame-in-context:oracle − none | +0.2000 [+0.0250, +0.3250] | 0.0199 | 40 |
| definition-in-context − none | +0.1750 [+0.0500, +0.3250] | 0.0199 | 40 |
| store:oracle+definition-in-context − none | +0.1750 [+0.0500, +0.3250] | 0.0199 | 40 |

## Readers (frames against the oracle frame)

| Reader | Terms | Frames | Precision | Recall | Cost per term |
|---|---|---|---|---|---|
| oracle | 77 | 77 | — | — | — |
| linker | 77 | 41 | 0.049 | 0.009 | forward_passes 9.9, forward_tokens 72.9, seconds 0.1 |
| typeprior | 77 | 41 | 0.854 | 0.152 | — |
| random | 77 | 77 | — | — | — |
| oracle:parent | 77 | 77 | — | — | — |

Scorer: 1465 unique of 1563 requests, 14903 forward tokens, 16.1 s, truncated 0.
