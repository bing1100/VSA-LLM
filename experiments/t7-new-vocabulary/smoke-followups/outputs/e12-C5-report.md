# E12 self-query, phase A (understanding) — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/lora) — SMOKE

Items `experiments/e9-retrofit/items/understanding-t7rood-smollm2-v1`; 17 of 18 concepts link. Conditions: `none`, `recall:own`, `symbolic`. Stores: `own` = /tmp/claude-1000/-home-bhux-workplace/71272da5-02b7-47b1-a057-5887c33fb83d/scratchpad/smoke-followups/e9/runs/t7rood/SmolLM2-135M-lora-C5-s1 (hrr, correlation).

| condition | family / subset | items | accuracy | − chance | hop 1 | bridge | hop 2 | pair (reverse) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | affordance/all | 8 | 0.688 | +0.354 | — | — | — | — |
| none | affordance/heldout | 2 | 0.750 | +0.417 | — | — | — | — |
| none | affordance/new | 2 | 0.000 | -0.333 | — | — | — | — |
| none | affordance/rare | 2 | 1.000 | +0.667 | — | — | — | — |
| none | affordance/seen | 2 | 1.000 | +0.667 | — | — | — | — |
| none | negation/all | 20 | 0.475 | -0.025 | — | — | — | — |
| none | negation/heldout | 4 | 0.375 | -0.125 | — | — | — | — |
| none | negation/new | 8 | 0.500 | +0.000 | — | — | — | — |
| none | negation/rare | 4 | 0.625 | +0.125 | — | — | — | — |
| none | negation/seen | 4 | 0.375 | -0.125 | — | — | — | — |
| none | paraphrase/all | 10 | 0.650 | +0.343 | — | — | — | — |
| none | paraphrase/heldout | 2 | 0.750 | +0.417 | — | — | — | — |
| none | paraphrase/new | 2 | 0.500 | +0.167 | — | — | — | — |
| none | paraphrase/rare | 4 | 0.500 | +0.233 | — | — | — | — |
| none | paraphrase/seen | 2 | 1.000 | +0.667 | — | — | — | — |
| none | reverse/all | 10 | 0.350 | -0.150 | — | — | — | — |
| none | reverse/heldout | 2 | 0.250 | -0.250 | — | — | — | — |
| none | reverse/new | 4 | 0.375 | -0.125 | — | — | — | — |
| none | reverse/rare | 2 | 0.250 | -0.250 | — | — | — | — |
| none | reverse/seen | 2 | 0.500 | +0.000 | — | — | — | — |
| recall:own | affordance/all | 8 | 0.812 | +0.479 | — | — | — | — |
| recall:own | affordance/heldout | 2 | 0.500 | +0.167 | — | — | — | — |
| recall:own | affordance/new | 2 | 1.000 | +0.667 | — | — | — | — |
| recall:own | affordance/rare | 2 | 1.000 | +0.667 | — | — | — | — |
| recall:own | affordance/seen | 2 | 0.750 | +0.417 | — | — | — | — |
| recall:own | negation/all | 20 | 0.500 | +0.000 | — | — | — | — |
| recall:own | negation/heldout | 4 | 0.500 | +0.000 | — | — | — | — |
| recall:own | negation/new | 8 | 0.500 | +0.000 | — | — | — | — |
| recall:own | negation/rare | 4 | 0.500 | +0.000 | — | — | — | — |
| recall:own | negation/seen | 4 | 0.500 | +0.000 | — | — | — | — |
| recall:own | paraphrase/all | 10 | 0.900 | +0.593 | — | — | — | — |
| recall:own | paraphrase/heldout | 2 | 1.000 | +0.667 | — | — | — | — |
| recall:own | paraphrase/new | 2 | 1.000 | +0.667 | — | — | — | — |
| recall:own | paraphrase/rare | 4 | 0.750 | +0.483 | — | — | — | — |
| recall:own | paraphrase/seen | 2 | 1.000 | +0.667 | — | — | — | — |
| recall:own | reverse/all | 10 | 0.700 | +0.200 | — | — | — | 1.000 |
| recall:own | reverse/heldout | 2 | 0.250 | -0.250 | — | — | — | 1.000 |
| recall:own | reverse/new | 4 | 0.875 | +0.375 | — | — | — | 1.000 |
| recall:own | reverse/rare | 2 | 1.000 | +0.500 | — | — | — | 1.000 |
| recall:own | reverse/seen | 2 | 0.500 | +0.000 | — | — | — | 1.000 |
| symbolic | affordance/all | 8 | 0.875 | +0.542 | — | — | — | — |
| symbolic | affordance/heldout | 2 | 0.750 | +0.417 | — | — | — | — |
| symbolic | affordance/new | 2 | 1.000 | +0.667 | — | — | — | — |
| symbolic | affordance/rare | 2 | 1.000 | +0.667 | — | — | — | — |
| symbolic | affordance/seen | 2 | 0.750 | +0.417 | — | — | — | — |
| symbolic | negation/all | 20 | 0.500 | +0.000 | — | — | — | — |
| symbolic | negation/heldout | 4 | 0.500 | +0.000 | — | — | — | — |
| symbolic | negation/new | 8 | 0.500 | +0.000 | — | — | — | — |
| symbolic | negation/rare | 4 | 0.500 | +0.000 | — | — | — | — |
| symbolic | negation/seen | 4 | 0.500 | +0.000 | — | — | — | — |
| symbolic | paraphrase/all | 10 | 0.900 | +0.593 | — | — | — | — |
| symbolic | paraphrase/heldout | 2 | 1.000 | +0.667 | — | — | — | — |
| symbolic | paraphrase/new | 2 | 1.000 | +0.667 | — | — | — | — |
| symbolic | paraphrase/rare | 4 | 0.750 | +0.483 | — | — | — | — |
| symbolic | paraphrase/seen | 2 | 1.000 | +0.667 | — | — | — | — |
| symbolic | reverse/all | 10 | 0.700 | +0.200 | — | — | — | 1.000 |
| symbolic | reverse/heldout | 2 | 0.500 | +0.000 | — | — | — | 1.000 |
| symbolic | reverse/new | 4 | 0.875 | +0.375 | — | — | — | 1.000 |
| symbolic | reverse/rare | 2 | 0.750 | +0.250 | — | — | — | 1.000 |
| symbolic | reverse/seen | 2 | 0.500 | +0.000 | — | — | — | 1.000 |

`contrast`: twin / role contrast accuracy (chance 0.5); `item`: PMI-argmax accuracy; `decode`: share of the gold fillers the condition's context states (twins: the four critical slots of a pair).
