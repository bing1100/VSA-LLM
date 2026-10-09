# E12 — self-query — SMOKE

Pre-registration: `experiments/e12-self-query/preregistration.md`. Contrasts: units × seeds crossed model (Satterthwaite t, 95% CI, two-sided p; one seed: one-sample t), Holm within Q1 and within F1.

**SMOKE: these numbers do not count toward the pre-registered endpoints.**

## SmolLM2-135M

Runs: {'C0p': [1], 'C5': [1]}

### WP-UB items (affordance, negation, paraphrase, reverse)

| host model | condition | family / subset | accuracy | − chance | hop 1 | bridge | hop 2 | pair | seeds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0p | none | affordance/all | 0.750 | +0.417 | — | — | — | — | 1 |
| C0p | none | affordance/heldout | 0.750 | +0.417 | — | — | — | — | 1 |
| C0p | none | affordance/new | 0.250 | -0.083 | — | — | — | — | 1 |
| C0p | none | affordance/rare | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | none | affordance/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | none | negation/all | 0.475 | -0.025 | — | — | — | — | 1 |
| C0p | none | negation/heldout | 0.375 | -0.125 | — | — | — | — | 1 |
| C0p | none | negation/new | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | none | negation/rare | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | none | negation/seen | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | none | paraphrase/all | 0.650 | +0.343 | — | — | — | — | 1 |
| C0p | none | paraphrase/heldout | 0.750 | +0.417 | — | — | — | — | 1 |
| C0p | none | paraphrase/new | 0.500 | +0.167 | — | — | — | — | 1 |
| C0p | none | paraphrase/rare | 0.500 | +0.233 | — | — | — | — | 1 |
| C0p | none | paraphrase/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | none | reverse/all | 0.350 | -0.150 | — | — | — | — | 1 |
| C0p | none | reverse/heldout | 0.250 | -0.250 | — | — | — | — | 1 |
| C0p | none | reverse/new | 0.375 | -0.125 | — | — | — | — | 1 |
| C0p | none | reverse/rare | 0.250 | -0.250 | — | — | — | — | 1 |
| C0p | none | reverse/seen | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | recall:C5 | affordance/all | 0.812 | +0.479 | — | — | — | — | 1 |
| C0p | recall:C5 | affordance/heldout | 0.500 | +0.167 | — | — | — | — | 1 |
| C0p | recall:C5 | affordance/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | recall:C5 | affordance/rare | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | recall:C5 | affordance/seen | 0.750 | +0.417 | — | — | — | — | 1 |
| C0p | recall:C5 | negation/all | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | recall:C5 | negation/heldout | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | recall:C5 | negation/new | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | recall:C5 | negation/rare | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | recall:C5 | negation/seen | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | recall:C5 | paraphrase/all | 0.900 | +0.593 | — | — | — | — | 1 |
| C0p | recall:C5 | paraphrase/heldout | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | recall:C5 | paraphrase/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | recall:C5 | paraphrase/rare | 0.750 | +0.483 | — | — | — | — | 1 |
| C0p | recall:C5 | paraphrase/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | recall:C5 | reverse/all | 0.650 | +0.150 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/heldout | 0.250 | -0.250 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/new | 0.875 | +0.375 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/rare | 0.750 | +0.250 | — | — | — | 1.000 | 1 |
| C0p | recall:C5 | reverse/seen | 0.500 | +0.000 | — | — | — | 1.000 | 1 |
| C0p | symbolic | affordance/all | 0.875 | +0.542 | — | — | — | — | 1 |
| C0p | symbolic | affordance/heldout | 0.750 | +0.417 | — | — | — | — | 1 |
| C0p | symbolic | affordance/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | symbolic | affordance/rare | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | symbolic | affordance/seen | 0.750 | +0.417 | — | — | — | — | 1 |
| C0p | symbolic | negation/all | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | symbolic | negation/heldout | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | symbolic | negation/new | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | symbolic | negation/rare | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | symbolic | negation/seen | 0.500 | +0.000 | — | — | — | — | 1 |
| C0p | symbolic | paraphrase/all | 0.900 | +0.593 | — | — | — | — | 1 |
| C0p | symbolic | paraphrase/heldout | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | symbolic | paraphrase/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | symbolic | paraphrase/rare | 0.750 | +0.483 | — | — | — | — | 1 |
| C0p | symbolic | paraphrase/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C0p | symbolic | reverse/all | 0.650 | +0.150 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/heldout | 0.500 | +0.000 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/new | 0.750 | +0.250 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/rare | 0.750 | +0.250 | — | — | — | 1.000 | 1 |
| C0p | symbolic | reverse/seen | 0.500 | +0.000 | — | — | — | 1.000 | 1 |
| C5 | none | affordance/all | 0.688 | +0.354 | — | — | — | — | 1 |
| C5 | none | affordance/heldout | 0.750 | +0.417 | — | — | — | — | 1 |
| C5 | none | affordance/new | 0.000 | -0.333 | — | — | — | — | 1 |
| C5 | none | affordance/rare | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | none | affordance/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | none | negation/all | 0.475 | -0.025 | — | — | — | — | 1 |
| C5 | none | negation/heldout | 0.375 | -0.125 | — | — | — | — | 1 |
| C5 | none | negation/new | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | none | negation/rare | 0.625 | +0.125 | — | — | — | — | 1 |
| C5 | none | negation/seen | 0.375 | -0.125 | — | — | — | — | 1 |
| C5 | none | paraphrase/all | 0.650 | +0.343 | — | — | — | — | 1 |
| C5 | none | paraphrase/heldout | 0.750 | +0.417 | — | — | — | — | 1 |
| C5 | none | paraphrase/new | 0.500 | +0.167 | — | — | — | — | 1 |
| C5 | none | paraphrase/rare | 0.500 | +0.233 | — | — | — | — | 1 |
| C5 | none | paraphrase/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | none | reverse/all | 0.350 | -0.150 | — | — | — | — | 1 |
| C5 | none | reverse/heldout | 0.250 | -0.250 | — | — | — | — | 1 |
| C5 | none | reverse/new | 0.375 | -0.125 | — | — | — | — | 1 |
| C5 | none | reverse/rare | 0.250 | -0.250 | — | — | — | — | 1 |
| C5 | none | reverse/seen | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | recall:own | affordance/all | 0.812 | +0.479 | — | — | — | — | 1 |
| C5 | recall:own | affordance/heldout | 0.500 | +0.167 | — | — | — | — | 1 |
| C5 | recall:own | affordance/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | recall:own | affordance/rare | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | recall:own | affordance/seen | 0.750 | +0.417 | — | — | — | — | 1 |
| C5 | recall:own | negation/all | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | recall:own | negation/heldout | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | recall:own | negation/new | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | recall:own | negation/rare | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | recall:own | negation/seen | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | recall:own | paraphrase/all | 0.900 | +0.593 | — | — | — | — | 1 |
| C5 | recall:own | paraphrase/heldout | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | recall:own | paraphrase/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | recall:own | paraphrase/rare | 0.750 | +0.483 | — | — | — | — | 1 |
| C5 | recall:own | paraphrase/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | recall:own | reverse/all | 0.700 | +0.200 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/heldout | 0.250 | -0.250 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/new | 0.875 | +0.375 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/rare | 1.000 | +0.500 | — | — | — | 1.000 | 1 |
| C5 | recall:own | reverse/seen | 0.500 | +0.000 | — | — | — | 1.000 | 1 |
| C5 | symbolic | affordance/all | 0.875 | +0.542 | — | — | — | — | 1 |
| C5 | symbolic | affordance/heldout | 0.750 | +0.417 | — | — | — | — | 1 |
| C5 | symbolic | affordance/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | symbolic | affordance/rare | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | symbolic | affordance/seen | 0.750 | +0.417 | — | — | — | — | 1 |
| C5 | symbolic | negation/all | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | symbolic | negation/heldout | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | symbolic | negation/new | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | symbolic | negation/rare | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | symbolic | negation/seen | 0.500 | +0.000 | — | — | — | — | 1 |
| C5 | symbolic | paraphrase/all | 0.900 | +0.593 | — | — | — | — | 1 |
| C5 | symbolic | paraphrase/heldout | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | symbolic | paraphrase/new | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | symbolic | paraphrase/rare | 0.750 | +0.483 | — | — | — | — | 1 |
| C5 | symbolic | paraphrase/seen | 1.000 | +0.667 | — | — | — | — | 1 |
| C5 | symbolic | reverse/all | 0.700 | +0.200 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/heldout | 0.500 | +0.000 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/new | 0.875 | +0.375 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/rare | 0.750 | +0.250 | — | — | — | 1.000 | 1 |
| C5 | symbolic | reverse/seen | 0.500 | +0.000 | — | — | — | 1.000 | 1 |

