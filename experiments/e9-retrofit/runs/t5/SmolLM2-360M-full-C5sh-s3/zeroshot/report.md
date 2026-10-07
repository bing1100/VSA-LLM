# E9 track zero-shot (t5) — C5sh seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3421 | 0.3186 | 0.5774 |
| none | 0.3533 | 0.3110 | 0.5613 |
| mean_row | 0.3495 | 0.3269 | 0.5756 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0113 [-0.0238, +0.0006] | 1566 | 0.1339 |
| property | mean_row | -0.0074 [-0.0166, +0.0015] | 1566 | 0.1339 |
| paraphrase | none | +0.0077 [-0.0147, +0.0300] | 1566 | 0.7756 |
| paraphrase | mean_row | -0.0083 [-0.0262, +0.0102] | 1566 | 0.7756 |
| entailment | none | +0.0161 [-0.0018, +0.0345] | 1680 | 0.1759 |
| entailment | mean_row | +0.0018 [-0.0125, +0.0155] | 1680 | 0.8386 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3429 | 0.3464 | 0.5900 |
| none | 0.3500 | 0.3196 | 0.5600 |
| mean_row | 0.3494 | 0.3429 | 0.5767 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0071 [-0.0292, +0.0149] | 560 | 0.7736 |
| property | mean_row | -0.0065 [-0.0220, +0.0077] | 560 | 0.7736 |
| paraphrase | none | +0.0268 [-0.0125, +0.0661] | 560 | 0.3558 |
| paraphrase | mean_row | +0.0036 [-0.0268, +0.0357] | 560 | 0.8696 |
| entailment | none | +0.0300 [-0.0017, +0.0600] | 600 | 0.1359 |
| entailment | mean_row | +0.0133 [-0.0117, +0.0367] | 600 | 0.3458 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3416 | 0.3032 | 0.5704 |
| none | 0.3552 | 0.3062 | 0.5620 |
| mean_row | 0.3496 | 0.3181 | 0.5750 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0136 [-0.0272, +0.0013] | 1006 | 0.1399 |
| property | mean_row | -0.0080 [-0.0186, +0.0030] | 1006 | 0.1449 |
| paraphrase | none | -0.0030 [-0.0308, +0.0239] | 1006 | 0.8646 |
| paraphrase | mean_row | -0.0149 [-0.0368, +0.0070] | 1006 | 0.3978 |
| entailment | none | +0.0083 [-0.0139, +0.0287] | 1080 | 0.9475 |
| entailment | mean_row | -0.0046 [-0.0213, +0.0120] | 1080 | 0.9475 |
