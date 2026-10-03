# E9 track zero-shot (t5) — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3163 | 0.2663 | 0.5286 |
| none | 0.3174 | 0.2631 | 0.5280 |
| mean_row | 0.3167 | 0.2650 | 0.5339 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0011 [-0.0049, +0.0026] | 1566 | 1.0000 |
| property | mean_row | -0.0004 [-0.0036, +0.0028] | 1566 | 1.0000 |
| paraphrase | none | +0.0032 [-0.0051, +0.0115] | 1566 | 0.9895 |
| paraphrase | mean_row | +0.0013 [-0.0057, +0.0083] | 1566 | 0.9895 |
| entailment | none | +0.0006 [-0.0060, +0.0071] | 1680 | 0.9025 |
| entailment | mean_row | -0.0054 [-0.0113, +0.0000] | 1680 | 0.1619 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3190 | 0.2661 | 0.5367 |
| none | 0.3214 | 0.2607 | 0.5400 |
| mean_row | 0.3202 | 0.2625 | 0.5433 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0024 [-0.0089, +0.0048] | 560 | 1.0000 |
| property | mean_row | -0.0012 [-0.0065, +0.0042] | 560 | 1.0000 |
| paraphrase | none | +0.0054 [-0.0089, +0.0214] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0036 [-0.0089, +0.0179] | 560 | 1.0000 |
| entailment | none | -0.0033 [-0.0133, +0.0067] | 600 | 0.6497 |
| entailment | mean_row | -0.0067 [-0.0183, +0.0033] | 600 | 0.6497 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3148 | 0.2664 | 0.5241 |
| none | 0.3151 | 0.2644 | 0.5213 |
| mean_row | 0.3148 | 0.2664 | 0.5287 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0003 [-0.0050, +0.0046] | 1006 | 1.0000 |
| property | mean_row | +0.0000 [-0.0043, +0.0040] | 1006 | 1.0000 |
| paraphrase | none | +0.0020 [-0.0070, +0.0119] | 1006 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0080, +0.0080] | 1006 | 1.0000 |
| entailment | none | +0.0028 [-0.0056, +0.0111] | 1080 | 0.6397 |
| entailment | mean_row | -0.0046 [-0.0111, +0.0019] | 1080 | 0.4418 |
