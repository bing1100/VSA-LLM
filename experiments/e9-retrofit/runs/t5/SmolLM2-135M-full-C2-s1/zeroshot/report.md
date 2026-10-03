# E9 track zero-shot (t5) — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3280 | 0.3301 | 0.5619 |
| none | 0.3318 | 0.3295 | 0.5643 |
| mean_row | 0.3314 | 0.3378 | 0.5607 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0038 [-0.0074, -0.0002] | 1566 | 0.0700 |
| property | mean_row | -0.0034 [-0.0068, -0.0002] | 1566 | 0.0700 |
| paraphrase | none | +0.0006 [-0.0077, +0.0096] | 1566 | 0.9275 |
| paraphrase | mean_row | -0.0077 [-0.0153, +0.0000] | 1566 | 0.1339 |
| entailment | none | -0.0024 [-0.0083, +0.0036] | 1680 | 0.9555 |
| entailment | mean_row | +0.0012 [-0.0036, +0.0054] | 1680 | 0.9555 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3250 | 0.3321 | 0.5317 |
| none | 0.3268 | 0.3214 | 0.5333 |
| mean_row | 0.3298 | 0.3375 | 0.5300 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0018 [-0.0071, +0.0042] | 560 | 0.5497 |
| property | mean_row | -0.0048 [-0.0101, -0.0000] | 560 | 0.0940 |
| paraphrase | none | +0.0107 [-0.0018, +0.0232] | 560 | 0.2519 |
| paraphrase | mean_row | -0.0054 [-0.0179, +0.0054] | 560 | 0.4348 |
| entailment | none | -0.0017 [-0.0100, +0.0067] | 600 | 1.0000 |
| entailment | mean_row | +0.0017 [-0.0033, +0.0083] | 600 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3297 | 0.3290 | 0.5787 |
| none | 0.3347 | 0.3340 | 0.5815 |
| mean_row | 0.3323 | 0.3380 | 0.5778 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0050 [-0.0099, -0.0003] | 1006 | 0.0640 |
| property | mean_row | -0.0027 [-0.0070, +0.0017] | 1006 | 0.2069 |
| paraphrase | none | -0.0050 [-0.0179, +0.0070] | 1006 | 0.4668 |
| paraphrase | mean_row | -0.0089 [-0.0199, +0.0020] | 1006 | 0.2279 |
| entailment | none | -0.0028 [-0.0102, +0.0046] | 1080 | 1.0000 |
| entailment | mean_row | +0.0009 [-0.0046, +0.0065] | 1080 | 1.0000 |
