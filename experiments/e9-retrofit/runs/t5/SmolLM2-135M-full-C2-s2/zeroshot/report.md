# E9 track zero-shot (t5) — C2 seed 2 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3299 | 0.3340 | 0.5744 |
| none | 0.3278 | 0.3346 | 0.5702 |
| mean_row | 0.3280 | 0.3295 | 0.5661 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0021 [-0.0013, +0.0057] | 1566 | 0.4178 |
| property | mean_row | +0.0019 [-0.0013, +0.0051] | 1566 | 0.4178 |
| paraphrase | none | -0.0006 [-0.0089, +0.0077] | 1566 | 0.9525 |
| paraphrase | mean_row | +0.0045 [-0.0026, +0.0115] | 1566 | 0.4838 |
| entailment | none | +0.0042 [-0.0018, +0.0101] | 1680 | 0.2029 |
| entailment | mean_row | +0.0083 [+0.0018, +0.0149] | 1680 | 0.0320 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3256 | 0.3375 | 0.5450 |
| none | 0.3262 | 0.3321 | 0.5400 |
| mean_row | 0.3250 | 0.3321 | 0.5300 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0006 [-0.0065, +0.0054] | 560 | 1.0000 |
| property | mean_row | +0.0006 [-0.0036, +0.0048] | 560 | 1.0000 |
| paraphrase | none | +0.0054 [-0.0071, +0.0196] | 560 | 0.9055 |
| paraphrase | mean_row | +0.0054 [-0.0054, +0.0179] | 560 | 0.9055 |
| entailment | none | +0.0050 [-0.0067, +0.0167] | 600 | 0.4838 |
| entailment | mean_row | +0.0150 [+0.0050, +0.0267] | 600 | 0.0100 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3323 | 0.3320 | 0.5907 |
| none | 0.3287 | 0.3360 | 0.5870 |
| mean_row | 0.3297 | 0.3280 | 0.5861 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0036 [-0.0010, +0.0083] | 1006 | 0.2859 |
| property | mean_row | +0.0027 [-0.0013, +0.0066] | 1006 | 0.2859 |
| paraphrase | none | -0.0040 [-0.0149, +0.0060] | 1006 | 0.8876 |
| paraphrase | mean_row | +0.0040 [-0.0050, +0.0129] | 1006 | 0.8876 |
| entailment | none | +0.0037 [-0.0028, +0.0111] | 1080 | 0.5677 |
| entailment | mean_row | +0.0046 [-0.0028, +0.0130] | 1080 | 0.5677 |
