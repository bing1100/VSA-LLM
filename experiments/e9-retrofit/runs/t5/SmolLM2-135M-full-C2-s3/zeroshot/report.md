# E9 track zero-shot (t5) — C2 seed 3 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3291 | 0.3244 | 0.5637 |
| none | 0.3301 | 0.3238 | 0.5649 |
| mean_row | 0.3308 | 0.3244 | 0.5643 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0011 [-0.0047, +0.0026] | 1566 | 0.6257 |
| property | mean_row | -0.0017 [-0.0049, +0.0015] | 1566 | 0.6257 |
| paraphrase | none | +0.0006 [-0.0077, +0.0089] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0077, +0.0077] | 1566 | 1.0000 |
| entailment | none | -0.0012 [-0.0065, +0.0042] | 1680 | 1.0000 |
| entailment | mean_row | -0.0006 [-0.0054, +0.0042] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3238 | 0.3393 | 0.5283 |
| none | 0.3250 | 0.3304 | 0.5350 |
| mean_row | 0.3244 | 0.3357 | 0.5333 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0012 [-0.0065, +0.0048] | 560 | 1.0000 |
| property | mean_row | -0.0006 [-0.0054, +0.0042] | 560 | 1.0000 |
| paraphrase | none | +0.0089 [-0.0054, +0.0232] | 560 | 0.4698 |
| paraphrase | mean_row | +0.0036 [-0.0089, +0.0161] | 560 | 0.6547 |
| entailment | none | -0.0067 [-0.0150, +0.0017] | 600 | 0.2979 |
| entailment | mean_row | -0.0050 [-0.0150, +0.0050] | 600 | 0.4068 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3320 | 0.3161 | 0.5833 |
| none | 0.3330 | 0.3201 | 0.5815 |
| mean_row | 0.3343 | 0.3181 | 0.5815 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0010 [-0.0056, +0.0033] | 1006 | 0.6197 |
| property | mean_row | -0.0023 [-0.0066, +0.0017] | 1006 | 0.5517 |
| paraphrase | none | -0.0040 [-0.0149, +0.0070] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0020 [-0.0109, +0.0070] | 1006 | 1.0000 |
| entailment | none | +0.0019 [-0.0056, +0.0093] | 1080 | 1.0000 |
| entailment | mean_row | +0.0019 [-0.0037, +0.0074] | 1080 | 1.0000 |
