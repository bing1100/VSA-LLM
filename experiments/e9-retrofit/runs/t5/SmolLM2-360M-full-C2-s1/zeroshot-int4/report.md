# E9 track zero-shot (t5) — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3448 | 0.3416 | 0.5851 |
| none | 0.3470 | 0.3410 | 0.5815 |
| mean_row | 0.3457 | 0.3397 | 0.5810 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0021 [-0.0053, +0.0013] | 1566 | 0.4518 |
| property | mean_row | -0.0009 [-0.0038, +0.0021] | 1566 | 0.6097 |
| paraphrase | none | +0.0006 [-0.0077, +0.0089] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0019 [-0.0051, +0.0089] | 1566 | 1.0000 |
| entailment | none | +0.0036 [-0.0018, +0.0095] | 1680 | 0.3358 |
| entailment | mean_row | +0.0042 [-0.0012, +0.0095] | 1680 | 0.3358 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3333 | 0.3500 | 0.5633 |
| none | 0.3345 | 0.3554 | 0.5567 |
| mean_row | 0.3321 | 0.3500 | 0.5617 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0012 [-0.0066, +0.0048] | 560 | 1.0000 |
| property | mean_row | +0.0012 [-0.0030, +0.0054] | 560 | 1.0000 |
| paraphrase | none | -0.0054 [-0.0214, +0.0107] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0125, +0.0143] | 560 | 1.0000 |
| entailment | none | +0.0067 [-0.0017, +0.0167] | 600 | 0.3998 |
| entailment | mean_row | +0.0017 [-0.0067, +0.0100] | 600 | 0.8416 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3512 | 0.3370 | 0.5972 |
| none | 0.3539 | 0.3330 | 0.5954 |
| mean_row | 0.3532 | 0.3340 | 0.5917 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0027 [-0.0066, +0.0013] | 1006 | 0.4318 |
| property | mean_row | -0.0020 [-0.0060, +0.0017] | 1006 | 0.4318 |
| paraphrase | none | +0.0040 [-0.0060, +0.0139] | 1006 | 0.9455 |
| paraphrase | mean_row | +0.0030 [-0.0050, +0.0109] | 1006 | 0.9455 |
| entailment | none | +0.0019 [-0.0056, +0.0093] | 1080 | 0.7066 |
| entailment | mean_row | +0.0056 [-0.0009, +0.0130] | 1080 | 0.2739 |
