# E9 track zero-shot (t5) — C2 seed 2 (Qwen/Qwen3-0.6B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.2991 | 0.2446 | 0.5435 |
| none | 0.3033 | 0.2497 | 0.5417 |
| mean_row | 0.3010 | 0.2471 | 0.5429 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0043 [-0.0081, -0.0006] | 1566 | 0.0440 |
| property | mean_row | -0.0019 [-0.0053, +0.0017] | 1566 | 0.2769 |
| paraphrase | none | -0.0051 [-0.0140, +0.0038] | 1566 | 0.6497 |
| paraphrase | mean_row | -0.0026 [-0.0109, +0.0057] | 1566 | 0.6497 |
| entailment | none | +0.0018 [-0.0048, +0.0078] | 1680 | 1.0000 |
| entailment | mean_row | +0.0006 [-0.0054, +0.0065] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3048 | 0.2464 | 0.5450 |
| none | 0.3077 | 0.2554 | 0.5367 |
| mean_row | 0.3048 | 0.2482 | 0.5400 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0030 [-0.0077, +0.0018] | 560 | 0.4698 |
| property | mean_row | +0.0000 [-0.0054, +0.0054] | 560 | 0.9925 |
| paraphrase | none | -0.0089 [-0.0232, +0.0054] | 560 | 0.5397 |
| paraphrase | mean_row | -0.0018 [-0.0161, +0.0143] | 560 | 0.9415 |
| entailment | none | +0.0083 [-0.0033, +0.0217] | 600 | 0.3898 |
| entailment | mean_row | +0.0050 [-0.0050, +0.0167] | 600 | 0.4308 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.2959 | 0.2435 | 0.5426 |
| none | 0.3009 | 0.2465 | 0.5444 |
| mean_row | 0.2989 | 0.2465 | 0.5444 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0050 [-0.0099, -0.0000] | 1006 | 0.0800 |
| property | mean_row | -0.0030 [-0.0073, +0.0013] | 1006 | 0.1659 |
| paraphrase | none | -0.0030 [-0.0149, +0.0089] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0030 [-0.0129, +0.0070] | 1006 | 1.0000 |
| entailment | none | -0.0019 [-0.0102, +0.0065] | 1080 | 1.0000 |
| entailment | mean_row | -0.0019 [-0.0102, +0.0056] | 1080 | 1.0000 |
