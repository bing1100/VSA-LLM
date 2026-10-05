# E9 track zero-shot (t5) — C2 seed 3 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3193 | 0.2446 | 0.5339 |
| none | 0.3191 | 0.2497 | 0.5286 |
| mean_row | 0.3193 | 0.2395 | 0.5381 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0002 [-0.0032, +0.0038] | 1566 | 1.0000 |
| property | mean_row | +0.0000 [-0.0036, +0.0036] | 1566 | 1.0000 |
| paraphrase | none | -0.0051 [-0.0140, +0.0032] | 1566 | 0.4918 |
| paraphrase | mean_row | +0.0051 [-0.0026, +0.0128] | 1566 | 0.4918 |
| entailment | none | +0.0054 [+0.0000, +0.0107] | 1680 | 0.1099 |
| entailment | mean_row | -0.0042 [-0.0107, +0.0018] | 1680 | 0.2059 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3173 | 0.2589 | 0.5383 |
| none | 0.3179 | 0.2571 | 0.5383 |
| mean_row | 0.3131 | 0.2482 | 0.5433 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0006 [-0.0065, +0.0054] | 560 | 0.8596 |
| property | mean_row | +0.0042 [-0.0030, +0.0113] | 560 | 0.4518 |
| paraphrase | none | +0.0018 [-0.0125, +0.0161] | 560 | 0.8546 |
| paraphrase | mean_row | +0.0107 [-0.0036, +0.0250] | 560 | 0.3778 |
| entailment | none | +0.0000 [-0.0067, +0.0067] | 600 | 1.0000 |
| entailment | mean_row | -0.0050 [-0.0150, +0.0050] | 600 | 0.7736 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3204 | 0.2366 | 0.5315 |
| none | 0.3197 | 0.2455 | 0.5231 |
| mean_row | 0.3227 | 0.2346 | 0.5352 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0007 [-0.0030, +0.0046] | 1006 | 0.8006 |
| property | mean_row | -0.0023 [-0.0063, +0.0017] | 1006 | 0.5737 |
| paraphrase | none | -0.0089 [-0.0189, +0.0010] | 1006 | 0.2179 |
| paraphrase | mean_row | +0.0020 [-0.0070, +0.0109] | 1006 | 0.7256 |
| entailment | none | +0.0083 [+0.0009, +0.0157] | 1080 | 0.0680 |
| entailment | mean_row | -0.0037 [-0.0111, +0.0037] | 1080 | 0.4468 |
