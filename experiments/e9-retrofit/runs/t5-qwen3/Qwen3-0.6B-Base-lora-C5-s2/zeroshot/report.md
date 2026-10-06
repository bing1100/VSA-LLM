# E9 track zero-shot (t5) — C5 seed 2 (Qwen/Qwen3-0.6B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4285 | 0.2848 | 0.6744 |
| none | 0.3099 | 0.2752 | 0.5423 |
| mean_row | 0.2865 | 0.2810 | 0.5411 |
| random_frame | 0.2908 | 0.2810 | 0.5220 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1186 [+0.1028, +0.1350] | 1566 | 0.0030 |
| property | mean_row | +0.1420 [+0.1279, +0.1558] | 1566 | 0.0030 |
| property | random_frame | +0.1377 [+0.1239, +0.1524] | 1566 | 0.0030 |
| paraphrase | none | +0.0096 [-0.0160, +0.0338] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0038 [-0.0185, +0.0243] | 1566 | 1.0000 |
| paraphrase | random_frame | +0.0038 [-0.0185, +0.0262] | 1566 | 1.0000 |
| entailment | none | +0.1321 [+0.1107, +0.1524] | 1680 | 0.0030 |
| entailment | mean_row | +0.1333 [+0.1143, +0.1530] | 1680 | 0.0030 |
| entailment | random_frame | +0.1524 [+0.1321, +0.1732] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4190 | 0.2571 | 0.6817 |
| none | 0.3006 | 0.2536 | 0.5300 |
| mean_row | 0.2714 | 0.2696 | 0.5450 |
| random_frame | 0.2720 | 0.2679 | 0.5233 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1185 [+0.0905, +0.1464] | 560 | 0.0030 |
| property | mean_row | +0.1476 [+0.1244, +0.1696] | 560 | 0.0030 |
| property | random_frame | +0.1470 [+0.1226, +0.1714] | 560 | 0.0030 |
| paraphrase | none | +0.0036 [-0.0411, +0.0464] | 560 | 1.0000 |
| paraphrase | mean_row | -0.0125 [-0.0465, +0.0232] | 560 | 1.0000 |
| paraphrase | random_frame | -0.0107 [-0.0483, +0.0286] | 560 | 1.0000 |
| entailment | none | +0.1517 [+0.1167, +0.1884] | 600 | 0.0030 |
| entailment | mean_row | +0.1367 [+0.1067, +0.1700] | 600 | 0.0030 |
| entailment | random_frame | +0.1583 [+0.1233, +0.1967] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4337 | 0.3002 | 0.6704 |
| none | 0.3151 | 0.2873 | 0.5491 |
| mean_row | 0.2949 | 0.2873 | 0.5389 |
| random_frame | 0.3012 | 0.2883 | 0.5213 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1186 [+0.0981, +0.1405] | 1006 | 0.0030 |
| property | mean_row | +0.1388 [+0.1209, +0.1557] | 1006 | 0.0030 |
| property | random_frame | +0.1325 [+0.1143, +0.1498] | 1006 | 0.0030 |
| paraphrase | none | +0.0129 [-0.0169, +0.0427] | 1006 | 1.0000 |
| paraphrase | mean_row | +0.0129 [-0.0139, +0.0417] | 1006 | 1.0000 |
| paraphrase | random_frame | +0.0119 [-0.0159, +0.0417] | 1006 | 1.0000 |
| entailment | none | +0.1213 [+0.0954, +0.1482] | 1080 | 0.0030 |
| entailment | mean_row | +0.1315 [+0.1092, +0.1556] | 1080 | 0.0030 |
| entailment | random_frame | +0.1491 [+0.1231, +0.1750] | 1080 | 0.0030 |
