# E9 track zero-shot (t5) — C5tr seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5081 | 0.4189 | 0.7000 |
| none | 0.3555 | 0.3257 | 0.5565 |
| mean_row | 0.3269 | 0.3512 | 0.5595 |
| random_frame | 0.3378 | 0.3819 | 0.5589 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1526 [+0.1335, +0.1699] | 1566 | 0.0030 |
| property | mean_row | +0.1811 [+0.1656, +0.1971] | 1566 | 0.0030 |
| property | random_frame | +0.1703 [+0.1539, +0.1862] | 1566 | 0.0030 |
| paraphrase | none | +0.0932 [+0.0632, +0.1220] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0677 [+0.0389, +0.0939] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0370 [+0.0083, +0.0639] | 1566 | 0.0150 |
| entailment | none | +0.1435 [+0.1196, +0.1679] | 1680 | 0.0030 |
| entailment | mean_row | +0.1405 [+0.1208, +0.1595] | 1680 | 0.0030 |
| entailment | random_frame | +0.1411 [+0.1220, +0.1613] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5149 | 0.4054 | 0.7017 |
| none | 0.3512 | 0.3393 | 0.5467 |
| mean_row | 0.3232 | 0.3732 | 0.5483 |
| random_frame | 0.3232 | 0.3804 | 0.5450 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1637 [+0.1309, +0.1964] | 560 | 0.0030 |
| property | mean_row | +0.1917 [+0.1661, +0.2190] | 560 | 0.0030 |
| property | random_frame | +0.1917 [+0.1643, +0.2214] | 560 | 0.0030 |
| paraphrase | none | +0.0661 [+0.0178, +0.1143] | 560 | 0.0150 |
| paraphrase | mean_row | +0.0321 [-0.0108, +0.0768] | 560 | 0.3138 |
| paraphrase | random_frame | +0.0250 [-0.0214, +0.0696] | 560 | 0.3138 |
| entailment | none | +0.1550 [+0.1167, +0.1950] | 600 | 0.0030 |
| entailment | mean_row | +0.1533 [+0.1217, +0.1867] | 600 | 0.0030 |
| entailment | random_frame | +0.1567 [+0.1200, +0.1933] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5043 | 0.4264 | 0.6991 |
| none | 0.3579 | 0.3181 | 0.5620 |
| mean_row | 0.3290 | 0.3390 | 0.5657 |
| random_frame | 0.3459 | 0.3827 | 0.5667 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1465 [+0.1246, +0.1677] | 1006 | 0.0030 |
| property | mean_row | +0.1753 [+0.1564, +0.1952] | 1006 | 0.0030 |
| property | random_frame | +0.1584 [+0.1385, +0.1783] | 1006 | 0.0030 |
| paraphrase | none | +0.1083 [+0.0736, +0.1431] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0875 [+0.0537, +0.1213] | 1006 | 0.0030 |
| paraphrase | random_frame | +0.0437 [+0.0109, +0.0795] | 1006 | 0.0080 |
| entailment | none | +0.1370 [+0.1065, +0.1667] | 1080 | 0.0030 |
| entailment | mean_row | +0.1333 [+0.1093, +0.1583] | 1080 | 0.0030 |
| entailment | random_frame | +0.1324 [+0.1083, +0.1583] | 1080 | 0.0030 |
