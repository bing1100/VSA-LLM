# E9 track zero-shot (t5) — C5 seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4802 | 0.4042 | 0.6810 |
| none | 0.3529 | 0.3321 | 0.5625 |
| mean_row | 0.3472 | 0.3557 | 0.5685 |
| random_frame | 0.3267 | 0.3819 | 0.5607 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1273 [+0.1111, +0.1435] | 1566 | 0.0030 |
| property | mean_row | +0.1330 [+0.1181, +0.1479] | 1566 | 0.0030 |
| property | random_frame | +0.1535 [+0.1371, +0.1701] | 1566 | 0.0030 |
| paraphrase | none | +0.0722 [+0.0447, +0.1009] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0485 [+0.0243, +0.0734] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0223 [-0.0057, +0.0498] | 1566 | 0.1199 |
| entailment | none | +0.1185 [+0.0958, +0.1405] | 1680 | 0.0030 |
| entailment | mean_row | +0.1125 [+0.0929, +0.1315] | 1680 | 0.0030 |
| entailment | random_frame | +0.1202 [+0.1000, +0.1399] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4756 | 0.3768 | 0.6817 |
| none | 0.3488 | 0.3357 | 0.5667 |
| mean_row | 0.3369 | 0.3571 | 0.5650 |
| random_frame | 0.3125 | 0.3911 | 0.5483 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1268 [+0.0976, +0.1572] | 560 | 0.0030 |
| property | mean_row | +0.1387 [+0.1131, +0.1643] | 560 | 0.0030 |
| property | random_frame | +0.1631 [+0.1345, +0.1905] | 560 | 0.0030 |
| paraphrase | none | +0.0411 [-0.0054, +0.0893] | 560 | 0.2759 |
| paraphrase | mean_row | +0.0196 [-0.0214, +0.0589] | 560 | 0.7916 |
| paraphrase | random_frame | -0.0143 [-0.0571, +0.0268] | 560 | 0.7916 |
| entailment | none | +0.1150 [+0.0767, +0.1517] | 600 | 0.0030 |
| entailment | mean_row | +0.1167 [+0.0850, +0.1500] | 600 | 0.0030 |
| entailment | random_frame | +0.1333 [+0.0983, +0.1683] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4828 | 0.4195 | 0.6806 |
| none | 0.3552 | 0.3300 | 0.5602 |
| mean_row | 0.3529 | 0.3549 | 0.5704 |
| random_frame | 0.3347 | 0.3767 | 0.5676 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1276 [+0.1080, +0.1471] | 1006 | 0.0030 |
| property | mean_row | +0.1299 [+0.1130, +0.1481] | 1006 | 0.0030 |
| property | random_frame | +0.1481 [+0.1282, +0.1687] | 1006 | 0.0030 |
| paraphrase | none | +0.0895 [+0.0547, +0.1233] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0646 [+0.0338, +0.0964] | 1006 | 0.0030 |
| paraphrase | random_frame | +0.0427 [+0.0089, +0.0785] | 1006 | 0.0190 |
| entailment | none | +0.1204 [+0.0935, +0.1472] | 1080 | 0.0030 |
| entailment | mean_row | +0.1102 [+0.0870, +0.1343] | 1080 | 0.0030 |
| entailment | random_frame | +0.1130 [+0.0889, +0.1361] | 1080 | 0.0030 |
