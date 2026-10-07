# E9 track zero-shot (t5) — C5ut seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4779 | 0.3831 | 0.6929 |
| none | 0.3555 | 0.3269 | 0.5518 |
| mean_row | 0.3453 | 0.3544 | 0.5810 |
| random_frame | 0.3242 | 0.3736 | 0.5601 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1224 [+0.1058, +0.1394] | 1566 | 0.0030 |
| property | mean_row | +0.1326 [+0.1171, +0.1473] | 1566 | 0.0030 |
| property | random_frame | +0.1537 [+0.1358, +0.1709] | 1566 | 0.0030 |
| paraphrase | none | +0.0562 [+0.0287, +0.0837] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0287 [+0.0025, +0.0543] | 1566 | 0.0760 |
| paraphrase | random_frame | +0.0096 [-0.0172, +0.0364] | 1566 | 0.5457 |
| entailment | none | +0.1411 [+0.1185, +0.1625] | 1680 | 0.0030 |
| entailment | mean_row | +0.1119 [+0.0929, +0.1310] | 1680 | 0.0030 |
| entailment | random_frame | +0.1327 [+0.1125, +0.1530] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4774 | 0.3839 | 0.7117 |
| none | 0.3512 | 0.3357 | 0.5533 |
| mean_row | 0.3429 | 0.3500 | 0.5700 |
| random_frame | 0.3077 | 0.3839 | 0.5567 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1262 [+0.0976, +0.1560] | 560 | 0.0030 |
| property | mean_row | +0.1345 [+0.1095, +0.1625] | 560 | 0.0030 |
| property | random_frame | +0.1696 [+0.1429, +0.1982] | 560 | 0.0030 |
| paraphrase | none | +0.0482 [+0.0018, +0.0964] | 560 | 0.1349 |
| paraphrase | mean_row | +0.0339 [-0.0089, +0.0786] | 560 | 0.2739 |
| paraphrase | random_frame | +0.0000 [-0.0464, +0.0482] | 560 | 1.0000 |
| entailment | none | +0.1583 [+0.1183, +0.1967] | 600 | 0.0030 |
| entailment | mean_row | +0.1417 [+0.1100, +0.1750] | 600 | 0.0030 |
| entailment | random_frame | +0.1550 [+0.1217, +0.1917] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4781 | 0.3827 | 0.6824 |
| none | 0.3579 | 0.3221 | 0.5509 |
| mean_row | 0.3466 | 0.3569 | 0.5870 |
| random_frame | 0.3333 | 0.3678 | 0.5620 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1203 [+0.1007, +0.1395] | 1006 | 0.0030 |
| property | mean_row | +0.1315 [+0.1140, +0.1498] | 1006 | 0.0030 |
| property | random_frame | +0.1448 [+0.1236, +0.1653] | 1006 | 0.0030 |
| paraphrase | none | +0.0606 [+0.0268, +0.0934] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0258 [-0.0060, +0.0557] | 1006 | 0.2619 |
| paraphrase | random_frame | +0.0149 [-0.0189, +0.0467] | 1006 | 0.4028 |
| entailment | none | +0.1315 [+0.1037, +0.1583] | 1080 | 0.0030 |
| entailment | mean_row | +0.0954 [+0.0722, +0.1185] | 1080 | 0.0030 |
| entailment | random_frame | +0.1204 [+0.0963, +0.1435] | 1080 | 0.0030 |
