# E9 track zero-shot (t5) — C5ut seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4830 | 0.3895 | 0.6869 |
| none | 0.3504 | 0.3212 | 0.5595 |
| mean_row | 0.3446 | 0.3384 | 0.5744 |
| random_frame | 0.3276 | 0.3499 | 0.5601 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1326 [+0.1162, +0.1488] | 1566 | 0.0030 |
| property | mean_row | +0.1384 [+0.1237, +0.1535] | 1566 | 0.0030 |
| property | random_frame | +0.1554 [+0.1394, +0.1722] | 1566 | 0.0030 |
| paraphrase | none | +0.0683 [+0.0396, +0.0952] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0511 [+0.0249, +0.0760] | 1566 | 0.0040 |
| paraphrase | random_frame | +0.0396 [+0.0115, +0.0664] | 1566 | 0.0050 |
| entailment | none | +0.1274 [+0.1054, +0.1494] | 1680 | 0.0030 |
| entailment | mean_row | +0.1125 [+0.0935, +0.1304] | 1680 | 0.0030 |
| entailment | random_frame | +0.1268 [+0.1071, +0.1470] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4917 | 0.3821 | 0.7017 |
| none | 0.3464 | 0.3411 | 0.5567 |
| mean_row | 0.3393 | 0.3232 | 0.5667 |
| random_frame | 0.3131 | 0.3375 | 0.5550 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1452 [+0.1167, +0.1738] | 560 | 0.0030 |
| property | mean_row | +0.1524 [+0.1268, +0.1780] | 560 | 0.0030 |
| property | random_frame | +0.1786 [+0.1512, +0.2060] | 560 | 0.0030 |
| paraphrase | none | +0.0411 [-0.0071, +0.0911] | 560 | 0.1279 |
| paraphrase | mean_row | +0.0589 [+0.0161, +0.1018] | 560 | 0.0360 |
| paraphrase | random_frame | +0.0446 [-0.0018, +0.0875] | 560 | 0.1279 |
| entailment | none | +0.1450 [+0.1067, +0.1833] | 600 | 0.0030 |
| entailment | mean_row | +0.1350 [+0.1017, +0.1700] | 600 | 0.0030 |
| entailment | random_frame | +0.1467 [+0.1133, +0.1817] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4781 | 0.3936 | 0.6787 |
| none | 0.3526 | 0.3101 | 0.5611 |
| mean_row | 0.3476 | 0.3469 | 0.5787 |
| random_frame | 0.3357 | 0.3569 | 0.5630 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1256 [+0.1067, +0.1458] | 1006 | 0.0030 |
| property | mean_row | +0.1306 [+0.1143, +0.1485] | 1006 | 0.0030 |
| property | random_frame | +0.1425 [+0.1233, +0.1627] | 1006 | 0.0030 |
| paraphrase | none | +0.0835 [+0.0497, +0.1163] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0467 [+0.0159, +0.0775] | 1006 | 0.0060 |
| paraphrase | random_frame | +0.0368 [+0.0050, +0.0696] | 1006 | 0.0350 |
| entailment | none | +0.1176 [+0.0898, +0.1444] | 1080 | 0.0030 |
| entailment | mean_row | +0.1000 [+0.0787, +0.1222] | 1080 | 0.0030 |
| entailment | random_frame | +0.1157 [+0.0917, +0.1398] | 1080 | 0.0030 |
