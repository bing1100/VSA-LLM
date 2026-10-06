# E9 track zero-shot (t5) — C5 seed 2 (Qwen/Qwen3-1.7B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4772 | 0.3787 | 0.7089 |
| none | 0.3231 | 0.3416 | 0.5625 |
| mean_row | 0.3203 | 0.3372 | 0.5679 |
| random_frame | 0.3091 | 0.3359 | 0.5589 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1541 [+0.1377, +0.1697] | 1566 | 0.0030 |
| property | mean_row | +0.1569 [+0.1426, +0.1716] | 1566 | 0.0030 |
| property | random_frame | +0.1682 [+0.1530, +0.1841] | 1566 | 0.0030 |
| paraphrase | none | +0.0370 [+0.0109, +0.0626] | 1566 | 0.0050 |
| paraphrase | mean_row | +0.0415 [+0.0185, +0.0632] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0428 [+0.0192, +0.0670] | 1566 | 0.0030 |
| entailment | none | +0.1464 [+0.1256, +0.1667] | 1680 | 0.0030 |
| entailment | mean_row | +0.1411 [+0.1214, +0.1607] | 1680 | 0.0030 |
| entailment | random_frame | +0.1500 [+0.1292, +0.1708] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4714 | 0.3571 | 0.6933 |
| none | 0.3095 | 0.3339 | 0.5567 |
| mean_row | 0.3036 | 0.3250 | 0.5450 |
| random_frame | 0.2786 | 0.3554 | 0.5450 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1619 [+0.1345, +0.1899] | 560 | 0.0030 |
| property | mean_row | +0.1679 [+0.1446, +0.1917] | 560 | 0.0030 |
| property | random_frame | +0.1929 [+0.1679, +0.2208] | 560 | 0.0030 |
| paraphrase | none | +0.0232 [-0.0179, +0.0679] | 560 | 0.6477 |
| paraphrase | mean_row | +0.0321 [-0.0089, +0.0696] | 560 | 0.3808 |
| paraphrase | random_frame | +0.0018 [-0.0393, +0.0429] | 560 | 0.9925 |
| entailment | none | +0.1367 [+0.1033, +0.1733] | 600 | 0.0030 |
| entailment | mean_row | +0.1483 [+0.1167, +0.1833] | 600 | 0.0030 |
| entailment | random_frame | +0.1483 [+0.1150, +0.1833] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4805 | 0.3907 | 0.7176 |
| none | 0.3307 | 0.3459 | 0.5657 |
| mean_row | 0.3297 | 0.3439 | 0.5806 |
| random_frame | 0.3260 | 0.3250 | 0.5667 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1498 [+0.1295, +0.1696] | 1006 | 0.0030 |
| property | mean_row | +0.1508 [+0.1335, +0.1683] | 1006 | 0.0030 |
| property | random_frame | +0.1544 [+0.1352, +0.1723] | 1006 | 0.0030 |
| paraphrase | none | +0.0447 [+0.0129, +0.0775] | 1006 | 0.0050 |
| paraphrase | mean_row | +0.0467 [+0.0179, +0.0755] | 1006 | 0.0040 |
| paraphrase | random_frame | +0.0656 [+0.0348, +0.0964] | 1006 | 0.0030 |
| entailment | none | +0.1519 [+0.1259, +0.1796] | 1080 | 0.0030 |
| entailment | mean_row | +0.1370 [+0.1130, +0.1620] | 1080 | 0.0030 |
| entailment | random_frame | +0.1509 [+0.1269, +0.1759] | 1080 | 0.0030 |
