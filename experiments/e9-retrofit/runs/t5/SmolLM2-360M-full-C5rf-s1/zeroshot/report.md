# E9 track zero-shot (t5) — C5rf seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4487 | 0.3621 | 0.6679 |
| none | 0.3512 | 0.3257 | 0.5649 |
| mean_row | 0.3491 | 0.3359 | 0.5690 |
| random_frame | 0.3263 | 0.3576 | 0.5482 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0975 [+0.0817, +0.1130] | 1566 | 0.0030 |
| property | mean_row | +0.0996 [+0.0851, +0.1137] | 1566 | 0.0030 |
| property | random_frame | +0.1224 [+0.1066, +0.1377] | 1566 | 0.0030 |
| paraphrase | none | +0.0364 [+0.0096, +0.0626] | 1566 | 0.0150 |
| paraphrase | mean_row | +0.0262 [+0.0038, +0.0504] | 1566 | 0.0660 |
| paraphrase | random_frame | +0.0045 [-0.0230, +0.0313] | 1566 | 0.7476 |
| entailment | none | +0.1030 [+0.0821, +0.1238] | 1680 | 0.0030 |
| entailment | mean_row | +0.0988 [+0.0804, +0.1179] | 1680 | 0.0030 |
| entailment | random_frame | +0.1196 [+0.1006, +0.1393] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4554 | 0.3679 | 0.6750 |
| none | 0.3440 | 0.3286 | 0.5717 |
| mean_row | 0.3357 | 0.3214 | 0.5767 |
| random_frame | 0.3143 | 0.3661 | 0.5450 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1113 [+0.0851, +0.1375] | 560 | 0.0030 |
| property | mean_row | +0.1196 [+0.0970, +0.1440] | 560 | 0.0030 |
| property | random_frame | +0.1411 [+0.1167, +0.1661] | 560 | 0.0030 |
| paraphrase | none | +0.0393 [-0.0036, +0.0821] | 560 | 0.1739 |
| paraphrase | mean_row | +0.0464 [+0.0071, +0.0857] | 560 | 0.0870 |
| paraphrase | random_frame | +0.0018 [-0.0393, +0.0447] | 560 | 0.9685 |
| entailment | none | +0.1033 [+0.0683, +0.1383] | 600 | 0.0030 |
| entailment | mean_row | +0.0983 [+0.0667, +0.1317] | 600 | 0.0030 |
| entailment | random_frame | +0.1300 [+0.0950, +0.1667] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4450 | 0.3588 | 0.6639 |
| none | 0.3552 | 0.3241 | 0.5611 |
| mean_row | 0.3565 | 0.3439 | 0.5648 |
| random_frame | 0.3330 | 0.3529 | 0.5500 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0898 [+0.0716, +0.1080] | 1006 | 0.0030 |
| property | mean_row | +0.0885 [+0.0729, +0.1057] | 1006 | 0.0030 |
| property | random_frame | +0.1120 [+0.0934, +0.1312] | 1006 | 0.0030 |
| paraphrase | none | +0.0348 [+0.0020, +0.0666] | 1006 | 0.1349 |
| paraphrase | mean_row | +0.0149 [-0.0139, +0.0437] | 1006 | 0.6697 |
| paraphrase | random_frame | +0.0060 [-0.0258, +0.0368] | 1006 | 0.7566 |
| entailment | none | +0.1028 [+0.0778, +0.1278] | 1080 | 0.0030 |
| entailment | mean_row | +0.0991 [+0.0759, +0.1222] | 1080 | 0.0030 |
| entailment | random_frame | +0.1139 [+0.0916, +0.1380] | 1080 | 0.0030 |
