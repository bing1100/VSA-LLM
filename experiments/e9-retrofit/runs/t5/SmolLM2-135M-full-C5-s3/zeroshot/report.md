# E9 track zero-shot (t5) — C5 seed 3 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4372 | 0.3691 | 0.6649 |
| none | 0.3287 | 0.3423 | 0.5673 |
| mean_row | 0.3191 | 0.3276 | 0.5631 |
| random_frame | 0.3067 | 0.3321 | 0.5488 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1086 [+0.0907, +0.1249] | 1566 | 0.0030 |
| property | mean_row | +0.1181 [+0.1043, +0.1322] | 1566 | 0.0030 |
| property | random_frame | +0.1305 [+0.1149, +0.1467] | 1566 | 0.0030 |
| paraphrase | none | +0.0268 [+0.0000, +0.0537] | 1566 | 0.0550 |
| paraphrase | mean_row | +0.0415 [+0.0179, +0.0658] | 1566 | 0.0060 |
| paraphrase | random_frame | +0.0370 [+0.0121, +0.0632] | 1566 | 0.0100 |
| entailment | none | +0.0976 [+0.0762, +0.1185] | 1680 | 0.0030 |
| entailment | mean_row | +0.1018 [+0.0833, +0.1202] | 1680 | 0.0030 |
| entailment | random_frame | +0.1161 [+0.0958, +0.1363] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4440 | 0.3857 | 0.6550 |
| none | 0.3280 | 0.3357 | 0.5483 |
| mean_row | 0.3190 | 0.2929 | 0.5250 |
| random_frame | 0.3018 | 0.3179 | 0.5033 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1161 [+0.0893, +0.1441] | 560 | 0.0030 |
| property | mean_row | +0.1250 [+0.1006, +0.1506] | 560 | 0.0030 |
| property | random_frame | +0.1423 [+0.1155, +0.1685] | 560 | 0.0030 |
| paraphrase | none | +0.0500 [+0.0054, +0.0929] | 560 | 0.0320 |
| paraphrase | mean_row | +0.0929 [+0.0536, +0.1339] | 560 | 0.0030 |
| paraphrase | random_frame | +0.0679 [+0.0232, +0.1107] | 560 | 0.0060 |
| entailment | none | +0.1067 [+0.0733, +0.1433] | 600 | 0.0030 |
| entailment | mean_row | +0.1300 [+0.0983, +0.1633] | 600 | 0.0030 |
| entailment | random_frame | +0.1517 [+0.1200, +0.1850] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4334 | 0.3598 | 0.6704 |
| none | 0.3290 | 0.3459 | 0.5778 |
| mean_row | 0.3191 | 0.3469 | 0.5843 |
| random_frame | 0.3095 | 0.3400 | 0.5741 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1044 [+0.0828, +0.1262] | 1006 | 0.0030 |
| property | mean_row | +0.1143 [+0.0971, +0.1322] | 1006 | 0.0030 |
| property | random_frame | +0.1239 [+0.1057, +0.1435] | 1006 | 0.0030 |
| paraphrase | none | +0.0139 [-0.0189, +0.0477] | 1006 | 0.7336 |
| paraphrase | mean_row | +0.0129 [-0.0149, +0.0417] | 1006 | 0.7336 |
| paraphrase | random_frame | +0.0199 [-0.0109, +0.0527] | 1006 | 0.6567 |
| entailment | none | +0.0926 [+0.0667, +0.1194] | 1080 | 0.0030 |
| entailment | mean_row | +0.0861 [+0.0630, +0.1084] | 1080 | 0.0030 |
| entailment | random_frame | +0.0963 [+0.0713, +0.1194] | 1080 | 0.0030 |
