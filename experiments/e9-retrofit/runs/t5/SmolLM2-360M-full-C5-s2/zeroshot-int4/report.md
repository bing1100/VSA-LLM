# E9 track zero-shot (t5) — C5 seed 2 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4506 | 0.3736 | 0.6405 |
| none | 0.3425 | 0.3282 | 0.5702 |
| mean_row | 0.3297 | 0.3397 | 0.5476 |
| random_frame | 0.3142 | 0.3289 | 0.5357 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1081 [+0.0917, +0.1247] | 1566 | 0.0030 |
| property | mean_row | +0.1209 [+0.1064, +0.1358] | 1566 | 0.0030 |
| property | random_frame | +0.1364 [+0.1207, +0.1513] | 1566 | 0.0030 |
| paraphrase | none | +0.0453 [+0.0179, +0.0734] | 1566 | 0.0060 |
| paraphrase | mean_row | +0.0338 [+0.0102, +0.0575] | 1566 | 0.0060 |
| paraphrase | random_frame | +0.0447 [+0.0192, +0.0715] | 1566 | 0.0060 |
| entailment | none | +0.0702 [+0.0494, +0.0923] | 1680 | 0.0030 |
| entailment | mean_row | +0.0929 [+0.0744, +0.1119] | 1680 | 0.0030 |
| entailment | random_frame | +0.1048 [+0.0851, +0.1238] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4702 | 0.3732 | 0.6383 |
| none | 0.3411 | 0.3196 | 0.5433 |
| mean_row | 0.3417 | 0.3357 | 0.5433 |
| random_frame | 0.3185 | 0.3232 | 0.5233 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1292 [+0.1012, +0.1571] | 560 | 0.0030 |
| property | mean_row | +0.1286 [+0.1053, +0.1530] | 560 | 0.0030 |
| property | random_frame | +0.1518 [+0.1256, +0.1786] | 560 | 0.0030 |
| paraphrase | none | +0.0536 [+0.0071, +0.1018] | 560 | 0.0750 |
| paraphrase | mean_row | +0.0375 [-0.0036, +0.0786] | 560 | 0.0780 |
| paraphrase | random_frame | +0.0500 [+0.0071, +0.0929] | 560 | 0.0750 |
| entailment | none | +0.0950 [+0.0617, +0.1333] | 600 | 0.0030 |
| entailment | mean_row | +0.0950 [+0.0650, +0.1267] | 600 | 0.0030 |
| entailment | random_frame | +0.1150 [+0.0817, +0.1483] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4397 | 0.3738 | 0.6417 |
| none | 0.3433 | 0.3330 | 0.5852 |
| mean_row | 0.3231 | 0.3419 | 0.5500 |
| random_frame | 0.3118 | 0.3320 | 0.5426 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0964 [+0.0772, +0.1163] | 1006 | 0.0030 |
| property | mean_row | +0.1166 [+0.0984, +0.1345] | 1006 | 0.0030 |
| property | random_frame | +0.1279 [+0.1093, +0.1471] | 1006 | 0.0030 |
| paraphrase | none | +0.0408 [+0.0030, +0.0755] | 1006 | 0.0800 |
| paraphrase | mean_row | +0.0318 [+0.0000, +0.0606] | 1006 | 0.0800 |
| paraphrase | random_frame | +0.0417 [+0.0080, +0.0736] | 1006 | 0.0450 |
| entailment | none | +0.0565 [+0.0324, +0.0833] | 1080 | 0.0030 |
| entailment | mean_row | +0.0917 [+0.0694, +0.1148] | 1080 | 0.0030 |
| entailment | random_frame | +0.0991 [+0.0759, +0.1222] | 1080 | 0.0030 |
