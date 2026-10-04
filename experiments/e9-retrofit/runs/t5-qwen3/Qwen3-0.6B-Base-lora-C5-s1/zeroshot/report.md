# E9 track zero-shot (t5) — C5 seed 1 (Qwen/Qwen3-0.6B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4225 | 0.3020 | 0.6649 |
| none | 0.3129 | 0.2842 | 0.5417 |
| mean_row | 0.2816 | 0.2848 | 0.5357 |
| random_frame | 0.2840 | 0.2822 | 0.5101 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1096 [+0.0941, +0.1260] | 1566 | 0.0030 |
| property | mean_row | +0.1409 [+0.1277, +0.1547] | 1566 | 0.0030 |
| property | random_frame | +0.1386 [+0.1249, +0.1528] | 1566 | 0.0030 |
| paraphrase | none | +0.0179 [-0.0089, +0.0434] | 1566 | 0.2719 |
| paraphrase | mean_row | +0.0172 [-0.0038, +0.0396] | 1566 | 0.2719 |
| paraphrase | random_frame | +0.0198 [-0.0019, +0.0428] | 1566 | 0.2519 |
| entailment | none | +0.1232 [+0.1030, +0.1435] | 1680 | 0.0030 |
| entailment | mean_row | +0.1292 [+0.1095, +0.1482] | 1680 | 0.0030 |
| entailment | random_frame | +0.1548 [+0.1351, +0.1756] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4131 | 0.2589 | 0.6600 |
| none | 0.3083 | 0.2500 | 0.5383 |
| mean_row | 0.2679 | 0.2554 | 0.5400 |
| random_frame | 0.2589 | 0.2643 | 0.4933 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1048 [+0.0786, +0.1321] | 560 | 0.0030 |
| property | mean_row | +0.1452 [+0.1232, +0.1685] | 560 | 0.0030 |
| property | random_frame | +0.1542 [+0.1292, +0.1786] | 560 | 0.0030 |
| paraphrase | none | +0.0089 [-0.0321, +0.0518] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0036 [-0.0322, +0.0393] | 560 | 1.0000 |
| paraphrase | random_frame | -0.0054 [-0.0446, +0.0339] | 560 | 1.0000 |
| entailment | none | +0.1217 [+0.0850, +0.1583] | 600 | 0.0030 |
| entailment | mean_row | +0.1200 [+0.0883, +0.1533] | 600 | 0.0030 |
| entailment | random_frame | +0.1667 [+0.1333, +0.2067] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4278 | 0.3260 | 0.6676 |
| none | 0.3154 | 0.3032 | 0.5435 |
| mean_row | 0.2893 | 0.3012 | 0.5333 |
| random_frame | 0.2979 | 0.2922 | 0.5194 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1123 [+0.0928, +0.1326] | 1006 | 0.0030 |
| property | mean_row | +0.1385 [+0.1226, +0.1551] | 1006 | 0.0030 |
| property | random_frame | +0.1299 [+0.1130, +0.1481] | 1006 | 0.0030 |
| paraphrase | none | +0.0229 [-0.0099, +0.0567] | 1006 | 0.1919 |
| paraphrase | mean_row | +0.0249 [-0.0040, +0.0557] | 1006 | 0.1919 |
| paraphrase | random_frame | +0.0338 [+0.0060, +0.0636] | 1006 | 0.0510 |
| entailment | none | +0.1241 [+0.0991, +0.1500] | 1080 | 0.0030 |
| entailment | mean_row | +0.1343 [+0.1102, +0.1593] | 1080 | 0.0030 |
| entailment | random_frame | +0.1481 [+0.1241, +0.1741] | 1080 | 0.0030 |
