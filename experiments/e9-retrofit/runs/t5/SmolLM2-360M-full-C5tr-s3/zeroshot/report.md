# E9 track zero-shot (t5) — C5tr seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4996 | 0.4093 | 0.7000 |
| none | 0.3501 | 0.3238 | 0.5542 |
| mean_row | 0.3338 | 0.3557 | 0.5673 |
| random_frame | 0.3427 | 0.3716 | 0.5643 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1494 [+0.1322, +0.1675] | 1566 | 0.0030 |
| property | mean_row | +0.1658 [+0.1507, +0.1811] | 1566 | 0.0030 |
| property | random_frame | +0.1569 [+0.1407, +0.1737] | 1566 | 0.0030 |
| paraphrase | none | +0.0856 [+0.0543, +0.1156] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0536 [+0.0262, +0.0805] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0377 [+0.0102, +0.0645] | 1566 | 0.0130 |
| entailment | none | +0.1458 [+0.1232, +0.1690] | 1680 | 0.0030 |
| entailment | mean_row | +0.1327 [+0.1125, +0.1518] | 1680 | 0.0030 |
| entailment | random_frame | +0.1357 [+0.1167, +0.1548] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4976 | 0.3982 | 0.7100 |
| none | 0.3423 | 0.3304 | 0.5550 |
| mean_row | 0.3232 | 0.3750 | 0.5500 |
| random_frame | 0.3238 | 0.3679 | 0.5450 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1554 [+0.1232, +0.1887] | 560 | 0.0030 |
| property | mean_row | +0.1744 [+0.1488, +0.2018] | 560 | 0.0030 |
| property | random_frame | +0.1738 [+0.1452, +0.2036] | 560 | 0.0030 |
| paraphrase | none | +0.0679 [+0.0161, +0.1196] | 560 | 0.0420 |
| paraphrase | mean_row | +0.0232 [-0.0250, +0.0697] | 560 | 0.4718 |
| paraphrase | random_frame | +0.0304 [-0.0179, +0.0768] | 560 | 0.4718 |
| entailment | none | +0.1550 [+0.1150, +0.1950] | 600 | 0.0030 |
| entailment | mean_row | +0.1600 [+0.1300, +0.1917] | 600 | 0.0030 |
| entailment | random_frame | +0.1650 [+0.1333, +0.1983] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5007 | 0.4155 | 0.6944 |
| none | 0.3545 | 0.3201 | 0.5537 |
| mean_row | 0.3396 | 0.3449 | 0.5769 |
| random_frame | 0.3532 | 0.3738 | 0.5750 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1461 [+0.1259, +0.1680] | 1006 | 0.0030 |
| property | mean_row | +0.1610 [+0.1421, +0.1809] | 1006 | 0.0030 |
| property | random_frame | +0.1474 [+0.1279, +0.1677] | 1006 | 0.0030 |
| paraphrase | none | +0.0954 [+0.0606, +0.1322] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0706 [+0.0398, +0.1024] | 1006 | 0.0030 |
| paraphrase | random_frame | +0.0417 [+0.0080, +0.0736] | 1006 | 0.0180 |
| entailment | none | +0.1407 [+0.1111, +0.1704] | 1080 | 0.0030 |
| entailment | mean_row | +0.1176 [+0.0935, +0.1417] | 1080 | 0.0030 |
| entailment | random_frame | +0.1194 [+0.0972, +0.1426] | 1080 | 0.0030 |
