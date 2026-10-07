# E9 track zero-shot (t5) — C5tr seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5064 | 0.4093 | 0.6964 |
| none | 0.3480 | 0.3218 | 0.5548 |
| mean_row | 0.3304 | 0.3614 | 0.5649 |
| random_frame | 0.3350 | 0.3761 | 0.5601 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1584 [+0.1405, +0.1767] | 1566 | 0.0030 |
| property | mean_row | +0.1760 [+0.1596, +0.1922] | 1566 | 0.0030 |
| property | random_frame | +0.1713 [+0.1547, +0.1884] | 1566 | 0.0030 |
| paraphrase | none | +0.0875 [+0.0581, +0.1181] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0479 [+0.0204, +0.0741] | 1566 | 0.0060 |
| paraphrase | random_frame | +0.0332 [+0.0051, +0.0600] | 1566 | 0.0140 |
| entailment | none | +0.1417 [+0.1190, +0.1649] | 1680 | 0.0030 |
| entailment | mean_row | +0.1315 [+0.1131, +0.1512] | 1680 | 0.0030 |
| entailment | random_frame | +0.1363 [+0.1173, +0.1571] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5054 | 0.4089 | 0.7050 |
| none | 0.3464 | 0.3357 | 0.5517 |
| mean_row | 0.3280 | 0.3929 | 0.5517 |
| random_frame | 0.3167 | 0.3929 | 0.5350 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1589 [+0.1286, +0.1917] | 560 | 0.0030 |
| property | mean_row | +0.1774 [+0.1494, +0.2060] | 560 | 0.0030 |
| property | random_frame | +0.1887 [+0.1607, +0.2185] | 560 | 0.0030 |
| paraphrase | none | +0.0732 [+0.0268, +0.1196] | 560 | 0.0090 |
| paraphrase | mean_row | +0.0161 [-0.0304, +0.0625] | 560 | 0.9995 |
| paraphrase | random_frame | +0.0161 [-0.0321, +0.0625] | 560 | 0.9995 |
| entailment | none | +0.1533 [+0.1133, +0.1950] | 600 | 0.0030 |
| entailment | mean_row | +0.1533 [+0.1200, +0.1883] | 600 | 0.0030 |
| entailment | random_frame | +0.1700 [+0.1350, +0.2067] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5070 | 0.4095 | 0.6917 |
| none | 0.3489 | 0.3141 | 0.5565 |
| mean_row | 0.3317 | 0.3439 | 0.5722 |
| random_frame | 0.3453 | 0.3668 | 0.5741 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1581 [+0.1368, +0.1796] | 1006 | 0.0030 |
| property | mean_row | +0.1753 [+0.1564, +0.1948] | 1006 | 0.0030 |
| property | random_frame | +0.1617 [+0.1411, +0.1836] | 1006 | 0.0030 |
| paraphrase | none | +0.0954 [+0.0596, +0.1332] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0656 [+0.0318, +0.1004] | 1006 | 0.0030 |
| paraphrase | random_frame | +0.0427 [+0.0080, +0.0765] | 1006 | 0.0170 |
| entailment | none | +0.1352 [+0.1065, +0.1648] | 1080 | 0.0030 |
| entailment | mean_row | +0.1194 [+0.0972, +0.1426] | 1080 | 0.0030 |
| entailment | random_frame | +0.1176 [+0.0926, +0.1426] | 1080 | 0.0030 |
