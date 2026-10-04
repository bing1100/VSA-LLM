# E9 track zero-shot (t5) — C5 seed 1 (Qwen/Qwen3-1.7B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4908 | 0.3902 | 0.7173 |
| none | 0.3240 | 0.3212 | 0.5750 |
| mean_row | 0.3252 | 0.3257 | 0.5786 |
| random_frame | 0.3159 | 0.3480 | 0.5714 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1669 [+0.1507, +0.1831] | 1566 | 0.0030 |
| property | mean_row | +0.1656 [+0.1516, +0.1814] | 1566 | 0.0030 |
| property | random_frame | +0.1750 [+0.1592, +0.1907] | 1566 | 0.0030 |
| paraphrase | none | +0.0690 [+0.0421, +0.0964] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0645 [+0.0415, +0.0881] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0421 [+0.0160, +0.0670] | 1566 | 0.0030 |
| entailment | none | +0.1423 [+0.1220, +0.1631] | 1680 | 0.0030 |
| entailment | mean_row | +0.1387 [+0.1190, +0.1583] | 1680 | 0.0030 |
| entailment | random_frame | +0.1458 [+0.1262, +0.1661] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4857 | 0.3643 | 0.7067 |
| none | 0.3018 | 0.2982 | 0.5517 |
| mean_row | 0.3196 | 0.3054 | 0.5517 |
| random_frame | 0.2917 | 0.3446 | 0.5483 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1839 [+0.1571, +0.2125] | 560 | 0.0030 |
| property | mean_row | +0.1661 [+0.1429, +0.1911] | 560 | 0.0030 |
| property | random_frame | +0.1940 [+0.1679, +0.2214] | 560 | 0.0030 |
| paraphrase | none | +0.0661 [+0.0179, +0.1107] | 560 | 0.0140 |
| paraphrase | mean_row | +0.0589 [+0.0179, +0.0982] | 560 | 0.0120 |
| paraphrase | random_frame | +0.0196 [-0.0233, +0.0625] | 560 | 0.3918 |
| entailment | none | +0.1550 [+0.1200, +0.1933] | 600 | 0.0030 |
| entailment | mean_row | +0.1550 [+0.1233, +0.1883] | 600 | 0.0030 |
| entailment | random_frame | +0.1583 [+0.1250, +0.1933] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4937 | 0.4046 | 0.7231 |
| none | 0.3363 | 0.3340 | 0.5880 |
| mean_row | 0.3284 | 0.3370 | 0.5935 |
| random_frame | 0.3294 | 0.3499 | 0.5843 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1574 [+0.1378, +0.1773] | 1006 | 0.0030 |
| property | mean_row | +0.1653 [+0.1481, +0.1829] | 1006 | 0.0030 |
| property | random_frame | +0.1643 [+0.1448, +0.1832] | 1006 | 0.0030 |
| paraphrase | none | +0.0706 [+0.0378, +0.1044] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0676 [+0.0378, +0.0964] | 1006 | 0.0030 |
| paraphrase | random_frame | +0.0547 [+0.0249, +0.0865] | 1006 | 0.0030 |
| entailment | none | +0.1352 [+0.1102, +0.1639] | 1080 | 0.0030 |
| entailment | mean_row | +0.1296 [+0.1056, +0.1556] | 1080 | 0.0030 |
| entailment | random_frame | +0.1389 [+0.1148, +0.1630] | 1080 | 0.0030 |
