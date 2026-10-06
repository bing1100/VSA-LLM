# E9 track zero-shot (t5) — C5 seed 2 (Qwen/Qwen3-0.6B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4195 | 0.2701 | 0.6798 |
| none | 0.3216 | 0.2714 | 0.5399 |
| mean_row | 0.2940 | 0.2656 | 0.5595 |
| random_frame | 0.2903 | 0.2739 | 0.5345 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0979 [+0.0817, +0.1135] | 1566 | 0.0030 |
| property | mean_row | +0.1256 [+0.1115, +0.1396] | 1566 | 0.0030 |
| property | random_frame | +0.1292 [+0.1152, +0.1443] | 1566 | 0.0030 |
| paraphrase | none | -0.0013 [-0.0275, +0.0255] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0045 [-0.0172, +0.0268] | 1566 | 1.0000 |
| paraphrase | random_frame | -0.0038 [-0.0281, +0.0205] | 1566 | 1.0000 |
| entailment | none | +0.1399 [+0.1167, +0.1619] | 1680 | 0.0030 |
| entailment | mean_row | +0.1202 [+0.1006, +0.1393] | 1680 | 0.0030 |
| entailment | random_frame | +0.1452 [+0.1244, +0.1667] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4161 | 0.2625 | 0.6683 |
| none | 0.3054 | 0.2518 | 0.5483 |
| mean_row | 0.2810 | 0.2482 | 0.5567 |
| random_frame | 0.2726 | 0.2536 | 0.5333 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1107 [+0.0827, +0.1387] | 560 | 0.0030 |
| property | mean_row | +0.1351 [+0.1113, +0.1589] | 560 | 0.0030 |
| property | random_frame | +0.1435 [+0.1173, +0.1685] | 560 | 0.0030 |
| paraphrase | none | +0.0107 [-0.0339, +0.0589] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0143 [-0.0232, +0.0518] | 560 | 1.0000 |
| paraphrase | random_frame | +0.0089 [-0.0268, +0.0482] | 560 | 1.0000 |
| entailment | none | +0.1200 [+0.0817, +0.1583] | 600 | 0.0030 |
| entailment | mean_row | +0.1117 [+0.0783, +0.1450] | 600 | 0.0030 |
| entailment | random_frame | +0.1350 [+0.1000, +0.1700] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4215 | 0.2744 | 0.6861 |
| none | 0.3307 | 0.2823 | 0.5352 |
| mean_row | 0.3012 | 0.2753 | 0.5611 |
| random_frame | 0.3002 | 0.2853 | 0.5352 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0908 [+0.0706, +0.1113] | 1006 | 0.0030 |
| property | mean_row | +0.1203 [+0.1027, +0.1378] | 1006 | 0.0030 |
| property | random_frame | +0.1213 [+0.1030, +0.1392] | 1006 | 0.0030 |
| paraphrase | none | -0.0080 [-0.0408, +0.0239] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0010 [-0.0298, +0.0258] | 1006 | 1.0000 |
| paraphrase | random_frame | -0.0109 [-0.0417, +0.0189] | 1006 | 1.0000 |
| entailment | none | +0.1509 [+0.1231, +0.1806] | 1080 | 0.0030 |
| entailment | mean_row | +0.1250 [+0.1009, +0.1491] | 1080 | 0.0030 |
| entailment | random_frame | +0.1509 [+0.1250, +0.1778] | 1080 | 0.0030 |
