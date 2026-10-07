# E9 track zero-shot (t5) — C5rf seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4779 | 0.3838 | 0.6810 |
| none | 0.3504 | 0.3174 | 0.5619 |
| mean_row | 0.3465 | 0.3480 | 0.5667 |
| random_frame | 0.3274 | 0.3461 | 0.5583 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1275 [+0.1107, +0.1445] | 1566 | 0.0030 |
| property | mean_row | +0.1313 [+0.1156, +0.1462] | 1566 | 0.0030 |
| property | random_frame | +0.1505 [+0.1343, +0.1669] | 1566 | 0.0030 |
| paraphrase | none | +0.0664 [+0.0383, +0.0939] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0358 [+0.0108, +0.0607] | 1566 | 0.0180 |
| paraphrase | random_frame | +0.0377 [+0.0089, +0.0645] | 1566 | 0.0180 |
| entailment | none | +0.1190 [+0.0970, +0.1411] | 1680 | 0.0030 |
| entailment | mean_row | +0.1143 [+0.0952, +0.1339] | 1680 | 0.0030 |
| entailment | random_frame | +0.1226 [+0.1024, +0.1435] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4768 | 0.3714 | 0.6917 |
| none | 0.3446 | 0.3286 | 0.5667 |
| mean_row | 0.3417 | 0.3304 | 0.5583 |
| random_frame | 0.3077 | 0.3536 | 0.5567 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1321 [+0.1042, +0.1595] | 560 | 0.0030 |
| property | mean_row | +0.1351 [+0.1107, +0.1607] | 560 | 0.0030 |
| property | random_frame | +0.1690 [+0.1429, +0.1953] | 560 | 0.0030 |
| paraphrase | none | +0.0429 [+0.0000, +0.0893] | 560 | 0.1589 |
| paraphrase | mean_row | +0.0411 [+0.0000, +0.0857] | 560 | 0.1589 |
| paraphrase | random_frame | +0.0179 [-0.0268, +0.0643] | 560 | 0.4448 |
| entailment | none | +0.1250 [+0.0867, +0.1633] | 600 | 0.0030 |
| entailment | mean_row | +0.1333 [+0.1017, +0.1667] | 600 | 0.0030 |
| entailment | random_frame | +0.1350 [+0.1000, +0.1684] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4785 | 0.3907 | 0.6750 |
| none | 0.3535 | 0.3111 | 0.5593 |
| mean_row | 0.3492 | 0.3579 | 0.5713 |
| random_frame | 0.3383 | 0.3419 | 0.5593 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1249 [+0.1057, +0.1445] | 1006 | 0.0030 |
| property | mean_row | +0.1292 [+0.1107, +0.1474] | 1006 | 0.0030 |
| property | random_frame | +0.1402 [+0.1209, +0.1594] | 1006 | 0.0030 |
| paraphrase | none | +0.0795 [+0.0437, +0.1133] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0328 [+0.0030, +0.0616] | 1006 | 0.0350 |
| paraphrase | random_frame | +0.0487 [+0.0159, +0.0795] | 1006 | 0.0040 |
| entailment | none | +0.1157 [+0.0889, +0.1444] | 1080 | 0.0030 |
| entailment | mean_row | +0.1037 [+0.0806, +0.1287] | 1080 | 0.0030 |
| entailment | random_frame | +0.1157 [+0.0898, +0.1417] | 1080 | 0.0030 |
