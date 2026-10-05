# E9 track zero-shot (t5) — C5 seed 3 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4500 | 0.3493 | 0.6494 |
| none | 0.3455 | 0.3295 | 0.5702 |
| mean_row | 0.3348 | 0.3346 | 0.5595 |
| random_frame | 0.3278 | 0.3333 | 0.5470 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1045 [+0.0873, +0.1211] | 1566 | 0.0030 |
| property | mean_row | +0.1152 [+0.1005, +0.1301] | 1566 | 0.0030 |
| property | random_frame | +0.1222 [+0.1068, +0.1381] | 1566 | 0.0030 |
| paraphrase | none | +0.0198 [-0.0070, +0.0466] | 1566 | 0.4708 |
| paraphrase | mean_row | +0.0147 [-0.0089, +0.0377] | 1566 | 0.4758 |
| paraphrase | random_frame | +0.0160 [-0.0096, +0.0421] | 1566 | 0.4758 |
| entailment | none | +0.0792 [+0.0583, +0.1006] | 1680 | 0.0030 |
| entailment | mean_row | +0.0899 [+0.0714, +0.1083] | 1680 | 0.0030 |
| entailment | random_frame | +0.1024 [+0.0827, +0.1226] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4613 | 0.3321 | 0.6367 |
| none | 0.3381 | 0.3286 | 0.5400 |
| mean_row | 0.3381 | 0.3232 | 0.5600 |
| random_frame | 0.3143 | 0.3071 | 0.5483 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1232 [+0.0952, +0.1512] | 560 | 0.0030 |
| property | mean_row | +0.1232 [+0.1006, +0.1476] | 560 | 0.0030 |
| property | random_frame | +0.1470 [+0.1220, +0.1720] | 560 | 0.0030 |
| paraphrase | none | +0.0036 [-0.0411, +0.0465] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0089 [-0.0304, +0.0482] | 560 | 1.0000 |
| paraphrase | random_frame | +0.0250 [-0.0161, +0.0679] | 560 | 0.8516 |
| entailment | none | +0.0967 [+0.0617, +0.1317] | 600 | 0.0030 |
| entailment | mean_row | +0.0767 [+0.0483, +0.1083] | 600 | 0.0030 |
| entailment | random_frame | +0.0883 [+0.0533, +0.1233] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4437 | 0.3588 | 0.6565 |
| none | 0.3496 | 0.3300 | 0.5870 |
| mean_row | 0.3330 | 0.3410 | 0.5593 |
| random_frame | 0.3353 | 0.3479 | 0.5463 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0941 [+0.0745, +0.1143] | 1006 | 0.0030 |
| property | mean_row | +0.1107 [+0.0931, +0.1286] | 1006 | 0.0030 |
| property | random_frame | +0.1083 [+0.0901, +0.1266] | 1006 | 0.0030 |
| paraphrase | none | +0.0288 [-0.0070, +0.0626] | 1006 | 0.3358 |
| paraphrase | mean_row | +0.0179 [-0.0139, +0.0467] | 1006 | 0.5697 |
| paraphrase | random_frame | +0.0109 [-0.0219, +0.0427] | 1006 | 0.5697 |
| entailment | none | +0.0694 [+0.0426, +0.0963] | 1080 | 0.0030 |
| entailment | mean_row | +0.0972 [+0.0750, +0.1204] | 1080 | 0.0030 |
| entailment | random_frame | +0.1102 [+0.0879, +0.1333] | 1080 | 0.0030 |
