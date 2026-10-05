# E9 track zero-shot (t5) — C5 seed 2 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4406 | 0.3672 | 0.6685 |
| none | 0.3316 | 0.3282 | 0.5601 |
| mean_row | 0.3133 | 0.3321 | 0.5673 |
| random_frame | 0.3078 | 0.3276 | 0.5500 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1090 [+0.0903, +0.1260] | 1566 | 0.0030 |
| property | mean_row | +0.1273 [+0.1128, +0.1422] | 1566 | 0.0030 |
| property | random_frame | +0.1328 [+0.1179, +0.1481] | 1566 | 0.0030 |
| paraphrase | none | +0.0390 [+0.0121, +0.0658] | 1566 | 0.0080 |
| paraphrase | mean_row | +0.0351 [+0.0115, +0.0600] | 1566 | 0.0080 |
| paraphrase | random_frame | +0.0396 [+0.0147, +0.0651] | 1566 | 0.0060 |
| entailment | none | +0.1083 [+0.0875, +0.1310] | 1680 | 0.0030 |
| entailment | mean_row | +0.1012 [+0.0833, +0.1202] | 1680 | 0.0030 |
| entailment | random_frame | +0.1185 [+0.0982, +0.1399] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4470 | 0.3607 | 0.6500 |
| none | 0.3310 | 0.3286 | 0.5383 |
| mean_row | 0.3095 | 0.3107 | 0.5217 |
| random_frame | 0.2917 | 0.3214 | 0.5267 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1161 [+0.0893, +0.1440] | 560 | 0.0030 |
| property | mean_row | +0.1375 [+0.1137, +0.1625] | 560 | 0.0030 |
| property | random_frame | +0.1554 [+0.1292, +0.1810] | 560 | 0.0030 |
| paraphrase | none | +0.0321 [-0.0125, +0.0768] | 560 | 0.2119 |
| paraphrase | mean_row | +0.0500 [+0.0089, +0.0911] | 560 | 0.0750 |
| paraphrase | random_frame | +0.0393 [-0.0071, +0.0839] | 560 | 0.2119 |
| entailment | none | +0.1117 [+0.0767, +0.1483] | 600 | 0.0030 |
| entailment | mean_row | +0.1283 [+0.0983, +0.1583] | 600 | 0.0030 |
| entailment | random_frame | +0.1233 [+0.0883, +0.1583] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4370 | 0.3708 | 0.6787 |
| none | 0.3320 | 0.3280 | 0.5722 |
| mean_row | 0.3154 | 0.3439 | 0.5926 |
| random_frame | 0.3168 | 0.3310 | 0.5630 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1050 [+0.0828, +0.1266] | 1006 | 0.0030 |
| property | mean_row | +0.1216 [+0.1040, +0.1402] | 1006 | 0.0030 |
| property | random_frame | +0.1203 [+0.1007, +0.1385] | 1006 | 0.0030 |
| paraphrase | none | +0.0427 [+0.0080, +0.0775] | 1006 | 0.0390 |
| paraphrase | mean_row | +0.0268 [-0.0030, +0.0567] | 1006 | 0.0900 |
| paraphrase | random_frame | +0.0398 [+0.0089, +0.0716] | 1006 | 0.0390 |
| entailment | none | +0.1065 [+0.0796, +0.1343] | 1080 | 0.0030 |
| entailment | mean_row | +0.0861 [+0.0620, +0.1093] | 1080 | 0.0030 |
| entailment | random_frame | +0.1157 [+0.0907, +0.1417] | 1080 | 0.0030 |
