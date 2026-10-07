# E9 track zero-shot (t5) — C5ut seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4706 | 0.3697 | 0.6810 |
| none | 0.3538 | 0.3314 | 0.5565 |
| mean_row | 0.3423 | 0.3461 | 0.5702 |
| random_frame | 0.3291 | 0.3397 | 0.5619 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1169 [+0.1005, +0.1330] | 1566 | 0.0030 |
| property | mean_row | +0.1284 [+0.1130, +0.1428] | 1566 | 0.0030 |
| property | random_frame | +0.1415 [+0.1256, +0.1560] | 1566 | 0.0030 |
| paraphrase | none | +0.0383 [+0.0109, +0.0645] | 1566 | 0.0270 |
| paraphrase | mean_row | +0.0236 [-0.0013, +0.0485] | 1566 | 0.0680 |
| paraphrase | random_frame | +0.0300 [+0.0038, +0.0562] | 1566 | 0.0500 |
| entailment | none | +0.1244 [+0.1018, +0.1470] | 1680 | 0.0030 |
| entailment | mean_row | +0.1107 [+0.0911, +0.1298] | 1680 | 0.0030 |
| entailment | random_frame | +0.1190 [+0.0994, +0.1381] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4690 | 0.3589 | 0.6867 |
| none | 0.3518 | 0.3357 | 0.5667 |
| mean_row | 0.3363 | 0.3411 | 0.5583 |
| random_frame | 0.3226 | 0.3393 | 0.5467 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1173 [+0.0887, +0.1470] | 560 | 0.0030 |
| property | mean_row | +0.1327 [+0.1083, +0.1601] | 560 | 0.0030 |
| property | random_frame | +0.1464 [+0.1214, +0.1744] | 560 | 0.0030 |
| paraphrase | none | +0.0232 [-0.0232, +0.0696] | 560 | 0.9205 |
| paraphrase | mean_row | +0.0179 [-0.0232, +0.0590] | 560 | 0.9205 |
| paraphrase | random_frame | +0.0196 [-0.0214, +0.0608] | 560 | 0.9205 |
| entailment | none | +0.1200 [+0.0817, +0.1600] | 600 | 0.0030 |
| entailment | mean_row | +0.1283 [+0.0950, +0.1617] | 600 | 0.0030 |
| entailment | random_frame | +0.1400 [+0.1050, +0.1767] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4715 | 0.3757 | 0.6778 |
| none | 0.3549 | 0.3290 | 0.5509 |
| mean_row | 0.3456 | 0.3489 | 0.5769 |
| random_frame | 0.3327 | 0.3400 | 0.5704 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1166 [+0.0971, +0.1372] | 1006 | 0.0030 |
| property | mean_row | +0.1259 [+0.1083, +0.1441] | 1006 | 0.0030 |
| property | random_frame | +0.1388 [+0.1186, +0.1581] | 1006 | 0.0030 |
| paraphrase | none | +0.0467 [+0.0119, +0.0795] | 1006 | 0.0270 |
| paraphrase | mean_row | +0.0268 [-0.0050, +0.0567] | 1006 | 0.1139 |
| paraphrase | random_frame | +0.0358 [+0.0040, +0.0686] | 1006 | 0.0700 |
| entailment | none | +0.1269 [+0.0981, +0.1546] | 1080 | 0.0030 |
| entailment | mean_row | +0.1009 [+0.0768, +0.1250] | 1080 | 0.0030 |
| entailment | random_frame | +0.1074 [+0.0843, +0.1306] | 1080 | 0.0030 |
