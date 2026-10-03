# E9 track zero-shot (t5) — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4355 | 0.3576 | 0.6458 |
| none | 0.3280 | 0.3263 | 0.5601 |
| mean_row | 0.3142 | 0.3110 | 0.5643 |
| random_frame | 0.3025 | 0.3129 | 0.5470 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1075 [+0.0892, +0.1243] | 1566 | 0.0030 |
| property | mean_row | +0.1213 [+0.1069, +0.1352] | 1566 | 0.0030 |
| property | random_frame | +0.1330 [+0.1169, +0.1488] | 1566 | 0.0030 |
| paraphrase | none | +0.0313 [+0.0051, +0.0568] | 1566 | 0.0110 |
| paraphrase | mean_row | +0.0466 [+0.0243, +0.0696] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0447 [+0.0204, +0.0683] | 1566 | 0.0030 |
| entailment | none | +0.0857 [+0.0643, +0.1077] | 1680 | 0.0030 |
| entailment | mean_row | +0.0815 [+0.0631, +0.0988] | 1680 | 0.0030 |
| entailment | random_frame | +0.0988 [+0.0792, +0.1196] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4339 | 0.3589 | 0.6250 |
| none | 0.3238 | 0.3304 | 0.5500 |
| mean_row | 0.3107 | 0.2929 | 0.5267 |
| random_frame | 0.2982 | 0.3179 | 0.5183 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1101 [+0.0851, +0.1357] | 560 | 0.0030 |
| property | mean_row | +0.1232 [+0.1000, +0.1482] | 560 | 0.0030 |
| property | random_frame | +0.1357 [+0.1089, +0.1631] | 560 | 0.0030 |
| paraphrase | none | +0.0286 [-0.0161, +0.0714] | 560 | 0.2199 |
| paraphrase | mean_row | +0.0661 [+0.0250, +0.1089] | 560 | 0.0090 |
| paraphrase | random_frame | +0.0411 [+0.0018, +0.0839] | 560 | 0.0880 |
| entailment | none | +0.0750 [+0.0400, +0.1117] | 600 | 0.0030 |
| entailment | mean_row | +0.0983 [+0.0683, +0.1300] | 600 | 0.0030 |
| entailment | random_frame | +0.1067 [+0.0750, +0.1417] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4364 | 0.3569 | 0.6574 |
| none | 0.3304 | 0.3241 | 0.5657 |
| mean_row | 0.3161 | 0.3211 | 0.5852 |
| random_frame | 0.3048 | 0.3101 | 0.5630 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1060 [+0.0835, +0.1292] | 1006 | 0.0030 |
| property | mean_row | +0.1203 [+0.1030, +0.1385] | 1006 | 0.0030 |
| property | random_frame | +0.1315 [+0.1130, +0.1508] | 1006 | 0.0030 |
| paraphrase | none | +0.0328 [-0.0010, +0.0646] | 1006 | 0.0670 |
| paraphrase | mean_row | +0.0358 [+0.0050, +0.0646] | 1006 | 0.0620 |
| paraphrase | random_frame | +0.0467 [+0.0149, +0.0785] | 1006 | 0.0150 |
| entailment | none | +0.0917 [+0.0657, +0.1194] | 1080 | 0.0030 |
| entailment | mean_row | +0.0722 [+0.0509, +0.0935] | 1080 | 0.0030 |
| entailment | random_frame | +0.0944 [+0.0704, +0.1194] | 1080 | 0.0030 |
