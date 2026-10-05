# E9 track zero-shot (t5) — C5 seed 2 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4119 | 0.2842 | 0.5940 |
| none | 0.3186 | 0.2695 | 0.5202 |
| mean_row | 0.2997 | 0.2644 | 0.5304 |
| random_frame | 0.2976 | 0.2759 | 0.5196 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0932 [+0.0768, +0.1086] | 1566 | 0.0030 |
| property | mean_row | +0.1122 [+0.0988, +0.1254] | 1566 | 0.0030 |
| property | random_frame | +0.1143 [+0.1000, +0.1286] | 1566 | 0.0030 |
| paraphrase | none | +0.0147 [-0.0109, +0.0383] | 1566 | 0.5137 |
| paraphrase | mean_row | +0.0198 [-0.0019, +0.0409] | 1566 | 0.2429 |
| paraphrase | random_frame | +0.0083 [-0.0141, +0.0313] | 1566 | 0.5137 |
| entailment | none | +0.0738 [+0.0542, +0.0941] | 1680 | 0.0030 |
| entailment | mean_row | +0.0637 [+0.0470, +0.0815] | 1680 | 0.0030 |
| entailment | random_frame | +0.0744 [+0.0565, +0.0935] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4262 | 0.2804 | 0.5967 |
| none | 0.3256 | 0.2839 | 0.5167 |
| mean_row | 0.3018 | 0.2679 | 0.5317 |
| random_frame | 0.3060 | 0.2732 | 0.5017 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1006 [+0.0756, +0.1262] | 560 | 0.0030 |
| property | mean_row | +0.1244 [+0.1030, +0.1482] | 560 | 0.0030 |
| property | random_frame | +0.1202 [+0.0964, +0.1452] | 560 | 0.0030 |
| paraphrase | none | -0.0036 [-0.0447, +0.0375] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0125 [-0.0232, +0.0483] | 560 | 1.0000 |
| paraphrase | random_frame | +0.0071 [-0.0304, +0.0429] | 560 | 1.0000 |
| entailment | none | +0.0800 [+0.0467, +0.1117] | 600 | 0.0030 |
| entailment | mean_row | +0.0650 [+0.0367, +0.0933] | 600 | 0.0030 |
| entailment | random_frame | +0.0950 [+0.0633, +0.1267] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4039 | 0.2863 | 0.5926 |
| none | 0.3148 | 0.2614 | 0.5222 |
| mean_row | 0.2985 | 0.2624 | 0.5296 |
| random_frame | 0.2929 | 0.2773 | 0.5296 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0891 [+0.0699, +0.1097] | 1006 | 0.0030 |
| property | mean_row | +0.1054 [+0.0898, +0.1223] | 1006 | 0.0030 |
| property | random_frame | +0.1110 [+0.0938, +0.1296] | 1006 | 0.0030 |
| paraphrase | none | +0.0249 [-0.0040, +0.0567] | 1006 | 0.2369 |
| paraphrase | mean_row | +0.0239 [-0.0020, +0.0497] | 1006 | 0.2369 |
| paraphrase | random_frame | +0.0089 [-0.0189, +0.0368] | 1006 | 0.5547 |
| entailment | none | +0.0704 [+0.0435, +0.0945] | 1080 | 0.0030 |
| entailment | mean_row | +0.0630 [+0.0426, +0.0833] | 1080 | 0.0030 |
| entailment | random_frame | +0.0630 [+0.0398, +0.0852] | 1080 | 0.0030 |
