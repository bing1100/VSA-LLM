# E9 track zero-shot (t5) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4604 | 0.3653 | 0.6708 |
| none | 0.3499 | 0.3180 | 0.5595 |
| mean_row | 0.3438 | 0.3461 | 0.5613 |
| random_frame | 0.3255 | 0.3461 | 0.5631 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1105 [+0.0943, +0.1258] | 1566 | 0.0030 |
| property | mean_row | +0.1166 [+0.1017, +0.1309] | 1566 | 0.0030 |
| property | random_frame | +0.1350 [+0.1196, +0.1509] | 1566 | 0.0030 |
| paraphrase | none | +0.0473 [+0.0185, +0.0741] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0192 [-0.0064, +0.0434] | 1566 | 0.3218 |
| paraphrase | random_frame | +0.0192 [-0.0083, +0.0453] | 1566 | 0.3218 |
| entailment | none | +0.1113 [+0.0899, +0.1327] | 1680 | 0.0030 |
| entailment | mean_row | +0.1095 [+0.0905, +0.1274] | 1680 | 0.0030 |
| entailment | random_frame | +0.1077 [+0.0887, +0.1262] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4655 | 0.3500 | 0.6633 |
| none | 0.3476 | 0.3214 | 0.5617 |
| mean_row | 0.3345 | 0.3339 | 0.5617 |
| random_frame | 0.3077 | 0.3339 | 0.5533 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1179 [+0.0899, +0.1458] | 560 | 0.0030 |
| property | mean_row | +0.1310 [+0.1071, +0.1565] | 560 | 0.0030 |
| property | random_frame | +0.1577 [+0.1315, +0.1833] | 560 | 0.0030 |
| paraphrase | none | +0.0286 [-0.0196, +0.0750] | 560 | 0.6657 |
| paraphrase | mean_row | +0.0161 [-0.0268, +0.0571] | 560 | 0.9135 |
| paraphrase | random_frame | +0.0161 [-0.0286, +0.0607] | 560 | 0.9135 |
| entailment | none | +0.1017 [+0.0650, +0.1383] | 600 | 0.0030 |
| entailment | mean_row | +0.1017 [+0.0700, +0.1350] | 600 | 0.0030 |
| entailment | random_frame | +0.1100 [+0.0733, +0.1450] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4576 | 0.3738 | 0.6750 |
| none | 0.3512 | 0.3161 | 0.5583 |
| mean_row | 0.3489 | 0.3529 | 0.5611 |
| random_frame | 0.3353 | 0.3529 | 0.5685 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1064 [+0.0875, +0.1246] | 1006 | 0.0030 |
| property | mean_row | +0.1087 [+0.0918, +0.1266] | 1006 | 0.0030 |
| property | random_frame | +0.1223 [+0.1037, +0.1405] | 1006 | 0.0030 |
| paraphrase | none | +0.0577 [+0.0229, +0.0895] | 1006 | 0.0060 |
| paraphrase | mean_row | +0.0209 [-0.0109, +0.0507] | 1006 | 0.4518 |
| paraphrase | random_frame | +0.0209 [-0.0119, +0.0517] | 1006 | 0.4518 |
| entailment | none | +0.1167 [+0.0898, +0.1426] | 1080 | 0.0030 |
| entailment | mean_row | +0.1139 [+0.0926, +0.1370] | 1080 | 0.0030 |
| entailment | random_frame | +0.1065 [+0.0833, +0.1296] | 1080 | 0.0030 |
