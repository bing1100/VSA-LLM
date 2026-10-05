# E9 track zero-shot (t5) — C2 seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3531 | 0.3199 | 0.5744 |
| none | 0.3542 | 0.3180 | 0.5690 |
| mean_row | 0.3506 | 0.3212 | 0.5702 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0011 [-0.0043, +0.0019] | 1566 | 0.5007 |
| property | mean_row | +0.0026 [-0.0000, +0.0055] | 1566 | 0.1579 |
| paraphrase | none | +0.0019 [-0.0051, +0.0089] | 1566 | 1.0000 |
| paraphrase | mean_row | -0.0013 [-0.0077, +0.0051] | 1566 | 1.0000 |
| entailment | none | +0.0054 [-0.0006, +0.0113] | 1680 | 0.1959 |
| entailment | mean_row | +0.0042 [-0.0006, +0.0095] | 1680 | 0.1959 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3536 | 0.3214 | 0.5900 |
| none | 0.3530 | 0.3179 | 0.5850 |
| mean_row | 0.3476 | 0.3214 | 0.5800 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0006 [-0.0048, +0.0060] | 560 | 0.7686 |
| property | mean_row | +0.0060 [+0.0018, +0.0107] | 560 | 0.0120 |
| paraphrase | none | +0.0036 [-0.0071, +0.0125] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0107, +0.0107] | 560 | 1.0000 |
| entailment | none | +0.0050 [-0.0067, +0.0167] | 600 | 0.4728 |
| entailment | mean_row | +0.0100 [+0.0000, +0.0200] | 600 | 0.1559 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3529 | 0.3191 | 0.5657 |
| none | 0.3549 | 0.3181 | 0.5602 |
| mean_row | 0.3522 | 0.3211 | 0.5648 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0020 [-0.0060, +0.0020] | 1006 | 0.6277 |
| property | mean_row | +0.0007 [-0.0033, +0.0043] | 1006 | 0.8096 |
| paraphrase | none | +0.0010 [-0.0089, +0.0109] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0020 [-0.0109, +0.0070] | 1006 | 1.0000 |
| entailment | none | +0.0056 [+0.0000, +0.0120] | 1080 | 0.1779 |
| entailment | mean_row | +0.0009 [-0.0046, +0.0065] | 1080 | 0.8656 |
