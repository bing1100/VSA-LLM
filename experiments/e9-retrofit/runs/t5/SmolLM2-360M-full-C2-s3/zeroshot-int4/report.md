# E9 track zero-shot (t5) — C2 seed 3 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3438 | 0.3557 | 0.5726 |
| none | 0.3459 | 0.3512 | 0.5786 |
| mean_row | 0.3472 | 0.3493 | 0.5744 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0021 [-0.0051, +0.0009] | 1566 | 0.1869 |
| property | mean_row | -0.0034 [-0.0066, -0.0004] | 1566 | 0.0480 |
| paraphrase | none | +0.0045 [-0.0032, +0.0115] | 1566 | 0.2889 |
| paraphrase | mean_row | +0.0064 [-0.0006, +0.0134] | 1566 | 0.1979 |
| entailment | none | -0.0060 [-0.0125, +0.0006] | 1680 | 0.1819 |
| entailment | mean_row | -0.0018 [-0.0077, +0.0036] | 1680 | 0.6557 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3357 | 0.3500 | 0.5617 |
| none | 0.3339 | 0.3446 | 0.5650 |
| mean_row | 0.3345 | 0.3429 | 0.5600 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0018 [-0.0024, +0.0065] | 560 | 0.9135 |
| property | mean_row | +0.0012 [-0.0036, +0.0060] | 560 | 0.9135 |
| paraphrase | none | +0.0054 [-0.0054, +0.0161] | 560 | 0.6297 |
| paraphrase | mean_row | +0.0071 [-0.0054, +0.0196] | 560 | 0.6297 |
| entailment | none | -0.0033 [-0.0150, +0.0083] | 600 | 1.0000 |
| entailment | mean_row | +0.0017 [-0.0067, +0.0100] | 600 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3482 | 0.3588 | 0.5787 |
| none | 0.3526 | 0.3549 | 0.5861 |
| mean_row | 0.3542 | 0.3529 | 0.5824 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0043 [-0.0083, -0.0003] | 1006 | 0.0350 |
| property | mean_row | -0.0060 [-0.0099, -0.0020] | 1006 | 0.0060 |
| paraphrase | none | +0.0040 [-0.0060, +0.0139] | 1006 | 0.5067 |
| paraphrase | mean_row | +0.0060 [-0.0030, +0.0159] | 1006 | 0.4998 |
| entailment | none | -0.0074 [-0.0158, +0.0009] | 1080 | 0.2099 |
| entailment | mean_row | -0.0037 [-0.0111, +0.0037] | 1080 | 0.4378 |
