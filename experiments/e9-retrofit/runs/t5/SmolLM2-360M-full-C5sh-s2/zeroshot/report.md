# E9 track zero-shot (t5) — C5sh seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3408 | 0.3225 | 0.5732 |
| none | 0.3489 | 0.3250 | 0.5619 |
| mean_row | 0.3463 | 0.3295 | 0.5708 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0081 [-0.0209, +0.0036] | 1566 | 0.3478 |
| property | mean_row | -0.0055 [-0.0136, +0.0026] | 1566 | 0.3478 |
| paraphrase | none | -0.0026 [-0.0262, +0.0204] | 1566 | 0.9795 |
| paraphrase | mean_row | -0.0070 [-0.0249, +0.0102] | 1566 | 0.9795 |
| entailment | none | +0.0113 [-0.0077, +0.0292] | 1680 | 0.4838 |
| entailment | mean_row | +0.0024 [-0.0119, +0.0167] | 1680 | 0.7976 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3440 | 0.3286 | 0.5850 |
| none | 0.3500 | 0.3304 | 0.5650 |
| mean_row | 0.3405 | 0.3464 | 0.5750 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0060 [-0.0286, +0.0161] | 560 | 1.0000 |
| property | mean_row | +0.0036 [-0.0113, +0.0185] | 560 | 1.0000 |
| paraphrase | none | -0.0018 [-0.0411, +0.0375] | 560 | 0.9885 |
| paraphrase | mean_row | -0.0179 [-0.0464, +0.0107] | 560 | 0.4938 |
| entailment | none | +0.0200 [-0.0117, +0.0517] | 600 | 0.4658 |
| entailment | mean_row | +0.0100 [-0.0133, +0.0334] | 600 | 0.4658 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3390 | 0.3191 | 0.5667 |
| none | 0.3482 | 0.3221 | 0.5602 |
| mean_row | 0.3496 | 0.3201 | 0.5685 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0093 [-0.0239, +0.0056] | 1006 | 0.2109 |
| property | mean_row | -0.0106 [-0.0206, -0.0010] | 1006 | 0.0700 |
| paraphrase | none | -0.0030 [-0.0308, +0.0249] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0010 [-0.0229, +0.0219] | 1006 | 1.0000 |
| entailment | none | +0.0065 [-0.0167, +0.0269] | 1080 | 1.0000 |
| entailment | mean_row | -0.0019 [-0.0185, +0.0139] | 1080 | 1.0000 |
