# E9 track zero-shot (t5) — C6m seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3442 | 0.3212 | 0.5702 |
| none | 0.3525 | 0.3167 | 0.5637 |
| mean_row | 0.3516 | 0.3218 | 0.5595 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0083 [-0.0170, +0.0004] | 1566 | 0.1279 |
| property | mean_row | -0.0074 [-0.0160, +0.0011] | 1566 | 0.1279 |
| paraphrase | none | +0.0045 [-0.0128, +0.0211] | 1566 | 1.0000 |
| paraphrase | mean_row | -0.0006 [-0.0179, +0.0160] | 1566 | 1.0000 |
| entailment | none | +0.0065 [-0.0065, +0.0208] | 1680 | 0.3628 |
| entailment | mean_row | +0.0107 [-0.0018, +0.0244] | 1680 | 0.2399 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3458 | 0.3214 | 0.5550 |
| none | 0.3464 | 0.3321 | 0.5683 |
| mean_row | 0.3458 | 0.3411 | 0.5667 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0006 [-0.0155, +0.0143] | 560 | 1.0000 |
| property | mean_row | -0.0000 [-0.0149, +0.0149] | 560 | 1.0000 |
| paraphrase | none | -0.0107 [-0.0411, +0.0196] | 560 | 0.5587 |
| paraphrase | mean_row | -0.0196 [-0.0500, +0.0089] | 560 | 0.4278 |
| entailment | none | -0.0133 [-0.0367, +0.0100] | 600 | 0.5817 |
| entailment | mean_row | -0.0117 [-0.0350, +0.0117] | 600 | 0.5817 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3433 | 0.3211 | 0.5787 |
| none | 0.3559 | 0.3082 | 0.5611 |
| mean_row | 0.3549 | 0.3111 | 0.5556 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0126 [-0.0225, -0.0030] | 1006 | 0.0300 |
| property | mean_row | -0.0116 [-0.0212, -0.0020] | 1006 | 0.0300 |
| paraphrase | none | +0.0129 [-0.0080, +0.0338] | 1006 | 0.4718 |
| paraphrase | mean_row | +0.0099 [-0.0109, +0.0308] | 1006 | 0.4718 |
| entailment | none | +0.0176 [-0.0019, +0.0361] | 1080 | 0.0860 |
| entailment | mean_row | +0.0231 [+0.0046, +0.0407] | 1080 | 0.0360 |
