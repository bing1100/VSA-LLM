# E9 track zero-shot (t5) — C2 seed 2 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3484 | 0.3538 | 0.5744 |
| none | 0.3472 | 0.3582 | 0.5744 |
| mean_row | 0.3487 | 0.3538 | 0.5750 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0013 [-0.0019, +0.0047] | 1566 | 0.9595 |
| property | mean_row | -0.0002 [-0.0032, +0.0028] | 1566 | 0.9595 |
| paraphrase | none | -0.0045 [-0.0128, +0.0038] | 1566 | 0.6817 |
| paraphrase | mean_row | +0.0000 [-0.0070, +0.0070] | 1566 | 1.0000 |
| entailment | none | +0.0000 [-0.0060, +0.0060] | 1680 | 1.0000 |
| entailment | mean_row | -0.0006 [-0.0065, +0.0054] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3351 | 0.3375 | 0.5550 |
| none | 0.3393 | 0.3446 | 0.5667 |
| mean_row | 0.3369 | 0.3500 | 0.5633 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0042 [-0.0101, +0.0012] | 560 | 0.2259 |
| property | mean_row | -0.0018 [-0.0071, +0.0036] | 560 | 0.5037 |
| paraphrase | none | -0.0071 [-0.0196, +0.0054] | 560 | 0.3138 |
| paraphrase | mean_row | -0.0125 [-0.0232, -0.0018] | 560 | 0.0620 |
| entailment | none | -0.0117 [-0.0217, -0.0033] | 600 | 0.0020 |
| entailment | mean_row | -0.0083 [-0.0183, +0.0017] | 600 | 0.1709 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3559 | 0.3628 | 0.5852 |
| none | 0.3516 | 0.3658 | 0.5787 |
| mean_row | 0.3552 | 0.3559 | 0.5815 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0043 [+0.0007, +0.0083] | 1006 | 0.0580 |
| property | mean_row | +0.0007 [-0.0030, +0.0043] | 1006 | 0.6607 |
| paraphrase | none | -0.0030 [-0.0139, +0.0080] | 1006 | 0.6387 |
| paraphrase | mean_row | +0.0070 [-0.0020, +0.0159] | 1006 | 0.3698 |
| entailment | none | +0.0065 [-0.0009, +0.0148] | 1080 | 0.2259 |
| entailment | mean_row | +0.0037 [-0.0028, +0.0111] | 1080 | 0.3368 |
