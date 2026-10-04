# E9 track zero-shot (t5) — C2 seed 1 (Qwen/Qwen3-1.7B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3172 | 0.3499 | 0.5827 |
| none | 0.3178 | 0.3480 | 0.5786 |
| mean_row | 0.3172 | 0.3487 | 0.5792 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0006 [-0.0034, +0.0021] | 1566 | 1.0000 |
| property | mean_row | -0.0000 [-0.0026, +0.0026] | 1566 | 1.0000 |
| paraphrase | none | +0.0019 [-0.0051, +0.0089] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0013 [-0.0045, +0.0070] | 1566 | 1.0000 |
| entailment | none | +0.0042 [-0.0030, +0.0113] | 1680 | 0.5597 |
| entailment | mean_row | +0.0036 [-0.0042, +0.0113] | 1680 | 0.5597 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3125 | 0.3589 | 0.5617 |
| none | 0.3149 | 0.3500 | 0.5650 |
| mean_row | 0.3107 | 0.3571 | 0.5600 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0024 [-0.0071, +0.0024] | 560 | 0.6577 |
| property | mean_row | +0.0018 [-0.0024, +0.0060] | 560 | 0.6577 |
| paraphrase | none | +0.0089 [-0.0018, +0.0196] | 560 | 0.3038 |
| paraphrase | mean_row | +0.0018 [-0.0089, +0.0125] | 560 | 0.8446 |
| entailment | none | -0.0033 [-0.0150, +0.0083] | 600 | 1.0000 |
| entailment | mean_row | +0.0017 [-0.0133, +0.0167] | 600 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3197 | 0.3449 | 0.5944 |
| none | 0.3194 | 0.3469 | 0.5861 |
| mean_row | 0.3207 | 0.3439 | 0.5898 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0003 [-0.0030, +0.0036] | 1006 | 1.0000 |
| property | mean_row | -0.0010 [-0.0043, +0.0023] | 1006 | 1.0000 |
| paraphrase | none | -0.0020 [-0.0109, +0.0070] | 1006 | 1.0000 |
| paraphrase | mean_row | +0.0010 [-0.0060, +0.0080] | 1006 | 1.0000 |
| entailment | none | +0.0083 [-0.0009, +0.0185] | 1080 | 0.1519 |
| entailment | mean_row | +0.0046 [-0.0046, +0.0130] | 1080 | 0.3638 |
