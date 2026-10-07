# E9 track zero-shot (t5) — C6m seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3467 | 0.3123 | 0.5649 |
| none | 0.3523 | 0.3123 | 0.5542 |
| mean_row | 0.3529 | 0.3110 | 0.5571 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0055 [-0.0140, +0.0028] | 1566 | 0.3078 |
| property | mean_row | -0.0062 [-0.0147, +0.0021] | 1566 | 0.3078 |
| paraphrase | none | +0.0000 [-0.0166, +0.0179] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0013 [-0.0166, +0.0192] | 1566 | 1.0000 |
| entailment | none | +0.0107 [-0.0036, +0.0250] | 1680 | 0.2859 |
| entailment | mean_row | +0.0077 [-0.0065, +0.0220] | 1680 | 0.2859 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3494 | 0.3089 | 0.5633 |
| none | 0.3500 | 0.3304 | 0.5533 |
| mean_row | 0.3506 | 0.3286 | 0.5567 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0006 [-0.0149, +0.0149] | 560 | 1.0000 |
| property | mean_row | -0.0012 [-0.0155, +0.0137] | 560 | 1.0000 |
| paraphrase | none | -0.0214 [-0.0518, +0.0071] | 560 | 0.3378 |
| paraphrase | mean_row | -0.0196 [-0.0518, +0.0107] | 560 | 0.3378 |
| entailment | none | +0.0100 [-0.0133, +0.0333] | 600 | 0.8496 |
| entailment | mean_row | +0.0067 [-0.0150, +0.0300] | 600 | 0.8496 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3453 | 0.3141 | 0.5657 |
| none | 0.3535 | 0.3022 | 0.5546 |
| mean_row | 0.3542 | 0.3012 | 0.5574 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0083 [-0.0186, +0.0017] | 1006 | 0.1599 |
| property | mean_row | -0.0089 [-0.0192, +0.0010] | 1006 | 0.1599 |
| paraphrase | none | +0.0119 [-0.0109, +0.0358] | 1006 | 0.5437 |
| paraphrase | mean_row | +0.0129 [-0.0089, +0.0358] | 1006 | 0.5437 |
| entailment | none | +0.0111 [-0.0065, +0.0269] | 1080 | 0.4478 |
| entailment | mean_row | +0.0083 [-0.0093, +0.0250] | 1080 | 0.4478 |
