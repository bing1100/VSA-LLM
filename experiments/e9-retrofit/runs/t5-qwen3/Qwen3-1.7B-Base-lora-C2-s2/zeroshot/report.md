# E9 track zero-shot (t5) — C2 seed 2 (Qwen/Qwen3-1.7B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3110 | 0.3576 | 0.5798 |
| none | 0.3112 | 0.3602 | 0.5833 |
| mean_row | 0.3116 | 0.3633 | 0.5827 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0002 [-0.0032, +0.0028] | 1566 | 1.0000 |
| property | mean_row | -0.0006 [-0.0030, +0.0019] | 1566 | 1.0000 |
| paraphrase | none | -0.0026 [-0.0089, +0.0038] | 1566 | 0.4928 |
| paraphrase | mean_row | -0.0057 [-0.0121, +0.0006] | 1566 | 0.1959 |
| entailment | none | -0.0036 [-0.0107, +0.0030] | 1680 | 0.6637 |
| entailment | mean_row | -0.0030 [-0.0095, +0.0036] | 1680 | 0.6637 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.2952 | 0.3732 | 0.5767 |
| none | 0.2970 | 0.3786 | 0.5833 |
| mean_row | 0.2940 | 0.3839 | 0.5850 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0018 [-0.0071, +0.0036] | 560 | 1.0000 |
| property | mean_row | +0.0012 [-0.0024, +0.0054] | 560 | 1.0000 |
| paraphrase | none | -0.0054 [-0.0179, +0.0071] | 560 | 0.4788 |
| paraphrase | mean_row | -0.0107 [-0.0232, +0.0018] | 560 | 0.2779 |
| entailment | none | -0.0067 [-0.0183, +0.0050] | 600 | 0.4178 |
| entailment | mean_row | -0.0083 [-0.0217, +0.0033] | 600 | 0.4178 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3197 | 0.3489 | 0.5815 |
| none | 0.3191 | 0.3499 | 0.5833 |
| mean_row | 0.3214 | 0.3519 | 0.5815 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0007 [-0.0030, +0.0046] | 1006 | 0.7726 |
| property | mean_row | -0.0017 [-0.0046, +0.0013] | 1006 | 0.5757 |
| paraphrase | none | -0.0010 [-0.0080, +0.0060] | 1006 | 0.9695 |
| paraphrase | mean_row | -0.0030 [-0.0099, +0.0040] | 1006 | 0.9695 |
| entailment | none | -0.0019 [-0.0111, +0.0065] | 1080 | 1.0000 |
| entailment | mean_row | +0.0000 [-0.0074, +0.0074] | 1080 | 1.0000 |
