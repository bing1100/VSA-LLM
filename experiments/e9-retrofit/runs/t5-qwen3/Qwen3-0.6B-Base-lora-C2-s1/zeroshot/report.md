# E9 track zero-shot (t5) — C2 seed 1 (Qwen/Qwen3-0.6B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3031 | 0.2765 | 0.5196 |
| none | 0.3031 | 0.2816 | 0.5214 |
| mean_row | 0.3037 | 0.2842 | 0.5196 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0000 [-0.0030, +0.0030] | 1566 | 1.0000 |
| property | mean_row | -0.0006 [-0.0038, +0.0026] | 1566 | 1.0000 |
| paraphrase | none | -0.0051 [-0.0121, +0.0019] | 1566 | 0.1849 |
| paraphrase | mean_row | -0.0077 [-0.0153, -0.0006] | 1566 | 0.0600 |
| entailment | none | -0.0018 [-0.0083, +0.0048] | 1680 | 1.0000 |
| entailment | mean_row | +0.0000 [-0.0060, +0.0060] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3024 | 0.2607 | 0.5367 |
| none | 0.3042 | 0.2607 | 0.5367 |
| mean_row | 0.3054 | 0.2661 | 0.5367 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0018 [-0.0065, +0.0024] | 560 | 0.6757 |
| property | mean_row | -0.0030 [-0.0089, +0.0030] | 560 | 0.6757 |
| paraphrase | none | +0.0000 [-0.0089, +0.0107] | 560 | 1.0000 |
| paraphrase | mean_row | -0.0054 [-0.0161, +0.0054] | 560 | 0.8036 |
| entailment | none | +0.0000 [-0.0117, +0.0133] | 600 | 1.0000 |
| entailment | mean_row | +0.0000 [-0.0117, +0.0117] | 600 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3035 | 0.2853 | 0.5102 |
| none | 0.3025 | 0.2932 | 0.5130 |
| mean_row | 0.3028 | 0.2942 | 0.5102 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0010 [-0.0030, +0.0050] | 1006 | 1.0000 |
| property | mean_row | +0.0007 [-0.0033, +0.0046] | 1006 | 1.0000 |
| paraphrase | none | -0.0080 [-0.0179, +0.0010] | 1006 | 0.1519 |
| paraphrase | mean_row | -0.0089 [-0.0189, +0.0000] | 1006 | 0.1519 |
| entailment | none | -0.0028 [-0.0102, +0.0056] | 1080 | 1.0000 |
| entailment | mean_row | +0.0000 [-0.0065, +0.0065] | 1080 | 1.0000 |
