# E9 track zero-shot (t5) — C2 seed 1 (Qwen/Qwen3-0.6B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3089 | 0.2548 | 0.5304 |
| none | 0.3069 | 0.2554 | 0.5315 |
| mean_row | 0.3069 | 0.2554 | 0.5339 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0019 [-0.0015, +0.0053] | 1566 | 0.5857 |
| property | mean_row | +0.0019 [-0.0015, +0.0051] | 1566 | 0.5857 |
| paraphrase | none | -0.0006 [-0.0083, +0.0070] | 1566 | 1.0000 |
| paraphrase | mean_row | -0.0006 [-0.0083, +0.0070] | 1566 | 1.0000 |
| entailment | none | -0.0012 [-0.0065, +0.0042] | 1680 | 0.7236 |
| entailment | mean_row | -0.0036 [-0.0095, +0.0018] | 1680 | 0.5137 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.2946 | 0.2357 | 0.5383 |
| none | 0.2851 | 0.2393 | 0.5350 |
| mean_row | 0.2875 | 0.2393 | 0.5417 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0095 [+0.0036, +0.0155] | 560 | 0.0080 |
| property | mean_row | +0.0071 [+0.0018, +0.0131] | 560 | 0.0110 |
| paraphrase | none | -0.0036 [-0.0179, +0.0107] | 560 | 1.0000 |
| paraphrase | mean_row | -0.0036 [-0.0179, +0.0107] | 560 | 1.0000 |
| entailment | none | +0.0033 [-0.0050, +0.0117] | 600 | 1.0000 |
| entailment | mean_row | -0.0033 [-0.0133, +0.0050] | 600 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3168 | 0.2654 | 0.5259 |
| none | 0.3191 | 0.2644 | 0.5296 |
| mean_row | 0.3178 | 0.2644 | 0.5296 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0023 [-0.0066, +0.0020] | 1006 | 0.5977 |
| property | mean_row | -0.0010 [-0.0053, +0.0033] | 1006 | 0.6327 |
| paraphrase | none | +0.0010 [-0.0070, +0.0099] | 1006 | 1.0000 |
| paraphrase | mean_row | +0.0010 [-0.0089, +0.0099] | 1006 | 1.0000 |
| entailment | none | -0.0037 [-0.0102, +0.0028] | 1080 | 0.5637 |
| entailment | mean_row | -0.0037 [-0.0111, +0.0028] | 1080 | 0.5637 |
