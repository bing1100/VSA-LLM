# E9 track zero-shot (t4) — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4930 | 0.5303 | 0.7123 |
| none | 0.4921 | 0.5276 | 0.7170 |
| mean_row | 0.4942 | 0.5310 | 0.7156 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0009 [-0.0022, +0.0040] | 1484 | 0.9235 |
| property | mean_row | -0.0011 [-0.0043, +0.0020] | 1484 | 0.9235 |
| paraphrase | none | +0.0027 [-0.0047, +0.0101] | 1484 | 1.0000 |
| paraphrase | mean_row | -0.0007 [-0.0094, +0.0074] | 1484 | 1.0000 |
| entailment | none | -0.0047 [-0.0101, +0.0007] | 1484 | 0.1879 |
| entailment | mean_row | -0.0034 [-0.0088, +0.0013] | 1484 | 0.2209 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4757 | 0.5318 | 0.7101 |
| none | 0.4727 | 0.5307 | 0.7135 |
| mean_row | 0.4757 | 0.5329 | 0.7146 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0030 [-0.0011, +0.0071] | 897 | 0.2879 |
| property | mean_row | -0.0000 [-0.0037, +0.0037] | 897 | 0.9965 |
| paraphrase | none | +0.0011 [-0.0089, +0.0100] | 897 | 1.0000 |
| paraphrase | mean_row | -0.0011 [-0.0111, +0.0089] | 897 | 1.0000 |
| entailment | none | -0.0033 [-0.0100, +0.0033] | 897 | 0.6437 |
| entailment | mean_row | -0.0045 [-0.0123, +0.0033] | 897 | 0.6437 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5196 | 0.5281 | 0.7155 |
| none | 0.5219 | 0.5230 | 0.7223 |
| mean_row | 0.5224 | 0.5281 | 0.7172 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0023 [-0.0074, +0.0028] | 587 | 0.5677 |
| property | mean_row | -0.0028 [-0.0080, +0.0023] | 587 | 0.5677 |
| paraphrase | none | +0.0051 [-0.0068, +0.0171] | 587 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0153, +0.0136] | 587 | 1.0000 |
| entailment | none | -0.0068 [-0.0153, +0.0000] | 587 | 0.2159 |
| entailment | mean_row | -0.0017 [-0.0069, +0.0034] | 587 | 0.7896 |
