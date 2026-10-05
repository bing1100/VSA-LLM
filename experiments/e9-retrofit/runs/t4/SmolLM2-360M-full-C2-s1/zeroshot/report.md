# E9 track zero-shot (t4) — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.6220 | 0.6024 | 0.7918 |
| none | 0.6213 | 0.5977 | 0.7978 |
| mean_row | 0.6224 | 0.6011 | 0.7972 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0007 [-0.0018, +0.0031] | 1484 | 1.0000 |
| property | mean_row | -0.0004 [-0.0029, +0.0018] | 1484 | 1.0000 |
| paraphrase | none | +0.0047 [-0.0020, +0.0115] | 1484 | 0.4158 |
| paraphrase | mean_row | +0.0013 [-0.0054, +0.0081] | 1484 | 0.7856 |
| entailment | none | -0.0061 [-0.0115, -0.0013] | 1484 | 0.0220 |
| entailment | mean_row | -0.0054 [-0.0094, -0.0013] | 1484 | 0.0220 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.6232 | 0.6299 | 0.7938 |
| none | 0.6239 | 0.6243 | 0.8016 |
| mean_row | 0.6232 | 0.6276 | 0.8004 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0007 [-0.0045, +0.0026] | 897 | 1.0000 |
| property | mean_row | +0.0000 [-0.0030, +0.0030] | 897 | 1.0000 |
| paraphrase | none | +0.0056 [-0.0033, +0.0156] | 897 | 0.6317 |
| paraphrase | mean_row | +0.0022 [-0.0078, +0.0112] | 897 | 0.7276 |
| entailment | none | -0.0078 [-0.0156, +0.0000] | 897 | 0.0800 |
| entailment | mean_row | -0.0067 [-0.0123, -0.0011] | 897 | 0.0800 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.6201 | 0.5605 | 0.7888 |
| none | 0.6173 | 0.5571 | 0.7922 |
| mean_row | 0.6212 | 0.5605 | 0.7922 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0028 [-0.0006, +0.0062] | 587 | 0.2459 |
| property | mean_row | -0.0011 [-0.0051, +0.0028] | 587 | 0.6767 |
| paraphrase | none | +0.0034 [-0.0052, +0.0136] | 587 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0085, +0.0085] | 587 | 1.0000 |
| entailment | none | -0.0034 [-0.0085, +0.0000] | 587 | 0.5677 |
| entailment | mean_row | -0.0034 [-0.0085, +0.0000] | 587 | 0.5677 |
