# E9 track zero-shot (t4) — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5631 | 0.4987 | 0.8026 |
| none | 0.5631 | 0.4966 | 0.8026 |
| mean_row | 0.5633 | 0.5020 | 0.8012 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0000 [-0.0034, +0.0031] | 1484 | 1.0000 |
| property | mean_row | -0.0002 [-0.0031, +0.0025] | 1484 | 1.0000 |
| paraphrase | none | +0.0020 [-0.0061, +0.0094] | 1484 | 0.7236 |
| paraphrase | mean_row | -0.0034 [-0.0101, +0.0034] | 1484 | 0.7236 |
| entailment | none | +0.0000 [-0.0047, +0.0047] | 1484 | 1.0000 |
| entailment | mean_row | +0.0013 [-0.0034, +0.0061] | 1484 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5452 | 0.4872 | 0.8071 |
| none | 0.5440 | 0.4838 | 0.8094 |
| mean_row | 0.5448 | 0.4916 | 0.8071 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0011 [-0.0030, +0.0048] | 897 | 1.0000 |
| property | mean_row | +0.0004 [-0.0033, +0.0037] | 897 | 1.0000 |
| paraphrase | none | +0.0033 [-0.0067, +0.0134] | 897 | 0.7396 |
| paraphrase | mean_row | -0.0045 [-0.0134, +0.0045] | 897 | 0.7396 |
| entailment | none | -0.0022 [-0.0089, +0.0034] | 897 | 1.0000 |
| entailment | mean_row | +0.0000 [-0.0067, +0.0067] | 897 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5906 | 0.5162 | 0.7956 |
| none | 0.5923 | 0.5162 | 0.7922 |
| mean_row | 0.5917 | 0.5179 | 0.7922 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0017 [-0.0074, +0.0040] | 587 | 1.0000 |
| property | mean_row | -0.0011 [-0.0057, +0.0034] | 587 | 1.0000 |
| paraphrase | none | +0.0000 [-0.0119, +0.0119] | 587 | 1.0000 |
| paraphrase | mean_row | -0.0017 [-0.0119, +0.0085] | 587 | 1.0000 |
| entailment | none | +0.0034 [-0.0017, +0.0102] | 587 | 0.5117 |
| entailment | mean_row | +0.0034 [+0.0000, +0.0085] | 587 | 0.5117 |
