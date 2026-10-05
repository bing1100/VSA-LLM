# E9 track zero-shot (t4) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.6204 | 0.5896 | 0.7884 |
| none | 0.6242 | 0.5997 | 0.7918 |
| mean_row | 0.6202 | 0.6024 | 0.7938 |
| random_frame | 0.6166 | 0.6024 | 0.7844 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0038 [-0.0088, +0.0011] | 1484 | 0.3808 |
| property | mean_row | +0.0002 [-0.0040, +0.0045] | 1484 | 0.9665 |
| property | random_frame | +0.0038 [-0.0018, +0.0092] | 1484 | 0.3858 |
| paraphrase | none | -0.0101 [-0.0222, +0.0020] | 1484 | 0.1089 |
| paraphrase | mean_row | -0.0128 [-0.0236, -0.0020] | 1484 | 0.0870 |
| paraphrase | random_frame | -0.0128 [-0.0249, -0.0007] | 1484 | 0.0960 |
| entailment | none | -0.0034 [-0.0108, +0.0040] | 1484 | 0.8016 |
| entailment | mean_row | -0.0054 [-0.0135, +0.0020] | 1484 | 0.6567 |
| entailment | random_frame | +0.0040 [-0.0047, +0.0135] | 1484 | 0.8016 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.6224 | 0.6221 | 0.7871 |
| none | 0.6236 | 0.6276 | 0.7960 |
| mean_row | 0.6213 | 0.6265 | 0.8004 |
| random_frame | 0.6198 | 0.6343 | 0.7926 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0011 [-0.0063, +0.0041] | 897 | 1.0000 |
| property | mean_row | +0.0011 [-0.0037, +0.0063] | 897 | 1.0000 |
| property | random_frame | +0.0026 [-0.0033, +0.0086] | 897 | 1.0000 |
| paraphrase | none | -0.0056 [-0.0201, +0.0089] | 897 | 0.9355 |
| paraphrase | mean_row | -0.0045 [-0.0190, +0.0089] | 897 | 0.9355 |
| paraphrase | random_frame | -0.0123 [-0.0279, +0.0033] | 897 | 0.4288 |
| entailment | none | -0.0089 [-0.0190, +0.0000] | 897 | 0.1719 |
| entailment | mean_row | -0.0134 [-0.0234, -0.0033] | 897 | 0.0300 |
| entailment | random_frame | -0.0056 [-0.0156, +0.0045] | 897 | 0.2879 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.6173 | 0.5400 | 0.7905 |
| none | 0.6252 | 0.5571 | 0.7853 |
| mean_row | 0.6184 | 0.5656 | 0.7836 |
| random_frame | 0.6116 | 0.5537 | 0.7717 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0080 [-0.0176, +0.0011] | 587 | 0.2639 |
| property | mean_row | -0.0011 [-0.0091, +0.0062] | 587 | 0.7236 |
| property | random_frame | +0.0057 [-0.0051, +0.0165] | 587 | 0.6297 |
| paraphrase | none | -0.0170 [-0.0375, +0.0034] | 587 | 0.2239 |
| paraphrase | mean_row | -0.0256 [-0.0460, -0.0085] | 587 | 0.0270 |
| paraphrase | random_frame | -0.0136 [-0.0358, +0.0068] | 587 | 0.2239 |
| entailment | none | +0.0051 [-0.0068, +0.0170] | 587 | 0.7236 |
| entailment | mean_row | +0.0068 [-0.0068, +0.0187] | 587 | 0.7236 |
| entailment | random_frame | +0.0187 [+0.0017, +0.0358] | 587 | 0.1139 |
