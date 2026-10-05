# E9 track zero-shot (t4) — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5743 | 0.7204 | 0.7534 |
| none | 0.5701 | 0.7109 | 0.7433 |
| mean_row | 0.5665 | 0.7116 | 0.7433 |
| random_frame | 0.5624 | 0.7116 | 0.7487 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0043 [-0.0013, +0.0099] | 1484 | 0.1239 |
| property | mean_row | +0.0079 [+0.0022, +0.0135] | 1484 | 0.0040 |
| property | random_frame | +0.0119 [+0.0054, +0.0186] | 1484 | 0.0030 |
| paraphrase | none | +0.0094 [-0.0040, +0.0236] | 1484 | 0.5697 |
| paraphrase | mean_row | +0.0088 [-0.0054, +0.0222] | 1484 | 0.5697 |
| paraphrase | random_frame | +0.0088 [-0.0068, +0.0236] | 1484 | 0.5697 |
| entailment | none | +0.0101 [+0.0020, +0.0189] | 1484 | 0.0840 |
| entailment | mean_row | +0.0101 [+0.0013, +0.0189] | 1484 | 0.0840 |
| entailment | random_frame | +0.0047 [-0.0047, +0.0142] | 1484 | 0.3498 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5782 | 0.7492 | 0.7525 |
| none | 0.5767 | 0.7336 | 0.7436 |
| mean_row | 0.5756 | 0.7324 | 0.7503 |
| random_frame | 0.5719 | 0.7414 | 0.7514 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0015 [-0.0048, +0.0082] | 897 | 0.6897 |
| property | mean_row | +0.0026 [-0.0030, +0.0085] | 897 | 0.6897 |
| property | random_frame | +0.0063 [-0.0004, +0.0134] | 897 | 0.2129 |
| paraphrase | none | +0.0156 [-0.0011, +0.0312] | 897 | 0.1499 |
| paraphrase | mean_row | +0.0167 [+0.0011, +0.0334] | 897 | 0.1319 |
| paraphrase | random_frame | +0.0078 [-0.0089, +0.0256] | 897 | 0.4248 |
| entailment | none | +0.0089 [-0.0011, +0.0190] | 897 | 0.2849 |
| entailment | mean_row | +0.0022 [-0.0067, +0.0111] | 897 | 1.0000 |
| entailment | random_frame | +0.0011 [-0.0089, +0.0111] | 897 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5684 | 0.6763 | 0.7547 |
| none | 0.5599 | 0.6763 | 0.7428 |
| mean_row | 0.5525 | 0.6797 | 0.7325 |
| random_frame | 0.5480 | 0.6661 | 0.7445 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0085 [-0.0017, +0.0187] | 587 | 0.0960 |
| property | mean_row | +0.0159 [+0.0051, +0.0267] | 587 | 0.0140 |
| property | random_frame | +0.0204 [+0.0080, +0.0341] | 587 | 0.0030 |
| paraphrase | none | +0.0000 [-0.0239, +0.0221] | 587 | 1.0000 |
| paraphrase | mean_row | -0.0034 [-0.0290, +0.0204] | 587 | 1.0000 |
| paraphrase | random_frame | +0.0102 [-0.0204, +0.0392] | 587 | 1.0000 |
| entailment | none | +0.0119 [-0.0017, +0.0273] | 587 | 0.2839 |
| entailment | mean_row | +0.0221 [+0.0051, +0.0409] | 587 | 0.0330 |
| entailment | random_frame | +0.0102 [-0.0069, +0.0290] | 587 | 0.2879 |
