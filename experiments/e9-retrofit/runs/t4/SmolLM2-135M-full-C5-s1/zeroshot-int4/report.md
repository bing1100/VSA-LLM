# E9 track zero-shot (t4) — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5094 | 0.5593 | 0.7150 |
| none | 0.5043 | 0.5539 | 0.7170 |
| mean_row | 0.5022 | 0.5593 | 0.7129 |
| random_frame | 0.4946 | 0.5539 | 0.7102 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0052 [-0.0020, +0.0128] | 1484 | 0.1649 |
| property | mean_row | +0.0072 [+0.0007, +0.0139] | 1484 | 0.0660 |
| property | random_frame | +0.0148 [+0.0072, +0.0231] | 1484 | 0.0030 |
| paraphrase | none | +0.0054 [-0.0115, +0.0222] | 1484 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0162, +0.0162] | 1484 | 1.0000 |
| paraphrase | random_frame | +0.0054 [-0.0115, +0.0222] | 1484 | 1.0000 |
| entailment | none | -0.0020 [-0.0121, +0.0081] | 1484 | 1.0000 |
| entailment | mean_row | +0.0020 [-0.0067, +0.0115] | 1484 | 1.0000 |
| entailment | random_frame | +0.0047 [-0.0061, +0.0155] | 1484 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4935 | 0.5552 | 0.7124 |
| none | 0.4890 | 0.5552 | 0.7090 |
| mean_row | 0.4894 | 0.5641 | 0.7113 |
| random_frame | 0.4790 | 0.5608 | 0.7124 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0045 [-0.0045, +0.0137] | 897 | 0.6577 |
| property | mean_row | +0.0041 [-0.0034, +0.0115] | 897 | 0.6577 |
| property | random_frame | +0.0145 [+0.0048, +0.0238] | 897 | 0.0120 |
| paraphrase | none | +0.0000 [-0.0212, +0.0201] | 897 | 1.0000 |
| paraphrase | mean_row | -0.0089 [-0.0268, +0.0100] | 897 | 1.0000 |
| paraphrase | random_frame | -0.0056 [-0.0256, +0.0156] | 897 | 1.0000 |
| entailment | none | +0.0033 [-0.0089, +0.0156] | 897 | 1.0000 |
| entailment | mean_row | +0.0011 [-0.0100, +0.0112] | 897 | 1.0000 |
| entailment | random_frame | +0.0000 [-0.0112, +0.0111] | 897 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5338 | 0.5656 | 0.7189 |
| none | 0.5275 | 0.5520 | 0.7291 |
| mean_row | 0.5219 | 0.5520 | 0.7155 |
| random_frame | 0.5185 | 0.5434 | 0.7070 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0062 [-0.0062, +0.0199] | 587 | 0.3358 |
| property | mean_row | +0.0119 [-0.0000, +0.0244] | 587 | 0.1119 |
| property | random_frame | +0.0153 [+0.0034, +0.0284] | 587 | 0.0360 |
| paraphrase | none | +0.0136 [-0.0136, +0.0392] | 587 | 0.7296 |
| paraphrase | mean_row | +0.0136 [-0.0136, +0.0409] | 587 | 0.7296 |
| paraphrase | random_frame | +0.0221 [-0.0068, +0.0511] | 587 | 0.4198 |
| entailment | none | -0.0102 [-0.0273, +0.0068] | 587 | 0.7706 |
| entailment | mean_row | +0.0034 [-0.0119, +0.0187] | 587 | 0.7706 |
| entailment | random_frame | +0.0119 [-0.0068, +0.0341] | 587 | 0.7706 |
