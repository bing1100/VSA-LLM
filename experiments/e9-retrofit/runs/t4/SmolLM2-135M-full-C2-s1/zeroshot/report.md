# E9 track zero-shot (t4) — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5672 | 0.6947 | 0.7446 |
| none | 0.5730 | 0.7069 | 0.7466 |
| mean_row | 0.5685 | 0.6995 | 0.7460 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0058 [-0.0092, -0.0027] | 1484 | 0.0020 |
| property | mean_row | -0.0013 [-0.0047, +0.0020] | 1484 | 0.4478 |
| paraphrase | none | -0.0121 [-0.0209, -0.0040] | 1484 | 0.0120 |
| paraphrase | mean_row | -0.0047 [-0.0135, +0.0040] | 1484 | 0.3208 |
| entailment | none | -0.0020 [-0.0074, +0.0027] | 1484 | 0.9955 |
| entailment | mean_row | -0.0013 [-0.0061, +0.0034] | 1484 | 0.9955 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5741 | 0.7146 | 0.7436 |
| none | 0.5793 | 0.7246 | 0.7447 |
| mean_row | 0.5738 | 0.7146 | 0.7480 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0052 [-0.0093, -0.0015] | 897 | 0.0120 |
| property | mean_row | +0.0004 [-0.0033, +0.0041] | 897 | 0.8196 |
| paraphrase | none | -0.0100 [-0.0212, +0.0000] | 897 | 0.1379 |
| paraphrase | mean_row | +0.0000 [-0.0101, +0.0111] | 897 | 1.0000 |
| entailment | none | -0.0011 [-0.0078, +0.0056] | 897 | 0.8806 |
| entailment | mean_row | -0.0045 [-0.0111, +0.0011] | 897 | 0.4058 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5565 | 0.6644 | 0.7462 |
| none | 0.5633 | 0.6797 | 0.7496 |
| mean_row | 0.5605 | 0.6763 | 0.7428 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0068 [-0.0131, -0.0011] | 587 | 0.0440 |
| property | mean_row | -0.0040 [-0.0108, +0.0023] | 587 | 0.2159 |
| paraphrase | none | -0.0153 [-0.0290, -0.0017] | 587 | 0.0700 |
| paraphrase | mean_row | -0.0119 [-0.0273, +0.0034] | 587 | 0.1439 |
| entailment | none | -0.0034 [-0.0102, +0.0034] | 587 | 0.9355 |
| entailment | mean_row | +0.0034 [-0.0034, +0.0119] | 587 | 0.9355 |
