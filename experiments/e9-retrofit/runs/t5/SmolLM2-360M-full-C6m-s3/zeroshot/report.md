# E9 track zero-shot (t5) — C6m seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3463 | 0.3263 | 0.5690 |
| none | 0.3540 | 0.3091 | 0.5601 |
| mean_row | 0.3538 | 0.3161 | 0.5583 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0077 [-0.0170, +0.0013] | 1566 | 0.1859 |
| property | mean_row | -0.0074 [-0.0168, +0.0011] | 1566 | 0.1859 |
| paraphrase | none | +0.0172 [-0.0007, +0.0364] | 1566 | 0.1299 |
| paraphrase | mean_row | +0.0102 [-0.0077, +0.0294] | 1566 | 0.2709 |
| entailment | none | +0.0089 [-0.0048, +0.0232] | 1680 | 0.2619 |
| entailment | mean_row | +0.0107 [-0.0030, +0.0244] | 1680 | 0.2619 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3440 | 0.3232 | 0.5750 |
| none | 0.3494 | 0.3286 | 0.5633 |
| mean_row | 0.3500 | 0.3339 | 0.5650 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0054 [-0.0196, +0.0089] | 560 | 0.9155 |
| property | mean_row | -0.0060 [-0.0208, +0.0089] | 560 | 0.9155 |
| paraphrase | none | -0.0054 [-0.0375, +0.0268] | 560 | 1.0000 |
| paraphrase | mean_row | -0.0107 [-0.0429, +0.0196] | 560 | 1.0000 |
| entailment | none | +0.0117 [-0.0117, +0.0350] | 600 | 0.7556 |
| entailment | mean_row | +0.0100 [-0.0117, +0.0333] | 600 | 0.7556 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3476 | 0.3280 | 0.5657 |
| none | 0.3565 | 0.2982 | 0.5583 |
| mean_row | 0.3559 | 0.3062 | 0.5546 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0089 [-0.0189, +0.0020] | 1006 | 0.2159 |
| property | mean_row | -0.0083 [-0.0182, +0.0023] | 1006 | 0.2159 |
| paraphrase | none | +0.0298 [+0.0079, +0.0527] | 1006 | 0.0220 |
| paraphrase | mean_row | +0.0219 [-0.0010, +0.0448] | 1006 | 0.0650 |
| entailment | none | +0.0074 [-0.0111, +0.0250] | 1080 | 0.4998 |
| entailment | mean_row | +0.0111 [-0.0065, +0.0278] | 1080 | 0.4998 |
