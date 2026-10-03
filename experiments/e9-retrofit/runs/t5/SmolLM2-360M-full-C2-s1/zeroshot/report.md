# E9 track zero-shot (t5) — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3525 | 0.3250 | 0.5708 |
| none | 0.3506 | 0.3269 | 0.5661 |
| mean_row | 0.3514 | 0.3250 | 0.5667 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0019 [-0.0013, +0.0051] | 1566 | 0.5077 |
| property | mean_row | +0.0011 [-0.0021, +0.0043] | 1566 | 0.5617 |
| paraphrase | none | -0.0019 [-0.0102, +0.0064] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0000 [-0.0083, +0.0083] | 1566 | 1.0000 |
| entailment | none | +0.0048 [-0.0012, +0.0107] | 1680 | 0.2499 |
| entailment | mean_row | +0.0042 [-0.0018, +0.0107] | 1680 | 0.2499 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3470 | 0.3375 | 0.5783 |
| none | 0.3476 | 0.3339 | 0.5683 |
| mean_row | 0.3446 | 0.3357 | 0.5700 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0006 [-0.0048, +0.0036] | 560 | 0.7976 |
| property | mean_row | +0.0024 [-0.0030, +0.0083] | 560 | 0.7976 |
| paraphrase | none | +0.0036 [-0.0107, +0.0179] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0018 [-0.0125, +0.0179] | 560 | 1.0000 |
| entailment | none | +0.0100 [-0.0017, +0.0217] | 600 | 0.2239 |
| entailment | mean_row | +0.0083 [-0.0017, +0.0200] | 600 | 0.2239 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3555 | 0.3181 | 0.5667 |
| none | 0.3522 | 0.3231 | 0.5648 |
| mean_row | 0.3552 | 0.3191 | 0.5648 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0033 [-0.0010, +0.0076] | 1006 | 0.2619 |
| property | mean_row | +0.0003 [-0.0033, +0.0046] | 1006 | 0.8756 |
| paraphrase | none | -0.0050 [-0.0159, +0.0050] | 1006 | 0.7316 |
| paraphrase | mean_row | -0.0010 [-0.0099, +0.0080] | 1006 | 0.8906 |
| entailment | none | +0.0019 [-0.0046, +0.0083] | 1080 | 1.0000 |
| entailment | mean_row | +0.0019 [-0.0056, +0.0093] | 1080 | 1.0000 |
