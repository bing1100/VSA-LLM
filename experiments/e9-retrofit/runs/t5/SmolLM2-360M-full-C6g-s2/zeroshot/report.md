# E9 track zero-shot (t5) — C6g seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3842 | 0.3142 | 0.6077 |
| none | 0.3525 | 0.3244 | 0.5619 |
| mean_row | 0.3567 | 0.3257 | 0.5625 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0317 [+0.0198, +0.0430] | 1566 | 0.0020 |
| property | mean_row | +0.0275 [+0.0155, +0.0385] | 1566 | 0.0020 |
| paraphrase | none | -0.0102 [-0.0307, +0.0109] | 1566 | 0.5597 |
| paraphrase | mean_row | -0.0115 [-0.0319, +0.0096] | 1566 | 0.5597 |
| entailment | none | +0.0458 [+0.0297, +0.0637] | 1680 | 0.0020 |
| entailment | mean_row | +0.0452 [+0.0286, +0.0625] | 1680 | 0.0020 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3863 | 0.3214 | 0.6050 |
| none | 0.3458 | 0.3429 | 0.5700 |
| mean_row | 0.3500 | 0.3393 | 0.5650 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0405 [+0.0202, +0.0601] | 560 | 0.0020 |
| property | mean_row | +0.0363 [+0.0161, +0.0554] | 560 | 0.0020 |
| paraphrase | none | -0.0214 [-0.0571, +0.0143] | 560 | 0.5217 |
| paraphrase | mean_row | -0.0179 [-0.0536, +0.0179] | 560 | 0.5217 |
| entailment | none | +0.0350 [+0.0083, +0.0617] | 600 | 0.0120 |
| entailment | mean_row | +0.0400 [+0.0133, +0.0667] | 600 | 0.0080 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3830 | 0.3101 | 0.6093 |
| none | 0.3562 | 0.3141 | 0.5574 |
| mean_row | 0.3605 | 0.3181 | 0.5611 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0268 [+0.0136, +0.0417] | 1006 | 0.0020 |
| property | mean_row | +0.0225 [+0.0089, +0.0378] | 1006 | 0.0050 |
| paraphrase | none | -0.0040 [-0.0298, +0.0219] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0080 [-0.0318, +0.0179] | 1006 | 1.0000 |
| entailment | none | +0.0519 [+0.0306, +0.0741] | 1080 | 0.0020 |
| entailment | mean_row | +0.0481 [+0.0250, +0.0713] | 1080 | 0.0020 |
