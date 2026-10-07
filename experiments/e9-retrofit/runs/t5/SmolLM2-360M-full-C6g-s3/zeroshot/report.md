# E9 track zero-shot (t5) — C6g seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3863 | 0.3250 | 0.6107 |
| none | 0.3563 | 0.3346 | 0.5631 |
| mean_row | 0.3561 | 0.3359 | 0.5637 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0300 [+0.0189, +0.0411] | 1566 | 0.0020 |
| property | mean_row | +0.0302 [+0.0187, +0.0411] | 1566 | 0.0020 |
| paraphrase | none | -0.0096 [-0.0307, +0.0121] | 1566 | 0.6617 |
| paraphrase | mean_row | -0.0109 [-0.0319, +0.0102] | 1566 | 0.6617 |
| entailment | none | +0.0476 [+0.0304, +0.0655] | 1680 | 0.0020 |
| entailment | mean_row | +0.0470 [+0.0298, +0.0643] | 1680 | 0.0020 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3976 | 0.3321 | 0.6267 |
| none | 0.3476 | 0.3429 | 0.5700 |
| mean_row | 0.3524 | 0.3536 | 0.5717 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0500 [+0.0310, +0.0679] | 560 | 0.0020 |
| property | mean_row | +0.0452 [+0.0268, +0.0637] | 560 | 0.0020 |
| paraphrase | none | -0.0107 [-0.0500, +0.0250] | 560 | 0.6057 |
| paraphrase | mean_row | -0.0214 [-0.0607, +0.0143] | 560 | 0.5817 |
| entailment | none | +0.0567 [+0.0283, +0.0850] | 600 | 0.0020 |
| entailment | mean_row | +0.0550 [+0.0267, +0.0833] | 600 | 0.0020 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3801 | 0.3211 | 0.6019 |
| none | 0.3612 | 0.3300 | 0.5593 |
| mean_row | 0.3582 | 0.3260 | 0.5593 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0189 [+0.0050, +0.0328] | 1006 | 0.0110 |
| property | mean_row | +0.0219 [+0.0080, +0.0361] | 1006 | 0.0080 |
| paraphrase | none | -0.0089 [-0.0348, +0.0179] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0050 [-0.0318, +0.0209] | 1006 | 1.0000 |
| entailment | none | +0.0426 [+0.0204, +0.0657] | 1080 | 0.0040 |
| entailment | mean_row | +0.0426 [+0.0194, +0.0658] | 1080 | 0.0040 |
