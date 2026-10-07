# E9 track zero-shot (t5) — C6g seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3782 | 0.3206 | 0.5988 |
| none | 0.3536 | 0.3167 | 0.5619 |
| mean_row | 0.3527 | 0.3212 | 0.5625 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0247 [+0.0132, +0.0364] | 1566 | 0.0020 |
| property | mean_row | +0.0255 [+0.0143, +0.0370] | 1566 | 0.0020 |
| paraphrase | none | +0.0038 [-0.0179, +0.0243] | 1566 | 1.0000 |
| paraphrase | mean_row | -0.0006 [-0.0230, +0.0217] | 1566 | 1.0000 |
| entailment | none | +0.0369 [+0.0196, +0.0548] | 1680 | 0.0020 |
| entailment | mean_row | +0.0363 [+0.0196, +0.0542] | 1680 | 0.0020 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3887 | 0.3411 | 0.6083 |
| none | 0.3488 | 0.3250 | 0.5750 |
| mean_row | 0.3470 | 0.3357 | 0.5783 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0399 [+0.0208, +0.0595] | 560 | 0.0020 |
| property | mean_row | +0.0417 [+0.0226, +0.0607] | 560 | 0.0020 |
| paraphrase | none | +0.0161 [-0.0196, +0.0500] | 560 | 0.8616 |
| paraphrase | mean_row | +0.0054 [-0.0321, +0.0393] | 560 | 0.8616 |
| entailment | none | +0.0333 [+0.0083, +0.0600] | 600 | 0.0160 |
| entailment | mean_row | +0.0300 [+0.0050, +0.0567] | 600 | 0.0160 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3724 | 0.3091 | 0.5935 |
| none | 0.3562 | 0.3121 | 0.5546 |
| mean_row | 0.3559 | 0.3131 | 0.5537 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0162 [+0.0030, +0.0295] | 1006 | 0.0360 |
| property | mean_row | +0.0166 [+0.0033, +0.0302] | 1006 | 0.0360 |
| paraphrase | none | -0.0030 [-0.0288, +0.0239] | 1006 | 1.0000 |
| paraphrase | mean_row | -0.0040 [-0.0298, +0.0229] | 1006 | 1.0000 |
| entailment | none | +0.0389 [+0.0157, +0.0620] | 1080 | 0.0040 |
| entailment | mean_row | +0.0398 [+0.0167, +0.0630] | 1080 | 0.0040 |
