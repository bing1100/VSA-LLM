# E9 track zero-shot (t5) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4434 | 0.3787 | 0.6310 |
| none | 0.3484 | 0.3474 | 0.5673 |
| mean_row | 0.3310 | 0.3442 | 0.5577 |
| random_frame | 0.3218 | 0.3410 | 0.5452 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0949 [+0.0788, +0.1109] | 1566 | 0.0030 |
| property | mean_row | +0.1124 [+0.0981, +0.1262] | 1566 | 0.0030 |
| property | random_frame | +0.1215 [+0.1056, +0.1369] | 1566 | 0.0030 |
| paraphrase | none | +0.0313 [+0.0032, +0.0587] | 1566 | 0.0310 |
| paraphrase | mean_row | +0.0345 [+0.0089, +0.0575] | 1566 | 0.0210 |
| paraphrase | random_frame | +0.0377 [+0.0128, +0.0632] | 1566 | 0.0210 |
| entailment | none | +0.0637 [+0.0428, +0.0857] | 1680 | 0.0030 |
| entailment | mean_row | +0.0732 [+0.0553, +0.0923] | 1680 | 0.0030 |
| entailment | random_frame | +0.0857 [+0.0673, +0.1060] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4583 | 0.3643 | 0.6100 |
| none | 0.3435 | 0.3393 | 0.5467 |
| mean_row | 0.3405 | 0.3357 | 0.5600 |
| random_frame | 0.3131 | 0.3286 | 0.5350 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1149 [+0.0881, +0.1411] | 560 | 0.0030 |
| property | mean_row | +0.1179 [+0.0964, +0.1393] | 560 | 0.0030 |
| property | random_frame | +0.1452 [+0.1202, +0.1708] | 560 | 0.0030 |
| paraphrase | none | +0.0250 [-0.0214, +0.0750] | 560 | 0.2909 |
| paraphrase | mean_row | +0.0286 [-0.0071, +0.0679] | 560 | 0.2759 |
| paraphrase | random_frame | +0.0357 [-0.0054, +0.0750] | 560 | 0.2519 |
| entailment | none | +0.0633 [+0.0283, +0.0983] | 600 | 0.0030 |
| entailment | mean_row | +0.0500 [+0.0200, +0.0783] | 600 | 0.0030 |
| entailment | random_frame | +0.0750 [+0.0433, +0.1067] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4351 | 0.3867 | 0.6426 |
| none | 0.3512 | 0.3519 | 0.5787 |
| mean_row | 0.3257 | 0.3489 | 0.5565 |
| random_frame | 0.3267 | 0.3479 | 0.5509 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0838 [+0.0639, +0.1037] | 1006 | 0.0030 |
| property | mean_row | +0.1093 [+0.0921, +0.1282] | 1006 | 0.0030 |
| property | random_frame | +0.1083 [+0.0891, +0.1269] | 1006 | 0.0030 |
| paraphrase | none | +0.0348 [-0.0010, +0.0686] | 1006 | 0.0690 |
| paraphrase | mean_row | +0.0378 [+0.0089, +0.0676] | 1006 | 0.0510 |
| paraphrase | random_frame | +0.0388 [+0.0060, +0.0726] | 1006 | 0.0510 |
| entailment | none | +0.0639 [+0.0380, +0.0898] | 1080 | 0.0030 |
| entailment | mean_row | +0.0861 [+0.0630, +0.1102] | 1080 | 0.0030 |
| entailment | random_frame | +0.0917 [+0.0676, +0.1157] | 1080 | 0.0030 |
