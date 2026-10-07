# E9 track zero-shot (t5) — C5sh seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3399 | 0.3116 | 0.5530 |
| none | 0.3516 | 0.3129 | 0.5583 |
| mean_row | 0.3480 | 0.3327 | 0.5679 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0117 [-0.0236, +0.0000] | 1566 | 0.0860 |
| property | mean_row | -0.0081 [-0.0164, -0.0002] | 1566 | 0.0860 |
| paraphrase | none | -0.0013 [-0.0236, +0.0211] | 1566 | 0.9475 |
| paraphrase | mean_row | -0.0211 [-0.0396, -0.0032] | 1566 | 0.0420 |
| entailment | none | -0.0054 [-0.0244, +0.0143] | 1680 | 0.5887 |
| entailment | mean_row | -0.0149 [-0.0286, -0.0012] | 1680 | 0.0860 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3369 | 0.3161 | 0.5533 |
| none | 0.3488 | 0.3232 | 0.5650 |
| mean_row | 0.3417 | 0.3536 | 0.5700 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0119 [-0.0315, +0.0089] | 560 | 0.4918 |
| property | mean_row | -0.0048 [-0.0185, +0.0077] | 560 | 0.4918 |
| paraphrase | none | -0.0071 [-0.0429, +0.0286] | 560 | 0.7496 |
| paraphrase | mean_row | -0.0375 [-0.0679, -0.0089] | 560 | 0.0340 |
| entailment | none | -0.0117 [-0.0433, +0.0200] | 600 | 0.5027 |
| entailment | mean_row | -0.0167 [-0.0400, +0.0067] | 600 | 0.3858 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3416 | 0.3091 | 0.5528 |
| none | 0.3532 | 0.3072 | 0.5546 |
| mean_row | 0.3516 | 0.3211 | 0.5667 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0116 [-0.0255, +0.0027] | 1006 | 0.1119 |
| property | mean_row | -0.0099 [-0.0202, +0.0003] | 1006 | 0.1119 |
| paraphrase | none | +0.0020 [-0.0269, +0.0308] | 1006 | 0.9065 |
| paraphrase | mean_row | -0.0119 [-0.0338, +0.0109] | 1006 | 0.6857 |
| entailment | none | -0.0019 [-0.0250, +0.0194] | 1080 | 0.9235 |
| entailment | mean_row | -0.0139 [-0.0315, +0.0028] | 1080 | 0.2399 |
