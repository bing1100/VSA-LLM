# E9 track zero-shot (t5) — C2 seed 2 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3163 | 0.2510 | 0.5238 |
| none | 0.3161 | 0.2548 | 0.5202 |
| mean_row | 0.3163 | 0.2478 | 0.5179 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0002 [-0.0038, +0.0040] | 1566 | 1.0000 |
| property | mean_row | +0.0000 [-0.0032, +0.0032] | 1566 | 1.0000 |
| paraphrase | none | -0.0038 [-0.0121, +0.0038] | 1566 | 0.7916 |
| paraphrase | mean_row | +0.0032 [-0.0038, +0.0109] | 1566 | 0.7916 |
| entailment | none | +0.0036 [-0.0030, +0.0101] | 1680 | 0.3598 |
| entailment | mean_row | +0.0060 [+0.0000, +0.0119] | 1680 | 0.1159 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3238 | 0.2607 | 0.5417 |
| none | 0.3214 | 0.2625 | 0.5333 |
| mean_row | 0.3214 | 0.2500 | 0.5350 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0024 [-0.0030, +0.0077] | 560 | 0.5757 |
| property | mean_row | +0.0024 [-0.0024, +0.0071] | 560 | 0.5757 |
| paraphrase | none | -0.0018 [-0.0143, +0.0108] | 560 | 0.9315 |
| paraphrase | mean_row | +0.0107 [+0.0000, +0.0214] | 560 | 0.1519 |
| entailment | none | +0.0083 [-0.0033, +0.0200] | 600 | 0.3338 |
| entailment | mean_row | +0.0067 [-0.0017, +0.0167] | 600 | 0.3338 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3121 | 0.2455 | 0.5139 |
| none | 0.3131 | 0.2505 | 0.5130 |
| mean_row | 0.3135 | 0.2465 | 0.5083 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0010 [-0.0060, +0.0040] | 1006 | 1.0000 |
| property | mean_row | -0.0013 [-0.0056, +0.0027] | 1006 | 1.0000 |
| paraphrase | none | -0.0050 [-0.0159, +0.0050] | 1006 | 0.7456 |
| paraphrase | mean_row | -0.0010 [-0.0109, +0.0080] | 1006 | 0.9045 |
| entailment | none | +0.0009 [-0.0083, +0.0093] | 1080 | 0.9565 |
| entailment | mean_row | +0.0056 [-0.0028, +0.0130] | 1080 | 0.4098 |
