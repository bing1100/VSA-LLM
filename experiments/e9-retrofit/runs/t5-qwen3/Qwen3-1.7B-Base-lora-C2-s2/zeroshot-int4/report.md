# E9 track zero-shot (t5) — C2 seed 2 (Qwen/Qwen3-1.7B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3189 | 0.2784 | 0.5506 |
| none | 0.3201 | 0.2752 | 0.5500 |
| mean_row | 0.3195 | 0.2765 | 0.5518 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0013 [-0.0040, +0.0015] | 1566 | 0.7796 |
| property | mean_row | -0.0006 [-0.0034, +0.0019] | 1566 | 0.7796 |
| paraphrase | none | +0.0032 [-0.0032, +0.0096] | 1566 | 0.7436 |
| paraphrase | mean_row | +0.0019 [-0.0045, +0.0083] | 1566 | 0.7436 |
| entailment | none | +0.0006 [-0.0054, +0.0065] | 1680 | 1.0000 |
| entailment | mean_row | -0.0012 [-0.0077, +0.0054] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3125 | 0.2929 | 0.5500 |
| none | 0.3113 | 0.2964 | 0.5517 |
| mean_row | 0.3119 | 0.2964 | 0.5550 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0012 [-0.0036, +0.0060] | 560 | 1.0000 |
| property | mean_row | +0.0006 [-0.0036, +0.0048] | 560 | 1.0000 |
| paraphrase | none | -0.0036 [-0.0143, +0.0054] | 560 | 1.0000 |
| paraphrase | mean_row | -0.0036 [-0.0125, +0.0054] | 560 | 1.0000 |
| entailment | none | -0.0017 [-0.0100, +0.0067] | 600 | 0.9635 |
| entailment | mean_row | -0.0050 [-0.0167, +0.0067] | 600 | 0.9635 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3224 | 0.2704 | 0.5509 |
| none | 0.3250 | 0.2634 | 0.5491 |
| mean_row | 0.3237 | 0.2654 | 0.5500 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0027 [-0.0063, +0.0010] | 1006 | 0.4018 |
| property | mean_row | -0.0013 [-0.0046, +0.0020] | 1006 | 0.4568 |
| paraphrase | none | +0.0070 [+0.0000, +0.0149] | 1006 | 0.1779 |
| paraphrase | mean_row | +0.0050 [-0.0030, +0.0129] | 1006 | 0.2919 |
| entailment | none | +0.0019 [-0.0065, +0.0102] | 1080 | 1.0000 |
| entailment | mean_row | +0.0009 [-0.0074, +0.0093] | 1080 | 1.0000 |
