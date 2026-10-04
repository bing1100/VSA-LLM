# E9 track zero-shot (t5) — C2 seed 1 (Qwen/Qwen3-1.7B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3335 | 0.2765 | 0.5470 |
| none | 0.3327 | 0.2771 | 0.5470 |
| mean_row | 0.3321 | 0.2746 | 0.5452 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0009 [-0.0021, +0.0038] | 1566 | 0.5817 |
| property | mean_row | +0.0015 [-0.0009, +0.0040] | 1566 | 0.4478 |
| paraphrase | none | -0.0006 [-0.0077, +0.0064] | 1566 | 1.0000 |
| paraphrase | mean_row | +0.0019 [-0.0045, +0.0083] | 1566 | 1.0000 |
| entailment | none | +0.0000 [-0.0054, +0.0054] | 1680 | 1.0000 |
| entailment | mean_row | +0.0018 [-0.0036, +0.0077] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3250 | 0.2964 | 0.5700 |
| none | 0.3250 | 0.3000 | 0.5700 |
| mean_row | 0.3232 | 0.2929 | 0.5650 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0000 [-0.0048, +0.0048] | 560 | 0.9925 |
| property | mean_row | +0.0018 [-0.0024, +0.0060] | 560 | 0.8096 |
| paraphrase | none | -0.0036 [-0.0143, +0.0071] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0036 [-0.0089, +0.0161] | 560 | 1.0000 |
| entailment | none | +0.0000 [-0.0083, +0.0100] | 600 | 1.0000 |
| entailment | mean_row | +0.0050 [-0.0050, +0.0167] | 600 | 0.8976 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3383 | 0.2654 | 0.5343 |
| none | 0.3370 | 0.2644 | 0.5343 |
| mean_row | 0.3370 | 0.2644 | 0.5343 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0013 [-0.0023, +0.0056] | 1006 | 0.8216 |
| property | mean_row | +0.0013 [-0.0017, +0.0046] | 1006 | 0.8216 |
| paraphrase | none | +0.0010 [-0.0080, +0.0099] | 1006 | 1.0000 |
| paraphrase | mean_row | +0.0010 [-0.0060, +0.0080] | 1006 | 1.0000 |
| entailment | none | +0.0000 [-0.0074, +0.0065] | 1080 | 1.0000 |
| entailment | mean_row | +0.0000 [-0.0065, +0.0065] | 1080 | 1.0000 |
