# E9 track zero-shot (t5) — C2 seed 2 (Qwen/Qwen3-0.6B-Base/lora)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3025 | 0.2848 | 0.5393 |
| none | 0.3006 | 0.2778 | 0.5411 |
| mean_row | 0.3033 | 0.2816 | 0.5399 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0019 [-0.0015, +0.0053] | 1566 | 0.5617 |
| property | mean_row | -0.0009 [-0.0038, +0.0021] | 1566 | 0.5807 |
| paraphrase | none | +0.0070 [-0.0006, +0.0147] | 1566 | 0.1699 |
| paraphrase | mean_row | +0.0032 [-0.0038, +0.0109] | 1566 | 0.4628 |
| entailment | none | -0.0018 [-0.0083, +0.0054] | 1680 | 1.0000 |
| entailment | mean_row | -0.0006 [-0.0065, +0.0060] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.2970 | 0.2643 | 0.5433 |
| none | 0.3006 | 0.2500 | 0.5467 |
| mean_row | 0.2964 | 0.2571 | 0.5417 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0036 [-0.0095, +0.0024] | 560 | 0.4698 |
| property | mean_row | +0.0006 [-0.0048, +0.0060] | 560 | 0.8376 |
| paraphrase | none | +0.0143 [+0.0018, +0.0286] | 560 | 0.0740 |
| paraphrase | mean_row | +0.0071 [-0.0054, +0.0196] | 560 | 0.2879 |
| entailment | none | -0.0033 [-0.0117, +0.0050] | 600 | 1.0000 |
| entailment | mean_row | +0.0017 [-0.0067, +0.0100] | 600 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3055 | 0.2962 | 0.5370 |
| none | 0.3005 | 0.2932 | 0.5380 |
| mean_row | 0.3072 | 0.2952 | 0.5389 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0050 [+0.0007, +0.0093] | 1006 | 0.0680 |
| property | mean_row | -0.0017 [-0.0053, +0.0020] | 1006 | 0.3768 |
| paraphrase | none | +0.0030 [-0.0060, +0.0129] | 1006 | 1.0000 |
| paraphrase | mean_row | +0.0010 [-0.0080, +0.0099] | 1006 | 1.0000 |
| entailment | none | -0.0009 [-0.0102, +0.0083] | 1080 | 1.0000 |
| entailment | mean_row | -0.0019 [-0.0111, +0.0065] | 1080 | 1.0000 |
