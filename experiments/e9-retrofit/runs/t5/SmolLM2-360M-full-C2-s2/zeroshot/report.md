# E9 track zero-shot (t5) — C2 seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3519 | 0.3193 | 0.5625 |
| none | 0.3504 | 0.3206 | 0.5643 |
| mean_row | 0.3514 | 0.3231 | 0.5607 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0015 [-0.0017, +0.0047] | 1566 | 0.7136 |
| property | mean_row | +0.0004 [-0.0028, +0.0036] | 1566 | 0.7976 |
| paraphrase | none | -0.0013 [-0.0083, +0.0057] | 1566 | 0.7896 |
| paraphrase | mean_row | -0.0038 [-0.0115, +0.0032] | 1566 | 0.6977 |
| entailment | none | -0.0018 [-0.0071, +0.0036] | 1680 | 1.0000 |
| entailment | mean_row | +0.0018 [-0.0036, +0.0071] | 1680 | 1.0000 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3488 | 0.3232 | 0.5717 |
| none | 0.3470 | 0.3304 | 0.5733 |
| mean_row | 0.3500 | 0.3357 | 0.5700 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0018 [-0.0030, +0.0071] | 560 | 1.0000 |
| property | mean_row | -0.0012 [-0.0071, +0.0048] | 560 | 1.0000 |
| paraphrase | none | -0.0071 [-0.0196, +0.0054] | 560 | 0.3018 |
| paraphrase | mean_row | -0.0125 [-0.0286, +0.0018] | 560 | 0.2499 |
| entailment | none | -0.0017 [-0.0083, +0.0050] | 600 | 1.0000 |
| entailment | mean_row | +0.0017 [-0.0050, +0.0083] | 600 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3535 | 0.3171 | 0.5574 |
| none | 0.3522 | 0.3151 | 0.5593 |
| mean_row | 0.3522 | 0.3161 | 0.5556 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0013 [-0.0027, +0.0056] | 1006 | 0.8636 |
| property | mean_row | +0.0013 [-0.0020, +0.0046] | 1006 | 0.8636 |
| paraphrase | none | +0.0020 [-0.0060, +0.0099] | 1006 | 1.0000 |
| paraphrase | mean_row | +0.0010 [-0.0070, +0.0089] | 1006 | 1.0000 |
| entailment | none | -0.0019 [-0.0093, +0.0056] | 1080 | 1.0000 |
| entailment | mean_row | +0.0019 [-0.0056, +0.0093] | 1080 | 1.0000 |
