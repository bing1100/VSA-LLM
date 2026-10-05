# E9 track zero-shot (t4) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

WP-C7 items `experiments/t4-chemistry/items`: 671 of 672 terms link ({'heldout': 671, 'unlinked': 1}; {'linked': 671, 'synthetic': 299, 'heldout': 372}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5624 | 0.4987 | 0.7891 |
| none | 0.5651 | 0.5061 | 0.7864 |
| mean_row | 0.5636 | 0.4973 | 0.7864 |
| random_frame | 0.5588 | 0.4987 | 0.7850 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0027 [-0.0083, +0.0025] | 1484 | 0.6237 |
| property | mean_row | -0.0011 [-0.0067, +0.0038] | 1484 | 0.6687 |
| property | random_frame | +0.0036 [-0.0018, +0.0094] | 1484 | 0.6237 |
| paraphrase | none | -0.0074 [-0.0209, +0.0054] | 1484 | 0.8606 |
| paraphrase | mean_row | +0.0013 [-0.0101, +0.0128] | 1484 | 1.0000 |
| paraphrase | random_frame | +0.0000 [-0.0135, +0.0135] | 1484 | 1.0000 |
| entailment | none | +0.0027 [-0.0047, +0.0101] | 1484 | 1.0000 |
| entailment | mean_row | +0.0027 [-0.0054, +0.0108] | 1484 | 1.0000 |
| entailment | random_frame | +0.0040 [-0.0034, +0.0121] | 1484 | 0.9685 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5448 | 0.4861 | 0.7882 |
| none | 0.5448 | 0.4905 | 0.7926 |
| mean_row | 0.5425 | 0.4749 | 0.7882 |
| random_frame | 0.5433 | 0.4994 | 0.7915 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0000 [-0.0063, +0.0071] | 897 | 1.0000 |
| property | mean_row | +0.0022 [-0.0041, +0.0082] | 897 | 1.0000 |
| property | random_frame | +0.0015 [-0.0048, +0.0078] | 897 | 1.0000 |
| paraphrase | none | -0.0045 [-0.0201, +0.0100] | 897 | 0.6347 |
| paraphrase | mean_row | +0.0111 [-0.0011, +0.0245] | 897 | 0.2999 |
| paraphrase | random_frame | -0.0134 [-0.0290, +0.0022] | 897 | 0.2999 |
| entailment | none | -0.0045 [-0.0145, +0.0045] | 897 | 1.0000 |
| entailment | mean_row | +0.0000 [-0.0100, +0.0100] | 897 | 1.0000 |
| entailment | random_frame | -0.0033 [-0.0134, +0.0056] | 897 | 1.0000 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.5894 | 0.5179 | 0.7905 |
| none | 0.5963 | 0.5298 | 0.7768 |
| mean_row | 0.5957 | 0.5315 | 0.7836 |
| random_frame | 0.5826 | 0.4974 | 0.7751 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | -0.0068 [-0.0165, +0.0023] | 587 | 0.4258 |
| property | mean_row | -0.0062 [-0.0159, +0.0034] | 587 | 0.4258 |
| property | random_frame | +0.0068 [-0.0034, +0.0165] | 587 | 0.4258 |
| paraphrase | none | -0.0119 [-0.0341, +0.0085] | 587 | 0.5097 |
| paraphrase | mean_row | -0.0136 [-0.0375, +0.0085] | 587 | 0.5097 |
| paraphrase | random_frame | +0.0204 [-0.0034, +0.0443] | 587 | 0.3148 |
| entailment | none | +0.0136 [+0.0017, +0.0273] | 587 | 0.0720 |
| entailment | mean_row | +0.0068 [-0.0051, +0.0204] | 587 | 0.3568 |
| entailment | random_frame | +0.0153 [+0.0034, +0.0290] | 587 | 0.0450 |
