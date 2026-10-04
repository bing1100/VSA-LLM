# E9 track zero-shot (t5) — C5 seed 1 (Qwen/Qwen3-0.6B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3785 | 0.2676 | 0.6512 |
| none | 0.3084 | 0.2133 | 0.5500 |
| mean_row | 0.2803 | 0.2356 | 0.5375 |
| random_frame | 0.2678 | 0.2656 | 0.5190 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0700 [+0.0545, +0.0856] | 1566 | 0.0030 |
| property | mean_row | +0.0981 [+0.0860, +0.1113] | 1566 | 0.0030 |
| property | random_frame | +0.1107 [+0.0977, +0.1243] | 1566 | 0.0030 |
| paraphrase | none | +0.0543 [+0.0281, +0.0805] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0319 [+0.0115, +0.0524] | 1566 | 0.0080 |
| paraphrase | random_frame | +0.0019 [-0.0198, +0.0249] | 1566 | 0.8716 |
| entailment | none | +0.1012 [+0.0780, +0.1256] | 1680 | 0.0030 |
| entailment | mean_row | +0.1137 [+0.0946, +0.1315] | 1680 | 0.0030 |
| entailment | random_frame | +0.1321 [+0.1119, +0.1524] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3696 | 0.2161 | 0.6367 |
| none | 0.2952 | 0.1821 | 0.5067 |
| mean_row | 0.2786 | 0.1946 | 0.5267 |
| random_frame | 0.2512 | 0.2304 | 0.4950 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0744 [+0.0458, +0.1024] | 560 | 0.0030 |
| property | mean_row | +0.0911 [+0.0708, +0.1119] | 560 | 0.0030 |
| property | random_frame | +0.1185 [+0.0952, +0.1423] | 560 | 0.0030 |
| paraphrase | none | +0.0339 [-0.0071, +0.0732] | 560 | 0.3898 |
| paraphrase | mean_row | +0.0214 [-0.0125, +0.0571] | 560 | 0.4898 |
| paraphrase | random_frame | -0.0143 [-0.0518, +0.0233] | 560 | 0.5027 |
| entailment | none | +0.1300 [+0.0883, +0.1700] | 600 | 0.0030 |
| entailment | mean_row | +0.1100 [+0.0783, +0.1417] | 600 | 0.0030 |
| entailment | random_frame | +0.1417 [+0.1067, +0.1767] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3834 | 0.2962 | 0.6593 |
| none | 0.3158 | 0.2306 | 0.5741 |
| mean_row | 0.2813 | 0.2584 | 0.5435 |
| random_frame | 0.2770 | 0.2853 | 0.5324 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0676 [+0.0484, +0.0878] | 1006 | 0.0030 |
| property | mean_row | +0.1021 [+0.0861, +0.1170] | 1006 | 0.0030 |
| property | random_frame | +0.1064 [+0.0898, +0.1236] | 1006 | 0.0030 |
| paraphrase | none | +0.0656 [+0.0338, +0.0964] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0378 [+0.0119, +0.0646] | 1006 | 0.0160 |
| paraphrase | random_frame | +0.0109 [-0.0179, +0.0398] | 1006 | 0.4598 |
| entailment | none | +0.0852 [+0.0574, +0.1148] | 1080 | 0.0030 |
| entailment | mean_row | +0.1157 [+0.0926, +0.1389] | 1080 | 0.0030 |
| entailment | random_frame | +0.1269 [+0.1019, +0.1528] | 1080 | 0.0030 |
