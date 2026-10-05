# E9 track zero-shot (t5) — C5 seed 3 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4019 | 0.2989 | 0.5893 |
| none | 0.3208 | 0.2656 | 0.5232 |
| mean_row | 0.3050 | 0.2554 | 0.5333 |
| random_frame | 0.2978 | 0.2656 | 0.5179 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0811 [+0.0647, +0.0971] | 1566 | 0.0030 |
| property | mean_row | +0.0968 [+0.0832, +0.1101] | 1566 | 0.0030 |
| property | random_frame | +0.1041 [+0.0898, +0.1188] | 1566 | 0.0030 |
| paraphrase | none | +0.0332 [+0.0089, +0.0568] | 1566 | 0.0160 |
| paraphrase | mean_row | +0.0434 [+0.0211, +0.0645] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0332 [+0.0115, +0.0549] | 1566 | 0.0160 |
| entailment | none | +0.0661 [+0.0452, +0.0881] | 1680 | 0.0030 |
| entailment | mean_row | +0.0560 [+0.0387, +0.0738] | 1680 | 0.0030 |
| entailment | random_frame | +0.0714 [+0.0524, +0.0893] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4143 | 0.3179 | 0.5983 |
| none | 0.3280 | 0.2875 | 0.5267 |
| mean_row | 0.3107 | 0.2714 | 0.5367 |
| random_frame | 0.2940 | 0.2714 | 0.5033 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0863 [+0.0589, +0.1137] | 560 | 0.0030 |
| property | mean_row | +0.1036 [+0.0815, +0.1262] | 560 | 0.0030 |
| property | random_frame | +0.1202 [+0.0964, +0.1440] | 560 | 0.0030 |
| paraphrase | none | +0.0304 [-0.0125, +0.0732] | 560 | 0.1979 |
| paraphrase | mean_row | +0.0464 [+0.0089, +0.0822] | 560 | 0.0600 |
| paraphrase | random_frame | +0.0464 [+0.0071, +0.0821] | 560 | 0.0600 |
| entailment | none | +0.0717 [+0.0350, +0.1100] | 600 | 0.0030 |
| entailment | mean_row | +0.0617 [+0.0333, +0.0917] | 600 | 0.0030 |
| entailment | random_frame | +0.0950 [+0.0617, +0.1250] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3950 | 0.2883 | 0.5843 |
| none | 0.3168 | 0.2535 | 0.5213 |
| mean_row | 0.3019 | 0.2465 | 0.5315 |
| random_frame | 0.2999 | 0.2624 | 0.5259 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0782 [+0.0600, +0.0978] | 1006 | 0.0030 |
| property | mean_row | +0.0931 [+0.0772, +0.1097] | 1006 | 0.0030 |
| property | random_frame | +0.0951 [+0.0779, +0.1137] | 1006 | 0.0030 |
| paraphrase | none | +0.0348 [+0.0040, +0.0646] | 1006 | 0.0640 |
| paraphrase | mean_row | +0.0417 [+0.0169, +0.0676] | 1006 | 0.0150 |
| paraphrase | random_frame | +0.0258 [-0.0010, +0.0537] | 1006 | 0.0660 |
| entailment | none | +0.0630 [+0.0370, +0.0889] | 1080 | 0.0030 |
| entailment | mean_row | +0.0528 [+0.0306, +0.0750] | 1080 | 0.0030 |
| entailment | random_frame | +0.0583 [+0.0361, +0.0806] | 1080 | 0.0030 |
