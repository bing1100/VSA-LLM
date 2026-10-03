# E9 track zero-shot (t5) — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4042 | 0.2925 | 0.5732 |
| none | 0.3148 | 0.2676 | 0.5244 |
| mean_row | 0.2937 | 0.2727 | 0.5208 |
| random_frame | 0.2971 | 0.2676 | 0.5220 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0894 [+0.0736, +0.1045] | 1566 | 0.0030 |
| property | mean_row | +0.1105 [+0.0973, +0.1230] | 1566 | 0.0030 |
| property | random_frame | +0.1071 [+0.0937, +0.1209] | 1566 | 0.0030 |
| paraphrase | none | +0.0249 [+0.0006, +0.0498] | 1566 | 0.1000 |
| paraphrase | mean_row | +0.0198 [-0.0013, +0.0415] | 1566 | 0.1000 |
| paraphrase | random_frame | +0.0249 [+0.0045, +0.0473] | 1566 | 0.0600 |
| entailment | none | +0.0488 [+0.0286, +0.0690] | 1680 | 0.0030 |
| entailment | mean_row | +0.0524 [+0.0369, +0.0685] | 1680 | 0.0030 |
| entailment | random_frame | +0.0512 [+0.0333, +0.0702] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4208 | 0.2911 | 0.5800 |
| none | 0.3232 | 0.2732 | 0.5300 |
| mean_row | 0.2857 | 0.2625 | 0.5183 |
| random_frame | 0.2851 | 0.2732 | 0.5033 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0976 [+0.0696, +0.1244] | 560 | 0.0030 |
| property | mean_row | +0.1351 [+0.1137, +0.1583] | 560 | 0.0030 |
| property | random_frame | +0.1357 [+0.1119, +0.1607] | 560 | 0.0030 |
| paraphrase | none | +0.0179 [-0.0250, +0.0607] | 560 | 0.7776 |
| paraphrase | mean_row | +0.0286 [-0.0089, +0.0661] | 560 | 0.5037 |
| paraphrase | random_frame | +0.0179 [-0.0214, +0.0572] | 560 | 0.7776 |
| entailment | none | +0.0500 [+0.0133, +0.0833] | 600 | 0.0090 |
| entailment | mean_row | +0.0617 [+0.0350, +0.0917] | 600 | 0.0030 |
| entailment | random_frame | +0.0767 [+0.0467, +0.1100] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3950 | 0.2932 | 0.5694 |
| none | 0.3101 | 0.2644 | 0.5213 |
| mean_row | 0.2982 | 0.2783 | 0.5222 |
| random_frame | 0.3038 | 0.2644 | 0.5324 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0848 [+0.0663, +0.1047] | 1006 | 0.0030 |
| property | mean_row | +0.0968 [+0.0805, +0.1140] | 1006 | 0.0030 |
| property | random_frame | +0.0911 [+0.0739, +0.1083] | 1006 | 0.0030 |
| paraphrase | none | +0.0288 [-0.0010, +0.0577] | 1006 | 0.1239 |
| paraphrase | mean_row | +0.0149 [-0.0099, +0.0398] | 1006 | 0.2579 |
| paraphrase | random_frame | +0.0288 [+0.0030, +0.0547] | 1006 | 0.1079 |
| entailment | none | +0.0481 [+0.0222, +0.0741] | 1080 | 0.0030 |
| entailment | mean_row | +0.0472 [+0.0269, +0.0685] | 1080 | 0.0030 |
| entailment | random_frame | +0.0370 [+0.0130, +0.0602] | 1080 | 0.0030 |
