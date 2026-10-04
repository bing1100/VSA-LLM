# E9 track zero-shot (t5) — C5 seed 1 (Qwen/Qwen3-1.7B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4163 | 0.3167 | 0.6268 |
| none | 0.3231 | 0.2625 | 0.5286 |
| mean_row | 0.3172 | 0.2905 | 0.5417 |
| random_frame | 0.3116 | 0.2918 | 0.5381 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0932 [+0.0788, +0.1083] | 1566 | 0.0030 |
| property | mean_row | +0.0992 [+0.0875, +0.1109] | 1566 | 0.0030 |
| property | random_frame | +0.1047 [+0.0919, +0.1171] | 1566 | 0.0030 |
| paraphrase | none | +0.0543 [+0.0268, +0.0811] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0262 [+0.0038, +0.0473] | 1566 | 0.0540 |
| paraphrase | random_frame | +0.0249 [+0.0019, +0.0473] | 1566 | 0.0540 |
| entailment | none | +0.0982 [+0.0768, +0.1196] | 1680 | 0.0030 |
| entailment | mean_row | +0.0851 [+0.0679, +0.1018] | 1680 | 0.0030 |
| entailment | random_frame | +0.0887 [+0.0708, +0.1065] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.3952 | 0.3125 | 0.6150 |
| none | 0.3077 | 0.2696 | 0.5217 |
| mean_row | 0.2964 | 0.2839 | 0.5350 |
| random_frame | 0.2756 | 0.2946 | 0.5233 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0875 [+0.0643, +0.1119] | 560 | 0.0030 |
| property | mean_row | +0.0988 [+0.0797, +0.1190] | 560 | 0.0030 |
| property | random_frame | +0.1196 [+0.0976, +0.1411] | 560 | 0.0030 |
| paraphrase | none | +0.0429 [-0.0018, +0.0875] | 560 | 0.2189 |
| paraphrase | mean_row | +0.0286 [-0.0089, +0.0643] | 560 | 0.2499 |
| paraphrase | random_frame | +0.0179 [-0.0214, +0.0554] | 560 | 0.3858 |
| entailment | none | +0.0933 [+0.0600, +0.1267] | 600 | 0.0030 |
| entailment | mean_row | +0.0800 [+0.0517, +0.1117] | 600 | 0.0030 |
| entailment | random_frame | +0.0917 [+0.0617, +0.1217] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4281 | 0.3191 | 0.6333 |
| none | 0.3317 | 0.2584 | 0.5324 |
| mean_row | 0.3287 | 0.2942 | 0.5454 |
| random_frame | 0.3317 | 0.2903 | 0.5463 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0964 [+0.0775, +0.1153] | 1006 | 0.0030 |
| property | mean_row | +0.0994 [+0.0848, +0.1150] | 1006 | 0.0030 |
| property | random_frame | +0.0964 [+0.0805, +0.1130] | 1006 | 0.0030 |
| paraphrase | none | +0.0606 [+0.0268, +0.0934] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0249 [-0.0020, +0.0517] | 1006 | 0.1259 |
| paraphrase | random_frame | +0.0288 [+0.0000, +0.0577] | 1006 | 0.1259 |
| entailment | none | +0.1009 [+0.0722, +0.1278] | 1080 | 0.0030 |
| entailment | mean_row | +0.0880 [+0.0657, +0.1102] | 1080 | 0.0030 |
| entailment | random_frame | +0.0870 [+0.0648, +0.1102] | 1080 | 0.0030 |
