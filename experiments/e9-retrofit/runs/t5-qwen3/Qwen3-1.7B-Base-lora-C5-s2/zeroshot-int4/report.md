# E9 track zero-shot (t5) — C5 seed 2 (Qwen/Qwen3-1.7B-Base/lora, int4-A)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4168 | 0.3059 | 0.6393 |
| none | 0.3374 | 0.2835 | 0.5506 |
| mean_row | 0.3186 | 0.3123 | 0.5417 |
| random_frame | 0.3225 | 0.2925 | 0.5411 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0794 [+0.0656, +0.0932] | 1566 | 0.0030 |
| property | mean_row | +0.0981 [+0.0866, +0.1103] | 1566 | 0.0030 |
| property | random_frame | +0.0943 [+0.0817, +0.1069] | 1566 | 0.0030 |
| paraphrase | none | +0.0223 [-0.0045, +0.0485] | 1566 | 0.3058 |
| paraphrase | mean_row | -0.0064 [-0.0275, +0.0147] | 1566 | 0.5807 |
| paraphrase | random_frame | +0.0134 [-0.0096, +0.0358] | 1566 | 0.5637 |
| entailment | none | +0.0887 [+0.0696, +0.1077] | 1680 | 0.0030 |
| entailment | mean_row | +0.0976 [+0.0792, +0.1155] | 1680 | 0.0030 |
| entailment | random_frame | +0.0982 [+0.0792, +0.1167] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4060 | 0.2821 | 0.6367 |
| none | 0.3292 | 0.2875 | 0.5617 |
| mean_row | 0.3012 | 0.3000 | 0.5250 |
| random_frame | 0.2958 | 0.2982 | 0.5233 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0768 [+0.0554, +0.0988] | 560 | 0.0030 |
| property | mean_row | +0.1048 [+0.0863, +0.1238] | 560 | 0.0030 |
| property | random_frame | +0.1101 [+0.0899, +0.1310] | 560 | 0.0030 |
| paraphrase | none | -0.0054 [-0.0446, +0.0393] | 560 | 1.0000 |
| paraphrase | mean_row | -0.0179 [-0.0536, +0.0179] | 560 | 1.0000 |
| paraphrase | random_frame | -0.0161 [-0.0554, +0.0250] | 560 | 1.0000 |
| entailment | none | +0.0750 [+0.0400, +0.1083] | 600 | 0.0030 |
| entailment | mean_row | +0.1117 [+0.0817, +0.1433] | 600 | 0.0030 |
| entailment | random_frame | +0.1133 [+0.0800, +0.1467] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4228 | 0.3191 | 0.6407 |
| none | 0.3419 | 0.2813 | 0.5444 |
| mean_row | 0.3284 | 0.3191 | 0.5509 |
| random_frame | 0.3373 | 0.2893 | 0.5509 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0808 [+0.0639, +0.0991] | 1006 | 0.0030 |
| property | mean_row | +0.0944 [+0.0805, +0.1093] | 1006 | 0.0030 |
| property | random_frame | +0.0855 [+0.0706, +0.0994] | 1006 | 0.0030 |
| paraphrase | none | +0.0378 [+0.0060, +0.0706] | 1006 | 0.0840 |
| paraphrase | mean_row | +0.0000 [-0.0278, +0.0278] | 1006 | 0.9905 |
| paraphrase | random_frame | +0.0298 [+0.0040, +0.0567] | 1006 | 0.0840 |
| entailment | none | +0.0963 [+0.0722, +0.1222] | 1080 | 0.0030 |
| entailment | mean_row | +0.0898 [+0.0676, +0.1120] | 1080 | 0.0030 |
| entailment | random_frame | +0.0898 [+0.0667, +0.1139] | 1080 | 0.0030 |
