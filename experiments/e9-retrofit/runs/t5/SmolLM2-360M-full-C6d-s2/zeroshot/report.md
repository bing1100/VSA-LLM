# E9 track zero-shot (t5) — C6d seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4325 | 0.3429 | 0.6333 |
| none | 0.3538 | 0.3231 | 0.5607 |
| mean_row | 0.3559 | 0.3218 | 0.5690 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0788 [+0.0656, +0.0922] | 1566 | 0.0020 |
| property | mean_row | +0.0766 [+0.0634, +0.0900] | 1566 | 0.0020 |
| paraphrase | none | +0.0198 [-0.0057, +0.0453] | 1566 | 0.1939 |
| paraphrase | mean_row | +0.0211 [-0.0032, +0.0453] | 1566 | 0.1939 |
| entailment | none | +0.0726 [+0.0547, +0.0917] | 1680 | 0.0020 |
| entailment | mean_row | +0.0643 [+0.0464, +0.0833] | 1680 | 0.0020 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4470 | 0.3518 | 0.6550 |
| none | 0.3506 | 0.3375 | 0.5683 |
| mean_row | 0.3518 | 0.3339 | 0.5783 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0964 [+0.0726, +0.1208] | 560 | 0.0020 |
| property | mean_row | +0.0952 [+0.0714, +0.1202] | 560 | 0.0020 |
| paraphrase | none | +0.0143 [-0.0250, +0.0536] | 560 | 0.8356 |
| paraphrase | mean_row | +0.0179 [-0.0232, +0.0572] | 560 | 0.8356 |
| entailment | none | +0.0867 [+0.0550, +0.1183] | 600 | 0.0020 |
| entailment | mean_row | +0.0767 [+0.0433, +0.1100] | 600 | 0.0020 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4245 | 0.3380 | 0.6213 |
| none | 0.3555 | 0.3151 | 0.5565 |
| mean_row | 0.3582 | 0.3151 | 0.5639 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0689 [+0.0537, +0.0848] | 1006 | 0.0020 |
| property | mean_row | +0.0663 [+0.0510, +0.0825] | 1006 | 0.0020 |
| paraphrase | none | +0.0229 [-0.0070, +0.0527] | 1006 | 0.2639 |
| paraphrase | mean_row | +0.0229 [-0.0060, +0.0527] | 1006 | 0.2639 |
| entailment | none | +0.0648 [+0.0407, +0.0898] | 1080 | 0.0020 |
| entailment | mean_row | +0.0574 [+0.0333, +0.0824] | 1080 | 0.0020 |
