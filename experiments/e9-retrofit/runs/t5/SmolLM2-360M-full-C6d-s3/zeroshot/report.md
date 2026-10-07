# E9 track zero-shot (t5) — C6d seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4244 | 0.3487 | 0.6375 |
| none | 0.3533 | 0.3263 | 0.5661 |
| mean_row | 0.3508 | 0.3295 | 0.5607 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0711 [+0.0575, +0.0847] | 1566 | 0.0020 |
| property | mean_row | +0.0736 [+0.0600, +0.0873] | 1566 | 0.0020 |
| paraphrase | none | +0.0223 [-0.0026, +0.0473] | 1566 | 0.1599 |
| paraphrase | mean_row | +0.0192 [-0.0051, +0.0441] | 1566 | 0.1599 |
| entailment | none | +0.0714 [+0.0524, +0.0899] | 1680 | 0.0020 |
| entailment | mean_row | +0.0768 [+0.0583, +0.0958] | 1680 | 0.0020 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4250 | 0.3518 | 0.6467 |
| none | 0.3512 | 0.3375 | 0.5700 |
| mean_row | 0.3470 | 0.3411 | 0.5667 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0738 [+0.0482, +0.0982] | 560 | 0.0020 |
| property | mean_row | +0.0780 [+0.0530, +0.1018] | 560 | 0.0020 |
| paraphrase | none | +0.0143 [-0.0268, +0.0536] | 560 | 1.0000 |
| paraphrase | mean_row | +0.0107 [-0.0286, +0.0518] | 560 | 1.0000 |
| entailment | none | +0.0767 [+0.0433, +0.1117] | 600 | 0.0020 |
| entailment | mean_row | +0.0800 [+0.0467, +0.1150] | 600 | 0.0020 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4241 | 0.3469 | 0.6324 |
| none | 0.3545 | 0.3201 | 0.5639 |
| mean_row | 0.3529 | 0.3231 | 0.5574 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0696 [+0.0533, +0.0868] | 1006 | 0.0020 |
| property | mean_row | +0.0712 [+0.0547, +0.0888] | 1006 | 0.0020 |
| paraphrase | none | +0.0268 [-0.0020, +0.0567] | 1006 | 0.1819 |
| paraphrase | mean_row | +0.0239 [-0.0050, +0.0547] | 1006 | 0.1819 |
| entailment | none | +0.0685 [+0.0454, +0.0926] | 1080 | 0.0020 |
| entailment | mean_row | +0.0750 [+0.0528, +0.0981] | 1080 | 0.0020 |
