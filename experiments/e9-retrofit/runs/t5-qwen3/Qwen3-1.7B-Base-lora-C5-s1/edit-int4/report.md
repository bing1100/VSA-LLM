# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2613 | 0.2555 | 0.2850 | 0.5767 | 0.3492 | 0.2773 | -1.1069 | 5.3151 |
| none | 0.2060 | 0.2030 | 0.2183 | 0.5000 | 0.3250 | 0.2230 | -1.2214 | 6.1478 |
| mean_row | 0.1985 | 0.2006 | 0.1900 | 0.5100 | 0.3264 | 0.2067 | -1.5841 | 5.6756 |
| random_frame | 0.2031 | 0.2018 | 0.2083 | 0.5050 | 0.3401 | 0.2119 | -1.5088 | 5.7130 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0553 [+0.0383, +0.0716] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0628 [+0.0517, +0.0746] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0582 [+0.0458, +0.0713] | 1529 | 0.0030 | yes |
| property_new | none | +0.0525 [+0.0342, +0.0716] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0549 [+0.0419, +0.0680] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0537 [+0.0399, +0.0675] | 1229 | 0.0030 | yes |
| property_category | none | +0.0667 [+0.0183, +0.1133] | 300 | 0.0110 | yes |
| property_category | mean_row | +0.0950 [+0.0650, +0.1267] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0767 [+0.0433, +0.1100] | 300 | 0.0030 | yes |
| entailment | none | +0.0767 [+0.0317, +0.1200] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.0667 [+0.0367, +0.1000] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0717 [+0.0383, +0.1067] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0242 [-0.0046, +0.0530] | 1529 | 0.1839 | no |
| paraphrase | mean_row | +0.0229 [+0.0007, +0.0451] | 1529 | 0.1409 | no |
| paraphrase | random_frame | +0.0092 [-0.0137, +0.0347] | 1529 | 0.4988 | no |
| statement_accuracy | none | +0.0543 [+0.0327, +0.0759] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0706 [+0.0530, +0.0876] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0654 [+0.0484, +0.0824] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1145 [+0.0298, +0.1998] | 1529 | 0.0080 | yes |
| statement_margin | mean_row | +0.4773 [+0.4245, +0.5347] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.4019 [+0.3396, +0.4678] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.8327 [-0.9379, -0.7364] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.3605 [-0.4102, -0.3132] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.3979 [-0.4620, -0.3375] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.492 → 0.527 (0.510) | +0.3808 [+0.1580, +0.6550] | +0.1962 [+0.0601, +0.3487] | 0.485 | +0.2201 [+0.0518, +0.4109] | 0.515 | 0.0029 / 0.4676 | 0.509 |
| seen | 100 | 0.460 → 0.535 (0.490) | +0.7375 [+0.3037, +1.2474] | +0.3289 [+0.0856, +0.5976] | 0.490 | +0.4568 [+0.1625, +0.7897] | 0.517 | 0.0034 / 0.2562 | 0.513 |
| heldout | 100 | 0.525 → 0.520 (0.530) | +0.0240 [-0.0777, +0.1252] | +0.0634 [-0.0340, +0.1576] | 0.480 | -0.0166 [-0.1555, +0.1246] | 0.512 | 0.0023 / 0.4676 | 0.504 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
