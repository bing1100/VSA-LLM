# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C5-s2`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2681 | 0.2718 | 0.2533 | 0.5567 | 0.4075 | 0.2649 | -1.3865 | 6.0012 |
| none | 0.2201 | 0.2258 | 0.1967 | 0.5000 | 0.2747 | 0.2335 | -1.3645 | 6.4802 |
| mean_row | 0.1835 | 0.1892 | 0.1600 | 0.5000 | 0.4061 | 0.2237 | -1.8583 | 6.3878 |
| random_frame | 0.2005 | 0.2038 | 0.1867 | 0.5083 | 0.4101 | 0.2302 | -1.7311 | 6.3201 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0481 [+0.0304, +0.0661] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0847 [+0.0706, +0.0984] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0677 [+0.0536, +0.0818] | 1529 | 0.0030 | yes |
| property_new | none | +0.0460 [+0.0248, +0.0675] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0826 [+0.0667, +0.0985] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0679 [+0.0521, +0.0830] | 1229 | 0.0030 | yes |
| property_category | none | +0.0567 [+0.0167, +0.0950] | 300 | 0.0060 | yes |
| property_category | mean_row | +0.0933 [+0.0616, +0.1267] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0667 [+0.0350, +0.1000] | 300 | 0.0030 | yes |
| entailment | none | +0.0567 [+0.0150, +0.1017] | 600 | 0.0160 | yes |
| entailment | mean_row | +0.0567 [+0.0267, +0.0867] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0483 [+0.0150, +0.0817] | 600 | 0.0160 | yes |
| paraphrase | none | +0.1328 [+0.1033, +0.1629] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0013 [-0.0249, +0.0268] | 1529 | 1.0000 | no |
| paraphrase | random_frame | -0.0026 [-0.0288, +0.0235] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0314 [+0.0105, +0.0510] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0412 [+0.0255, +0.0569] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0347 [+0.0177, +0.0504] | 1529 | 0.0030 | yes |
| statement_margin | none | -0.0221 [-0.1065, +0.0656] | 1529 | 0.6227 | no |
| statement_margin | mean_row | +0.4717 [+0.4051, +0.5431] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.3446 [+0.2726, +0.4182] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.4790 [-0.5819, -0.3823] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.3866 [-0.4542, -0.3201] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.3189 [-0.3931, -0.2463] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.475 → 0.502 (0.480) | +0.3205 [+0.1150, +0.5771] | +0.1414 [+0.0088, +0.3027] | 0.535 | +0.0157 [-0.0948, +0.1264] | 0.515 | 0.0012 / 0.1620 | 0.517 |
| seen | 100 | 0.455 → 0.510 (0.470) | +0.6183 [+0.2220, +1.0893] | +0.2936 [+0.0591, +0.5761] | 0.550 | +0.1449 [-0.0210, +0.3299] | 0.522 | 0.0007 / 0.1337 | 0.527 |
| heldout | 100 | 0.495 → 0.495 (0.490) | +0.0226 [-0.1114, +0.1551] | -0.0108 [-0.1087, +0.0882] | 0.520 | -0.1136 [-0.2576, +0.0291] | 0.507 | 0.0018 / 0.1620 | 0.507 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
