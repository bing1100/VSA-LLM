# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C5-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3404 | 0.3397 | 0.3433 | 0.5800 | 0.4460 | 0.3362 | -0.7124 | 4.5309 |
| none | 0.1910 | 0.2055 | 0.1317 | 0.4517 | 0.4192 | 0.2309 | -1.1373 | 4.5556 |
| mean_row | 0.1857 | 0.1981 | 0.1350 | 0.4667 | 0.3911 | 0.2224 | -1.3931 | 5.1140 |
| random_frame | 0.2011 | 0.2050 | 0.1850 | 0.4450 | 0.4454 | 0.2309 | -1.3884 | 5.1435 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.1494 [+0.1295, +0.1694] | 1529 | 0.0030 | yes |
| property | mean_row | +0.1547 [+0.1380, +0.1714] | 1529 | 0.0030 | yes |
| property | random_frame | +0.1393 [+0.1210, +0.1586] | 1529 | 0.0030 | yes |
| property_new | none | +0.1343 [+0.1123, +0.1566] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.1416 [+0.1233, +0.1607] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.1347 [+0.1139, +0.1554] | 1229 | 0.0030 | yes |
| property_category | none | +0.2117 [+0.1733, +0.2483] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.2083 [+0.1750, +0.2417] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.1583 [+0.1200, +0.1950] | 300 | 0.0030 | yes |
| entailment | none | +0.1283 [+0.0900, +0.1667] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.1133 [+0.0817, +0.1467] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.1350 [+0.1033, +0.1700] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0268 [-0.0033, +0.0562] | 1529 | 0.1779 | no |
| paraphrase | mean_row | +0.0549 [+0.0288, +0.0824] | 1529 | 0.0030 | yes |
| paraphrase | random_frame | +0.0007 [-0.0275, +0.0294] | 1529 | 0.9685 | no |
| statement_accuracy | none | +0.1053 [+0.0831, +0.1275] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.1138 [+0.0948, +0.1341] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.1053 [+0.0857, +0.1262] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.4249 [+0.3467, +0.4994] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.6807 [+0.6012, +0.7604] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.6760 [+0.5875, +0.7695] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.0247 [-0.1109, +0.0581] | 1529 | 0.5847 | no |
| statement_loss | mean_row | -0.5831 [-0.6488, -0.5153] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.6126 [-0.7082, -0.5301] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.420 → 0.465 (0.438) | +0.5652 [+0.2522, +0.9882] | +0.2863 [+0.0776, +0.5577] | 0.445 | +0.2497 [+0.1137, +0.4141] | 0.593 | 0.0010 / 0.1757 | 0.493 |
| seen | 100 | 0.450 → 0.535 (0.490) | +1.0936 [+0.4568, +1.8583] | +0.5662 [+0.1578, +1.0634] | 0.500 | +0.4479 [+0.1787, +0.7743] | 0.585 | 0.0010 / 0.1157 | 0.538 |
| heldout | 100 | 0.390 → 0.395 (0.385) | +0.0368 [-0.0186, +0.0964] | +0.0063 [-0.0384, +0.0488] | 0.390 | +0.0515 [-0.0296, +0.1310] | 0.600 | 0.0010 / 0.1757 | 0.444 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
