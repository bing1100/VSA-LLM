# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2472 | 0.2632 | 0.1817 | 0.5217 | 0.4147 | 0.2557 | -1.4881 | 6.0209 |
| none | 0.1939 | 0.1993 | 0.1717 | 0.4817 | 0.2878 | 0.2289 | -1.4084 | 6.7118 |
| mean_row | 0.1910 | 0.2030 | 0.1417 | 0.4683 | 0.4166 | 0.2119 | -1.8875 | 6.3597 |
| random_frame | 0.1956 | 0.2075 | 0.1467 | 0.4900 | 0.4114 | 0.2204 | -1.7601 | 6.2941 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0533 [+0.0356, +0.0713] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0562 [+0.0425, +0.0687] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0517 [+0.0363, +0.0664] | 1529 | 0.0030 | yes |
| property_new | none | +0.0639 [+0.0439, +0.0838] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0602 [+0.0452, +0.0749] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0557 [+0.0403, +0.0724] | 1229 | 0.0030 | yes |
| property_category | none | +0.0100 [-0.0217, +0.0417] | 300 | 0.5767 | no |
| property_category | mean_row | +0.0400 [+0.0117, +0.0683] | 300 | 0.0240 | yes |
| property_category | random_frame | +0.0350 [+0.0017, +0.0667] | 300 | 0.0820 | no |
| entailment | none | +0.0400 [-0.0033, +0.0867] | 600 | 0.1299 | no |
| entailment | mean_row | +0.0533 [+0.0233, +0.0850] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0317 [-0.0017, +0.0650] | 600 | 0.1299 | no |
| paraphrase | none | +0.1269 [+0.0948, +0.1589] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | -0.0020 [-0.0268, +0.0242] | 1529 | 1.0000 | no |
| paraphrase | random_frame | +0.0033 [-0.0249, +0.0301] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0268 [+0.0065, +0.0471] | 1529 | 0.0060 | yes |
| statement_accuracy | mean_row | +0.0438 [+0.0281, +0.0608] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0353 [+0.0190, +0.0530] | 1529 | 0.0030 | yes |
| statement_margin | none | -0.0797 [-0.1597, +0.0016] | 1529 | 0.0550 | no |
| statement_margin | mean_row | +0.3993 [+0.3301, +0.4679] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.2720 [+0.2032, +0.3467] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.6909 [-0.7808, -0.5969] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.3388 [-0.4024, -0.2751] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.2733 [-0.3440, -0.2036] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.468 → 0.485 (0.472) | +0.4366 [+0.1977, +0.7462] | +0.2053 [+0.0457, +0.3968] | 0.475 | +0.0391 [-0.1109, +0.1812] | 0.527 | 0.0055 / 0.3233 | 0.495 |
| seen | 100 | 0.490 → 0.530 (0.505) | +0.8122 [+0.3431, +1.3753] | +0.4076 [+0.1242, +0.7491] | 0.520 | +0.2434 [+0.0582, +0.4490] | 0.500 | 0.0055 / 0.3233 | 0.516 |
| heldout | 100 | 0.445 → 0.440 (0.440) | +0.0611 [-0.0510, +0.1773] | +0.0031 [-0.1128, +0.1176] | 0.430 | -0.1652 [-0.3756, +0.0176] | 0.555 | 0.0056 / 0.2854 | 0.469 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
