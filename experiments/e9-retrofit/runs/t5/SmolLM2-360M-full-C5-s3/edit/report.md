# E9 dimension 3 — zero-shot by ontology editing — C5 seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s3`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2623 | 0.2616 | 0.2650 | 0.5333 | 0.4022 | 0.2525 | -0.9834 | 4.5187 |
| none | 0.1897 | 0.1924 | 0.1783 | 0.5117 | 0.3388 | 0.2256 | -1.0558 | 4.7383 |
| mean_row | 0.2050 | 0.2079 | 0.1933 | 0.4933 | 0.3800 | 0.2132 | -1.1265 | 4.6309 |
| random_frame | 0.2070 | 0.2055 | 0.2133 | 0.5083 | 0.4075 | 0.2211 | -1.1304 | 4.6476 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0726 [+0.0579, +0.0870] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0572 [+0.0435, +0.0713] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0553 [+0.0399, +0.0700] | 1529 | 0.0030 | yes |
| property_new | none | +0.0692 [+0.0521, +0.0850] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0537 [+0.0386, +0.0688] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0561 [+0.0407, +0.0724] | 1229 | 0.0030 | yes |
| property_category | none | +0.0867 [+0.0483, +0.1283] | 300 | 0.0040 | yes |
| property_category | mean_row | +0.0717 [+0.0367, +0.1034] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0517 [+0.0150, +0.0883] | 300 | 0.0120 | yes |
| entailment | none | +0.0217 [-0.0117, +0.0550] | 600 | 0.2579 | no |
| entailment | mean_row | +0.0400 [+0.0117, +0.0683] | 600 | 0.0270 | yes |
| entailment | random_frame | +0.0250 [-0.0050, +0.0550] | 600 | 0.2579 | no |
| paraphrase | none | +0.0634 [+0.0366, +0.0909] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0222 [-0.0020, +0.0445] | 1529 | 0.1479 | no |
| paraphrase | random_frame | -0.0052 [-0.0320, +0.0216] | 1529 | 0.7466 | no |
| statement_accuracy | none | +0.0268 [+0.0118, +0.0438] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0392 [+0.0249, +0.0536] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0314 [+0.0170, +0.0477] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0724 [+0.0355, +0.1132] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1431 [+0.1107, +0.1752] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1470 [+0.1099, +0.1862] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2196 [-0.2629, -0.1837] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1123 [-0.1431, -0.0813] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1290 [-0.1701, -0.0900] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.427 → 0.455 (0.440) | +0.3686 [+0.1843, +0.5906] | +0.1384 [+0.0271, +0.2812] | 0.470 | +0.2781 [+0.1394, +0.4446] | 0.547 | 0.0015 / 0.1366 | 0.488 |
| seen | 100 | 0.400 → 0.445 (0.430) | +0.6524 [+0.2942, +1.0572] | +0.2523 [+0.0573, +0.5035] | 0.450 | +0.5032 [+0.2519, +0.7939] | 0.547 | 0.0021 / 0.1366 | 0.477 |
| heldout | 100 | 0.455 → 0.465 (0.450) | +0.0849 [-0.0176, +0.1811] | +0.0246 [-0.0581, +0.1031] | 0.490 | +0.0531 [-0.0382, +0.1356] | 0.547 | 0.0009 / 0.1177 | 0.499 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
