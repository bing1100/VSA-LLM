# E9 dimension 3 — zero-shot by ontology editing — C5ut seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5ut-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2587 | 0.2612 | 0.2483 | 0.5517 | 0.4088 | 0.2472 | -0.9830 | 4.5643 |
| none | 0.1893 | 0.1912 | 0.1817 | 0.5183 | 0.3479 | 0.2263 | -1.0660 | 4.7794 |
| mean_row | 0.1975 | 0.1989 | 0.1917 | 0.4867 | 0.3721 | 0.2119 | -1.1528 | 4.6710 |
| random_frame | 0.2027 | 0.2010 | 0.2100 | 0.5000 | 0.3990 | 0.2152 | -1.1713 | 4.6976 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0693 [+0.0540, +0.0844] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0612 [+0.0477, +0.0746] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0559 [+0.0409, +0.0713] | 1529 | 0.0030 | yes |
| property_new | none | +0.0700 [+0.0541, +0.0879] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0622 [+0.0476, +0.0777] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0602 [+0.0448, +0.0769] | 1229 | 0.0030 | yes |
| property_category | none | +0.0667 [+0.0267, +0.1100] | 300 | 0.0040 | yes |
| property_category | mean_row | +0.0567 [+0.0250, +0.0900] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0383 [+0.0000, +0.0783] | 300 | 0.0530 | no |
| entailment | none | +0.0333 [+0.0000, +0.0667] | 600 | 0.0620 | no |
| entailment | mean_row | +0.0650 [+0.0350, +0.0950] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0517 [+0.0217, +0.0833] | 600 | 0.0040 | yes |
| paraphrase | none | +0.0608 [+0.0347, +0.0877] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0366 [+0.0105, +0.0621] | 1529 | 0.0180 | yes |
| paraphrase | random_frame | +0.0098 [-0.0170, +0.0353] | 1529 | 0.4778 | no |
| statement_accuracy | none | +0.0209 [+0.0052, +0.0379] | 1529 | 0.0170 | yes |
| statement_accuracy | mean_row | +0.0353 [+0.0209, +0.0497] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0320 [+0.0170, +0.0477] | 1529 | 0.0040 | yes |
| statement_margin | none | +0.0830 [+0.0455, +0.1237] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1698 [+0.1356, +0.2069] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1883 [+0.1466, +0.2330] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2151 [-0.2557, -0.1784] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1067 [-0.1427, -0.0724] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1332 [-0.1762, -0.0941] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.438 → 0.455 (0.458) | +0.3459 [+0.1802, +0.5618] | +0.1381 [+0.0456, +0.2450] | 0.475 | +0.2142 [+0.0947, +0.3668] | 0.550 | 0.0015 / 0.1221 | 0.490 |
| seen | 100 | 0.405 → 0.445 (0.445) | +0.6452 [+0.3124, +1.0258] | +0.2275 [+0.0683, +0.4083] | 0.460 | +0.3813 [+0.1382, +0.6675] | 0.560 | 0.0015 / 0.1221 | 0.483 |
| heldout | 100 | 0.470 → 0.465 (0.470) | +0.0466 [-0.0129, +0.1070] | +0.0487 [-0.0345, +0.1425] | 0.490 | +0.0470 [-0.0263, +0.1165] | 0.540 | 0.0015 / 0.1043 | 0.496 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
