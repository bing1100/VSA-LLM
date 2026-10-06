# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C5-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2927 | 0.2913 | 0.2983 | 0.5550 | 0.4604 | 0.2982 | -0.8487 | 4.2756 |
| none | 0.1952 | 0.2022 | 0.1667 | 0.4583 | 0.3891 | 0.2289 | -1.1915 | 4.9514 |
| mean_row | 0.2005 | 0.2132 | 0.1483 | 0.4533 | 0.4375 | 0.2269 | -1.4306 | 4.7159 |
| random_frame | 0.2109 | 0.2124 | 0.2050 | 0.4883 | 0.4343 | 0.2283 | -1.3671 | 4.7346 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0974 [+0.0804, +0.1145] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0922 [+0.0788, +0.1060] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0818 [+0.0680, +0.0965] | 1529 | 0.0030 | yes |
| property_new | none | +0.0891 [+0.0712, +0.1078] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0781 [+0.0631, +0.0928] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0789 [+0.0635, +0.0948] | 1229 | 0.0030 | yes |
| property_category | none | +0.1317 [+0.0883, +0.1750] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.1500 [+0.1183, +0.1833] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0933 [+0.0583, +0.1300] | 300 | 0.0030 | yes |
| entailment | none | +0.0967 [+0.0600, +0.1367] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.1017 [+0.0717, +0.1317] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0667 [+0.0350, +0.0983] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0713 [+0.0412, +0.1034] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0229 [-0.0033, +0.0484] | 1529 | 0.1000 | no |
| paraphrase | random_frame | +0.0262 [+0.0007, +0.0517] | 1529 | 0.1000 | no |
| statement_accuracy | none | +0.0693 [+0.0484, +0.0896] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0713 [+0.0549, +0.0876] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0700 [+0.0523, +0.0863] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.3428 [+0.2550, +0.4346] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.5819 [+0.5020, +0.6628] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.5184 [+0.4349, +0.6044] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.6759 [-0.7802, -0.5716] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.4403 [-0.5202, -0.3609] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.4590 [-0.5498, -0.3693] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.427 → 0.458 (0.432) | +0.4878 [+0.2028, +0.8810] | +0.2364 [+0.0812, +0.4386] | 0.445 | +0.1524 [+0.0501, +0.2933] | 0.579 | 0.0008 / 0.1066 | 0.487 |
| seen | 100 | 0.435 → 0.490 (0.445) | +0.9355 [+0.3401, +1.6560] | +0.4645 [+0.1579, +0.8475] | 0.490 | +0.2540 [+0.0466, +0.5017] | 0.560 | 0.0005 / 0.0542 | 0.511 |
| heldout | 100 | 0.420 → 0.425 (0.420) | +0.0402 [-0.0233, +0.1021] | +0.0082 [-0.0553, +0.0701] | 0.400 | +0.0508 [-0.0094, +0.1143] | 0.598 | 0.0011 / 0.1066 | 0.460 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
