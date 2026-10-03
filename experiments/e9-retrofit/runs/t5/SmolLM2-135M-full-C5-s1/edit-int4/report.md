# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2283 | 0.2400 | 0.1800 | 0.5383 | 0.3663 | 0.2525 | -1.3600 | 6.8845 |
| none | 0.1988 | 0.2071 | 0.1650 | 0.4883 | 0.3630 | 0.2348 | -1.4933 | 7.3029 |
| mean_row | 0.1923 | 0.2055 | 0.1383 | 0.5117 | 0.3715 | 0.2387 | -1.4612 | 6.9882 |
| random_frame | 0.1933 | 0.1981 | 0.1733 | 0.5217 | 0.3898 | 0.2283 | -1.4629 | 6.9962 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0294 [+0.0150, +0.0435] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0360 [+0.0245, +0.0478] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0350 [+0.0232, +0.0474] | 1529 | 0.0030 | yes |
| property_new | none | +0.0330 [+0.0171, +0.0484] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0346 [+0.0207, +0.0480] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0419 [+0.0285, +0.0553] | 1229 | 0.0030 | yes |
| property_category | none | +0.0150 [-0.0133, +0.0467] | 300 | 0.7236 | no |
| property_category | mean_row | +0.0417 [+0.0150, +0.0683] | 300 | 0.0060 | yes |
| property_category | random_frame | +0.0067 [-0.0217, +0.0367] | 300 | 0.7236 | no |
| entailment | none | +0.0500 [+0.0217, +0.0800] | 600 | 0.0090 | yes |
| entailment | mean_row | +0.0267 [+0.0017, +0.0500] | 600 | 0.0920 | no |
| entailment | random_frame | +0.0167 [-0.0067, +0.0417] | 600 | 0.2109 | no |
| paraphrase | none | +0.0033 [-0.0222, +0.0294] | 1529 | 1.0000 | no |
| paraphrase | mean_row | -0.0052 [-0.0288, +0.0183] | 1529 | 1.0000 | no |
| paraphrase | random_frame | -0.0235 [-0.0477, +0.0013] | 1529 | 0.2069 | no |
| statement_accuracy | none | +0.0177 [+0.0052, +0.0294] | 1529 | 0.0060 | yes |
| statement_accuracy | mean_row | +0.0137 [+0.0033, +0.0249] | 1529 | 0.0130 | yes |
| statement_accuracy | random_frame | +0.0242 [+0.0118, +0.0366] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1333 [+0.1024, +0.1661] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1012 [+0.0792, +0.1259] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1029 [+0.0758, +0.1298] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.4184 [-0.4603, -0.3780] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1037 [-0.1316, -0.0775] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1116 [-0.1456, -0.0785] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.465 → 0.495 (0.480) | +0.7442 [+0.5841, +0.9168] | +0.3170 [+0.2028, +0.4362] | 0.475 | +0.9376 [+0.7787, +1.0982] | 0.492 | 0.0005 / 0.1055 | 0.487 |
| seen | 100 | 0.445 → 0.495 (0.470) | +0.9112 [+0.6467, +1.2048] | +0.3622 [+0.1882, +0.5481] | 0.470 | +1.0046 [+0.7578, +1.2645] | 0.470 | 0.0002 / 0.0495 | 0.478 |
| heldout | 100 | 0.485 → 0.495 (0.490) | +0.5773 [+0.4113, +0.7397] | +0.2717 [+0.1190, +0.4266] | 0.480 | +0.8706 [+0.6763, +1.0633] | 0.515 | 0.0008 / 0.1055 | 0.496 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
