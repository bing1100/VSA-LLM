# E9 dimension 3 — zero-shot by ontology editing — C2 seed 3 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C2-s3`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2001 | 0.2079 | 0.1683 | 0.5000 | 0.3250 | 0.2283 | -1.4740 | 7.1462 |
| none | 0.2014 | 0.2111 | 0.1617 | 0.4933 | 0.3303 | 0.2296 | -1.4728 | 7.1494 |
| mean_row | 0.2024 | 0.2116 | 0.1650 | 0.4967 | 0.3250 | 0.2276 | -1.4722 | 7.1462 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0013 [-0.0056, +0.0033] | 1529 | 0.6197 | no |
| property | mean_row | -0.0023 [-0.0062, +0.0020] | 1529 | 0.6197 | no |
| property_new | none | -0.0033 [-0.0081, +0.0020] | 1229 | 0.3318 | no |
| property_new | mean_row | -0.0037 [-0.0085, +0.0012] | 1229 | 0.3318 | no |
| property_category | none | +0.0067 [-0.0017, +0.0167] | 300 | 0.3638 | no |
| property_category | mean_row | +0.0033 [-0.0050, +0.0117] | 300 | 0.5297 | no |
| entailment | none | +0.0067 [-0.0017, +0.0167] | 600 | 0.3758 | no |
| entailment | mean_row | +0.0033 [-0.0050, +0.0117] | 600 | 0.5377 | no |
| paraphrase | none | -0.0052 [-0.0150, +0.0046] | 1529 | 0.6897 | no |
| paraphrase | mean_row | +0.0000 [-0.0085, +0.0092] | 1529 | 1.0000 | no |
| statement_accuracy | none | -0.0013 [-0.0059, +0.0033] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0007 [-0.0039, +0.0052] | 1529 | 1.0000 | no |
| statement_margin | none | -0.0012 [-0.0041, +0.0018] | 1529 | 0.4468 | no |
| statement_margin | mean_row | -0.0018 [-0.0045, +0.0009] | 1529 | 0.4178 | no |
| statement_loss | none | -0.0032 [-0.0058, -0.0006] | 1529 | 0.0280 | yes |
| statement_loss | mean_row | -0.0000 [-0.0025, +0.0025] | 1529 | 0.9885 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.477 → 0.477 (0.477) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.495 | 0.0000 / 0.0000 | 0.477 |
| seen | 100 | 0.485 → 0.485 (0.485) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.453 | 0.0000 / 0.0000 | 0.465 |
| heldout | 100 | 0.470 → 0.470 (0.470) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.537 | 0.0000 / 0.0000 | 0.487 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
