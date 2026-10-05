# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C2-s2`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2122 | 0.2079 | 0.2300 | 0.5033 | 0.3362 | 0.2191 | -1.2980 | 5.5588 |
| none | 0.2077 | 0.2022 | 0.2300 | 0.5083 | 0.3349 | 0.2204 | -1.2974 | 5.5597 |
| mean_row | 0.2067 | 0.2014 | 0.2283 | 0.5000 | 0.3342 | 0.2171 | -1.2986 | 5.5589 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0046 [+0.0013, +0.0078] | 1529 | 0.0180 | yes |
| property | mean_row | +0.0056 [+0.0013, +0.0098] | 1529 | 0.0190 | yes |
| property_new | none | +0.0057 [+0.0024, +0.0098] | 1229 | 0.0040 | yes |
| property_new | mean_row | +0.0065 [+0.0020, +0.0114] | 1229 | 0.0040 | yes |
| property_category | none | +0.0000 [-0.0083, +0.0083] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0017 [-0.0083, +0.0117] | 300 | 1.0000 | no |
| entailment | none | -0.0050 [-0.0133, +0.0033] | 600 | 0.6357 | no |
| entailment | mean_row | +0.0033 [-0.0067, +0.0133] | 600 | 0.6437 | no |
| paraphrase | none | +0.0013 [-0.0072, +0.0098] | 1529 | 1.0000 | no |
| paraphrase | mean_row | +0.0020 [-0.0065, +0.0111] | 1529 | 1.0000 | no |
| statement_accuracy | none | -0.0013 [-0.0052, +0.0026] | 1529 | 0.9615 | no |
| statement_accuracy | mean_row | +0.0020 [-0.0026, +0.0065] | 1529 | 0.9615 | no |
| statement_margin | none | -0.0007 [-0.0033, +0.0020] | 1529 | 1.0000 | no |
| statement_margin | mean_row | +0.0005 [-0.0020, +0.0031] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0008 [-0.0030, +0.0014] | 1529 | 0.8996 | no |
| statement_loss | mean_row | -0.0001 [-0.0021, +0.0019] | 1529 | 0.9335 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.470 → 0.470 (0.470) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.516 | 0.0000 / 0.0000 | 0.481 |
| seen | 100 | 0.475 → 0.475 (0.475) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.485 | 0.0000 / 0.0000 | 0.473 |
| heldout | 100 | 0.465 → 0.465 (0.465) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.547 | 0.0000 / 0.0000 | 0.488 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
