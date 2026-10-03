# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2093 | 0.2059 | 0.2233 | 0.5000 | 0.3336 | 0.2191 | -1.2706 | 5.4929 |
| none | 0.2077 | 0.2050 | 0.2183 | 0.4917 | 0.3309 | 0.2165 | -1.2714 | 5.4944 |
| mean_row | 0.2122 | 0.2087 | 0.2267 | 0.4983 | 0.3329 | 0.2211 | -1.2700 | 5.4922 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0016 [-0.0029, +0.0062] | 1529 | 0.5067 | no |
| property | mean_row | -0.0029 [-0.0072, +0.0013] | 1529 | 0.4218 | no |
| property_new | none | +0.0008 [-0.0037, +0.0053] | 1229 | 0.8286 | no |
| property_new | mean_row | -0.0028 [-0.0069, +0.0012] | 1229 | 0.3698 | no |
| property_category | none | +0.0050 [-0.0067, +0.0167] | 300 | 0.9695 | no |
| property_category | mean_row | -0.0033 [-0.0150, +0.0083] | 300 | 0.9695 | no |
| entailment | none | +0.0083 [-0.0000, +0.0183] | 600 | 0.2219 | no |
| entailment | mean_row | +0.0017 [-0.0067, +0.0100] | 600 | 0.8726 | no |
| paraphrase | none | +0.0026 [-0.0065, +0.0118] | 1529 | 1.0000 | no |
| paraphrase | mean_row | +0.0007 [-0.0078, +0.0092] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0026 [-0.0007, +0.0065] | 1529 | 0.3758 | no |
| statement_accuracy | mean_row | -0.0020 [-0.0059, +0.0020] | 1529 | 0.4008 | no |
| statement_margin | none | +0.0008 [-0.0018, +0.0034] | 1529 | 1.0000 | no |
| statement_margin | mean_row | -0.0006 [-0.0031, +0.0018] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0014 [-0.0033, +0.0007] | 1529 | 0.3538 | no |
| statement_loss | mean_row | +0.0008 [-0.0013, +0.0029] | 1529 | 0.4238 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.482 → 0.482 (0.482) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.475 | +0.0000 [+0.0000, +0.0000] | 0.520 | 0.0000 / 0.0000 | 0.492 |
| seen | 100 | 0.500 → 0.500 (0.500) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.485 | 0.0000 / 0.0000 | 0.488 |
| heldout | 100 | 0.465 → 0.465 (0.465) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.555 | 0.0000 / 0.0000 | 0.493 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
