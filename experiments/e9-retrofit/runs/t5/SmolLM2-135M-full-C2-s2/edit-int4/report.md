# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C2-s2`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1982 | 0.2083 | 0.1567 | 0.4967 | 0.3322 | 0.2237 | -1.5250 | 7.1518 |
| none | 0.1959 | 0.2059 | 0.1550 | 0.4917 | 0.3309 | 0.2237 | -1.5257 | 7.1550 |
| mean_row | 0.1978 | 0.2071 | 0.1600 | 0.4917 | 0.3277 | 0.2263 | -1.5234 | 7.1507 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0023 [-0.0016, +0.0062] | 1529 | 0.6497 | no |
| property | mean_row | +0.0003 [-0.0043, +0.0043] | 1529 | 0.9285 | no |
| property_new | none | +0.0024 [-0.0024, +0.0069] | 1229 | 0.7096 | no |
| property_new | mean_row | +0.0012 [-0.0037, +0.0061] | 1229 | 0.7096 | no |
| property_category | none | +0.0017 [-0.0050, +0.0083] | 300 | 0.9855 | no |
| property_category | mean_row | -0.0033 [-0.0100, +0.0033] | 300 | 0.9855 | no |
| entailment | none | +0.0050 [-0.0083, +0.0183] | 600 | 0.8016 | no |
| entailment | mean_row | +0.0050 [-0.0050, +0.0150] | 600 | 0.8016 | no |
| paraphrase | none | +0.0013 [-0.0085, +0.0111] | 1529 | 0.8336 | no |
| paraphrase | mean_row | +0.0046 [-0.0052, +0.0144] | 1529 | 0.7336 | no |
| statement_accuracy | none | +0.0000 [-0.0039, +0.0033] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | -0.0026 [-0.0072, +0.0020] | 1529 | 0.5977 | no |
| statement_margin | none | +0.0006 [-0.0023, +0.0038] | 1529 | 0.6717 | no |
| statement_margin | mean_row | -0.0016 [-0.0045, +0.0013] | 1529 | 0.5397 | no |
| statement_loss | none | -0.0032 [-0.0060, -0.0007] | 1529 | 0.0340 | yes |
| statement_loss | mean_row | +0.0012 [-0.0014, +0.0036] | 1529 | 0.3768 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.505 → 0.505 (0.505) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.490 | +0.0000 [+0.0000, +0.0000] | 0.490 | 0.0000 / 0.0000 | 0.495 |
| seen | 100 | 0.515 → 0.515 (0.515) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.455 | 0.0000 / 0.0000 | 0.479 |
| heldout | 100 | 0.495 → 0.495 (0.495) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.510 | +0.0000 [+0.0000, +0.0000] | 0.525 | 0.0000 / 0.0000 | 0.510 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
