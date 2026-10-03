# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2011 | 0.2116 | 0.1583 | 0.4917 | 0.3316 | 0.2269 | -1.5072 | 7.1327 |
| none | 0.2044 | 0.2140 | 0.1650 | 0.4883 | 0.3303 | 0.2256 | -1.5078 | 7.1370 |
| mean_row | 0.2008 | 0.2099 | 0.1633 | 0.4917 | 0.3277 | 0.2243 | -1.5071 | 7.1318 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0033 [-0.0072, +0.0007] | 1529 | 0.2559 | no |
| property | mean_row | +0.0003 [-0.0033, +0.0043] | 1529 | 0.9195 | no |
| property_new | none | -0.0024 [-0.0069, +0.0020] | 1229 | 0.6677 | no |
| property_new | mean_row | +0.0016 [-0.0024, +0.0061] | 1229 | 0.6677 | no |
| property_category | none | -0.0067 [-0.0167, +0.0017] | 300 | 0.4098 | no |
| property_category | mean_row | -0.0050 [-0.0133, +0.0017] | 300 | 0.4098 | no |
| entailment | none | +0.0033 [-0.0050, +0.0133] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0000 [-0.0100, +0.0100] | 600 | 1.0000 | no |
| paraphrase | none | +0.0013 [-0.0085, +0.0118] | 1529 | 0.8526 | no |
| paraphrase | mean_row | +0.0039 [-0.0039, +0.0124] | 1529 | 0.7676 | no |
| statement_accuracy | none | +0.0013 [-0.0033, +0.0059] | 1529 | 0.7476 | no |
| statement_accuracy | mean_row | +0.0026 [-0.0026, +0.0078] | 1529 | 0.7476 | no |
| statement_margin | none | +0.0005 [-0.0023, +0.0034] | 1529 | 1.0000 | no |
| statement_margin | mean_row | -0.0001 [-0.0029, +0.0028] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0042 [-0.0067, -0.0017] | 1529 | 0.0020 | yes |
| statement_loss | mean_row | +0.0009 [-0.0015, +0.0034] | 1529 | 0.4148 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.500 → 0.500 (0.500) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.490 | +0.0000 [+0.0000, +0.0000] | 0.494 | 0.0000 / 0.0000 | 0.495 |
| seen | 100 | 0.500 → 0.500 (0.500) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.463 | 0.0000 / 0.0000 | 0.480 |
| heldout | 100 | 0.500 → 0.500 (0.500) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.500 | +0.0000 [+0.0000, +0.0000] | 0.525 | 0.0000 / 0.0000 | 0.508 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
