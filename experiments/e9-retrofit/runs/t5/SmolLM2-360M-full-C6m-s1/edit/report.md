# E9 dimension 3 — zero-shot by ontology editing — C6m seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C6m-s1`, channel `source`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1920 | 0.1965 | 0.1733 | 0.5150 | 0.3277 | 0.2250 | -1.0269 | 4.5892 |
| none | 0.1936 | 0.1985 | 0.1733 | 0.5017 | 0.3309 | 0.2256 | -1.0275 | 4.5899 |
| mean_row | 0.1933 | 0.1977 | 0.1750 | 0.5017 | 0.3303 | 0.2237 | -1.0248 | 4.5877 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0016 [-0.0052, +0.0016] | 1529 | 0.7956 | no |
| property | mean_row | -0.0013 [-0.0049, +0.0023] | 1529 | 0.7956 | no |
| property_new | none | -0.0020 [-0.0061, +0.0020] | 1229 | 0.6797 | no |
| property_new | mean_row | -0.0012 [-0.0057, +0.0028] | 1229 | 0.6797 | no |
| property_category | none | +0.0000 [-0.0083, +0.0100] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0017 [-0.0083, +0.0050] | 300 | 1.0000 | no |
| entailment | none | +0.0133 [+0.0033, +0.0250] | 600 | 0.0250 | yes |
| entailment | mean_row | +0.0133 [+0.0033, +0.0233] | 600 | 0.0140 | yes |
| paraphrase | none | -0.0033 [-0.0118, +0.0052] | 1529 | 1.0000 | no |
| paraphrase | mean_row | -0.0026 [-0.0105, +0.0052] | 1529 | 1.0000 | no |
| statement_accuracy | none | -0.0007 [-0.0046, +0.0033] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0013 [-0.0033, +0.0059] | 1529 | 1.0000 | no |
| statement_margin | none | +0.0006 [-0.0016, +0.0028] | 1529 | 0.6127 | no |
| statement_margin | mean_row | -0.0021 [-0.0044, +0.0002] | 1529 | 0.1599 | no |
| statement_loss | none | -0.0007 [-0.0025, +0.0010] | 1529 | 0.4678 | no |
| statement_loss | mean_row | +0.0015 [-0.0003, +0.0034] | 1529 | 0.2139 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.458 → 0.458 (0.458) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.465 | +0.0000 [+0.0000, +0.0000] | 0.534 | 0.0000 / 0.0000 | 0.483 |
| seen | 100 | 0.460 → 0.460 (0.460) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.532 | 0.0000 / 0.0000 | 0.482 |
| heldout | 100 | 0.455 → 0.455 (0.455) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.535 | 0.0000 / 0.0000 | 0.484 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
