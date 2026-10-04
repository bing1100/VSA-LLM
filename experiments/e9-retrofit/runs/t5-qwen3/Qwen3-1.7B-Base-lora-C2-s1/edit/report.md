# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1828 | 0.1977 | 0.1217 | 0.4350 | 0.3859 | 0.2243 | -1.0614 | 4.3900 |
| none | 0.1854 | 0.2006 | 0.1233 | 0.4383 | 0.3957 | 0.2243 | -1.0612 | 4.3908 |
| mean_row | 0.1821 | 0.1965 | 0.1233 | 0.4417 | 0.3859 | 0.2250 | -1.0623 | 4.3905 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0026 [-0.0062, +0.0007] | 1529 | 0.3278 | no |
| property | mean_row | +0.0007 [-0.0029, +0.0046] | 1529 | 0.7776 | no |
| property_new | none | -0.0028 [-0.0069, +0.0012] | 1229 | 0.3418 | no |
| property_new | mean_row | +0.0012 [-0.0037, +0.0057] | 1229 | 0.6767 | no |
| property_category | none | -0.0017 [-0.0083, +0.0033] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0017 [-0.0067, +0.0033] | 300 | 1.0000 | no |
| entailment | none | -0.0033 [-0.0133, +0.0067] | 600 | 0.7276 | no |
| entailment | mean_row | -0.0067 [-0.0184, +0.0067] | 600 | 0.7276 | no |
| paraphrase | none | -0.0098 [-0.0196, -0.0007] | 1529 | 0.0640 | no |
| paraphrase | mean_row | +0.0000 [-0.0092, +0.0098] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0000 [-0.0039, +0.0039] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | -0.0007 [-0.0052, +0.0033] | 1529 | 1.0000 | no |
| statement_margin | none | -0.0002 [-0.0027, +0.0023] | 1529 | 0.9435 | no |
| statement_margin | mean_row | +0.0009 [-0.0017, +0.0032] | 1529 | 0.9435 | no |
| statement_loss | none | -0.0008 [-0.0029, +0.0012] | 1529 | 0.8296 | no |
| statement_loss | mean_row | -0.0005 [-0.0025, +0.0016] | 1529 | 0.8296 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.435 → 0.435 (0.435) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.455 | +0.0000 [+0.0000, +0.0000] | 0.586 | 0.0000 / 0.0000 | 0.484 |
| seen | 100 | 0.465 → 0.465 (0.465) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.520 | +0.0000 [+0.0000, +0.0000] | 0.575 | 0.0000 / 0.0000 | 0.516 |
| heldout | 100 | 0.405 → 0.405 (0.405) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.390 | +0.0000 [+0.0000, +0.0000] | 0.598 | 0.0000 / 0.0000 | 0.447 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
