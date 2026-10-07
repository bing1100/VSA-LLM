# E9 dimension 3 — zero-shot by ontology editing — C6m seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C6m-s3`, channel `source`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1942 | 0.1989 | 0.1750 | 0.5067 | 0.3296 | 0.2237 | -1.0291 | 4.6000 |
| none | 0.1936 | 0.1989 | 0.1717 | 0.5050 | 0.3264 | 0.2250 | -1.0290 | 4.5990 |
| mean_row | 0.1946 | 0.2010 | 0.1683 | 0.5033 | 0.3309 | 0.2198 | -1.0287 | 4.5988 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0007 [-0.0029, +0.0043] | 1529 | 1.0000 | no |
| property | mean_row | -0.0003 [-0.0039, +0.0033] | 1529 | 1.0000 | no |
| property_new | none | +0.0000 [-0.0037, +0.0037] | 1229 | 1.0000 | no |
| property_new | mean_row | -0.0020 [-0.0057, +0.0012] | 1229 | 0.6097 | no |
| property_category | none | +0.0033 [-0.0067, +0.0133] | 300 | 0.6587 | no |
| property_category | mean_row | +0.0067 [-0.0034, +0.0183] | 300 | 0.6157 | no |
| entailment | none | +0.0017 [-0.0067, +0.0100] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0033 [-0.0050, +0.0133] | 600 | 1.0000 | no |
| paraphrase | none | +0.0033 [-0.0046, +0.0111] | 1529 | 0.9235 | no |
| paraphrase | mean_row | -0.0013 [-0.0085, +0.0059] | 1529 | 0.9235 | no |
| statement_accuracy | none | -0.0013 [-0.0052, +0.0026] | 1529 | 0.6487 | no |
| statement_accuracy | mean_row | +0.0039 [+0.0000, +0.0085] | 1529 | 0.1419 | no |
| statement_margin | none | -0.0001 [-0.0024, +0.0023] | 1529 | 1.0000 | no |
| statement_margin | mean_row | -0.0004 [-0.0026, +0.0019] | 1529 | 1.0000 | no |
| statement_loss | none | +0.0010 [-0.0008, +0.0029] | 1529 | 0.3318 | no |
| statement_loss | mean_row | +0.0012 [-0.0006, +0.0029] | 1529 | 0.3318 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.463 → 0.463 (0.463) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.465 | +0.0000 [+0.0000, +0.0000] | 0.531 | 0.0000 / 0.0000 | 0.484 |
| seen | 100 | 0.465 → 0.465 (0.465) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.532 | 0.0000 / 0.0000 | 0.484 |
| heldout | 100 | 0.460 → 0.460 (0.460) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.530 | 0.0000 / 0.0000 | 0.485 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
