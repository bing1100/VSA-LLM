# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C2-s2`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1991 | 0.2006 | 0.1933 | 0.4783 | 0.3800 | 0.2067 | -1.3600 | 6.5011 |
| none | 0.1978 | 0.1993 | 0.1917 | 0.4717 | 0.3806 | 0.2067 | -1.3576 | 6.4998 |
| mean_row | 0.1972 | 0.1977 | 0.1950 | 0.4817 | 0.3859 | 0.2047 | -1.3589 | 6.5000 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0013 [-0.0020, +0.0049] | 1529 | 0.4838 | no |
| property | mean_row | +0.0020 [-0.0010, +0.0052] | 1529 | 0.4738 | no |
| property_new | none | +0.0012 [-0.0024, +0.0053] | 1229 | 0.6027 | no |
| property_new | mean_row | +0.0028 [-0.0008, +0.0065] | 1229 | 0.2659 | no |
| property_category | none | +0.0017 [-0.0050, +0.0100] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0017 [-0.0083, +0.0033] | 300 | 1.0000 | no |
| entailment | none | +0.0067 [-0.0050, +0.0200] | 600 | 0.7096 | no |
| entailment | mean_row | -0.0033 [-0.0133, +0.0050] | 600 | 0.7096 | no |
| paraphrase | none | -0.0007 [-0.0092, +0.0078] | 1529 | 0.9575 | no |
| paraphrase | mean_row | -0.0059 [-0.0137, +0.0020] | 1529 | 0.3238 | no |
| statement_accuracy | none | +0.0000 [-0.0039, +0.0039] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0020 [-0.0020, +0.0059] | 1529 | 0.8056 | no |
| statement_margin | none | -0.0024 [-0.0054, +0.0005] | 1529 | 0.2319 | no |
| statement_margin | mean_row | -0.0011 [-0.0039, +0.0018] | 1529 | 0.4708 | no |
| statement_loss | none | +0.0013 [-0.0012, +0.0037] | 1529 | 0.5657 | no |
| statement_loss | mean_row | +0.0012 [-0.0012, +0.0034] | 1529 | 0.5657 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.480 → 0.480 (0.480) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.497 | 0.0000 / 0.0000 | 0.486 |
| seen | 100 | 0.450 → 0.450 (0.450) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.495 | 0.0000 / 0.0000 | 0.471 |
| heldout | 100 | 0.510 → 0.510 (0.510) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.490 | +0.0000 [+0.0000, +0.0000] | 0.500 | 0.0000 / 0.0000 | 0.500 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
