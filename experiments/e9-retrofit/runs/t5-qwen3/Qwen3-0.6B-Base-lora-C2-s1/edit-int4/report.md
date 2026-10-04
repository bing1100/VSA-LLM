# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2054 | 0.2071 | 0.1983 | 0.4967 | 0.3080 | 0.2230 | -1.3291 | 6.3392 |
| none | 0.2083 | 0.2103 | 0.2000 | 0.4950 | 0.3100 | 0.2198 | -1.3292 | 6.3399 |
| mean_row | 0.2077 | 0.2095 | 0.2000 | 0.4900 | 0.3094 | 0.2211 | -1.3284 | 6.3379 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0029 [-0.0072, +0.0010] | 1529 | 0.3798 | no |
| property | mean_row | -0.0023 [-0.0065, +0.0020] | 1529 | 0.3798 | no |
| property_new | none | -0.0033 [-0.0085, +0.0016] | 1229 | 0.4458 | no |
| property_new | mean_row | -0.0024 [-0.0077, +0.0024] | 1229 | 0.4458 | no |
| property_category | none | -0.0017 [-0.0083, +0.0050] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0017 [-0.0100, +0.0067] | 300 | 1.0000 | no |
| entailment | none | +0.0017 [-0.0100, +0.0133] | 600 | 0.8976 | no |
| entailment | mean_row | +0.0067 [-0.0050, +0.0183] | 600 | 0.5837 | no |
| paraphrase | none | -0.0020 [-0.0105, +0.0065] | 1529 | 1.0000 | no |
| paraphrase | mean_row | -0.0013 [-0.0098, +0.0072] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0033 [-0.0007, +0.0078] | 1529 | 0.3458 | no |
| statement_accuracy | mean_row | +0.0020 [-0.0026, +0.0065] | 1529 | 0.4798 | no |
| statement_margin | none | +0.0001 [-0.0029, +0.0032] | 1529 | 1.0000 | no |
| statement_margin | mean_row | -0.0007 [-0.0035, +0.0022] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0008 [-0.0032, +0.0016] | 1529 | 0.5777 | no |
| statement_loss | mean_row | +0.0013 [-0.0010, +0.0036] | 1529 | 0.5777 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.510 → 0.510 (0.510) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.510 | +0.0000 [+0.0000, +0.0000] | 0.512 | 0.0000 / 0.0000 | 0.511 |
| seen | 100 | 0.540 → 0.540 (0.540) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.550 | +0.0000 [+0.0000, +0.0000] | 0.470 | 0.0000 / 0.0000 | 0.517 |
| heldout | 100 | 0.480 → 0.480 (0.480) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.555 | 0.0000 / 0.0000 | 0.499 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
