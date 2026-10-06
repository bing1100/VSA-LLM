# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C2-s2`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2073 | 0.2132 | 0.1833 | 0.4917 | 0.3237 | 0.2178 | -1.3413 | 6.3203 |
| none | 0.2083 | 0.2144 | 0.1833 | 0.4817 | 0.3257 | 0.2204 | -1.3418 | 6.3241 |
| mean_row | 0.2080 | 0.2132 | 0.1867 | 0.4883 | 0.3257 | 0.2152 | -1.3437 | 6.3209 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0010 [-0.0052, +0.0029] | 1529 | 1.0000 | no |
| property | mean_row | -0.0007 [-0.0046, +0.0033] | 1529 | 1.0000 | no |
| property_new | none | -0.0012 [-0.0057, +0.0033] | 1229 | 1.0000 | no |
| property_new | mean_row | +0.0000 [-0.0045, +0.0049] | 1229 | 1.0000 | no |
| property_category | none | +0.0000 [-0.0083, +0.0083] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0033 [-0.0117, +0.0050] | 300 | 1.0000 | no |
| entailment | none | +0.0100 [-0.0017, +0.0217] | 600 | 0.2039 | no |
| entailment | mean_row | +0.0033 [-0.0067, +0.0133] | 600 | 0.6617 | no |
| paraphrase | none | -0.0020 [-0.0105, +0.0065] | 1529 | 1.0000 | no |
| paraphrase | mean_row | -0.0020 [-0.0105, +0.0066] | 1529 | 1.0000 | no |
| statement_accuracy | none | -0.0026 [-0.0072, +0.0020] | 1529 | 0.5797 | no |
| statement_accuracy | mean_row | +0.0026 [-0.0026, +0.0078] | 1529 | 0.5797 | no |
| statement_margin | none | +0.0005 [-0.0021, +0.0034] | 1529 | 0.7336 | no |
| statement_margin | mean_row | +0.0024 [-0.0001, +0.0052] | 1529 | 0.1259 | no |
| statement_loss | none | -0.0038 [-0.0061, -0.0016] | 1529 | 0.0060 | yes |
| statement_loss | mean_row | -0.0006 [-0.0028, +0.0015] | 1529 | 0.5757 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.502 → 0.502 (0.502) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.505 | +0.0000 [+0.0000, +0.0000] | 0.464 | 0.0000 / 0.0000 | 0.490 |
| seen | 100 | 0.530 → 0.530 (0.530) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.490 | +0.0000 [+0.0000, +0.0000] | 0.407 | 0.0000 / 0.0000 | 0.470 |
| heldout | 100 | 0.475 → 0.475 (0.475) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.520 | +0.0000 [+0.0000, +0.0000] | 0.520 | 0.0000 / 0.0000 | 0.504 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
