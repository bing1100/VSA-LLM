# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C2-s2`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1861 | 0.1977 | 0.1383 | 0.4600 | 0.3859 | 0.2256 | -1.0428 | 4.3334 |
| none | 0.1861 | 0.1985 | 0.1350 | 0.4567 | 0.3872 | 0.2243 | -1.0445 | 4.3364 |
| mean_row | 0.1828 | 0.1941 | 0.1367 | 0.4650 | 0.3846 | 0.2269 | -1.0427 | 4.3328 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0000 [-0.0033, +0.0036] | 1529 | 1.0000 | no |
| property | mean_row | +0.0033 [+0.0000, +0.0069] | 1529 | 0.1419 | no |
| property_new | none | -0.0008 [-0.0045, +0.0033] | 1229 | 0.7686 | no |
| property_new | mean_row | +0.0037 [-0.0004, +0.0077] | 1229 | 0.1739 | no |
| property_category | none | +0.0033 [-0.0017, +0.0100] | 300 | 0.8876 | no |
| property_category | mean_row | +0.0017 [-0.0033, +0.0067] | 300 | 0.8876 | no |
| entailment | none | +0.0033 [-0.0067, +0.0133] | 600 | 0.8856 | no |
| entailment | mean_row | -0.0050 [-0.0167, +0.0067] | 600 | 0.8856 | no |
| paraphrase | none | -0.0013 [-0.0098, +0.0072] | 1529 | 1.0000 | no |
| paraphrase | mean_row | +0.0013 [-0.0078, +0.0098] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0013 [-0.0033, +0.0059] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | -0.0013 [-0.0065, +0.0033] | 1529 | 1.0000 | no |
| statement_margin | none | +0.0018 [-0.0008, +0.0044] | 1529 | 0.3758 | no |
| statement_margin | mean_row | -0.0001 [-0.0025, +0.0024] | 1529 | 0.9385 | no |
| statement_loss | none | -0.0030 [-0.0050, -0.0010] | 1529 | 0.0120 | yes |
| statement_loss | mean_row | +0.0006 [-0.0014, +0.0026] | 1529 | 0.5517 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.465 → 0.465 (0.465) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.435 | +0.0000 [+0.0000, +0.0000] | 0.583 | 0.0000 / 0.0000 | 0.487 |
| seen | 100 | 0.485 → 0.485 (0.485) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.590 | 0.0000 / 0.0000 | 0.510 |
| heldout | 100 | 0.445 → 0.445 (0.445) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.400 | +0.0000 [+0.0000, +0.0000] | 0.575 | 0.0000 / 0.0000 | 0.463 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
