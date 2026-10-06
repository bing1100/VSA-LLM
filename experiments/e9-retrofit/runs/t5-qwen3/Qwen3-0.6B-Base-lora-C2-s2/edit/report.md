# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C2-s2`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1949 | 0.1977 | 0.1833 | 0.4917 | 0.3466 | 0.2289 | -1.1140 | 4.5985 |
| none | 0.1916 | 0.1949 | 0.1783 | 0.4950 | 0.3434 | 0.2296 | -1.1143 | 4.6008 |
| mean_row | 0.1939 | 0.1965 | 0.1833 | 0.4933 | 0.3492 | 0.2276 | -1.1134 | 4.5971 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0033 [+0.0000, +0.0069] | 1529 | 0.1479 | no |
| property | mean_row | +0.0010 [-0.0029, +0.0049] | 1529 | 0.6777 | no |
| property_new | none | +0.0028 [-0.0012, +0.0069] | 1229 | 0.3818 | no |
| property_new | mean_row | +0.0012 [-0.0033, +0.0057] | 1229 | 0.6617 | no |
| property_category | none | +0.0050 [-0.0017, +0.0133] | 300 | 0.4878 | no |
| property_category | mean_row | +0.0000 [-0.0083, +0.0083] | 300 | 1.0000 | no |
| entailment | none | -0.0033 [-0.0100, +0.0033] | 600 | 0.8976 | no |
| entailment | mean_row | -0.0017 [-0.0133, +0.0100] | 600 | 0.8976 | no |
| paraphrase | none | +0.0033 [-0.0059, +0.0124] | 1529 | 1.0000 | no |
| paraphrase | mean_row | -0.0026 [-0.0111, +0.0052] | 1529 | 1.0000 | no |
| statement_accuracy | none | -0.0007 [-0.0046, +0.0033] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0013 [-0.0033, +0.0059] | 1529 | 1.0000 | no |
| statement_margin | none | +0.0003 [-0.0021, +0.0028] | 1529 | 1.0000 | no |
| statement_margin | mean_row | -0.0006 [-0.0032, +0.0019] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0023 [-0.0042, -0.0004] | 1529 | 0.0400 | yes |
| statement_loss | mean_row | +0.0013 [-0.0006, +0.0032] | 1529 | 0.1639 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.450 → 0.450 (0.450) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.555 | 0.0000 / 0.0000 | 0.488 |
| seen | 100 | 0.490 → 0.490 (0.490) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.490 | +0.0000 [+0.0000, +0.0000] | 0.532 | 0.0000 / 0.0000 | 0.503 |
| heldout | 100 | 0.410 → 0.410 (0.410) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.450 | +0.0000 [+0.0000, +0.0000] | 0.578 | 0.0000 / 0.0000 | 0.469 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
