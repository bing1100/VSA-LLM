# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2024 | 0.2067 | 0.1850 | 0.4717 | 0.3165 | 0.2139 | -1.2987 | 6.3461 |
| none | 0.2011 | 0.2063 | 0.1800 | 0.4667 | 0.3120 | 0.2132 | -1.2994 | 6.3468 |
| mean_row | 0.2014 | 0.2059 | 0.1833 | 0.4650 | 0.3205 | 0.2158 | -1.3006 | 6.3479 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0013 [-0.0026, +0.0052] | 1529 | 1.0000 | no |
| property | mean_row | +0.0010 [-0.0026, +0.0049] | 1529 | 1.0000 | no |
| property_new | none | +0.0004 [-0.0037, +0.0045] | 1229 | 1.0000 | no |
| property_new | mean_row | +0.0008 [-0.0037, +0.0053] | 1229 | 1.0000 | no |
| property_category | none | +0.0050 [-0.0067, +0.0150] | 300 | 0.8976 | no |
| property_category | mean_row | +0.0017 [-0.0067, +0.0100] | 300 | 0.8976 | no |
| entailment | none | +0.0050 [-0.0050, +0.0167] | 600 | 0.5357 | no |
| entailment | mean_row | +0.0067 [-0.0033, +0.0183] | 600 | 0.5357 | no |
| paraphrase | none | +0.0046 [-0.0039, +0.0131] | 1529 | 0.6797 | no |
| paraphrase | mean_row | -0.0039 [-0.0124, +0.0039] | 1529 | 0.6797 | no |
| statement_accuracy | none | +0.0007 [-0.0020, +0.0033] | 1529 | 0.8406 | no |
| statement_accuracy | mean_row | -0.0020 [-0.0059, +0.0020] | 1529 | 0.8096 | no |
| statement_margin | none | +0.0007 [-0.0025, +0.0038] | 1529 | 0.6707 | no |
| statement_margin | mean_row | +0.0019 [-0.0011, +0.0049] | 1529 | 0.4638 | no |
| statement_loss | none | -0.0007 [-0.0033, +0.0018] | 1529 | 0.6337 | no |
| statement_loss | mean_row | -0.0018 [-0.0044, +0.0006] | 1529 | 0.3078 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.495 → 0.495 (0.495) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.485 | +0.0000 [+0.0000, +0.0000] | 0.486 | 0.0000 / 0.0000 | 0.489 |
| seen | 100 | 0.485 → 0.485 (0.485) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.450 | +0.0000 [+0.0000, +0.0000] | 0.497 | 0.0000 / 0.0000 | 0.477 |
| heldout | 100 | 0.505 → 0.505 (0.505) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.520 | +0.0000 [+0.0000, +0.0000] | 0.475 | 0.0000 / 0.0000 | 0.499 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
