# E9 dimension 3 — zero-shot by ontology editing — C2 seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C2-s3`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2112 | 0.2059 | 0.2333 | 0.5150 | 0.3499 | 0.2191 | -1.2757 | 5.5314 |
| none | 0.2109 | 0.2059 | 0.2317 | 0.5167 | 0.3473 | 0.2184 | -1.2772 | 5.5331 |
| mean_row | 0.2126 | 0.2067 | 0.2367 | 0.5100 | 0.3499 | 0.2178 | -1.2765 | 5.5329 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0003 [-0.0033, +0.0039] | 1529 | 1.0000 | no |
| property | mean_row | -0.0013 [-0.0049, +0.0026] | 1529 | 1.0000 | no |
| property_new | none | +0.0000 [-0.0041, +0.0041] | 1229 | 1.0000 | no |
| property_new | mean_row | -0.0008 [-0.0049, +0.0033] | 1229 | 1.0000 | no |
| property_category | none | +0.0017 [-0.0067, +0.0100] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0033 [-0.0133, +0.0067] | 300 | 1.0000 | no |
| entailment | none | -0.0017 [-0.0117, +0.0083] | 600 | 0.8696 | no |
| entailment | mean_row | +0.0050 [-0.0050, +0.0167] | 600 | 0.8696 | no |
| paraphrase | none | +0.0026 [-0.0065, +0.0118] | 1529 | 1.0000 | no |
| paraphrase | mean_row | +0.0000 [-0.0079, +0.0085] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0007 [-0.0039, +0.0052] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0013 [-0.0033, +0.0059] | 1529 | 1.0000 | no |
| statement_margin | none | +0.0016 [-0.0010, +0.0041] | 1529 | 0.4758 | no |
| statement_margin | mean_row | +0.0008 [-0.0017, +0.0035] | 1529 | 0.5267 | no |
| statement_loss | none | -0.0017 [-0.0040, +0.0003] | 1529 | 0.2159 | no |
| statement_loss | mean_row | -0.0015 [-0.0036, +0.0005] | 1529 | 0.2159 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.492 → 0.492 (0.492) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.475 | +0.0000 [+0.0000, +0.0000] | 0.524 | 0.0000 / 0.0000 | 0.496 |
| seen | 100 | 0.515 → 0.515 (0.515) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.500 | +0.0000 [+0.0000, +0.0000] | 0.475 | 0.0000 / 0.0000 | 0.496 |
| heldout | 100 | 0.470 → 0.470 (0.470) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.450 | +0.0000 [+0.0000, +0.0000] | 0.573 | 0.0000 / 0.0000 | 0.492 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
