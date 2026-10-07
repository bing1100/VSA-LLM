# E9 dimension 3 — zero-shot by ontology editing — C6d seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C6d-s3`, channel `source`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1936 | 0.1973 | 0.1783 | 0.5167 | 0.3309 | 0.2184 | -1.0555 | 4.6951 |
| none | 0.1952 | 0.1989 | 0.1800 | 0.5100 | 0.3342 | 0.2158 | -1.0574 | 4.6948 |
| mean_row | 0.1952 | 0.1998 | 0.1767 | 0.5167 | 0.3362 | 0.2139 | -1.0551 | 4.6906 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0016 [-0.0052, +0.0020] | 1529 | 0.8776 | no |
| property | mean_row | -0.0016 [-0.0069, +0.0033] | 1529 | 0.8776 | no |
| property_new | none | -0.0016 [-0.0053, +0.0016] | 1229 | 0.7876 | no |
| property_new | mean_row | -0.0024 [-0.0077, +0.0025] | 1229 | 0.7876 | no |
| property_category | none | -0.0017 [-0.0133, +0.0083] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0017 [-0.0117, +0.0150] | 300 | 1.0000 | no |
| entailment | none | +0.0067 [-0.0050, +0.0200] | 600 | 0.7556 | no |
| entailment | mean_row | +0.0000 [-0.0150, +0.0150] | 600 | 1.0000 | no |
| paraphrase | none | -0.0033 [-0.0124, +0.0052] | 1529 | 0.7456 | no |
| paraphrase | mean_row | -0.0052 [-0.0164, +0.0052] | 1529 | 0.7456 | no |
| statement_accuracy | none | +0.0026 [-0.0013, +0.0072] | 1529 | 0.3218 | no |
| statement_accuracy | mean_row | +0.0046 [-0.0007, +0.0098] | 1529 | 0.2399 | no |
| statement_margin | none | +0.0019 [-0.0010, +0.0046] | 1529 | 0.3458 | no |
| statement_margin | mean_row | -0.0004 [-0.0049, +0.0043] | 1529 | 0.9035 | no |
| statement_loss | none | +0.0003 [-0.0020, +0.0026] | 1529 | 0.8316 | no |
| statement_loss | mean_row | +0.0045 [-0.0016, +0.0105] | 1529 | 0.2919 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.440 → 0.440 (0.440) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.450 | +0.0000 [+0.0000, +0.0000] | 0.541 | 0.0000 / 0.0000 | 0.473 |
| seen | 100 | 0.410 → 0.410 (0.410) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.420 | +0.0000 [+0.0000, +0.0000] | 0.535 | 0.0000 / 0.0000 | 0.448 |
| heldout | 100 | 0.470 → 0.470 (0.470) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.547 | 0.0000 / 0.0000 | 0.497 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
