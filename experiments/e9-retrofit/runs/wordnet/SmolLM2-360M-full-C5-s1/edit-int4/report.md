# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3077 | 0.1683 | 0.4008 | 0.6275 | 0.5395 | 0.2373 | -1.7928 | 8.7293 |
| none | 0.3062 | 0.1696 | 0.3975 | 0.6258 | 0.5395 | 0.2378 | -1.7905 | 8.7269 |
| mean_row | 0.3062 | 0.1708 | 0.3967 | 0.6208 | 0.5395 | 0.2373 | -1.7912 | 8.7275 |
| random_frame | 0.3052 | 0.1683 | 0.3967 | 0.6217 | 0.5415 | 0.2358 | -1.7929 | 8.7302 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0015 [-0.0030, +0.0060] | 1001 | 1.0000 | no |
| property | mean_row | +0.0015 [-0.0035, +0.0065] | 1001 | 1.0000 | no |
| property | random_frame | +0.0025 [-0.0020, +0.0070] | 1001 | 0.8576 | no |
| property_new | none | -0.0012 [-0.0063, +0.0037] | 401 | 1.0000 | no |
| property_new | mean_row | -0.0025 [-0.0075, +0.0025] | 401 | 1.0000 | no |
| property_new | random_frame | +0.0000 [-0.0050, +0.0050] | 401 | 1.0000 | no |
| property_category | none | +0.0033 [-0.0025, +0.0100] | 600 | 0.7226 | no |
| property_category | mean_row | +0.0042 [-0.0025, +0.0109] | 600 | 0.7226 | no |
| property_category | random_frame | +0.0042 [-0.0017, +0.0108] | 600 | 0.7226 | no |
| entailment | none | +0.0017 [-0.0050, +0.0083] | 600 | 0.7126 | no |
| entailment | mean_row | +0.0067 [+0.0008, +0.0133] | 600 | 0.1229 | no |
| entailment | random_frame | +0.0058 [+0.0000, +0.0125] | 600 | 0.1639 | no |
| paraphrase | none | +0.0000 [-0.0110, +0.0110] | 1001 | 1.0000 | no |
| paraphrase | mean_row | +0.0000 [-0.0130, +0.0120] | 1001 | 1.0000 | no |
| paraphrase | random_frame | -0.0020 [-0.0140, +0.0100] | 1001 | 1.0000 | no |
| statement_accuracy | none | -0.0005 [-0.0035, +0.0025] | 1001 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0000 [-0.0030, +0.0030] | 1001 | 1.0000 | no |
| statement_accuracy | random_frame | +0.0015 [-0.0015, +0.0050] | 1001 | 1.0000 | no |
| statement_margin | none | -0.0023 [-0.0047, +0.0001] | 1001 | 0.1949 | no |
| statement_margin | mean_row | -0.0016 [-0.0039, +0.0008] | 1001 | 0.3478 | no |
| statement_margin | random_frame | +0.0001 [-0.0028, +0.0028] | 1001 | 0.9485 | no |
| statement_loss | none | +0.0024 [+0.0004, +0.0045] | 1001 | 0.0660 | no |
| statement_loss | mean_row | +0.0018 [-0.0002, +0.0037] | 1001 | 0.1799 | no |
| statement_loss | random_frame | -0.0009 [-0.0031, +0.0015] | 1001 | 0.4678 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.198 → 0.193 (0.193) | +0.0039 [-0.0106, +0.0183] | +0.0039 [-0.0111, +0.0204] | 0.193 | -0.0115 [-0.0244, +0.0007] | 0.785 | 0.0001 / 0.0440 | 0.257 |
| seen | 100 | 0.205 → 0.210 (0.210) | +0.0112 [-0.0087, +0.0301] | +0.0046 [-0.0183, +0.0263] | 0.205 | -0.0190 [-0.0389, +0.0007] | 0.770 | 0.0002 / 0.0440 | 0.274 |
| heldout | 100 | 0.190 → 0.175 (0.175) | -0.0034 [-0.0228, +0.0168] | +0.0032 [-0.0178, +0.0276] | 0.180 | -0.0041 [-0.0206, +0.0121] | 0.800 | 0.0000 / 0.0000 | 0.240 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
