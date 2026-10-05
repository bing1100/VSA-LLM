# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1976 | 0.1873 | 0.2117 | 0.4550 | 0.2827 | 0.2841 | -1.0442 | 3.6717 |
| none | 0.1927 | 0.1800 | 0.2100 | 0.4533 | 0.2841 | 0.2827 | -1.0558 | 3.6903 |
| mean_row | 0.1962 | 0.1813 | 0.2167 | 0.4550 | 0.2897 | 0.2827 | -1.0507 | 3.6789 |
| random_frame | 0.1934 | 0.1825 | 0.2083 | 0.4517 | 0.2869 | 0.2771 | -1.0525 | 3.6754 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0049 [-0.0063, +0.0162] | 711 | 1.0000 | no |
| property | mean_row | +0.0014 [-0.0077, +0.0105] | 711 | 1.0000 | no |
| property | random_frame | +0.0042 [-0.0070, +0.0162] | 711 | 1.0000 | no |
| property_new | none | +0.0073 [-0.0061, +0.0219] | 411 | 1.0000 | no |
| property_new | mean_row | +0.0061 [-0.0061, +0.0182] | 411 | 1.0000 | no |
| property_new | random_frame | +0.0049 [-0.0085, +0.0195] | 411 | 1.0000 | no |
| property_category | none | +0.0017 [-0.0167, +0.0200] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0050 [-0.0200, +0.0100] | 300 | 1.0000 | no |
| property_category | random_frame | +0.0033 [-0.0150, +0.0217] | 300 | 1.0000 | no |
| entailment | none | +0.0017 [-0.0167, +0.0200] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0000 [-0.0150, +0.0167] | 600 | 1.0000 | no |
| entailment | random_frame | +0.0033 [-0.0150, +0.0233] | 600 | 1.0000 | no |
| paraphrase | none | -0.0014 [-0.0239, +0.0211] | 711 | 1.0000 | no |
| paraphrase | mean_row | -0.0070 [-0.0281, +0.0141] | 711 | 1.0000 | no |
| paraphrase | random_frame | -0.0042 [-0.0281, +0.0197] | 711 | 1.0000 | no |
| statement_accuracy | none | +0.0014 [-0.0057, +0.0098] | 711 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0014 [-0.0084, +0.0113] | 711 | 1.0000 | no |
| statement_accuracy | random_frame | +0.0070 [-0.0014, +0.0155] | 711 | 0.3688 | no |
| statement_margin | none | +0.0116 [+0.0053, +0.0177] | 711 | 0.0030 | yes |
| statement_margin | mean_row | +0.0064 [+0.0011, +0.0118] | 711 | 0.0200 | yes |
| statement_margin | random_frame | +0.0083 [+0.0015, +0.0146] | 711 | 0.0200 | yes |
| statement_loss | none | -0.0186 [-0.0251, -0.0122] | 711 | 0.0030 | yes |
| statement_loss | mean_row | -0.0072 [-0.0128, -0.0013] | 711 | 0.0380 | yes |
| statement_loss | random_frame | -0.0037 [-0.0111, +0.0039] | 711 | 0.3228 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.323 → 0.323 (0.320) | +0.0161 [-0.0527, +0.0801] | -0.0096 [-0.0754, +0.0490] | 0.320 | +0.0007 [-0.0734, +0.0709] | 0.728 | 0.0388 / 2.3465 | 0.395 |
| seen | 100 | 0.335 → 0.335 (0.335) | +0.0764 [-0.0101, +0.1600] | +0.0418 [-0.0502, +0.1210] | 0.350 | +0.0778 [-0.0113, +0.1670] | 0.713 | 0.0133 / 0.7894 | 0.414 |
| heldout | 100 | 0.310 → 0.310 (0.305) | -0.0442 [-0.1371, +0.0478] | -0.0611 [-0.1523, +0.0272] | 0.290 | -0.0764 [-0.1876, +0.0253] | 0.743 | 0.0643 / 2.3465 | 0.374 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
