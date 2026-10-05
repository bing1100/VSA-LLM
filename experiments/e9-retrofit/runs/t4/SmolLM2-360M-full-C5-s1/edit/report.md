# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1751 | 0.1667 | 0.1867 | 0.4867 | 0.3544 | 0.2940 | -0.9805 | 3.1128 |
| none | 0.1695 | 0.1606 | 0.1817 | 0.4800 | 0.3615 | 0.2925 | -0.9858 | 3.1220 |
| mean_row | 0.1709 | 0.1557 | 0.1917 | 0.4833 | 0.3685 | 0.2954 | -0.9840 | 3.1168 |
| random_frame | 0.1751 | 0.1618 | 0.1933 | 0.4917 | 0.3502 | 0.2925 | -0.9827 | 3.1139 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0056 [-0.0035, +0.0148] | 711 | 0.8246 | no |
| property | mean_row | +0.0042 [-0.0049, +0.0134] | 711 | 0.8246 | no |
| property | random_frame | +0.0000 [-0.0091, +0.0098] | 711 | 1.0000 | no |
| property_new | none | +0.0061 [-0.0061, +0.0195] | 411 | 0.8096 | no |
| property_new | mean_row | +0.0109 [+0.0000, +0.0231] | 411 | 0.2759 | no |
| property_new | random_frame | +0.0049 [-0.0073, +0.0182] | 411 | 0.8096 | no |
| property_category | none | +0.0050 [-0.0067, +0.0167] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0050 [-0.0167, +0.0067] | 300 | 1.0000 | no |
| property_category | random_frame | -0.0067 [-0.0200, +0.0067] | 300 | 1.0000 | no |
| entailment | none | +0.0067 [-0.0100, +0.0233] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0033 [-0.0150, +0.0217] | 600 | 1.0000 | no |
| entailment | random_frame | -0.0050 [-0.0250, +0.0167] | 600 | 1.0000 | no |
| paraphrase | none | -0.0070 [-0.0254, +0.0113] | 711 | 1.0000 | no |
| paraphrase | mean_row | -0.0141 [-0.0352, +0.0070] | 711 | 0.6507 | no |
| paraphrase | random_frame | +0.0042 [-0.0169, +0.0253] | 711 | 1.0000 | no |
| statement_accuracy | none | +0.0014 [-0.0028, +0.0070] | 711 | 1.0000 | no |
| statement_accuracy | mean_row | -0.0014 [-0.0084, +0.0042] | 711 | 1.0000 | no |
| statement_accuracy | random_frame | +0.0014 [-0.0028, +0.0070] | 711 | 1.0000 | no |
| statement_margin | none | +0.0052 [+0.0011, +0.0094] | 711 | 0.0330 | yes |
| statement_margin | mean_row | +0.0035 [-0.0002, +0.0073] | 711 | 0.1359 | no |
| statement_margin | random_frame | +0.0021 [-0.0024, +0.0070] | 711 | 0.3738 | no |
| statement_loss | none | -0.0092 [-0.0133, -0.0052] | 711 | 0.0030 | yes |
| statement_loss | mean_row | -0.0040 [-0.0076, -0.0005] | 711 | 0.0480 | yes |
| statement_loss | random_frame | -0.0012 [-0.0058, +0.0035] | 711 | 0.6237 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.312 → 0.315 (0.318) | +0.0604 [+0.0090, +0.1121] | -0.0030 [-0.0471, +0.0392] | 0.325 | +0.0430 [-0.0190, +0.1048] | 0.736 | 0.0350 / 1.0405 | 0.394 |
| seen | 100 | 0.290 → 0.295 (0.300) | +0.0985 [+0.0352, +0.1644] | +0.0138 [-0.0449, +0.0687] | 0.320 | +0.0876 [+0.0094, +0.1714] | 0.745 | 0.0243 / 0.9730 | 0.382 |
| heldout | 100 | 0.335 → 0.335 (0.335) | +0.0222 [-0.0570, +0.1013] | -0.0198 [-0.0897, +0.0430] | 0.330 | -0.0017 [-0.0978, +0.0893] | 0.728 | 0.0457 / 1.0405 | 0.406 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
