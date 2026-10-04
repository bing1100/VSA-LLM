# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2884 | 0.2909 | 0.2783 | 0.5467 | 0.4572 | 0.2838 | -0.9427 | 4.2087 |
| none | 0.2005 | 0.2059 | 0.1783 | 0.4883 | 0.3355 | 0.2309 | -1.1798 | 4.9133 |
| mean_row | 0.2014 | 0.2128 | 0.1550 | 0.4883 | 0.4519 | 0.2263 | -1.4533 | 4.5213 |
| random_frame | 0.2090 | 0.2095 | 0.2067 | 0.4900 | 0.4513 | 0.2394 | -1.3714 | 4.5940 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0880 [+0.0706, +0.1053] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0870 [+0.0736, +0.1007] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0795 [+0.0641, +0.0945] | 1529 | 0.0030 | yes |
| property_new | none | +0.0850 [+0.0667, +0.1046] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0781 [+0.0639, +0.0936] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0814 [+0.0651, +0.0980] | 1229 | 0.0030 | yes |
| property_category | none | +0.1000 [+0.0633, +0.1383] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.1233 [+0.0917, +0.1550] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0717 [+0.0383, +0.1067] | 300 | 0.0030 | yes |
| entailment | none | +0.0583 [+0.0167, +0.1017] | 600 | 0.0130 | yes |
| entailment | mean_row | +0.0583 [+0.0300, +0.0883] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0567 [+0.0267, +0.0900] | 600 | 0.0040 | yes |
| paraphrase | none | +0.1216 [+0.0929, +0.1504] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0052 [-0.0196, +0.0288] | 1529 | 1.0000 | no |
| paraphrase | random_frame | +0.0059 [-0.0216, +0.0327] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0530 [+0.0340, +0.0733] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0576 [+0.0412, +0.0733] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0445 [+0.0288, +0.0615] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.2371 [+0.1476, +0.3248] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.5106 [+0.4334, +0.5888] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.4288 [+0.3437, +0.5157] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.7046 [-0.8072, -0.5982] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.3126 [-0.3875, -0.2353] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.3853 [-0.4825, -0.2900] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.417 → 0.465 (0.438) | +0.4760 [+0.1883, +0.8528] | +0.2132 [+0.0587, +0.4123] | 0.440 | +0.1777 [+0.0680, +0.3189] | 0.566 | 0.0022 / 0.2066 | 0.485 |
| seen | 100 | 0.435 → 0.505 (0.450) | +0.9606 [+0.3649, +1.6558] | +0.4330 [+0.1280, +0.7984] | 0.480 | +0.3122 [+0.1045, +0.5719] | 0.550 | 0.0019 / 0.1076 | 0.510 |
| heldout | 100 | 0.400 → 0.425 (0.425) | -0.0086 [-0.0783, +0.0573] | -0.0066 [-0.0695, +0.0555] | 0.400 | +0.0431 [-0.0335, +0.1222] | 0.583 | 0.0026 / 0.2066 | 0.457 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
