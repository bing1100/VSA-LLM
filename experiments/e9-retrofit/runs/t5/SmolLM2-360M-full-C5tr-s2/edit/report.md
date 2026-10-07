# E9 dimension 3 — zero-shot by ontology editing — C5tr seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5tr-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3156 | 0.3031 | 0.3667 | 0.6050 | 0.4676 | 0.3022 | -0.8756 | 4.4865 |
| none | 0.1952 | 0.1993 | 0.1783 | 0.5233 | 0.3355 | 0.2256 | -1.0716 | 4.8436 |
| mean_row | 0.2011 | 0.1924 | 0.2367 | 0.5000 | 0.4716 | 0.2204 | -1.2427 | 4.7726 |
| random_frame | 0.2165 | 0.1969 | 0.2967 | 0.5133 | 0.4702 | 0.2440 | -1.1677 | 4.7509 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.1203 [+0.0997, +0.1390] | 1529 | 0.0030 | yes |
| property | mean_row | +0.1145 [+0.0974, +0.1308] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0991 [+0.0827, +0.1158] | 1529 | 0.0030 | yes |
| property_new | none | +0.1037 [+0.0830, +0.1257] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.1107 [+0.0928, +0.1294] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.1062 [+0.0871, +0.1261] | 1229 | 0.0030 | yes |
| property_category | none | +0.1883 [+0.1383, +0.2367] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.1300 [+0.0933, +0.1667] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0700 [+0.0300, +0.1100] | 300 | 0.0030 | yes |
| entailment | none | +0.0817 [+0.0417, +0.1233] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.1050 [+0.0700, +0.1417] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0917 [+0.0583, +0.1250] | 600 | 0.0030 | yes |
| paraphrase | none | +0.1321 [+0.1014, +0.1642] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | -0.0039 [-0.0334, +0.0235] | 1529 | 1.0000 | no |
| paraphrase | random_frame | -0.0026 [-0.0314, +0.0262] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0765 [+0.0562, +0.0981] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0818 [+0.0654, +0.0988] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0582 [+0.0412, +0.0765] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1960 [+0.1443, +0.2524] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.3671 [+0.3235, +0.4136] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.2921 [+0.2470, +0.3400] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.3570 [-0.4183, -0.3008] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.2861 [-0.3320, -0.2412] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.2644 [-0.3140, -0.2179] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.438 → 0.463 (0.458) | +0.3462 [+0.1907, +0.5517] | +0.1159 [+0.0175, +0.2267] | 0.465 | +0.2564 [+0.1422, +0.4009] | 0.546 | 0.0005 / 0.0862 | 0.488 |
| seen | 100 | 0.390 → 0.445 (0.430) | +0.6554 [+0.3352, +1.0114] | +0.2632 [+0.1083, +0.4409] | 0.430 | +0.4649 [+0.2384, +0.7104] | 0.557 | 0.0001 / 0.0238 | 0.471 |
| heldout | 100 | 0.485 → 0.480 (0.485) | +0.0370 [-0.0375, +0.1072] | -0.0315 [-0.1298, +0.0567] | 0.500 | +0.0480 [-0.0238, +0.1153] | 0.535 | 0.0008 / 0.0862 | 0.504 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
