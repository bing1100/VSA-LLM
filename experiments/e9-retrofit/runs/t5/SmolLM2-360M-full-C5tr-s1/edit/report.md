# E9 dimension 3 — zero-shot by ontology editing — C5tr seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5tr-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3123 | 0.3096 | 0.3233 | 0.6000 | 0.4768 | 0.2878 | -0.9246 | 4.5005 |
| none | 0.1942 | 0.1973 | 0.1817 | 0.5117 | 0.3329 | 0.2263 | -1.0798 | 4.8375 |
| mean_row | 0.1995 | 0.1937 | 0.2233 | 0.5117 | 0.4513 | 0.2198 | -1.2407 | 4.7395 |
| random_frame | 0.2162 | 0.2010 | 0.2783 | 0.5117 | 0.4611 | 0.2394 | -1.1736 | 4.7409 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.1181 [+0.0991, +0.1364] | 1529 | 0.0030 | yes |
| property | mean_row | +0.1128 [+0.0961, +0.1288] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0961 [+0.0791, +0.1135] | 1529 | 0.0030 | yes |
| property_new | none | +0.1123 [+0.0924, +0.1334] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.1159 [+0.0993, +0.1338] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.1086 [+0.0903, +0.1273] | 1229 | 0.0030 | yes |
| property_category | none | +0.1417 [+0.0966, +0.1900] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.1000 [+0.0650, +0.1383] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0450 [+0.0050, +0.0850] | 300 | 0.0310 | yes |
| entailment | none | +0.0883 [+0.0500, +0.1300] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.0883 [+0.0583, +0.1183] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0883 [+0.0567, +0.1217] | 600 | 0.0030 | yes |
| paraphrase | none | +0.1439 [+0.1138, +0.1740] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0255 [-0.0007, +0.0530] | 1529 | 0.1199 | no |
| paraphrase | random_frame | +0.0157 [-0.0124, +0.0445] | 1529 | 0.2819 | no |
| statement_accuracy | none | +0.0615 [+0.0425, +0.0818] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0680 [+0.0523, +0.0863] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0484 [+0.0307, +0.0674] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1552 [+0.1036, +0.2082] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.3161 [+0.2726, +0.3626] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.2490 [+0.2002, +0.3020] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.3369 [-0.3952, -0.2855] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.2389 [-0.2808, -0.1972] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.2403 [-0.2883, -0.1965] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.432 → 0.460 (0.453) | +0.4160 [+0.2363, +0.6533] | +0.1641 [+0.0537, +0.2913] | 0.465 | +0.2728 [+0.1549, +0.4252] | 0.550 | 0.0003 / 0.0669 | 0.488 |
| seen | 100 | 0.395 → 0.445 (0.430) | +0.7531 [+0.3866, +1.1641] | +0.3339 [+0.1457, +0.5570] | 0.450 | +0.4656 [+0.2412, +0.7369] | 0.560 | 0.0001 / 0.0128 | 0.480 |
| heldout | 100 | 0.470 → 0.475 (0.475) | +0.0789 [-0.0000, +0.1582] | -0.0058 [-0.1001, +0.0788] | 0.480 | +0.0799 [+0.0079, +0.1548] | 0.540 | 0.0004 / 0.0669 | 0.497 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
