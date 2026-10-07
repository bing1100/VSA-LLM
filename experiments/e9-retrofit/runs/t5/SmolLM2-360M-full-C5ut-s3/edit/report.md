# E9 dimension 3 — zero-shot by ontology editing — C5ut seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5ut-s3`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2619 | 0.2624 | 0.2600 | 0.5500 | 0.4114 | 0.2525 | -1.0003 | 4.5470 |
| none | 0.1900 | 0.1932 | 0.1767 | 0.5200 | 0.3290 | 0.2224 | -1.0615 | 4.7545 |
| mean_row | 0.1991 | 0.2002 | 0.1950 | 0.4867 | 0.3800 | 0.2132 | -1.1448 | 4.6321 |
| random_frame | 0.2070 | 0.2050 | 0.2150 | 0.4983 | 0.4101 | 0.2145 | -1.1700 | 4.6843 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0719 [+0.0569, +0.0873] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0628 [+0.0490, +0.0765] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0549 [+0.0402, +0.0700] | 1529 | 0.0030 | yes |
| property_new | none | +0.0692 [+0.0533, +0.0854] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0622 [+0.0484, +0.0777] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0574 [+0.0431, +0.0736] | 1229 | 0.0030 | yes |
| property_category | none | +0.0833 [+0.0467, +0.1217] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.0650 [+0.0317, +0.1000] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0450 [+0.0083, +0.0833] | 300 | 0.0160 | yes |
| entailment | none | +0.0300 [-0.0050, +0.0650] | 600 | 0.1019 | no |
| entailment | mean_row | +0.0633 [+0.0333, +0.0950] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0517 [+0.0200, +0.0850] | 600 | 0.0040 | yes |
| paraphrase | none | +0.0824 [+0.0562, +0.1086] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0314 [+0.0052, +0.0569] | 1529 | 0.0480 | yes |
| paraphrase | random_frame | +0.0013 [-0.0255, +0.0262] | 1529 | 0.9385 | no |
| statement_accuracy | none | +0.0301 [+0.0144, +0.0471] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0392 [+0.0255, +0.0543] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0379 [+0.0235, +0.0530] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0611 [+0.0239, +0.1016] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1445 [+0.1130, +0.1783] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1696 [+0.1302, +0.2116] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2075 [-0.2484, -0.1704] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.0851 [-0.1168, -0.0545] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1373 [-0.1799, -0.0996] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.435 → 0.458 (0.455) | +0.3690 [+0.1916, +0.5945] | +0.1521 [+0.0413, +0.2846] | 0.475 | +0.2286 [+0.1073, +0.3752] | 0.544 | 0.0014 / 0.1574 | 0.489 |
| seen | 100 | 0.405 → 0.455 (0.445) | +0.6527 [+0.2983, +1.0457] | +0.2877 [+0.0885, +0.5212] | 0.460 | +0.3997 [+0.1650, +0.6648] | 0.550 | 0.0011 / 0.1574 | 0.485 |
| heldout | 100 | 0.465 → 0.460 (0.465) | +0.0853 [+0.0145, +0.1573] | +0.0166 [-0.0535, +0.0887] | 0.490 | +0.0574 [-0.0175, +0.1322] | 0.537 | 0.0018 / 0.1118 | 0.494 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
