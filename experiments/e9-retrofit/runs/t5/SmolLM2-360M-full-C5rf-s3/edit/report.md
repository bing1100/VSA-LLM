# E9 dimension 3 — zero-shot by ontology editing — C5rf seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5rf-s3`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2629 | 0.2628 | 0.2633 | 0.5583 | 0.4081 | 0.2583 | -0.9583 | 4.4918 |
| none | 0.1864 | 0.1900 | 0.1717 | 0.5183 | 0.3466 | 0.2191 | -1.0632 | 4.7641 |
| mean_row | 0.2011 | 0.2006 | 0.2033 | 0.4867 | 0.3833 | 0.2132 | -1.1467 | 4.6296 |
| random_frame | 0.2031 | 0.2022 | 0.2067 | 0.4883 | 0.4303 | 0.2263 | -1.1480 | 4.6496 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0765 [+0.0605, +0.0919] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0618 [+0.0477, +0.0755] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0598 [+0.0451, +0.0759] | 1529 | 0.0030 | yes |
| property_new | none | +0.0728 [+0.0557, +0.0899] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0622 [+0.0472, +0.0781] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0606 [+0.0460, +0.0769] | 1229 | 0.0030 | yes |
| property_category | none | +0.0917 [+0.0533, +0.1317] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.0600 [+0.0283, +0.0933] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0567 [+0.0183, +0.0967] | 300 | 0.0030 | yes |
| entailment | none | +0.0400 [+0.0066, +0.0767] | 600 | 0.0230 | yes |
| entailment | mean_row | +0.0717 [+0.0450, +0.1017] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0700 [+0.0417, +0.0983] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0615 [+0.0340, +0.0889] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0249 [+0.0006, +0.0491] | 1529 | 0.1019 | no |
| paraphrase | random_frame | -0.0222 [-0.0477, +0.0059] | 1529 | 0.1309 | no |
| statement_accuracy | none | +0.0392 [+0.0229, +0.0576] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0451 [+0.0307, +0.0608] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0320 [+0.0164, +0.0484] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1049 [+0.0658, +0.1491] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1884 [+0.1567, +0.2244] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1897 [+0.1498, +0.2308] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2722 [-0.3153, -0.2327] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1378 [-0.1700, -0.1066] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1577 [-0.1965, -0.1196] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.432 → 0.463 (0.450) | +0.3853 [+0.2076, +0.6096] | +0.1510 [+0.0523, +0.2776] | 0.485 | +0.2300 [+0.1141, +0.3677] | 0.542 | 0.0008 / 0.1279 | 0.494 |
| seen | 100 | 0.410 → 0.465 (0.430) | +0.7039 [+0.3490, +1.1047] | +0.2492 [+0.0652, +0.4581] | 0.470 | +0.4037 [+0.1854, +0.6515] | 0.542 | 0.0006 / 0.0650 | 0.490 |
| heldout | 100 | 0.455 → 0.460 (0.470) | +0.0666 [+0.0004, +0.1420] | +0.0528 [-0.0118, +0.1148] | 0.500 | +0.0564 [-0.0073, +0.1250] | 0.542 | 0.0010 / 0.1279 | 0.499 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
