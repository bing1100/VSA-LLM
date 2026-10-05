# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2652 | 0.2673 | 0.2567 | 0.5317 | 0.4245 | 0.2636 | -0.9719 | 4.4867 |
| none | 0.1916 | 0.1924 | 0.1883 | 0.5100 | 0.3375 | 0.2250 | -1.0595 | 4.7576 |
| mean_row | 0.2054 | 0.2030 | 0.2150 | 0.4700 | 0.3826 | 0.2198 | -1.1395 | 4.6149 |
| random_frame | 0.2067 | 0.2014 | 0.2283 | 0.4867 | 0.4218 | 0.2269 | -1.1235 | 4.6219 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0736 [+0.0582, +0.0889] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0598 [+0.0468, +0.0736] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0585 [+0.0445, +0.0736] | 1529 | 0.0030 | yes |
| property_new | none | +0.0749 [+0.0578, +0.0928] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0643 [+0.0504, +0.0789] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0659 [+0.0509, +0.0822] | 1229 | 0.0030 | yes |
| property_category | none | +0.0683 [+0.0300, +0.1084] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.0417 [+0.0100, +0.0717] | 300 | 0.0200 | yes |
| property_category | random_frame | +0.0283 [-0.0083, +0.0633] | 300 | 0.1549 | no |
| entailment | none | +0.0217 [-0.0150, +0.0567] | 600 | 0.2369 | no |
| entailment | mean_row | +0.0617 [+0.0317, +0.0917] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0450 [+0.0133, +0.0767] | 600 | 0.0100 | yes |
| paraphrase | none | +0.0870 [+0.0595, +0.1138] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0419 [+0.0170, +0.0661] | 1529 | 0.0030 | yes |
| paraphrase | random_frame | +0.0026 [-0.0242, +0.0301] | 1529 | 0.8806 | no |
| statement_accuracy | none | +0.0386 [+0.0216, +0.0556] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0438 [+0.0288, +0.0595] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0366 [+0.0209, +0.0523] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0876 [+0.0492, +0.1269] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1676 [+0.1344, +0.1992] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1515 [+0.1121, +0.1891] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2709 [-0.3137, -0.2321] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1281 [-0.1581, -0.0976] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1351 [-0.1733, -0.0963] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.440 → 0.475 (0.455) | +0.3927 [+0.2201, +0.6177] | +0.1470 [+0.0418, +0.2784] | 0.485 | +0.2251 [+0.1065, +0.3750] | 0.545 | 0.0004 / 0.1268 | 0.500 |
| seen | 100 | 0.405 → 0.475 (0.445) | +0.6740 [+0.3292, +1.0743] | +0.2298 [+0.0346, +0.4636] | 0.470 | +0.4232 [+0.1973, +0.6777] | 0.550 | 0.0006 / 0.1268 | 0.496 |
| heldout | 100 | 0.475 → 0.475 (0.465) | +0.1113 [+0.0483, +0.1740] | +0.0643 [-0.0039, +0.1302] | 0.500 | +0.0269 [-0.0591, +0.1141] | 0.540 | 0.0001 / 0.0297 | 0.504 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
