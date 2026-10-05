# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C5-s2`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2345 | 0.2482 | 0.1783 | 0.5283 | 0.3669 | 0.2538 | -1.3693 | 6.8505 |
| none | 0.1975 | 0.2103 | 0.1450 | 0.4967 | 0.3839 | 0.2322 | -1.5226 | 7.3466 |
| mean_row | 0.1900 | 0.2046 | 0.1300 | 0.4933 | 0.3715 | 0.2269 | -1.4902 | 7.0036 |
| random_frame | 0.1965 | 0.2091 | 0.1450 | 0.4967 | 0.3695 | 0.2328 | -1.4835 | 6.9886 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0370 [+0.0232, +0.0517] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0445 [+0.0330, +0.0566] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0379 [+0.0255, +0.0504] | 1529 | 0.0030 | yes |
| property_new | none | +0.0378 [+0.0224, +0.0541] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0435 [+0.0305, +0.0574] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0391 [+0.0252, +0.0537] | 1229 | 0.0030 | yes |
| property_category | none | +0.0333 [+0.0033, +0.0650] | 300 | 0.0620 | no |
| property_category | mean_row | +0.0483 [+0.0233, +0.0750] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0333 [+0.0033, +0.0617] | 300 | 0.0620 | no |
| entailment | none | +0.0317 [+0.0000, +0.0633] | 600 | 0.0590 | no |
| entailment | mean_row | +0.0350 [+0.0083, +0.0600] | 600 | 0.0300 | yes |
| entailment | random_frame | +0.0317 [+0.0050, +0.0583] | 600 | 0.0460 | yes |
| paraphrase | none | -0.0170 [-0.0425, +0.0092] | 1529 | 0.6447 | no |
| paraphrase | mean_row | -0.0046 [-0.0281, +0.0183] | 1529 | 1.0000 | no |
| paraphrase | random_frame | -0.0026 [-0.0281, +0.0209] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0216 [+0.0078, +0.0347] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0268 [+0.0164, +0.0366] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0209 [+0.0092, +0.0327] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1533 [+0.1215, +0.1874] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1209 [+0.0973, +0.1451] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1143 [+0.0857, +0.1443] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.4961 [-0.5448, -0.4536] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1532 [-0.1804, -0.1264] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1381 [-0.1723, -0.1067] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.463 → 0.497 (0.475) | +0.6378 [+0.4894, +0.8055] | +0.3315 [+0.2278, +0.4406] | 0.475 | +0.8931 [+0.7259, +1.0606] | 0.484 | 0.0012 / 0.3322 | 0.485 |
| seen | 100 | 0.430 → 0.490 (0.460) | +0.8525 [+0.5996, +1.1300] | +0.3883 [+0.2220, +0.5550] | 0.470 | +1.1376 [+0.8730, +1.4074] | 0.465 | 0.0020 / 0.3322 | 0.475 |
| heldout | 100 | 0.495 → 0.505 (0.490) | +0.4232 [+0.2904, +0.5559] | +0.2746 [+0.1463, +0.3971] | 0.480 | +0.6486 [+0.4668, +0.8232] | 0.502 | 0.0004 / 0.0493 | 0.496 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
