# E9 dimension 3 — zero-shot by ontology editing — C5rf seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5rf-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2279 | 0.2339 | 0.2033 | 0.5333 | 0.3519 | 0.2322 | -1.0068 | 4.5671 |
| none | 0.1952 | 0.1945 | 0.1983 | 0.5133 | 0.3407 | 0.2256 | -1.0584 | 4.7270 |
| mean_row | 0.2008 | 0.2050 | 0.1833 | 0.5033 | 0.3506 | 0.2178 | -1.0958 | 4.6415 |
| random_frame | 0.2070 | 0.2116 | 0.1883 | 0.5050 | 0.3663 | 0.2276 | -1.0994 | 4.6290 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0327 [+0.0209, +0.0448] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0271 [+0.0160, +0.0383] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0209 [+0.0101, +0.0324] | 1529 | 0.0030 | yes |
| property_new | none | +0.0395 [+0.0269, +0.0533] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0289 [+0.0163, +0.0415] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0224 [+0.0094, +0.0358] | 1229 | 0.0030 | yes |
| property_category | none | +0.0050 [-0.0267, +0.0383] | 300 | 0.8026 | no |
| property_category | mean_row | +0.0200 [-0.0067, +0.0467] | 300 | 0.5097 | no |
| property_category | random_frame | +0.0150 [-0.0150, +0.0450] | 300 | 0.6797 | no |
| entailment | none | +0.0200 [-0.0100, +0.0500] | 600 | 0.2119 | no |
| entailment | mean_row | +0.0300 [+0.0033, +0.0567] | 600 | 0.0990 | no |
| entailment | random_frame | +0.0283 [+0.0000, +0.0550] | 600 | 0.1319 | no |
| paraphrase | none | +0.0111 [-0.0111, +0.0334] | 1529 | 0.6927 | no |
| paraphrase | mean_row | +0.0013 [-0.0183, +0.0216] | 1529 | 0.9655 | no |
| paraphrase | random_frame | -0.0144 [-0.0366, +0.0078] | 1529 | 0.6927 | no |
| statement_accuracy | none | +0.0065 [-0.0059, +0.0196] | 1529 | 0.7076 | no |
| statement_accuracy | mean_row | +0.0144 [+0.0026, +0.0268] | 1529 | 0.0750 | no |
| statement_accuracy | random_frame | +0.0046 [-0.0098, +0.0190] | 1529 | 0.7076 | no |
| statement_margin | none | +0.0516 [+0.0267, +0.0789] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.0890 [+0.0660, +0.1132] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.0926 [+0.0661, +0.1209] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.1599 [-0.1895, -0.1324] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.0743 [-0.0971, -0.0506] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.0618 [-0.0892, -0.0363] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.430 → 0.465 (0.455) | +0.3554 [+0.1751, +0.5846] | +0.1187 [+0.0041, +0.2514] | 0.475 | +0.1909 [+0.0656, +0.3450] | 0.540 | 0.0005 / 0.0592 | 0.491 |
| seen | 100 | 0.405 → 0.470 (0.450) | +0.6711 [+0.3132, +1.1031] | +0.2440 [+0.0566, +0.4802] | 0.480 | +0.3466 [+0.1078, +0.6251] | 0.537 | 0.0008 / 0.0592 | 0.494 |
| heldout | 100 | 0.455 → 0.460 (0.460) | +0.0398 [-0.0762, +0.1431] | -0.0065 [-0.1303, +0.0942] | 0.470 | +0.0351 [-0.0581, +0.1285] | 0.542 | 0.0002 / 0.0332 | 0.488 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
