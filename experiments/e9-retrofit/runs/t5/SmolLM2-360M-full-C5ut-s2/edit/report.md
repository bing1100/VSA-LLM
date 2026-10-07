# E9 dimension 3 — zero-shot by ontology editing — C5ut seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5ut-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2626 | 0.2673 | 0.2433 | 0.5467 | 0.4192 | 0.2570 | -0.9785 | 4.5164 |
| none | 0.1926 | 0.1969 | 0.1750 | 0.5033 | 0.3479 | 0.2269 | -1.0588 | 4.7552 |
| mean_row | 0.2011 | 0.2030 | 0.1933 | 0.4933 | 0.3963 | 0.2204 | -1.1481 | 4.6392 |
| random_frame | 0.2031 | 0.2022 | 0.2067 | 0.5100 | 0.4277 | 0.2198 | -1.1382 | 4.6602 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0700 [+0.0546, +0.0857] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0615 [+0.0484, +0.0746] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0595 [+0.0438, +0.0746] | 1529 | 0.0030 | yes |
| property_new | none | +0.0704 [+0.0533, +0.0863] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0643 [+0.0492, +0.0793] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0651 [+0.0480, +0.0822] | 1229 | 0.0030 | yes |
| property_category | none | +0.0683 [+0.0300, +0.1083] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.0500 [+0.0200, +0.0800] | 300 | 0.0060 | yes |
| property_category | random_frame | +0.0367 [+0.0000, +0.0750] | 300 | 0.0590 | no |
| entailment | none | +0.0433 [+0.0100, +0.0783] | 600 | 0.0180 | yes |
| entailment | mean_row | +0.0533 [+0.0250, +0.0833] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0367 [+0.0050, +0.0683] | 600 | 0.0220 | yes |
| paraphrase | none | +0.0713 [+0.0438, +0.1001] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0229 [-0.0026, +0.0484] | 1529 | 0.1859 | no |
| paraphrase | random_frame | -0.0085 [-0.0334, +0.0183] | 1529 | 0.5497 | no |
| statement_accuracy | none | +0.0301 [+0.0124, +0.0477] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0366 [+0.0216, +0.0517] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0373 [+0.0196, +0.0543] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0803 [+0.0430, +0.1214] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1697 [+0.1348, +0.2057] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1598 [+0.1186, +0.2042] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2388 [-0.2845, -0.1981] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1228 [-0.1590, -0.0856] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1438 [-0.1866, -0.1023] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.435 → 0.458 (0.448) | +0.4162 [+0.2318, +0.6483] | +0.1764 [+0.0524, +0.3246] | 0.475 | +0.2808 [+0.1470, +0.4418] | 0.547 | 0.0000 / 0.0000 | 0.490 |
| seen | 100 | 0.405 → 0.455 (0.440) | +0.7217 [+0.3509, +1.1358] | +0.2850 [+0.0673, +0.5239] | 0.460 | +0.4578 [+0.1999, +0.7465] | 0.545 | 0.0000 / 0.0000 | 0.483 |
| heldout | 100 | 0.465 → 0.460 (0.455) | +0.1107 [+0.0270, +0.1962] | +0.0678 [-0.0251, +0.1612] | 0.490 | +0.1039 [+0.0182, +0.1955] | 0.550 | 0.0000 / 0.0000 | 0.497 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
