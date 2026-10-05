# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s2`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2757 | 0.2771 | 0.2700 | 0.5450 | 0.3937 | 0.2531 | -1.2139 | 5.4051 |
| none | 0.2037 | 0.2038 | 0.2033 | 0.5067 | 0.3218 | 0.2224 | -1.3120 | 5.7051 |
| mean_row | 0.2194 | 0.2128 | 0.2467 | 0.5350 | 0.4075 | 0.2204 | -1.4381 | 5.5977 |
| random_frame | 0.2158 | 0.2055 | 0.2583 | 0.5067 | 0.3976 | 0.2086 | -1.4174 | 5.5817 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0719 [+0.0549, +0.0886] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0562 [+0.0428, +0.0700] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0598 [+0.0441, +0.0765] | 1529 | 0.0030 | yes |
| property_new | none | +0.0732 [+0.0549, +0.0919] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0643 [+0.0480, +0.0814] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0716 [+0.0541, +0.0903] | 1229 | 0.0030 | yes |
| property_category | none | +0.0667 [+0.0250, +0.1067] | 300 | 0.0090 | yes |
| property_category | mean_row | +0.0233 [-0.0117, +0.0567] | 300 | 0.3678 | no |
| property_category | random_frame | +0.0117 [-0.0217, +0.0467] | 300 | 0.5697 | no |
| entailment | none | +0.0383 [+0.0050, +0.0733] | 600 | 0.1019 | no |
| entailment | mean_row | +0.0100 [-0.0183, +0.0383] | 600 | 0.5297 | no |
| entailment | random_frame | +0.0383 [+0.0050, +0.0700] | 600 | 0.1019 | no |
| paraphrase | none | +0.0719 [+0.0425, +0.1001] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | -0.0137 [-0.0425, +0.0131] | 1529 | 0.5897 | no |
| paraphrase | random_frame | -0.0039 [-0.0334, +0.0223] | 1529 | 0.8216 | no |
| statement_accuracy | none | +0.0307 [+0.0137, +0.0484] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0327 [+0.0170, +0.0484] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0445 [+0.0288, +0.0615] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0981 [+0.0453, +0.1495] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.2243 [+0.1821, +0.2686] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.2035 [+0.1577, +0.2501] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2999 [-0.3561, -0.2464] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1925 [-0.2306, -0.1576] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1765 [-0.2215, -0.1363] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.460 → 0.490 (0.475) | +0.3952 [+0.1954, +0.6408] | +0.1466 [+0.0038, +0.3059] | 0.470 | +0.2411 [+0.1117, +0.4025] | 0.535 | 0.0010 / 0.1220 | 0.497 |
| seen | 100 | 0.440 → 0.500 (0.475) | +0.6869 [+0.2867, +1.1153] | +0.2491 [-0.0019, +0.5387] | 0.490 | +0.3908 [+0.1470, +0.6548] | 0.505 | 0.0009 / 0.1110 | 0.498 |
| heldout | 100 | 0.480 → 0.480 (0.475) | +0.1035 [-0.0051, +0.2133] | +0.0441 [-0.0632, +0.1560] | 0.450 | +0.0913 [-0.0231, +0.2142] | 0.565 | 0.0011 / 0.1220 | 0.494 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
