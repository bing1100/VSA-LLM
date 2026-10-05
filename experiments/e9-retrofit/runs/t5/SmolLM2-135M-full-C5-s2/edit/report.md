# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C5-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2390 | 0.2616 | 0.1467 | 0.5317 | 0.4264 | 0.2538 | -1.0970 | 5.1355 |
| none | 0.1956 | 0.2075 | 0.1467 | 0.4833 | 0.4101 | 0.2139 | -1.2909 | 5.8415 |
| mean_row | 0.1897 | 0.2067 | 0.1200 | 0.4850 | 0.4480 | 0.2119 | -1.2898 | 5.2908 |
| random_frame | 0.1942 | 0.2095 | 0.1317 | 0.4733 | 0.4362 | 0.2184 | -1.2716 | 5.3217 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0435 [+0.0275, +0.0608] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0494 [+0.0363, +0.0634] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0448 [+0.0307, +0.0595] | 1529 | 0.0030 | yes |
| property_new | none | +0.0541 [+0.0362, +0.0716] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0549 [+0.0403, +0.0704] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0521 [+0.0362, +0.0675] | 1229 | 0.0030 | yes |
| property_category | none | +0.0000 [-0.0367, +0.0367] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0267 [+0.0000, +0.0517] | 300 | 0.1739 | no |
| property_category | random_frame | +0.0150 [-0.0150, +0.0434] | 300 | 0.6837 | no |
| entailment | none | +0.0483 [+0.0150, +0.0850] | 600 | 0.0060 | yes |
| entailment | mean_row | +0.0467 [+0.0150, +0.0783] | 600 | 0.0060 | yes |
| entailment | random_frame | +0.0583 [+0.0267, +0.0917] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0164 [-0.0105, +0.0451] | 1529 | 0.4618 | no |
| paraphrase | mean_row | -0.0216 [-0.0465, +0.0033] | 1529 | 0.3058 | no |
| paraphrase | random_frame | -0.0098 [-0.0373, +0.0170] | 1529 | 0.5057 | no |
| statement_accuracy | none | +0.0399 [+0.0249, +0.0556] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0419 [+0.0294, +0.0550] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0353 [+0.0209, +0.0497] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1939 [+0.1565, +0.2322] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1928 [+0.1648, +0.2225] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1746 [+0.1399, +0.2121] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.7060 [-0.7562, -0.6576] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1553 [-0.1850, -0.1263] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1861 [-0.2242, -0.1495] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.415 → 0.472 (0.440) | +0.8028 [+0.6452, +0.9874] | +0.3365 [+0.2420, +0.4388] | 0.470 | +0.8551 [+0.7078, +1.0265] | 0.525 | 0.0006 / 0.0926 | 0.488 |
| seen | 100 | 0.400 → 0.485 (0.440) | +1.0274 [+0.7539, +1.3193] | +0.3693 [+0.2163, +0.5382] | 0.480 | +1.0932 [+0.8341, +1.3785] | 0.492 | 0.0010 / 0.0926 | 0.486 |
| heldout | 100 | 0.430 → 0.460 (0.440) | +0.5782 [+0.4253, +0.7372] | +0.3038 [+0.1938, +0.4209] | 0.460 | +0.6171 [+0.4626, +0.7800] | 0.557 | 0.0003 / 0.0471 | 0.488 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
