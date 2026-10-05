# E9 dimension 3 — zero-shot by ontology editing — C5 seed 3 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C5-s3`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2158 | 0.2246 | 0.1800 | 0.5133 | 0.3689 | 0.2472 | -1.3976 | 6.8843 |
| none | 0.1949 | 0.2030 | 0.1617 | 0.4983 | 0.3506 | 0.2374 | -1.5068 | 7.3097 |
| mean_row | 0.1861 | 0.1945 | 0.1517 | 0.4983 | 0.3636 | 0.2387 | -1.4702 | 6.9525 |
| random_frame | 0.1975 | 0.2059 | 0.1633 | 0.5083 | 0.3833 | 0.2354 | -1.4603 | 6.9578 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0209 [+0.0078, +0.0334] | 1529 | 0.0040 | yes |
| property | mean_row | +0.0298 [+0.0190, +0.0412] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0183 [+0.0065, +0.0311] | 1529 | 0.0040 | yes |
| property_new | none | +0.0216 [+0.0081, +0.0358] | 1229 | 0.0040 | yes |
| property_new | mean_row | +0.0301 [+0.0179, +0.0423] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0187 [+0.0053, +0.0325] | 1229 | 0.0040 | yes |
| property_category | none | +0.0183 [-0.0133, +0.0500] | 300 | 0.5677 | no |
| property_category | mean_row | +0.0283 [+0.0050, +0.0517] | 300 | 0.0540 | no |
| property_category | random_frame | +0.0167 [-0.0100, +0.0450] | 300 | 0.5677 | no |
| entailment | none | +0.0150 [-0.0150, +0.0450] | 600 | 0.7496 | no |
| entailment | mean_row | +0.0150 [-0.0083, +0.0367] | 600 | 0.7496 | no |
| entailment | random_frame | +0.0050 [-0.0233, +0.0350] | 600 | 0.7596 | no |
| paraphrase | none | +0.0183 [-0.0065, +0.0438] | 1529 | 0.5277 | no |
| paraphrase | mean_row | +0.0052 [-0.0183, +0.0275] | 1529 | 0.6867 | no |
| paraphrase | random_frame | -0.0144 [-0.0386, +0.0098] | 1529 | 0.5277 | no |
| statement_accuracy | none | +0.0098 [-0.0013, +0.0229] | 1529 | 0.2159 | no |
| statement_accuracy | mean_row | +0.0085 [-0.0020, +0.0203] | 1529 | 0.2159 | no |
| statement_accuracy | random_frame | +0.0118 [+0.0013, +0.0235] | 1529 | 0.1139 | no |
| statement_margin | none | +0.1091 [+0.0793, +0.1422] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.0725 [+0.0508, +0.0950] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.0627 [+0.0366, +0.0900] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.4253 [-0.4681, -0.3854] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.0681 [-0.0955, -0.0423] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.0735 [-0.1073, -0.0410] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.470 → 0.495 (0.472) | +0.8667 [+0.6942, +1.0501] | +0.4390 [+0.3116, +0.5694] | 0.495 | +1.0578 [+0.8701, +1.2578] | 0.485 | 0.0017 / 0.2418 | 0.492 |
| seen | 100 | 0.445 → 0.495 (0.460) | +1.0668 [+0.7668, +1.3831] | +0.4828 [+0.2910, +0.6903] | 0.500 | +1.2698 [+0.9247, +1.6227] | 0.450 | 0.0013 / 0.1101 | 0.481 |
| heldout | 100 | 0.495 → 0.495 (0.485) | +0.6667 [+0.4831, +0.8733] | +0.3953 [+0.2131, +0.5701] | 0.490 | +0.8459 [+0.6671, +1.0360] | 0.520 | 0.0021 / 0.2418 | 0.501 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
