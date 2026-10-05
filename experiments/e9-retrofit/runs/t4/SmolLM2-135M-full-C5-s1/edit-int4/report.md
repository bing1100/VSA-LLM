# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2060 | 0.1971 | 0.2183 | 0.5033 | 0.3446 | 0.2996 | -1.0654 | 3.9222 |
| none | 0.1969 | 0.1959 | 0.1983 | 0.4833 | 0.3333 | 0.2968 | -1.0794 | 3.9430 |
| mean_row | 0.1990 | 0.1946 | 0.2050 | 0.4950 | 0.3418 | 0.2954 | -1.0716 | 3.9288 |
| random_frame | 0.1990 | 0.1946 | 0.2050 | 0.4950 | 0.3404 | 0.2940 | -1.0692 | 3.9262 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0091 [-0.0021, +0.0211] | 711 | 0.3718 | no |
| property | mean_row | +0.0070 [-0.0028, +0.0169] | 711 | 0.4318 | no |
| property | random_frame | +0.0070 [-0.0049, +0.0190] | 711 | 0.4318 | no |
| property_new | none | +0.0012 [-0.0134, +0.0158] | 411 | 1.0000 | no |
| property_new | mean_row | +0.0024 [-0.0109, +0.0158] | 411 | 1.0000 | no |
| property_new | random_frame | +0.0024 [-0.0122, +0.0170] | 411 | 1.0000 | no |
| property_category | none | +0.0200 [+0.0017, +0.0383] | 300 | 0.1469 | no |
| property_category | mean_row | +0.0133 [-0.0017, +0.0300] | 300 | 0.2139 | no |
| property_category | random_frame | +0.0133 [-0.0067, +0.0317] | 300 | 0.2209 | no |
| entailment | none | +0.0200 [+0.0000, +0.0400] | 600 | 0.2099 | no |
| entailment | mean_row | +0.0083 [-0.0100, +0.0250] | 600 | 0.8256 | no |
| entailment | random_frame | +0.0083 [-0.0133, +0.0283] | 600 | 0.8256 | no |
| paraphrase | none | +0.0113 [-0.0127, +0.0366] | 711 | 1.0000 | no |
| paraphrase | mean_row | +0.0028 [-0.0197, +0.0253] | 711 | 1.0000 | no |
| paraphrase | random_frame | +0.0042 [-0.0197, +0.0295] | 711 | 1.0000 | no |
| statement_accuracy | none | +0.0028 [-0.0056, +0.0113] | 711 | 0.8576 | no |
| statement_accuracy | mean_row | +0.0042 [-0.0042, +0.0127] | 711 | 0.8576 | no |
| statement_accuracy | random_frame | +0.0056 [-0.0028, +0.0141] | 711 | 0.8576 | no |
| statement_margin | none | +0.0140 [+0.0082, +0.0202] | 711 | 0.0030 | yes |
| statement_margin | mean_row | +0.0062 [+0.0013, +0.0113] | 711 | 0.0220 | yes |
| statement_margin | random_frame | +0.0039 [-0.0025, +0.0103] | 711 | 0.2349 | no |
| statement_loss | none | -0.0208 [-0.0269, -0.0150] | 711 | 0.0030 | yes |
| statement_loss | mean_row | -0.0066 [-0.0115, -0.0018] | 711 | 0.0040 | yes |
| statement_loss | random_frame | -0.0040 [-0.0100, +0.0020] | 711 | 0.1849 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.343 → 0.347 (0.343) | +0.1005 [+0.0119, +0.1813] | +0.0684 [-0.0216, +0.1572] | 0.330 | +0.0618 [-0.0318, +0.1518] | 0.700 | 0.0499 / 1.6644 | 0.409 |
| seen | 100 | 0.370 → 0.375 (0.370) | +0.1436 [+0.0257, +0.2609] | +0.0365 [-0.0804, +0.1481] | 0.350 | +0.0953 [-0.0245, +0.2108] | 0.657 | 0.0342 / 1.1695 | 0.426 |
| heldout | 100 | 0.315 → 0.320 (0.315) | +0.0574 [-0.0746, +0.1813] | +0.1003 [-0.0245, +0.2254] | 0.310 | +0.0282 [-0.1201, +0.1685] | 0.743 | 0.0656 / 1.6644 | 0.390 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
