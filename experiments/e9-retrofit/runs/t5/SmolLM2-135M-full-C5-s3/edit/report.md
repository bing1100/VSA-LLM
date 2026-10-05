# E9 dimension 3 — zero-shot by ontology editing — C5 seed 3 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C5-s3`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2319 | 0.2587 | 0.1217 | 0.5317 | 0.4369 | 0.2518 | -1.1180 | 5.2265 |
| none | 0.1952 | 0.2050 | 0.1550 | 0.4783 | 0.4055 | 0.2145 | -1.2898 | 5.8226 |
| mean_row | 0.1903 | 0.2063 | 0.1250 | 0.4850 | 0.4421 | 0.2112 | -1.2897 | 5.3062 |
| random_frame | 0.1939 | 0.2087 | 0.1333 | 0.4700 | 0.4362 | 0.2139 | -1.2700 | 5.3428 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0366 [+0.0213, +0.0520] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0415 [+0.0291, +0.0549] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0379 [+0.0242, +0.0517] | 1529 | 0.0030 | yes |
| property_new | none | +0.0537 [+0.0354, +0.0712] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0525 [+0.0386, +0.0671] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0500 [+0.0346, +0.0651] | 1229 | 0.0030 | yes |
| property_category | none | -0.0333 [-0.0667, +0.0033] | 300 | 0.2189 | no |
| property_category | mean_row | -0.0033 [-0.0283, +0.0233] | 300 | 1.0000 | no |
| property_category | random_frame | -0.0117 [-0.0417, +0.0200] | 300 | 1.0000 | no |
| entailment | none | +0.0533 [+0.0233, +0.0833] | 600 | 0.0040 | yes |
| entailment | mean_row | +0.0467 [+0.0183, +0.0733] | 600 | 0.0040 | yes |
| entailment | random_frame | +0.0617 [+0.0317, +0.0900] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0314 [+0.0033, +0.0582] | 1529 | 0.1169 | no |
| paraphrase | mean_row | -0.0052 [-0.0314, +0.0190] | 1529 | 1.0000 | no |
| paraphrase | random_frame | +0.0007 [-0.0262, +0.0249] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0373 [+0.0229, +0.0523] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0405 [+0.0281, +0.0536] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0379 [+0.0235, +0.0530] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1718 [+0.1387, +0.2072] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1717 [+0.1430, +0.2008] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1520 [+0.1180, +0.1882] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.5961 [-0.6424, -0.5541] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.0798 [-0.1093, -0.0495] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1163 [-0.1532, -0.0816] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.420 → 0.482 (0.458) | +1.0206 [+0.8264, +1.2335] | +0.4571 [+0.3312, +0.5894] | 0.475 | +1.0312 [+0.8285, +1.2503] | 0.524 | 0.0015 / 0.2043 | 0.493 |
| seen | 100 | 0.425 → 0.515 (0.480) | +1.4195 [+1.0784, +1.7864] | +0.5370 [+0.3347, +0.7518] | 0.500 | +1.4722 [+1.1409, +1.8418] | 0.497 | 0.0012 / 0.1290 | 0.504 |
| heldout | 100 | 0.415 → 0.450 (0.435) | +0.6216 [+0.4672, +0.7813] | +0.3771 [+0.2211, +0.5370] | 0.450 | +0.5903 [+0.3851, +0.7901] | 0.550 | 0.0019 / 0.2043 | 0.479 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
