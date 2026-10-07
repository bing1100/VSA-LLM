# E9 dimension 3 — zero-shot by ontology editing — C5sh seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5sh-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2011 | 0.2026 | 0.1950 | 0.5033 | 0.3532 | 0.2217 | -1.0590 | 4.6334 |
| none | 0.1969 | 0.1993 | 0.1867 | 0.5117 | 0.3342 | 0.2276 | -1.0360 | 4.6799 |
| mean_row | 0.1920 | 0.1985 | 0.1650 | 0.4917 | 0.3407 | 0.2217 | -1.0625 | 4.6303 |
| random_frame | 0.1969 | 0.1993 | 0.1867 | 0.5083 | 0.3407 | 0.2328 | -1.0530 | 4.6297 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0043 [-0.0052, +0.0134] | 1529 | 0.7636 | no |
| property | mean_row | +0.0092 [+0.0003, +0.0180] | 1529 | 0.1499 | no |
| property | random_frame | +0.0043 [-0.0049, +0.0128] | 1529 | 0.7636 | no |
| property_new | none | +0.0033 [-0.0061, +0.0126] | 1229 | 1.0000 | no |
| property_new | mean_row | +0.0041 [-0.0057, +0.0134] | 1229 | 1.0000 | no |
| property_new | random_frame | +0.0033 [-0.0061, +0.0126] | 1229 | 1.0000 | no |
| property_category | none | +0.0083 [-0.0183, +0.0350] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0300 [+0.0083, +0.0517] | 300 | 0.0210 | yes |
| property_category | random_frame | +0.0083 [-0.0167, +0.0300] | 300 | 1.0000 | no |
| entailment | none | -0.0083 [-0.0300, +0.0133] | 600 | 0.9415 | no |
| entailment | mean_row | +0.0117 [-0.0067, +0.0317] | 600 | 0.7736 | no |
| entailment | random_frame | -0.0050 [-0.0267, +0.0150] | 600 | 0.9415 | no |
| paraphrase | none | +0.0190 [-0.0000, +0.0366] | 1529 | 0.1829 | no |
| paraphrase | mean_row | +0.0124 [-0.0059, +0.0301] | 1529 | 0.3918 | no |
| paraphrase | random_frame | +0.0124 [-0.0059, +0.0301] | 1529 | 0.3918 | no |
| statement_accuracy | none | -0.0059 [-0.0164, +0.0052] | 1529 | 0.6717 | no |
| statement_accuracy | mean_row | +0.0000 [-0.0105, +0.0098] | 1529 | 1.0000 | no |
| statement_accuracy | random_frame | -0.0111 [-0.0222, +0.0000] | 1529 | 0.1709 | no |
| statement_margin | none | -0.0229 [-0.0416, -0.0037] | 1529 | 0.0630 | no |
| statement_margin | mean_row | +0.0036 [-0.0111, +0.0182] | 1529 | 0.9615 | no |
| statement_margin | random_frame | -0.0060 [-0.0229, +0.0107] | 1529 | 0.9615 | no |
| statement_loss | none | -0.0465 [-0.0686, -0.0254] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | +0.0031 [-0.0121, +0.0189] | 1529 | 1.0000 | no |
| statement_loss | random_frame | +0.0038 [-0.0136, +0.0210] | 1529 | 1.0000 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).
