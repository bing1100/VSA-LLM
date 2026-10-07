# E9 dimension 3 — zero-shot by ontology editing — C5sh seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5sh-s3`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1962 | 0.2026 | 0.1700 | 0.5183 | 0.3492 | 0.2184 | -1.0622 | 4.6453 |
| none | 0.1916 | 0.1949 | 0.1783 | 0.5150 | 0.3421 | 0.2224 | -1.0356 | 4.6957 |
| mean_row | 0.1956 | 0.2018 | 0.1700 | 0.5067 | 0.3362 | 0.2171 | -1.0582 | 4.6294 |
| random_frame | 0.1956 | 0.2014 | 0.1717 | 0.5000 | 0.3519 | 0.2191 | -1.0595 | 4.6419 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0046 [-0.0049, +0.0141] | 1529 | 1.0000 | no |
| property | mean_row | +0.0007 [-0.0069, +0.0082] | 1529 | 1.0000 | no |
| property | random_frame | +0.0007 [-0.0082, +0.0095] | 1529 | 1.0000 | no |
| property_new | none | +0.0077 [-0.0024, +0.0179] | 1229 | 0.4288 | no |
| property_new | mean_row | +0.0008 [-0.0077, +0.0094] | 1229 | 1.0000 | no |
| property_new | random_frame | +0.0012 [-0.0081, +0.0110] | 1229 | 1.0000 | no |
| property_category | none | -0.0083 [-0.0333, +0.0183] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0000 [-0.0200, +0.0217] | 300 | 1.0000 | no |
| property_category | random_frame | -0.0017 [-0.0250, +0.0217] | 300 | 1.0000 | no |
| entailment | none | +0.0033 [-0.0200, +0.0267] | 600 | 0.8316 | no |
| entailment | mean_row | +0.0117 [-0.0100, +0.0333] | 600 | 0.6157 | no |
| entailment | random_frame | +0.0183 [-0.0033, +0.0400] | 600 | 0.3718 | no |
| paraphrase | none | +0.0072 [-0.0124, +0.0268] | 1529 | 1.0000 | no |
| paraphrase | mean_row | +0.0131 [-0.0046, +0.0307] | 1529 | 0.5037 | no |
| paraphrase | random_frame | -0.0026 [-0.0216, +0.0164] | 1529 | 1.0000 | no |
| statement_accuracy | none | -0.0039 [-0.0151, +0.0072] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0013 [-0.0085, +0.0111] | 1529 | 1.0000 | no |
| statement_accuracy | random_frame | -0.0007 [-0.0111, +0.0098] | 1529 | 1.0000 | no |
| statement_margin | none | -0.0266 [-0.0474, -0.0060] | 1529 | 0.0390 | no |
| statement_margin | mean_row | -0.0039 [-0.0192, +0.0111] | 1529 | 1.0000 | no |
| statement_margin | random_frame | -0.0027 [-0.0214, +0.0154] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0504 [-0.0742, -0.0290] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | +0.0160 [-0.0013, +0.0338] | 1529 | 0.1259 | no |
| statement_loss | random_frame | +0.0034 [-0.0168, +0.0244] | 1529 | 0.7166 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).
