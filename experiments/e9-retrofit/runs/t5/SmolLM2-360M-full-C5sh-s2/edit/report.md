# E9 dimension 3 — zero-shot by ontology editing — C5sh seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5sh-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1978 | 0.2006 | 0.1867 | 0.5000 | 0.3368 | 0.2145 | -1.0628 | 4.6345 |
| none | 0.1906 | 0.1928 | 0.1817 | 0.5167 | 0.3414 | 0.2198 | -1.0406 | 4.6946 |
| mean_row | 0.1946 | 0.2010 | 0.1683 | 0.4917 | 0.3342 | 0.2224 | -1.0651 | 4.6402 |
| random_frame | 0.1969 | 0.2022 | 0.1750 | 0.5100 | 0.3506 | 0.2224 | -1.0607 | 4.6417 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0072 [-0.0020, +0.0164] | 1529 | 0.4138 | no |
| property | mean_row | +0.0033 [-0.0046, +0.0114] | 1529 | 0.8396 | no |
| property | random_frame | +0.0010 [-0.0082, +0.0098] | 1529 | 0.8636 | no |
| property_new | none | +0.0077 [-0.0016, +0.0175] | 1229 | 0.3808 | no |
| property_new | mean_row | -0.0004 [-0.0098, +0.0081] | 1229 | 1.0000 | no |
| property_new | random_frame | -0.0016 [-0.0122, +0.0081] | 1229 | 1.0000 | no |
| property_category | none | +0.0050 [-0.0200, +0.0283] | 300 | 0.7136 | no |
| property_category | mean_row | +0.0183 [+0.0000, +0.0383] | 300 | 0.2189 | no |
| property_category | random_frame | +0.0117 [-0.0117, +0.0367] | 300 | 0.7136 | no |
| entailment | none | -0.0167 [-0.0400, +0.0067] | 600 | 0.6267 | no |
| entailment | mean_row | +0.0083 [-0.0117, +0.0300] | 600 | 0.9155 | no |
| entailment | random_frame | -0.0100 [-0.0333, +0.0117] | 600 | 0.9155 | no |
| paraphrase | none | -0.0046 [-0.0235, +0.0144] | 1529 | 1.0000 | no |
| paraphrase | mean_row | +0.0026 [-0.0144, +0.0196] | 1529 | 1.0000 | no |
| paraphrase | random_frame | -0.0137 [-0.0320, +0.0052] | 1529 | 0.5127 | no |
| statement_accuracy | none | -0.0052 [-0.0151, +0.0046] | 1529 | 0.3718 | no |
| statement_accuracy | mean_row | -0.0078 [-0.0177, +0.0020] | 1529 | 0.3718 | no |
| statement_accuracy | random_frame | -0.0078 [-0.0183, +0.0026] | 1529 | 0.3718 | no |
| statement_margin | none | -0.0222 [-0.0412, -0.0025] | 1529 | 0.0780 | no |
| statement_margin | mean_row | +0.0023 [-0.0117, +0.0166] | 1529 | 1.0000 | no |
| statement_margin | random_frame | -0.0020 [-0.0218, +0.0161] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0601 [-0.0830, -0.0403] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.0057 [-0.0199, +0.0084] | 1529 | 0.8876 | no |
| statement_loss | random_frame | -0.0072 [-0.0268, +0.0122] | 1529 | 0.8876 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).
