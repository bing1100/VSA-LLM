# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3117 | 0.1683 | 0.4075 | 0.5817 | 0.5764 | 0.2562 | -1.7014 | 8.4506 |
| none | 0.3112 | 0.1671 | 0.4075 | 0.5825 | 0.5774 | 0.2552 | -1.7016 | 8.4502 |
| mean_row | 0.3107 | 0.1621 | 0.4100 | 0.5850 | 0.5804 | 0.2542 | -1.7016 | 8.4503 |
| random_frame | 0.3097 | 0.1658 | 0.4058 | 0.5833 | 0.5704 | 0.2562 | -1.7034 | 8.4519 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0005 [-0.0035, +0.0045] | 1001 | 1.0000 | no |
| property | mean_row | +0.0010 [-0.0040, +0.0060] | 1001 | 1.0000 | no |
| property | random_frame | +0.0020 [-0.0025, +0.0070] | 1001 | 1.0000 | no |
| property_new | none | +0.0012 [-0.0050, +0.0075] | 401 | 1.0000 | no |
| property_new | mean_row | +0.0062 [+0.0012, +0.0125] | 401 | 0.1529 | no |
| property_new | random_frame | +0.0025 [-0.0050, +0.0100] | 401 | 1.0000 | no |
| property_category | none | +0.0000 [-0.0058, +0.0058] | 600 | 1.0000 | no |
| property_category | mean_row | -0.0025 [-0.0092, +0.0042] | 600 | 1.0000 | no |
| property_category | random_frame | +0.0017 [-0.0042, +0.0075] | 600 | 1.0000 | no |
| entailment | none | -0.0008 [-0.0075, +0.0050] | 600 | 1.0000 | no |
| entailment | mean_row | -0.0033 [-0.0100, +0.0025] | 600 | 0.9745 | no |
| entailment | random_frame | -0.0017 [-0.0058, +0.0025] | 600 | 1.0000 | no |
| paraphrase | none | -0.0010 [-0.0100, +0.0080] | 1001 | 1.0000 | no |
| paraphrase | mean_row | -0.0040 [-0.0140, +0.0060] | 1001 | 1.0000 | no |
| paraphrase | random_frame | +0.0060 [-0.0040, +0.0160] | 1001 | 0.8786 | no |
| statement_accuracy | none | +0.0010 [-0.0020, +0.0040] | 1001 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0020 [-0.0005, +0.0050] | 1001 | 0.5997 | no |
| statement_accuracy | random_frame | +0.0000 [-0.0035, +0.0035] | 1001 | 1.0000 | no |
| statement_margin | none | +0.0001 [-0.0023, +0.0024] | 1001 | 1.0000 | no |
| statement_margin | mean_row | +0.0001 [-0.0023, +0.0024] | 1001 | 1.0000 | no |
| statement_margin | random_frame | +0.0019 [-0.0009, +0.0046] | 1001 | 0.5397 | no |
| statement_loss | none | +0.0004 [-0.0015, +0.0025] | 1001 | 1.0000 | no |
| statement_loss | mean_row | +0.0003 [-0.0017, +0.0024] | 1001 | 1.0000 | no |
| statement_loss | random_frame | -0.0013 [-0.0035, +0.0011] | 1001 | 0.8906 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.172 → 0.177 (0.175) | +0.0045 [-0.0113, +0.0206] | +0.0071 [-0.0104, +0.0240] | 0.182 | -0.0098 [-0.0221, +0.0029] | 0.804 | 0.0000 / 0.0097 | 0.243 |
| seen | 100 | 0.205 → 0.205 (0.200) | +0.0040 [-0.0159, +0.0265] | +0.0124 [-0.0108, +0.0365] | 0.210 | -0.0082 [-0.0247, +0.0078] | 0.807 | 0.0000 / 0.0000 | 0.276 |
| heldout | 100 | 0.140 → 0.150 (0.150) | +0.0049 [-0.0198, +0.0308] | +0.0018 [-0.0229, +0.0273] | 0.155 | -0.0113 [-0.0297, +0.0074] | 0.800 | 0.0001 / 0.0097 | 0.209 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
