# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2443 | 0.1633 | 0.2983 | 0.4983 | 0.4745 | 0.2453 | -1.7758 | 8.6746 |
| none | 0.2468 | 0.1683 | 0.2992 | 0.4975 | 0.4725 | 0.2458 | -1.7769 | 8.6739 |
| mean_row | 0.2453 | 0.1696 | 0.2958 | 0.4950 | 0.4755 | 0.2453 | -1.7775 | 8.6733 |
| random_frame | 0.2463 | 0.1696 | 0.2975 | 0.5000 | 0.4685 | 0.2463 | -1.7777 | 8.6742 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0025 [-0.0075, +0.0025] | 1001 | 1.0000 | no |
| property | mean_row | -0.0010 [-0.0055, +0.0035] | 1001 | 1.0000 | no |
| property | random_frame | -0.0020 [-0.0070, +0.0030] | 1001 | 1.0000 | no |
| property_new | none | -0.0050 [-0.0112, +0.0012] | 401 | 0.2599 | no |
| property_new | mean_row | -0.0062 [-0.0125, +0.0000] | 401 | 0.1949 | no |
| property_new | random_frame | -0.0062 [-0.0137, +0.0000] | 401 | 0.2599 | no |
| property_category | none | -0.0008 [-0.0075, +0.0058] | 600 | 1.0000 | no |
| property_category | mean_row | +0.0025 [-0.0033, +0.0092] | 600 | 1.0000 | no |
| property_category | random_frame | +0.0008 [-0.0050, +0.0067] | 600 | 1.0000 | no |
| entailment | none | +0.0008 [-0.0058, +0.0075] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0033 [-0.0017, +0.0083] | 600 | 0.7886 | no |
| entailment | random_frame | -0.0017 [-0.0067, +0.0042] | 600 | 1.0000 | no |
| paraphrase | none | +0.0020 [-0.0090, +0.0130] | 1001 | 1.0000 | no |
| paraphrase | mean_row | -0.0010 [-0.0110, +0.0090] | 1001 | 1.0000 | no |
| paraphrase | random_frame | +0.0060 [-0.0050, +0.0180] | 1001 | 1.0000 | no |
| statement_accuracy | none | -0.0005 [-0.0025, +0.0015] | 1001 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0000 [-0.0020, +0.0020] | 1001 | 1.0000 | no |
| statement_accuracy | random_frame | -0.0010 [-0.0030, +0.0005] | 1001 | 1.0000 | no |
| statement_margin | none | +0.0011 [-0.0013, +0.0037] | 1001 | 0.5817 | no |
| statement_margin | mean_row | +0.0017 [-0.0010, +0.0044] | 1001 | 0.5817 | no |
| statement_margin | random_frame | +0.0019 [-0.0013, +0.0049] | 1001 | 0.5817 | no |
| statement_loss | none | +0.0006 [-0.0016, +0.0028] | 1001 | 1.0000 | no |
| statement_loss | mean_row | +0.0013 [-0.0010, +0.0035] | 1001 | 0.8096 | no |
| statement_loss | random_frame | +0.0004 [-0.0021, +0.0028] | 1001 | 1.0000 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.253 → 0.253 (0.250) | +0.0081 [-0.0171, +0.0333] | +0.0099 [-0.0187, +0.0383] | 0.235 | +0.0163 [-0.0032, +0.0367] | 0.744 | 0.0002 / 0.0469 | 0.314 |
| seen | 100 | 0.270 → 0.270 (0.265) | +0.0306 [-0.0048, +0.0674] | +0.0187 [-0.0154, +0.0560] | 0.245 | +0.0215 [-0.0109, +0.0608] | 0.748 | 0.0003 / 0.0469 | 0.329 |
| heldout | 100 | 0.235 → 0.235 (0.235) | -0.0144 [-0.0511, +0.0214] | +0.0011 [-0.0437, +0.0463] | 0.225 | +0.0110 [-0.0092, +0.0313] | 0.740 | 0.0000 / 0.0000 | 0.298 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
