# E9 dimension 3 — zero-shot by ontology editing — C5tr seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5tr-s3`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3123 | 0.3023 | 0.3533 | 0.5850 | 0.4722 | 0.3028 | -0.8986 | 4.4663 |
| none | 0.1906 | 0.1953 | 0.1717 | 0.5183 | 0.3244 | 0.2269 | -1.0753 | 4.8494 |
| mean_row | 0.1949 | 0.1859 | 0.2317 | 0.4967 | 0.4559 | 0.2171 | -1.2424 | 4.7397 |
| random_frame | 0.2152 | 0.1969 | 0.2900 | 0.5100 | 0.4657 | 0.2446 | -1.1553 | 4.7098 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.1216 [+0.1033, +0.1400] | 1529 | 0.0030 | yes |
| property | mean_row | +0.1174 [+0.1017, +0.1334] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0971 [+0.0811, +0.1141] | 1529 | 0.0030 | yes |
| property_new | none | +0.1070 [+0.0862, +0.1282] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.1164 [+0.0997, +0.1338] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.1054 [+0.0871, +0.1245] | 1229 | 0.0030 | yes |
| property_category | none | +0.1817 [+0.1333, +0.2300] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.1217 [+0.0866, +0.1583] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0633 [+0.0250, +0.1033] | 300 | 0.0030 | yes |
| entailment | none | +0.0667 [+0.0267, +0.1083] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.0883 [+0.0567, +0.1200] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0750 [+0.0450, +0.1050] | 600 | 0.0030 | yes |
| paraphrase | none | +0.1478 [+0.1158, +0.1785] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0164 [-0.0118, +0.0438] | 1529 | 0.4858 | no |
| paraphrase | random_frame | +0.0065 [-0.0222, +0.0334] | 1529 | 0.6407 | no |
| statement_accuracy | none | +0.0759 [+0.0549, +0.0981] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0857 [+0.0674, +0.1034] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0582 [+0.0419, +0.0765] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1767 [+0.1268, +0.2310] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.3438 [+0.2989, +0.3894] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.2567 [+0.2120, +0.3045] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.3831 [-0.4431, -0.3289] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.2734 [-0.3179, -0.2311] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.2435 [-0.2944, -0.1970] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.438 → 0.468 (0.453) | +0.4467 [+0.2736, +0.6797] | +0.2144 [+0.0979, +0.3654] | 0.470 | +0.3300 [+0.1988, +0.5050] | 0.544 | 0.0002 / 0.0280 | 0.491 |
| seen | 100 | 0.400 → 0.460 (0.430) | +0.7176 [+0.3651, +1.1403] | +0.3541 [+0.1336, +0.6216] | 0.460 | +0.5137 [+0.2606, +0.8210] | 0.547 | 0.0002 / 0.0244 | 0.486 |
| heldout | 100 | 0.475 → 0.475 (0.475) | +0.1758 [+0.0954, +0.2684] | +0.0747 [-0.0044, +0.1625] | 0.480 | +0.1464 [+0.0580, +0.2400] | 0.540 | 0.0002 / 0.0280 | 0.497 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
