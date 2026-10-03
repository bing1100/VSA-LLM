# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2423 | 0.2628 | 0.1583 | 0.5233 | 0.4356 | 0.2583 | -1.0833 | 5.1772 |
| none | 0.1949 | 0.2083 | 0.1400 | 0.4867 | 0.4199 | 0.2165 | -1.2883 | 5.8488 |
| mean_row | 0.1920 | 0.2091 | 0.1217 | 0.4817 | 0.4349 | 0.2152 | -1.2906 | 5.2870 |
| random_frame | 0.1920 | 0.2116 | 0.1117 | 0.4917 | 0.4500 | 0.2204 | -1.2841 | 5.3444 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0474 [+0.0311, +0.0641] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0504 [+0.0370, +0.0638] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0504 [+0.0353, +0.0644] | 1529 | 0.0030 | yes |
| property_new | none | +0.0545 [+0.0354, +0.0728] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0537 [+0.0386, +0.0692] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0513 [+0.0342, +0.0683] | 1229 | 0.0030 | yes |
| property_category | none | +0.0183 [-0.0183, +0.0550] | 300 | 0.3468 | no |
| property_category | mean_row | +0.0367 [+0.0100, +0.0633] | 300 | 0.0180 | yes |
| property_category | random_frame | +0.0467 [+0.0150, +0.0784] | 300 | 0.0150 | yes |
| entailment | none | +0.0367 [+0.0033, +0.0683] | 600 | 0.0620 | no |
| entailment | mean_row | +0.0417 [+0.0150, +0.0683] | 600 | 0.0060 | yes |
| entailment | random_frame | +0.0317 [+0.0017, +0.0617] | 600 | 0.0620 | no |
| paraphrase | none | +0.0157 [-0.0124, +0.0432] | 1529 | 0.8126 | no |
| paraphrase | mean_row | +0.0007 [-0.0255, +0.0275] | 1529 | 1.0000 | no |
| paraphrase | random_frame | -0.0144 [-0.0419, +0.0124] | 1529 | 0.8126 | no |
| statement_accuracy | none | +0.0419 [+0.0262, +0.0576] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0432 [+0.0301, +0.0569] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0379 [+0.0229, +0.0530] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.2049 [+0.1677, +0.2438] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.2073 [+0.1778, +0.2366] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.2007 [+0.1661, +0.2366] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.6717 [-0.7201, -0.6216] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1098 [-0.1392, -0.0812] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1673 [-0.2014, -0.1306] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.438 → 0.495 (0.472) | +0.8880 [+0.7090, +1.0877] | +0.3742 [+0.2566, +0.5001] | 0.470 | +0.8888 [+0.7207, +1.0735] | 0.530 | 0.0012 / 0.1095 | 0.497 |
| seen | 100 | 0.425 → 0.520 (0.475) | +1.1892 [+0.8811, +1.5228] | +0.4963 [+0.3026, +0.7053] | 0.480 | +1.1719 [+0.8858, +1.4661] | 0.507 | 0.0006 / 0.1070 | 0.502 |
| heldout | 100 | 0.450 → 0.470 (0.470) | +0.5867 [+0.4269, +0.7524] | +0.2520 [+0.1216, +0.3900] | 0.460 | +0.6057 [+0.4357, +0.7868] | 0.552 | 0.0018 / 0.1095 | 0.491 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
