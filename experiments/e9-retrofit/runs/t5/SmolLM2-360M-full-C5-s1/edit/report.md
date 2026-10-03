# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2492 | 0.2522 | 0.2367 | 0.5367 | 0.3865 | 0.2564 | -0.9957 | 4.5431 |
| none | 0.1946 | 0.1985 | 0.1783 | 0.5117 | 0.3479 | 0.2256 | -1.0629 | 4.7536 |
| mean_row | 0.1962 | 0.1989 | 0.1850 | 0.4950 | 0.3669 | 0.2178 | -1.1093 | 4.6458 |
| random_frame | 0.2112 | 0.2091 | 0.2200 | 0.5000 | 0.4016 | 0.2328 | -1.1159 | 4.6415 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0546 [+0.0402, +0.0687] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0530 [+0.0409, +0.0661] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0379 [+0.0255, +0.0520] | 1529 | 0.0030 | yes |
| property_new | none | +0.0537 [+0.0378, +0.0696] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0533 [+0.0395, +0.0683] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0431 [+0.0289, +0.0574] | 1229 | 0.0030 | yes |
| property_category | none | +0.0583 [+0.0250, +0.0950] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.0517 [+0.0200, +0.0867] | 300 | 0.0040 | yes |
| property_category | random_frame | +0.0167 [-0.0183, +0.0550] | 300 | 0.3978 | no |
| entailment | none | +0.0250 [-0.0034, +0.0567] | 600 | 0.1199 | no |
| entailment | mean_row | +0.0417 [+0.0150, +0.0683] | 600 | 0.0180 | yes |
| entailment | random_frame | +0.0367 [+0.0050, +0.0650] | 600 | 0.0400 | yes |
| paraphrase | none | +0.0386 [+0.0137, +0.0621] | 1529 | 0.0090 | yes |
| paraphrase | mean_row | +0.0196 [-0.0046, +0.0425] | 1529 | 0.2259 | no |
| paraphrase | random_frame | -0.0150 [-0.0392, +0.0092] | 1529 | 0.2339 | no |
| statement_accuracy | none | +0.0307 [+0.0157, +0.0477] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0386 [+0.0255, +0.0530] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0235 [+0.0098, +0.0386] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0673 [+0.0361, +0.1013] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1137 [+0.0847, +0.1418] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1203 [+0.0887, +0.1527] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2105 [-0.2467, -0.1782] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1027 [-0.1303, -0.0763] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.0984 [-0.1290, -0.0695] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.427 → 0.453 (0.453) | +0.3661 [+0.1896, +0.5947] | +0.1308 [+0.0224, +0.2553] | 0.485 | +0.2744 [+0.1399, +0.4365] | 0.547 | 0.0010 / 0.1459 | 0.492 |
| seen | 100 | 0.400 → 0.455 (0.450) | +0.6927 [+0.3412, +1.1008] | +0.2697 [+0.0876, +0.4739] | 0.470 | +0.4523 [+0.1879, +0.7586] | 0.547 | 0.0014 / 0.1459 | 0.488 |
| heldout | 100 | 0.455 → 0.450 (0.455) | +0.0394 [-0.0488, +0.1216] | -0.0082 [-0.0987, +0.0768] | 0.500 | +0.0965 [+0.0155, +0.1760] | 0.547 | 0.0005 / 0.0936 | 0.496 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
