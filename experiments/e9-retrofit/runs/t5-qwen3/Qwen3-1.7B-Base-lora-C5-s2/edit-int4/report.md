# E9 dimension 3 — zero-shot by ontology editing — C5 seed 2 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C5-s2`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2626 | 0.2587 | 0.2783 | 0.5667 | 0.3663 | 0.2989 | -1.0451 | 5.4278 |
| none | 0.2129 | 0.2164 | 0.1983 | 0.4833 | 0.3322 | 0.2191 | -1.1852 | 5.8959 |
| mean_row | 0.2083 | 0.2168 | 0.1733 | 0.4833 | 0.3564 | 0.2184 | -1.5217 | 5.8249 |
| random_frame | 0.2073 | 0.2140 | 0.1800 | 0.4783 | 0.3525 | 0.2283 | -1.5109 | 5.8492 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0497 [+0.0330, +0.0667] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0543 [+0.0419, +0.0664] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0553 [+0.0412, +0.0697] | 1529 | 0.0030 | yes |
| property_new | none | +0.0423 [+0.0240, +0.0598] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0419 [+0.0285, +0.0553] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0448 [+0.0293, +0.0594] | 1229 | 0.0030 | yes |
| property_category | none | +0.0800 [+0.0350, +0.1233] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.1050 [+0.0767, +0.1350] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.0983 [+0.0667, +0.1333] | 300 | 0.0030 | yes |
| entailment | none | +0.0833 [+0.0450, +0.1233] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.0833 [+0.0550, +0.1117] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0883 [+0.0583, +0.1183] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0340 [+0.0059, +0.0628] | 1529 | 0.0510 | no |
| paraphrase | mean_row | +0.0098 [-0.0124, +0.0320] | 1529 | 0.4638 | no |
| paraphrase | random_frame | +0.0137 [-0.0105, +0.0379] | 1529 | 0.4638 | no |
| statement_accuracy | none | +0.0798 [+0.0582, +0.1014] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0804 [+0.0634, +0.0974] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0706 [+0.0523, +0.0889] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1402 [+0.0667, +0.2176] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.4767 [+0.4212, +0.5357] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.4659 [+0.4022, +0.5355] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.4681 [-0.5731, -0.3632] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.3970 [-0.4426, -0.3523] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.4214 [-0.4799, -0.3628] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.480 → 0.510 (0.497) | +0.3694 [+0.1436, +0.6702] | +0.1673 [+0.0087, +0.3665] | 0.460 | +0.1307 [+0.0050, +0.2780] | 0.516 | 0.0037 / 0.3411 | 0.494 |
| seen | 100 | 0.475 → 0.520 (0.500) | +0.7650 [+0.3198, +1.2886] | +0.3671 [+0.0810, +0.7195] | 0.470 | +0.2771 [+0.0523, +0.5243] | 0.505 | 0.0055 / 0.3411 | 0.497 |
| heldout | 100 | 0.485 → 0.500 (0.495) | -0.0261 [-0.1289, +0.0739] | -0.0326 [-0.1339, +0.0695] | 0.450 | -0.0157 [-0.1581, +0.1242] | 0.527 | 0.0018 / 0.1578 | 0.490 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
