# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (Qwen/Qwen3-1.7B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3362 | 0.3417 | 0.3133 | 0.5933 | 0.4467 | 0.3421 | -0.6683 | 4.6099 |
| none | 0.1808 | 0.1965 | 0.1167 | 0.4500 | 0.3852 | 0.2302 | -1.1214 | 4.4772 |
| mean_row | 0.1825 | 0.1924 | 0.1417 | 0.4683 | 0.4061 | 0.2296 | -1.3642 | 5.2003 |
| random_frame | 0.1956 | 0.1989 | 0.1817 | 0.4800 | 0.4284 | 0.2269 | -1.3303 | 5.2161 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.1553 [+0.1351, +0.1756] | 1529 | 0.0030 | yes |
| property | mean_row | +0.1537 [+0.1370, +0.1710] | 1529 | 0.0030 | yes |
| property | random_frame | +0.1406 [+0.1216, +0.1586] | 1529 | 0.0030 | yes |
| property_new | none | +0.1452 [+0.1225, +0.1692] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.1493 [+0.1294, +0.1709] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.1428 [+0.1221, +0.1648] | 1229 | 0.0030 | yes |
| property_category | none | +0.1967 [+0.1600, +0.2333] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.1717 [+0.1417, +0.2017] | 300 | 0.0030 | yes |
| property_category | random_frame | +0.1317 [+0.0983, +0.1633] | 300 | 0.0030 | yes |
| entailment | none | +0.1433 [+0.1050, +0.1817] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.1250 [+0.0883, +0.1600] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.1133 [+0.0767, +0.1500] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0615 [+0.0307, +0.0909] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0405 [+0.0137, +0.0667] | 1529 | 0.0060 | yes |
| paraphrase | random_frame | +0.0183 [-0.0092, +0.0451] | 1529 | 0.2099 | no |
| statement_accuracy | none | +0.1118 [+0.0909, +0.1341] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.1125 [+0.0935, +0.1321] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.1151 [+0.0955, +0.1360] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.4531 [+0.3762, +0.5351] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.6959 [+0.6193, +0.7743] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.6620 [+0.5795, +0.7452] | 1529 | 0.0030 | yes |
| statement_loss | none | +0.1327 [+0.0433, +0.2208] | 1529 | 0.0040 | no |
| statement_loss | mean_row | -0.5904 [-0.6562, -0.5275] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.6062 [-0.6888, -0.5300] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.395 → 0.445 (0.412) | +0.5862 [+0.2671, +0.9967] | +0.3068 [+0.1159, +0.5500] | 0.445 | +0.3037 [+0.1461, +0.5004] | 0.580 | 0.0005 / 0.1165 | 0.482 |
| seen | 100 | 0.390 → 0.475 (0.425) | +1.1229 [+0.4705, +1.8691] | +0.5634 [+0.1988, +0.9930] | 0.470 | +0.5590 [+0.2445, +0.9355] | 0.583 | 0.0008 / 0.1165 | 0.504 |
| heldout | 100 | 0.400 → 0.415 (0.400) | +0.0494 [+0.0070, +0.0913] | +0.0502 [-0.0045, +0.1079] | 0.420 | +0.0484 [-0.0227, +0.1228] | 0.578 | 0.0003 / 0.0628 | 0.460 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
