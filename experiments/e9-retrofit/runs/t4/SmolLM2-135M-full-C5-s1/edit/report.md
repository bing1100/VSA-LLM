# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1835 | 0.1691 | 0.2033 | 0.5017 | 0.3544 | 0.3080 | -0.9627 | 3.2743 |
| none | 0.1800 | 0.1655 | 0.2000 | 0.4950 | 0.3586 | 0.3010 | -0.9720 | 3.2937 |
| mean_row | 0.1871 | 0.1764 | 0.2017 | 0.5083 | 0.3713 | 0.3010 | -0.9686 | 3.2802 |
| random_frame | 0.1835 | 0.1727 | 0.1983 | 0.4850 | 0.3755 | 0.3010 | -0.9684 | 3.2795 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0035 [-0.0091, +0.0155] | 711 | 1.0000 | no |
| property | mean_row | -0.0035 [-0.0148, +0.0070] | 711 | 1.0000 | no |
| property | random_frame | +0.0000 [-0.0113, +0.0113] | 711 | 1.0000 | no |
| property_new | none | +0.0036 [-0.0110, +0.0195] | 411 | 1.0000 | no |
| property_new | mean_row | -0.0073 [-0.0207, +0.0073] | 411 | 1.0000 | no |
| property_new | random_frame | -0.0036 [-0.0182, +0.0109] | 411 | 1.0000 | no |
| property_category | none | +0.0033 [-0.0133, +0.0200] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0017 [-0.0167, +0.0200] | 300 | 1.0000 | no |
| property_category | random_frame | +0.0050 [-0.0133, +0.0233] | 300 | 1.0000 | no |
| entailment | none | +0.0067 [-0.0133, +0.0267] | 600 | 1.0000 | no |
| entailment | mean_row | -0.0067 [-0.0267, +0.0133] | 600 | 1.0000 | no |
| entailment | random_frame | +0.0167 [-0.0033, +0.0383] | 600 | 0.4138 | no |
| paraphrase | none | -0.0042 [-0.0309, +0.0225] | 711 | 0.7916 | no |
| paraphrase | mean_row | -0.0169 [-0.0422, +0.0113] | 711 | 0.4778 | no |
| paraphrase | random_frame | -0.0211 [-0.0478, +0.0056] | 711 | 0.3928 | no |
| statement_accuracy | none | +0.0070 [+0.0000, +0.0141] | 711 | 0.1219 | no |
| statement_accuracy | mean_row | +0.0070 [+0.0014, +0.0141] | 711 | 0.0360 | yes |
| statement_accuracy | random_frame | +0.0070 [+0.0000, +0.0141] | 711 | 0.1219 | no |
| statement_margin | none | +0.0092 [+0.0038, +0.0145] | 711 | 0.0030 | yes |
| statement_margin | mean_row | +0.0059 [+0.0012, +0.0107] | 711 | 0.0200 | yes |
| statement_margin | random_frame | +0.0057 [-0.0003, +0.0117] | 711 | 0.0610 | no |
| statement_loss | none | -0.0194 [-0.0249, -0.0140] | 711 | 0.0030 | yes |
| statement_loss | mean_row | -0.0059 [-0.0107, -0.0012] | 711 | 0.0240 | yes |
| statement_loss | random_frame | -0.0052 [-0.0113, +0.0010] | 711 | 0.1049 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.360 → 0.362 (0.367) | +0.0504 [-0.0261, +0.1177] | +0.0506 [-0.0177, +0.1185] | 0.370 | +0.0688 [-0.0022, +0.1355] | 0.682 | 0.0410 / 1.7195 | 0.433 |
| seen | 100 | 0.390 → 0.395 (0.390) | +0.0745 [-0.0222, +0.1651] | +0.0181 [-0.0578, +0.0951] | 0.370 | +0.0880 [-0.0044, +0.1759] | 0.675 | 0.0351 / 1.7195 | 0.447 |
| heldout | 100 | 0.330 → 0.330 (0.345) | +0.0263 [-0.0787, +0.1256] | +0.0832 [-0.0319, +0.1931] | 0.370 | +0.0497 [-0.0573, +0.1555] | 0.690 | 0.0469 / 1.0851 | 0.418 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
