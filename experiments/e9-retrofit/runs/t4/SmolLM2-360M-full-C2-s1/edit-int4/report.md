# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1955 | 0.1813 | 0.2150 | 0.4667 | 0.2813 | 0.2883 | -1.0454 | 3.6635 |
| none | 0.1962 | 0.1837 | 0.2133 | 0.4583 | 0.2883 | 0.2869 | -1.0479 | 3.6648 |
| mean_row | 0.1990 | 0.1861 | 0.2167 | 0.4600 | 0.2869 | 0.2869 | -1.0477 | 3.6653 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0007 [-0.0070, +0.0056] | 711 | 0.8926 | no |
| property | mean_row | -0.0035 [-0.0113, +0.0042] | 711 | 0.8536 | no |
| property_new | none | -0.0024 [-0.0109, +0.0061] | 411 | 0.6537 | no |
| property_new | mean_row | -0.0049 [-0.0134, +0.0036] | 411 | 0.5857 | no |
| property_category | none | +0.0017 [-0.0083, +0.0117] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0017 [-0.0167, +0.0133] | 300 | 1.0000 | no |
| entailment | none | +0.0083 [-0.0033, +0.0217] | 600 | 0.4638 | no |
| entailment | mean_row | +0.0067 [-0.0033, +0.0167] | 600 | 0.4638 | no |
| paraphrase | none | -0.0070 [-0.0225, +0.0084] | 711 | 0.8956 | no |
| paraphrase | mean_row | -0.0056 [-0.0211, +0.0098] | 711 | 0.8956 | no |
| statement_accuracy | none | +0.0014 [-0.0042, +0.0070] | 711 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0014 [+0.0000, +0.0042] | 711 | 1.0000 | no |
| statement_margin | none | +0.0024 [+0.0006, +0.0044] | 711 | 0.0200 | yes |
| statement_margin | mean_row | +0.0022 [+0.0005, +0.0040] | 711 | 0.0200 | yes |
| statement_loss | none | -0.0013 [-0.0030, +0.0003] | 711 | 0.1000 | no |
| statement_loss | mean_row | -0.0018 [-0.0033, -0.0003] | 711 | 0.0420 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.325 → 0.325 (0.325) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.320 | +0.0000 [+0.0000, +0.0000] | 0.729 | 0.0000 / 0.0000 | 0.396 |
| seen | 100 | 0.345 → 0.345 (0.345) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.360 | +0.0000 [+0.0000, +0.0000] | 0.715 | 0.0000 / 0.0000 | 0.424 |
| heldout | 100 | 0.305 → 0.305 (0.305) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.280 | +0.0000 [+0.0000, +0.0000] | 0.743 | 0.0000 / 0.0000 | 0.366 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
