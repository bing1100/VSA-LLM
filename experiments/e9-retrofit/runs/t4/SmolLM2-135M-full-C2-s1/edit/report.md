# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1885 | 0.1752 | 0.2067 | 0.4900 | 0.3572 | 0.3052 | -0.9818 | 3.3189 |
| none | 0.1850 | 0.1715 | 0.2033 | 0.5083 | 0.3615 | 0.2996 | -0.9796 | 3.3171 |
| mean_row | 0.1835 | 0.1679 | 0.2050 | 0.4983 | 0.3671 | 0.2954 | -0.9818 | 3.3188 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0035 [-0.0035, +0.0113] | 711 | 0.3838 | no |
| property | mean_row | +0.0049 [-0.0021, +0.0120] | 711 | 0.3838 | no |
| property_new | none | +0.0036 [-0.0049, +0.0134] | 411 | 0.5447 | no |
| property_new | mean_row | +0.0073 [-0.0012, +0.0158] | 411 | 0.3138 | no |
| property_category | none | +0.0033 [-0.0100, +0.0167] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0017 [-0.0100, +0.0133] | 300 | 1.0000 | no |
| entailment | none | -0.0183 [-0.0367, -0.0017] | 600 | 0.0940 | no |
| entailment | mean_row | -0.0083 [-0.0250, +0.0067] | 600 | 0.3468 | no |
| paraphrase | none | -0.0042 [-0.0225, +0.0155] | 711 | 0.7296 | no |
| paraphrase | mean_row | -0.0098 [-0.0295, +0.0084] | 711 | 0.6337 | no |
| statement_accuracy | none | +0.0056 [-0.0014, +0.0127] | 711 | 0.1419 | no |
| statement_accuracy | mean_row | +0.0098 [+0.0014, +0.0183] | 711 | 0.0380 | yes |
| statement_margin | none | -0.0022 [-0.0045, +0.0001] | 711 | 0.1399 | no |
| statement_margin | mean_row | +0.0000 [-0.0024, +0.0025] | 711 | 0.9855 | no |
| statement_loss | none | +0.0018 [-0.0001, +0.0037] | 711 | 0.1279 | no |
| statement_loss | mean_row | +0.0001 [-0.0017, +0.0019] | 711 | 0.8646 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.355 → 0.355 (0.355) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.350 | +0.0000 [+0.0000, +0.0000] | 0.685 | 0.0000 / 0.0000 | 0.421 |
| seen | 100 | 0.385 → 0.385 (0.385) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.370 | +0.0000 [+0.0000, +0.0000] | 0.685 | 0.0000 / 0.0000 | 0.444 |
| heldout | 100 | 0.325 → 0.325 (0.325) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.330 | +0.0000 [+0.0000, +0.0000] | 0.685 | 0.0000 / 0.0000 | 0.396 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
