# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3112 | 0.1696 | 0.4058 | 0.5817 | 0.5764 | 0.2512 | -1.7035 | 8.4419 |
| none | 0.3117 | 0.1733 | 0.4042 | 0.5833 | 0.5754 | 0.2522 | -1.7043 | 8.4421 |
| mean_row | 0.3097 | 0.1696 | 0.4033 | 0.5808 | 0.5734 | 0.2527 | -1.7039 | 8.4422 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0005 [-0.0040, +0.0030] | 1001 | 1.0000 | no |
| property | mean_row | +0.0015 [-0.0025, +0.0060] | 1001 | 1.0000 | no |
| property_new | none | -0.0037 [-0.0087, +0.0000] | 401 | 0.1959 | no |
| property_new | mean_row | +0.0000 [-0.0050, +0.0050] | 401 | 1.0000 | no |
| property_category | none | +0.0017 [-0.0033, +0.0067] | 600 | 1.0000 | no |
| property_category | mean_row | +0.0025 [-0.0042, +0.0092] | 600 | 1.0000 | no |
| entailment | none | -0.0017 [-0.0042, +0.0000] | 600 | 0.5337 | no |
| entailment | mean_row | +0.0008 [-0.0025, +0.0050] | 600 | 0.8456 | no |
| paraphrase | none | +0.0010 [-0.0080, +0.0100] | 1001 | 1.0000 | no |
| paraphrase | mean_row | +0.0030 [-0.0070, +0.0130] | 1001 | 1.0000 | no |
| statement_accuracy | none | -0.0010 [-0.0035, +0.0015] | 1001 | 0.7396 | no |
| statement_accuracy | mean_row | -0.0015 [-0.0045, +0.0015] | 1001 | 0.7396 | no |
| statement_margin | none | +0.0008 [-0.0010, +0.0028] | 1001 | 0.7516 | no |
| statement_margin | mean_row | +0.0005 [-0.0014, +0.0025] | 1001 | 0.7516 | no |
| statement_loss | none | -0.0002 [-0.0017, +0.0013] | 1001 | 1.0000 | no |
| statement_loss | mean_row | -0.0003 [-0.0019, +0.0012] | 1001 | 1.0000 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.172 → 0.172 (0.172) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.185 | +0.0000 [+0.0000, +0.0000] | 0.800 | 0.0000 / 0.0000 | 0.241 |
| seen | 100 | 0.200 → 0.200 (0.200) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.210 | +0.0000 [+0.0000, +0.0000] | 0.807 | 0.0000 / 0.0000 | 0.273 |
| heldout | 100 | 0.145 → 0.145 (0.145) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.160 | +0.0000 [+0.0000, +0.0000] | 0.792 | 0.0000 / 0.0000 | 0.208 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
