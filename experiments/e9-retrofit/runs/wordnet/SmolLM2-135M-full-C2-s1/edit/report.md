# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2443 | 0.1671 | 0.2958 | 0.4967 | 0.4665 | 0.2463 | -1.7716 | 8.6596 |
| none | 0.2433 | 0.1658 | 0.2950 | 0.4942 | 0.4715 | 0.2448 | -1.7706 | 8.6593 |
| mean_row | 0.2418 | 0.1683 | 0.2908 | 0.4967 | 0.4605 | 0.2453 | -1.7701 | 8.6583 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0010 [-0.0025, +0.0050] | 1001 | 0.6937 | no |
| property | mean_row | +0.0025 [-0.0015, +0.0070] | 1001 | 0.5297 | no |
| property_new | none | +0.0012 [+0.0000, +0.0037] | 401 | 1.0000 | no |
| property_new | mean_row | -0.0012 [-0.0062, +0.0025] | 401 | 1.0000 | no |
| property_category | none | +0.0008 [-0.0050, +0.0067] | 600 | 0.8826 | no |
| property_category | mean_row | +0.0050 [-0.0008, +0.0117] | 600 | 0.2459 | no |
| entailment | none | +0.0025 [-0.0017, +0.0067] | 600 | 0.7176 | no |
| entailment | mean_row | +0.0000 [-0.0042, +0.0042] | 600 | 1.0000 | no |
| paraphrase | none | -0.0050 [-0.0150, +0.0050] | 1001 | 0.5697 | no |
| paraphrase | mean_row | +0.0060 [-0.0050, +0.0170] | 1001 | 0.5697 | no |
| statement_accuracy | none | +0.0015 [-0.0010, +0.0040] | 1001 | 0.6617 | no |
| statement_accuracy | mean_row | +0.0010 [-0.0015, +0.0040] | 1001 | 0.6617 | no |
| statement_margin | none | -0.0011 [-0.0032, +0.0010] | 1001 | 0.3378 | no |
| statement_margin | mean_row | -0.0015 [-0.0036, +0.0007] | 1001 | 0.3378 | no |
| statement_loss | none | +0.0002 [-0.0014, +0.0019] | 1001 | 0.7556 | no |
| statement_loss | mean_row | +0.0013 [-0.0004, +0.0030] | 1001 | 0.2519 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.250 → 0.250 (0.250) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.233 | +0.0000 [+0.0000, +0.0000] | 0.744 | 0.0000 / 0.0000 | 0.311 |
| seen | 100 | 0.270 → 0.270 (0.270) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.240 | +0.0000 [+0.0000, +0.0000] | 0.748 | 0.0000 / 0.0000 | 0.326 |
| heldout | 100 | 0.230 → 0.230 (0.230) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.225 | +0.0000 [+0.0000, +0.0000] | 0.740 | 0.0000 / 0.0000 | 0.296 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
