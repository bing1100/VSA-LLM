# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2842 | 0.1559 | 0.3700 | 0.5833 | 0.5145 | 0.2413 | -1.8238 | 9.2370 |
| none | 0.2857 | 0.1608 | 0.3692 | 0.5817 | 0.5215 | 0.2423 | -1.8233 | 9.2367 |
| mean_row | 0.2852 | 0.1596 | 0.3692 | 0.5850 | 0.5135 | 0.2433 | -1.8231 | 9.2372 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0015 [-0.0060, +0.0030] | 1001 | 1.0000 | no |
| property | mean_row | -0.0010 [-0.0050, +0.0030] | 1001 | 1.0000 | no |
| property_new | none | -0.0050 [-0.0125, +0.0012] | 401 | 0.4018 | no |
| property_new | mean_row | -0.0037 [-0.0100, +0.0012] | 401 | 0.4018 | no |
| property_category | none | +0.0008 [-0.0050, +0.0075] | 600 | 1.0000 | no |
| property_category | mean_row | +0.0008 [-0.0058, +0.0075] | 600 | 1.0000 | no |
| entailment | none | +0.0017 [-0.0033, +0.0067] | 600 | 1.0000 | no |
| entailment | mean_row | -0.0017 [-0.0083, +0.0050] | 600 | 1.0000 | no |
| paraphrase | none | -0.0070 [-0.0180, +0.0050] | 1001 | 0.5577 | no |
| paraphrase | mean_row | +0.0010 [-0.0100, +0.0120] | 1001 | 0.9425 | no |
| statement_accuracy | none | -0.0010 [-0.0030, +0.0010] | 1001 | 0.4678 | no |
| statement_accuracy | mean_row | -0.0020 [-0.0040, -0.0005] | 1001 | 0.0780 | no |
| statement_margin | none | -0.0005 [-0.0027, +0.0017] | 1001 | 1.0000 | no |
| statement_margin | mean_row | -0.0007 [-0.0029, +0.0015] | 1001 | 1.0000 | no |
| statement_loss | none | +0.0002 [-0.0015, +0.0019] | 1001 | 1.0000 | no |
| statement_loss | mean_row | -0.0002 [-0.0018, +0.0015] | 1001 | 1.0000 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.292 → 0.292 (0.292) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.302 | +0.0000 [+0.0000, +0.0000] | 0.679 | 0.0000 / 0.0000 | 0.366 |
| seen | 100 | 0.315 → 0.315 (0.315) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.310 | +0.0000 [+0.0000, +0.0000] | 0.685 | 0.0000 / 0.0000 | 0.382 |
| heldout | 100 | 0.270 → 0.270 (0.270) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.295 | +0.0000 [+0.0000, +0.0000] | 0.672 | 0.0000 / 0.0000 | 0.350 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
