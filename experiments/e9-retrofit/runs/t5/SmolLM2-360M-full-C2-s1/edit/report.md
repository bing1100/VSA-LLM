# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1939 | 0.2006 | 0.1667 | 0.4933 | 0.3440 | 0.2230 | -1.0261 | 4.5969 |
| none | 0.1929 | 0.1973 | 0.1750 | 0.5050 | 0.3375 | 0.2263 | -1.0251 | 4.5954 |
| mean_row | 0.1933 | 0.1981 | 0.1733 | 0.4983 | 0.3381 | 0.2243 | -1.0255 | 4.5959 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0010 [-0.0023, +0.0043] | 1529 | 1.0000 | no |
| property | mean_row | +0.0007 [-0.0023, +0.0033] | 1529 | 1.0000 | no |
| property_new | none | +0.0033 [+0.0000, +0.0069] | 1229 | 0.1659 | no |
| property_new | mean_row | +0.0024 [+0.0000, +0.0053] | 1229 | 0.1659 | no |
| property_category | none | -0.0083 [-0.0183, +0.0000] | 300 | 0.1419 | no |
| property_category | mean_row | -0.0067 [-0.0150, +0.0000] | 300 | 0.1419 | no |
| entailment | none | -0.0117 [-0.0233, -0.0017] | 600 | 0.0880 | no |
| entailment | mean_row | -0.0050 [-0.0133, +0.0033] | 600 | 0.3318 | no |
| paraphrase | none | +0.0065 [-0.0026, +0.0157] | 1529 | 0.2819 | no |
| paraphrase | mean_row | +0.0059 [-0.0020, +0.0137] | 1529 | 0.2819 | no |
| statement_accuracy | none | -0.0033 [-0.0078, +0.0013] | 1529 | 0.4098 | no |
| statement_accuracy | mean_row | -0.0013 [-0.0052, +0.0020] | 1529 | 0.5977 | no |
| statement_margin | none | -0.0010 [-0.0034, +0.0013] | 1529 | 0.7816 | no |
| statement_margin | mean_row | -0.0006 [-0.0029, +0.0016] | 1529 | 0.7816 | no |
| statement_loss | none | +0.0015 [-0.0002, +0.0033] | 1529 | 0.1659 | no |
| statement_loss | mean_row | +0.0010 [-0.0005, +0.0027] | 1529 | 0.2329 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.453 → 0.453 (0.453) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.465 | +0.0000 [+0.0000, +0.0000] | 0.530 | 0.0000 / 0.0000 | 0.480 |
| seen | 100 | 0.455 → 0.455 (0.455) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.525 | 0.0000 / 0.0000 | 0.478 |
| heldout | 100 | 0.450 → 0.450 (0.450) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.535 | 0.0000 / 0.0000 | 0.482 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
