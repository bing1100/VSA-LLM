# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C2-s2`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1975 | 0.2014 | 0.1817 | 0.5100 | 0.3401 | 0.2237 | -1.0264 | 4.6069 |
| none | 0.1975 | 0.2006 | 0.1850 | 0.5167 | 0.3355 | 0.2237 | -1.0276 | 4.6073 |
| mean_row | 0.1946 | 0.1989 | 0.1767 | 0.5067 | 0.3336 | 0.2250 | -1.0281 | 4.6076 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0000 [-0.0039, +0.0039] | 1529 | 1.0000 | no |
| property | mean_row | +0.0029 [-0.0010, +0.0069] | 1529 | 0.3458 | no |
| property_new | none | +0.0008 [-0.0037, +0.0049] | 1229 | 0.7666 | no |
| property_new | mean_row | +0.0024 [-0.0016, +0.0065] | 1229 | 0.5837 | no |
| property_category | none | -0.0033 [-0.0150, +0.0083] | 300 | 0.9255 | no |
| property_category | mean_row | +0.0050 [-0.0050, +0.0150] | 300 | 0.9255 | no |
| entailment | none | -0.0067 [-0.0200, +0.0050] | 600 | 0.6677 | no |
| entailment | mean_row | +0.0033 [-0.0067, +0.0133] | 600 | 0.6677 | no |
| paraphrase | none | +0.0046 [-0.0039, +0.0131] | 1529 | 0.3088 | no |
| paraphrase | mean_row | +0.0065 [-0.0013, +0.0144] | 1529 | 0.2239 | no |
| statement_accuracy | none | +0.0000 [-0.0033, +0.0033] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | -0.0013 [-0.0052, +0.0026] | 1529 | 1.0000 | no |
| statement_margin | none | +0.0012 [-0.0010, +0.0034] | 1529 | 0.2659 | no |
| statement_margin | mean_row | +0.0018 [-0.0003, +0.0039] | 1529 | 0.2079 | no |
| statement_loss | none | -0.0003 [-0.0021, +0.0014] | 1529 | 0.8236 | no |
| statement_loss | mean_row | -0.0007 [-0.0024, +0.0009] | 1529 | 0.8236 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.448 → 0.448 (0.448) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.465 | +0.0000 [+0.0000, +0.0000] | 0.531 | 0.0000 / 0.0000 | 0.479 |
| seen | 100 | 0.445 → 0.445 (0.445) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.532 | 0.0000 / 0.0000 | 0.476 |
| heldout | 100 | 0.450 → 0.450 (0.450) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.530 | 0.0000 / 0.0000 | 0.481 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
