# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1985 | 0.2148 | 0.1317 | 0.4733 | 0.4107 | 0.2119 | -1.2431 | 5.6041 |
| none | 0.1988 | 0.2148 | 0.1333 | 0.4733 | 0.4075 | 0.2171 | -1.2442 | 5.6121 |
| mean_row | 0.1991 | 0.2136 | 0.1400 | 0.4767 | 0.4153 | 0.2126 | -1.2417 | 5.6033 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0003 [-0.0049, +0.0043] | 1529 | 1.0000 | no |
| property | mean_row | -0.0007 [-0.0049, +0.0033] | 1529 | 1.0000 | no |
| property_new | none | +0.0000 [-0.0049, +0.0049] | 1229 | 1.0000 | no |
| property_new | mean_row | +0.0012 [-0.0033, +0.0057] | 1229 | 1.0000 | no |
| property_category | none | -0.0017 [-0.0100, +0.0067] | 300 | 0.8256 | no |
| property_category | mean_row | -0.0083 [-0.0183, +0.0017] | 300 | 0.2559 | no |
| entailment | none | +0.0000 [-0.0117, +0.0117] | 600 | 1.0000 | no |
| entailment | mean_row | -0.0033 [-0.0150, +0.0067] | 600 | 1.0000 | no |
| paraphrase | none | +0.0033 [-0.0072, +0.0150] | 1529 | 0.8736 | no |
| paraphrase | mean_row | -0.0046 [-0.0150, +0.0059] | 1529 | 0.8736 | no |
| statement_accuracy | none | -0.0052 [-0.0092, -0.0020] | 1529 | 0.0020 | no |
| statement_accuracy | mean_row | -0.0007 [-0.0039, +0.0026] | 1529 | 0.8236 | no |
| statement_margin | none | +0.0010 [-0.0015, +0.0036] | 1529 | 0.5737 | no |
| statement_margin | mean_row | -0.0014 [-0.0039, +0.0011] | 1529 | 0.5737 | no |
| statement_loss | none | -0.0079 [-0.0102, -0.0058] | 1529 | 0.0020 | yes |
| statement_loss | mean_row | +0.0008 [-0.0013, +0.0029] | 1529 | 0.4318 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.475 → 0.475 (0.475) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.475 | +0.0000 [+0.0000, +0.0000] | 0.505 | 0.0000 / 0.0000 | 0.485 |
| seen | 100 | 0.500 → 0.500 (0.500) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.472 | 0.0000 / 0.0000 | 0.484 |
| heldout | 100 | 0.450 → 0.450 (0.450) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.537 | 0.0000 / 0.0000 | 0.483 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
