# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1699 | 0.1843 | 0.1550 | 0.4600 | 0.3366 | 0.1748 | -1.6966 | 5.8222 |
| none | 0.1757 | 0.1923 | 0.1583 | 0.4717 | 0.3399 | 0.1716 | -1.6951 | 5.8216 |
| mean_row | 0.1708 | 0.1859 | 0.1550 | 0.4700 | 0.3317 | 0.1716 | -1.6985 | 5.8243 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0057 [-0.0131, +0.0016] | 612 | 0.2959 | no |
| property | mean_row | -0.0008 [-0.0074, +0.0049] | 612 | 0.8576 | no |
| property_new | none | -0.0080 [-0.0176, +0.0016] | 312 | 0.2419 | no |
| property_new | mean_row | -0.0016 [-0.0081, +0.0048] | 312 | 0.8486 | no |
| property_category | none | -0.0033 [-0.0150, +0.0067] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0000 [-0.0100, +0.0100] | 300 | 1.0000 | no |
| entailment | none | -0.0117 [-0.0233, -0.0017] | 600 | 0.0740 | no |
| entailment | mean_row | -0.0100 [-0.0217, +0.0000] | 600 | 0.0950 | no |
| paraphrase | none | -0.0033 [-0.0229, +0.0147] | 612 | 1.0000 | no |
| paraphrase | mean_row | +0.0049 [-0.0131, +0.0229] | 612 | 1.0000 | no |
| statement_accuracy | none | +0.0033 [+0.0000, +0.0082] | 612 | 0.5517 | no |
| statement_accuracy | mean_row | +0.0033 [-0.0033, +0.0098] | 612 | 0.5517 | no |
| statement_margin | none | -0.0015 [-0.0041, +0.0011] | 612 | 0.3058 | no |
| statement_margin | mean_row | +0.0019 [-0.0008, +0.0044] | 612 | 0.3058 | no |
| statement_loss | none | +0.0007 [-0.0014, +0.0026] | 612 | 0.5477 | no |
| statement_loss | mean_row | -0.0021 [-0.0040, -0.0003] | 612 | 0.0620 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.240 → 0.240 (0.240) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.230 | +0.0000 [+0.0000, +0.0000] | 0.750 | 0.0000 / 0.0000 | 0.305 |
| seen | 100 | 0.240 → 0.240 (0.240) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.200 | +0.0000 [+0.0000, +0.0000] | 0.764 | 0.0000 / 0.0000 | 0.286 |
| heldout | 100 | 0.240 → 0.240 (0.240) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.260 | +0.0000 [+0.0000, +0.0000] | 0.736 | 0.0000 / 0.0000 | 0.320 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
