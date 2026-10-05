# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1653 | 0.1594 | 0.1733 | 0.4750 | 0.3685 | 0.2954 | -0.9909 | 3.1336 |
| none | 0.1702 | 0.1594 | 0.1850 | 0.4683 | 0.3586 | 0.2911 | -0.9921 | 3.1340 |
| mean_row | 0.1646 | 0.1582 | 0.1733 | 0.4750 | 0.3671 | 0.2954 | -0.9924 | 3.1347 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0049 [-0.0113, +0.0014] | 711 | 0.3038 | no |
| property | mean_row | +0.0007 [-0.0049, +0.0063] | 711 | 0.9145 | no |
| property_new | none | +0.0000 [-0.0085, +0.0085] | 411 | 1.0000 | no |
| property_new | mean_row | +0.0012 [-0.0061, +0.0085] | 411 | 1.0000 | no |
| property_category | none | -0.0117 [-0.0217, -0.0033] | 300 | 0.0400 | no |
| property_category | mean_row | +0.0000 [-0.0083, +0.0083] | 300 | 1.0000 | no |
| entailment | none | +0.0067 [-0.0067, +0.0200] | 600 | 0.8056 | no |
| entailment | mean_row | +0.0000 [-0.0133, +0.0133] | 600 | 1.0000 | no |
| paraphrase | none | +0.0098 [-0.0042, +0.0267] | 711 | 0.4838 | no |
| paraphrase | mean_row | +0.0014 [-0.0113, +0.0155] | 711 | 0.8856 | no |
| statement_accuracy | none | +0.0042 [-0.0028, +0.0113] | 711 | 0.6737 | no |
| statement_accuracy | mean_row | +0.0000 [-0.0056, +0.0056] | 711 | 1.0000 | no |
| statement_margin | none | +0.0013 [-0.0004, +0.0030] | 711 | 0.1739 | no |
| statement_margin | mean_row | +0.0015 [-0.0002, +0.0033] | 711 | 0.1739 | no |
| statement_loss | none | -0.0004 [-0.0018, +0.0011] | 711 | 0.6347 | no |
| statement_loss | mean_row | -0.0011 [-0.0026, +0.0004] | 711 | 0.2819 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.325 → 0.325 (0.325) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.320 | +0.0000 [+0.0000, +0.0000] | 0.731 | 0.0000 / 0.0000 | 0.396 |
| seen | 100 | 0.315 → 0.315 (0.315) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.340 | +0.0000 [+0.0000, +0.0000] | 0.733 | 0.0000 / 0.0000 | 0.401 |
| heldout | 100 | 0.335 → 0.335 (0.335) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.300 | +0.0000 [+0.0000, +0.0000] | 0.730 | 0.0000 / 0.0000 | 0.390 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
