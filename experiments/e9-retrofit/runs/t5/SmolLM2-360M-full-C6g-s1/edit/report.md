# E9 dimension 3 — zero-shot by ontology editing — C6g seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C6g-s1`, channel `source`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1959 | 0.2006 | 0.1767 | 0.5167 | 0.3270 | 0.2256 | -1.0359 | 4.6786 |
| none | 0.1933 | 0.1977 | 0.1750 | 0.5233 | 0.3224 | 0.2250 | -1.0353 | 4.6787 |
| mean_row | 0.1929 | 0.1969 | 0.1767 | 0.5167 | 0.3257 | 0.2250 | -1.0363 | 4.6791 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0026 [-0.0010, +0.0062] | 1529 | 0.3118 | no |
| property | mean_row | +0.0029 [-0.0007, +0.0065] | 1529 | 0.3118 | no |
| property_new | none | +0.0028 [-0.0008, +0.0065] | 1229 | 0.1389 | no |
| property_new | mean_row | +0.0037 [+0.0004, +0.0073] | 1229 | 0.1000 | no |
| property_category | none | +0.0017 [-0.0100, +0.0133] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0000 [-0.0117, +0.0117] | 300 | 1.0000 | no |
| entailment | none | -0.0067 [-0.0183, +0.0033] | 600 | 0.6417 | no |
| entailment | mean_row | +0.0000 [-0.0117, +0.0117] | 600 | 1.0000 | no |
| paraphrase | none | +0.0046 [-0.0026, +0.0118] | 1529 | 0.5317 | no |
| paraphrase | mean_row | +0.0013 [-0.0065, +0.0098] | 1529 | 0.8116 | no |
| statement_accuracy | none | +0.0007 [-0.0039, +0.0059] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0007 [-0.0039, +0.0059] | 1529 | 1.0000 | no |
| statement_margin | none | -0.0007 [-0.0030, +0.0017] | 1529 | 1.0000 | no |
| statement_margin | mean_row | +0.0004 [-0.0025, +0.0032] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0001 [-0.0020, +0.0016] | 1529 | 1.0000 | no |
| statement_loss | mean_row | -0.0005 [-0.0029, +0.0021] | 1529 | 1.0000 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.432 → 0.432 (0.432) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.450 | +0.0000 [+0.0000, +0.0000] | 0.537 | 0.0000 / 0.0000 | 0.469 |
| seen | 100 | 0.420 → 0.420 (0.420) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.440 | +0.0000 [+0.0000, +0.0000] | 0.530 | 0.0000 / 0.0000 | 0.459 |
| heldout | 100 | 0.445 → 0.445 (0.445) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.545 | 0.0000 / 0.0000 | 0.480 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
