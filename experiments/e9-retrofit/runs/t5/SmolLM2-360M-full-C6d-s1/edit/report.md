# E9 dimension 3 — zero-shot by ontology editing — C6d seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C6d-s1`, channel `source`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1933 | 0.1973 | 0.1767 | 0.5233 | 0.3362 | 0.2191 | -1.0557 | 4.7073 |
| none | 0.1936 | 0.1973 | 0.1783 | 0.5183 | 0.3362 | 0.2165 | -1.0559 | 4.7078 |
| mean_row | 0.1910 | 0.1961 | 0.1700 | 0.5067 | 0.3440 | 0.2165 | -1.0555 | 4.7031 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0003 [-0.0046, +0.0039] | 1529 | 0.9395 | no |
| property | mean_row | +0.0023 [-0.0029, +0.0075] | 1529 | 0.8916 | no |
| property_new | none | +0.0000 [-0.0045, +0.0045] | 1229 | 1.0000 | no |
| property_new | mean_row | +0.0012 [-0.0045, +0.0069] | 1229 | 1.0000 | no |
| property_category | none | -0.0017 [-0.0150, +0.0117] | 300 | 0.8976 | no |
| property_category | mean_row | +0.0067 [-0.0067, +0.0200] | 300 | 0.7616 | no |
| entailment | none | +0.0050 [-0.0067, +0.0167] | 600 | 0.4998 | no |
| entailment | mean_row | +0.0167 [+0.0050, +0.0284] | 600 | 0.0100 | yes |
| paraphrase | none | +0.0000 [-0.0085, +0.0092] | 1529 | 1.0000 | no |
| paraphrase | mean_row | -0.0078 [-0.0183, +0.0039] | 1529 | 0.3818 | no |
| statement_accuracy | none | +0.0026 [-0.0020, +0.0072] | 1529 | 0.6437 | no |
| statement_accuracy | mean_row | +0.0026 [-0.0026, +0.0078] | 1529 | 0.6437 | no |
| statement_margin | none | +0.0002 [-0.0023, +0.0028] | 1529 | 1.0000 | no |
| statement_margin | mean_row | -0.0002 [-0.0048, +0.0046] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0005 [-0.0027, +0.0016] | 1529 | 0.6827 | no |
| statement_loss | mean_row | +0.0042 [-0.0018, +0.0103] | 1529 | 0.3478 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.443 → 0.443 (0.443) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.445 | +0.0000 [+0.0000, +0.0000] | 0.541 | 0.0000 / 0.0000 | 0.472 |
| seen | 100 | 0.415 → 0.415 (0.415) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.430 | +0.0000 [+0.0000, +0.0000] | 0.532 | 0.0000 / 0.0000 | 0.454 |
| heldout | 100 | 0.470 → 0.470 (0.470) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.550 | 0.0000 / 0.0000 | 0.490 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
