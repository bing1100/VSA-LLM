# E9 dimension 3 — zero-shot by ontology editing — C2 seed 2 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-135M-full-C2-s2`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1910 | 0.2059 | 0.1300 | 0.4800 | 0.3957 | 0.2139 | -1.2391 | 5.6114 |
| none | 0.1939 | 0.2091 | 0.1317 | 0.4783 | 0.4042 | 0.2126 | -1.2397 | 5.6154 |
| mean_row | 0.1916 | 0.2059 | 0.1333 | 0.4817 | 0.4061 | 0.2139 | -1.2393 | 5.6093 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0029 [-0.0069, +0.0010] | 1529 | 0.3398 | no |
| property | mean_row | -0.0007 [-0.0049, +0.0036] | 1529 | 0.8356 | no |
| property_new | none | -0.0033 [-0.0077, +0.0012] | 1229 | 0.3178 | no |
| property_new | mean_row | +0.0000 [-0.0049, +0.0049] | 1229 | 1.0000 | no |
| property_category | none | -0.0017 [-0.0117, +0.0083] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0033 [-0.0133, +0.0050] | 300 | 1.0000 | no |
| entailment | none | +0.0017 [-0.0100, +0.0133] | 600 | 1.0000 | no |
| entailment | mean_row | -0.0017 [-0.0117, +0.0100] | 600 | 1.0000 | no |
| paraphrase | none | -0.0085 [-0.0190, +0.0020] | 1529 | 0.1559 | no |
| paraphrase | mean_row | -0.0105 [-0.0216, +0.0007] | 1529 | 0.1559 | no |
| statement_accuracy | none | +0.0013 [-0.0026, +0.0059] | 1529 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0000 [-0.0026, +0.0026] | 1529 | 1.0000 | no |
| statement_margin | none | +0.0006 [-0.0020, +0.0031] | 1529 | 1.0000 | no |
| statement_margin | mean_row | +0.0002 [-0.0022, +0.0028] | 1529 | 1.0000 | no |
| statement_loss | none | -0.0040 [-0.0063, -0.0016] | 1529 | 0.0020 | yes |
| statement_loss | mean_row | +0.0021 [+0.0000, +0.0041] | 1529 | 0.0500 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.468 → 0.468 (0.468) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.507 | 0.0000 / 0.0000 | 0.484 |
| seen | 100 | 0.485 → 0.485 (0.485) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.475 | 0.0000 / 0.0000 | 0.480 |
| heldout | 100 | 0.450 → 0.450 (0.450) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.540 | 0.0000 / 0.0000 | 0.487 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
