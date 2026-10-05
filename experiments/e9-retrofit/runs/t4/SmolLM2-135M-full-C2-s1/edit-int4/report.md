# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t4/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1934 | 0.1922 | 0.1950 | 0.4867 | 0.3347 | 0.3108 | -1.0803 | 3.9662 |
| none | 0.1920 | 0.1946 | 0.1883 | 0.4850 | 0.3277 | 0.3066 | -1.0798 | 3.9652 |
| mean_row | 0.1899 | 0.1922 | 0.1867 | 0.4817 | 0.3361 | 0.3094 | -1.0795 | 3.9658 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0014 [-0.0063, +0.0091] | 711 | 0.8716 | no |
| property | mean_row | +0.0035 [-0.0049, +0.0113] | 711 | 0.8716 | no |
| property_new | none | -0.0024 [-0.0122, +0.0073] | 411 | 1.0000 | no |
| property_new | mean_row | +0.0000 [-0.0085, +0.0085] | 411 | 1.0000 | no |
| property_category | none | +0.0067 [-0.0067, +0.0200] | 300 | 0.5937 | no |
| property_category | mean_row | +0.0083 [-0.0067, +0.0233] | 300 | 0.5937 | no |
| entailment | none | +0.0017 [-0.0117, +0.0167] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0050 [-0.0083, +0.0183] | 600 | 1.0000 | no |
| paraphrase | none | +0.0070 [-0.0113, +0.0253] | 711 | 1.0000 | no |
| paraphrase | mean_row | -0.0014 [-0.0211, +0.0169] | 711 | 1.0000 | no |
| statement_accuracy | none | +0.0042 [-0.0014, +0.0113] | 711 | 0.4818 | no |
| statement_accuracy | mean_row | +0.0014 [-0.0056, +0.0085] | 711 | 0.8326 | no |
| statement_margin | none | -0.0004 [-0.0028, +0.0020] | 711 | 0.9575 | no |
| statement_margin | mean_row | -0.0008 [-0.0030, +0.0015] | 711 | 0.9575 | no |
| statement_loss | none | +0.0010 [-0.0009, +0.0029] | 711 | 0.5997 | no |
| statement_loss | mean_row | +0.0004 [-0.0014, +0.0022] | 711 | 0.6807 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 397 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.335 → 0.335 (0.335) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.335 | +0.0000 [+0.0000, +0.0000] | 0.691 | 0.0000 / 0.0000 | 0.404 |
| seen | 100 | 0.370 → 0.370 (0.370) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.370 | +0.0000 [+0.0000, +0.0000] | 0.645 | 0.0000 / 0.0000 | 0.431 |
| heldout | 100 | 0.300 → 0.300 (0.300) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.300 | +0.0000 [+0.0000, +0.0000] | 0.738 | 0.0000 / 0.0000 | 0.374 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
