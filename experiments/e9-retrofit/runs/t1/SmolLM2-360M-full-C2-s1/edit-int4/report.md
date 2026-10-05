# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1846 | 0.1795 | 0.1900 | 0.5033 | 0.3644 | 0.1765 | -1.7704 | 6.3334 |
| none | 0.1830 | 0.1779 | 0.1883 | 0.5050 | 0.3627 | 0.1748 | -1.7699 | 6.3336 |
| mean_row | 0.1871 | 0.1891 | 0.1850 | 0.5083 | 0.3562 | 0.1765 | -1.7707 | 6.3325 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0016 [-0.0057, +0.0090] | 612 | 1.0000 | no |
| property | mean_row | -0.0025 [-0.0098, +0.0041] | 612 | 1.0000 | no |
| property_new | none | +0.0016 [-0.0064, +0.0096] | 312 | 0.8526 | no |
| property_new | mean_row | -0.0096 [-0.0192, -0.0016] | 312 | 0.0840 | no |
| property_category | none | +0.0017 [-0.0100, +0.0133] | 300 | 0.9015 | no |
| property_category | mean_row | +0.0050 [-0.0067, +0.0167] | 300 | 0.9015 | no |
| entailment | none | -0.0017 [-0.0117, +0.0083] | 600 | 0.8596 | no |
| entailment | mean_row | -0.0050 [-0.0150, +0.0050] | 600 | 0.8596 | no |
| paraphrase | none | +0.0016 [-0.0131, +0.0163] | 612 | 0.9205 | no |
| paraphrase | mean_row | +0.0082 [-0.0065, +0.0229] | 612 | 0.7136 | no |
| statement_accuracy | none | +0.0016 [+0.0000, +0.0049] | 612 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0000 [-0.0049, +0.0049] | 612 | 1.0000 | no |
| statement_margin | none | -0.0005 [-0.0027, +0.0019] | 612 | 1.0000 | no |
| statement_margin | mean_row | +0.0004 [-0.0018, +0.0025] | 612 | 1.0000 | no |
| statement_loss | none | -0.0002 [-0.0022, +0.0017] | 612 | 0.8436 | no |
| statement_loss | mean_row | +0.0008 [-0.0009, +0.0027] | 612 | 0.7076 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.245 → 0.245 (0.245) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.240 | +0.0000 [+0.0000, +0.0000] | 0.753 | 0.0000 / 0.0000 | 0.313 |
| seen | 100 | 0.285 → 0.285 (0.285) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.270 | +0.0000 [+0.0000, +0.0000] | 0.736 | 0.0000 / 0.0000 | 0.350 |
| heldout | 100 | 0.205 → 0.205 (0.205) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.210 | +0.0000 [+0.0000, +0.0000] | 0.769 | 0.0000 / 0.0000 | 0.274 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
