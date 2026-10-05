# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2937 | 0.1608 | 0.3825 | 0.5892 | 0.5195 | 0.2423 | -1.7842 | 9.1761 |
| none | 0.2897 | 0.1608 | 0.3758 | 0.5925 | 0.5115 | 0.2438 | -1.7847 | 9.1768 |
| mean_row | 0.2892 | 0.1596 | 0.3758 | 0.5933 | 0.5225 | 0.2408 | -1.7846 | 9.1773 |
| random_frame | 0.2902 | 0.1608 | 0.3767 | 0.5925 | 0.5145 | 0.2403 | -1.7828 | 9.1748 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0040 [-0.0010, +0.0095] | 1001 | 0.3018 | no |
| property | mean_row | +0.0045 [+0.0000, +0.0090] | 1001 | 0.2009 | no |
| property | random_frame | +0.0035 [-0.0025, +0.0095] | 1001 | 0.3018 | no |
| property_new | none | +0.0000 [-0.0087, +0.0087] | 401 | 1.0000 | no |
| property_new | mean_row | +0.0012 [-0.0062, +0.0087] | 401 | 1.0000 | no |
| property_new | random_frame | +0.0000 [-0.0087, +0.0087] | 401 | 1.0000 | no |
| property_category | none | +0.0067 [+0.0008, +0.0125] | 600 | 0.0810 | no |
| property_category | mean_row | +0.0067 [+0.0008, +0.0125] | 600 | 0.0810 | no |
| property_category | random_frame | +0.0058 [-0.0017, +0.0142] | 600 | 0.1729 | no |
| entailment | none | -0.0033 [-0.0108, +0.0033] | 600 | 0.8216 | no |
| entailment | mean_row | -0.0042 [-0.0108, +0.0017] | 600 | 0.6897 | no |
| entailment | random_frame | -0.0033 [-0.0117, +0.0050] | 600 | 0.8216 | no |
| paraphrase | none | +0.0080 [-0.0020, +0.0190] | 1001 | 0.4828 | no |
| paraphrase | mean_row | -0.0030 [-0.0130, +0.0060] | 1001 | 0.8496 | no |
| paraphrase | random_frame | +0.0050 [-0.0070, +0.0170] | 1001 | 0.8496 | no |
| statement_accuracy | none | -0.0015 [-0.0055, +0.0020] | 1001 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0015 [-0.0025, +0.0055] | 1001 | 1.0000 | no |
| statement_accuracy | random_frame | +0.0020 [-0.0015, +0.0060] | 1001 | 0.9715 | no |
| statement_margin | none | +0.0006 [-0.0024, +0.0034] | 1001 | 1.0000 | no |
| statement_margin | mean_row | +0.0004 [-0.0027, +0.0034] | 1001 | 1.0000 | no |
| statement_margin | random_frame | -0.0013 [-0.0050, +0.0021] | 1001 | 1.0000 | no |
| statement_loss | none | -0.0006 [-0.0033, +0.0019] | 1001 | 1.0000 | no |
| statement_loss | mean_row | -0.0012 [-0.0038, +0.0014] | 1001 | 1.0000 | no |
| statement_loss | random_frame | +0.0013 [-0.0020, +0.0047] | 1001 | 1.0000 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.285 → 0.280 (0.285) | +0.0096 [-0.0195, +0.0390] | +0.0042 [-0.0239, +0.0322] | 0.300 | +0.0219 [-0.0121, +0.0717] | 0.688 | 0.0000 / 0.0000 | 0.359 |
| seen | 100 | 0.300 → 0.295 (0.305) | +0.0140 [-0.0253, +0.0563] | +0.0018 [-0.0349, +0.0403] | 0.315 | +0.0441 [-0.0157, +0.1461] | 0.690 | 0.0000 / 0.0000 | 0.374 |
| heldout | 100 | 0.270 → 0.265 (0.265) | +0.0052 [-0.0326, +0.0466] | +0.0067 [-0.0365, +0.0522] | 0.285 | -0.0004 [-0.0224, +0.0235] | 0.685 | 0.0000 / 0.0000 | 0.343 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
