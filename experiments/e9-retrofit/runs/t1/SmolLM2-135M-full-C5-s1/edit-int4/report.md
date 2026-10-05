# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1773 | 0.1827 | 0.1717 | 0.4950 | 0.3660 | 0.1928 | -1.5273 | 6.1234 |
| none | 0.1773 | 0.1795 | 0.1750 | 0.4933 | 0.3578 | 0.1912 | -1.5284 | 6.1300 |
| mean_row | 0.1724 | 0.1747 | 0.1700 | 0.4933 | 0.3676 | 0.1912 | -1.5303 | 6.1300 |
| random_frame | 0.1748 | 0.1811 | 0.1683 | 0.4833 | 0.3693 | 0.1895 | -1.5262 | 6.1271 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0000 [-0.0082, +0.0090] | 612 | 1.0000 | no |
| property | mean_row | +0.0049 [-0.0033, +0.0131] | 612 | 0.9805 | no |
| property | random_frame | +0.0025 [-0.0074, +0.0123] | 612 | 1.0000 | no |
| property_new | none | +0.0032 [-0.0096, +0.0160] | 312 | 1.0000 | no |
| property_new | mean_row | +0.0080 [-0.0048, +0.0209] | 312 | 0.8126 | no |
| property_new | random_frame | +0.0016 [-0.0112, +0.0160] | 312 | 1.0000 | no |
| property_category | none | -0.0033 [-0.0150, +0.0083] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0017 [-0.0083, +0.0133] | 300 | 1.0000 | no |
| property_category | random_frame | +0.0033 [-0.0100, +0.0183] | 300 | 1.0000 | no |
| entailment | none | +0.0017 [-0.0100, +0.0133] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0017 [-0.0083, +0.0117] | 600 | 1.0000 | no |
| entailment | random_frame | +0.0117 [+0.0000, +0.0250] | 600 | 0.2069 | no |
| paraphrase | none | +0.0082 [-0.0114, +0.0278] | 612 | 1.0000 | no |
| paraphrase | mean_row | -0.0016 [-0.0212, +0.0180] | 612 | 1.0000 | no |
| paraphrase | random_frame | -0.0033 [-0.0229, +0.0180] | 612 | 1.0000 | no |
| statement_accuracy | none | +0.0016 [-0.0033, +0.0065] | 612 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0016 [+0.0000, +0.0049] | 612 | 1.0000 | no |
| statement_accuracy | random_frame | +0.0033 [+0.0000, +0.0082] | 612 | 0.8636 | no |
| statement_margin | none | +0.0011 [-0.0028, +0.0050] | 612 | 1.0000 | no |
| statement_margin | mean_row | +0.0030 [-0.0009, +0.0068] | 612 | 0.4318 | no |
| statement_margin | random_frame | -0.0011 [-0.0057, +0.0032] | 612 | 1.0000 | no |
| statement_loss | none | -0.0066 [-0.0100, -0.0031] | 612 | 0.0030 | yes |
| statement_loss | mean_row | -0.0066 [-0.0099, -0.0032] | 612 | 0.0040 | yes |
| statement_loss | random_frame | -0.0037 [-0.0077, +0.0004] | 612 | 0.0900 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.265 → 0.265 (0.263) | -0.0414 [-0.0870, +0.0067] | -0.0338 [-0.0771, +0.0100] | 0.270 | -0.0633 [-0.1413, +0.0154] | 0.730 | 0.0012 / 0.2813 | 0.339 |
| seen | 100 | 0.270 → 0.270 (0.265) | -0.0549 [-0.1244, +0.0089] | -0.0464 [-0.1045, +0.0113] | 0.250 | -0.1004 [-0.2098, +0.0094] | 0.734 | 0.0019 / 0.2813 | 0.331 |
| heldout | 100 | 0.260 → 0.260 (0.260) | -0.0280 [-0.0989, +0.0397] | -0.0212 [-0.0957, +0.0471] | 0.290 | -0.0262 [-0.1438, +0.0937] | 0.726 | 0.0005 / 0.0639 | 0.346 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
