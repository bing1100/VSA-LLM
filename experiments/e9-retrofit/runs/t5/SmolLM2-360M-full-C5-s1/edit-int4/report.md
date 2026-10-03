# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2688 | 0.2665 | 0.2783 | 0.5500 | 0.3833 | 0.2525 | -1.2527 | 5.5262 |
| none | 0.2063 | 0.2042 | 0.2150 | 0.5067 | 0.3152 | 0.2171 | -1.3186 | 5.7118 |
| mean_row | 0.2090 | 0.2026 | 0.2350 | 0.5017 | 0.3754 | 0.2126 | -1.4228 | 5.6680 |
| random_frame | 0.2220 | 0.2120 | 0.2633 | 0.5083 | 0.3819 | 0.2165 | -1.3940 | 5.6377 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0625 [+0.0471, +0.0788] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0598 [+0.0461, +0.0739] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0468 [+0.0314, +0.0631] | 1529 | 0.0030 | yes |
| property_new | none | +0.0622 [+0.0452, +0.0797] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0639 [+0.0492, +0.0793] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0545 [+0.0382, +0.0712] | 1229 | 0.0030 | yes |
| property_category | none | +0.0633 [+0.0250, +0.1033] | 300 | 0.0060 | yes |
| property_category | mean_row | +0.0433 [+0.0083, +0.0800] | 300 | 0.0400 | yes |
| property_category | random_frame | +0.0150 [-0.0250, +0.0534] | 300 | 0.4868 | no |
| entailment | none | +0.0433 [+0.0100, +0.0784] | 600 | 0.0240 | yes |
| entailment | mean_row | +0.0483 [+0.0167, +0.0800] | 600 | 0.0120 | yes |
| entailment | random_frame | +0.0417 [+0.0100, +0.0733] | 600 | 0.0240 | yes |
| paraphrase | none | +0.0680 [+0.0405, +0.0948] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0078 [-0.0157, +0.0320] | 1529 | 1.0000 | no |
| paraphrase | random_frame | +0.0013 [-0.0255, +0.0275] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0353 [+0.0203, +0.0510] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0399 [+0.0255, +0.0543] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0360 [+0.0203, +0.0510] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0659 [+0.0204, +0.1099] | 1529 | 0.0060 | yes |
| statement_margin | mean_row | +0.1701 [+0.1311, +0.2090] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1413 [+0.0993, +0.1833] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.1856 [-0.2327, -0.1400] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1418 [-0.1799, -0.1031] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1115 [-0.1499, -0.0732] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.458 → 0.472 (0.477) | +0.3940 [+0.1858, +0.6519] | +0.1266 [-0.0232, +0.2848] | 0.485 | +0.1941 [+0.0285, +0.3917] | 0.525 | 0.0012 / 0.1676 | 0.493 |
| seen | 100 | 0.440 → 0.480 (0.480) | +0.8029 [+0.4029, +1.2637] | +0.2964 [+0.0380, +0.5568] | 0.490 | +0.4329 [+0.1437, +0.7654] | 0.502 | 0.0017 / 0.1676 | 0.491 |
| heldout | 100 | 0.475 → 0.465 (0.475) | -0.0148 [-0.1712, +0.1239] | -0.0433 [-0.2017, +0.0950] | 0.480 | -0.0447 [-0.1770, +0.0933] | 0.547 | 0.0007 / 0.0698 | 0.495 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
