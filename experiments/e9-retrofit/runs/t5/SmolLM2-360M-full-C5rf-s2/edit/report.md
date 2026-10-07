# E9 dimension 3 — zero-shot by ontology editing — C5rf seed 2 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5rf-s2`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2629 | 0.2653 | 0.2533 | 0.5600 | 0.4264 | 0.2623 | -0.9793 | 4.5089 |
| none | 0.1913 | 0.1937 | 0.1817 | 0.5050 | 0.3479 | 0.2256 | -1.0608 | 4.7500 |
| mean_row | 0.2031 | 0.2026 | 0.2050 | 0.4867 | 0.3911 | 0.2171 | -1.1460 | 4.6328 |
| random_frame | 0.2063 | 0.2063 | 0.2067 | 0.5000 | 0.4075 | 0.2217 | -1.1440 | 4.6534 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0716 [+0.0572, +0.0867] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0598 [+0.0468, +0.0739] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0566 [+0.0422, +0.0716] | 1529 | 0.0030 | yes |
| property_new | none | +0.0716 [+0.0557, +0.0883] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0627 [+0.0476, +0.0785] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0590 [+0.0431, +0.0757] | 1229 | 0.0030 | yes |
| property_category | none | +0.0717 [+0.0350, +0.1100] | 300 | 0.0030 | yes |
| property_category | mean_row | +0.0483 [+0.0167, +0.0783] | 300 | 0.0040 | yes |
| property_category | random_frame | +0.0467 [+0.0117, +0.0817] | 300 | 0.0130 | yes |
| entailment | none | +0.0550 [+0.0217, +0.0883] | 600 | 0.0030 | yes |
| entailment | mean_row | +0.0733 [+0.0466, +0.1050] | 600 | 0.0030 | yes |
| entailment | random_frame | +0.0600 [+0.0300, +0.0900] | 600 | 0.0030 | yes |
| paraphrase | none | +0.0785 [+0.0510, +0.1047] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0353 [+0.0105, +0.0589] | 1529 | 0.0060 | yes |
| paraphrase | random_frame | +0.0190 [-0.0072, +0.0458] | 1529 | 0.1669 | no |
| statement_accuracy | none | +0.0366 [+0.0196, +0.0543] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0451 [+0.0301, +0.0615] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0405 [+0.0249, +0.0569] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.0815 [+0.0424, +0.1195] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.1666 [+0.1319, +0.2021] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.1647 [+0.1254, +0.2072] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2412 [-0.2845, -0.2028] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1239 [-0.1587, -0.0914] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1445 [-0.1850, -0.1055] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.448 → 0.468 (0.455) | +0.3987 [+0.2090, +0.6524] | +0.1300 [+0.0253, +0.2608] | 0.485 | +0.2899 [+0.1675, +0.4501] | 0.545 | 0.0006 / 0.1515 | 0.497 |
| seen | 100 | 0.415 → 0.470 (0.450) | +0.7214 [+0.3479, +1.1467] | +0.2117 [+0.0233, +0.4434] | 0.470 | +0.4524 [+0.2160, +0.7173] | 0.550 | 0.0009 / 0.1515 | 0.494 |
| heldout | 100 | 0.480 → 0.465 (0.460) | +0.0759 [-0.0196, +0.1774] | +0.0484 [-0.0316, +0.1288] | 0.500 | +0.1273 [+0.0357, +0.2213] | 0.540 | 0.0002 / 0.0286 | 0.500 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
