# E9 dimension 3 — zero-shot by ontology editing — C5 seed 3 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s3`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2773 | 0.2811 | 0.2617 | 0.5433 | 0.4048 | 0.2492 | -1.2016 | 5.3285 |
| none | 0.2037 | 0.2010 | 0.2150 | 0.5183 | 0.3277 | 0.2171 | -1.3059 | 5.5626 |
| mean_row | 0.2099 | 0.2046 | 0.2317 | 0.5100 | 0.4042 | 0.2086 | -1.4366 | 5.5211 |
| random_frame | 0.2184 | 0.2128 | 0.2417 | 0.5000 | 0.3905 | 0.2112 | -1.4089 | 5.5072 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0736 [+0.0569, +0.0906] | 1529 | 0.0030 | yes |
| property | mean_row | +0.0674 [+0.0530, +0.0814] | 1529 | 0.0030 | yes |
| property | random_frame | +0.0589 [+0.0432, +0.0749] | 1529 | 0.0030 | yes |
| property_new | none | +0.0801 [+0.0614, +0.0989] | 1229 | 0.0030 | yes |
| property_new | mean_row | +0.0765 [+0.0602, +0.0924] | 1229 | 0.0030 | yes |
| property_new | random_frame | +0.0683 [+0.0509, +0.0859] | 1229 | 0.0030 | yes |
| property_category | none | +0.0467 [+0.0050, +0.0883] | 300 | 0.0810 | no |
| property_category | mean_row | +0.0300 [-0.0050, +0.0650] | 300 | 0.2139 | no |
| property_category | random_frame | +0.0200 [-0.0200, +0.0600] | 300 | 0.3628 | no |
| entailment | none | +0.0250 [-0.0083, +0.0583] | 600 | 0.1579 | no |
| entailment | mean_row | +0.0333 [+0.0050, +0.0617] | 600 | 0.0440 | yes |
| entailment | random_frame | +0.0433 [+0.0133, +0.0733] | 600 | 0.0150 | yes |
| paraphrase | none | +0.0772 [+0.0484, +0.1040] | 1529 | 0.0030 | yes |
| paraphrase | mean_row | +0.0007 [-0.0255, +0.0255] | 1529 | 0.9835 | no |
| paraphrase | random_frame | +0.0144 [-0.0131, +0.0412] | 1529 | 0.6537 | no |
| statement_accuracy | none | +0.0320 [+0.0164, +0.0497] | 1529 | 0.0030 | yes |
| statement_accuracy | mean_row | +0.0405 [+0.0255, +0.0562] | 1529 | 0.0030 | yes |
| statement_accuracy | random_frame | +0.0379 [+0.0209, +0.0543] | 1529 | 0.0030 | yes |
| statement_margin | none | +0.1042 [+0.0573, +0.1533] | 1529 | 0.0030 | yes |
| statement_margin | mean_row | +0.2350 [+0.1954, +0.2780] | 1529 | 0.0030 | yes |
| statement_margin | random_frame | +0.2072 [+0.1628, +0.2560] | 1529 | 0.0030 | yes |
| statement_loss | none | -0.2340 [-0.2868, -0.1874] | 1529 | 0.0030 | yes |
| statement_loss | mean_row | -0.1926 [-0.2293, -0.1558] | 1529 | 0.0030 | yes |
| statement_loss | random_frame | -0.1787 [-0.2243, -0.1349] | 1529 | 0.0030 | yes |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.455 → 0.487 (0.475) | +0.4541 [+0.2357, +0.7145] | +0.1652 [+0.0339, +0.3193] | 0.480 | +0.3221 [+0.1657, +0.5037] | 0.535 | 0.0013 / 0.2439 | 0.500 |
| seen | 100 | 0.430 → 0.490 (0.465) | +0.8447 [+0.4253, +1.3268] | +0.2848 [+0.0588, +0.5401] | 0.500 | +0.5173 [+0.2463, +0.8349] | 0.505 | 0.0002 / 0.0307 | 0.498 |
| heldout | 100 | 0.480 → 0.485 (0.485) | +0.0636 [-0.0606, +0.1813] | +0.0456 [-0.0639, +0.1591] | 0.460 | +0.1270 [-0.0002, +0.2530] | 0.565 | 0.0023 / 0.2439 | 0.500 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
