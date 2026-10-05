# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-135M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1724 | 0.1987 | 0.1450 | 0.4817 | 0.3333 | 0.1748 | -1.6839 | 5.8084 |
| none | 0.1740 | 0.1939 | 0.1533 | 0.4700 | 0.3431 | 0.1781 | -1.6837 | 5.8113 |
| mean_row | 0.1716 | 0.1891 | 0.1533 | 0.4783 | 0.3480 | 0.1765 | -1.6879 | 5.8121 |
| random_frame | 0.1724 | 0.1859 | 0.1583 | 0.4733 | 0.3399 | 0.1716 | -1.6887 | 5.8119 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0016 [-0.0082, +0.0049] | 612 | 1.0000 | no |
| property | mean_row | +0.0008 [-0.0065, +0.0090] | 612 | 1.0000 | no |
| property | random_frame | +0.0000 [-0.0090, +0.0098] | 612 | 1.0000 | no |
| property_new | none | +0.0048 [-0.0048, +0.0144] | 312 | 0.4678 | no |
| property_new | mean_row | +0.0096 [+0.0000, +0.0192] | 312 | 0.1679 | no |
| property_new | random_frame | +0.0128 [+0.0000, +0.0256] | 312 | 0.1679 | no |
| property_category | none | -0.0083 [-0.0167, +0.0000] | 300 | 0.1539 | no |
| property_category | mean_row | -0.0083 [-0.0200, +0.0033] | 300 | 0.1849 | no |
| property_category | random_frame | -0.0133 [-0.0250, -0.0033] | 300 | 0.0810 | no |
| entailment | none | +0.0117 [-0.0017, +0.0250] | 600 | 0.2549 | no |
| entailment | mean_row | +0.0033 [-0.0100, +0.0150] | 600 | 0.7176 | no |
| entailment | random_frame | +0.0083 [-0.0067, +0.0233] | 600 | 0.7176 | no |
| paraphrase | none | -0.0098 [-0.0294, +0.0098] | 612 | 0.7456 | no |
| paraphrase | mean_row | -0.0147 [-0.0327, +0.0033] | 612 | 0.4048 | no |
| paraphrase | random_frame | -0.0065 [-0.0294, +0.0163] | 612 | 0.7456 | no |
| statement_accuracy | none | -0.0033 [-0.0082, +0.0000] | 612 | 0.7826 | no |
| statement_accuracy | mean_row | -0.0016 [-0.0049, +0.0000] | 612 | 0.7826 | no |
| statement_accuracy | random_frame | +0.0033 [+0.0000, +0.0082] | 612 | 0.7826 | no |
| statement_margin | none | -0.0002 [-0.0041, +0.0036] | 612 | 0.8956 | no |
| statement_margin | mean_row | +0.0040 [+0.0003, +0.0077] | 612 | 0.1139 | no |
| statement_margin | random_frame | +0.0048 [+0.0001, +0.0091] | 612 | 0.1139 | no |
| statement_loss | none | -0.0029 [-0.0065, +0.0007] | 612 | 0.2359 | no |
| statement_loss | mean_row | -0.0037 [-0.0072, -0.0002] | 612 | 0.1169 | no |
| statement_loss | random_frame | -0.0035 [-0.0075, +0.0008] | 612 | 0.2359 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.240 → 0.240 (0.237) | -0.0158 [-0.0612, +0.0278] | -0.0186 [-0.0534, +0.0179] | 0.230 | -0.0000 [-0.0765, +0.0710] | 0.747 | 0.0018 / 0.4039 | 0.304 |
| seen | 100 | 0.235 → 0.235 (0.230) | -0.0128 [-0.0774, +0.0507] | -0.0169 [-0.0701, +0.0388] | 0.200 | +0.0355 [-0.0639, +0.1445] | 0.756 | 0.0033 / 0.4039 | 0.284 |
| heldout | 100 | 0.245 → 0.245 (0.245) | -0.0187 [-0.0871, +0.0431] | -0.0204 [-0.0695, +0.0275] | 0.260 | -0.0355 [-0.1438, +0.0672] | 0.739 | 0.0004 / 0.0505 | 0.323 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
