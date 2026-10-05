# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1846 | 0.1843 | 0.1850 | 0.5183 | 0.3399 | 0.1781 | -1.7535 | 6.3163 |
| none | 0.1863 | 0.1827 | 0.1900 | 0.5017 | 0.3497 | 0.1830 | -1.7546 | 6.3180 |
| mean_row | 0.1846 | 0.1763 | 0.1933 | 0.5083 | 0.3497 | 0.1830 | -1.7564 | 6.3184 |
| random_frame | 0.1879 | 0.1811 | 0.1950 | 0.5033 | 0.3611 | 0.1797 | -1.7557 | 6.3186 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0016 [-0.0090, +0.0049] | 612 | 1.0000 | no |
| property | mean_row | +0.0000 [-0.0074, +0.0074] | 612 | 1.0000 | no |
| property | random_frame | -0.0033 [-0.0123, +0.0049] | 612 | 1.0000 | no |
| property_new | none | +0.0016 [-0.0096, +0.0128] | 312 | 1.0000 | no |
| property_new | mean_row | +0.0080 [-0.0016, +0.0177] | 312 | 0.4858 | no |
| property_new | random_frame | +0.0032 [-0.0080, +0.0144] | 312 | 1.0000 | no |
| property_category | none | -0.0050 [-0.0167, +0.0050] | 300 | 0.3928 | no |
| property_category | mean_row | -0.0083 [-0.0200, +0.0017] | 300 | 0.3688 | no |
| property_category | random_frame | -0.0100 [-0.0233, +0.0033] | 300 | 0.3688 | no |
| entailment | none | +0.0167 [+0.0033, +0.0317] | 600 | 0.0720 | no |
| entailment | mean_row | +0.0100 [-0.0033, +0.0233] | 600 | 0.1859 | no |
| entailment | random_frame | +0.0150 [+0.0017, +0.0284] | 600 | 0.0920 | no |
| paraphrase | none | -0.0098 [-0.0294, +0.0082] | 612 | 0.6257 | no |
| paraphrase | mean_row | -0.0098 [-0.0278, +0.0082] | 612 | 0.6257 | no |
| paraphrase | random_frame | -0.0212 [-0.0425, -0.0016] | 612 | 0.1139 | no |
| statement_accuracy | none | -0.0049 [-0.0114, +0.0000] | 612 | 0.3178 | no |
| statement_accuracy | mean_row | -0.0049 [-0.0114, +0.0000] | 612 | 0.3178 | no |
| statement_accuracy | random_frame | -0.0016 [-0.0082, +0.0049] | 612 | 0.8386 | no |
| statement_margin | none | +0.0011 [-0.0028, +0.0051] | 612 | 0.8616 | no |
| statement_margin | mean_row | +0.0029 [-0.0007, +0.0068] | 612 | 0.3808 | no |
| statement_margin | random_frame | +0.0022 [-0.0035, +0.0076] | 612 | 0.8616 | no |
| statement_loss | none | -0.0017 [-0.0051, +0.0016] | 612 | 0.6417 | no |
| statement_loss | mean_row | -0.0021 [-0.0054, +0.0012] | 612 | 0.5637 | no |
| statement_loss | random_frame | -0.0023 [-0.0067, +0.0021] | 612 | 0.6417 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.245 → 0.240 (0.242) | -0.0309 [-0.0631, +0.0008] | -0.0039 [-0.0398, +0.0370] | 0.230 | -0.0660 [-0.1128, -0.0174] | 0.753 | 0.0014 / 0.2453 | 0.305 |
| seen | 100 | 0.280 → 0.275 (0.275) | -0.0319 [-0.0800, +0.0155] | -0.0012 [-0.0580, +0.0564] | 0.250 | -0.1125 [-0.1836, -0.0388] | 0.739 | 0.0025 / 0.2453 | 0.334 |
| heldout | 100 | 0.210 → 0.205 (0.210) | -0.0299 [-0.0753, +0.0144] | -0.0067 [-0.0562, +0.0420] | 0.210 | -0.0195 [-0.0830, +0.0446] | 0.766 | 0.0003 / 0.0517 | 0.274 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
