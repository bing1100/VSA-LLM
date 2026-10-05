# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1944 | 0.1811 | 0.2083 | 0.5200 | 0.3186 | 0.1846 | -1.6918 | 5.6961 |
| none | 0.1977 | 0.1827 | 0.2133 | 0.5167 | 0.3235 | 0.1879 | -1.6937 | 5.6974 |
| mean_row | 0.2002 | 0.1859 | 0.2150 | 0.5200 | 0.3399 | 0.1863 | -1.6926 | 5.6966 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0033 [-0.0098, +0.0033] | 612 | 0.4298 | no |
| property | mean_row | -0.0057 [-0.0131, +0.0016] | 612 | 0.2999 | no |
| property_new | none | -0.0016 [-0.0096, +0.0064] | 312 | 0.8916 | no |
| property_new | mean_row | -0.0048 [-0.0160, +0.0048] | 312 | 0.8916 | no |
| property_category | none | -0.0050 [-0.0150, +0.0050] | 300 | 0.4198 | no |
| property_category | mean_row | -0.0067 [-0.0167, +0.0017] | 300 | 0.4198 | no |
| entailment | none | +0.0033 [-0.0050, +0.0133] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0000 [-0.0067, +0.0067] | 600 | 1.0000 | no |
| paraphrase | none | -0.0049 [-0.0163, +0.0065] | 612 | 0.4578 | no |
| paraphrase | mean_row | -0.0212 [-0.0359, -0.0082] | 612 | 0.0080 | no |
| statement_accuracy | none | -0.0033 [-0.0082, +0.0000] | 612 | 0.5337 | no |
| statement_accuracy | mean_row | -0.0016 [-0.0082, +0.0033] | 612 | 0.7396 | no |
| statement_margin | none | +0.0020 [-0.0001, +0.0040] | 612 | 0.1239 | no |
| statement_margin | mean_row | +0.0008 [-0.0015, +0.0029] | 612 | 0.5347 | no |
| statement_loss | none | -0.0013 [-0.0029, +0.0004] | 612 | 0.2919 | no |
| statement_loss | mean_row | -0.0004 [-0.0021, +0.0013] | 612 | 0.6247 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.217 → 0.217 (0.217) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.200 | +0.0000 [+0.0000, +0.0000] | 0.790 | 0.0000 / 0.0000 | 0.276 |
| seen | 100 | 0.240 → 0.240 (0.240) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.230 | +0.0000 [+0.0000, +0.0000] | 0.796 | 0.0000 / 0.0000 | 0.307 |
| heldout | 100 | 0.195 → 0.195 (0.195) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.170 | +0.0000 [+0.0000, +0.0000] | 0.784 | 0.0000 / 0.0000 | 0.244 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
