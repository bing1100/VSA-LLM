# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-360M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 13). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.3082 | 0.1696 | 0.4008 | 0.6358 | 0.5514 | 0.2388 | -1.7702 | 8.6886 |
| none | 0.3107 | 0.1721 | 0.4033 | 0.6325 | 0.5504 | 0.2383 | -1.7703 | 8.6878 |
| mean_row | 0.3062 | 0.1696 | 0.3975 | 0.6358 | 0.5504 | 0.2398 | -1.7682 | 8.6872 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0025 [-0.0070, +0.0020] | 1001 | 0.6797 | no |
| property | mean_row | +0.0020 [-0.0025, +0.0070] | 1001 | 0.6797 | no |
| property_new | none | -0.0025 [-0.0100, +0.0050] | 401 | 1.0000 | no |
| property_new | mean_row | +0.0000 [-0.0062, +0.0075] | 401 | 1.0000 | no |
| property_category | none | -0.0025 [-0.0083, +0.0033] | 600 | 0.6277 | no |
| property_category | mean_row | +0.0033 [-0.0025, +0.0100] | 600 | 0.6277 | no |
| entailment | none | +0.0033 [-0.0025, +0.0092] | 600 | 0.6657 | no |
| entailment | mean_row | +0.0000 [-0.0050, +0.0050] | 600 | 1.0000 | no |
| paraphrase | none | +0.0010 [-0.0060, +0.0080] | 1001 | 1.0000 | no |
| paraphrase | mean_row | +0.0010 [-0.0080, +0.0100] | 1001 | 1.0000 | no |
| statement_accuracy | none | +0.0005 [-0.0015, +0.0025] | 1001 | 1.0000 | no |
| statement_accuracy | mean_row | -0.0010 [-0.0035, +0.0020] | 1001 | 1.0000 | no |
| statement_margin | none | +0.0001 [-0.0019, +0.0020] | 1001 | 0.9625 | no |
| statement_margin | mean_row | -0.0020 [-0.0040, -0.0002] | 1001 | 0.0600 | no |
| statement_loss | none | +0.0008 [-0.0007, +0.0022] | 1001 | 0.3098 | no |
| statement_loss | mean_row | +0.0014 [-0.0002, +0.0031] | 1001 | 0.1719 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 400 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.185 → 0.185 (0.185) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.190 | +0.0000 [+0.0000, +0.0000] | 0.789 | 0.0000 / 0.0000 | 0.251 |
| seen | 100 | 0.205 → 0.205 (0.205) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.210 | +0.0000 [+0.0000, +0.0000] | 0.767 | 0.0000 / 0.0000 | 0.274 |
| heldout | 100 | 0.165 → 0.165 (0.165) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.170 | +0.0000 [+0.0000, +0.0000] | 0.810 | 0.0000 / 0.0000 | 0.228 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
