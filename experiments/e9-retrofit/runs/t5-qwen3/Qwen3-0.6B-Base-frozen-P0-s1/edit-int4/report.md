# E9 dimension 3 — zero-shot by ontology editing — P0 seed 1 (Qwen/Qwen3-0.6B-Base/frozen)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-frozen-P0-s1`, channel `none`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1933 | 0.2067 | 0.1383 | 0.4750 | 0.4277 | 0.2086 | -1.0334 | 9.1278 |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.495 → 0.495 (0.495) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.505 | +0.0000 [+0.0000, +0.0000] | 0.465 | 0.0000 / 0.0000 | 0.488 |
| seen | 100 | 0.480 → 0.480 (0.480) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.490 | +0.0000 [+0.0000, +0.0000] | 0.435 | 0.0000 / 0.0000 | 0.467 |
| heldout | 100 | 0.510 → 0.510 (0.510) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.520 | +0.0000 [+0.0000, +0.0000] | 0.495 | 0.0000 / 0.0000 | 0.508 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
