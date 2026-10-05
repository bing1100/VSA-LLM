# E9 dimension 3 — zero-shot by ontology editing — P0 seed 1 (HuggingFaceTB/SmolLM2-360M/frozen)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-360M-frozen-P0-s1`, channel `none`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1904 | 0.1843 | 0.1967 | 0.4933 | 0.3301 | 0.1765 | -1.6933 | 6.5106 |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.242 → 0.242 (0.242) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.250 | +0.0000 [+0.0000, +0.0000] | 0.745 | 0.0000 / 0.0000 | 0.317 |
| seen | 100 | 0.280 → 0.280 (0.280) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.260 | +0.0000 [+0.0000, +0.0000] | 0.724 | 0.0000 / 0.0000 | 0.341 |
| heldout | 100 | 0.205 → 0.205 (0.205) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.240 | +0.0000 [+0.0000, +0.0000] | 0.766 | 0.0000 / 0.0000 | 0.290 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
