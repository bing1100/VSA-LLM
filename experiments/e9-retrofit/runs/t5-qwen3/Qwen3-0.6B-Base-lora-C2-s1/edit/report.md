# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (Qwen/Qwen3-0.6B-Base/lora)

Run `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C2-s1`, channel `free`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 0). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1946 | 0.2026 | 0.1617 | 0.4750 | 0.3748 | 0.2341 | -1.1036 | 4.6241 |
| none | 0.1956 | 0.2038 | 0.1617 | 0.4783 | 0.3748 | 0.2322 | -1.1019 | 4.6253 |
| mean_row | 0.1978 | 0.2055 | 0.1667 | 0.4783 | 0.3721 | 0.2322 | -1.1014 | 4.6230 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0010 [-0.0043, +0.0023] | 1529 | 0.6747 | no |
| property | mean_row | -0.0033 [-0.0069, +0.0000] | 1529 | 0.1479 | no |
| property_new | none | -0.0012 [-0.0053, +0.0024] | 1229 | 0.5817 | no |
| property_new | mean_row | -0.0028 [-0.0069, +0.0008] | 1229 | 0.3578 | no |
| property_category | none | +0.0000 [-0.0067, +0.0083] | 300 | 1.0000 | no |
| property_category | mean_row | -0.0050 [-0.0133, +0.0017] | 300 | 0.4738 | no |
| entailment | none | -0.0033 [-0.0083, +0.0000] | 600 | 0.5157 | no |
| entailment | mean_row | -0.0033 [-0.0100, +0.0033] | 600 | 0.5157 | no |
| paraphrase | none | +0.0000 [-0.0092, +0.0092] | 1529 | 1.0000 | no |
| paraphrase | mean_row | +0.0026 [-0.0065, +0.0118] | 1529 | 1.0000 | no |
| statement_accuracy | none | +0.0020 [-0.0020, +0.0065] | 1529 | 0.5117 | no |
| statement_accuracy | mean_row | +0.0020 [-0.0007, +0.0052] | 1529 | 0.5117 | no |
| statement_margin | none | -0.0017 [-0.0043, +0.0009] | 1529 | 0.1819 | no |
| statement_margin | mean_row | -0.0022 [-0.0046, +0.0003] | 1529 | 0.1659 | no |
| statement_loss | none | -0.0012 [-0.0032, +0.0008] | 1529 | 0.5077 | no |
| statement_loss | mean_row | +0.0011 [-0.0008, +0.0030] | 1529 | 0.5077 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 363 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.470 → 0.470 (0.470) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.470 | +0.0000 [+0.0000, +0.0000] | 0.566 | 0.0000 / 0.0000 | 0.498 |
| seen | 100 | 0.485 → 0.485 (0.485) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.480 | +0.0000 [+0.0000, +0.0000] | 0.560 | 0.0000 / 0.0000 | 0.506 |
| heldout | 100 | 0.455 → 0.455 (0.455) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.460 | +0.0000 [+0.0000, +0.0000] | 0.573 | 0.0000 / 0.0000 | 0.490 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
