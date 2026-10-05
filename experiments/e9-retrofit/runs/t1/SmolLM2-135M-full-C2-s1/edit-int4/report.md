# E9 dimension 3 — zero-shot by ontology editing — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-135M-full-C2-s1`, channel `free`, weights: **int4-A**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.1667 | 0.1859 | 0.1467 | 0.4950 | 0.3644 | 0.1879 | -1.5368 | 6.1692 |
| none | 0.1724 | 0.1891 | 0.1550 | 0.4900 | 0.3480 | 0.1928 | -1.5350 | 6.1684 |
| mean_row | 0.1716 | 0.1875 | 0.1550 | 0.4950 | 0.3513 | 0.1961 | -1.5365 | 6.1687 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | -0.0057 [-0.0131, +0.0008] | 612 | 0.2479 | no |
| property | mean_row | -0.0049 [-0.0123, +0.0016] | 612 | 0.2479 | no |
| property_new | none | -0.0032 [-0.0128, +0.0048] | 312 | 1.0000 | no |
| property_new | mean_row | -0.0016 [-0.0112, +0.0080] | 312 | 1.0000 | no |
| property_category | none | -0.0083 [-0.0200, +0.0017] | 300 | 0.1939 | no |
| property_category | mean_row | -0.0083 [-0.0183, +0.0000] | 300 | 0.1939 | no |
| entailment | none | +0.0050 [-0.0067, +0.0167] | 600 | 1.0000 | no |
| entailment | mean_row | +0.0000 [-0.0117, +0.0100] | 600 | 1.0000 | no |
| paraphrase | none | +0.0163 [-0.0049, +0.0359] | 612 | 0.2639 | no |
| paraphrase | mean_row | +0.0131 [-0.0049, +0.0327] | 612 | 0.2639 | no |
| statement_accuracy | none | -0.0049 [-0.0114, +0.0000] | 612 | 0.1009 | no |
| statement_accuracy | mean_row | -0.0082 [-0.0163, -0.0016] | 612 | 0.0280 | no |
| statement_margin | none | -0.0018 [-0.0045, +0.0010] | 612 | 0.3978 | no |
| statement_margin | mean_row | -0.0003 [-0.0031, +0.0024] | 612 | 0.8606 | no |
| statement_loss | none | +0.0008 [-0.0012, +0.0028] | 612 | 0.8356 | no |
| statement_loss | mean_row | +0.0005 [-0.0015, +0.0026] | 612 | 0.8356 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts. **This model has no composed rows: the edit cannot change it (after = before).**

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.258 → 0.258 (0.258) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.260 | +0.0000 [+0.0000, +0.0000] | 0.724 | 0.0000 / 0.0000 | 0.329 |
| seen | 100 | 0.270 → 0.270 (0.270) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.240 | +0.0000 [+0.0000, +0.0000] | 0.726 | 0.0000 / 0.0000 | 0.324 |
| heldout | 100 | 0.245 → 0.245 (0.245) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0.280 | +0.0000 [+0.0000, +0.0000] | 0.721 | 0.0000 / 0.0000 | 0.332 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
