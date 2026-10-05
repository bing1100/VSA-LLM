# E9 dimension 3 — zero-shot by ontology editing — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Run `experiments/e9-retrofit/runs/t1/SmolLM2-360M-full-C5-s1`, channel `compose`, weights: **bf16 (as trained)**. No weight is updated: new words and edits change only the alias table and the composer's frame schedule.

## (a) New words

300 of 300 invented names link to their new entry ({'linked': 300}; spurious sub-word links inside the names: 3). Chance: property 0.200, entailment 0.500.

| source | property | property_new | property_category | entailment | paraphrase | statement_accuracy | statement_margin | statement_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| own | 0.2002 | 0.1763 | 0.2250 | 0.5200 | 0.3415 | 0.1846 | -1.6885 | 5.6910 |
| none | 0.2002 | 0.1763 | 0.2250 | 0.5250 | 0.3317 | 0.1830 | -1.6898 | 5.6943 |
| mean_row | 0.1961 | 0.1763 | 0.2167 | 0.5217 | 0.3350 | 0.1846 | -1.6898 | 5.6925 |
| random_frame | 0.2026 | 0.1795 | 0.2267 | 0.5233 | 0.3317 | 0.1830 | -1.6911 | 5.6945 |

`own` − baseline (paired bootstrap over items; Holm over baselines within each test):

| test | baseline | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---:|---:|---|
| property | none | +0.0000 [-0.0082, +0.0090] | 612 | 1.0000 | no |
| property | mean_row | +0.0041 [-0.0033, +0.0114] | 612 | 1.0000 | no |
| property | random_frame | -0.0025 [-0.0114, +0.0057] | 612 | 1.0000 | no |
| property_new | none | +0.0000 [-0.0096, +0.0081] | 312 | 1.0000 | no |
| property_new | mean_row | +0.0000 [-0.0096, +0.0096] | 312 | 1.0000 | no |
| property_new | random_frame | -0.0032 [-0.0144, +0.0080] | 312 | 1.0000 | no |
| property_category | none | +0.0000 [-0.0150, +0.0150] | 300 | 1.0000 | no |
| property_category | mean_row | +0.0083 [-0.0033, +0.0200] | 300 | 0.6477 | no |
| property_category | random_frame | -0.0017 [-0.0150, +0.0117] | 300 | 1.0000 | no |
| entailment | none | -0.0050 [-0.0167, +0.0083] | 600 | 1.0000 | no |
| entailment | mean_row | -0.0017 [-0.0150, +0.0117] | 600 | 1.0000 | no |
| entailment | random_frame | -0.0033 [-0.0167, +0.0117] | 600 | 1.0000 | no |
| paraphrase | none | +0.0098 [-0.0114, +0.0294] | 612 | 1.0000 | no |
| paraphrase | mean_row | +0.0065 [-0.0114, +0.0261] | 612 | 1.0000 | no |
| paraphrase | random_frame | +0.0098 [-0.0098, +0.0294] | 612 | 1.0000 | no |
| statement_accuracy | none | +0.0016 [-0.0033, +0.0065] | 612 | 1.0000 | no |
| statement_accuracy | mean_row | +0.0000 [+0.0000, +0.0000] | 612 | 1.0000 | no |
| statement_accuracy | random_frame | +0.0016 [-0.0033, +0.0082] | 612 | 1.0000 | no |
| statement_margin | none | +0.0012 [-0.0023, +0.0048] | 612 | 0.9235 | no |
| statement_margin | mean_row | +0.0013 [-0.0021, +0.0049] | 612 | 0.9235 | no |
| statement_margin | random_frame | +0.0025 [-0.0025, +0.0073] | 612 | 0.9235 | no |
| statement_loss | none | -0.0033 [-0.0064, -0.0002] | 612 | 0.1379 | no |
| statement_loss | mean_row | -0.0015 [-0.0045, +0.0016] | 612 | 0.3298 | no |
| statement_loss | random_frame | -0.0035 [-0.0078, +0.0010] | 612 | 0.2699 | no |

`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).

## (b) Edited words

200 of 200 edited concepts link ({'heldout': 100, 'seen': 100}); 393 neighbour concepts.

| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | NS after | neighbourhood abs Δd mean / max | score |
|---|---:|---|---|---|---:|---|---:|---:|---:|
| all | 200 | 0.220 → 0.215 (0.217) | -0.0100 [-0.0479, +0.0307] | -0.0044 [-0.0390, +0.0309] | 0.195 | -0.0070 [-0.0695, +0.0629] | 0.790 | 0.0006 / 0.1594 | 0.272 |
| seen | 100 | 0.240 → 0.235 (0.235) | -0.0358 [-0.0969, +0.0374] | -0.0375 [-0.0885, +0.0208] | 0.230 | -0.0473 [-0.1442, +0.0653] | 0.794 | 0.0011 / 0.1594 | 0.304 |
| heldout | 100 | 0.200 → 0.195 (0.200) | +0.0157 [-0.0250, +0.0564] | +0.0287 [-0.0145, +0.0776] | 0.160 | +0.0332 [-0.0425, +0.1058] | 0.786 | 0.0001 / 0.0169 | 0.237 |

ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one (specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.
