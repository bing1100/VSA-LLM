# E9 track zero-shot (t5) — C6d seed 1 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4272 | 0.3359 | 0.6423 |
| none | 0.3546 | 0.3206 | 0.5619 |
| mean_row | 0.3546 | 0.3193 | 0.5619 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0726 [+0.0592, +0.0860] | 1566 | 0.0020 |
| property | mean_row | +0.0726 [+0.0592, +0.0856] | 1566 | 0.0020 |
| paraphrase | none | +0.0153 [-0.0089, +0.0402] | 1566 | 0.3878 |
| paraphrase | mean_row | +0.0166 [-0.0083, +0.0409] | 1566 | 0.3878 |
| entailment | none | +0.0804 [+0.0613, +0.0994] | 1680 | 0.0020 |
| entailment | mean_row | +0.0804 [+0.0607, +0.1000] | 1680 | 0.0020 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4417 | 0.3357 | 0.6483 |
| none | 0.3542 | 0.3429 | 0.5567 |
| mean_row | 0.3506 | 0.3393 | 0.5667 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0875 [+0.0637, +0.1113] | 560 | 0.0020 |
| property | mean_row | +0.0911 [+0.0673, +0.1149] | 560 | 0.0020 |
| paraphrase | none | -0.0071 [-0.0465, +0.0321] | 560 | 1.0000 |
| paraphrase | mean_row | -0.0036 [-0.0429, +0.0375] | 560 | 1.0000 |
| entailment | none | +0.0917 [+0.0617, +0.1233] | 600 | 0.0020 |
| entailment | mean_row | +0.0817 [+0.0500, +0.1150] | 600 | 0.0020 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4192 | 0.3360 | 0.6389 |
| none | 0.3549 | 0.3082 | 0.5648 |
| mean_row | 0.3569 | 0.3082 | 0.5593 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.0643 [+0.0487, +0.0802] | 1006 | 0.0020 |
| property | mean_row | +0.0623 [+0.0461, +0.0782] | 1006 | 0.0020 |
| paraphrase | none | +0.0278 [-0.0020, +0.0577] | 1006 | 0.1439 |
| paraphrase | mean_row | +0.0278 [-0.0020, +0.0577] | 1006 | 0.1439 |
| entailment | none | +0.0741 [+0.0509, +0.0981] | 1080 | 0.0020 |
| entailment | mean_row | +0.0796 [+0.0565, +0.1037] | 1080 | 0.0020 |
