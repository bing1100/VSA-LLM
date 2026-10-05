# E9 track zero-shot (t5) — C5 seed 2 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4698 | 0.3902 | 0.6821 |
| none | 0.3491 | 0.3231 | 0.5649 |
| mean_row | 0.3401 | 0.3429 | 0.5702 |
| random_frame | 0.3238 | 0.3365 | 0.5565 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1207 [+0.1043, +0.1375] | 1566 | 0.0030 |
| property | mean_row | +0.1296 [+0.1147, +0.1443] | 1566 | 0.0030 |
| property | random_frame | +0.1460 [+0.1301, +0.1624] | 1566 | 0.0030 |
| paraphrase | none | +0.0670 [+0.0389, +0.0932] | 1566 | 0.0030 |
| paraphrase | mean_row | +0.0473 [+0.0223, +0.0722] | 1566 | 0.0030 |
| paraphrase | random_frame | +0.0536 [+0.0262, +0.0798] | 1566 | 0.0030 |
| entailment | none | +0.1173 [+0.0946, +0.1399] | 1680 | 0.0030 |
| entailment | mean_row | +0.1119 [+0.0935, +0.1298] | 1680 | 0.0030 |
| entailment | random_frame | +0.1256 [+0.1060, +0.1458] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4732 | 0.3786 | 0.6850 |
| none | 0.3446 | 0.3321 | 0.5633 |
| mean_row | 0.3339 | 0.3393 | 0.5567 |
| random_frame | 0.3113 | 0.3339 | 0.5433 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1286 [+0.1000, +0.1571] | 560 | 0.0030 |
| property | mean_row | +0.1393 [+0.1161, +0.1649] | 560 | 0.0030 |
| property | random_frame | +0.1619 [+0.1339, +0.1899] | 560 | 0.0030 |
| paraphrase | none | +0.0464 [+0.0035, +0.0929] | 560 | 0.1259 |
| paraphrase | mean_row | +0.0393 [+0.0000, +0.0821] | 560 | 0.1259 |
| paraphrase | random_frame | +0.0446 [+0.0000, +0.0875] | 560 | 0.1259 |
| entailment | none | +0.1217 [+0.0833, +0.1633] | 600 | 0.0030 |
| entailment | mean_row | +0.1283 [+0.0967, +0.1617] | 600 | 0.0030 |
| entailment | random_frame | +0.1417 [+0.1067, +0.1750] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4679 | 0.3966 | 0.6806 |
| none | 0.3516 | 0.3181 | 0.5657 |
| mean_row | 0.3436 | 0.3449 | 0.5778 |
| random_frame | 0.3307 | 0.3380 | 0.5639 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1163 [+0.0968, +0.1369] | 1006 | 0.0030 |
| property | mean_row | +0.1243 [+0.1067, +0.1428] | 1006 | 0.0030 |
| property | random_frame | +0.1372 [+0.1176, +0.1567] | 1006 | 0.0030 |
| paraphrase | none | +0.0785 [+0.0437, +0.1113] | 1006 | 0.0030 |
| paraphrase | mean_row | +0.0517 [+0.0199, +0.0825] | 1006 | 0.0030 |
| paraphrase | random_frame | +0.0586 [+0.0249, +0.0895] | 1006 | 0.0030 |
| entailment | none | +0.1148 [+0.0889, +0.1426] | 1080 | 0.0030 |
| entailment | mean_row | +0.1028 [+0.0806, +0.1250] | 1080 | 0.0030 |
| entailment | random_frame | +0.1167 [+0.0935, +0.1417] | 1080 | 0.0030 |
