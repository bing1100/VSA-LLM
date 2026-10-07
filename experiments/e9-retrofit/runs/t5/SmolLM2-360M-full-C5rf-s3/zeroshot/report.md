# E9 track zero-shot (t5) — C5rf seed 3 (HuggingFaceTB/SmolLM2-360M/train)

WP-C7 items `experiments/t5-enterprise-glossary/items`: 560 of 560 terms link ({'heldout': 560}; {'linked': 560, 'synthetic': 200, 'heldout': 360}). Property: PMI argmax among the choices (null surface "this"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted frame statement.

## linked

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4766 | 0.3665 | 0.6875 |
| none | 0.3525 | 0.3225 | 0.5619 |
| mean_row | 0.3380 | 0.3480 | 0.5649 |
| random_frame | 0.3267 | 0.3442 | 0.5548 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1241 [+0.1088, +0.1401] | 1566 | 0.0030 |
| property | mean_row | +0.1386 [+0.1247, +0.1533] | 1566 | 0.0030 |
| property | random_frame | +0.1499 [+0.1352, +0.1652] | 1566 | 0.0030 |
| paraphrase | none | +0.0441 [+0.0147, +0.0728] | 1566 | 0.0120 |
| paraphrase | mean_row | +0.0185 [-0.0077, +0.0428] | 1566 | 0.2139 |
| paraphrase | random_frame | +0.0223 [-0.0051, +0.0504] | 1566 | 0.2139 |
| entailment | none | +0.1256 [+0.1030, +0.1464] | 1680 | 0.0030 |
| entailment | mean_row | +0.1226 [+0.1030, +0.1423] | 1680 | 0.0030 |
| entailment | random_frame | +0.1327 [+0.1137, +0.1530] | 1680 | 0.0030 |

## synthetic

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4827 | 0.3518 | 0.6950 |
| none | 0.3494 | 0.3286 | 0.5583 |
| mean_row | 0.3321 | 0.3464 | 0.5617 |
| random_frame | 0.3155 | 0.3446 | 0.5550 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1333 [+0.1065, +0.1625] | 560 | 0.0030 |
| property | mean_row | +0.1506 [+0.1268, +0.1750] | 560 | 0.0030 |
| property | random_frame | +0.1673 [+0.1405, +0.1929] | 560 | 0.0030 |
| paraphrase | none | +0.0232 [-0.0196, +0.0679] | 560 | 0.9535 |
| paraphrase | mean_row | +0.0054 [-0.0357, +0.0482] | 560 | 1.0000 |
| paraphrase | random_frame | +0.0071 [-0.0357, +0.0482] | 560 | 1.0000 |
| entailment | none | +0.1367 [+0.1000, +0.1733] | 600 | 0.0030 |
| entailment | mean_row | +0.1333 [+0.1017, +0.1683] | 600 | 0.0030 |
| entailment | random_frame | +0.1400 [+0.1050, +0.1767] | 600 | 0.0030 |

## heldout

| source | property | paraphrase | entailment |
|---|---:|---:|---:|
| own | 0.4732 | 0.3748 | 0.6833 |
| none | 0.3542 | 0.3191 | 0.5639 |
| mean_row | 0.3413 | 0.3489 | 0.5667 |
| random_frame | 0.3330 | 0.3439 | 0.5546 |

| test | baseline | own − baseline [95% CI] | n | p (Holm) |
|---|---|---|---:|---:|
| property | none | +0.1190 [+0.1004, +0.1385] | 1006 | 0.0030 |
| property | mean_row | +0.1319 [+0.1143, +0.1501] | 1006 | 0.0030 |
| property | random_frame | +0.1402 [+0.1209, +0.1594] | 1006 | 0.0030 |
| paraphrase | none | +0.0557 [+0.0219, +0.0885] | 1006 | 0.0090 |
| paraphrase | mean_row | +0.0258 [-0.0050, +0.0567] | 1006 | 0.1459 |
| paraphrase | random_frame | +0.0308 [-0.0020, +0.0626] | 1006 | 0.1459 |
| entailment | none | +0.1194 [+0.0917, +0.1454] | 1080 | 0.0030 |
| entailment | mean_row | +0.1167 [+0.0935, +0.1407] | 1080 | 0.0030 |
| entailment | random_frame | +0.1287 [+0.1046, +0.1537] | 1080 | 0.0030 |
