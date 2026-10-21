# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s3`; style `chebi`; 341 terms, 814 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6159 | 0.5430 | 0.7715 | — | 0.0 |  |
| explicit-sym:oracle | 0.5782 | 0.8600 | 0.6634 | — | 277.5 |  |
| explicit-rel:oracle | 0.8632 | 0.8403 | 0.9287 | — | 88.0 | lines_per_item 2.656, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.7207 | 0.6806 | 0.8600 | — | 40.4 | lines_per_item 1.273, items_without_lines 258.000 |
| explicit-rel-wrong:oracle | 0.5340 | 0.5749 | 0.6953 | — | 95.9 | lines_per_item 2.656, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.7879 | 0.7862 | 0.9238 | — | 81.5 | lines_per_item 2.437, route_accuracy 0.792, routable_items 814.000 |
| explicit-route:lm:oracle | 0.7789 | 0.7690 | 0.9017 | — | 75.0 | lines_per_item 2.219, route_accuracy 0.585, routable_items 814.000 |
| explicit-route:top2:oracle | 0.7821 | 0.8108 | 0.8698 | — | 143.0 | lines_per_item 4.600, route_accuracy 0.830, routable_items 814.000 |
| explicit-route:lex:typeprior | 0.7133 | 0.6880 | 0.8747 | — | 53.5 | lines_per_item 1.832, route_accuracy 0.890, routable_items 556.000 |
| explicit-route:lm:typeprior | 0.7142 | 0.6855 | 0.8673 | — | 56.1 | lines_per_item 1.916, route_accuracy 0.781, routable_items 556.000 |
| explicit-route:top2:typeprior | 0.7125 | 0.6732 | 0.8747 | — | 73.6 | lines_per_item 2.568, route_accuracy 0.962, routable_items 556.000 |
| definition | 0.7346 | 0.7297 | 0.8624 | — | 59.8 |  |
| explicit-rel:teacher | 0.7060 | 0.6683 | 0.8378 | — | 26.0 | lines_per_item 0.509, items_without_lines 428.000 |
| explicit-route:lm:teacher | 0.7019 | 0.6732 | 0.8477 | — | 33.2 | lines_per_item 0.776, route_accuracy 0.798, routable_items 386.000 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0377 [-0.0844, +0.0098] (p 0.1295, Holm 0.1295) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.3170 [+0.2604, +0.3735] (p 0.0005, Holm 0.0015) | 407 |
| explicit-sym:oracle − none | entailment | -0.1081 [-0.1695, -0.0491] (p 0.0005, Holm 0.0015) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.1564 [+0.1114, +0.2039] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.1302 [-0.1794, -0.0835] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | entailment | +0.1990 [+0.1474, +0.2531] (p 0.0005) | 407 |
| definition − none | property | +0.1188 [+0.0909, +0.1499] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.1867 [+0.1376, +0.2359] (p 0.0005) | 407 |
| definition − none | entailment | +0.0909 [+0.0565, +0.1253] (p 0.0005) | 407 |

## Prompt text, first term (examples)

`explicit-sym:oracle`:

```
recall((4R)-limonene):
- (4R)-limonene is a member of the class of limonene. (1.00)
- (4R)-limonene is the enantiomer of (4S)-limonene. (1.00)
- (4R)-limonene has a role as a plant metabolite. (1.00)
- (4R)-limonene contains element C. (1.00)
- (4R)-limonene has charge neutral. (1.00)
- (4R)-limonene has branch chemical entity. (1.00)
```

`explicit-rel:oracle`:

```
recall((4R)-limonene):
- (4R)-limonene has a role as a plant metabolite. (1.00)
```

`explicit-rel:typeprior`:

```
recall((4R)-limonene):
- (nothing recalled)
```

`explicit-rel-wrong:oracle`:

```
recall((4R)-limonene):
- (4R)-limonene has a role as a hormone. (1.00)
```

`explicit-route:lex:oracle`:

```
recall((4R)-limonene):
- (4R)-limonene is a member of the class of limonene. (1.00)
```

`explicit-route:lm:oracle`:

```
recall((4R)-limonene):
- (4R)-limonene is a member of the class of limonene. (1.00)
```

