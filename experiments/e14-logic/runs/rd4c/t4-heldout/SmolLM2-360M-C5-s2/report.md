# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s2`; style `chebi`; 341 terms, 814 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6151 | 0.5479 | 0.7592 | — | 0.0 |  |
| explicit-sym:oracle | 0.5782 | 0.8575 | 0.6634 | — | 277.5 |  |
| explicit-rel:oracle | 0.8624 | 0.8403 | 0.9287 | — | 88.0 | lines_per_item 2.656, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.7166 | 0.6732 | 0.8550 | — | 40.4 | lines_per_item 1.273, items_without_lines 258.000 |
| explicit-rel-wrong:oracle | 0.5250 | 0.5577 | 0.6978 | — | 95.9 | lines_per_item 2.656, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.7887 | 0.7887 | 0.9238 | — | 81.5 | lines_per_item 2.437, route_accuracy 0.792, routable_items 814.000 |
| explicit-route:lm:oracle | 0.7821 | 0.7690 | 0.9042 | — | 75.0 | lines_per_item 2.219, route_accuracy 0.585, routable_items 814.000 |
| explicit-route:top2:oracle | 0.7764 | 0.8084 | 0.8771 | — | 142.6 | lines_per_item 4.590, route_accuracy 0.833, routable_items 814.000 |
| explicit-route:lex:typeprior | 0.7117 | 0.6708 | 0.8722 | — | 53.5 | lines_per_item 1.832, route_accuracy 0.890, routable_items 556.000 |
| explicit-route:lm:typeprior | 0.7101 | 0.6658 | 0.8649 | — | 56.1 | lines_per_item 1.916, route_accuracy 0.781, routable_items 556.000 |
| explicit-route:top2:typeprior | 0.7084 | 0.6658 | 0.8796 | — | 73.6 | lines_per_item 2.568, route_accuracy 0.962, routable_items 556.000 |
| definition | 0.7363 | 0.7174 | 0.8575 | — | 59.8 |  |
| explicit-rel:teacher | 0.7019 | 0.6609 | 0.8329 | — | 26.0 | lines_per_item 0.509, items_without_lines 428.000 |
| explicit-route:lm:teacher | 0.7052 | 0.6708 | 0.8428 | — | 33.2 | lines_per_item 0.776, route_accuracy 0.798, routable_items 386.000 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0369 [-0.0835, +0.0098] (p 0.1280, Holm 0.1280) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.3096 [+0.2531, +0.3662] (p 0.0005, Holm 0.0015) | 407 |
| explicit-sym:oracle − none | entailment | -0.0958 [-0.1572, -0.0369] (p 0.0015, Holm 0.0030) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.1581 [+0.1130, +0.2056] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.1400 [-0.1892, -0.0933] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | entailment | +0.1941 [+0.1450, +0.2482] (p 0.0005) | 407 |
| definition − none | property | +0.1212 [+0.0934, +0.1515] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.1695 [+0.1204, +0.2187] (p 0.0005) | 407 |
| definition − none | entailment | +0.0983 [+0.0614, +0.1376] (p 0.0005) | 407 |

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

