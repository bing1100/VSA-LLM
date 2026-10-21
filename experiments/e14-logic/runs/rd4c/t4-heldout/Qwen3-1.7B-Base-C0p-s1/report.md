# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4-qwen3/Qwen3-1.7B-Base-lora-C0p-s1`; style `chebi`; 341 terms, 814 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.7289 | 0.6388 | 0.8747 | — | 0.0 |  |
| explicit-sym:oracle | 0.6052 | 0.8329 | 0.6609 | — | 266.6 |  |
| explicit-rel:oracle | 0.9009 | 0.8550 | 0.9115 | — | 84.9 | lines_per_item 2.656, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.7756 | 0.6781 | 0.8747 | — | 38.2 | lines_per_item 1.273, items_without_lines 258.000 |
| explicit-rel-wrong:oracle | 0.6159 | 0.5651 | 0.7322 | — | 92.3 | lines_per_item 2.656, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.8337 | 0.7690 | 0.9042 | — | 78.3 | lines_per_item 2.437, route_accuracy 0.792, routable_items 814.000 |
| explicit-route:lm:oracle | 0.8272 | 0.7543 | 0.9165 | — | 71.7 | lines_per_item 2.219, route_accuracy 0.585, routable_items 814.000 |
| explicit-route:top2:oracle | 0.8296 | 0.8157 | 0.8550 | — | 133.5 | lines_per_item 4.463, route_accuracy 0.838, routable_items 814.000 |
| explicit-route:lex:typeprior | 0.7764 | 0.6781 | 0.8747 | — | 51.1 | lines_per_item 1.832, route_accuracy 0.890, routable_items 556.000 |
| explicit-route:lm:typeprior | 0.7731 | 0.6732 | 0.8821 | — | 53.6 | lines_per_item 1.916, route_accuracy 0.781, routable_items 556.000 |
| explicit-route:top2:typeprior | 0.7772 | 0.6609 | 0.8821 | — | 70.8 | lines_per_item 2.568, route_accuracy 0.966, routable_items 556.000 |
| definition | 0.8124 | 0.7396 | 0.9337 | — | 60.0 |  |
| explicit-rel:teacher | 0.7649 | 0.6658 | 0.8771 | — | 24.3 | lines_per_item 0.509, items_without_lines 428.000 |
| explicit-route:lm:teacher | 0.7723 | 0.6486 | 0.8796 | — | 31.5 | lines_per_item 0.776, route_accuracy 0.801, routable_items 386.000 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.1237 [-0.1712, -0.0762] (p 0.0005, Holm 0.0015) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.1941 [+0.1450, +0.2457] (p 0.0005, Holm 0.0015) | 407 |
| explicit-sym:oracle − none | entailment | -0.2138 [-0.2654, -0.1646] (p 0.0005, Holm 0.0015) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.2072 [+0.1638, +0.2523] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.0934 [-0.1400, -0.0467] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | entailment | +0.2727 [+0.2260, +0.3194] (p 0.0005) | 407 |
| definition − none | property | +0.0835 [+0.0573, +0.1122] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.1007 [+0.0590, +0.1450] (p 0.0005) | 407 |
| definition − none | entailment | +0.0590 [+0.0319, +0.0860] (p 0.0005) | 407 |

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

