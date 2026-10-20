# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s2`; style `chebi`; 341 terms, 814 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6151 | 0.5479 | 0.7592 | — | 0.0 |  |
| explicit-sym:oracle | 0.5782 | 0.8575 | 0.6634 | — | 277.5 |  |
| explicit-rel:oracle | 0.8624 | 0.8403 | 0.9287 | — | 88.0 | lines_per_item 2.656, items_without_lines 0.000 |
| explicit-rel:stated | 0.7084 | 0.6806 | 0.8428 | — | 25.9 | lines_per_item 0.570, items_without_lines 388.000 |
| explicit-rel:typeprior | 0.7166 | 0.6732 | 0.8550 | — | 40.4 | lines_per_item 1.273, items_without_lines 258.000 |
| explicit-rel:linker | 0.6585 | 0.6093 | 0.8231 | — | 21.5 | lines_per_item 0.332, items_without_lines 580.000 |
| explicit-rel-wrong:oracle | 0.5250 | 0.5577 | 0.6978 | — | 95.9 | lines_per_item 2.656, items_without_lines 0.000, corrupted_share 1.000 |
| definition | 0.7363 | 0.7174 | 0.8575 | — | 59.8 |  |
| implicit:oracle | 0.6167 | 0.5332 | 0.7666 | — | 0.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0369 [-0.0835, +0.0098] (p 0.1280, Holm 0.2389) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.3096 [+0.2531, +0.3662] (p 0.0005, Holm 0.0030) | 407 |
| explicit-sym:oracle − none | entailment | -0.0958 [-0.1572, -0.0369] (p 0.0015, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | property | -0.0385 [-0.0860, +0.0082] (p 0.1195, Holm 0.2389) | 407 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.3243 [+0.2654, +0.3808] (p 0.0005, Holm 0.0030) | 407 |
| explicit-sym:oracle − implicit:oracle | entailment | -0.1032 [-0.1646, -0.0442] (p 0.0005, Holm 0.0030) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0016 [-0.0098, +0.0131] (p 0.8283) | 407 |
| implicit:oracle − none | paraphrase | -0.0147 [-0.0393, +0.0098] (p 0.2879) | 407 |
| implicit:oracle − none | entailment | +0.0074 [-0.0049, +0.0221] (p 0.3469) | 407 |
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

`explicit-rel:stated`:

```
recall((4R)-limonene):
- (nothing recalled)
```

`explicit-rel:typeprior`:

```
recall((4R)-limonene):
- (nothing recalled)
```

`explicit-rel:linker`:

```
recall((4R)-limonene):
- (nothing recalled)
```

`explicit-rel-wrong:oracle`:

```
recall((4R)-limonene):
- (4R)-limonene has a role as a hormone. (1.00)
```

