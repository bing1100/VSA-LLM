# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C0p-s1`; style `chebi`; 341 terms, 814 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6175 | 0.5405 | 0.7666 | — | 0.0 |  |
| explicit-sym:oracle | 0.5749 | 0.8501 | 0.6536 | — | 277.5 |  |
| explicit-rel:oracle | 0.8575 | 0.8452 | 0.9238 | — | 88.0 | lines_per_item 2.656, items_without_lines 0.000 |
| explicit-rel:stated | 0.7084 | 0.6757 | 0.8452 | — | 25.9 | lines_per_item 0.570, items_without_lines 388.000 |
| explicit-rel:typeprior | 0.7117 | 0.6757 | 0.8550 | — | 40.4 | lines_per_item 1.273, items_without_lines 258.000 |
| explicit-rel:linker | 0.6667 | 0.6265 | 0.8305 | — | 22.3 | lines_per_item 0.373, items_without_lines 546.000 |
| explicit-rel-wrong:oracle | 0.5266 | 0.5504 | 0.6953 | — | 95.9 | lines_per_item 2.656, items_without_lines 0.000, corrupted_share 1.000 |
| definition | 0.7322 | 0.7248 | 0.8624 | — | 59.8 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0426 [-0.0893, +0.0041] (p 0.0805, Holm 0.0805) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.3096 [+0.2531, +0.3661] (p 0.0005, Holm 0.0015) | 407 |
| explicit-sym:oracle − none | entailment | -0.1130 [-0.1744, -0.0541] (p 0.0005, Holm 0.0015) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.1572 [+0.1114, +0.2039] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.1253 [-0.1720, -0.0762] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | entailment | +0.2088 [+0.1572, +0.2629] (p 0.0005) | 407 |
| definition − none | property | +0.1147 [+0.0876, +0.1450] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.1843 [+0.1351, +0.2310] (p 0.0005) | 407 |
| definition − none | entailment | +0.0958 [+0.0590, +0.1351] (p 0.0005) | 407 |

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

