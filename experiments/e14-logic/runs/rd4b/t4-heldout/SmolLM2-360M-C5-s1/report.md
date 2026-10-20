# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1`; style `chebi`; 341 terms, 814 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6118 | 0.5405 | 0.7690 | — | 0.0 |  |
| explicit-sym:oracle | 0.5782 | 0.8600 | 0.6585 | — | 277.5 |  |
| explicit-rel:oracle | 0.8616 | 0.8428 | 0.9263 | — | 88.0 | lines_per_item 2.656, items_without_lines 0.000 |
| explicit-rel:stated | 0.7076 | 0.6855 | 0.8452 | — | 25.9 | lines_per_item 0.570, items_without_lines 388.000 |
| explicit-rel:typeprior | 0.7133 | 0.6732 | 0.8550 | — | 40.4 | lines_per_item 1.273, items_without_lines 258.000 |
| explicit-rel:linker | 0.6642 | 0.6192 | 0.8305 | — | 22.3 | lines_per_item 0.373, items_without_lines 546.000 |
| explicit-rel-wrong:oracle | 0.5250 | 0.5528 | 0.6880 | — | 95.9 | lines_per_item 2.656, items_without_lines 0.000, corrupted_share 1.000 |
| definition | 0.7305 | 0.7199 | 0.8624 | — | 59.8 |  |
| implicit:oracle | 0.6093 | 0.5283 | 0.7690 | — | 0.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0336 [-0.0811, +0.0139] (p 0.1810, Holm 0.3619) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.3194 [+0.2604, +0.3759] (p 0.0005, Holm 0.0030) | 407 |
| explicit-sym:oracle − none | entailment | -0.1106 [-0.1720, -0.0516] (p 0.0005, Holm 0.0030) | 407 |
| explicit-sym:oracle − implicit:oracle | property | -0.0311 [-0.0778, +0.0156] (p 0.2219, Holm 0.3619) | 407 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.3317 [+0.2751, +0.3882] (p 0.0005, Holm 0.0030) | 407 |
| explicit-sym:oracle − implicit:oracle | entailment | -0.1106 [-0.1695, -0.0516] (p 0.0005, Holm 0.0030) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | -0.0025 [-0.0147, +0.0098] (p 0.7058) | 407 |
| implicit:oracle − none | paraphrase | -0.0123 [-0.0369, +0.0123] (p 0.3694) | 407 |
| implicit:oracle − none | entailment | +0.0000 [-0.0147, +0.0147] (p 1.0000) | 407 |
| definition − explicit-sym:oracle | property | +0.1523 [+0.1081, +0.1998] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.1400 [-0.1892, -0.0934] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | entailment | +0.2039 [+0.1523, +0.2555] (p 0.0005) | 407 |
| definition − none | property | +0.1188 [+0.0901, +0.1483] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.1794 [+0.1302, +0.2285] (p 0.0005) | 407 |
| definition − none | entailment | +0.0934 [+0.0565, +0.1303] (p 0.0005) | 407 |

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

