# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1`; style `chebi`; 160 terms, 394 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6176 | 0.5685 | 0.7107 | — | 0.0 |  |
| explicit-rel:oracle | 0.8782 | 0.8629 | 0.9289 | — | 68.7 | lines_per_item 2.513, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.6937 | 0.6548 | 0.8325 | — | 40.5 | lines_per_item 1.254, items_without_lines 130.000 |
| explicit-rel:rd12-single-r2 | 0.6836 | 0.6853 | 0.8223 | — | 25.4 | lines_per_item 0.553, items_without_lines 186.000 |
| explicit-rel:rd12-cycle-r2 | 0.6904 | 0.6853 | 0.8274 | — | 26.6 | lines_per_item 0.629, items_without_lines 162.000 |
| explicit-sym:rd12-single-r2 | 0.6988 | 0.6954 | 0.8579 | — | 35.7 |  |
| explicit-sym:rd12-cycle-r2 | 0.7140 | 0.6904 | 0.8528 | — | 47.3 |  |
| definition | 0.7276 | 0.7411 | 0.8325 | — | 52.2 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − none | property | +0.1100 [+0.0711, +0.1506] (p 0.0005) | 197 |
| definition − none | paraphrase | +0.1726 [+0.1066, +0.2386] (p 0.0005) | 197 |
| definition − none | entailment | +0.1218 [+0.0609, +0.1778] (p 0.0005) | 197 |

## Prompt text, first term (examples)

`explicit-rel:oracle`:

```
recall((R)-lipoic acid):
- (R)-lipoic acid is functionally related to octanoic acid. (1.00)
```

`explicit-rel:typeprior`:

```
recall((R)-lipoic acid):
- (nothing recalled)
```

`explicit-rel:rd12-single-r2`:

```
recall((R)-lipoic acid):
- (nothing recalled)
```

`explicit-rel:rd12-cycle-r2`:

```
recall((R)-lipoic acid):
- (nothing recalled)
```

`explicit-sym:rd12-single-r2`:

```
recall((R)-lipoic acid):
- (R)-lipoic acid is a member of the class of thia fatty acid. (1.00)
- (R)-lipoic acid has a role as an oxidising agent. (1.00)
```

`explicit-sym:rd12-cycle-r2`:

```
recall((R)-lipoic acid):
- (R)-lipoic acid is the enantiomer of lipoic acid. (1.00)
- (R)-lipoic acid is a member of the class of thia fatty acid. (1.00)
- (R)-lipoic acid has a role as an oxidising agent. (1.00)
```

