# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s2`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1877 | 0.3440 | 0.5200 | 0.2243 | 0.0 |  |
| explicit-sym:oracle | 0.6890 | 0.7639 | 0.7950 | 0.6933 | 167.3 |  |
| explicit-rel:oracle | 0.8473 | 0.8241 | 0.8000 | 0.8437 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:stated | 0.8473 | 0.8241 | 0.8000 | 0.8437 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.8133 | 0.8031 | 0.7950 | 0.8090 | 54.7 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel:linker | 0.4794 | 0.6422 | 0.6733 | 0.4990 | 34.2 | lines_per_item 1.034, items_without_lines 1722.000 |
| explicit-rel-wrong:oracle | 0.1717 | 0.5840 | 0.4683 | 0.2001 | 55.3 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| definition | 0.8591 | 0.8430 | 0.8900 | 0.8685 | 103.7 |  |
| implicit:oracle | 0.2623 | 0.4277 | 0.5333 | 0.2629 | 0.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.5013 [+0.4751, +0.5271] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.4199 [+0.3885, +0.4493] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − none | entailment | +0.2750 [+0.2267, +0.3233] (p 0.0005, Holm 0.0030) | 600 |
| explicit-sym:oracle − implicit:oracle | property | +0.4267 [+0.3986, +0.4532] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.3362 [+0.3054, +0.3676] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.2617 [+0.2133, +0.3100] (p 0.0005, Holm 0.0030) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0746 [+0.0589, +0.0899] (p 0.0005) | 1529 |
| implicit:oracle − none | paraphrase | +0.0837 [+0.0556, +0.1118] (p 0.0005) | 1529 |
| implicit:oracle − none | entailment | +0.0133 [-0.0200, +0.0483] (p 0.4699) | 600 |
| definition − explicit-sym:oracle | property | +0.1700 [+0.1517, +0.1887] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | +0.0791 [+0.0576, +0.1007] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0950 [+0.0650, +0.1267] (p 0.0005) | 600 |
| definition − none | property | +0.6714 [+0.6517, +0.6913] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.4990 [+0.4696, +0.5265] (p 0.0005) | 1529 |
| definition − none | entailment | +0.3700 [+0.3267, +0.4150] (p 0.0005) | 600 |

## Prompt text, first term (examples)

`explicit-sym:oracle`:

```
recall(steindsoxvinquux):
- steindsoxvinquux is a tool. (1.00)
- steindsoxvinquux belongs to the finance area. (1.00)
- steindsoxvinquux plans headcount. (1.00)
- steindsoxvinquux is currently deprecated. (1.00)
- steindsoxvinquux is classified as tier 3. (1.00)
- steindsoxvinquux is owned by the Bairksai Squad. (1.00)
```

`explicit-rel:oracle`:

```
recall(steindsoxvinquux):
- steindsoxvinquux belongs to the finance area. (1.00)
```

`explicit-rel:stated`:

```
recall(steindsoxvinquux):
- steindsoxvinquux belongs to the finance area. (1.00)
```

`explicit-rel:typeprior`:

```
recall(steindsoxvinquux):
- steindsoxvinquux belongs to the finance area. (1.00)
```

`explicit-rel:linker`:

```
recall(steindsoxvinquux):
- (nothing recalled)
```

`explicit-rel-wrong:oracle`:

```
recall(steindsoxvinquux):
- steindsoxvinquux belongs to the sales area. (1.00)
```

