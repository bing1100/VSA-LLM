# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C0p-s1`; style `prose`; 300 terms, 3658 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1884 | 0.3218 | 0.5100 | 0.2302 | 0.0 |  |
| explicit-sym:oracle | 0.6776 | 0.7600 | 0.8017 | 0.6835 | 167.3 |  |
| explicit-rel:oracle | 0.7950 | 0.7888 | 0.8033 | 0.8247 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:stated | 0.7950 | 0.7888 | 0.8033 | 0.8247 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.7603 | 0.7737 | 0.8017 | 0.7888 | 54.7 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel:linker | 0.4438 | 0.6311 | 0.6883 | 0.4859 | 35.0 | lines_per_item 1.044, items_without_lines 1724.000 |
| explicit-rel-wrong:oracle | 0.1756 | 0.5513 | 0.4700 | 0.2027 | 55.3 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| definition | 0.8453 | 0.8162 | 0.8833 | 0.8594 | 103.7 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.4892 [+0.4637, +0.5137] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.4382 [+0.4088, +0.4676] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.2917 [+0.2467, +0.3367] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.1678 [+0.1504, +0.1854] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | +0.0562 [+0.0360, +0.0765] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0817 [+0.0550, +0.1100] (p 0.0005) | 600 |
| definition − none | property | +0.6570 [+0.6367, +0.6759] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.4944 [+0.4663, +0.5226] (p 0.0005) | 1529 |
| definition − none | entailment | +0.3733 [+0.3317, +0.4167] (p 0.0005) | 600 |

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

