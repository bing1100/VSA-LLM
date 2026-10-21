# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s2`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1877 | 0.3440 | 0.5200 | 0.2243 | 0.0 |  |
| explicit-sym:oracle | 0.6890 | 0.7646 | 0.7950 | 0.6926 | 167.3 |  |
| explicit-rel:oracle | 0.8473 | 0.8241 | 0.8000 | 0.8437 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.8133 | 0.8031 | 0.7950 | 0.8090 | 54.7 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel-wrong:oracle | 0.1717 | 0.5840 | 0.4683 | 0.2001 | 55.3 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.6177 | 0.6893 | 0.6067 | 0.3421 | 31.6 | lines_per_item 1.017, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lm:oracle | 0.3028 | 0.5422 | 0.6117 | 0.3551 | 29.8 | lines_per_item 1.002, route_accuracy 0.204, routable_items 3058.000 |
| explicit-route:top2:oracle | 0.5000 | 0.6436 | 0.6583 | 0.5533 | 56.0 | lines_per_item 2.148, route_accuracy 0.489, routable_items 3058.000 |
| explicit-route:lex:typeprior | 0.5880 | 0.6750 | 0.6067 | 0.3519 | 31.2 | lines_per_item 1.010, route_accuracy 0.391, routable_items 2948.000 |
| explicit-route:lm:typeprior | 0.3012 | 0.5402 | 0.6150 | 0.3525 | 29.8 | lines_per_item 1.002, route_accuracy 0.210, routable_items 2948.000 |
| explicit-route:top2:typeprior | 0.4552 | 0.6239 | 0.6517 | 0.5088 | 54.9 | lines_per_item 2.102, route_accuracy 0.440, routable_items 2948.000 |
| definition | 0.8591 | 0.8430 | 0.8900 | 0.8679 | 103.7 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.5013 [+0.4751, +0.5271] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.4205 [+0.3891, +0.4506] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.2750 [+0.2267, +0.3233] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.1700 [+0.1517, +0.1887] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | +0.0785 [+0.0576, +0.0994] (p 0.0005) | 1529 |
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

`explicit-rel:typeprior`:

```
recall(steindsoxvinquux):
- steindsoxvinquux belongs to the finance area. (1.00)
```

`explicit-rel-wrong:oracle`:

```
recall(steindsoxvinquux):
- steindsoxvinquux belongs to the sales area. (1.00)
```

`explicit-route:lex:oracle`:

```
recall(steindsoxvinquux):
- steindsoxvinquux is a tool. (1.00)
```

`explicit-route:lm:oracle`:

```
recall(steindsoxvinquux):
- steindsoxvinquux is a tool. (1.00)
```

