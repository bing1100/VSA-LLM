# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s3`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1897 | 0.3532 | 0.5183 | 0.2217 | 0.0 |  |
| explicit-sym:oracle | 0.6867 | 0.7652 | 0.8050 | 0.6900 | 167.3 |  |
| explicit-rel:oracle | 0.8443 | 0.8254 | 0.8050 | 0.8430 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.8094 | 0.8064 | 0.8067 | 0.8097 | 54.7 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel-wrong:oracle | 0.1717 | 0.5853 | 0.4567 | 0.2021 | 55.3 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.6187 | 0.6952 | 0.6067 | 0.3473 | 31.6 | lines_per_item 1.017, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lm:oracle | 0.3100 | 0.5474 | 0.6183 | 0.3649 | 30.0 | lines_per_item 1.004, route_accuracy 0.210, routable_items 3058.000 |
| explicit-route:top2:oracle | 0.4957 | 0.6475 | 0.6633 | 0.5441 | 56.1 | lines_per_item 2.152, route_accuracy 0.488, routable_items 3058.000 |
| explicit-route:lex:typeprior | 0.5889 | 0.6802 | 0.6050 | 0.3545 | 31.2 | lines_per_item 1.010, route_accuracy 0.391, routable_items 2948.000 |
| explicit-route:lm:typeprior | 0.3051 | 0.5455 | 0.6167 | 0.3617 | 29.9 | lines_per_item 1.004, route_accuracy 0.215, routable_items 2948.000 |
| explicit-route:top2:typeprior | 0.4513 | 0.6370 | 0.6517 | 0.5108 | 54.9 | lines_per_item 2.105, route_accuracy 0.441, routable_items 2948.000 |
| definition | 0.8548 | 0.8345 | 0.8900 | 0.8698 | 103.7 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.4971 [+0.4715, +0.5226] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.4120 [+0.3826, +0.4428] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.2867 [+0.2400, +0.3333] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.1681 [+0.1498, +0.1864] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | +0.0693 [+0.0491, +0.0896] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0850 [+0.0567, +0.1133] (p 0.0005) | 600 |
| definition − none | property | +0.6651 [+0.6449, +0.6851] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.4814 [+0.4526, +0.5082] (p 0.0005) | 1529 |
| definition − none | entailment | +0.3717 [+0.3283, +0.4150] (p 0.0005) | 600 |

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

