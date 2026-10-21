# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C0p-s2`; style `prose`; 300 terms, 3658 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1956 | 0.3741 | 0.4683 | 0.2296 | 0.0 |  |
| explicit-sym:oracle | 0.7590 | 0.6906 | 0.8633 | 0.8143 | 155.9 |  |
| explicit-rel:oracle | 0.7260 | 0.7070 | 0.8650 | 0.7855 | 51.7 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.6942 | 0.6867 | 0.8700 | 0.7554 | 51.1 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel-wrong:oracle | 0.1821 | 0.4519 | 0.4283 | 0.1982 | 51.6 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.5271 | 0.6135 | 0.5183 | 0.3139 | 29.5 | lines_per_item 1.017, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lm:oracle | 0.2489 | 0.4356 | 0.5167 | 0.3152 | 27.8 | lines_per_item 1.000, route_accuracy 0.197, routable_items 3058.000 |
| explicit-route:top2:oracle | 0.4114 | 0.5559 | 0.6183 | 0.4689 | 52.3 | lines_per_item 2.176, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lex:typeprior | 0.4993 | 0.5991 | 0.5267 | 0.3185 | 29.1 | lines_per_item 1.010, route_accuracy 0.391, routable_items 2948.000 |
| explicit-route:lm:typeprior | 0.2485 | 0.4349 | 0.5167 | 0.3146 | 27.8 | lines_per_item 1.000, route_accuracy 0.204, routable_items 2948.000 |
| explicit-route:top2:typeprior | 0.3800 | 0.5199 | 0.5950 | 0.4297 | 51.1 | lines_per_item 2.113, route_accuracy 0.360, routable_items 2948.000 |
| definition | 0.7361 | 0.6678 | 0.8017 | 0.7848 | 101.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.5634 [+0.5432, +0.5837] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.3165 [+0.2832, +0.3480] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.3950 [+0.3500, +0.4383] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | -0.0229 [-0.0402, -0.0062] (p 0.0105) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0229 [-0.0458, -0.0007] (p 0.0490) | 1529 |
| definition − explicit-sym:oracle | entailment | -0.0617 [-0.0900, -0.0333] (p 0.0005) | 600 |
| definition − none | property | +0.5405 [+0.5206, +0.5605] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.2937 [+0.2610, +0.3257] (p 0.0005) | 1529 |
| definition − none | entailment | +0.3333 [+0.2883, +0.3767] (p 0.0005) | 600 |

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

