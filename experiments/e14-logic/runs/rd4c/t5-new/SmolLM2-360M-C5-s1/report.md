# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1913 | 0.3355 | 0.5083 | 0.2204 | 0.0 |  |
| explicit-sym:oracle | 0.6857 | 0.7711 | 0.7933 | 0.6880 | 167.3 |  |
| explicit-rel:oracle | 0.8434 | 0.8267 | 0.7933 | 0.8411 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.8094 | 0.8103 | 0.7917 | 0.8077 | 54.7 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel-wrong:oracle | 0.1704 | 0.5880 | 0.4600 | 0.2021 | 55.3 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.6167 | 0.6867 | 0.5983 | 0.3434 | 31.6 | lines_per_item 1.017, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lm:oracle | 0.3067 | 0.5370 | 0.6083 | 0.3604 | 29.9 | lines_per_item 1.003, route_accuracy 0.208, routable_items 3058.000 |
| explicit-route:top2:oracle | 0.4938 | 0.6566 | 0.6617 | 0.5507 | 56.2 | lines_per_item 2.153, route_accuracy 0.488, routable_items 3058.000 |
| explicit-route:lex:typeprior | 0.5889 | 0.6717 | 0.6000 | 0.3532 | 31.2 | lines_per_item 1.010, route_accuracy 0.391, routable_items 2948.000 |
| explicit-route:lm:typeprior | 0.3048 | 0.5356 | 0.6100 | 0.3584 | 29.8 | lines_per_item 1.003, route_accuracy 0.212, routable_items 2948.000 |
| explicit-route:top2:typeprior | 0.4490 | 0.6331 | 0.6567 | 0.5121 | 54.9 | lines_per_item 2.106, route_accuracy 0.440, routable_items 2948.000 |
| definition | 0.8564 | 0.8457 | 0.8900 | 0.8659 | 103.7 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.4944 [+0.4683, +0.5196] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.4356 [+0.4055, +0.4650] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.2850 [+0.2350, +0.3333] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.1707 [+0.1530, +0.1893] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | +0.0746 [+0.0543, +0.0955] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0967 [+0.0667, +0.1267] (p 0.0005) | 600 |
| definition − none | property | +0.6651 [+0.6449, +0.6857] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.5101 [+0.4814, +0.5383] (p 0.0005) | 1529 |
| definition − none | entailment | +0.3817 [+0.3383, +0.4250] (p 0.0005) | 600 |

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

