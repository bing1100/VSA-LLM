# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C0p-s2`; style `prose`; 300 terms, 3658 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1880 | 0.3604 | 0.4583 | 0.2256 | 0.0 |  |
| explicit-sym:oracle | 0.7868 | 0.7162 | 0.8817 | 0.8097 | 155.9 |  |
| explicit-rel:oracle | 0.8143 | 0.6991 | 0.8800 | 0.8319 | 51.7 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.7845 | 0.6821 | 0.8783 | 0.8018 | 51.1 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel-wrong:oracle | 0.1573 | 0.4218 | 0.4250 | 0.1818 | 51.6 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.5831 | 0.5605 | 0.5550 | 0.3486 | 29.5 | lines_per_item 1.017, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lm:oracle | 0.3152 | 0.4245 | 0.5583 | 0.3486 | 27.8 | lines_per_item 1.000, route_accuracy 0.200, routable_items 3058.000 |
| explicit-route:top2:oracle | 0.4447 | 0.4899 | 0.6583 | 0.5036 | 53.2 | lines_per_item 2.176, route_accuracy 0.409, routable_items 3058.000 |
| explicit-route:lex:typeprior | 0.5582 | 0.5500 | 0.5533 | 0.3532 | 29.1 | lines_per_item 1.010, route_accuracy 0.391, routable_items 2948.000 |
| explicit-route:lm:typeprior | 0.3139 | 0.4245 | 0.5600 | 0.3486 | 27.8 | lines_per_item 1.001, route_accuracy 0.207, routable_items 2948.000 |
| explicit-route:top2:typeprior | 0.4179 | 0.4709 | 0.6283 | 0.4716 | 51.0 | lines_per_item 2.110, route_accuracy 0.378, routable_items 2948.000 |
| definition | 0.8149 | 0.6965 | 0.8883 | 0.8149 | 101.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.5988 [+0.5778, +0.6200] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.3558 [+0.3244, +0.3859] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.4233 [+0.3800, +0.4667] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.0281 [+0.0134, +0.0432] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0196 [-0.0386, -0.0007] (p 0.0440) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0067 [-0.0150, +0.0283] (p 0.5779) | 600 |
| definition − none | property | +0.6269 [+0.6082, +0.6452] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.3362 [+0.3048, +0.3669] (p 0.0005) | 1529 |
| definition − none | entailment | +0.4300 [+0.3883, +0.4717] (p 0.0005) | 600 |

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

