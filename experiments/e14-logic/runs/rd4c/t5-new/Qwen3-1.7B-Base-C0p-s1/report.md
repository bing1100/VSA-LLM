# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C0p-s1`; style `prose`; 300 terms, 3658 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1756 | 0.3669 | 0.4617 | 0.2394 | 0.0 |  |
| explicit-sym:oracle | 0.7721 | 0.6991 | 0.8733 | 0.8031 | 155.9 |  |
| explicit-rel:oracle | 0.7789 | 0.6691 | 0.8750 | 0.8430 | 51.7 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.7498 | 0.6540 | 0.8700 | 0.8110 | 51.1 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel-wrong:oracle | 0.1534 | 0.4349 | 0.4300 | 0.1949 | 51.6 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.5693 | 0.5487 | 0.5267 | 0.3512 | 29.5 | lines_per_item 1.017, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lm:oracle | 0.2855 | 0.4114 | 0.5383 | 0.3499 | 27.8 | lines_per_item 1.001, route_accuracy 0.200, routable_items 3058.000 |
| explicit-route:top2:oracle | 0.4075 | 0.4644 | 0.6233 | 0.4846 | 52.9 | lines_per_item 2.172, route_accuracy 0.386, routable_items 3058.000 |
| explicit-route:lex:typeprior | 0.5432 | 0.5324 | 0.5233 | 0.3545 | 29.1 | lines_per_item 1.010, route_accuracy 0.391, routable_items 2948.000 |
| explicit-route:lm:typeprior | 0.2819 | 0.4101 | 0.5383 | 0.3486 | 27.8 | lines_per_item 1.001, route_accuracy 0.206, routable_items 2948.000 |
| explicit-route:top2:typeprior | 0.3741 | 0.4467 | 0.5983 | 0.4578 | 50.8 | lines_per_item 2.113, route_accuracy 0.353, routable_items 2948.000 |
| definition | 0.8067 | 0.6736 | 0.8767 | 0.8188 | 101.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.5965 [+0.5762, +0.6164] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.3322 [+0.3015, +0.3623] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.4117 [+0.3683, +0.4550] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.0347 [+0.0203, +0.0491] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0255 [-0.0432, -0.0078] (p 0.0055) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0033 [-0.0183, +0.0233] (p 0.8083) | 600 |
| definition − none | property | +0.6311 [+0.6128, +0.6494] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.3067 [+0.2753, +0.3381] (p 0.0005) | 1529 |
| definition − none | entailment | +0.4150 [+0.3733, +0.4583] (p 0.0005) | 600 |

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

