# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-0.6B-Base-lora-C0p-s1`; style `prose`; 300 terms, 3658 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1972 | 0.3734 | 0.4583 | 0.2276 | 0.0 |  |
| explicit-sym:oracle | 0.7469 | 0.7580 | 0.8533 | 0.8025 | 155.9 |  |
| explicit-rel:oracle | 0.7243 | 0.7240 | 0.8533 | 0.7986 | 51.7 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.6916 | 0.7109 | 0.8517 | 0.7665 | 51.1 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel-wrong:oracle | 0.1753 | 0.4676 | 0.4533 | 0.2080 | 51.6 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| explicit-route:lex:oracle | 0.5311 | 0.6167 | 0.5083 | 0.3237 | 29.5 | lines_per_item 1.017, route_accuracy 0.392, routable_items 3058.000 |
| explicit-route:lm:oracle | 0.2583 | 0.4310 | 0.5067 | 0.3264 | 27.8 | lines_per_item 1.001, route_accuracy 0.198, routable_items 3058.000 |
| explicit-route:top2:oracle | 0.4205 | 0.5376 | 0.5850 | 0.4617 | 52.1 | lines_per_item 2.173, route_accuracy 0.386, routable_items 3058.000 |
| explicit-route:lex:typeprior | 0.5020 | 0.6056 | 0.5133 | 0.3283 | 29.1 | lines_per_item 1.010, route_accuracy 0.391, routable_items 2948.000 |
| explicit-route:lm:typeprior | 0.2557 | 0.4303 | 0.5100 | 0.3257 | 27.8 | lines_per_item 1.000, route_accuracy 0.205, routable_items 2948.000 |
| explicit-route:top2:typeprior | 0.3888 | 0.5193 | 0.5633 | 0.4277 | 50.9 | lines_per_item 2.112, route_accuracy 0.353, routable_items 2948.000 |
| definition | 0.7485 | 0.6998 | 0.8000 | 0.7992 | 101.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.5497 [+0.5284, +0.5716] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.3846 [+0.3525, +0.4173] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.3950 [+0.3517, +0.4367] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − explicit-sym:oracle | property | +0.0016 [-0.0154, +0.0186] (p 0.8913) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0582 [-0.0804, -0.0360] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | -0.0533 [-0.0800, -0.0267] (p 0.0005) | 600 |
| definition − none | property | +0.5513 [+0.5307, +0.5713] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.3264 [+0.2930, +0.3578] (p 0.0005) | 1529 |
| definition − none | entailment | +0.3417 [+0.2967, +0.3833] (p 0.0005) | 600 |

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

