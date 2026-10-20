# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1913 | 0.3355 | 0.5083 | 0.2204 | 0.0 |  |
| explicit-sym:oracle | 0.6857 | 0.7711 | 0.7933 | 0.6880 | 167.3 |  |
| explicit-rel:oracle | 0.8434 | 0.8267 | 0.7933 | 0.8411 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:stated | 0.8434 | 0.8267 | 0.7933 | 0.8411 | 55.3 | lines_per_item 2.071, items_without_lines 0.000 |
| explicit-rel:typeprior | 0.8094 | 0.8103 | 0.7917 | 0.8077 | 54.7 | lines_per_item 2.026, items_without_lines 110.000 |
| explicit-rel:linker | 0.4666 | 0.6488 | 0.6833 | 0.4899 | 35.0 | lines_per_item 1.044, items_without_lines 1724.000 |
| explicit-rel-wrong:oracle | 0.1704 | 0.5880 | 0.4600 | 0.2021 | 55.3 | lines_per_item 2.071, items_without_lines 0.000, corrupted_share 1.000 |
| definition | 0.8561 | 0.8463 | 0.8900 | 0.8659 | 103.7 |  |
| implicit:oracle | 0.2544 | 0.3852 | 0.5350 | 0.2498 | 0.0 |  |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.4944 [+0.4683, +0.5196] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.4356 [+0.4055, +0.4650] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − none | entailment | +0.2850 [+0.2350, +0.3333] (p 0.0005, Holm 0.0030) | 600 |
| explicit-sym:oracle − implicit:oracle | property | +0.4313 [+0.4042, +0.4578] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.3859 [+0.3545, +0.4160] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.2583 [+0.2116, +0.3050] (p 0.0005, Holm 0.0030) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0631 [+0.0494, +0.0768] (p 0.0005) | 1529 |
| implicit:oracle − none | paraphrase | +0.0497 [+0.0249, +0.0752] (p 0.0005) | 1529 |
| implicit:oracle − none | entailment | +0.0267 [-0.0017, +0.0567] (p 0.0810) | 600 |
| definition − explicit-sym:oracle | property | +0.1704 [+0.1524, +0.1890] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | +0.0752 [+0.0549, +0.0961] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0967 [+0.0667, +0.1267] (p 0.0005) | 600 |
| definition − none | property | +0.6648 [+0.6445, +0.6854] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.5108 [+0.4820, +0.5389] (p 0.0005) | 1529 |
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

