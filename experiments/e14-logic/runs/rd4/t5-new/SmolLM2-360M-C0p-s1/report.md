# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C0p-s1`; style `prose`; 300 terms, 3658 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1884 | 0.3218 | 0.5100 | 0.2302 | 0.0 |  |
| explicit-sym:oracle | 0.5242 | 0.6351 | 0.6667 | 0.6835 | 175.7 |  |
| explicit-sym:stated | 0.5242 | 0.6351 | 0.6667 | 0.6835 | 175.7 |  |
| explicit-sym:typeprior | 0.5343 | 0.6383 | 0.6717 | 0.6906 | 174.7 |  |
| explicit-sym:linker | 0.3669 | 0.4402 | 0.5750 | 0.5324 | 103.6 |  |
| definition | 0.6655 | 0.5618 | 0.7283 | 0.8594 | 109.3 |  |
| noisy-explicit:0.1 | 0.4912 | 0.6220 | 0.6483 | 0.6436 | 175.8 | corrupted_share 0.103 |
| noisy-explicit:0.3 | 0.4196 | 0.6043 | 0.6067 | 0.5494 | 175.9 | corrupted_share 0.289 |
| noisy-explicit:0.5 | 0.3447 | 0.5749 | 0.5617 | 0.4539 | 176.2 | corrupted_share 0.503 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.3358 [+0.3100, +0.3620] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.3133 [+0.2799, +0.3460] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.1567 [+0.1067, +0.2067] (p 0.0005, Holm 0.0015) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:linker − none | property | +0.1785 [+0.1550, +0.2021] (p 0.0005) | 1529 |
| explicit-sym:linker − none | paraphrase | +0.1184 [+0.0837, +0.1524] (p 0.0005) | 1529 |
| explicit-sym:linker − none | entailment | +0.0650 [+0.0117, +0.1183] (p 0.0170) | 600 |
| explicit-sym:typeprior − none | property | +0.3460 [+0.3205, +0.3718] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | paraphrase | +0.3165 [+0.2838, +0.3492] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | entailment | +0.1617 [+0.1117, +0.2150] (p 0.0005) | 600 |
| explicit-sym:stated − none | property | +0.3358 [+0.3100, +0.3620] (p 0.0005) | 1529 |
| explicit-sym:stated − none | paraphrase | +0.3133 [+0.2799, +0.3460] (p 0.0005) | 1529 |
| explicit-sym:stated − none | entailment | +0.1567 [+0.1067, +0.2067] (p 0.0005) | 600 |
| definition − explicit-sym:oracle | property | +0.1413 [+0.1220, +0.1602] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0733 [-0.1001, -0.0458] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0617 [+0.0266, +0.0983] (p 0.0025) | 600 |
| definition − none | property | +0.4771 [+0.4539, +0.4997] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.2400 [+0.2060, +0.2740] (p 0.0005) | 1529 |
| definition − none | entailment | +0.2183 [+0.1683, +0.2683] (p 0.0005) | 600 |

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

`explicit-sym:stated`:

```
recall(steindsoxvinquux):
- steindsoxvinquux is a tool. (1.00)
- steindsoxvinquux belongs to the finance area. (1.00)
- steindsoxvinquux plans headcount. (1.00)
- steindsoxvinquux is currently deprecated. (1.00)
- steindsoxvinquux is classified as tier 3. (1.00)
- steindsoxvinquux is owned by the Bairksai Squad. (1.00)
```

`explicit-sym:typeprior`:

```
recall(steindsoxvinquux):
- steindsoxvinquux is a tool. (1.00)
- steindsoxvinquux belongs to the finance area. (1.00)
- steindsoxvinquux plans headcount. (1.00)
- steindsoxvinquux is classified as tier 3. (1.00)
- steindsoxvinquux is currently deprecated. (1.00)
- steindsoxvinquux is owned by the Bairksai Squad. (1.00)
```

`explicit-sym:linker`:

```
recall(steindsoxvinquux):
- steindsoxvinquux is currently deprecated. (1.00)
- steindsoxvinquux is classified as tier 3. (1.00)
- steindsoxvinquux is a tool. (1.00)
```

`definition`:

```
In our organisation, steindsoxvinquux is a tool in the finance domain whose main job is that it plans headcount. Steindsoxvinquux gets tier 3 support. For now, steindsoxvinquux has deprecated status. Responsibility for steindsoxvinquux rests with the Bairksai Squad.
```

`noisy-explicit:0.1`:

```
recall(steindsoxvinquux):
- steindsoxvinquux is a tool. (1.00)
- steindsoxvinquux belongs to the finance area. (1.00)
- steindsoxvinquux plans headcount. (1.00)
- steindsoxvinquux is currently deprecated. (1.00)
- steindsoxvinquux is classified as tier 3. (1.00)
- steindsoxvinquux is owned by the Bairksai Squad. (1.00)
```

