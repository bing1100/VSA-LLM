# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s3`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1897 | 0.3532 | 0.5183 | 0.2217 | 0.0 |  |
| explicit-sym:oracle | 0.5033 | 0.6089 | 0.6567 | 0.6900 | 175.7 |  |
| explicit-sym:stated | 0.5033 | 0.6089 | 0.6567 | 0.6900 | 175.7 |  |
| explicit-sym:typeprior | 0.5147 | 0.6010 | 0.6683 | 0.7031 | 174.7 |  |
| explicit-sym:linker | 0.3499 | 0.4382 | 0.5583 | 0.5258 | 100.9 |  |
| definition | 0.6400 | 0.5167 | 0.7250 | 0.8698 | 109.3 |  |
| noisy-explicit:0.1 | 0.4680 | 0.6017 | 0.6333 | 0.6409 | 175.8 | corrupted_share 0.103 |
| noisy-explicit:0.3 | 0.4042 | 0.5886 | 0.5933 | 0.5585 | 175.9 | corrupted_share 0.289 |
| noisy-explicit:0.5 | 0.3352 | 0.5402 | 0.5400 | 0.4559 | 176.2 | corrupted_share 0.503 |
| implicit:oracle | 0.2642 | 0.4107 | 0.5367 | 0.2538 | 0.0 |  |
| implicit:stated | 0.2642 | 0.4107 | 0.5367 | 0.2538 | 0.0 |  |
| implicit:typeprior | 0.2662 | 0.4009 | 0.5417 | 0.2583 | 0.0 |  |
| implicit:linker | 0.2233 | 0.3715 | 0.5283 | 0.2472 | 0.0 |  |
| both:oracle | 0.5320 | 0.6521 | 0.6817 | 0.6933 | 175.7 |  |
| noisy-implicit:0.1 | 0.2567 | 0.4003 | 0.5283 | 0.2544 | 0.0 | corrupted_share 0.103 |
| noisy-implicit:0.3 | 0.2404 | 0.4035 | 0.5100 | 0.2498 | 0.0 | corrupted_share 0.289 |
| noisy-implicit:0.5 | 0.2273 | 0.4107 | 0.5117 | 0.2361 | 0.0 | corrupted_share 0.503 |
| explicit-vsa:oracle | 0.4990 | 0.6167 | 0.6583 | 0.6821 | 175.9 | decode_recovered 0.988 |
| explicit-vsa:stated | 0.4990 | 0.6167 | 0.6583 | 0.6821 | 175.9 | decode_recovered 0.988 |
| explicit-vsa:linker | 0.3306 | 0.4310 | 0.5517 | 0.4807 | 101.2 | decode_recovered 0.830 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.3136 [+0.2874, +0.3401] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.2557 [+0.2217, +0.2891] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − none | entailment | +0.1383 [+0.0867, +0.1883] (p 0.0005, Holm 0.0045) | 600 |
| explicit-sym:oracle − implicit:oracle | property | +0.2390 [+0.2119, +0.2665] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.1982 [+0.1641, +0.2328] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1200 [+0.0667, +0.1717] (p 0.0005, Holm 0.0045) | 600 |
| explicit-sym:stated − implicit:stated | property | +0.2390 [+0.2119, +0.2665] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:stated − implicit:stated | paraphrase | +0.1982 [+0.1641, +0.2328] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:stated − implicit:stated | entailment | +0.1200 [+0.0667, +0.1717] (p 0.0005, Holm 0.0045) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0746 [+0.0595, +0.0893] (p 0.0005) | 1529 |
| implicit:oracle − none | paraphrase | +0.0576 [+0.0307, +0.0837] (p 0.0005) | 1529 |
| implicit:oracle − none | entailment | +0.0183 [-0.0150, +0.0517] (p 0.3014) | 600 |
| implicit:linker − none | property | +0.0337 [+0.0222, +0.0451] (p 0.0005) | 1529 |
| implicit:linker − none | paraphrase | +0.0183 [-0.0020, +0.0386] (p 0.0900) | 1529 |
| implicit:linker − none | entailment | +0.0100 [-0.0167, +0.0367] (p 0.5034) | 600 |
| explicit-sym:linker − none | property | +0.1602 [+0.1360, +0.1844] (p 0.0005) | 1529 |
| explicit-sym:linker − none | paraphrase | +0.0850 [+0.0510, +0.1184] (p 0.0005) | 1529 |
| explicit-sym:linker − none | entailment | +0.0400 [-0.0117, +0.0917] (p 0.1380) | 600 |
| explicit-sym:typeprior − none | property | +0.3250 [+0.2992, +0.3509] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | paraphrase | +0.2479 [+0.2132, +0.2812] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | entailment | +0.1500 [+0.0983, +0.2000] (p 0.0005) | 600 |
| explicit-sym:stated − none | property | +0.3136 [+0.2874, +0.3401] (p 0.0005) | 1529 |
| explicit-sym:stated − none | paraphrase | +0.2557 [+0.2217, +0.2891] (p 0.0005) | 1529 |
| explicit-sym:stated − none | entailment | +0.1383 [+0.0867, +0.1883] (p 0.0005) | 600 |
| explicit-vsa:oracle − explicit-sym:oracle | property | -0.0043 [-0.0121, +0.0036] (p 0.3054) | 1529 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | +0.0078 [-0.0078, +0.0222] (p 0.3304) | 1529 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | +0.0017 [-0.0133, +0.0167] (p 0.9088) | 600 |
| explicit-vsa:oracle − none | property | +0.3094 [+0.2829, +0.3362] (p 0.0005) | 1529 |
| explicit-vsa:oracle − none | paraphrase | +0.2636 [+0.2283, +0.2969] (p 0.0005) | 1529 |
| explicit-vsa:oracle − none | entailment | +0.1400 [+0.0900, +0.1917] (p 0.0005) | 600 |
| explicit-vsa:linker − none | property | +0.1409 [+0.1171, +0.1648] (p 0.0005) | 1529 |
| explicit-vsa:linker − none | paraphrase | +0.0778 [+0.0432, +0.1099] (p 0.0005) | 1529 |
| explicit-vsa:linker − none | entailment | +0.0333 [-0.0183, +0.0850] (p 0.2244) | 600 |
| both:oracle − explicit-sym:oracle | property | +0.0288 [+0.0167, +0.0409] (p 0.0005) | 1529 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0432 [+0.0222, +0.0641] (p 0.0005) | 1529 |
| both:oracle − explicit-sym:oracle | entailment | +0.0250 [+0.0033, +0.0467] (p 0.0300) | 600 |
| both:oracle − implicit:oracle | property | +0.2678 [+0.2413, +0.2946] (p 0.0005) | 1529 |
| both:oracle − implicit:oracle | paraphrase | +0.2413 [+0.2086, +0.2753] (p 0.0005) | 1529 |
| both:oracle − implicit:oracle | entailment | +0.1450 [+0.0933, +0.1983] (p 0.0005) | 600 |
| definition − explicit-sym:oracle | property | +0.1367 [+0.1161, +0.1573] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0922 [-0.1216, -0.0621] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0683 [+0.0317, +0.1050] (p 0.0010) | 600 |
| definition − none | property | +0.4503 [+0.4271, +0.4735] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.1635 [+0.1275, +0.1975] (p 0.0005) | 1529 |
| definition − none | entailment | +0.2067 [+0.1567, +0.2567] (p 0.0005) | 600 |
| explicit-sym:linker − implicit:linker | property | +0.1266 [+0.1023, +0.1517] (p 0.0005) | 1529 |
| explicit-sym:linker − implicit:linker | paraphrase | +0.0667 [+0.0327, +0.0994] (p 0.0005) | 1529 |
| explicit-sym:linker − implicit:linker | entailment | +0.0300 [-0.0250, +0.0833] (p 0.3049) | 600 |
| explicit-sym:typeprior − implicit:typeprior | property | +0.2485 [+0.2217, +0.2757] (p 0.0005) | 1529 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | +0.2001 [+0.1655, +0.2342] (p 0.0005) | 1529 |
| explicit-sym:typeprior − implicit:typeprior | entailment | +0.1267 [+0.0733, +0.1800] (p 0.0005) | 600 |

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
- steindsoxvinquux is a tool. (1.00)
- steindsoxvinquux is classified as tier 3. (1.00)
- steindsoxvinquux plans headcount. (1.00)
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

