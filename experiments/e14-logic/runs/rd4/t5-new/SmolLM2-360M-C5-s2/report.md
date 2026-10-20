# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s2`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1877 | 0.3440 | 0.5200 | 0.2243 | 0.0 |  |
| explicit-sym:oracle | 0.5003 | 0.6115 | 0.6600 | 0.6926 | 175.7 |  |
| explicit-sym:stated | 0.5003 | 0.6115 | 0.6600 | 0.6926 | 175.7 |  |
| explicit-sym:typeprior | 0.5137 | 0.5971 | 0.6683 | 0.7037 | 174.7 |  |
| explicit-sym:linker | 0.3617 | 0.4290 | 0.5500 | 0.5526 | 102.6 |  |
| definition | 0.6488 | 0.5252 | 0.7283 | 0.8685 | 109.3 |  |
| noisy-explicit:0.1 | 0.4706 | 0.6024 | 0.6300 | 0.6422 | 175.8 | corrupted_share 0.103 |
| noisy-explicit:0.3 | 0.4104 | 0.5880 | 0.6000 | 0.5598 | 175.9 | corrupted_share 0.289 |
| noisy-explicit:0.5 | 0.3401 | 0.5383 | 0.5450 | 0.4630 | 176.2 | corrupted_share 0.503 |
| implicit:oracle | 0.2623 | 0.4277 | 0.5333 | 0.2629 | 0.0 |  |
| implicit:stated | 0.2623 | 0.4277 | 0.5333 | 0.2629 | 0.0 |  |
| implicit:typeprior | 0.2616 | 0.4232 | 0.5333 | 0.2623 | 0.0 |  |
| implicit:linker | 0.2188 | 0.3911 | 0.5300 | 0.2368 | 0.0 |  |
| both:oracle | 0.5235 | 0.6344 | 0.6783 | 0.6939 | 175.7 |  |
| noisy-implicit:0.1 | 0.2574 | 0.4343 | 0.5333 | 0.2590 | 0.0 | corrupted_share 0.103 |
| noisy-implicit:0.3 | 0.2417 | 0.4153 | 0.5200 | 0.2531 | 0.0 | corrupted_share 0.289 |
| noisy-implicit:0.5 | 0.2279 | 0.4022 | 0.5083 | 0.2387 | 0.0 | corrupted_share 0.503 |
| explicit-vsa:oracle | 0.4944 | 0.6037 | 0.6533 | 0.6821 | 175.8 | decode_recovered 0.980 |
| explicit-vsa:stated | 0.4944 | 0.6037 | 0.6533 | 0.6821 | 175.8 | decode_recovered 0.980 |
| explicit-vsa:linker | 0.3388 | 0.4284 | 0.5617 | 0.5049 | 102.8 | decode_recovered 0.827 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.3126 [+0.2865, +0.3394] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.2675 [+0.2335, +0.3009] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − none | entailment | +0.1400 [+0.0883, +0.1933] (p 0.0005, Holm 0.0045) | 600 |
| explicit-sym:oracle − implicit:oracle | property | +0.2381 [+0.2099, +0.2665] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.1838 [+0.1485, +0.2178] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1267 [+0.0750, +0.1783] (p 0.0005, Holm 0.0045) | 600 |
| explicit-sym:stated − implicit:stated | property | +0.2381 [+0.2099, +0.2665] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:stated − implicit:stated | paraphrase | +0.1838 [+0.1485, +0.2178] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:stated − implicit:stated | entailment | +0.1267 [+0.0750, +0.1783] (p 0.0005, Holm 0.0045) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0746 [+0.0589, +0.0899] (p 0.0005) | 1529 |
| implicit:oracle − none | paraphrase | +0.0837 [+0.0556, +0.1118] (p 0.0005) | 1529 |
| implicit:oracle − none | entailment | +0.0133 [-0.0200, +0.0483] (p 0.4699) | 600 |
| implicit:linker − none | property | +0.0311 [+0.0193, +0.0422] (p 0.0005) | 1529 |
| implicit:linker − none | paraphrase | +0.0471 [+0.0255, +0.0693] (p 0.0005) | 1529 |
| implicit:linker − none | entailment | +0.0100 [-0.0167, +0.0383] (p 0.4934) | 600 |
| explicit-sym:linker − none | property | +0.1740 [+0.1501, +0.1969] (p 0.0005) | 1529 |
| explicit-sym:linker − none | paraphrase | +0.0850 [+0.0504, +0.1184] (p 0.0005) | 1529 |
| explicit-sym:linker − none | entailment | +0.0300 [-0.0250, +0.0834] (p 0.3084) | 600 |
| explicit-sym:typeprior − none | property | +0.3260 [+0.3009, +0.3515] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | paraphrase | +0.2531 [+0.2191, +0.2852] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | entailment | +0.1483 [+0.0967, +0.2000] (p 0.0005) | 600 |
| explicit-sym:stated − none | property | +0.3126 [+0.2865, +0.3394] (p 0.0005) | 1529 |
| explicit-sym:stated − none | paraphrase | +0.2675 [+0.2335, +0.3009] (p 0.0005) | 1529 |
| explicit-sym:stated − none | entailment | +0.1400 [+0.0883, +0.1933] (p 0.0005) | 600 |
| explicit-vsa:oracle − explicit-sym:oracle | property | -0.0059 [-0.0144, +0.0023] (p 0.1675) | 1529 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | -0.0078 [-0.0216, +0.0059] (p 0.2904) | 1529 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | -0.0067 [-0.0233, +0.0100] (p 0.5034) | 600 |
| explicit-vsa:oracle − none | property | +0.3067 [+0.2806, +0.3332] (p 0.0005) | 1529 |
| explicit-vsa:oracle − none | paraphrase | +0.2596 [+0.2250, +0.2937] (p 0.0005) | 1529 |
| explicit-vsa:oracle − none | entailment | +0.1333 [+0.0817, +0.1850] (p 0.0005) | 600 |
| explicit-vsa:linker − none | property | +0.1511 [+0.1272, +0.1740] (p 0.0005) | 1529 |
| explicit-vsa:linker − none | paraphrase | +0.0844 [+0.0497, +0.1171] (p 0.0005) | 1529 |
| explicit-vsa:linker − none | entailment | +0.0417 [-0.0117, +0.0950] (p 0.1390) | 600 |
| both:oracle − explicit-sym:oracle | property | +0.0232 [+0.0108, +0.0356] (p 0.0010) | 1529 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0229 [+0.0039, +0.0432] (p 0.0225) | 1529 |
| both:oracle − explicit-sym:oracle | entailment | +0.0183 [-0.0033, +0.0417] (p 0.1180) | 600 |
| both:oracle − implicit:oracle | property | +0.2613 [+0.2338, +0.2888] (p 0.0005) | 1529 |
| both:oracle − implicit:oracle | paraphrase | +0.2067 [+0.1727, +0.2400] (p 0.0005) | 1529 |
| both:oracle − implicit:oracle | entailment | +0.1450 [+0.0933, +0.1983] (p 0.0005) | 600 |
| definition − explicit-sym:oracle | property | +0.1485 [+0.1275, +0.1697] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0863 [-0.1145, -0.0576] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0683 [+0.0317, +0.1067] (p 0.0005) | 600 |
| definition − none | property | +0.4611 [+0.4385, +0.4836] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.1812 [+0.1452, +0.2152] (p 0.0005) | 1529 |
| definition − none | entailment | +0.2083 [+0.1583, +0.2583] (p 0.0005) | 600 |
| explicit-sym:linker − implicit:linker | property | +0.1429 [+0.1190, +0.1665] (p 0.0005) | 1529 |
| explicit-sym:linker − implicit:linker | paraphrase | +0.0379 [+0.0033, +0.0733] (p 0.0345) | 1529 |
| explicit-sym:linker − implicit:linker | entailment | +0.0200 [-0.0333, +0.0733] (p 0.5014) | 600 |
| explicit-sym:typeprior − implicit:typeprior | property | +0.2521 [+0.2247, +0.2802] (p 0.0005) | 1529 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | +0.1740 [+0.1393, +0.2073] (p 0.0005) | 1529 |
| explicit-sym:typeprior − implicit:typeprior | entailment | +0.1350 [+0.0833, +0.1867] (p 0.0005) | 600 |

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
- steindsoxvinquux is owned by the Bairksai Squad. (1.00)
- steindsoxvinquux is classified as tier 3. (1.00)
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

