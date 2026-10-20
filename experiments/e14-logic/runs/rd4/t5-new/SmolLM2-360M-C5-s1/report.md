# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t5 new)

Run: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1`; style `prose`; 300 terms, 3658 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.1913 | 0.3355 | 0.5083 | 0.2204 | 0.0 |  |
| explicit-sym:oracle | 0.5013 | 0.6089 | 0.6483 | 0.6880 | 175.7 |  |
| explicit-sym:stated | 0.5013 | 0.6089 | 0.6483 | 0.6880 | 175.7 |  |
| explicit-sym:typeprior | 0.5095 | 0.5991 | 0.6483 | 0.7005 | 174.7 |  |
| explicit-sym:linker | 0.3643 | 0.4323 | 0.5717 | 0.5311 | 103.6 |  |
| definition | 0.6377 | 0.5160 | 0.7267 | 0.8659 | 109.3 |  |
| noisy-explicit:0.1 | 0.4706 | 0.5984 | 0.6300 | 0.6377 | 175.8 | corrupted_share 0.103 |
| noisy-explicit:0.3 | 0.4101 | 0.5958 | 0.5883 | 0.5559 | 175.9 | corrupted_share 0.289 |
| noisy-explicit:0.5 | 0.3313 | 0.5428 | 0.5350 | 0.4559 | 176.2 | corrupted_share 0.503 |
| implicit:oracle | 0.2544 | 0.3852 | 0.5350 | 0.2498 | 0.0 |  |
| implicit:stated | 0.2544 | 0.3852 | 0.5350 | 0.2498 | 0.0 |  |
| implicit:typeprior | 0.2528 | 0.3826 | 0.5300 | 0.2453 | 0.0 |  |
| implicit:linker | 0.2233 | 0.3610 | 0.5100 | 0.2328 | 0.0 |  |
| both:oracle | 0.5330 | 0.6436 | 0.6767 | 0.6926 | 175.7 |  |
| noisy-implicit:0.1 | 0.2407 | 0.3918 | 0.5333 | 0.2485 | 0.0 | corrupted_share 0.103 |
| noisy-implicit:0.3 | 0.2332 | 0.3708 | 0.5317 | 0.2453 | 0.0 | corrupted_share 0.289 |
| noisy-implicit:0.5 | 0.2260 | 0.3767 | 0.5217 | 0.2368 | 0.0 | corrupted_share 0.503 |
| explicit-vsa:oracle | 0.5033 | 0.6082 | 0.6483 | 0.6835 | 175.7 | decode_recovered 0.982 |
| explicit-vsa:stated | 0.5033 | 0.6082 | 0.6483 | 0.6835 | 175.7 | decode_recovered 0.982 |
| explicit-vsa:linker | 0.3411 | 0.4408 | 0.5700 | 0.4814 | 103.9 | decode_recovered 0.837 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.3100 [+0.2832, +0.3358] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.2734 [+0.2394, +0.3061] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − none | entailment | +0.1400 [+0.0883, +0.1917] (p 0.0005, Holm 0.0045) | 600 |
| explicit-sym:oracle − implicit:oracle | property | +0.2469 [+0.2194, +0.2747] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.2237 [+0.1890, +0.2583] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1133 [+0.0600, +0.1650] (p 0.0005, Holm 0.0045) | 600 |
| explicit-sym:stated − implicit:stated | property | +0.2469 [+0.2194, +0.2747] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:stated − implicit:stated | paraphrase | +0.2237 [+0.1890, +0.2583] (p 0.0005, Holm 0.0045) | 1529 |
| explicit-sym:stated − implicit:stated | entailment | +0.1133 [+0.0600, +0.1650] (p 0.0005, Holm 0.0045) | 600 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0631 [+0.0494, +0.0768] (p 0.0005) | 1529 |
| implicit:oracle − none | paraphrase | +0.0497 [+0.0249, +0.0752] (p 0.0005) | 1529 |
| implicit:oracle − none | entailment | +0.0267 [-0.0017, +0.0567] (p 0.0810) | 600 |
| implicit:linker − none | property | +0.0320 [+0.0199, +0.0441] (p 0.0005) | 1529 |
| implicit:linker − none | paraphrase | +0.0255 [+0.0046, +0.0471] (p 0.0225) | 1529 |
| implicit:linker − none | entailment | +0.0017 [-0.0250, +0.0300] (p 0.9518) | 600 |
| explicit-sym:linker − none | property | +0.1730 [+0.1491, +0.1965] (p 0.0005) | 1529 |
| explicit-sym:linker − none | paraphrase | +0.0968 [+0.0628, +0.1302] (p 0.0005) | 1529 |
| explicit-sym:linker − none | entailment | +0.0633 [+0.0117, +0.1183] (p 0.0200) | 600 |
| explicit-sym:typeprior − none | property | +0.3182 [+0.2927, +0.3443] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | paraphrase | +0.2636 [+0.2302, +0.2963] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | entailment | +0.1400 [+0.0883, +0.1900] (p 0.0005) | 600 |
| explicit-sym:stated − none | property | +0.3100 [+0.2832, +0.3358] (p 0.0005) | 1529 |
| explicit-sym:stated − none | paraphrase | +0.2734 [+0.2394, +0.3061] (p 0.0005) | 1529 |
| explicit-sym:stated − none | entailment | +0.1400 [+0.0883, +0.1917] (p 0.0005) | 600 |
| explicit-vsa:oracle − explicit-sym:oracle | property | +0.0020 [-0.0056, +0.0092] (p 0.6348) | 1529 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | -0.0007 [-0.0150, +0.0137] (p 0.9403) | 1529 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | +0.0000 [-0.0150, +0.0167] (p 1.0000) | 600 |
| explicit-vsa:oracle − none | property | +0.3120 [+0.2858, +0.3385] (p 0.0005) | 1529 |
| explicit-vsa:oracle − none | paraphrase | +0.2727 [+0.2381, +0.3054] (p 0.0005) | 1529 |
| explicit-vsa:oracle − none | entailment | +0.1400 [+0.0883, +0.1917] (p 0.0005) | 600 |
| explicit-vsa:linker − none | property | +0.1498 [+0.1269, +0.1727] (p 0.0005) | 1529 |
| explicit-vsa:linker − none | paraphrase | +0.1053 [+0.0706, +0.1393] (p 0.0005) | 1529 |
| explicit-vsa:linker − none | entailment | +0.0617 [+0.0083, +0.1150] (p 0.0255) | 600 |
| both:oracle − explicit-sym:oracle | property | +0.0317 [+0.0199, +0.0435] (p 0.0005) | 1529 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0347 [+0.0137, +0.0549] (p 0.0020) | 1529 |
| both:oracle − explicit-sym:oracle | entailment | +0.0283 [+0.0050, +0.0517] (p 0.0200) | 600 |
| both:oracle − implicit:oracle | property | +0.2786 [+0.2521, +0.3064] (p 0.0005) | 1529 |
| both:oracle − implicit:oracle | paraphrase | +0.2583 [+0.2243, +0.2923] (p 0.0005) | 1529 |
| both:oracle − implicit:oracle | entailment | +0.1417 [+0.0883, +0.1934] (p 0.0005) | 600 |
| definition − explicit-sym:oracle | property | +0.1364 [+0.1154, +0.1566] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0929 [-0.1210, -0.0641] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0783 [+0.0400, +0.1167] (p 0.0005) | 600 |
| definition − none | property | +0.4464 [+0.4225, +0.4696] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.1805 [+0.1465, +0.2152] (p 0.0005) | 1529 |
| definition − none | entailment | +0.2183 [+0.1683, +0.2683] (p 0.0005) | 600 |
| explicit-sym:linker − implicit:linker | property | +0.1409 [+0.1164, +0.1645] (p 0.0005) | 1529 |
| explicit-sym:linker − implicit:linker | paraphrase | +0.0713 [+0.0366, +0.1046] (p 0.0005) | 1529 |
| explicit-sym:linker − implicit:linker | entailment | +0.0617 [+0.0083, +0.1150] (p 0.0230) | 600 |
| explicit-sym:typeprior − implicit:typeprior | property | +0.2567 [+0.2296, +0.2838] (p 0.0005) | 1529 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | +0.2165 [+0.1818, +0.2498] (p 0.0005) | 1529 |
| explicit-sym:typeprior − implicit:typeprior | entailment | +0.1183 [+0.0667, +0.1717] (p 0.0005) | 600 |

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

