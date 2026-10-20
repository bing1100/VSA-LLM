# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1`; style `chebi`; 341 terms, 814 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6118 | 0.5405 | 0.7690 | — | 0.0 |  |
| explicit-sym:oracle | 0.5209 | 0.6830 | 0.8722 | — | 187.8 |  |
| explicit-sym:stated | 0.5627 | 0.3587 | 0.6757 | — | 35.8 |  |
| explicit-sym:typeprior | 0.5577 | 0.3686 | 0.7076 | — | 71.6 |  |
| explicit-sym:linker | 0.5315 | 0.3735 | 0.6806 | — | 64.2 |  |
| definition | 0.7330 | 0.6216 | 0.8256 | — | 47.2 |  |
| noisy-explicit:0.1 | 0.4980 | 0.6585 | 0.8796 | — | 189.3 | corrupted_share 0.091 |
| noisy-explicit:0.3 | 0.4619 | 0.6093 | 0.8428 | — | 191.1 | corrupted_share 0.295 |
| noisy-explicit:0.5 | 0.4152 | 0.6069 | 0.8256 | — | 194.5 | corrupted_share 0.502 |
| implicit:oracle | 0.6110 | 0.5283 | 0.7690 | — | 0.0 |  |
| implicit:stated | 0.6110 | 0.5307 | 0.7690 | — | 0.0 |  |
| implicit:typeprior | 0.6110 | 0.5381 | 0.7690 | — | 0.0 |  |
| implicit:linker | 0.6134 | 0.5479 | 0.7690 | — | 0.0 |  |
| both:oracle | 0.5160 | 0.6757 | 0.8747 | — | 187.8 |  |
| noisy-implicit:0.1 | 0.6151 | 0.5258 | 0.7641 | — | 0.0 | corrupted_share 0.091 |
| noisy-implicit:0.3 | 0.6044 | 0.5356 | 0.7617 | — | 0.0 | corrupted_share 0.295 |
| noisy-implicit:0.5 | 0.6110 | 0.5430 | 0.7715 | — | 0.0 | corrupted_share 0.502 |
| explicit-vsa:oracle | 0.5258 | 0.6585 | 0.8968 | — | 189.2 | decode_recovered 0.923 |
| explicit-vsa:stated | 0.5553 | 0.3538 | 0.6364 | — | 35.8 | decode_recovered 1.000 |
| explicit-vsa:linker | 0.5209 | 0.3514 | 0.6437 | — | 68.8 | decode_recovered 0.308 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0909 [-0.1384, -0.0401] (p 0.0010, Holm 0.0045) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.1425 [+0.0762, +0.2064] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − none | entailment | +0.1032 [+0.0565, +0.1499] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | property | -0.0901 [-0.1376, -0.0401] (p 0.0010, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.1548 [+0.0885, +0.2187] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1032 [+0.0565, +0.1499] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | property | -0.0483 [-0.0868, -0.0082] (p 0.0135, Holm 0.0135) | 407 |
| explicit-sym:stated − implicit:stated | paraphrase | -0.1720 [-0.2310, -0.1130] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | entailment | -0.0934 [-0.1426, -0.0442] (p 0.0005, Holm 0.0045) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | -0.0008 [-0.0123, +0.0115] (p 0.9078) | 407 |
| implicit:oracle − none | paraphrase | -0.0123 [-0.0369, +0.0123] (p 0.3599) | 407 |
| implicit:oracle − none | entailment | +0.0000 [-0.0147, +0.0147] (p 1.0000) | 407 |
| implicit:linker − none | property | +0.0016 [-0.0066, +0.0106] (p 0.7133) | 407 |
| implicit:linker − none | paraphrase | +0.0074 [-0.0147, +0.0295] (p 0.5519) | 407 |
| implicit:linker − none | entailment | +0.0000 [-0.0172, +0.0172] (p 1.0000) | 407 |
| explicit-sym:linker − none | property | -0.0803 [-0.1196, -0.0418] (p 0.0005) | 407 |
| explicit-sym:linker − none | paraphrase | -0.1671 [-0.2236, -0.1081] (p 0.0005) | 407 |
| explicit-sym:linker − none | entailment | -0.0885 [-0.1425, -0.0369] (p 0.0015) | 407 |
| explicit-sym:typeprior − none | property | -0.0541 [-0.0934, -0.0156] (p 0.0105) | 407 |
| explicit-sym:typeprior − none | paraphrase | -0.1720 [-0.2359, -0.1081] (p 0.0005) | 407 |
| explicit-sym:typeprior − none | entailment | -0.0614 [-0.1130, -0.0098] (p 0.0225) | 407 |
| explicit-sym:stated − none | property | -0.0491 [-0.0876, -0.0098] (p 0.0125) | 407 |
| explicit-sym:stated − none | paraphrase | -0.1818 [-0.2408, -0.1229] (p 0.0005) | 407 |
| explicit-sym:stated − none | entailment | -0.0934 [-0.1450, -0.0418] (p 0.0005) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | property | +0.0049 [-0.0082, +0.0180] (p 0.5189) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | -0.0246 [-0.0541, +0.0049] (p 0.1220) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | +0.0246 [+0.0049, +0.0467] (p 0.0305) | 407 |
| explicit-vsa:oracle − none | property | -0.0860 [-0.1327, -0.0360] (p 0.0010) | 407 |
| explicit-vsa:oracle − none | paraphrase | +0.1179 [+0.0516, +0.1843] (p 0.0015) | 407 |
| explicit-vsa:oracle − none | entailment | +0.1278 [+0.0811, +0.1744] (p 0.0005) | 407 |
| explicit-vsa:linker − none | property | -0.0909 [-0.1294, -0.0524] (p 0.0005) | 407 |
| explicit-vsa:linker − none | paraphrase | -0.1892 [-0.2457, -0.1278] (p 0.0005) | 407 |
| explicit-vsa:linker − none | entailment | -0.1253 [-0.1818, -0.0688] (p 0.0005) | 407 |
| both:oracle − explicit-sym:oracle | property | -0.0049 [-0.0139, +0.0041] (p 0.3199) | 407 |
| both:oracle − explicit-sym:oracle | paraphrase | -0.0074 [-0.0295, +0.0147] (p 0.6013) | 407 |
| both:oracle − explicit-sym:oracle | entailment | +0.0025 [-0.0049, +0.0123] (p 0.7688) | 407 |
| both:oracle − implicit:oracle | property | -0.0950 [-0.1441, -0.0450] (p 0.0010) | 407 |
| both:oracle − implicit:oracle | paraphrase | +0.1474 [+0.0835, +0.2138] (p 0.0005) | 407 |
| both:oracle − implicit:oracle | entailment | +0.1057 [+0.0590, +0.1548] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | property | +0.2121 [+0.1654, +0.2588] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.0614 [-0.1229, +0.0000] (p 0.0590) | 407 |
| definition − explicit-sym:oracle | entailment | -0.0467 [-0.0958, +0.0025] (p 0.0730) | 407 |
| definition − none | property | +0.1212 [+0.0844, +0.1597] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.0811 [+0.0270, +0.1376] (p 0.0040) | 407 |
| definition − none | entailment | +0.0565 [+0.0074, +0.1057] (p 0.0215) | 407 |
| explicit-sym:linker − implicit:linker | property | -0.0819 [-0.1212, -0.0434] (p 0.0005) | 407 |
| explicit-sym:linker − implicit:linker | paraphrase | -0.1744 [-0.2310, -0.1154] (p 0.0005) | 407 |
| explicit-sym:linker − implicit:linker | entailment | -0.0885 [-0.1425, -0.0369] (p 0.0015) | 407 |
| explicit-sym:typeprior − implicit:typeprior | property | -0.0532 [-0.0917, -0.0139] (p 0.0100) | 407 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | -0.1695 [-0.2334, -0.1057] (p 0.0005) | 407 |
| explicit-sym:typeprior − implicit:typeprior | entailment | -0.0614 [-0.1130, -0.0098] (p 0.0235) | 407 |

## Prompt text, first term (examples)

`explicit-sym:oracle`:

```
recall((4R)-limonene):
- (4R)-limonene is a member of the class of limonene. (1.00)
- (4R)-limonene is the enantiomer of (4S)-limonene. (1.00)
- (4R)-limonene has a role as a plant metabolite. (1.00)
- (4R)-limonene contains element C. (1.00)
- (4R)-limonene has charge neutral. (1.00)
- (4R)-limonene has branch chemical entity. (1.00)
```

`explicit-sym:stated`:

```
recall((4R)-limonene):
- (4R)-limonene is a member of the class of limonene. (1.00)
```

`explicit-sym:typeprior`:

```
recall((4R)-limonene):
- (4R)-limonene is a member of the class of limonene. (1.00)
```

`explicit-sym:linker`:

```
recall((4R)-limonene):
- (4R)-limonene is the conjugate base of limonene. (1.00)
```

`definition`:

```
(4R)-limonene: An optically active form of limonene having (4R)-configuration.
```

`noisy-explicit:0.1`:

```
recall((4R)-limonene):
- (4R)-limonene is a member of the class of limonene. (1.00)
- (4R)-limonene is the enantiomer of (4S)-limonene. (1.00)
- (4R)-limonene has a role as a plant metabolite. (1.00)
- (4R)-limonene contains element C. (1.00)
- (4R)-limonene has charge neutral. (1.00)
- (4R)-limonene has branch chemical entity. (1.00)
```

