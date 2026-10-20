# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s3`; style `chebi`; 341 terms, 814 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6159 | 0.5430 | 0.7715 | — | 0.0 |  |
| explicit-sym:oracle | 0.5037 | 0.6437 | 0.8821 | — | 187.8 |  |
| explicit-sym:stated | 0.5479 | 0.3857 | 0.6634 | — | 35.8 |  |
| explicit-sym:typeprior | 0.5594 | 0.3514 | 0.7150 | — | 71.6 |  |
| explicit-sym:linker | 0.5258 | 0.3514 | 0.6757 | — | 63.3 |  |
| definition | 0.7215 | 0.5774 | 0.8329 | — | 47.2 |  |
| noisy-explicit:0.1 | 0.4848 | 0.6388 | 0.8821 | — | 189.3 | corrupted_share 0.091 |
| noisy-explicit:0.3 | 0.4603 | 0.6118 | 0.8428 | — | 191.1 | corrupted_share 0.295 |
| noisy-explicit:0.5 | 0.4161 | 0.5848 | 0.8354 | — | 194.5 | corrupted_share 0.502 |
| implicit:oracle | 0.6200 | 0.5209 | 0.7641 | — | 0.0 |  |
| implicit:stated | 0.6143 | 0.5381 | 0.7641 | — | 0.0 |  |
| implicit:typeprior | 0.6134 | 0.5332 | 0.7666 | — | 0.0 |  |
| implicit:linker | 0.6200 | 0.5479 | 0.7813 | — | 0.0 |  |
| both:oracle | 0.5061 | 0.6511 | 0.8747 | — | 187.8 |  |
| noisy-implicit:0.1 | 0.6167 | 0.5258 | 0.7617 | — | 0.0 | corrupted_share 0.091 |
| noisy-implicit:0.3 | 0.6183 | 0.5430 | 0.7666 | — | 0.0 | corrupted_share 0.295 |
| noisy-implicit:0.5 | 0.6192 | 0.5405 | 0.7690 | — | 0.0 | corrupted_share 0.502 |
| explicit-vsa:oracle | 0.5102 | 0.6511 | 0.8870 | — | 189.1 | decode_recovered 0.888 |
| explicit-vsa:stated | 0.5389 | 0.3661 | 0.6388 | — | 35.8 | decode_recovered 1.000 |
| explicit-vsa:linker | 0.4947 | 0.2973 | 0.6388 | — | 67.9 | decode_recovered 0.304 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.1122 [-0.1605, -0.0614] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.1007 [+0.0344, +0.1671] (p 0.0030, Holm 0.0045) | 407 |
| explicit-sym:oracle − none | entailment | +0.1106 [+0.0639, +0.1572] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | property | -0.1163 [-0.1646, -0.0663] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.1229 [+0.0590, +0.1916] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1179 [+0.0713, +0.1671] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | property | -0.0663 [-0.1032, -0.0287] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | paraphrase | -0.1523 [-0.2088, -0.0934] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | entailment | -0.1007 [-0.1523, -0.0491] (p 0.0005, Holm 0.0045) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0041 [-0.0057, +0.0147] (p 0.4409) | 407 |
| implicit:oracle − none | paraphrase | -0.0221 [-0.0442, +0.0025] (p 0.0810) | 407 |
| implicit:oracle − none | entailment | -0.0074 [-0.0221, +0.0074] (p 0.3949) | 407 |
| implicit:linker − none | property | +0.0041 [-0.0057, +0.0139] (p 0.3709) | 407 |
| implicit:linker − none | paraphrase | +0.0049 [-0.0172, +0.0270] (p 0.7173) | 407 |
| implicit:linker − none | entailment | +0.0098 [-0.0025, +0.0221] (p 0.1465) | 407 |
| explicit-sym:linker − none | property | -0.0901 [-0.1302, -0.0483] (p 0.0005) | 407 |
| explicit-sym:linker − none | paraphrase | -0.1916 [-0.2482, -0.1351] (p 0.0005) | 407 |
| explicit-sym:linker − none | entailment | -0.0958 [-0.1450, -0.0442] (p 0.0005) | 407 |
| explicit-sym:typeprior − none | property | -0.0565 [-0.0934, -0.0197] (p 0.0040) | 407 |
| explicit-sym:typeprior − none | paraphrase | -0.1916 [-0.2531, -0.1327] (p 0.0005) | 407 |
| explicit-sym:typeprior − none | entailment | -0.0565 [-0.1081, -0.0098] (p 0.0240) | 407 |
| explicit-sym:stated − none | property | -0.0680 [-0.1048, -0.0303] (p 0.0010) | 407 |
| explicit-sym:stated − none | paraphrase | -0.1572 [-0.2162, -0.0983] (p 0.0005) | 407 |
| explicit-sym:stated − none | entailment | -0.1081 [-0.1597, -0.0589] (p 0.0005) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | property | +0.0066 [-0.0057, +0.0188] (p 0.2989) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | +0.0074 [-0.0197, +0.0369] (p 0.6798) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | +0.0049 [-0.0147, +0.0246] (p 0.6858) | 407 |
| explicit-vsa:oracle − none | property | -0.1057 [-0.1532, -0.0549] (p 0.0010) | 407 |
| explicit-vsa:oracle − none | paraphrase | +0.1081 [+0.0418, +0.1769] (p 0.0015) | 407 |
| explicit-vsa:oracle − none | entailment | +0.1155 [+0.0663, +0.1646] (p 0.0005) | 407 |
| explicit-vsa:linker − none | property | -0.1212 [-0.1622, -0.0811] (p 0.0005) | 407 |
| explicit-vsa:linker − none | paraphrase | -0.2457 [-0.3022, -0.1892] (p 0.0005) | 407 |
| explicit-vsa:linker − none | entailment | -0.1327 [-0.1843, -0.0835] (p 0.0005) | 407 |
| both:oracle − explicit-sym:oracle | property | +0.0025 [-0.0057, +0.0106] (p 0.5469) | 407 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0074 [-0.0123, +0.0270] (p 0.5534) | 407 |
| both:oracle − explicit-sym:oracle | entailment | -0.0074 [-0.0197, +0.0049] (p 0.3419) | 407 |
| both:oracle − implicit:oracle | property | -0.1138 [-0.1613, -0.0631] (p 0.0005) | 407 |
| both:oracle − implicit:oracle | paraphrase | +0.1302 [+0.0663, +0.1966] (p 0.0005) | 407 |
| both:oracle − implicit:oracle | entailment | +0.1106 [+0.0639, +0.1572] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | property | +0.2179 [+0.1720, +0.2645] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.0663 [-0.1351, +0.0000] (p 0.0585) | 407 |
| definition − explicit-sym:oracle | entailment | -0.0491 [-0.0958, -0.0025] (p 0.0475) | 407 |
| definition − none | property | +0.1057 [+0.0696, +0.1441] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.0344 [-0.0221, +0.0934] (p 0.2389) | 407 |
| definition − none | entailment | +0.0614 [+0.0172, +0.1081] (p 0.0100) | 407 |
| explicit-sym:linker − implicit:linker | property | -0.0942 [-0.1351, -0.0532] (p 0.0005) | 407 |
| explicit-sym:linker − implicit:linker | paraphrase | -0.1966 [-0.2531, -0.1400] (p 0.0005) | 407 |
| explicit-sym:linker − implicit:linker | entailment | -0.1057 [-0.1548, -0.0565] (p 0.0005) | 407 |
| explicit-sym:typeprior − implicit:typeprior | property | -0.0541 [-0.0909, -0.0180] (p 0.0055) | 407 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | -0.1818 [-0.2408, -0.1204] (p 0.0005) | 407 |
| explicit-sym:typeprior − implicit:typeprior | entailment | -0.0516 [-0.1007, -0.0025] (p 0.0425) | 407 |

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
- (4R)-limonene has part limonene. (1.00)
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

