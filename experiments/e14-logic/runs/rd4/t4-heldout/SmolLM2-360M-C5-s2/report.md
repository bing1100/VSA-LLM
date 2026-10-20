# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s2`; style `chebi`; 341 terms, 814 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6151 | 0.5479 | 0.7592 | — | 0.0 |  |
| explicit-sym:oracle | 0.5094 | 0.6658 | 0.8747 | — | 187.8 |  |
| explicit-sym:stated | 0.5545 | 0.3563 | 0.6781 | — | 35.8 |  |
| explicit-sym:typeprior | 0.5651 | 0.3661 | 0.7101 | — | 71.6 |  |
| explicit-sym:linker | 0.5373 | 0.3440 | 0.6708 | — | 64.5 |  |
| definition | 0.7322 | 0.6020 | 0.8329 | — | 47.2 |  |
| noisy-explicit:0.1 | 0.4980 | 0.6560 | 0.8747 | — | 189.3 | corrupted_share 0.091 |
| noisy-explicit:0.3 | 0.4693 | 0.5995 | 0.8452 | — | 191.1 | corrupted_share 0.295 |
| noisy-explicit:0.5 | 0.4120 | 0.5946 | 0.8305 | — | 194.5 | corrupted_share 0.502 |
| implicit:oracle | 0.6183 | 0.5332 | 0.7666 | — | 0.0 |  |
| implicit:stated | 0.6208 | 0.5504 | 0.7592 | — | 0.0 |  |
| implicit:typeprior | 0.6159 | 0.5307 | 0.7715 | — | 0.0 |  |
| implicit:linker | 0.6257 | 0.5455 | 0.7740 | — | 0.0 |  |
| both:oracle | 0.5168 | 0.6781 | 0.8698 | — | 187.8 |  |
| noisy-implicit:0.1 | 0.6159 | 0.5356 | 0.7641 | — | 0.0 | corrupted_share 0.091 |
| noisy-implicit:0.3 | 0.6102 | 0.5430 | 0.7592 | — | 0.0 | corrupted_share 0.295 |
| noisy-implicit:0.5 | 0.6143 | 0.5405 | 0.7641 | — | 0.0 | corrupted_share 0.502 |
| explicit-vsa:oracle | 0.5143 | 0.6634 | 0.8747 | — | 189.8 | decode_recovered 0.903 |
| explicit-vsa:stated | 0.5471 | 0.3514 | 0.6536 | — | 35.8 | decode_recovered 1.000 |
| explicit-vsa:linker | 0.5111 | 0.3268 | 0.6314 | — | 69.8 | decode_recovered 0.310 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.1057 [-0.1523, -0.0557] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.1179 [+0.0541, +0.1843] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − none | entailment | +0.1155 [+0.0663, +0.1646] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | property | -0.1089 [-0.1564, -0.0581] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.1327 [+0.0688, +0.1966] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1081 [+0.0590, +0.1572] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | property | -0.0663 [-0.1048, -0.0270] (p 0.0015, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | paraphrase | -0.1941 [-0.2531, -0.1375] (p 0.0005, Holm 0.0045) | 407 |
| explicit-sym:stated − implicit:stated | entailment | -0.0811 [-0.1327, -0.0295] (p 0.0010, Holm 0.0045) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0033 [-0.0082, +0.0156] (p 0.6213) | 407 |
| implicit:oracle − none | paraphrase | -0.0147 [-0.0393, +0.0098] (p 0.2729) | 407 |
| implicit:oracle − none | entailment | +0.0074 [-0.0049, +0.0221] (p 0.3469) | 407 |
| implicit:linker − none | property | +0.0106 [-0.0008, +0.0221] (p 0.0650) | 407 |
| implicit:linker − none | paraphrase | -0.0025 [-0.0270, +0.0221] (p 0.9093) | 407 |
| implicit:linker − none | entailment | +0.0147 [+0.0000, +0.0319] (p 0.0795) | 407 |
| explicit-sym:linker − none | property | -0.0778 [-0.1171, -0.0385] (p 0.0005) | 407 |
| explicit-sym:linker − none | paraphrase | -0.2039 [-0.2629, -0.1449] (p 0.0005) | 407 |
| explicit-sym:linker − none | entailment | -0.0885 [-0.1450, -0.0344] (p 0.0005) | 407 |
| explicit-sym:typeprior − none | property | -0.0500 [-0.0885, -0.0115] (p 0.0115) | 407 |
| explicit-sym:typeprior − none | paraphrase | -0.1818 [-0.2457, -0.1179] (p 0.0005) | 407 |
| explicit-sym:typeprior − none | entailment | -0.0491 [-0.1007, +0.0000] (p 0.0560) | 407 |
| explicit-sym:stated − none | property | -0.0606 [-0.0991, -0.0213] (p 0.0035) | 407 |
| explicit-sym:stated − none | paraphrase | -0.1916 [-0.2506, -0.1327] (p 0.0005) | 407 |
| explicit-sym:stated − none | entailment | -0.0811 [-0.1302, -0.0319] (p 0.0010) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | property | +0.0049 [-0.0082, +0.0180] (p 0.4684) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | -0.0025 [-0.0344, +0.0270] (p 0.9383) | 407 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | +0.0000 [-0.0172, +0.0172] (p 1.0000) | 407 |
| explicit-vsa:oracle − none | property | -0.1007 [-0.1482, -0.0508] (p 0.0005) | 407 |
| explicit-vsa:oracle − none | paraphrase | +0.1155 [+0.0491, +0.1818] (p 0.0010) | 407 |
| explicit-vsa:oracle − none | entailment | +0.1155 [+0.0663, +0.1646] (p 0.0005) | 407 |
| explicit-vsa:linker − none | property | -0.1040 [-0.1441, -0.0631] (p 0.0005) | 407 |
| explicit-vsa:linker − none | paraphrase | -0.2211 [-0.2826, -0.1572] (p 0.0005) | 407 |
| explicit-vsa:linker − none | entailment | -0.1278 [-0.1843, -0.0737] (p 0.0005) | 407 |
| both:oracle − explicit-sym:oracle | property | +0.0074 [-0.0041, +0.0180] (p 0.1825) | 407 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0123 [-0.0098, +0.0344] (p 0.3534) | 407 |
| both:oracle − explicit-sym:oracle | entailment | -0.0049 [-0.0172, +0.0074] (p 0.5279) | 407 |
| both:oracle − implicit:oracle | property | -0.1016 [-0.1491, -0.0500] (p 0.0010) | 407 |
| both:oracle − implicit:oracle | paraphrase | +0.1450 [+0.0786, +0.2113] (p 0.0005) | 407 |
| both:oracle − implicit:oracle | entailment | +0.1032 [+0.0541, +0.1523] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | property | +0.2228 [+0.1761, +0.2695] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.0639 [-0.1302, +0.0000] (p 0.0570) | 407 |
| definition − explicit-sym:oracle | entailment | -0.0418 [-0.0909, +0.0074] (p 0.1075) | 407 |
| definition − none | property | +0.1171 [+0.0803, +0.1548] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.0541 [-0.0025, +0.1130] (p 0.0710) | 407 |
| definition − none | entailment | +0.0737 [+0.0270, +0.1229] (p 0.0050) | 407 |
| explicit-sym:linker − implicit:linker | property | -0.0885 [-0.1269, -0.0500] (p 0.0005) | 407 |
| explicit-sym:linker − implicit:linker | paraphrase | -0.2015 [-0.2629, -0.1400] (p 0.0005) | 407 |
| explicit-sym:linker − implicit:linker | entailment | -0.1032 [-0.1597, -0.0491] (p 0.0005) | 407 |
| explicit-sym:typeprior − implicit:typeprior | property | -0.0508 [-0.0885, -0.0131] (p 0.0100) | 407 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | -0.1646 [-0.2260, -0.1007] (p 0.0005) | 407 |
| explicit-sym:typeprior − implicit:typeprior | entailment | -0.0614 [-0.1130, -0.0123] (p 0.0175) | 407 |

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
- (4R)-limonene is the conjugate acid of limonene. (1.00)
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

