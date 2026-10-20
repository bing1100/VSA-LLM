# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C0p-s1`; style `chebi`; 341 terms, 814 items; compose channel: False; store decoding: False.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.6175 | 0.5405 | 0.7666 | — | 0.0 |  |
| explicit-sym:oracle | 0.4930 | 0.6339 | 0.8649 | — | 187.8 |  |
| explicit-sym:stated | 0.5610 | 0.3489 | 0.6536 | — | 35.8 |  |
| explicit-sym:typeprior | 0.5618 | 0.3686 | 0.6904 | — | 71.6 |  |
| explicit-sym:linker | 0.5291 | 0.3808 | 0.6708 | — | 64.2 |  |
| definition | 0.7248 | 0.5946 | 0.8280 | — | 47.2 |  |
| noisy-explicit:0.1 | 0.4816 | 0.6290 | 0.8673 | — | 189.3 | corrupted_share 0.091 |
| noisy-explicit:0.3 | 0.4505 | 0.5971 | 0.8477 | — | 191.1 | corrupted_share 0.295 |
| noisy-explicit:0.5 | 0.4103 | 0.6020 | 0.8329 | — | 194.5 | corrupted_share 0.502 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.1245 [-0.1720, -0.0745] (p 0.0005, Holm 0.0015) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.0934 [+0.0246, +0.1622] (p 0.0090, Holm 0.0090) | 407 |
| explicit-sym:oracle − none | entailment | +0.0983 [+0.0516, +0.1474] (p 0.0005, Holm 0.0015) | 407 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:linker − none | property | -0.0885 [-0.1278, -0.0483] (p 0.0005) | 407 |
| explicit-sym:linker − none | paraphrase | -0.1597 [-0.2187, -0.0958] (p 0.0005) | 407 |
| explicit-sym:linker − none | entailment | -0.0958 [-0.1474, -0.0467] (p 0.0015) | 407 |
| explicit-sym:typeprior − none | property | -0.0557 [-0.0934, -0.0172] (p 0.0050) | 407 |
| explicit-sym:typeprior − none | paraphrase | -0.1720 [-0.2310, -0.1106] (p 0.0005) | 407 |
| explicit-sym:typeprior − none | entailment | -0.0762 [-0.1302, -0.0270] (p 0.0030) | 407 |
| explicit-sym:stated − none | property | -0.0565 [-0.0942, -0.0180] (p 0.0060) | 407 |
| explicit-sym:stated − none | paraphrase | -0.1916 [-0.2482, -0.1351] (p 0.0005) | 407 |
| explicit-sym:stated − none | entailment | -0.1130 [-0.1671, -0.0614] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | property | +0.2318 [+0.1867, +0.2776] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.0393 [-0.1057, +0.0270] (p 0.2584) | 407 |
| definition − explicit-sym:oracle | entailment | -0.0369 [-0.0860, +0.0123] (p 0.1615) | 407 |
| definition − none | property | +0.1073 [+0.0721, +0.1450] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.0541 [-0.0025, +0.1130] (p 0.0670) | 407 |
| definition − none | entailment | +0.0614 [+0.0147, +0.1081] (p 0.0125) | 407 |

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

