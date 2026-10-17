**SMOKE TEST — not a result.**

# E14 RD-4 / RD-7 — implicit against explicit recall of read frames (t4 heldout)

Run: `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1`; style `chebi`; 4 terms, 16 items; compose channel: True; store decoding: True.

## Means

| condition | property | paraphrase | entailment | statement | context tokens | note |
|---|---:|---:|---:|---:|---:|---|
| none | 0.4583 | 0.5000 | 0.8750 | — | 0.0 |  |
| explicit-sym:oracle | 0.3750 | 1.0000 | 1.0000 | — | 315.2 |  |
| explicit-sym:stated | 0.4167 | 0.3750 | 0.8750 | — | 46.5 |  |
| explicit-sym:typeprior | 0.4167 | 0.5000 | 0.6250 | — | 66.5 |  |
| explicit-sym:linker | 0.4167 | 0.7500 | 0.6250 | — | 63.5 |  |
| definition | 0.6250 | 0.3750 | 0.8750 | — | 49.2 |  |
| noisy-explicit:0.1 | 0.3750 | 1.0000 | 1.0000 | — | 324.8 | corrupted_share 0.111 |
| noisy-explicit:0.3 | 0.3750 | 1.0000 | 1.0000 | — | 324.0 | corrupted_share 0.256 |
| noisy-explicit:0.5 | 0.3333 | 0.8750 | 1.0000 | — | 326.0 | corrupted_share 0.447 |
| implicit:oracle | 0.4583 | 0.5000 | 0.8750 | — | 0.0 |  |
| implicit:stated | 0.4583 | 0.5000 | 0.8750 | — | 0.0 |  |
| implicit:typeprior | 0.4583 | 0.5000 | 0.8750 | — | 0.0 |  |
| implicit:linker | 0.4583 | 0.5000 | 0.8750 | — | 0.0 |  |
| both:oracle | 0.3750 | 1.0000 | 1.0000 | — | 315.2 |  |
| noisy-implicit:0.1 | 0.4583 | 0.5000 | 0.8750 | — | 0.0 | corrupted_share 0.111 |
| noisy-implicit:0.3 | 0.4583 | 0.5000 | 0.8750 | — | 0.0 | corrupted_share 0.256 |
| noisy-implicit:0.5 | 0.5000 | 0.6250 | 0.8750 | — | 0.0 | corrupted_share 0.447 |
| explicit-vsa:oracle | 0.3750 | 1.0000 | 1.0000 | — | 314.5 | decode_recovered 0.860 |
| explicit-vsa:stated | 0.4583 | 0.3750 | 1.0000 | — | 46.5 | decode_recovered 1.000 |
| explicit-vsa:linker | 0.2917 | 0.6250 | 0.5000 | — | 69.5 | decode_recovered 0.500 |

## Primary (Holm over contrasts × tests)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0833 [-0.4177, +0.2927] (p 0.8259, Holm 1.0000) | 8 |
| explicit-sym:oracle − none | paraphrase | +0.5000 [+0.1250, +0.8750] (p 0.0199, Holm 0.1791) | 8 |
| explicit-sym:oracle − none | entailment | +0.1250 [+0.0000, +0.3750] (p 0.7164, Holm 1.0000) | 8 |
| explicit-sym:oracle − implicit:oracle | property | -0.0833 [-0.4177, +0.2927] (p 0.8259, Holm 1.0000) | 8 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.5000 [+0.1250, +0.8750] (p 0.0199, Holm 0.1791) | 8 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1250 [+0.0000, +0.3750] (p 0.7164, Holm 1.0000) | 8 |
| explicit-sym:stated − implicit:stated | property | -0.0417 [-0.2510, +0.1260] (p 0.6567, Holm 1.0000) | 8 |
| explicit-sym:stated − implicit:stated | paraphrase | -0.1250 [-0.3750, +0.0000] (p 0.6169, Holm 1.0000) | 8 |
| explicit-sym:stated − implicit:stated | entailment | +0.0000 [-0.3750, +0.3750] (p 1.0000, Holm 1.0000) | 8 |

## Secondary (unadjusted)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| implicit:oracle − none | paraphrase | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| implicit:oracle − none | entailment | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| implicit:linker − none | property | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| implicit:linker − none | paraphrase | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| implicit:linker − none | entailment | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| explicit-sym:linker − none | property | -0.0417 [-0.2094, +0.1250] (p 0.8557) | 8 |
| explicit-sym:linker − none | paraphrase | +0.2500 [+0.0000, +0.6250] (p 0.2786) | 8 |
| explicit-sym:linker − none | entailment | -0.2500 [-0.6250, +0.2531] (p 0.4279) | 8 |
| explicit-sym:typeprior − none | property | -0.0417 [-0.2083, +0.0833] (p 0.5672) | 8 |
| explicit-sym:typeprior − none | paraphrase | +0.0000 [-0.3750, +0.2500] (p 1.0000) | 8 |
| explicit-sym:typeprior − none | entailment | -0.2500 [-0.5000, +0.0000] (p 0.1692) | 8 |
| explicit-sym:stated − none | property | -0.0417 [-0.2510, +0.1260] (p 0.6567) | 8 |
| explicit-sym:stated − none | paraphrase | -0.1250 [-0.3750, +0.0000] (p 0.6169) | 8 |
| explicit-sym:stated − none | entailment | +0.0000 [-0.3750, +0.3750] (p 1.0000) | 8 |
| explicit-vsa:oracle − explicit-sym:oracle | property | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| explicit-vsa:oracle − none | property | -0.0833 [-0.4177, +0.2927] (p 0.8259) | 8 |
| explicit-vsa:oracle − none | paraphrase | +0.5000 [+0.1250, +0.8750] (p 0.0199) | 8 |
| explicit-vsa:oracle − none | entailment | +0.1250 [+0.0000, +0.3750] (p 0.7164) | 8 |
| explicit-vsa:linker − none | property | -0.1667 [-0.4167, +0.0000] (p 0.1891) | 8 |
| explicit-vsa:linker − none | paraphrase | +0.1250 [-0.2500, +0.5000] (p 0.7761) | 8 |
| explicit-vsa:linker − none | entailment | -0.3750 [-0.7500, -0.1250] (p 0.0498) | 8 |
| both:oracle − explicit-sym:oracle | property | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| both:oracle − explicit-sym:oracle | entailment | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 8 |
| both:oracle − implicit:oracle | property | -0.0833 [-0.4177, +0.2927] (p 0.8259) | 8 |
| both:oracle − implicit:oracle | paraphrase | +0.5000 [+0.1250, +0.8750] (p 0.0199) | 8 |
| both:oracle − implicit:oracle | entailment | +0.1250 [+0.0000, +0.3750] (p 0.7164) | 8 |
| definition − explicit-sym:oracle | property | +0.2500 [-0.2083, +0.5844] (p 0.2985) | 8 |
| definition − explicit-sym:oracle | paraphrase | -0.6250 [-0.8781, -0.3750] (p 0.0100) | 8 |
| definition − explicit-sym:oracle | entailment | -0.1250 [-0.3750, +0.0000] (p 0.6169) | 8 |
| definition − none | property | +0.1667 [+0.0417, +0.3333] (p 0.0498) | 8 |
| definition − none | paraphrase | -0.1250 [-0.3750, +0.0000] (p 0.7164) | 8 |
| definition − none | entailment | +0.0000 [-0.3750, +0.3750] (p 1.0000) | 8 |
| explicit-sym:linker − implicit:linker | property | -0.0417 [-0.2094, +0.1250] (p 0.8557) | 8 |
| explicit-sym:linker − implicit:linker | paraphrase | +0.2500 [+0.0000, +0.6250] (p 0.2786) | 8 |
| explicit-sym:linker − implicit:linker | entailment | -0.2500 [-0.6250, +0.2531] (p 0.4279) | 8 |
| explicit-sym:typeprior − implicit:typeprior | property | -0.0417 [-0.2083, +0.0833] (p 0.5672) | 8 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | +0.0000 [-0.3750, +0.2500] (p 1.0000) | 8 |
| explicit-sym:typeprior − implicit:typeprior | entailment | -0.2500 [-0.5000, +0.0000] (p 0.1692) | 8 |

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

