**SMOKE TEST — not a result.**

# E11 read-to-learn — t4 heldout — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Item set `/tmp/claude-1000/-home-bhux-workplace-VSA-LLM/95cd3acd-b338-46c8-8df8-f781d6fdf776/scratchpad/t4h-noitems` (8 terms, 0 items); styles ['chebi'] (primary `chebi`); channel composes: True. No weight is updated except inside the gradient baseline, whose weights are restored after every term.

## Frames written by each reader

| style | reader | frames | empty | edges | precision | recall | F1 | stated recall | filler recall | relation acc. | relation acc. (ambiguous fillers) | cost / word |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chebi | linker | 8 | 0 | 2.75 | 0.091 | 0.018 | 0.030 | 0.154 | 0.113 | 0.200 | 0.222 (9) | forward_passes 34.8, forward_tokens 3512.8, seconds 8.6 |
| chebi | linker-random | 8 | 0 | 2.75 | 0.000 | 0.000 | 0.000 | 0.000 | 0.113 | — | — (0) | — |
| chebi | none | 8 | 8 | 0.00 | — | 0.000 | — | 0.000 | 0.113 | — | — (0) | — |
| chebi | oracle | 8 | 0 | 13.88 | 1.000 | 1.000 | 1.000 | 1.000 | 0.113 | 1.000 | 1.000 (75) | — |
| chebi | random | 8 | 0 | 13.38 | 0.290 | 0.279 | 0.284 | 0.000 | 0.113 | 0.979 | 0.750 (4) | — |
| chebi | typeprior | 8 | 0 | 3.00 | 0.417 | 0.090 | 0.148 | 0.769 | 0.113 | 0.917 | 0.909 (11) | — |

## Loss after the read terms in the evaluation text

5 windows, 7 occurrences (0 inside the term's own definition document, reported separately).

| condition | loss (other documents) | targets | loss (own document) | unlinked-token loss |
|---|---:|---:|---:|---:|
| chebi|oracle | 1.9605 | 56 | — | 2.1220 |
| chebi|typeprior | 1.9482 | 56 | — | 2.1221 |
| chebi|linker | 1.9835 | 56 | — | 2.1218 |
| chebi|random | 1.9813 | 56 | — | 2.1222 |
| chebi|linker-random | 1.9585 | 56 | — | 2.1220 |
| chebi|none | 1.9605 | 56 | — | 2.1218 |
| chebi|context | 1.9188 | 56 | — | — |
| chebi|context+linker | 1.9429 | 56 | — | — |

| condition − reference | part | relative change [95% CI] | Δ nats/token | p |
|---|---|---|---:|---:|
| chebi|oracle − none | other | -0.00% [-0.21, +1.13] | -0.0000 | 0.8410 |
| chebi|typeprior − none | other | -0.63% [-1.59, +0.89] | -0.0123 | 0.3487 |
| chebi|linker − none | other | +1.17% [+0.02, +2.57] | +0.0230 | 0.0103 |
| chebi|random − none | other | +1.06% [-0.24, +2.65] | +0.0208 | 0.4000 |
| chebi|linker-random − none | other | -0.10% [-0.31, +0.13] | -0.0020 | 0.5846 |
| chebi|context − none | other | -2.13% [-4.68, -0.05] | -0.0417 | 0.0103 |
| chebi|context+linker − none | other | -0.90% [-1.97, -0.03] | -0.0176 | 0.0103 |
| chebi|linker − random | other | +0.11% [-0.08, +0.26] | +0.0021 | 0.4205 |
| chebi|linker − linker-random | other | +1.27% [-0.01, +2.89] | +0.0249 | 0.1641 |
| chebi|context − linker | other | -3.26% [-7.07, -0.07] | -0.0647 | 0.0103 |
| chebi|context+linker − context | other | +1.26% [+0.02, +2.85] | +0.0241 | 0.0103 |

Share of the oracle frame's loss gain recovered by the linker reader: -1211.801.

In context (§16.3): 98.8 definition tokens per window (1.00 definitions; budget 1024), 70.6 per occurrence; 1.000 of the occurrences had their own definition in context.

## Locality

`{"applicable": true, "entries": 2000, "max_abs_row_change": 0.0}`

Timings (s): `{"readers": 69.1, "windows": 113.6}`; total 191s.