**SMOKE TEST — not a result.**

# E11 read-to-learn — t4 heldout — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Item set `experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1` (40 terms, 108 items); styles ['chebi'] (primary `chebi`); channel composes: True. No weight is updated except inside the gradient baseline, whose weights are restored after every term.

## Frames written by each reader

| style | reader | frames | empty | edges | precision | recall | F1 | stated recall | filler recall | relation acc. | relation acc. (ambiguous fillers) | cost / word |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chebi | linker | 40 | 4 | 2.35 | 0.074 | 0.013 | 0.023 | 0.146 | 0.104 | 0.190 | 0.184 (38) | forward_passes 33.2, forward_tokens 2558.7, seconds 0.8 |
| chebi | linker-all | 40 | 1 | 2.55 | 0.088 | 0.017 | 0.029 | 0.188 | 0.104 | 0.226 | 0.220 (41) | forward_passes 33.2, forward_tokens 2558.7, seconds 0.8 |
| chebi | none | 40 | 40 | 0.00 | — | 0.000 | — | 0.000 | 0.104 | — | — (0) | — |
| chebi | oracle | 40 | 0 | 13.07 | 1.000 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 1.000 (335) | — |
| chebi | pattern | 40 | 34 | 0.15 | 0.333 | 0.004 | 0.008 | 0.042 | 0.104 | 0.333 | 0.333 (6) | — |
| chebi | random | 40 | 0 | 12.22 | 0.274 | 0.256 | 0.265 | 0.000 | 0.104 | 0.988 | 0.846 (13) | — |
| chebi | stated | 40 | 6 | 1.20 | 1.000 | 0.092 | 0.168 | 1.000 | 0.104 | 1.000 | 1.000 (48) | — |
| chebi | typeprior | 40 | 1 | 2.60 | 0.356 | 0.071 | 0.118 | 0.771 | 0.104 | 0.849 | 0.841 (44) | — |

## Item tests (no definition in context unless the condition says `context`)

| condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| chebi|context | 0.6420 | 0.4630 | 0.8148 |
| chebi|context+linker | 0.6481 | 0.5185 | 0.8148 |
| chebi|context+oracle | 0.6667 | 0.5185 | 0.7593 |
| chebi|linker | 0.5926 | 0.5185 | 0.7037 |
| chebi|linker-all | 0.5926 | 0.5185 | 0.7037 |
| chebi|none | 0.5679 | 0.5000 | 0.7407 |
| chebi|oracle | 0.5864 | 0.5000 | 0.7037 |
| chebi|pattern | 0.5741 | 0.5185 | 0.7407 |
| chebi|random | 0.5802 | 0.4630 | 0.7222 |
| chebi|stated | 0.5926 | 0.4815 | 0.7222 |
| chebi|typeprior | 0.5926 | 0.5000 | 0.7037 |

| style | a − b | test | difference [95% CI] | n | p (Holm within pair) |
|---|---|---|---|---:|---:|
| chebi | linker − none | property | +0.0247 [-0.0062, +0.0741] | 54 | 0.6833 |
| chebi | linker − none | paraphrase | +0.0185 [+0.0000, +0.0556] | 54 | 0.8032 |
| chebi | linker − none | entailment | -0.0370 [-0.0926, +0.0000] | 54 | 0.6833 |
| chebi | oracle − none | property | +0.0185 [-0.0123, +0.0556] | 54 | 0.7732 |
| chebi | oracle − none | paraphrase | +0.0000 [-0.0926, +0.0926] | 54 | 1.0000 |
| chebi | oracle − none | entailment | -0.0370 [-0.0926, +0.0000] | 54 | 0.7732 |
| chebi | linker − oracle | property | +0.0062 [-0.0185, +0.0370] | 54 | 1.0000 |
| chebi | linker − oracle | paraphrase | +0.0185 [-0.0556, +0.1111] | 54 | 1.0000 |
| chebi | linker − oracle | entailment | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | linker − typeprior | property | +0.0000 [-0.0309, +0.0309] | 54 | 1.0000 |
| chebi | linker − typeprior | paraphrase | +0.0185 [-0.0741, +0.0931] | 54 | 1.0000 |
| chebi | linker − typeprior | entailment | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | linker − random | property | +0.0123 [-0.0309, +0.0556] | 54 | 1.0000 |
| chebi | linker − random | paraphrase | +0.0556 [-0.0185, +0.1296] | 54 | 0.7253 |
| chebi | linker − random | entailment | -0.0185 [-0.0556, +0.0000] | 54 | 1.0000 |
| chebi | linker − pattern | property | +0.0185 [-0.0185, +0.0679] | 54 | 0.9510 |
| chebi | linker − pattern | paraphrase | +0.0000 [-0.0556, +0.0556] | 54 | 1.0000 |
| chebi | linker − pattern | entailment | -0.0370 [-0.0926, +0.0000] | 54 | 0.7732 |
| chebi | linker − linker-all | property | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | linker − linker-all | paraphrase | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | linker − linker-all | entailment | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | stated − none | property | +0.0247 [+0.0000, +0.0617] | 54 | 0.3896 |
| chebi | stated − none | paraphrase | -0.0185 [-0.0926, +0.0370] | 54 | 1.0000 |
| chebi | stated − none | entailment | -0.0185 [-0.0741, +0.0370] | 54 | 1.0000 |
| chebi | context − linker | property | +0.0494 [-0.0494, +0.1481] | 54 | 0.6194 |
| chebi | context − linker | paraphrase | -0.0556 [-0.1852, +0.0741] | 54 | 0.6194 |
| chebi | context − linker | entailment | +0.1111 [-0.0370, +0.2593] | 54 | 0.5215 |
| chebi | context+linker − context | property | +0.0062 [-0.0370, +0.0432] | 54 | 1.0000 |
| chebi | context+linker − context | paraphrase | +0.0556 [-0.0370, +0.1481] | 54 | 1.0000 |
| chebi | context+linker − context | entailment | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | context − none | property | +0.0741 [-0.0248, +0.1790] | 54 | 0.4555 |
| chebi | context − none | paraphrase | -0.0370 [-0.1667, +0.0926] | 54 | 0.6913 |
| chebi | context − none | entailment | +0.0741 [-0.0741, +0.2222] | 54 | 0.6913 |

## Loss after the read terms in the evaluation text

48 windows, 64 occurrences (5 inside the term's own definition document, reported separately).

| condition | loss (other documents) | targets | loss (own document) | unlinked-token loss |
|---|---:|---:|---:|---:|
| chebi|oracle | 1.3458 | 470 | 1.4254 | 1.9827 |
| chebi|stated | 1.3533 | 470 | 1.4305 | 1.9826 |
| chebi|typeprior | 1.3478 | 470 | 1.4340 | 1.9828 |
| chebi|pattern | 1.3564 | 470 | 1.4312 | 1.9826 |
| chebi|linker | 1.3497 | 470 | 1.4404 | 1.9830 |
| chebi|linker-all | 1.3494 | 470 | 1.4356 | 1.9829 |
| chebi|random | 1.3588 | 470 | 1.4376 | 1.9826 |
| chebi|none | 1.3566 | 470 | 1.4245 | 1.9827 |

| condition − none | part | relative change [95% CI] | Δ nats/token | p |
|---|---|---|---:|---:|
| chebi|oracle | other | -0.80% [-1.01, -0.37] | -0.0108 | 0.0020 |
| chebi|oracle | own | +0.07% [-1.40, +0.67] | +0.0009 | 0.7615 |
| chebi|stated | other | -0.24% [-0.74, +0.63] | -0.0033 | 0.5994 |
| chebi|stated | own | +0.42% [-0.79, +0.92] | +0.0060 | 0.5894 |
| chebi|typeprior | other | -0.65% [-0.80, -0.29] | -0.0088 | 0.0100 |
| chebi|typeprior | own | +0.67% [+0.07, +0.92] | +0.0096 | 0.0023 |
| chebi|pattern | other | -0.02% [-0.16, +0.15] | -0.0002 | 0.8252 |
| chebi|pattern | own | +0.47% [+0.00, +0.67] | +0.0068 | 0.5252 |
| chebi|linker | other | -0.51% [-0.97, +0.16] | -0.0069 | 0.1479 |
| chebi|linker | own | +1.12% [+0.45, +1.40] | +0.0159 | 0.0023 |
| chebi|linker-all | other | -0.53% [-1.00, +0.14] | -0.0072 | 0.1299 |
| chebi|linker-all | own | +0.78% [+0.45, +0.92] | +0.0111 | 0.0023 |
| chebi|random | other | +0.16% [-0.32, +0.80] | +0.0022 | 0.5275 |
| chebi|random | own | +0.92% [+0.79, +0.97] | +0.0131 | 0.0023 |

Share of the oracle frame's loss gain recovered by the linker reader: 0.640.

## Locality

`{"applicable": true, "entries": 2000, "max_abs_row_change": 8.940696716308594e-08}`

Timings (s): `{"readers": 30.3, "persistence": 98.9, "context": 152.0, "windows": 22.7}`; total 314s.