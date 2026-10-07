**SMOKE TEST — not a result.**

# E11 read-to-learn — t4 heldout — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Item set `experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1` (40 terms, 108 items); styles ['chebi'] (primary `chebi`); channel composes: True. No weight is updated except inside the gradient baseline, whose weights are restored after every term.

## Frames written by each reader

| style | reader | frames | empty | edges | precision | recall | F1 | stated recall | filler recall | relation acc. | relation acc. (ambiguous fillers) | cost / word |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chebi | linker | 40 | 1 | 2.45 | 0.041 | 0.008 | 0.013 | 0.082 | 0.104 | 0.113 | 0.100 (40) | forward_passes 33.2, forward_tokens 2606.0, seconds 0.5 |
| chebi | linker-all | 40 | 1 | 2.48 | 0.040 | 0.008 | 0.013 | 0.082 | 0.104 | 0.113 | 0.100 (40) | forward_passes 33.2, forward_tokens 2606.0, seconds 0.5 |
| chebi | linker-joint | 40 | 7 | 1.43 | 0.053 | 0.006 | 0.010 | 0.061 | 0.104 | 0.139 | 0.130 (23) | forward_passes 36.6, forward_tokens 2851.3, seconds 0.5 |
| chebi | none | 40 | 40 | 0.00 | — | 0.000 | — | 0.000 | 0.104 | — | — (0) | — |
| chebi | oracle | 40 | 0 | 13.10 | 1.000 | 1.000 | 1.000 | 1.000 | 0.104 | 1.000 | 1.000 (337) | — |
| chebi | pattern | 40 | 35 | 0.12 | 0.200 | 0.002 | 0.004 | 0.020 | 0.104 | 0.200 | 0.200 (5) | — |
| chebi | random | 40 | 0 | 12.20 | 0.262 | 0.244 | 0.253 | 0.000 | 0.104 | 0.996 | 0.889 (9) | — |
| chebi | stated | 40 | 6 | 1.23 | 1.000 | 0.094 | 0.171 | 1.000 | 0.104 | 1.000 | 1.000 (49) | — |
| chebi | typeprior | 40 | 1 | 2.58 | 0.350 | 0.069 | 0.115 | 0.735 | 0.104 | 0.833 | 0.818 (44) | — |

## Item tests (no definition in context unless the condition says `context`)

| condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| chebi|context | 0.6605 | 0.4815 | 0.7778 |
| chebi|context+linker | 0.6852 | 0.5000 | 0.7963 |
| chebi|context+oracle | 0.6543 | 0.4444 | 0.7778 |
| chebi|linker | 0.6111 | 0.5370 | 0.7407 |
| chebi|linker-all | 0.6111 | 0.5370 | 0.7407 |
| chebi|linker-joint | 0.5926 | 0.4815 | 0.7593 |
| chebi|none | 0.6049 | 0.5185 | 0.7407 |
| chebi|oracle | 0.6049 | 0.5000 | 0.7222 |
| chebi|pattern | 0.6049 | 0.5185 | 0.7407 |
| chebi|random | 0.5988 | 0.4444 | 0.7222 |
| chebi|stated | 0.6111 | 0.5185 | 0.7407 |
| chebi|typeprior | 0.6173 | 0.4815 | 0.7037 |

| style | a − b | test | difference [95% CI] | n | p (Holm within pair) |
|---|---|---|---|---:|---:|
| chebi | linker − none | property | +0.0062 [-0.0309, +0.0556] | 54 | 1.0000 |
| chebi | linker − none | paraphrase | +0.0185 [-0.0370, +0.0741] | 54 | 1.0000 |
| chebi | linker − none | entailment | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | oracle − none | property | +0.0000 [-0.0309, +0.0370] | 54 | 1.0000 |
| chebi | oracle − none | paraphrase | -0.0185 [-0.1111, +0.0556] | 54 | 1.0000 |
| chebi | oracle − none | entailment | -0.0185 [-0.0556, +0.0000] | 54 | 1.0000 |
| chebi | linker − oracle | property | +0.0062 [-0.0247, +0.0370] | 54 | 1.0000 |
| chebi | linker − oracle | paraphrase | +0.0370 [-0.0370, +0.1116] | 54 | 1.0000 |
| chebi | linker − oracle | entailment | +0.0185 [+0.0000, +0.0556] | 54 | 1.0000 |
| chebi | linker − typeprior | property | -0.0062 [-0.0432, +0.0248] | 54 | 0.8152 |
| chebi | linker − typeprior | paraphrase | +0.0556 [-0.0185, +0.1296] | 54 | 0.8152 |
| chebi | linker − typeprior | entailment | +0.0370 [+0.0000, +0.0926] | 54 | 0.8152 |
| chebi | linker − random | property | +0.0123 [-0.0309, +0.0617] | 54 | 1.0000 |
| chebi | linker − random | paraphrase | +0.0926 [+0.0181, +0.1852] | 54 | 0.1558 |
| chebi | linker − random | entailment | +0.0185 [+0.0000, +0.0556] | 54 | 1.0000 |
| chebi | linker − pattern | property | +0.0062 [-0.0309, +0.0556] | 54 | 1.0000 |
| chebi | linker − pattern | paraphrase | +0.0185 [-0.0370, +0.0741] | 54 | 1.0000 |
| chebi | linker − pattern | entailment | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | linker − linker-all | property | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | linker − linker-all | paraphrase | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | linker − linker-all | entailment | +0.0000 [+0.0000, +0.0000] | 54 | 1.0000 |
| chebi | stated − none | property | +0.0062 [-0.0247, +0.0432] | 54 | 1.0000 |
| chebi | stated − none | paraphrase | +0.0000 [-0.0741, +0.0741] | 54 | 1.0000 |
| chebi | stated − none | entailment | +0.0000 [-0.0556, +0.0556] | 54 | 1.0000 |
| chebi | linker-joint − none | property | -0.0123 [-0.0309, +0.0000] | 54 | 0.7313 |
| chebi | linker-joint − none | paraphrase | -0.0370 [-0.0926, +0.0000] | 54 | 0.7313 |
| chebi | linker-joint − none | entailment | +0.0185 [+0.0000, +0.0556] | 54 | 0.7313 |
| chebi | linker-joint − typeprior | property | -0.0247 [-0.0679, +0.0062] | 54 | 0.3796 |
| chebi | linker-joint − typeprior | paraphrase | +0.0000 [-0.0556, +0.0741] | 54 | 1.0000 |
| chebi | linker-joint − typeprior | entailment | +0.0556 [+0.0000, +0.1296] | 54 | 0.3117 |
| chebi | linker-joint − oracle | property | -0.0123 [-0.0494, +0.0185] | 54 | 0.8511 |
| chebi | linker-joint − oracle | paraphrase | -0.0185 [-0.0926, +0.0556] | 54 | 0.8991 |
| chebi | linker-joint − oracle | entailment | +0.0370 [+0.0000, +0.0926] | 54 | 0.7552 |
| chebi | linker-joint − linker | property | -0.0185 [-0.0679, +0.0123] | 54 | 0.8032 |
| chebi | linker-joint − linker | paraphrase | -0.0556 [-0.1296, +0.0000] | 54 | 0.3057 |
| chebi | linker-joint − linker | entailment | +0.0185 [+0.0000, +0.0556] | 54 | 0.8032 |
| chebi | context − linker | property | +0.0494 [-0.0617, +0.1605] | 54 | 1.0000 |
| chebi | context − linker | paraphrase | -0.0556 [-0.2037, +0.0926] | 54 | 1.0000 |
| chebi | context − linker | entailment | +0.0370 [-0.0926, +0.1852] | 54 | 1.0000 |
| chebi | context+linker − context | property | +0.0247 [-0.0123, +0.0617] | 54 | 0.6174 |
| chebi | context+linker − context | paraphrase | +0.0185 [-0.0926, +0.1111] | 54 | 1.0000 |
| chebi | context+linker − context | entailment | +0.0185 [-0.0556, +0.1111] | 54 | 1.0000 |
| chebi | context − none | property | +0.0556 [-0.0617, +0.1605] | 54 | 1.0000 |
| chebi | context − none | paraphrase | -0.0370 [-0.1852, +0.1111] | 54 | 1.0000 |
| chebi | context − none | entailment | +0.0370 [-0.0926, +0.1852] | 54 | 1.0000 |

## Loss after the read terms in the evaluation text

48 windows, 64 occurrences (5 inside the term's own definition document, reported separately).

| condition | loss (other documents) | targets | loss (own document) | unlinked-token loss |
|---|---:|---:|---:|---:|
| chebi|oracle | 1.3443 | 470 | 1.4216 | 1.9827 |
| chebi|stated | 1.3520 | 470 | 1.4352 | 1.9826 |
| chebi|typeprior | 1.3469 | 470 | 1.4340 | 1.9827 |
| chebi|pattern | 1.3571 | 470 | 1.4294 | 1.9825 |
| chebi|linker | 1.3500 | 470 | 1.4356 | 1.9830 |
| chebi|linker-all | 1.3494 | 470 | 1.4356 | 1.9830 |
| chebi|linker-joint | 1.3518 | 470 | 1.4267 | 1.9828 |
| chebi|random | 1.3502 | 470 | 1.4453 | 1.9825 |
| chebi|none | 1.3566 | 470 | 1.4245 | 1.9827 |

| condition − none | part | relative change [95% CI] | Δ nats/token | p |
|---|---|---|---:|---:|
| chebi|oracle | other | -0.91% [-1.10, -0.51] | -0.0123 | 0.0040 |
| chebi|oracle | own | -0.20% [-1.40, +0.29] | -0.0029 | 0.8046 |
| chebi|stated | other | -0.34% [-0.84, +0.54] | -0.0046 | 0.4695 |
| chebi|stated | own | +0.76% [-0.79, +1.40] | +0.0108 | 0.4943 |
| chebi|typeprior | other | -0.72% [-0.89, -0.33] | -0.0097 | 0.0060 |
| chebi|typeprior | own | +0.67% [+0.07, +0.92] | +0.0096 | 0.0023 |
| chebi|pattern | other | +0.03% [-0.05, +0.26] | +0.0004 | 0.6074 |
| chebi|pattern | own | +0.34% [+0.00, +0.49] | +0.0049 | 0.4920 |
| chebi|linker | other | -0.49% [-0.92, +0.08] | -0.0066 | 0.1359 |
| chebi|linker | own | +0.78% [+0.45, +0.92] | +0.0111 | 0.0023 |
| chebi|linker-all | other | -0.53% [-0.99, +0.08] | -0.0072 | 0.1359 |
| chebi|linker-all | own | +0.78% [+0.45, +0.92] | +0.0111 | 0.0023 |
| chebi|linker-joint | other | -0.36% [-0.84, +0.25] | -0.0048 | 0.4675 |
| chebi|linker-joint | own | +0.15% [-1.70, +0.92] | +0.0022 | 0.7471 |
| chebi|random | other | -0.47% [-1.12, +0.67] | -0.0064 | 0.3796 |
| chebi|random | own | +1.46% [+0.79, +1.74] | +0.0209 | 0.0023 |

Share of the oracle frame's loss gain recovered by the linker reader: 0.540.

## Locality

`{"applicable": true, "entries": 2000, "max_abs_row_change": 8.940696716308594e-08}`

Timings (s): `{"readers": 41.6, "persistence": 92.2, "context": 78.0, "windows": 25.9}`; total 250s.