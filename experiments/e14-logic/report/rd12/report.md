# E14 RD-12 — interleaved passive ↔ active reading cycle (pilot: T4, SmolLM2-360M C5 s1)

| arm | round | frames/term | gold recall | loss after read terms | general loss | explicit property | implicit property | retracted |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cycle | 0 | 0.88 | 0.155 | 1.7881 | 2.5534 | 0.5533 | 0.6074 | — |
| cycle | 1 | 1.31 | 0.176 | 1.8001 | 2.5593 | 0.5448 | 0.6244 | — |
| cycle | 2 | 1.32 | 0.176 | 1.8182 | 2.5663 | 0.5516 | 0.6362 | — |
| gold | 0 | 7.66 | 1.000 | 1.7902 | 2.5538 | 0.5448 | 0.6007 | — |
| gold | 1 | 7.66 | 1.000 | 1.8072 | 2.5594 | 0.5381 | 0.6125 | — |
| gold | 2 | 7.66 | 1.000 | 1.8254 | 2.5663 | 0.5516 | 0.6261 | — |
| passive | 0 | 0.00 | 0.000 | 1.7832 | 2.5533 | 0.6125 | 0.6125 | — |
| passive | 1 | 0.00 | 0.000 | 1.7973 | 2.5595 | 0.6227 | 0.6261 | — |
| passive | 2 | 0.00 | 0.000 | 1.8150 | 2.5658 | 0.6311 | 0.6362 | — |
| single | 0 | 0.88 | 0.155 | 1.7871 | 2.5535 | 0.5584 | 0.6125 | — |
| single | 1 | 0.88 | 0.155 | 1.8023 | 2.5597 | 0.5601 | 0.6125 | — |
| single | 2 | 0.88 | 0.155 | 1.8195 | 2.5657 | 0.5635 | 0.6311 | — |

Loss after the read terms at the last round, a − b (paired over terms; negative = a better):

- cycle − passive: +0.0032 [-0.0028, +0.0095] (p 0.2999, n 112)
- cycle − single: -0.0013 [-0.0045, +0.0021] (p 0.4114, n 112)
- gold − passive: +0.0104 [+0.0007, +0.0205] (p 0.0365, n 112)
- single − passive: +0.0045 [-0.0022, +0.0120] (p 0.1945, n 112)
