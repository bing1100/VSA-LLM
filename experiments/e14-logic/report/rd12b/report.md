# E14 RD-12 — RD-12b, the cycle rerun with the `gentle` passive phase and no-training controls (pre-registration §18)

| arm | round | frames/term | gold recall | loss after read terms | general loss | explicit property | implicit property | selective property | none (plain null) | retracted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cycle | 0 | 0.88 | 0.155 | 1.7635 | 2.5482 | 0.5178 | 0.6142 | 0.6920 | 0.6108 | — |
| cycle | 1 | 1.31 | 0.176 | 1.7640 | 2.5477 | 0.5195 | 0.6108 | 0.6920 | 0.6142 | — |
| cycle | 2 | 1.32 | 0.176 | 1.7663 | 2.5473 | 0.5093 | 0.6193 | 0.6954 | 0.6159 | — |
| gold | 0 | 7.66 | 1.000 | 1.7646 | 2.5480 | 0.5567 | 0.6058 | 0.8714 | 0.6142 | — |
| gold | 1 | 7.66 | 1.000 | 1.7655 | 2.5477 | 0.5550 | 0.6108 | 0.8629 | 0.6176 | — |
| gold | 2 | 7.66 | 1.000 | 1.7691 | 2.5477 | 0.5601 | 0.6058 | 0.8663 | 0.6176 | — |
| passive | 0 | 0.00 | 0.000 | 1.7627 | 2.5482 | 0.5567 | 0.6074 | 0.6447 | 0.6074 | — |
| passive | 1 | 0.00 | 0.000 | 1.7643 | 2.5478 | 0.5651 | 0.6108 | 0.6464 | 0.6108 | — |
| passive | 2 | 0.00 | 0.000 | 1.7655 | 2.5475 | 0.5516 | 0.6125 | 0.6396 | 0.6125 | — |
| single | 0 | 0.88 | 0.155 | 1.7642 | 2.5481 | 0.5042 | 0.6091 | 0.6802 | 0.6142 | — |
| single | 1 | 0.88 | 0.155 | 1.7653 | 2.5477 | 0.5093 | 0.6159 | 0.6819 | 0.6108 | — |
| single | 2 | 0.88 | 0.155 | 1.7660 | 2.5476 | 0.5178 | 0.6159 | 0.6802 | 0.6142 | — |
| still-gold | 0 | 7.66 | 1.000 | 1.7654 | 2.5487 | 0.5448 | 0.6125 | 0.8731 | 0.6210 | — |
| still-none | 0 | 0.00 | 0.000 | 1.7641 | 2.5487 | 0.5584 | 0.6210 | 0.6447 | 0.6210 | — |

Loss after the read terms at the last round, a − b (paired over terms; negative = a better):

- cycle − passive: +0.0007 [-0.0022, +0.0038] (p 0.6253, n 112)
- cycle − single: +0.0002 [-0.0025, +0.0030] (p 0.8688, n 112)
- gold − passive: +0.0035 [-0.0014, +0.0085] (p 0.1590, n 112)
- single − passive: +0.0005 [-0.0026, +0.0037] (p 0.7688, n 112)
- passive − still-none: +0.0014 [-0.0034, +0.0060] (p 0.5529, n 112)
- gold − still-gold: +0.0036 [-0.0012, +0.0088] (p 0.1470, n 112)
- still-gold − still-none: +0.0013 [-0.0033, +0.0060] (p 0.6193, n 112)

**Recipe gate** (passive − still-none ≤ +0.5% of still-none's loss): drift +0.08% → passes.

Selective recall (each arm's frames, the asked relation's lines; context-free null) at the last round, a − b (paired over items; positive = a better):

- cycle − single, property: +0.0152 [+0.0034, +0.0271] (p 0.0060, n 197)
- cycle − single, paraphrase: +0.0203 [+0.0000, +0.0457] (p 0.1355, n 197)
- cycle − single, entailment: +0.0254 [+0.0051, +0.0508] (p 0.0100, n 197)
- cycle − passive, property: +0.0558 [+0.0237, +0.0897] (p 0.0010, n 197)
- cycle − passive, paraphrase: +0.1168 [+0.0609, +0.1726] (p 0.0005, n 197)
- cycle − passive, entailment: +0.0609 [+0.0254, +0.1015] (p 0.0030, n 197)
- gold − passive, property: +0.2267 [+0.1810, +0.2741] (p 0.0005, n 197)
- gold − passive, paraphrase: +0.2792 [+0.2081, +0.3503] (p 0.0005, n 197)
- gold − passive, entailment: +0.1371 [+0.0863, +0.1929] (p 0.0005, n 197)
- still-gold − still-none, property: +0.2284 [+0.1794, +0.2809] (p 0.0005, n 197)
- still-gold − still-none, paraphrase: +0.2386 [+0.1675, +0.3096] (p 0.0005, n 197)
- still-gold − still-none, entailment: +0.1168 [+0.0711, +0.1675] (p 0.0005, n 197)
