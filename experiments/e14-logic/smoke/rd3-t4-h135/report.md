**SMOKE TEST — not a result.**

# E14 RD-3 — choosing the relation on stated edges (t4 heldout, style `chebi`, host `HuggingFaceTB/SmolLM2-135M`)

| reader | terms | edges | precision | recall (all gold) | stated recall | stated F1 | ambiguous accuracy (n) |
|---|---:|---:|---:|---:|---:|---:|---|
| typeprior | 12 | 3.417 | 0.391 | 0.107 | 0.803 | 0.529 | 0.842 (19) |
| linker | 12 | 3.167 | 0.058 | 0.022 | 0.136 | 0.087 | 0.111 (18) |
| host | 12 | 3.167 | 0.342 | 0.089 | 0.636 | 0.406 | 0.733 (15) |
| mc-rel | 12 | 3.417 | 0.232 | 0.074 | 0.485 | 0.320 | 0.632 (19) |
| mc-rel-lm | 12 | 3.417 | 0.114 | 0.037 | 0.227 | 0.155 | 0.263 (19) |
| react | 12 | 3.833 | 0.168 | 0.068 | 0.455 | 0.253 | 0.611 (18) |

| a − b | stated F1 difference [95% CI] | n |
|---|---|---:|
| react − linker | +0.1659 [+0.0303, +0.3001] (p 0.0299, Holm 0.0299) | 11 |
| react − typeprior | -0.2766 [-0.4536, -0.1185] (p 0.0100, Holm 0.0199) | 11 |
| mc-rel − linker | +0.2335 [+0.0924, +0.3923] (p 0.0100) | 11 |
| mc-rel − typeprior | -0.2091 [-0.3830, -0.0718] (p 0.0398) | 11 |
| react − mc-rel | -0.0675 [-0.1455, -0.0064] (p 0.0299) | 11 |
| mc-rel − mc-rel-lm | +0.1648 [+0.0519, +0.2906] (p 0.0100) | 11 |
| mc-rel-lm − linker | +0.0687 [-0.0260, +0.1658] (p 0.2388) | 11 |
| react − host | -0.1538 [-0.3049, -0.0141] (p 0.0498) | 11 |
