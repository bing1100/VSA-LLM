**SMOKE TEST — not a result.**

# E14 RD-3 — choosing the relation on stated edges (t5 new, style `prose`, host `Qwen/Qwen3-0.6B-Base`)

| reader | terms | edges | precision | recall (all gold) | stated recall | stated F1 | ambiguous accuracy (n) |
|---|---:|---:|---:|---:|---:|---:|---|
| typeprior | 3 | 8.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (6) |
| linker | 3 | 4.333 | 0.800 | 0.425 | 0.425 | 0.554 | 0.500 (6) |
| host | 3 | 5.000 | 1.000 | 0.653 | 0.653 | 0.782 | — (0) |
| mc-rel | 3 | 8.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (6) |
| mc-rel-lm | 3 | 8.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (6) |
| react | 3 | 8.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (6) |

| a − b | stated F1 difference [95% CI] | n |
|---|---|---:|
| react − linker | +0.4462 [+0.3333, +0.5385] (p 0.0198, Holm 0.0396) | 3 |
| react − typeprior | +0.0000 [+0.0000, +0.0000] (p 1.0000, Holm 1.0000) | 3 |
| mc-rel − linker | +0.4462 [+0.3333, +0.5385] (p 0.0198) | 3 |
| mc-rel − typeprior | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 3 |
| react − mc-rel | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 3 |
| mc-rel − mc-rel-lm | +0.0000 [+0.0000, +0.0000] (p 1.0000) | 3 |
| mc-rel-lm − linker | +0.4462 [+0.3333, +0.5385] (p 0.0198) | 3 |
| react − host | +0.2183 [+0.0909, +0.3333] (p 0.0198) | 3 |
