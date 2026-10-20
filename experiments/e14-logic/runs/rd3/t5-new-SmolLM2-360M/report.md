# E14 RD-3 — choosing the relation on stated edges (t5 new, style `prose`, host `HuggingFaceTB/SmolLM2-360M`)

| reader | terms | edges | precision | recall (all gold) | stated recall | stated F1 | ambiguous accuracy (n) |
|---|---:|---:|---:|---:|---:|---:|---|
| typeprior | 300 | 7.340 | 0.918 | 0.915 | 0.915 | 0.916 | 0.700 (617) |
| linker | 300 | 4.097 | 0.727 | 0.433 | 0.433 | 0.518 | 0.328 (466) |
| host | 300 | 4.787 | 0.947 | 0.645 | 0.645 | 0.758 | 0.176 (17) |
| mc-rel | 300 | 7.383 | 0.915 | 0.915 | 0.915 | 0.915 | 0.676 (630) |
| mc-rel-lm | 300 | 7.383 | 0.885 | 0.885 | 0.885 | 0.885 | 0.557 (630) |
| react | 300 | 7.183 | 0.928 | 0.908 | 0.908 | 0.917 | 0.716 (573) |

| a − b | stated F1 difference [95% CI] | n |
|---|---|---:|
| react − linker | +0.3987 [+0.3696, +0.4288] (p 0.0005, Holm 0.0010) | 300 |
| react − typeprior | +0.0006 [-0.0114, +0.0130] (p 0.9158, Holm 0.9158) | 300 |
| mc-rel − linker | +0.3968 [+0.3678, +0.4267] (p 0.0005) | 300 |
| mc-rel − typeprior | -0.0013 [-0.0131, +0.0108] (p 0.8463) | 300 |
| react − mc-rel | +0.0019 [-0.0009, +0.0046] (p 0.1875) | 300 |
| mc-rel − mc-rel-lm | +0.0296 [+0.0231, +0.0367] (p 0.0005) | 300 |
| mc-rel-lm − linker | +0.3672 [+0.3382, +0.3965] (p 0.0005) | 300 |
| react − host | +0.1588 [+0.1455, +0.1725] (p 0.0005) | 300 |
