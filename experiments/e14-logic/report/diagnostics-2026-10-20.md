# E14 tier P — exploratory diagnostics (2026-10-20)

Exploratory diagnostics of E14 tier P (2026-10-20; not pre-registered, read after the primaries).

## (a) LG-B true/false accuracy by gold label (exec conditions state the conclusion)

| host | set | condition | label True | label False |
|---|---|---|---:|---:|
| SmolLM2-135M | taxonomy | text | 0.718 (600) | 0.263 (600) |
| SmolLM2-135M | taxonomy | exec-sym | 0.843 (600) | 0.228 (600) |
| SmolLM2-135M | taxonomy | exec-vsa | 0.820 (600) | 0.317 (600) |
| SmolLM2-135M | taxonomy | exec-vsa-conclusion | 0.743 (600) | 0.302 (600) |
| SmolLM2-135M | prop-rp | text | 0.983 (300) | 0.010 (300) |
| SmolLM2-135M | prop-rp | exec-sym | 0.957 (300) | 0.180 (300) |
| SmolLM2-135M | prop-rp | exec-vsa | 0.917 (300) | 0.273 (300) |
| SmolLM2-360M | taxonomy | text | 0.998 (600) | 0.015 (600) |
| SmolLM2-360M | taxonomy | exec-sym | 0.987 (600) | 0.053 (600) |
| SmolLM2-360M | taxonomy | exec-vsa | 0.987 (600) | 0.152 (600) |
| SmolLM2-360M | taxonomy | exec-vsa-conclusion | 0.543 (600) | 0.808 (600) |
| SmolLM2-360M | prop-rp | text | 0.987 (300) | 0.013 (300) |
| SmolLM2-360M | prop-rp | exec-sym | 0.983 (300) | 0.877 (300) |
| SmolLM2-360M | prop-rp | exec-vsa | 0.970 (300) | 0.937 (300) |
| Qwen3-0.6B | taxonomy | text | 0.998 (600) | 0.128 (600) |
| Qwen3-0.6B | taxonomy | exec-sym | 1.000 (600) | 1.000 (600) |
| Qwen3-0.6B | taxonomy | exec-vsa | 1.000 (600) | 1.000 (600) |
| Qwen3-0.6B | taxonomy | exec-vsa-conclusion | 1.000 (600) | 0.887 (600) |
| Qwen3-0.6B | prop-rp | text | 0.950 (300) | 0.060 (300) |
| Qwen3-0.6B | prop-rp | exec-sym | 1.000 (300) | 0.853 (300) |
| Qwen3-0.6B | prop-rp | exec-vsa | 1.000 (300) | 0.863 (300) |
| Qwen3-1.7B | taxonomy | text | 0.390 (600) | 0.997 (600) |
| Qwen3-1.7B | taxonomy | exec-sym | 1.000 (600) | 1.000 (600) |
| Qwen3-1.7B | taxonomy | exec-vsa | 1.000 (600) | 0.997 (600) |
| Qwen3-1.7B | taxonomy | exec-vsa-conclusion | 1.000 (600) | 0.998 (600) |
| Qwen3-1.7B | prop-rp | text | 0.443 (300) | 0.580 (300) |
| Qwen3-1.7B | prop-rp | exec-sym | 1.000 (300) | 1.000 (300) |
| Qwen3-1.7B | prop-rp | exec-vsa | 1.000 (300) | 1.000 (300) |

## (b) RD-4 property items, C5 s1–3 pooled: PMI-correct / raw-correct

**t4-heldout**

| condition | ALL | has_functional_parent | has_role | is_a |
|---|---:|---:|---:|---:|
| none | 0.614 / 0.733 | 0.886 / 0.866 | 0.318 / 0.738 | 0.732 / 0.708 |
| implicit:oracle | 0.616 / 0.732 | 0.889 / 0.872 | 0.330 / 0.738 | 0.728 / 0.706 |
| explicit-sym:oracle | 0.511 / 0.696 | 0.382 / 0.513 | 0.326 / 0.626 | 0.634 / 0.764 |
| explicit-sym:typeprior | 0.561 / 0.831 | 0.624 / 0.889 | 0.377 / 0.759 | 0.651 / 0.860 |
| explicit-sym:linker | 0.532 / 0.831 | 0.630 / 0.895 | 0.375 / 0.769 | 0.601 / 0.855 |
| noisy-explicit:0.3 | 0.464 / 0.657 | 0.308 / 0.496 | 0.295 / 0.558 | 0.582 / 0.738 |
| definition | 0.729 / 0.868 | 0.877 / 0.923 | 0.548 / 0.802 | 0.803 / 0.895 |

**t5-new**

| condition | ALL | all |
|---|---:|---:|
| none | 0.190 / 0.225 | 0.190 / 0.225 |
| implicit:oracle | 0.260 / 0.284 | 0.260 / 0.284 |
| explicit-sym:oracle | 0.502 / 0.756 | 0.502 / 0.756 |
| explicit-sym:typeprior | 0.513 / 0.762 | 0.513 / 0.762 |
| explicit-sym:linker | 0.359 / 0.576 | 0.359 / 0.576 |
| noisy-explicit:0.3 | 0.408 / 0.589 | 0.408 / 0.589 |
| definition | 0.642 / 0.927 | 0.642 / 0.927 |

## (c) T4 property items: does the oracle recall state the answer?

| relation | items | gold among the frame's partners | a distractor among them |
|---|---:|---:|---:|
| has_functional_parent | 39 | 35 | 0 |
| has_role | 130 | 128 | 0 |
| is_a | 238 | 236 | 0 |
