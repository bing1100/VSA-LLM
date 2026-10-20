# E14 RD-3 — pooled report (Holm per track over hosts × {react − linker, react − typeprior})

## t4 heldout — HuggingFaceTB/SmolLM2-360M

| reader | stated F1 | precision | stated recall | ambiguous accuracy |
|---|---:|---:|---:|---:|
| typeprior | 0.573 | 0.367 | 0.841 | 0.845 |
| linker | 0.101 | 0.064 | 0.151 | 0.165 |
| host | 0.461 | 0.350 | 0.652 | 0.755 |
| teacher | 0.603 | 0.672 | 0.660 | 0.964 |
| mc-rel | 0.446 | 0.284 | 0.664 | 0.702 |
| mc-rel-lm | 0.253 | 0.163 | 0.371 | 0.369 |
| react | 0.337 | 0.194 | 0.663 | 0.715 |

## t4 heldout — Qwen/Qwen3-1.7B-Base

| reader | stated F1 | precision | stated recall | ambiguous accuracy |
|---|---:|---:|---:|---:|
| typeprior | 0.573 | 0.367 | 0.841 | 0.845 |
| linker | 0.101 | 0.064 | 0.151 | 0.165 |
| host | 0.461 | 0.350 | 0.652 | 0.755 |
| teacher | 0.603 | 0.672 | 0.660 | 0.964 |
| mc-rel | 0.415 | 0.263 | 0.625 | 0.685 |
| mc-rel-lm | 0.170 | 0.111 | 0.239 | 0.247 |
| react | 0.320 | 0.192 | 0.613 | 0.684 |

## t5 new — HuggingFaceTB/SmolLM2-360M

| reader | stated F1 | precision | stated recall | ambiguous accuracy |
|---|---:|---:|---:|---:|
| typeprior | 0.916 | 0.918 | 0.915 | 0.700 |
| linker | 0.518 | 0.727 | 0.433 | 0.328 |
| host | 0.758 | 0.947 | 0.645 | 0.176 |
| mc-rel | 0.915 | 0.915 | 0.915 | 0.676 |
| mc-rel-lm | 0.885 | 0.885 | 0.885 | 0.557 |
| react | 0.917 | 0.928 | 0.908 | 0.716 |

## t5 new — Qwen/Qwen3-1.7B-Base

| reader | stated F1 | precision | stated recall | ambiguous accuracy |
|---|---:|---:|---:|---:|
| typeprior | 0.916 | 0.918 | 0.915 | 0.700 |
| linker | 0.518 | 0.727 | 0.433 | 0.328 |
| host | 0.758 | 0.947 | 0.645 | 0.176 |
| mc-rel | 0.937 | 0.937 | 0.937 | 0.776 |
| mc-rel-lm | 0.916 | 0.916 | 0.916 | 0.694 |
| react | 0.936 | 0.941 | 0.932 | 0.793 |

## Primary

| track | host | a − b | stated F1 difference [95% CI] | p (Holm, family) | n |
|---|---|---|---|---:|---:|
| t4 heldout | HuggingFaceTB/SmolLM2-360M | react − linker | +0.2362 [+0.1933, +0.2793] (p 0.0005, Holm 0.0010) | 0.0020 | 248 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M | react − typeprior | -0.2361 [-0.2766, -0.1959] (p 0.0005, Holm 0.0010) | 0.0020 | 248 |
| t4 heldout | Qwen/Qwen3-1.7B-Base | react − linker | +0.2186 [+0.1762, +0.2608] (p 0.0005, Holm 0.0010) | 0.0020 | 248 |
| t4 heldout | Qwen/Qwen3-1.7B-Base | react − typeprior | -0.2536 [-0.2976, -0.2105] (p 0.0005, Holm 0.0010) | 0.0020 | 248 |
| t5 new | HuggingFaceTB/SmolLM2-360M | react − linker | +0.3987 [+0.3696, +0.4288] (p 0.0005, Holm 0.0010) | 0.0020 | 300 |
| t5 new | HuggingFaceTB/SmolLM2-360M | react − typeprior | +0.0006 [-0.0114, +0.0130] (p 0.9158, Holm 0.9158) | 0.9158 | 300 |
| t5 new | Qwen/Qwen3-1.7B-Base | react − linker | +0.4180 [+0.3874, +0.4479] (p 0.0005, Holm 0.0010) | 0.0020 | 300 |
| t5 new | Qwen/Qwen3-1.7B-Base | react − typeprior | +0.0198 [+0.0070, +0.0324] (p 0.0030, Holm 0.0030) | 0.0060 | 300 |
