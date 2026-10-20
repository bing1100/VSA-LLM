# E14 LG-B verdict format — pooled report (pre-registration §14)

## HuggingFaceTB/SmolLM2-135M

**Primary: exec-vsa-verdict − text (taxonomy, by depth; Holm)**

- depth 1: +0.010 [-0.055, +0.075] (n 200, Holm p 1.0000)
- depth 2: -0.035 [-0.095, +0.020] (n 200, Holm p 1.0000)
- depth 3: -0.055 [-0.125, +0.015] (n 200, Holm p 0.6523)
- depth 4: -0.030 [-0.095, +0.030] (n 200, Holm p 1.0000)
- depth 5: -0.035 [-0.100, +0.030] (n 200, Holm p 1.0000)
- exec-vsa-verdict − exec-sym-verdict: -0.004 [-0.022, +0.013] (n 1200, p 0.6898)

Format effect (taxonomy):

- exec-vsa-verdict − exec-vsa: -0.083 [-0.109, -0.058] (n 1200, p 0.0005)
- exec-sym-verdict − exec-sym: -0.052 [-0.077, -0.028] (n 1200, p 0.0005)
- exec-vsa-conclusion-verdict − exec-vsa-conclusion: -0.045 [-0.089, +0.000] (n 1200, p 0.0540)
- exec-vsa-verdict − exec-vsa, false items only: +0.110 [+0.080, +0.142] (n 600, p 0.0005)

Rulebase:

- exec-vsa-verdict − text: -0.028 [-0.067, +0.010] (n 600, p 0.1645)
- exec-vsa-verdict − exec-vsa: -0.093 [-0.128, -0.060] (n 600, p 0.0005)

| set · condition | true items | false items |
|---|---:|---:|
| taxonomy-verdict · exec-sym | 0.850 (600) | 0.230 (600) |
| taxonomy-verdict · exec-sym-verdict | 0.607 (600) | 0.370 (600) |
| taxonomy-verdict · exec-vsa | 0.818 (600) | 0.317 (600) |
| taxonomy-verdict · exec-vsa-conclusion | 0.743 (600) | 0.302 (600) |
| taxonomy-verdict · exec-vsa-conclusion-verdict | 0.053 (600) | 0.902 (600) |
| taxonomy-verdict · exec-vsa-verdict | 0.542 (600) | 0.427 (600) |
| taxonomy-verdict · none | 0.997 (600) | 0.000 (600) |
| taxonomy-verdict · text | 0.725 (600) | 0.257 (600) |
| rulebase-verdict · exec-vsa | 0.847 (300) | 0.303 (300) |
| rulebase-verdict · exec-vsa-verdict | 0.580 (300) | 0.383 (300) |
| rulebase-verdict · text | 0.730 (300) | 0.290 (300) |

## HuggingFaceTB/SmolLM2-360M

**Primary: exec-vsa-verdict − text (taxonomy, by depth; Holm)**

- depth 1: +0.045 [-0.050, +0.140] (n 200, Holm p 0.7238)
- depth 2: -0.050 [-0.155, +0.055] (n 200, Holm p 0.7238)
- depth 3: -0.095 [-0.195, +0.000] (n 200, Holm p 0.2899)
- depth 4: -0.060 [-0.155, +0.035] (n 200, Holm p 0.7198)
- depth 5: -0.075 [-0.170, +0.020] (n 200, Holm p 0.5059)
- exec-vsa-verdict − exec-sym-verdict: +0.024 [+0.003, +0.046] (n 1200, p 0.0225)

Format effect (taxonomy):

- exec-vsa-verdict − exec-vsa: -0.068 [-0.106, -0.029] (n 1200, p 0.0010)
- exec-sym-verdict − exec-sym: -0.043 [-0.083, -0.002] (n 1200, p 0.0415)
- exec-vsa-conclusion-verdict − exec-vsa-conclusion: -0.172 [-0.204, -0.140] (n 1200, p 0.0005)
- exec-vsa-verdict − exec-vsa, false items only: +0.392 [+0.352, +0.432] (n 600, p 0.0005)

Rulebase:

- exec-vsa-verdict − text: -0.010 [-0.065, +0.045] (n 600, p 0.7488)
- exec-vsa-verdict − exec-vsa: -0.057 [-0.110, -0.002] (n 600, p 0.0445)

| set · condition | true items | false items |
|---|---:|---:|
| taxonomy-verdict · exec-sym | 0.987 (600) | 0.050 (600) |
| taxonomy-verdict · exec-sym-verdict | 0.442 (600) | 0.510 (600) |
| taxonomy-verdict · exec-vsa | 0.987 (600) | 0.148 (600) |
| taxonomy-verdict · exec-vsa-conclusion | 0.540 (600) | 0.807 (600) |
| taxonomy-verdict · exec-vsa-conclusion-verdict | 0.003 (600) | 0.998 (600) |
| taxonomy-verdict · exec-vsa-verdict | 0.460 (600) | 0.540 (600) |
| taxonomy-verdict · none | 0.940 (600) | 0.060 (600) |
| taxonomy-verdict · text | 0.998 (600) | 0.013 (600) |
| rulebase-verdict · exec-vsa | 0.987 (300) | 0.107 (300) |
| rulebase-verdict · exec-vsa-verdict | 0.467 (300) | 0.513 (300) |
| rulebase-verdict · text | 0.977 (300) | 0.023 (300) |

## Qwen/Qwen3-0.6B-Base

**Primary: exec-vsa-verdict − text (taxonomy, by depth; Holm)**

- depth 1: +0.465 [+0.395, +0.530] (n 200, Holm p 0.0025)
- depth 2: +0.445 [+0.375, +0.515] (n 200, Holm p 0.0025)
- depth 3: +0.470 [+0.400, +0.540] (n 200, Holm p 0.0025)
- depth 4: +0.425 [+0.355, +0.495] (n 200, Holm p 0.0025)
- depth 5: +0.435 [+0.365, +0.505] (n 200, Holm p 0.0025)
- exec-vsa-verdict − exec-sym-verdict: +0.000 [+0.000, +0.000] (n 1200, p 1.0000)

Format effect (taxonomy):

- exec-vsa-verdict − exec-vsa: +0.000 [+0.000, +0.000] (n 1200, p 1.0000)
- exec-sym-verdict − exec-sym: +0.000 [+0.000, +0.000] (n 1200, p 1.0000)
- exec-vsa-conclusion-verdict − exec-vsa-conclusion: +0.058 [+0.044, +0.071] (n 1200, p 0.0005)
- exec-vsa-verdict − exec-vsa, false items only: +0.000 [+0.000, +0.000] (n 600, p 1.0000)

Rulebase:

- exec-vsa-verdict − text: +0.387 [+0.348, +0.425] (n 600, p 0.0005)
- exec-vsa-verdict − exec-vsa: +0.002 [+0.000, +0.005] (n 600, p 0.7298)

| set · condition | true items | false items |
|---|---:|---:|
| taxonomy-verdict · exec-sym | 1.000 (600) | 1.000 (600) |
| taxonomy-verdict · exec-sym-verdict | 1.000 (600) | 1.000 (600) |
| taxonomy-verdict · exec-vsa | 1.000 (600) | 1.000 (600) |
| taxonomy-verdict · exec-vsa-conclusion | 1.000 (600) | 0.885 (600) |
| taxonomy-verdict · exec-vsa-conclusion-verdict | 1.000 (600) | 1.000 (600) |
| taxonomy-verdict · exec-vsa-verdict | 1.000 (600) | 1.000 (600) |
| taxonomy-verdict · none | 1.000 (600) | 0.000 (600) |
| taxonomy-verdict · text | 0.998 (600) | 0.122 (600) |
| rulebase-verdict · exec-vsa | 1.000 (300) | 0.997 (300) |
| rulebase-verdict · exec-vsa-verdict | 1.000 (300) | 1.000 (300) |
| rulebase-verdict · text | 0.833 (300) | 0.393 (300) |

## Qwen/Qwen3-1.7B-Base

**Primary: exec-vsa-verdict − text (taxonomy, by depth; Holm)**

- depth 1: +0.140 [+0.095, +0.190] (n 200, Holm p 0.0025)
- depth 2: +0.360 [+0.295, +0.430] (n 200, Holm p 0.0025)
- depth 3: +0.360 [+0.290, +0.430] (n 200, Holm p 0.0025)
- depth 4: +0.400 [+0.330, +0.470] (n 200, Holm p 0.0025)
- depth 5: +0.410 [+0.340, +0.480] (n 200, Holm p 0.0025)
- exec-vsa-verdict − exec-sym-verdict: -0.022 [-0.030, -0.013] (n 1200, p 0.0005)

Format effect (taxonomy):

- exec-vsa-verdict − exec-vsa: -0.028 [-0.038, -0.018] (n 1200, p 0.0005)
- exec-sym-verdict − exec-sym: -0.008 [-0.014, -0.003] (n 1200, p 0.0010)
- exec-vsa-conclusion-verdict − exec-vsa-conclusion: +0.001 [+0.000, +0.003] (n 1200, p 0.7498)
- exec-vsa-verdict − exec-vsa, false items only: +0.003 [+0.000, +0.008] (n 600, p 0.2859)

Rulebase:

- exec-vsa-verdict − text: +0.305 [+0.267, +0.343] (n 600, p 0.0005)
- exec-vsa-verdict − exec-vsa: -0.030 [-0.045, -0.017] (n 600, p 0.0005)

| set · condition | true items | false items |
|---|---:|---:|
| taxonomy-verdict · exec-sym | 1.000 (600) | 1.000 (600) |
| taxonomy-verdict · exec-sym-verdict | 0.983 (600) | 1.000 (600) |
| taxonomy-verdict · exec-vsa | 1.000 (600) | 0.997 (600) |
| taxonomy-verdict · exec-vsa-conclusion | 1.000 (600) | 0.998 (600) |
| taxonomy-verdict · exec-vsa-conclusion-verdict | 1.000 (600) | 1.000 (600) |
| taxonomy-verdict · exec-vsa-verdict | 0.940 (600) | 1.000 (600) |
| taxonomy-verdict · none | 0.998 (600) | 0.000 (600) |
| taxonomy-verdict · text | 0.390 (600) | 0.997 (600) |
| rulebase-verdict · exec-vsa | 1.000 (300) | 1.000 (300) |
| rulebase-verdict · exec-vsa-verdict | 0.940 (300) | 1.000 (300) |
| rulebase-verdict · text | 0.337 (300) | 0.993 (300) |

## Gate G1′ (pre-registration §14)

- HuggingFaceTB/SmolLM2-135M: fail
- HuggingFaceTB/SmolLM2-360M: fail
