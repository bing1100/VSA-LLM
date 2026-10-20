# E14 LG-P — pooled report

## HuggingFaceTB/SmolLM2-135M

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: +0.064 [-0.006, +0.135] (Holm p 0.1770, n 156)
- budget 30: +0.103 [+0.038, +0.173] (Holm p 0.0075, n 156)
- budget 100: +0.006 [-0.051, +0.064] (Holm p 0.8973, n 156)

Secondary (found on provable problems):

- score − sos, budget 10: -0.019 [-0.077, +0.045] (p 0.6198, n 156)
- score − sos, budget 30: -0.026 [-0.071, +0.019] (p 0.3619, n 156)
- score − sos, budget 100: -0.237 [-0.301, -0.173] (p 0.0005, n 156)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.186 | 0.186 | 1.000 |
| bfs|30 | 0.654 | 0.654 | 1.000 |
| bfs|100 | 0.769 | 0.769 | 1.000 |
| ceiling|100 | 1.000 | 1.000 | 1.000 |
| random|10 | 0.115 | 0.115 | 1.000 |
| random|30 | 0.513 | 0.513 | 1.000 |
| random|100 | 0.750 | 0.750 | 1.000 |
| score|10 | 0.179 | 0.179 | 1.000 |
| score|30 | 0.615 | 0.615 | 1.000 |
| score|100 | 0.756 | 0.756 | 1.000 |
| sos|10 | 0.199 | 0.199 | 1.000 |
| sos|30 | 0.641 | 0.641 | 1.000 |
| sos|100 | 0.994 | 0.994 | 1.000 |

## HuggingFaceTB/SmolLM2-360M

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: +0.019 [-0.051, +0.083] (Holm p 0.6303, n 156)
- budget 30: +0.090 [+0.019, +0.160] (Holm p 0.0285, n 156)
- budget 100: -0.058 [-0.103, -0.013] (Holm p 0.0430, n 156)

Secondary (found on provable problems):

- score − sos, budget 10: -0.064 [-0.122, -0.006] (p 0.0390, n 156)
- score − sos, budget 30: -0.038 [-0.090, +0.013] (p 0.2064, n 156)
- score − sos, budget 100: -0.301 [-0.372, -0.231] (p 0.0005, n 156)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.186 | 0.186 | 1.000 |
| bfs|30 | 0.654 | 0.654 | 1.000 |
| bfs|100 | 0.769 | 0.769 | 1.000 |
| ceiling|100 | 1.000 | 1.000 | 1.000 |
| random|10 | 0.115 | 0.115 | 1.000 |
| random|30 | 0.513 | 0.513 | 1.000 |
| random|100 | 0.750 | 0.750 | 1.000 |
| score|10 | 0.135 | 0.135 | 1.000 |
| score|30 | 0.603 | 0.603 | 1.000 |
| score|100 | 0.692 | 0.692 | 1.000 |
| sos|10 | 0.199 | 0.199 | 1.000 |
| sos|30 | 0.641 | 0.641 | 1.000 |
| sos|100 | 0.994 | 0.994 | 1.000 |

## Qwen/Qwen3-0.6B-Base

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: +0.032 [-0.038, +0.096] (Holm p 0.8208, n 156)
- budget 30: +0.096 [+0.032, +0.160] (Holm p 0.0060, n 156)
- budget 100: +0.006 [-0.051, +0.064] (Holm p 0.9153, n 156)

Secondary (found on provable problems):

- score − sos, budget 10: -0.051 [-0.115, +0.013] (p 0.1330, n 156)
- score − sos, budget 30: -0.032 [-0.083, +0.019] (p 0.2889, n 156)
- score − sos, budget 100: -0.237 [-0.301, -0.173] (p 0.0005, n 156)
- react − random, budget 10: -0.013 [-0.083, +0.058] (p 0.7713, n 156)
- react − random, budget 30: +0.038 [-0.032, +0.109] (p 0.3099, n 156)
- react − random, budget 100: -0.032 [-0.090, +0.026] (p 0.3289, n 156)
- react − score, budget 10: -0.045 [-0.103, +0.013] (p 0.1565, n 156)
- react − score, budget 30: -0.058 [-0.109, -0.006] (p 0.0350, n 156)
- react − score, budget 100: -0.038 [-0.090, +0.013] (p 0.1905, n 156)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.186 | 0.186 | 1.000 |
| bfs|30 | 0.654 | 0.654 | 1.000 |
| bfs|100 | 0.769 | 0.769 | 1.000 |
| ceiling|100 | 1.000 | 1.000 | 1.000 |
| random|10 | 0.115 | 0.115 | 1.000 |
| random|30 | 0.513 | 0.513 | 1.000 |
| random|100 | 0.750 | 0.750 | 1.000 |
| react|10 | 0.103 | 0.103 | 1.000 |
| react|30 | 0.551 | 0.551 | 1.000 |
| react|100 | 0.718 | 0.718 | 1.000 |
| score|10 | 0.147 | 0.147 | 1.000 |
| score|30 | 0.609 | 0.609 | 1.000 |
| score|100 | 0.756 | 0.756 | 1.000 |
| sos|10 | 0.199 | 0.199 | 1.000 |
| sos|30 | 0.641 | 0.641 | 1.000 |
| sos|100 | 0.994 | 0.994 | 1.000 |

## Qwen/Qwen3-1.7B-Base

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: +0.109 [+0.038, +0.179] (Holm p 0.0055, n 156)
- budget 30: +0.154 [+0.090, +0.224] (Holm p 0.0015, n 156)
- budget 100: +0.128 [+0.071, +0.192] (Holm p 0.0015, n 156)

Secondary (found on provable problems):

- score − sos, budget 10: +0.026 [-0.038, +0.090] (p 0.5104, n 156)
- score − sos, budget 30: +0.026 [-0.019, +0.071] (p 0.2984, n 156)
- score − sos, budget 100: -0.115 [-0.173, -0.064] (p 0.0005, n 156)
- react − random, budget 10: -0.013 [-0.077, +0.051] (p 0.7548, n 156)
- react − random, budget 30: +0.045 [-0.026, +0.115] (p 0.2249, n 156)
- react − random, budget 100: -0.071 [-0.122, -0.019] (p 0.0095, n 156)
- react − score, budget 10: -0.122 [-0.192, -0.051] (p 0.0005, n 156)
- react − score, budget 30: -0.109 [-0.160, -0.058] (p 0.0005, n 156)
- react − score, budget 100: -0.199 [-0.269, -0.135] (p 0.0005, n 156)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.186 | 0.186 | 1.000 |
| bfs|30 | 0.654 | 0.654 | 1.000 |
| bfs|100 | 0.769 | 0.769 | 1.000 |
| ceiling|100 | 1.000 | 1.000 | 1.000 |
| random|10 | 0.115 | 0.115 | 1.000 |
| random|30 | 0.513 | 0.513 | 1.000 |
| random|100 | 0.750 | 0.750 | 1.000 |
| react|10 | 0.103 | 0.103 | 1.000 |
| react|30 | 0.558 | 0.558 | 1.000 |
| react|100 | 0.679 | 0.679 | 1.000 |
| score|10 | 0.224 | 0.224 | 1.000 |
| score|30 | 0.667 | 0.667 | 1.000 |
| score|100 | 0.878 | 0.878 | 1.000 |
| sos|10 | 0.199 | 0.199 | 1.000 |
| sos|30 | 0.641 | 0.641 | 1.000 |
| sos|100 | 0.994 | 0.994 | 1.000 |

