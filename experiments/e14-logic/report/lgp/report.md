# E14 LG-P — pooled report

## HuggingFaceTB/SmolLM2-135M

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: +0.000 [-0.020, +0.020] (Holm p 1.0000, n 408)
- budget 30: -0.017 [-0.044, +0.010] (Holm p 0.5359, n 408)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.858 | 0.721 | 1.000 |
| bfs|30 | 0.914 | 0.831 | 1.000 |
| ceiling|30 | 1.000 | 1.000 | 1.000 |
| random|10 | 0.876 | 0.757 | 1.000 |
| random|30 | 0.959 | 0.919 | 1.000 |
| score|10 | 0.876 | 0.757 | 1.000 |
| score|30 | 0.950 | 0.902 | 1.000 |
| sos|10 | 0.925 | 0.853 | 1.000 |
| sos|30 | 1.000 | 1.000 | 1.000 |

## HuggingFaceTB/SmolLM2-360M

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: -0.005 [-0.027, +0.017] (Holm p 0.7653, n 408)
- budget 30: -0.054 [-0.081, -0.027] (Holm p 0.0010, n 408)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.858 | 0.721 | 1.000 |
| bfs|30 | 0.914 | 0.831 | 1.000 |
| ceiling|30 | 1.000 | 1.000 | 1.000 |
| random|10 | 0.876 | 0.757 | 1.000 |
| random|30 | 0.959 | 0.919 | 1.000 |
| score|10 | 0.874 | 0.752 | 1.000 |
| score|30 | 0.931 | 0.865 | 1.000 |
| sos|10 | 0.925 | 0.853 | 1.000 |
| sos|30 | 1.000 | 1.000 | 1.000 |

## Qwen/Qwen3-0.6B-Base

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: -0.007 [-0.027, +0.010] (Holm p 0.5359, n 408)
- budget 30: -0.049 [-0.078, -0.022] (Holm p 0.0010, n 408)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.858 | 0.721 | 1.000 |
| bfs|30 | 0.914 | 0.831 | 1.000 |
| ceiling|30 | 1.000 | 1.000 | 1.000 |
| cot|30 | 0.504 | 0.613 | 0.511 |
| random|10 | 0.876 | 0.757 | 1.000 |
| random|30 | 0.959 | 0.919 | 1.000 |
| react|10 | 0.863 | 0.730 | 1.000 |
| react|30 | 0.920 | 0.843 | 1.000 |
| score|10 | 0.873 | 0.750 | 1.000 |
| score|30 | 0.934 | 0.870 | 1.000 |
| sos|10 | 0.925 | 0.853 | 1.000 |
| sos|30 | 1.000 | 1.000 | 1.000 |
| unsound|10 | 0.512 | 1.000 | 0.511 |
| unsound|30 | 0.510 | 1.000 | 0.510 |

## Qwen/Qwen3-1.7B-Base

Primary — proofs found on provable problems, score − random (Holm over budgets):

- budget 10: +0.029 [+0.007, +0.054] (Holm p 0.0270, n 408)
- budget 30: +0.025 [-0.000, +0.051] (Holm p 0.0755, n 408)

| policy · budget | accuracy | found | soundness |
|---|---:|---:|---:|
| bfs|10 | 0.858 | 0.721 | 1.000 |
| bfs|30 | 0.914 | 0.831 | 1.000 |
| ceiling|30 | 1.000 | 1.000 | 1.000 |
| cot|30 | 0.540 | 0.380 | 0.574 |
| random|10 | 0.876 | 0.757 | 1.000 |
| random|30 | 0.959 | 0.919 | 1.000 |
| react|10 | 0.794 | 0.596 | 1.000 |
| react|30 | 0.843 | 0.691 | 1.000 |
| score|10 | 0.891 | 0.787 | 1.000 |
| score|30 | 0.971 | 0.944 | 1.000 |
| sos|10 | 0.925 | 0.853 | 1.000 |
| sos|30 | 1.000 | 1.000 | 1.000 |
| unsound|10 | 0.509 | 0.895 | 0.510 |
| unsound|30 | 0.506 | 0.914 | 0.509 |

