# Realistic capacity and calibration

| Algebra | D | Corr. | Degree | Top-1 | MRR | ECE | Brier | Coverage | Selective acc. |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| real_hrr | 256 | 0.00 | fixed16 | 0.788 | 0.849 | 0.025 | 0.114 | 0.622 | 0.947 |
| real_hrr | 256 | 0.00 | poisson16 | 0.764 | 0.830 | 0.028 | 0.119 | 0.654 | 0.918 |
| real_hrr | 256 | 0.00 | heavy_tail | 0.506 | 0.585 | 0.069 | 0.131 | 0.370 | 0.945 |
| real_hrr | 256 | 0.30 | fixed16 | 0.608 | 0.701 | 0.029 | 0.161 | 0.332 | 0.937 |
| real_hrr | 256 | 0.30 | poisson16 | 0.580 | 0.679 | 0.035 | 0.162 | 0.223 | 0.977 |
| real_hrr | 256 | 0.30 | heavy_tail | 0.466 | 0.541 | 0.057 | 0.147 | 0.353 | 0.882 |
| real_hrr | 512 | 0.00 | fixed16 | 0.990 | 0.994 | 0.007 | 0.008 | 1.000 | 0.990 |
| real_hrr | 512 | 0.00 | poisson16 | 0.985 | 0.991 | 0.003 | 0.012 | 1.000 | 0.985 |
| real_hrr | 512 | 0.00 | heavy_tail | 0.713 | 0.777 | 0.031 | 0.108 | 0.677 | 0.908 |
| real_hrr | 512 | 0.30 | fixed16 | 0.944 | 0.964 | 0.015 | 0.037 | 1.000 | 0.944 |
| real_hrr | 512 | 0.30 | poisson16 | 0.905 | 0.933 | 0.021 | 0.052 | 0.891 | 0.964 |
| real_hrr | 512 | 0.30 | heavy_tail | 0.810 | 0.851 | 0.046 | 0.066 | 0.698 | 0.990 |
| unitary_hrr | 256 | 0.00 | fixed16 | 0.823 | 0.878 | 0.031 | 0.097 | 0.764 | 0.932 |
| unitary_hrr | 256 | 0.00 | poisson16 | 0.784 | 0.845 | 0.032 | 0.116 | 0.694 | 0.930 |
| unitary_hrr | 256 | 0.00 | heavy_tail | 0.504 | 0.574 | 0.067 | 0.118 | 0.413 | 0.942 |
| unitary_hrr | 256 | 0.30 | fixed16 | 0.641 | 0.734 | 0.043 | 0.158 | 0.332 | 0.958 |
| unitary_hrr | 256 | 0.30 | poisson16 | 0.628 | 0.709 | 0.051 | 0.164 | 0.286 | 0.982 |
| unitary_hrr | 256 | 0.30 | heavy_tail | 0.467 | 0.543 | 0.033 | 0.131 | 0.351 | 0.897 |
| unitary_hrr | 512 | 0.00 | fixed16 | 0.995 | 0.997 | 0.001 | 0.004 | 1.000 | 0.995 |
| unitary_hrr | 512 | 0.00 | poisson16 | 0.986 | 0.991 | 0.001 | 0.008 | 1.000 | 0.986 |
| unitary_hrr | 512 | 0.00 | heavy_tail | 0.717 | 0.781 | 0.028 | 0.106 | 0.706 | 0.902 |
| unitary_hrr | 512 | 0.30 | fixed16 | 0.944 | 0.963 | 0.025 | 0.031 | 0.998 | 0.944 |
| unitary_hrr | 512 | 0.30 | poisson16 | 0.905 | 0.935 | 0.020 | 0.053 | 0.937 | 0.945 |
| unitary_hrr | 512 | 0.30 | heavy_tail | 0.800 | 0.833 | 0.036 | 0.074 | 0.703 | 0.986 |
| map | 256 | 0.00 | fixed16 | 0.861 | 0.909 | 0.029 | 0.085 | 0.745 | 0.960 |
| map | 256 | 0.00 | poisson16 | 0.801 | 0.859 | 0.038 | 0.111 | 0.744 | 0.921 |
| map | 256 | 0.00 | heavy_tail | 0.474 | 0.556 | 0.112 | 0.129 | 0.374 | 0.936 |
| map | 256 | 0.30 | fixed16 | 0.627 | 0.731 | 0.027 | 0.157 | 0.299 | 0.942 |
| map | 256 | 0.30 | poisson16 | 0.586 | 0.688 | 0.052 | 0.156 | 0.329 | 0.953 |
| map | 256 | 0.30 | heavy_tail | 0.462 | 0.529 | 0.067 | 0.140 | 0.391 | 0.829 |
| map | 512 | 0.00 | fixed16 | 0.993 | 0.996 | 0.005 | 0.006 | 1.000 | 0.993 |
| map | 512 | 0.00 | poisson16 | 0.988 | 0.992 | 0.002 | 0.010 | 1.000 | 0.988 |
| map | 512 | 0.00 | heavy_tail | 0.715 | 0.773 | 0.027 | 0.113 | 0.709 | 0.879 |
| map | 512 | 0.30 | fixed16 | 0.951 | 0.971 | 0.003 | 0.033 | 0.998 | 0.953 |
| map | 512 | 0.30 | poisson16 | 0.918 | 0.944 | 0.029 | 0.051 | 0.913 | 0.951 |
| map | 512 | 0.30 | heavy_tail | 0.830 | 0.863 | 0.039 | 0.068 | 0.730 | 0.990 |

## Gate decision

- ECE gate (`≤ 0.05`): **29/36 conditions pass**.
- Held-out selective accuracy (`≥ 0.95`, nonzero coverage): **17/36 pass**.
- Both calibration gates: **15/36 pass**.
- **Decision: do not promote a universal default backend or confidence policy.**

## Aggregate findings

- Dimension: 256: top1=0.632, ece=0.046; 512: top1=0.894, ece=0.019.
- Candidate correlation: 0.0: top1=0.799, evaluation_coverage=0.748; 0.3: top1=0.726, evaluation_coverage=0.598.
- Degree profile: fixed16: top1=0.847, ece=0.020, evaluation_selective_accuracy=0.958; heavy_tail: top1=0.622, ece=0.051, evaluation_selective_accuracy=0.924; poisson16: top1=0.819, ece=0.026, evaluation_selective_accuracy=0.958.
- No algebra dominates consistently enough to justify selection before degree-aware calibration and sharding.

Calibration and evaluation use disjoint seeds. Confidence is based only on the observable top-1/top-2 similarity gap.
