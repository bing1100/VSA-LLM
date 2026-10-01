# Experiment E0 — synthetic identifiability

## D0.2 split detection

### atomics

| Policy | Splits at convergence (raw) | Precision | Recall | ARI train usages | ARI held-out usages | False-split rate (raw) | Held-out cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.0 (0.0) | nan | 0.000 | nan | nan | 0.000 (0.000) | 0.9462 |
| m3 | 8.3 (24.0) | 0.963 | 1.000 | 1.000 | 0.042 | 0.008 (0.400) | 0.9423 |
| m3_anderson | 13.0 (24.0) | 0.637 | 0.917 | 0.995 | 0.048 | 0.142 (0.417) | 0.9448 |
| coherence_only | 8.7 (24.0) | 0.815 | 0.875 | 1.000 | 0.000 | 0.042 (0.425) | 0.9418 |
| random | 19.3 (24.0) | 0.120 | 0.292 | 0.018 | 0.203 | 0.425 (0.542) | 0.9432 |
| oracle | 0.0 (0.0) | nan | nan | 1.000 | 1.000 | nan (nan) | 0.9999 |

Passes: **True**.

### relations

| Policy | Splits at convergence (raw) | Precision | Recall | ARI train usages | ARI held-out usages | False-split rate (raw) | Held-out cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.0 (0.0) | nan | 0.000 | nan | nan | 0.000 (0.000) | 0.9022 |
| m3 | 2.0 (3.0) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 (0.250) | 0.9069 |
| m3_anderson | 2.0 (3.0) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 (0.250) | 0.9069 |
| coherence_only | 1.3 (3.0) | 1.000 | 0.667 | 1.000 | 0.000 | 0.000 (0.417) | 0.9000 |
| random | 2.7 (3.0) | 0.333 | 0.500 | nan | nan | 0.417 (0.500) | 0.9016 |
| oracle | 0.0 (0.0) | nan | nan | 1.000 | 1.000 | nan (nan) | 0.9998 |

Passes: **True**.

D0.2 gate: **PASS**.
