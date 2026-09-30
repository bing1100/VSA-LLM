# Experiment E0 — synthetic identifiability

## D0.1 contextual composition

| Teacher | M1(q) − M0, held-out concepts × held-out contexts (cosine, 95% CI) | Passes |
|---|---|---|
| static_0.0 | -0.0000 [-0.0000, -0.0000] | True |
| contextual_0.5 | +0.3585 [+0.3355, +0.3816] | True |
| contextual_1.0 | +0.3943 [+0.3084, +0.4802] | True |
| contextual_2.0 | +0.4325 [+0.2275, +0.6375] | True |
| contextual_4.0 | +0.4604 [+0.1238, +0.7969] | True |

D0.1 gate: **PASS**.

## D0.3 factored mapping (held-out concept cosine)

| Learner | Cosine |
|---|---:|
| free_k4 | 0.5979 |
| free_k8 | 0.6467 |
| free_k16 | 0.6805 |
| free_k32 | 0.6962 |
| induced_k4 | 0.9759 |
| induced_k8 | 0.9447 |
| induced_k16 | 0.9094 |
| induced_k32 | 0.8927 |
| hybrid_k4 | 0.9781 |
| hybrid_k8 | 0.9440 |
| hybrid_k16 | 0.9099 |
| hybrid_k32 | 0.8953 |
| m0 | 0.6230 |

D0.3 gate: **PASS**.

## D0.2 split detection

### atomics

| Policy | Splits at convergence (raw) | Precision | Recall | ARI train usages | ARI held-out usages | False-split rate (raw) | Held-out cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.0 (0.0) | nan | 0.000 | nan | nan | 0.000 (0.000) | 0.9462 |
| m3 | 8.0 (23.0) | 1.000 | 1.000 | 1.000 | 0.046 | 0.000 (0.375) | 0.9387 |
| m3_anderson | 11.0 (23.0) | 0.804 | 0.958 | 1.000 | 0.061 | 0.083 (0.383) | 0.9386 |
| coherence_only | 8.3 (24.0) | 0.884 | 0.917 | 1.000 | 0.102 | 0.025 (0.417) | 0.9404 |
| random | 19.3 (21.7) | 0.120 | 0.292 | 0.018 | 0.203 | 0.425 (0.483) | 0.9432 |
| oracle | 0.0 (0.0) | nan | nan | 1.000 | 1.000 | nan (nan) | 0.9999 |

Passes: **True**.

### relations

| Policy | Splits at convergence (raw) | Precision | Recall | ARI train usages | ARI held-out usages | False-split rate (raw) | Held-out cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.0 (0.0) | nan | 0.000 | nan | nan | 0.000 (0.000) | 0.9022 |
| m3 | 2.0 (3.0) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 (0.250) | 0.8394 |
| m3_anderson | 2.0 (3.0) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 (0.250) | 0.8394 |
| coherence_only | 1.3 (3.0) | 1.000 | 0.667 | 1.000 | 0.000 | 0.000 (0.417) | 0.8564 |
| random | 2.7 (3.0) | 0.333 | 0.500 | nan | nan | 0.417 (0.500) | 0.9016 |
| oracle | 0.0 (0.0) | nan | nan | 1.000 | 1.000 | nan (nan) | 0.9998 |

Passes: **True**.

D0.2 gate: **PASS**.

**G1 (all E0 gates): PASS.**
