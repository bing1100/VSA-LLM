# E3 — developmental recovery on WordNet

## atomics

| Policy | Splits | Precision | Recall | ARI | False-split rate | Test MRR | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| m3 | 70.7 | 0.066 | 0.291 | 0.014 | 0.185 | 0.0347 | 2070443 |
| none | 0.0 | nan | 0.000 | nan | 0.000 | 0.0336 | 2028800 |
| coherence_only | 233.0 | 0.031 | 0.460 | 0.009 | 0.228 | 0.0359 | 2152021 |
| random | 29.3 | 0.057 | 0.102 | 0.029 | 0.084 | 0.0269 | 2061909 |
| uniform | 0.0 | nan | 0.000 | nan | 0.000 | 0.0349 | 2068425 |

M3 − uniform enlargement (test MRR): -0.0002 [-0.0076, +0.0072]; M3 − none: +0.0011 [-0.0048, +0.0069]; M3 − random (ARI): +nan [+nan, +nan].
Gate: **FAIL**.

## relations

| Policy | Splits | Precision | Recall | ARI | False-split rate | Test MRR | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| m3 | 0.0 | nan | 0.000 | nan | 0.000 | 0.0373 | 2296576 |
| none | 0.0 | nan | 0.000 | nan | 0.000 | 0.0373 | 2296576 |
| coherence_only | 0.0 | nan | 0.000 | nan | 0.000 | 0.0374 | 2296576 |
| random | 0.0 | nan | 0.000 | nan | 0.000 | 0.0373 | 2296576 |
| uniform | 0.0 | nan | 0.000 | nan | 0.000 | 0.0374 | 2296576 |
| stem_cell | 0.0 | nan | 0.000 | 0.001 | 0.000 | 0.0380 | 2307056 |

M3 − uniform enlargement (test MRR): -0.0000 [-0.0001, +0.0001]; M3 − none: -0.0000 [-0.0000, +0.0000]; M3 − random (ARI): +nan [+nan, +nan].
Gate: **FAIL**.
