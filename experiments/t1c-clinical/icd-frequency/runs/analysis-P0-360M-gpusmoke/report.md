# T1c-F analysis — encoder P0-360M-gpusmoke (seeds [1])

Aggregates only (licensed data stay under the data root). Preregistration: `experiments/t1c-clinical/icd-frequency/preregistration.md`.

Codes: {'seen_with_eval_positive': 172, 'heldout_scored_all': 94, 'heldout_scored_eval': 18, 'rare_seen': 9, 'frequent_seen': 130}. Bootstrap replicates: 200.

## Endpoints (mean over seeds)

| condition | E1 held-out macro-AUC (all adm.) | E2 slope AUC ~ ln f | seen macro-AUC | rare | frequent | gap | held-out top-100 (generalized) |
|---|---:|---:|---:|---:|---:|---:|---:|
| free | 0.475 | 0.0016 | 0.516 | 0.647 | 0.520 | -0.128 | 0.011 |
| composed_head | 0.543 | 0.0218 | 0.580 | 0.382 | 0.591 | 0.209 | 0.011 |
| transe | 0.574 | 0.0139 | 0.533 | 0.575 | 0.547 | -0.028 | 0.039 |
| title | 0.474 | -0.0122 | 0.522 | 0.435 | 0.515 | 0.080 | 0.021 |
| random | 0.541 | 0.0062 | 0.494 | 0.560 | 0.500 | -0.060 | 0.011 |
| gram | 0.551 | 0.0291 | 0.587 | 0.493 | 0.614 | 0.121 | 0.059 |
| composed_free | 0.559 | 0.0213 | 0.534 | 0.473 | 0.558 | 0.085 | 0.027 |

## Primary comparisons (composed_head − free; Holm over E1, E2)

- E1: Δ = 0.0834 [-0.0213, 0.1911], p = 0.1400, Holm p = 0.2800
- E2: Δ = 0.0138 [-0.0515, 0.0880], p = 0.7100, Holm p = 0.7100

Decision: `{"available": true, "E1_supported": false, "E1_refuted": false, "E2_slope_reduced": false, "E2_supported": false, "E2_flattening_by_damage": false, "E1_specificity": {"random": "inconclusive", "title": "inconclusive", "transe": "inconclusive", "gram": "inconclusive", "composed_free": "inconclusive"}}`

## Per bin (seen codes on evaluation admissions)

| bin | seen / held-out codes | free AUC / top-10 / top-100 | composed_head AUC / top-10 / top-100 | transe AUC / top-10 / top-100 | title AUC / top-10 / top-100 | random AUC / top-10 / top-100 | gram AUC / top-10 / top-100 | composed_free AUC / top-10 / top-100 |
|---|---|---|---|---|---|---|---|---|
| [-14,-12) | 2 / 0 | 0.848 / 0.000 / 0.000 | 0.370 / 0.000 / 0.000 | 0.522 / 0.000 / 0.000 | 0.761 / 0.000 / 0.000 | 0.587 / 0.000 / 0.000 | 0.522 / 0.000 / 0.000 | 0.630 / 0.000 / 0.000 |
| [-12,-10) | 7 / 202 | 0.590 / 0.000 / 0.000 | 0.385 / 0.000 / 0.000 | 0.590 / 0.000 / 0.000 | 0.342 / 0.000 / 0.000 | 0.553 / 0.000 / 0.143 | 0.484 / 0.000 / 0.000 | 0.429 / 0.000 / 0.000 |
| [-10,-8) | 33 / 131 | 0.466 / 0.000 / 0.000 | 0.592 / 0.000 / 0.030 | 0.468 / 0.000 / 0.030 | 0.573 / 0.000 / 0.030 | 0.451 / 0.000 / 0.030 | 0.507 / 0.000 / 0.000 | 0.453 / 0.000 / 0.000 |
| [-8,-6) | 81 / 49 | 0.507 / 0.000 / 0.173 | 0.584 / 0.000 / 0.111 | 0.551 / 0.000 / 0.099 | 0.527 / 0.000 / 0.049 | 0.493 / 0.000 / 0.043 | 0.612 / 0.012 / 0.160 | 0.564 / 0.000 / 0.173 |
| [-6,-4) | 45 / 0 | 0.531 / 0.141 / 0.826 | 0.601 / 0.089 / 0.252 | 0.532 / 0.056 / 0.289 | 0.496 / 0.044 / 0.244 | 0.507 / 0.106 / 0.378 | 0.622 / 0.100 / 0.807 | 0.535 / 0.146 / 0.583 |
| [-4,-2) | 4 / 0 | 0.650 / 0.938 / 1.000 | 0.621 / 0.500 / 0.525 | 0.638 / 0.281 / 1.000 | 0.478 / 0.094 / 0.406 | 0.560 / 1.000 / 1.000 | 0.551 / 1.000 / 1.000 | 0.713 / 0.750 / 1.000 |
| [-2,0) | 0 / 0 | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |

## Frequency information in the code vectors

| condition | ridge R² (code vector) | at init | ridge R² (label parameters) | ρ(bias, ln f) | kNN agreement | at init | t-SNE kNN agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| free | 0.210 | -0.002 | 0.244 | 0.130 | 0.155 | -0.013 | 0.228 |
| composed_head | 0.160 | 0.085 | 0.161 | 0.110 | 0.397 | 0.387 | 0.164 |
| transe | 0.125 | 0.125 | 0.115 | 0.043 | 0.443 | 0.443 | 0.161 |
| title | 0.221 | 0.221 | 0.190 | 0.007 | 0.424 | 0.424 | 0.179 |
| random | -0.002 | -0.002 | 0.003 | 0.004 | 0.028 | 0.028 | 0.035 |
| gram | 0.339 | 0.088 | 0.365 | 0.275 | 0.479 | 0.431 | 0.212 |
| composed_free | 0.230 | 0.083 | 0.247 | 0.197 | 0.396 | 0.353 | 0.121 |
