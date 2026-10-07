# T1c-F analysis — encoder P0-135M-cpusmoke (seeds [1])

Aggregates only (licensed data stay under the data root). Preregistration: `experiments/t1c-clinical/icd-frequency/preregistration.md`.

Codes: {'seen_with_eval_positive': 48, 'heldout_scored_all': 31, 'heldout_scored_eval': 4, 'rare_seen': 1, 'frequent_seen': 37}. Bootstrap replicates: 50.

## Endpoints (mean over seeds)

| condition | E1 held-out macro-AUC (all adm.) | E2 slope AUC ~ ln f | seen macro-AUC | rare | frequent | gap | held-out top-100 (generalized) |
|---|---:|---:|---:|---:|---:|---:|---:|
| free | 0.484 | 0.0695 | 0.479 | 0.000 | 0.520 | 0.520 | 0.000 |
| composed_head | 0.410 | 0.0105 | 0.505 | 0.750 | 0.473 | -0.277 | 0.000 |
| transe | 0.479 | 0.0144 | 0.531 | 0.500 | 0.527 | 0.027 | 0.032 |
| title | 0.469 | 0.0438 | 0.566 | 0.250 | 0.579 | 0.329 | 0.065 |
| random | 0.478 | 0.0410 | 0.517 | 0.250 | 0.529 | 0.279 | 0.000 |
| gram | 0.511 | 0.0225 | 0.582 | 0.750 | 0.592 | -0.158 | 0.032 |
| composed_free | 0.489 | 0.0318 | 0.533 | 0.500 | 0.529 | 0.029 | 0.000 |

## Primary comparisons (composed_head − free; Holm over E1, E2)

- E1: Δ = -0.0634 [-0.1646, 0.0338], p = 0.3200, Holm p = 0.6400
- E2: Δ = -0.0330 [-0.1836, 0.1788], p = 0.5200, Holm p = 0.6400

Decision: `{"available": true, "E1_supported": false, "E1_refuted": false, "E2_slope_reduced": false, "E2_supported": false, "E2_flattening_by_damage": false, "E1_specificity": {"random": "inconclusive", "title": "inconclusive", "transe": "inconclusive", "gram": "inconclusive", "composed_free": "inconclusive"}}`

## Per bin (seen codes on evaluation admissions)

| bin | seen / held-out codes | free AUC / top-10 / top-100 | composed_head AUC / top-10 / top-100 | transe AUC / top-10 / top-100 | title AUC / top-10 / top-100 | random AUC / top-10 / top-100 | gram AUC / top-10 / top-100 | composed_free AUC / top-10 / top-100 |
|---|---|---|---|---|---|---|---|---|
| [-14,-12) | 0 / 0 | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
| [-12,-10) | 1 / 202 | 0.000 / 0.000 / 0.000 | 0.750 / 0.000 / 0.000 | 0.500 / 0.000 / 0.000 | 0.250 / 0.000 / 0.000 | 0.250 / 0.000 / 0.000 | 0.750 / 0.000 / 0.000 | 0.500 / 0.000 / 0.000 |
| [-10,-8) | 10 / 131 | 0.375 / 0.000 / 0.000 | 0.600 / 0.000 / 0.000 | 0.550 / 0.000 / 0.000 | 0.550 / 0.000 / 0.000 | 0.500 / 0.000 / 0.100 | 0.525 / 0.000 / 0.000 | 0.550 / 0.000 / 0.000 |
| [-8,-6) | 20 / 49 | 0.425 / 0.000 / 0.150 | 0.350 / 0.000 / 0.000 | 0.487 / 0.000 / 0.050 | 0.512 / 0.000 / 0.000 | 0.438 / 0.000 / 0.000 | 0.537 / 0.000 / 0.200 | 0.412 / 0.050 / 0.100 |
| [-6,-4) | 13 / 0 | 0.635 / 0.077 / 0.154 | 0.590 / 0.000 / 0.000 | 0.590 / 0.000 / 0.077 | 0.654 / 0.000 / 0.077 | 0.647 / 0.154 / 0.154 | 0.647 / 0.077 / 0.231 | 0.679 / 0.000 / 0.000 |
| [-4,-2) | 4 / 0 | 0.625 / 0.500 / 1.000 | 0.708 / 0.000 / 0.250 | 0.521 / 0.250 / 0.500 | 0.667 / 0.000 / 0.500 | 0.604 / 0.250 / 0.750 | 0.688 / 0.125 / 0.500 | 0.625 / 0.250 / 0.875 |
| [-2,0) | 0 / 0 | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |

## Frequency information in the code vectors

| condition | ridge R² (code vector) | ridge R² (label parameters) | ρ(bias, ln f) | kNN agreement | t-SNE kNN agreement |
|---|---:|---:|---:|---:|---:|
| free | 0.007 | 0.017 | 0.019 | -0.001 | -0.005 |
| composed_head | 0.089 | 0.091 | -0.003 | 0.387 | 0.151 |
| transe | 0.125 | 0.114 | -0.040 | 0.443 | 0.189 |
| title | 0.195 | 0.177 | 0.019 | 0.422 | 0.171 |
| random | -0.002 | -0.003 | 0.016 | 0.028 | 0.004 |
| gram | 0.158 | 0.164 | 0.031 | 0.438 | 0.231 |
| composed_free | 0.112 | 0.112 | 0.053 | 0.359 | 0.158 |
