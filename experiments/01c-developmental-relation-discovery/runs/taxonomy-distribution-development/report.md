# Experiment 01c.4 — taxonomy distributional retrieval

Nodes: **412**; edges: **240**; relation counts: `{'hypernym': 120, 'instance_hypernym': 120}`.

| Method | Distribution MRR | Distribution R@10 | Exact MRR | Cosine |
|---|---:|---:|---:|---:|
| offset | 0.2387 | 0.5785 | 0.2374 | 0.0824 |
| hrr | 0.1890 | 0.4378 | 0.1861 | 0.0424 |
| basis_offset_residual_hrr | 0.2317 | 0.5683 | 0.2303 | 0.1136 |
| basis_offset_diagonal_control | 0.2387 | 0.5847 | 0.2373 | 0.1167 |
| shuffled_basis_offset_residual_hrr | 0.1263 | 0.2958 | 0.1257 | 0.0280 |
| shuffled_basis_offset_diagonal_control | 0.1345 | 0.3064 | 0.1338 | 0.0311 |

Distribution-MRR gain over non-HRR control: **-0.0070**.
Gain over shuffled control: **+0.1054**.
Cosine delta from offset: **+0.0312**.
Development criteria: **FAIL**.
