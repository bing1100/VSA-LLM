# Experiment 01c.5 — complete parent-neighborhood retrieval

Nodes: **385**; edges: **301**; source queries: **149**.

| Method | Query MRR | Set R@10 | Hit@10 | Set NLL | Centroid cosine | Targets/query |
|---|---:|---:|---:|---:|---:|---:|
| offset | 0.3282 | 0.4009 | 0.6757 | 3.0249 | 0.1718 | 2.00 |
| low_rank | 0.4386 | 0.4640 | 0.7027 | 3.4548 | 0.2590 | 2.00 |
| basis_offset_residual_hrr | 0.3411 | 0.4099 | 0.6847 | 3.0816 | 0.1944 | 2.00 |
| basis_offset_diagonal_control | 0.3722 | 0.4234 | 0.6937 | 2.9450 | 0.2106 | 2.00 |
| shuffled_basis_offset_residual_hrr | 0.2413 | 0.3108 | 0.5315 | 3.9214 | 0.0974 | 2.00 |
| shuffled_basis_offset_diagonal_control | 0.2957 | 0.3153 | 0.5225 | 3.7559 | 0.1145 | 2.00 |

HRR MRR gain over matched non-HRR: **-0.0311**.
Set-recall gain: **-0.0135**.
Gain over shuffled HRR: **+0.0998**.
Development criteria: **FAIL**.
