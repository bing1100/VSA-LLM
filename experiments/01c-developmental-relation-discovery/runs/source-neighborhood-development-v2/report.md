# Experiment 01c.5 — complete parent-neighborhood retrieval

Nodes: **385**; edges: **301**; source queries: **149**.

Protocol: **2**.

| Method | Query MRR | MRR, parent seen in training | MRR, parent unseen | Set R@10 | Hit@10 | Set NLL | Centroid cosine | Targets/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| offset | 0.3255 | nan | 0.3255 | 0.3919 | 0.6306 | 3.1447 | 0.1575 | 2.00 |
| low_rank | 0.3378 | nan | 0.3378 | 0.3784 | 0.5946 | 3.7849 | 0.2178 | 2.00 |
| low_rank_matched | 0.2202 | nan | 0.2202 | 0.3333 | 0.5495 | 3.3588 | 0.0848 | 2.00 |
| basis_offset_residual_hrr | 0.3244 | nan | 0.3244 | 0.3514 | 0.5766 | 3.2933 | 0.1783 | 2.00 |
| basis_offset_diagonal_control | 0.3597 | nan | 0.3597 | 0.3919 | 0.6126 | 3.1476 | 0.1953 | 2.00 |
| basis_offset_rotated_diagonal_control | 0.3414 | nan | 0.3414 | 0.3694 | 0.6036 | 3.2400 | 0.1847 | 2.00 |
| shuffled_offset | 0.2365 | nan | 0.2365 | 0.2658 | 0.4144 | 3.8061 | 0.0654 | 2.00 |
| shuffled_low_rank | 0.3426 | nan | 0.3426 | 0.3468 | 0.5856 | 4.0673 | 0.2168 | 2.00 |
| shuffled_low_rank_matched | 0.2103 | nan | 0.2103 | 0.3198 | 0.5225 | 3.5967 | 0.0661 | 2.00 |
| shuffled_basis_offset_residual_hrr | 0.2734 | nan | 0.2734 | 0.2658 | 0.4324 | 4.0248 | 0.0816 | 2.00 |
| shuffled_basis_offset_diagonal_control | 0.2840 | nan | 0.2840 | 0.3018 | 0.4865 | 3.8894 | 0.0962 | 2.00 |
| shuffled_basis_offset_rotated_diagonal_control | 0.2660 | nan | 0.2660 | 0.2883 | 0.4685 | 3.9680 | 0.0864 | 2.00 |

HRR MRR gain over matched non-HRR: **-0.0169** (per-seed 95% CI [-0.0521, +0.0182]).
Set-recall gain: **-0.0180**.
Gain over shuffled HRR: **+0.0511** (95% CI [+0.0191, +0.0831]).
Development criteria (proxy): **FAIL**.

An empty unseen column means every held-out parent was also a training endpoint.
