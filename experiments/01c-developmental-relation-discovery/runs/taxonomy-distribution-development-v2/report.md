# Experiment 01c.4 — taxonomy distributional retrieval

Protocol: **2**. Nodes: **412**; edges: **240**; relation counts: `{'hypernym': 120, 'instance_hypernym': 120}`.

| Method | Distribution MRR | Distribution R@10 | Exact MRR | Cosine |
|---|---:|---:|---:|---:|
| offset | 0.2270 | 0.5785 | 0.2257 | 0.0845 |
| hrr | 0.1912 | 0.4590 | 0.1884 | 0.0437 |
| basis_offset_residual_hrr | 0.2325 | 0.5789 | 0.2310 | 0.1153 |
| basis_offset_diagonal_control | 0.2423 | 0.5741 | 0.2408 | 0.1187 |
| basis_offset_rotated_diagonal_control | 0.2339 | 0.5741 | 0.2324 | 0.1166 |
| low_rank_matched | 0.2199 | 0.4691 | 0.2186 | 0.0647 |
| shuffled_offset | 0.0595 | 0.0880 | 0.0593 | -0.0178 |
| shuffled_basis_offset_residual_hrr | 0.0673 | 0.1047 | 0.0671 | -0.0011 |
| shuffled_basis_offset_diagonal_control | 0.0694 | 0.1319 | 0.0691 | 0.0023 |
| shuffled_basis_offset_rotated_diagonal_control | 0.0679 | 0.1270 | 0.0677 | -0.0007 |
| shuffled_low_rank_matched | 0.1321 | 0.3263 | 0.1318 | 0.0370 |

Distribution-MRR gain over non-HRR control: **-0.0014** (per-seed 95% CI [-0.0065, +0.0037]).
Gain over shuffled control: **+0.1652** (95% CI [+0.1100, +0.2204]).
Cosine delta from offset: **+0.0308**.
Development criteria (proxy): **FAIL**.
