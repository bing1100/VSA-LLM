# Experiment 01c — reconstruction rescue

Nodes: **319**; edges: **168**; relation counts: `{'hypernym': 24, 'instance_hypernym': 24, 'member_meronym': 24, 'substance_meronym': 24, 'part_meronym': 24, 'antonym': 24, 'attribute': 24}`.

Primary condition: **node_disjoint / contextual_centered**.

| Method | Cosine | MSE | Target MRR | Recall@10 |
|---|---:|---:|---:|---:|
| additive | 0.2956 | 0.00183 | 0.3419 | 0.5854 |
| offset | 0.3440 | 0.00171 | 0.3650 | 0.6829 |
| hrr | 0.2716 | 0.00190 | 0.3753 | 0.5854 |
| residual_hrr | 0.2666 | 0.00191 | 0.3875 | 0.5610 |
| offset_residual_hrr | 0.2736 | 0.00189 | 0.3820 | 0.6829 |
| gated_offset_residual_hrr | 0.2356 | 0.00199 | 0.3468 | 0.6341 |
| basis_offset_residual_hrr | 0.2658 | 0.00191 | 0.3048 | 0.6098 |
| low_rank | 0.3509 | 0.00169 | 0.3844 | 0.6341 |
| shuffled_offset_residual_hrr | 0.0964 | 0.00235 | 0.2350 | 0.4390 |

## Exploratory joint criterion

Best reconstruction candidate: **offset_residual_hrr**.
Cosine gain over **offset**: **-0.0704**.
MRR delta from **hrr**: **+0.0067**.
Exploratory criteria: **FAIL**; promotion eligible: **False**; final gate: **FAIL**.

A smoke result is not evidence for promotion. Full 01c requires development/confirmation separation, multiple seeds, target audits, behavioral locality, and a second ontology.
