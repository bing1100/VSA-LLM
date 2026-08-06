# Experiment 01c — reconstruction rescue

Nodes: **319**; edges: **168**; relation counts: `{'hypernym': 24, 'instance_hypernym': 24, 'member_meronym': 24, 'substance_meronym': 24, 'part_meronym': 24, 'antonym': 24, 'attribute': 24}`.

Primary condition: **node_disjoint / contextual_centered**.

| Method | Cosine | MSE | Target MRR | Recall@10 |
|---|---:|---:|---:|---:|
| additive | 0.2956 | 0.00183 | 0.3419 | 0.5854 |
| offset | 0.3445 | 0.00171 | 0.3730 | 0.6585 |
| hrr | 0.2721 | 0.00190 | 0.3879 | 0.5854 |
| residual_hrr | 0.2990 | 0.00183 | 0.3329 | 0.5610 |
| offset_residual_hrr | 0.3477 | 0.00170 | 0.3784 | 0.6829 |
| gated_offset_residual_hrr | 0.3482 | 0.00170 | 0.3824 | 0.6829 |
| basis_offset_residual_hrr | 0.3387 | 0.00172 | 0.3687 | 0.6098 |
| low_rank | 0.3764 | 0.00162 | 0.3705 | 0.7073 |
| shuffled_offset_residual_hrr | 0.2018 | 0.00208 | 0.2793 | 0.4390 |

## Exploratory joint criterion

Best reconstruction candidate: **gated_offset_residual_hrr**.
Cosine gain over **offset**: **+0.0036**.
MRR delta from **hrr**: **-0.0056**.
Exploratory criteria: **FAIL**; promotion eligible: **False**; final gate: **FAIL**.

A smoke result is not evidence for promotion. Full 01c requires development/confirmation separation, multiple seeds, target audits, behavioral locality, and a second ontology.
