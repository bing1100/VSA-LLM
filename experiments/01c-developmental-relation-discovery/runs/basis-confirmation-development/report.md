# Experiment 01c — reconstruction rescue

Nodes: **966**; edges: **560**; relation counts: `{'hypernym': 80, 'instance_hypernym': 80, 'member_meronym': 80, 'substance_meronym': 80, 'part_meronym': 80, 'antonym': 80, 'attribute': 80}`.

Primary condition: **node_disjoint / contextual_centered**.

| Method | Cosine | MSE | Target MRR | Recall@10 |
|---|---:|---:|---:|---:|
| offset | 0.3219 | 0.00177 | 0.1858 | 0.3508 |
| hrr | 0.2480 | 0.00196 | 0.2407 | 0.3529 |
| basis_offset_residual_hrr | 0.3244 | 0.00176 | 0.2021 | 0.3602 |
| basis_offset_diagonal_control | 0.3274 | 0.00175 | 0.2011 | 0.3716 |
| shuffled_basis_offset_residual_hrr | 0.2293 | 0.00201 | 0.1813 | 0.3277 |
| shuffled_basis_offset_diagonal_control | 0.2302 | 0.00200 | 0.1836 | 0.3288 |

## Exploratory joint criterion

Best reconstruction candidate: **basis_offset_residual_hrr**.
Cosine gain over **offset**: **+0.0025**.
MRR delta from **hrr**: **-0.0386**.
Exploratory criteria: **FAIL**; promotion eligible: **False**; final gate: **FAIL**.

A smoke result is not evidence for promotion. Full 01c requires development/confirmation separation, multiple seeds, target audits, behavioral locality, and a second ontology.
