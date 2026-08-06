# Experiment 01c — reconstruction rescue

Nodes: **966**; edges: **560**; relation counts: `{'hypernym': 80, 'instance_hypernym': 80, 'member_meronym': 80, 'substance_meronym': 80, 'part_meronym': 80, 'antonym': 80, 'attribute': 80}`.

Primary condition: **node_disjoint / contextual_centered**.

| Method | Cosine | MSE | Target MRR | Recall@10 |
|---|---:|---:|---:|---:|
| additive | 0.2929 | 0.00184 | 0.2271 | 0.3447 |
| offset | 0.3488 | 0.00170 | 0.2013 | 0.3587 |
| hrr | 0.2592 | 0.00193 | 0.2437 | 0.3778 |
| residual_hrr | 0.2960 | 0.00183 | 0.2320 | 0.3566 |
| offset_residual_hrr | 0.3515 | 0.00169 | 0.2189 | 0.3709 |
| gated_offset_residual_hrr | 0.3513 | 0.00169 | 0.2213 | 0.3685 |
| basis_offset_residual_hrr | 0.3502 | 0.00169 | 0.2585 | 0.4087 |
| low_rank | 0.3130 | 0.00179 | 0.2018 | 0.3873 |
| shuffled_residual_hrr | 0.2914 | 0.00185 | 0.2291 | 0.3471 |
| shuffled_offset_residual_hrr | 0.2598 | 0.00193 | 0.1916 | 0.3300 |
| shuffled_gated_offset_residual_hrr | 0.2616 | 0.00192 | 0.1845 | 0.3348 |
| shuffled_basis_offset_residual_hrr | 0.2590 | 0.00193 | 0.1940 | 0.3269 |

## Exploratory joint criterion

Best reconstruction candidate: **offset_residual_hrr**.
Cosine gain over **offset**: **+0.0027**.
MRR delta from **hrr**: **-0.0248**.
Exploratory criteria: **FAIL**; promotion eligible: **False**; final gate: **FAIL**.

A smoke result is not evidence for promotion. Full 01c requires development/confirmation separation, multiple seeds, target audits, behavioral locality, and a second ontology.
