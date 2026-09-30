# Experiment 01c — reconstruction rescue

Protocol: **2**. Nodes: **974**; edges: **560**; relation counts: `{'hypernym': 80, 'instance_hypernym': 80, 'member_meronym': 80, 'substance_meronym': 80, 'part_meronym': 80, 'antonym': 80, 'attribute': 80}`.

Primary condition: **node_disjoint / contextual_centered**.

| Method | Cosine | MSE | Target MRR | Recall@10 |
|---|---:|---:|---:|---:|
| offset | 0.3788 | 0.00162 | 0.2007 | 0.3830 |
| hrr | 0.2750 | 0.00189 | 0.2356 | 0.3783 |
| hrr_identity | 0.2750 | 0.00189 | 0.2361 | 0.3783 |
| basis_offset_residual_hrr | 0.3831 | 0.00161 | 0.2123 | 0.4004 |
| basis_offset_diagonal_control | 0.3832 | 0.00161 | 0.2142 | 0.3978 |
| basis_offset_rotated_diagonal_control | 0.3840 | 0.00160 | 0.2129 | 0.3978 |
| low_rank_matched | 0.1306 | 0.00226 | 0.1205 | 0.2414 |
| shuffled_offset | 0.1721 | 0.00216 | 0.0967 | 0.1825 |
| shuffled_hrr | 0.2068 | 0.00207 | 0.2060 | 0.2951 |
| shuffled_basis_offset_residual_hrr | 0.1981 | 0.00209 | 0.1145 | 0.2040 |
| shuffled_basis_offset_diagonal_control | 0.1980 | 0.00209 | 0.1139 | 0.2185 |
| shuffled_basis_offset_rotated_diagonal_control | 0.1985 | 0.00209 | 0.1145 | 0.2113 |
| shuffled_low_rank_matched | 0.0769 | 0.00240 | 0.0881 | 0.2075 |

## Exploratory joint criterion (proxy)

Best reconstruction candidate: **basis_offset_residual_hrr** (selection: named in config).
Cosine gain over **offset**: **+0.0042** (per-seed 95% CI [-0.0081, +0.0166]).
MRR delta from **hrr**: **-0.0233**.
Exploratory criteria: **FAIL**; promotion eligible: **False**; final gate: **FAIL**.

Design-gate items this summarizer does **not** evaluate (the criteria above are proxies):

- 2 positive relation-residual R²
- 6 improves more than one relation family
- 7 reproduces on a second ontology/domain
- 8 stable across target paraphrases/templates
- 9 frozen-host behavioral locality

A smoke result is not evidence for promotion. Full 01c requires development/confirmation separation, multiple seeds, target audits, behavioral locality, and a second ontology.
