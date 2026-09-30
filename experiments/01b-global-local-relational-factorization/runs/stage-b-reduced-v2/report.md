# Experiment 01b Stage B — heterogeneous WordNet / frozen GPT-2

Protocol: **2**. Nodes: **974**; edges: **560**; relation counts: `{'hypernym': 80, 'instance_hypernym': 80, 'member_meronym': 80, 'substance_meronym': 80, 'part_meronym': 80, 'antonym': 80, 'attribute': 80}`.

## node_disjoint / static_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4506 |
| relation_mean | 0.1315 |
| offset | 0.3621 |
| hrr | 0.4479 |
| hrr_identity | 0.4478 |
| diagonal | 0.3942 |
| low_rank_identity | 0.3129 |
| low_rank_matched | 0.4106 |
| shuffled_offset | 0.2487 |
| shuffled_hrr | 0.4249 |
| shuffled_hrr_identity | 0.4250 |
| shuffled_diagonal | 0.3469 |
| shuffled_low_rank_identity | 0.1660 |
| shuffled_low_rank_matched | 0.2987 |

## node_disjoint / static_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4689 |
| relation_mean | 0.1417 |
| offset | 0.4147 |
| hrr | 0.4329 |
| hrr_identity | 0.4328 |
| diagonal | 0.4302 |
| low_rank_identity | 0.3374 |
| low_rank_matched | 0.2904 |
| shuffled_offset | 0.2633 |
| shuffled_hrr | 0.3870 |
| shuffled_hrr_identity | 0.3874 |
| shuffled_diagonal | 0.3824 |
| shuffled_low_rank_identity | 0.2522 |
| shuffled_low_rank_matched | 0.2184 |

## node_disjoint / contextual_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2243 |
| relation_mean | 0.0866 |
| offset | 0.1981 |
| hrr | 0.2128 |
| hrr_identity | 0.2103 |
| diagonal | 0.1799 |
| low_rank_identity | 0.2702 |
| low_rank_matched | 0.2029 |
| shuffled_offset | 0.1505 |
| shuffled_hrr | 0.2017 |
| shuffled_hrr_identity | 0.2019 |
| shuffled_diagonal | 0.1598 |
| shuffled_low_rank_identity | 0.1707 |
| shuffled_low_rank_matched | 0.1442 |

## node_disjoint / contextual_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2254 |
| relation_mean | 0.1108 |
| offset | 0.2096 |
| hrr | 0.2277 |
| hrr_identity | 0.2279 |
| diagonal | 0.2540 |
| low_rank_identity | 0.1926 |
| low_rank_matched | 0.1772 |
| shuffled_offset | 0.0883 |
| shuffled_hrr | 0.1818 |
| shuffled_hrr_identity | 0.1818 |
| shuffled_diagonal | 0.1881 |
| shuffled_low_rank_identity | 0.1364 |
| shuffled_low_rank_matched | 0.0950 |

## token_disjoint / static_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4407 |
| relation_mean | 0.1310 |
| offset | 0.3625 |
| hrr | 0.4506 |
| hrr_identity | 0.4506 |
| diagonal | 0.3882 |
| low_rank_identity | 0.3435 |
| low_rank_matched | 0.3758 |
| shuffled_offset | 0.2500 |
| shuffled_hrr | 0.4240 |
| shuffled_hrr_identity | 0.4240 |
| shuffled_diagonal | 0.3763 |
| shuffled_low_rank_identity | 0.1655 |
| shuffled_low_rank_matched | 0.2578 |

## token_disjoint / static_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4554 |
| relation_mean | 0.1526 |
| offset | 0.4370 |
| hrr | 0.4272 |
| hrr_identity | 0.4266 |
| diagonal | 0.4362 |
| low_rank_identity | 0.3637 |
| low_rank_matched | 0.2829 |
| shuffled_offset | 0.2877 |
| shuffled_hrr | 0.4161 |
| shuffled_hrr_identity | 0.4161 |
| shuffled_diagonal | 0.4241 |
| shuffled_low_rank_identity | 0.2610 |
| shuffled_low_rank_matched | 0.2036 |

## token_disjoint / contextual_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2447 |
| relation_mean | 0.0783 |
| offset | 0.2135 |
| hrr | 0.2445 |
| hrr_identity | 0.2436 |
| diagonal | 0.1818 |
| low_rank_identity | 0.2681 |
| low_rank_matched | 0.1833 |
| shuffled_offset | 0.1495 |
| shuffled_hrr | 0.2358 |
| shuffled_hrr_identity | 0.2413 |
| shuffled_diagonal | 0.1498 |
| shuffled_low_rank_identity | 0.1622 |
| shuffled_low_rank_matched | 0.1861 |

## token_disjoint / contextual_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2370 |
| relation_mean | 0.1072 |
| offset | 0.1884 |
| hrr | 0.2514 |
| hrr_identity | 0.2514 |
| diagonal | 0.2554 |
| low_rank_identity | 0.2074 |
| low_rank_matched | 0.1863 |
| shuffled_offset | 0.0790 |
| shuffled_hrr | 0.2153 |
| shuffled_hrr_identity | 0.2153 |
| shuffled_diagonal | 0.2161 |
| shuffled_low_rank_identity | 0.1296 |
| shuffled_low_rank_matched | 0.1169 |

## edge_disjoint / static_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4050 |
| relation_mean | 0.1327 |
| offset | 0.3479 |
| hrr | 0.4323 |
| hrr_identity | 0.4325 |
| diagonal | 0.3829 |
| low_rank_identity | 0.3679 |
| low_rank_matched | 0.4364 |
| shuffled_offset | 0.2209 |
| shuffled_hrr | 0.3924 |
| shuffled_hrr_identity | 0.3922 |
| shuffled_diagonal | 0.3182 |
| shuffled_low_rank_identity | 0.1351 |
| shuffled_low_rank_matched | 0.3283 |

## edge_disjoint / static_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4139 |
| relation_mean | 0.1643 |
| offset | 0.4332 |
| hrr | 0.4244 |
| hrr_identity | 0.4244 |
| diagonal | 0.4398 |
| low_rank_identity | 0.3876 |
| low_rank_matched | 0.3015 |
| shuffled_offset | 0.2611 |
| shuffled_hrr | 0.3672 |
| shuffled_hrr_identity | 0.3703 |
| shuffled_diagonal | 0.3547 |
| shuffled_low_rank_identity | 0.1869 |
| shuffled_low_rank_matched | 0.2134 |

## edge_disjoint / contextual_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2308 |
| relation_mean | 0.0809 |
| offset | 0.2013 |
| hrr | 0.2400 |
| hrr_identity | 0.2424 |
| diagonal | 0.1785 |
| low_rank_identity | 0.2876 |
| low_rank_matched | 0.1718 |
| shuffled_offset | 0.1334 |
| shuffled_hrr | 0.2170 |
| shuffled_hrr_identity | 0.2165 |
| shuffled_diagonal | 0.1330 |
| shuffled_low_rank_identity | 0.1393 |
| shuffled_low_rank_matched | 0.1229 |

## edge_disjoint / contextual_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2304 |
| relation_mean | 0.1200 |
| offset | 0.2310 |
| hrr | 0.2502 |
| hrr_identity | 0.2502 |
| diagonal | 0.2726 |
| low_rank_identity | 0.2682 |
| low_rank_matched | 0.1968 |
| shuffled_offset | 0.0925 |
| shuffled_hrr | 0.1982 |
| shuffled_hrr_identity | 0.1982 |
| shuffled_diagonal | 0.2046 |
| shuffled_low_rank_identity | 0.1109 |
| shuffled_low_rank_matched | 0.1083 |

## edge_disjoint_size_matched / static_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4050 |
| relation_mean | 0.1290 |
| offset | 0.3418 |
| hrr | 0.4346 |
| hrr_identity | 0.4346 |
| diagonal | 0.3748 |
| low_rank_identity | 0.3510 |
| low_rank_matched | 0.3397 |
| shuffled_offset | 0.2057 |
| shuffled_hrr | 0.3865 |
| shuffled_hrr_identity | 0.3865 |
| shuffled_diagonal | 0.3065 |
| shuffled_low_rank_identity | 0.1473 |
| shuffled_low_rank_matched | 0.2576 |

## edge_disjoint_size_matched / static_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.4152 |
| relation_mean | 0.1592 |
| offset | 0.4016 |
| hrr | 0.4092 |
| hrr_identity | 0.4097 |
| diagonal | 0.4149 |
| low_rank_identity | 0.3962 |
| low_rank_matched | 0.2822 |
| shuffled_offset | 0.2309 |
| shuffled_hrr | 0.3424 |
| shuffled_hrr_identity | 0.3433 |
| shuffled_diagonal | 0.3411 |
| shuffled_low_rank_identity | 0.2405 |
| shuffled_low_rank_matched | 0.1785 |

## edge_disjoint_size_matched / contextual_raw

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2308 |
| relation_mean | 0.0769 |
| offset | 0.1981 |
| hrr | 0.2454 |
| hrr_identity | 0.2477 |
| diagonal | 0.1853 |
| low_rank_identity | 0.2721 |
| low_rank_matched | 0.1703 |
| shuffled_offset | 0.1318 |
| shuffled_hrr | 0.2047 |
| shuffled_hrr_identity | 0.2055 |
| shuffled_diagonal | 0.1508 |
| shuffled_low_rank_identity | 0.1626 |
| shuffled_low_rank_matched | 0.1157 |

## edge_disjoint_size_matched / contextual_centered

| Method | Target MRR (test) |
|---|---:|
| additive | 0.2327 |
| relation_mean | 0.1156 |
| offset | 0.2408 |
| hrr | 0.2513 |
| hrr_identity | 0.2512 |
| diagonal | 0.2529 |
| low_rank_identity | 0.2378 |
| low_rank_matched | 0.1723 |
| shuffled_offset | 0.0791 |
| shuffled_hrr | 0.1976 |
| shuffled_hrr_identity | 0.1976 |
| shuffled_diagonal | 0.2042 |
| shuffled_low_rank_identity | 0.1173 |
| shuffled_low_rank_matched | 0.0856 |

## Stage-B proxy criteria

Candidate: **hrr** (selection: named in config); control: **additive**; primary MRR gain: **+0.0023**.
Gain over additive per seed: 95% CI [-0.0643, +0.0689]; over matched shuffled: [-0.0120, +0.1038].
Paired wins over additive: **1**; over matched shuffled control: **3**.
Primary cosine gain: **-0.0394**.
Retrieval criteria: **FAIL**; reconstruction criteria: **FAIL**; promotion eligible: **False**.

Design-gate items this summarizer does **not** evaluate (the criteria above are proxies):

- 1 beats weighted additive and the strongest equal-budget graph/unstructured baseline on held-out behavioral transfer
- 2 shifts the adaptation curve left
- 3 globally shared relations outperform edge-specific memorization on held-out transfer
- 4 passes recipe/edge/alias leakage audits and reproduces over at least three splits
- 5 transfers to a second relation or ontology family
- 6 preserves frozen-host locality
- 7 emits auditable relation parameters, edge salience, residual allocation, uncertainty and provenance

This tests frozen representation transfer, not vocabulary insertion or generated-answer behavior.
