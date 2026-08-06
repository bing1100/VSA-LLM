# Experiment 01b Stage B — heterogeneous WordNet / frozen GPT-2

Nodes: **966**; edges: **560**; relation counts: `{'hypernym': 80, 'instance_hypernym': 80, 'member_meronym': 80, 'substance_meronym': 80, 'part_meronym': 80, 'antonym': 80, 'attribute': 80}`.

## edge_disjoint / static_raw

| Method | Target MRR |
|---|---:|
| additive | 0.4239 |
| relation_mean | 0.1306 |
| offset | 0.3621 |
| hrr | 0.4491 |
| diagonal | 0.3927 |
| low_rank | 0.3692 |
| shuffled_hrr | 0.4215 |
| shuffled_diagonal | 0.3644 |
| shuffled_low_rank | 0.1689 |

## edge_disjoint / static_centered

| Method | Target MRR |
|---|---:|
| additive | 0.4386 |
| relation_mean | 0.1681 |
| offset | 0.4608 |
| hrr | 0.4481 |
| diagonal | 0.4538 |
| low_rank | 0.3768 |
| shuffled_hrr | 0.4478 |
| shuffled_diagonal | 0.4302 |
| shuffled_low_rank | 0.2027 |

## edge_disjoint / contextual_raw

| Method | Target MRR |
|---|---:|
| additive | 0.2356 |
| relation_mean | 0.0814 |
| offset | 0.1994 |
| hrr | 0.2528 |
| diagonal | 0.1800 |
| low_rank | 0.2779 |
| shuffled_hrr | 0.2284 |
| shuffled_diagonal | 0.1531 |
| shuffled_low_rank | 0.1665 |

## edge_disjoint / contextual_centered

| Method | Target MRR |
|---|---:|
| additive | 0.2390 |
| relation_mean | 0.1181 |
| offset | 0.2289 |
| hrr | 0.2556 |
| diagonal | 0.2764 |
| low_rank | 0.2722 |
| shuffled_hrr | 0.2322 |
| shuffled_diagonal | 0.2597 |
| shuffled_low_rank | 0.1552 |

## node_disjoint / static_raw

| Method | Target MRR |
|---|---:|
| additive | 0.4261 |
| relation_mean | 0.1082 |
| offset | 0.3349 |
| hrr | 0.4279 |
| diagonal | 0.3754 |
| low_rank | 0.3156 |
| shuffled_hrr | 0.3997 |
| shuffled_diagonal | 0.3581 |
| shuffled_low_rank | 0.1812 |

## node_disjoint / static_centered

| Method | Target MRR |
|---|---:|
| additive | 0.4370 |
| relation_mean | 0.1387 |
| offset | 0.4061 |
| hrr | 0.4198 |
| diagonal | 0.4036 |
| low_rank | 0.3428 |
| shuffled_hrr | 0.4195 |
| shuffled_diagonal | 0.4089 |
| shuffled_low_rank | 0.2574 |

## node_disjoint / contextual_raw

| Method | Target MRR |
|---|---:|
| additive | 0.2415 |
| relation_mean | 0.0779 |
| offset | 0.2216 |
| hrr | 0.2631 |
| diagonal | 0.2045 |
| low_rank | 0.2477 |
| shuffled_hrr | 0.2347 |
| shuffled_diagonal | 0.1677 |
| shuffled_low_rank | 0.2231 |

## node_disjoint / contextual_centered

| Method | Target MRR |
|---|---:|
| additive | 0.2520 |
| relation_mean | 0.1151 |
| offset | 0.2097 |
| hrr | 0.2713 |
| diagonal | 0.2652 |
| low_rank | 0.2076 |
| shuffled_hrr | 0.2516 |
| shuffled_diagonal | 0.2342 |
| shuffled_low_rank | 0.1436 |

## Stage-B gate

Best structured: **hrr**; best control: **additive**; primary MRR gain: **+0.0194**.
Paired wins over additive: **3**; over matched shuffled control: **3**.
Primary cosine gain: **-0.0280**.
Retrieval gate: **PASS**; reconstruction gate: **FAIL**.

This tests frozen representation transfer, not vocabulary insertion or generated-answer behavior.
