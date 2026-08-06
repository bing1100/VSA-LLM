# Experiment 01b Stage B — heterogeneous WordNet / frozen GPT-2

Nodes: **319**; edges: **168**; relation counts: `{'hypernym': 24, 'instance_hypernym': 24, 'member_meronym': 24, 'substance_meronym': 24, 'part_meronym': 24, 'antonym': 24, 'attribute': 24}`.

## edge_disjoint / static_centered

| Method | Target MRR |
|---|---:|
| additive | 0.4956 |
| relation_mean | 0.3136 |
| offset | 0.5153 |
| hrr | 0.4447 |
| diagonal | 0.4663 |
| low_rank | 0.4048 |
| shuffled_hrr | 0.4978 |

## edge_disjoint / contextual_centered

| Method | Target MRR |
|---|---:|
| additive | 0.3239 |
| relation_mean | 0.1862 |
| offset | 0.3313 |
| hrr | 0.3224 |
| diagonal | 0.3217 |
| low_rank | 0.2709 |
| shuffled_hrr | 0.3419 |

## node_disjoint / static_centered

| Method | Target MRR |
|---|---:|
| additive | 0.5917 |
| relation_mean | 0.2324 |
| offset | 0.4902 |
| hrr | 0.4290 |
| diagonal | 0.4846 |
| low_rank | 0.4212 |
| shuffled_hrr | 0.3023 |

## node_disjoint / contextual_centered

| Method | Target MRR |
|---|---:|
| additive | 0.3419 |
| relation_mean | 0.1499 |
| offset | 0.3650 |
| hrr | 0.3753 |
| diagonal | 0.3778 |
| low_rank | 0.3844 |
| shuffled_hrr | 0.2854 |

## Stage-B gate

Best structured: **low_rank**; best control: **additive**; primary MRR gain: **+0.0425**; gate: **PASS**.

This tests frozen representation transfer, not vocabulary insertion or generated-answer behavior.
