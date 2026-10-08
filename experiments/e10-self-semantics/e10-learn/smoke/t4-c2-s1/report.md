# E10.L erased (SMOKE)

## SmolLM2-360M-full-C5-s1

Seen 8947; probes 16; erased 10987 (gold on probes 35; typed coverage 1.000); content relations: is_a, has_role, has_functional_parent, has_parent_hydride, has_part, is_conjugate_acid_of, is_conjugate_base_of, is_tautomer_of, is_enantiomer_of, is_substituent_group_from, contains_element; 94 concept types.

Decoders: passive out-of-fold cosine 0.425; evidence 314 concepts / 549 observations (mean cosine 0.311). Primary rule: `holm+decoy`; decoy thresholds {'decompose/0': None, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| holm+decoy | decompose | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| holm | all | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| holm | decompose | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| bh | all | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| bh | decompose | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| knockoff | all | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| knockoff | decompose | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| decoy | all | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |
| decoy | decompose | 23 | 18 | 0 | — | 0.000 | 0.000 | 0.171 | 0.000 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 16 | 0.0000 (0) [0.000, 0.194] | 0.0625 (1) [0.011, 0.283] | 0.0625 (1) [0.011, 0.283] | 0.0000 (0) [0.000, 0.194] |
| relabel | 4 | 0.0000 (0) [0.000, 0.490] | 0.0000 (0) [0.000, 0.490] | 0.0000 (0) [0.000, 0.490] | 0.0000 (0) [0.000, 0.490] |
| swap | 18 | 0.0000 (0) [0.000, 0.176] | 0.0000 (0) [0.000, 0.176] | 0.0000 (0) [0.000, 0.176] | 0.0000 (0) [0.000, 0.176] |
| permute | 10 | 0.0000 (0) [0.000, 0.278] | 0.0000 (0) [0.000, 0.278] | 0.0000 (0) [0.000, 0.278] | 0.0000 (0) [0.000, 0.278] |

Shared pool: 105 candidates, 35 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.733 | 0.343 |
| correlate | 0.718 | 0.229 |
| prior | 0.652 | 0.371 |
| lre | 0.860 | 0.543 |
| amie | 0.514 | 0.029 |
| transe | 0.875 | 0.571 |
| rotate | 0.829 | 0.457 |
| complex | 0.827 | 0.486 |

Timings (s): {'setup': 0.25, 'decoders': 0.71, 'propose': 1.4, 'test': 0.03, 'complete-null': 2.62, 'decide': 0.12, 'nulls': 0.2, 'pool-readouts': 1.49, 'amie': 1.39, 'kge': 25.87}

