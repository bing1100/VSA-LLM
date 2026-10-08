# E10.L erased (SMOKE)

## SmolLM2-360M-full-C5-s1

Seen 2845; probes 85; erased 3346 (gold on probes 141; typed coverage 0.844); content relations: is_a, area, purpose, subtype_of, owned_by, depends_on, part_of, measured_by, governed_by, replaces, uses, produces, reports_to, computed_from, reported_in, approved_by, applies_to, describes, sponsored_by, produced_by; 138 concept types.

Decoders: passive out-of-fold cosine 0.385; evidence 374 concepts / 1149 observations (mean cosine 0.232). Primary rule: `holm+decoy`; decoy thresholds {'decompose/0': None, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 170 | 125 | 0 | — | 0.000 | 0.000 | 0.057 | 0.000 |
| holm+decoy | decompose | 170 | 125 | 0 | — | 0.000 | 0.000 | 0.057 | 0.000 |
| holm | all | 170 | 125 | 1 | 1.000 | 0.007 | 0.014 | 0.057 | 0.000 |
| holm | decompose | 170 | 125 | 1 | 1.000 | 0.007 | 0.014 | 0.057 | 0.000 |
| bh | all | 170 | 125 | 1 | 1.000 | 0.007 | 0.014 | 0.057 | 0.000 |
| bh | decompose | 170 | 125 | 1 | 1.000 | 0.007 | 0.014 | 0.057 | 0.000 |
| knockoff | all | 170 | 125 | 0 | — | 0.000 | 0.000 | 0.057 | 0.000 |
| knockoff | decompose | 170 | 125 | 0 | — | 0.000 | 0.000 | 0.057 | 0.000 |
| decoy | all | 170 | 125 | 0 | — | 0.000 | 0.000 | 0.057 | 0.000 |
| decoy | decompose | 170 | 125 | 0 | — | 0.000 | 0.000 | 0.057 | 0.000 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 131 | 0.0000 (0) [0.000, 0.028] | 0.0153 (2) [0.004, 0.054] | 0.0305 (4) [0.012, 0.076] | 0.0000 (0) [0.000, 0.028] |
| relabel | 0 | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] |
| swap | 121 | 0.0000 (0) [0.000, 0.031] | 0.0000 (0) [0.000, 0.031] | 0.0000 (0) [0.000, 0.031] | 0.0000 (0) [0.000, 0.031] |
| permute | 104 | 0.0000 (0) [0.000, 0.036] | 0.0000 (0) [0.000, 0.036] | 0.0000 (0) [0.000, 0.036] | 0.0000 (0) [0.000, 0.036] |

Shared pool: 423 candidates, 141 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.526 | 0.050 |
| correlate | 0.520 | 0.043 |
| prior | 0.496 | 0.000 |
| lre | 0.530 | 0.000 |
| amie | 0.500 | 0.000 |
| transe | 0.547 | 0.007 |
| rotate | 0.495 | 0.000 |
| complex | 0.489 | 0.007 |

Timings (s): {'setup': 0.03, 'decoders': 0.44, 'propose': 0.74, 'test': 0.09, 'complete-null': 1.67, 'decide': 0.1, 'nulls': 0.42, 'pool-readouts': 2.28, 'amie': 0.31, 'kge': 7.61}

