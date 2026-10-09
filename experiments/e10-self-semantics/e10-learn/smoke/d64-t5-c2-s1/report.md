# E10.L erased (SMOKE)

## SmolLM2-360M-full-C5-s1

Seen 2845; probes 70; erased 3346 (gold on probes 110; typed coverage 0.900); content relations: is_a, area, purpose, subtype_of, owned_by, depends_on, part_of, measured_by, governed_by, replaces, uses, produces, reports_to, computed_from, reported_in, approved_by, applies_to, describes, sponsored_by, produced_by; 138 concept types.

Decoders: passive out-of-fold cosine 0.385; evidence 330 concepts / 1036 observations (mean cosine 0.271). Primary rule: `holm+decoy`; decoy thresholds {'decompose/0': None, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 140 | 105 | 0 | — | 0.000 | 0.000 | 0.073 | 0.000 |
| holm+decoy | decompose | 140 | 105 | 0 | — | 0.000 | 0.000 | 0.073 | 0.000 |
| holm | all | 140 | 105 | 2 | 1.000 | 0.018 | 0.036 | 0.073 | 0.000 |
| holm | decompose | 140 | 105 | 2 | 1.000 | 0.018 | 0.036 | 0.073 | 0.000 |
| bh | all | 140 | 105 | 4 | 0.500 | 0.018 | 0.035 | 0.073 | 0.015 |
| bh | decompose | 140 | 105 | 4 | 0.500 | 0.018 | 0.035 | 0.073 | 0.015 |
| knockoff | all | 140 | 105 | 0 | — | 0.000 | 0.000 | 0.073 | 0.000 |
| knockoff | decompose | 140 | 105 | 0 | — | 0.000 | 0.000 | 0.073 | 0.000 |
| decoy | all | 140 | 105 | 0 | — | 0.000 | 0.000 | 0.073 | 0.000 |
| decoy | decompose | 140 | 105 | 0 | — | 0.000 | 0.000 | 0.073 | 0.000 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 87 | 0.0000 (0) [0.000, 0.042] | 0.0230 (2) [0.006, 0.080] | 0.0230 (2) [0.006, 0.080] | 0.0000 (0) [0.000, 0.042] |
| relabel | 0 | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] |
| swap | 103 | 0.0000 (0) [0.000, 0.036] | 0.0194 (2) [0.005, 0.068] | 0.0194 (2) [0.005, 0.068] | 0.0000 (0) [0.000, 0.036] |
| permute | 82 | 0.0000 (0) [0.000, 0.045] | 0.0000 (0) [0.000, 0.045] | 0.0488 (4) [0.019, 0.119] | 0.0000 (0) [0.000, 0.045] |

Shared pool: 330 candidates, 110 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.510 | 0.045 |
| correlate | 0.518 | 0.036 |
| prior | 0.518 | 0.000 |
| lre | 0.526 | 0.009 |
| amie | 0.500 | 0.000 |
| transe | 0.558 | 0.009 |
| rotate | 0.506 | 0.009 |
| complex | 0.573 | 0.109 |

Timings (s): {'setup': 0.18, 'decoders': 41.8, 'propose': 8.48, 'test': 6.44, 'complete-null': 58.65, 'decide': 10.31, 'nulls': 18.12, 'pool-readouts': 263.63, 'amie': 0.59, 'kge': 130.02}

