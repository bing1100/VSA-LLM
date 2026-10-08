# E10.L erased (SMOKE)

## SmolLM2-360M-full-C5-s1

Seen 2845; probes 29; erased 3346 (gold on probes 43; typed coverage 0.907); content relations: is_a, area, purpose, subtype_of, owned_by, depends_on, part_of, measured_by, governed_by, replaces, uses, produces, reports_to, computed_from, reported_in, approved_by, applies_to, describes, sponsored_by, produced_by; 138 concept types.

Decoders: passive out-of-fold cosine 0.231; evidence 374 concepts / 1149 observations (mean cosine 0.232). Primary rule: `holm+decoy`; decoy thresholds {'decompose/0': None, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 58 | 46 | 0 | — | 0.000 | 0.000 | 0.140 | 0.000 |
| holm+decoy | decompose | 58 | 46 | 0 | — | 0.000 | 0.000 | 0.140 | 0.000 |
| holm | all | 58 | 46 | 4 | 0.500 | 0.047 | 0.085 | 0.140 | 0.038 |
| holm | decompose | 58 | 46 | 4 | 0.500 | 0.047 | 0.085 | 0.140 | 0.038 |
| bh | all | 58 | 46 | 6 | 0.333 | 0.047 | 0.082 | 0.140 | 0.077 |
| bh | decompose | 58 | 46 | 6 | 0.333 | 0.047 | 0.082 | 0.140 | 0.077 |
| knockoff | all | 58 | 46 | 0 | — | 0.000 | 0.000 | 0.140 | 0.000 |
| knockoff | decompose | 58 | 46 | 0 | — | 0.000 | 0.000 | 0.140 | 0.000 |
| decoy | all | 58 | 46 | 0 | — | 0.000 | 0.000 | 0.140 | 0.000 |
| decoy | decompose | 58 | 46 | 0 | — | 0.000 | 0.000 | 0.140 | 0.000 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 49 | 0.0000 (0) [0.000, 0.073] | 0.0204 (1) [0.004, 0.107] | 0.0204 (1) [0.004, 0.107] | 0.0000 (0) [0.000, 0.073] |
| relabel | 6 | 0.0000 (0) [0.000, 0.390] | 0.0000 (0) [0.000, 0.390] | 0.0000 (0) [0.000, 0.390] | 0.0000 (0) [0.000, 0.390] |
| swap | 44 | 0.0000 (0) [0.000, 0.080] | 0.0227 (1) [0.004, 0.118] | 0.0227 (1) [0.004, 0.118] | 0.0000 (0) [0.000, 0.080] |
| permute | 34 | 0.0000 (0) [0.000, 0.102] | 0.0294 (1) [0.005, 0.149] | 0.0294 (1) [0.005, 0.149] | 0.0000 (0) [0.000, 0.102] |

Shared pool: 129 candidates, 43 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.549 | 0.070 |
| correlate | 0.524 | 0.070 |
| prior | 0.515 | 0.000 |
| lre | 0.609 | 0.140 |

Timings (s): {'setup': 0.13, 'decoders': 0.39, 'propose': 0.48, 'test': 0.04, 'complete-null': 1.18, 'decide': 0.11, 'nulls': 0.22, 'pool-readouts': 1.78}

