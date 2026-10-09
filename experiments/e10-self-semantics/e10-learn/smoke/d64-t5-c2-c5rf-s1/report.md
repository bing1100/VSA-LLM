# E10.L erased (SMOKE)

## SmolLM2-360M-full-C5rf-s1

Seen 2845; probes 70; erased 3346 (gold on probes 110; typed coverage 0.900); content relations: is_a, area, purpose, subtype_of, owned_by, depends_on, part_of, measured_by, governed_by, replaces, uses, produces, reports_to, computed_from, reported_in, approved_by, applies_to, describes, sponsored_by, produced_by; 138 concept types.

Decoders: passive out-of-fold cosine 0.364; evidence 330 concepts / 1036 observations (mean cosine 0.253). Primary rule: `holm+decoy`; decoy thresholds {'decompose/0': None, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 140 | 104 | 0 | — | 0.000 | 0.000 | 0.082 | 0.000 |
| holm+decoy | decompose | 140 | 104 | 0 | — | 0.000 | 0.000 | 0.082 | 0.000 |
| holm | all | 140 | 104 | 3 | 0.667 | 0.018 | 0.035 | 0.082 | 0.008 |
| holm | decompose | 140 | 104 | 3 | 0.667 | 0.018 | 0.035 | 0.082 | 0.008 |
| bh | all | 140 | 104 | 6 | 0.333 | 0.018 | 0.034 | 0.082 | 0.031 |
| bh | decompose | 140 | 104 | 6 | 0.333 | 0.018 | 0.034 | 0.082 | 0.031 |
| knockoff | all | 140 | 104 | 0 | — | 0.000 | 0.000 | 0.082 | 0.000 |
| knockoff | decompose | 140 | 104 | 0 | — | 0.000 | 0.000 | 0.082 | 0.000 |
| decoy | all | 140 | 104 | 0 | — | 0.000 | 0.000 | 0.082 | 0.000 |
| decoy | decompose | 140 | 104 | 0 | — | 0.000 | 0.000 | 0.082 | 0.000 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 87 | 0.0000 (0) [0.000, 0.042] | 0.0345 (3) [0.012, 0.097] | 0.0575 (5) [0.025, 0.128] | 0.0000 (0) [0.000, 0.042] |
| relabel | 0 | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] |
| swap | 103 | 0.0000 (0) [0.000, 0.036] | 0.0291 (3) [0.010, 0.082] | 0.0680 (7) [0.033, 0.134] | 0.0000 (0) [0.000, 0.036] |
| permute | 82 | 0.0000 (0) [0.000, 0.045] | 0.0122 (1) [0.002, 0.066] | 0.0122 (1) [0.002, 0.066] | 0.0000 (0) [0.000, 0.045] |

Shared pool: 330 candidates, 110 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.531 | 0.073 |
| correlate | 0.524 | 0.073 |
| prior | 0.518 | 0.000 |
| lre | 0.526 | 0.009 |
| amie | 0.500 | 0.000 |
| transe | 0.559 | 0.009 |
| rotate | 0.506 | 0.009 |
| complex | 0.573 | 0.109 |

Timings (s): {'setup': 0.09, 'decoders': 37.68, 'propose': 6.54, 'test': 10.59, 'complete-null': 61.35, 'decide': 12.61, 'nulls': 24.24, 'pool-readouts': 284.28, 'amie': 0.54, 'kge': 129.42}

