# E10.L erased (SMOKE)

## SmolLM2-360M-full-C5-s1

Seen 2845; probes 85; erased 3346 (gold on probes 141; typed coverage 0.844); content relations: is_a, area, purpose, subtype_of, owned_by, depends_on, part_of, measured_by, governed_by, replaces, uses, produces, reports_to, computed_from, reported_in, approved_by, applies_to, describes, sponsored_by, produced_by; 138 concept types.

Decoders: passive out-of-fold cosine 0.908; evidence 374 concepts / 1149 observations (mean cosine 0.232). Primary rule: `holm+decoy`; decoy thresholds {'decompose/0': -0.07, 'decompose/1': -0.07}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 111 | 85 | 7 | 1.000 | 0.050 | 0.095 | 0.787 | — |
| holm+decoy | decompose | 111 | 85 | 7 | 1.000 | 0.050 | 0.095 | 0.787 | — |
| holm | all | 111 | 85 | 7 | 1.000 | 0.050 | 0.095 | 0.787 | — |
| holm | decompose | 111 | 85 | 7 | 1.000 | 0.050 | 0.095 | 0.787 | — |
| bh | all | 111 | 85 | 18 | 1.000 | 0.128 | 0.226 | 0.787 | — |
| bh | decompose | 111 | 85 | 18 | 1.000 | 0.128 | 0.226 | 0.787 | — |
| knockoff | all | 111 | 85 | 0 | — | 0.000 | 0.000 | 0.787 | — |
| knockoff | decompose | 111 | 85 | 0 | — | 0.000 | 0.000 | 0.787 | — |
| decoy | all | 111 | 85 | 84 | 1.000 | 0.596 | 0.747 | 0.787 | — |
| decoy | decompose | 111 | 85 | 84 | 1.000 | 0.596 | 0.747 | 0.787 | — |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 0 | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] | — (0) [—, —] |
| relabel | 14 | 0.0000 (0) [0.000, 0.215] | 0.0000 (0) [0.000, 0.215] | 0.0000 (0) [0.000, 0.215] | 1.0000 (14) [0.785, 1.000] |
| swap | 84 | 0.0000 (0) [0.000, 0.044] | 0.0000 (0) [0.000, 0.044] | 0.0000 (0) [0.000, 0.044] | 0.9762 (82) [0.917, 0.993] |
| permute | 98 | 0.0204 (2) [0.006, 0.071] | 0.0204 (2) [0.006, 0.071] | 0.0204 (2) [0.006, 0.071] | 0.9694 (95) [0.914, 0.990] |

Shared pool: 423 candidates, 141 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 1.000 | 1.000 |
| correlate | 1.000 | 1.000 |
| prior | 0.496 | 0.000 |

Timings (s): {'setup': 0.03, 'decoders': 0.53, 'propose': 0.72, 'test': 0.07, 'complete-null': 1.27, 'decide': 0.08, 'nulls': 0.46, 'pool-readouts': 0.32}

