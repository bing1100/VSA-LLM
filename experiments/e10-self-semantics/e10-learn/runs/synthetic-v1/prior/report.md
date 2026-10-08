# E10.L synthetic (CALIBRATION)

## synthetic-s101

Seen 224; probes 132; erased 203 (gold on probes 203; typed coverage 0.833); content relations: is_a, has_attribute, part_of, has_part, member_of, similar_to, located_in; 10 concept types.

Decoders: passive out-of-fold cosine 0.537; evidence 224 concepts / 1792 observations (mean cosine 0.563). Primary rule: `holm+decoy`; decoy thresholds {'closure/0': None, 'closure/1': None, 'decompose/0': 0.19, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 284 | 284 | 31 | 0.742 | 0.113 | 0.197 | 0.488 | 0.043 |
| holm+decoy | closure | 20 | 20 | 0 | — | 0.000 | 0.000 | 0.069 | 0.000 |
| holm+decoy | decompose | 264 | 264 | 31 | 0.742 | 0.113 | 0.197 | 0.419 | 0.045 |
| holm | all | 284 | 284 | 161 | 0.503 | 0.399 | 0.445 | 0.488 | 0.432 |
| holm | closure | 20 | 20 | 6 | 1.000 | 0.030 | 0.057 | 0.069 | 0.000 |
| holm | decompose | 264 | 264 | 155 | 0.484 | 0.369 | 0.419 | 0.419 | 0.447 |
| bh | all | 284 | 284 | 252 | 0.377 | 0.468 | 0.418 | 0.488 | 0.849 |
| bh | closure | 20 | 20 | 11 | 0.909 | 0.049 | 0.093 | 0.069 | 0.167 |
| bh | decompose | 264 | 264 | 241 | 0.353 | 0.419 | 0.383 | 0.419 | 0.872 |
| knockoff | all | 284 | 284 | 230 | 0.365 | 0.414 | 0.388 | 0.488 | 0.789 |
| knockoff | closure | 20 | 20 | 8 | 1.000 | 0.039 | 0.076 | 0.069 | 0.000 |
| knockoff | decompose | 264 | 264 | 222 | 0.342 | 0.374 | 0.358 | 0.419 | 0.816 |
| decoy | all | 284 | 284 | 31 | 0.742 | 0.113 | 0.197 | 0.488 | 0.043 |
| decoy | closure | 20 | 20 | 0 | — | 0.000 | 0.000 | 0.069 | 0.000 |
| decoy | decompose | 264 | 264 | 31 | 0.742 | 0.113 | 0.197 | 0.419 | 0.045 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 270 | 0.0111 (3) [0.004, 0.032] | 0.3074 (83) [0.255, 0.365] | 0.7778 (210) [0.724, 0.823] | 0.0111 (3) [0.004, 0.032] |
| relabel | 95 | 0.0000 (0) [0.000, 0.039] | 0.0842 (8) [0.043, 0.157] | 0.1263 (12) [0.074, 0.208] | 0.0000 (0) [0.000, 0.039] |
| swap | 284 | 0.0035 (1) [0.001, 0.020] | 0.0387 (11) [0.022, 0.068] | 0.1655 (47) [0.127, 0.213] | 0.0035 (1) [0.001, 0.020] |
| permute | 257 | 0.0000 (0) [0.000, 0.015] | 0.0350 (9) [0.019, 0.065] | 0.0856 (22) [0.057, 0.126] | 0.0000 (0) [0.000, 0.015] |
| planted | 264 | 0.0152 (4) [0.006, 0.038] | 0.2689 (71) [0.219, 0.325] | 0.7803 (206) [0.727, 0.826] | 0.0189 (5) [0.008, 0.044] |

Shared pool: 609 candidates, 203 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.886 | 0.670 |
| correlate | 0.886 | 0.675 |
| prior | 0.530 | 0.000 |
| lre | 0.812 | 0.581 |
| amie | 0.635 | 0.271 |
| transe | 0.704 | 0.325 |
| rotate | 0.650 | 0.251 |
| complex | 0.672 | 0.286 |

Timings (s): {'setup': 0.01, 'decoders': 0.03, 'propose': 0.86, 'test': 0.54, 'complete-null': 0.88, 'decide': 0.12, 'nulls': 1.56, 'pool-readouts': 0.39, 'amie': 0.01, 'kge': 23.15}

## synthetic-s202

Seen 224; probes 136; erased 192 (gold on probes 192; typed coverage 0.776); content relations: is_a, has_attribute, part_of, has_part, member_of, similar_to, located_in; 12 concept types.

Decoders: passive out-of-fold cosine 0.530; evidence 224 concepts / 1792 observations (mean cosine 0.547). Primary rule: `holm+decoy`; decoy thresholds {'closure/0': None, 'closure/1': None, 'decompose/0': None, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 290 | 290 | 0 | — | 0.000 | 0.000 | 0.333 | 0.000 |
| holm+decoy | closure | 18 | 18 | 0 | — | 0.000 | 0.000 | 0.052 | 0.000 |
| holm+decoy | decompose | 272 | 272 | 0 | — | 0.000 | 0.000 | 0.281 | 0.000 |
| holm | all | 290 | 290 | 125 | 0.312 | 0.203 | 0.246 | 0.333 | 0.381 |
| holm | closure | 18 | 18 | 4 | 0.750 | 0.016 | 0.031 | 0.052 | 0.125 |
| holm | decompose | 272 | 272 | 121 | 0.298 | 0.188 | 0.230 | 0.281 | 0.390 |
| bh | all | 290 | 290 | 253 | 0.241 | 0.318 | 0.274 | 0.333 | 0.850 |
| bh | closure | 18 | 18 | 11 | 0.727 | 0.042 | 0.079 | 0.052 | 0.375 |
| bh | decompose | 272 | 272 | 242 | 0.219 | 0.276 | 0.244 | 0.281 | 0.867 |
| knockoff | all | 290 | 290 | 204 | 0.284 | 0.302 | 0.293 | 0.333 | 0.646 |
| knockoff | closure | 18 | 18 | 10 | 0.900 | 0.047 | 0.089 | 0.052 | 0.125 |
| knockoff | decompose | 272 | 272 | 194 | 0.253 | 0.255 | 0.254 | 0.281 | 0.665 |
| decoy | all | 290 | 290 | 0 | — | 0.000 | 0.000 | 0.333 | 0.000 |
| decoy | closure | 18 | 18 | 0 | — | 0.000 | 0.000 | 0.052 | 0.000 |
| decoy | decompose | 272 | 272 | 0 | — | 0.000 | 0.000 | 0.281 | 0.000 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 281 | 0.0000 (0) [0.000, 0.013] | 0.3274 (92) [0.275, 0.384] | 0.8470 (238) [0.800, 0.884] | 0.0000 (0) [0.000, 0.013] |
| relabel | 82 | 0.0000 (0) [0.000, 0.045] | 0.0366 (3) [0.013, 0.102] | 0.1220 (10) [0.068, 0.210] | 0.0000 (0) [0.000, 0.045] |
| swap | 290 | 0.0000 (0) [0.000, 0.013] | 0.0448 (13) [0.026, 0.075] | 0.1931 (56) [0.152, 0.242] | 0.0000 (0) [0.000, 0.013] |
| permute | 270 | 0.0000 (0) [0.000, 0.014] | 0.0148 (4) [0.006, 0.037] | 0.1000 (27) [0.070, 0.142] | 0.0000 (0) [0.000, 0.014] |
| planted | 272 | 0.0000 (0) [0.000, 0.014] | 0.3235 (88) [0.271, 0.381] | 0.8566 (233) [0.810, 0.893] | 0.0000 (0) [0.000, 0.014] |

Shared pool: 576 candidates, 192 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.849 | 0.604 |
| correlate | 0.869 | 0.646 |
| prior | 0.497 | 0.016 |
| lre | 0.736 | 0.474 |
| amie | 0.634 | 0.271 |
| transe | 0.697 | 0.354 |
| rotate | 0.660 | 0.286 |
| complex | 0.646 | 0.208 |

Timings (s): {'setup': 0.0, 'decoders': 0.02, 'propose': 0.64, 'test': 0.11, 'complete-null': 0.79, 'decide': 0.12, 'nulls': 1.65, 'pool-readouts': 0.39, 'amie': 0.01, 'kge': 22.72}

## synthetic-s303

Seen 224; probes 124; erased 182 (gold on probes 182; typed coverage 0.758); content relations: is_a, has_attribute, part_of, has_part, member_of, similar_to, located_in; 15 concept types.

Decoders: passive out-of-fold cosine 0.577; evidence 224 concepts / 1792 observations (mean cosine 0.591). Primary rule: `holm+decoy`; decoy thresholds {'closure/0': None, 'closure/1': None, 'decompose/0': None, 'decompose/1': None}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 265 | 265 | 0 | — | 0.000 | 0.000 | 0.401 | 0.000 |
| holm+decoy | closure | 17 | 17 | 0 | — | 0.000 | 0.000 | 0.071 | 0.000 |
| holm+decoy | decompose | 248 | 248 | 0 | — | 0.000 | 0.000 | 0.330 | 0.000 |
| holm | all | 265 | 265 | 108 | 0.380 | 0.225 | 0.283 | 0.401 | 0.349 |
| holm | closure | 17 | 17 | 4 | 1.000 | 0.022 | 0.043 | 0.071 | 0.000 |
| holm | decompose | 248 | 248 | 104 | 0.356 | 0.203 | 0.259 | 0.330 | 0.356 |
| bh | all | 265 | 265 | 221 | 0.308 | 0.374 | 0.337 | 0.401 | 0.797 |
| bh | closure | 17 | 17 | 11 | 1.000 | 0.060 | 0.114 | 0.071 | 0.000 |
| bh | decompose | 248 | 248 | 210 | 0.271 | 0.313 | 0.291 | 0.330 | 0.814 |
| knockoff | all | 265 | 265 | 206 | 0.306 | 0.346 | 0.325 | 0.401 | 0.745 |
| knockoff | closure | 17 | 17 | 12 | 0.833 | 0.055 | 0.103 | 0.071 | 0.500 |
| knockoff | decompose | 248 | 248 | 194 | 0.273 | 0.291 | 0.282 | 0.330 | 0.750 |
| decoy | all | 265 | 265 | 0 | — | 0.000 | 0.000 | 0.401 | 0.000 |
| decoy | closure | 17 | 17 | 0 | — | 0.000 | 0.000 | 0.071 | 0.000 |
| decoy | decompose | 248 | 248 | 0 | — | 0.000 | 0.000 | 0.330 | 0.000 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 254 | 0.0000 (0) [0.000, 0.015] | 0.2795 (71) [0.228, 0.338] | 0.7677 (195) [0.712, 0.815] | 0.0000 (0) [0.000, 0.015] |
| relabel | 97 | 0.0000 (0) [0.000, 0.038] | 0.0825 (8) [0.042, 0.154] | 0.1546 (15) [0.096, 0.240] | 0.0000 (0) [0.000, 0.038] |
| swap | 265 | 0.0000 (0) [0.000, 0.014] | 0.0189 (5) [0.008, 0.043] | 0.0679 (18) [0.043, 0.105] | 0.0000 (0) [0.000, 0.014] |
| permute | 246 | 0.0000 (0) [0.000, 0.015] | 0.0244 (6) [0.011, 0.052] | 0.0610 (15) [0.037, 0.098] | 0.0000 (0) [0.000, 0.015] |
| planted | 248 | 0.0000 (0) [0.000, 0.015] | 0.2661 (66) [0.215, 0.324] | 0.7379 (183) [0.680, 0.789] | 0.0000 (0) [0.000, 0.015] |

Shared pool: 546 candidates, 182 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.851 | 0.555 |
| correlate | 0.856 | 0.495 |
| prior | 0.491 | 0.000 |
| lre | 0.754 | 0.484 |
| amie | 0.637 | 0.275 |
| transe | 0.718 | 0.297 |
| rotate | 0.699 | 0.319 |
| complex | 0.639 | 0.253 |

Timings (s): {'setup': 0.0, 'decoders': 0.02, 'propose': 0.57, 'test': 0.09, 'complete-null': 0.69, 'decide': 0.1, 'nulls': 1.47, 'pool-readouts': 0.37, 'amie': 0.01, 'kge': 24.29}

## Pooled over seeds

Mean [95% t over seeds] for precision / recall; null rates pooled (accepted / tested).

- **holm+decoy** decompose precision 0.742 [—, —], recall 0.038 [-0.125, 0.200], accepted 10.3; nulls: complete 0.0037 (3/805); relabel 0.0000 (0/274); swap 0.0012 (1/839); permute 0.0000 (0/773); planted 0.0051 (4/784)
- **holm** decompose precision 0.379 [0.142, 0.616], recall 0.253 [0.003, 0.504], accepted 126.7; nulls: complete 0.3056 (246/805); relabel 0.0693 (19/274); swap 0.0346 (29/839); permute 0.0246 (19/773); planted 0.2870 (225/784)
- **bh** decompose precision 0.281 [0.114, 0.448], recall 0.336 [0.152, 0.520], accepted 231.0; nulls: complete 0.7988 (643/805); relabel 0.1350 (37/274); swap 0.1442 (121/839); permute 0.0828 (64/773); planted 0.7934 (622/784)
- **knockoff** decompose precision 0.289 [0.173, 0.406], recall 0.307 [0.155, 0.459], accepted 203.3; nulls: 
- **decoy** decompose precision 0.742 [—, —], recall 0.038 [-0.125, 0.200], accepted 10.3; nulls: complete 0.0037 (3/805); relabel 0.0000 (0/274); swap 0.0012 (1/839); permute 0.0000 (0/773); planted 0.0064 (5/784)

Pool (mean over seeds): amie.auc 0.636; amie.recall_at_0.8 0.272; complex.auc 0.653; complex.recall_at_0.8 0.249; correlate.auc 0.870; correlate.recall_at_0.8 0.605; decompose.auc 0.862; decompose.recall_at_0.8 0.610; lre.auc 0.767; lre.recall_at_0.8 0.513; prior.auc 0.506; prior.recall_at_0.8 0.005; rotate.auc 0.669; rotate.recall_at_0.8 0.285; transe.auc 0.707; transe.recall_at_0.8 0.325
