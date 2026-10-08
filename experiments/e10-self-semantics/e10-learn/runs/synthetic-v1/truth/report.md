# E10.L synthetic (CALIBRATION)

## synthetic-s101

Seen 224; probes 132; erased 203 (gold on probes 203; typed coverage 0.833); content relations: is_a, has_attribute, part_of, has_part, member_of, similar_to, located_in; 10 concept types.

Decoders: passive out-of-fold cosine 0.752; evidence 224 concepts / 1792 observations (mean cosine 0.778). Primary rule: `holm+decoy`; decoy thresholds {'closure/0': None, 'closure/1': None, 'decompose/0': 0.07, 'decompose/1': 0.06}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 272 | 272 | 146 | 0.918 | 0.660 | 0.768 | 0.773 | 0.104 |
| holm+decoy | closure | 8 | 8 | 0 | — | 0.000 | 0.000 | 0.010 | 0.000 |
| holm+decoy | decompose | 264 | 264 | 146 | 0.918 | 0.660 | 0.768 | 0.764 | 0.110 |
| holm | all | 272 | 272 | 151 | 0.914 | 0.680 | 0.780 | 0.773 | 0.113 |
| holm | closure | 8 | 8 | 2 | 1.000 | 0.010 | 0.020 | 0.010 | 0.000 |
| holm | decompose | 264 | 264 | 149 | 0.913 | 0.670 | 0.773 | 0.764 | 0.119 |
| bh | all | 272 | 272 | 215 | 0.721 | 0.764 | 0.742 | 0.773 | 0.522 |
| bh | closure | 8 | 8 | 3 | 0.667 | 0.010 | 0.019 | 0.010 | 0.167 |
| bh | decompose | 264 | 264 | 212 | 0.722 | 0.754 | 0.737 | 0.764 | 0.541 |
| knockoff | all | 272 | 272 | 210 | 0.700 | 0.724 | 0.712 | 0.773 | 0.548 |
| knockoff | closure | 8 | 8 | 4 | 0.500 | 0.010 | 0.019 | 0.010 | 0.333 |
| knockoff | decompose | 264 | 264 | 206 | 0.704 | 0.714 | 0.709 | 0.764 | 0.560 |
| decoy | all | 272 | 272 | 177 | 0.836 | 0.729 | 0.779 | 0.773 | 0.252 |
| decoy | closure | 8 | 8 | 0 | — | 0.000 | 0.000 | 0.010 | 0.000 |
| decoy | decompose | 264 | 264 | 177 | 0.836 | 0.729 | 0.779 | 0.764 | 0.266 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 241 | 0.0124 (3) [0.004, 0.036] | 0.0124 (3) [0.004, 0.036] | 0.2116 (51) [0.165, 0.268] | 0.0498 (12) [0.029, 0.085] |
| relabel | 90 | 0.0778 (7) [0.038, 0.152] | 0.1222 (11) [0.070, 0.206] | 0.1667 (15) [0.104, 0.257] | 0.0889 (8) [0.046, 0.166] |
| swap | 272 | 0.0184 (5) [0.008, 0.042] | 0.0221 (6) [0.010, 0.047] | 0.1213 (33) [0.088, 0.165] | 0.1213 (33) [0.088, 0.165] |
| permute | 256 | 0.0273 (7) [0.013, 0.055] | 0.0273 (7) [0.013, 0.055] | 0.0859 (22) [0.057, 0.127] | 0.1133 (29) [0.080, 0.158] |
| planted | 240 | 0.0042 (1) [0.001, 0.023] | 0.0125 (3) [0.004, 0.036] | 0.0500 (12) [0.029, 0.085] | 0.1000 (24) [0.068, 0.144] |

Shared pool: 609 candidates, 203 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.993 | 0.985 |
| correlate | 0.986 | 0.961 |
| prior | 0.530 | 0.000 |
| lre | 0.812 | 0.581 |
| amie | 0.635 | 0.271 |
| transe | 0.704 | 0.320 |
| rotate | 0.650 | 0.251 |
| complex | 0.672 | 0.286 |

Timings (s): {'setup': 0.01, 'decoders': 0.03, 'propose': 0.6, 'test': 0.47, 'complete-null': 0.65, 'decide': 0.11, 'nulls': 1.51, 'pool-readouts': 0.37, 'amie': 0.01, 'kge': 22.67}

## synthetic-s202

Seen 224; probes 136; erased 192 (gold on probes 192; typed coverage 0.776); content relations: is_a, has_attribute, part_of, has_part, member_of, similar_to, located_in; 12 concept types.

Decoders: passive out-of-fold cosine 0.765; evidence 224 concepts / 1792 observations (mean cosine 0.785). Primary rule: `holm+decoy`; decoy thresholds {'closure/0': None, 'closure/1': None, 'decompose/0': 0.06, 'decompose/1': 0.07}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 283 | 283 | 139 | 0.827 | 0.599 | 0.695 | 0.714 | 0.164 |
| holm+decoy | closure | 11 | 11 | 0 | — | 0.000 | 0.000 | 0.016 | 0.000 |
| holm+decoy | decompose | 272 | 272 | 139 | 0.827 | 0.599 | 0.695 | 0.698 | 0.174 |
| holm | all | 283 | 283 | 144 | 0.819 | 0.615 | 0.702 | 0.714 | 0.178 |
| holm | closure | 11 | 11 | 4 | 0.750 | 0.016 | 0.031 | 0.016 | 0.125 |
| holm | decompose | 272 | 272 | 140 | 0.821 | 0.599 | 0.693 | 0.698 | 0.181 |
| bh | all | 283 | 283 | 229 | 0.585 | 0.698 | 0.637 | 0.714 | 0.651 |
| bh | closure | 11 | 11 | 6 | 0.500 | 0.016 | 0.030 | 0.016 | 0.375 |
| bh | decompose | 272 | 272 | 223 | 0.587 | 0.682 | 0.631 | 0.698 | 0.667 |
| knockoff | all | 283 | 283 | 201 | 0.657 | 0.688 | 0.672 | 0.714 | 0.473 |
| knockoff | closure | 11 | 11 | 8 | 0.375 | 0.016 | 0.030 | 0.016 | 0.625 |
| knockoff | decompose | 272 | 272 | 193 | 0.668 | 0.672 | 0.670 | 0.698 | 0.464 |
| decoy | all | 283 | 283 | 193 | 0.684 | 0.688 | 0.686 | 0.714 | 0.418 |
| decoy | closure | 11 | 11 | 0 | — | 0.000 | 0.000 | 0.016 | 0.000 |
| decoy | decompose | 272 | 272 | 193 | 0.684 | 0.688 | 0.686 | 0.698 | 0.442 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 264 | 0.0189 (5) [0.008, 0.044] | 0.0265 (7) [0.013, 0.054] | 0.1326 (35) [0.097, 0.179] | 0.0341 (9) [0.018, 0.064] |
| relabel | 109 | 0.0092 (1) [0.002, 0.050] | 0.0367 (4) [0.014, 0.091] | 0.1376 (15) [0.085, 0.215] | 0.0550 (6) [0.025, 0.115] |
| swap | 283 | 0.0318 (9) [0.017, 0.059] | 0.0353 (10) [0.019, 0.064] | 0.1201 (34) [0.087, 0.163] | 0.1237 (35) [0.090, 0.167] |
| permute | 270 | 0.0222 (6) [0.010, 0.048] | 0.0259 (7) [0.013, 0.053] | 0.0778 (21) [0.051, 0.116] | 0.0889 (24) [0.060, 0.129] |
| planted | 245 | 0.0082 (2) [0.002, 0.029] | 0.0122 (3) [0.004, 0.035] | 0.0490 (12) [0.028, 0.084] | 0.0816 (20) [0.053, 0.123] |

Shared pool: 576 candidates, 192 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.992 | 0.984 |
| correlate | 0.989 | 0.984 |
| prior | 0.497 | 0.016 |
| lre | 0.736 | 0.474 |
| amie | 0.634 | 0.271 |
| transe | 0.700 | 0.328 |
| rotate | 0.660 | 0.302 |
| complex | 0.646 | 0.208 |

Timings (s): {'setup': 0.0, 'decoders': 0.02, 'propose': 0.62, 'test': 0.1, 'complete-null': 0.72, 'decide': 0.11, 'nulls': 1.58, 'pool-readouts': 0.4, 'amie': 0.01, 'kge': 26.09}

## synthetic-s303

Seen 224; probes 124; erased 182 (gold on probes 182; typed coverage 0.758); content relations: is_a, has_attribute, part_of, has_part, member_of, similar_to, located_in; 15 concept types.

Decoders: passive out-of-fold cosine 0.783; evidence 224 concepts / 1792 observations (mean cosine 0.802). Primary rule: `holm+decoy`; decoy thresholds {'closure/0': None, 'closure/1': None, 'decompose/0': 0.07, 'decompose/1': 0.06}.

| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |
|---|---|---|---|---|---|---|---|---|---|
| holm+decoy | all | 253 | 253 | 123 | 0.846 | 0.571 | 0.682 | 0.676 | 0.146 |
| holm+decoy | closure | 5 | 5 | 0 | — | 0.000 | 0.000 | 0.005 | 0.000 |
| holm+decoy | decompose | 248 | 248 | 123 | 0.846 | 0.571 | 0.682 | 0.670 | 0.151 |
| holm | all | 253 | 253 | 125 | 0.840 | 0.577 | 0.684 | 0.676 | 0.154 |
| holm | closure | 5 | 5 | 1 | 1.000 | 0.005 | 0.011 | 0.005 | 0.000 |
| holm | decompose | 248 | 248 | 124 | 0.839 | 0.571 | 0.680 | 0.670 | 0.159 |
| bh | all | 253 | 253 | 199 | 0.603 | 0.659 | 0.630 | 0.676 | 0.608 |
| bh | closure | 5 | 5 | 2 | 0.500 | 0.005 | 0.011 | 0.005 | 0.250 |
| bh | decompose | 248 | 248 | 197 | 0.604 | 0.654 | 0.628 | 0.670 | 0.619 |
| knockoff | all | 253 | 253 | 185 | 0.632 | 0.643 | 0.638 | 0.676 | 0.523 |
| knockoff | closure | 5 | 5 | 1 | 0.000 | 0.000 | 0.000 | 0.005 | 0.250 |
| knockoff | decompose | 248 | 248 | 184 | 0.636 | 0.643 | 0.639 | 0.670 | 0.532 |
| decoy | all | 253 | 253 | 165 | 0.727 | 0.659 | 0.692 | 0.676 | 0.346 |
| decoy | closure | 5 | 5 | 0 | — | 0.000 | 0.000 | 0.005 | 0.000 |
| decoy | decompose | 248 | 248 | 165 | 0.727 | 0.659 | 0.692 | 0.670 | 0.357 |

Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the real run's thresholds:

| null world | tested | holm+decoy | holm | bh | decoy |
|---|---|---|---|---|---|
| complete | 236 | 0.0212 (5) [0.009, 0.049] | 0.0424 (10) [0.023, 0.076] | 0.1907 (45) [0.146, 0.246] | 0.0339 (8) [0.017, 0.065] |
| relabel | 101 | 0.0693 (7) [0.034, 0.136] | 0.0891 (9) [0.048, 0.161] | 0.2475 (25) [0.174, 0.340] | 0.1485 (15) [0.092, 0.231] |
| swap | 253 | 0.0237 (6) [0.011, 0.051] | 0.0237 (6) [0.011, 0.051] | 0.1621 (41) [0.122, 0.212] | 0.1146 (29) [0.081, 0.160] |
| permute | 247 | 0.0283 (7) [0.014, 0.057] | 0.0364 (9) [0.019, 0.068] | 0.1377 (34) [0.100, 0.186] | 0.1296 (32) [0.093, 0.177] |
| planted | 217 | 0.0230 (5) [0.010, 0.053] | 0.0323 (7) [0.016, 0.065] | 0.1014 (22) [0.068, 0.149] | 0.0691 (15) [0.042, 0.111] |

Shared pool: 546 candidates, 182 erased edges.

| method | AUC | recall at precision 0.8 |
|---|---|---|
| decompose | 0.984 | 0.956 |
| correlate | 0.977 | 0.945 |
| prior | 0.491 | 0.000 |
| lre | 0.754 | 0.484 |
| amie | 0.637 | 0.275 |
| transe | 0.717 | 0.286 |
| rotate | 0.701 | 0.313 |
| complex | 0.639 | 0.253 |

Timings (s): {'setup': 0.0, 'decoders': 0.02, 'propose': 0.56, 'test': 0.09, 'complete-null': 0.63, 'decide': 0.1, 'nulls': 1.4, 'pool-readouts': 0.39, 'amie': 0.01, 'kge': 30.54}

## Pooled over seeds

Mean [95% t over seeds] for precision / recall; null rates pooled (accepted / tested).

- **holm+decoy** decompose precision 0.864 [0.745, 0.982], recall 0.610 [0.497, 0.723], accepted 136.0; nulls: complete 0.0175 (13/741); relabel 0.0500 (15/300); swap 0.0248 (20/808); permute 0.0259 (20/773); planted 0.0114 (8/702)
- **holm** decompose precision 0.858 [0.737, 0.978], recall 0.613 [0.487, 0.740], accepted 137.7; nulls: complete 0.0270 (20/741); relabel 0.0800 (24/300); swap 0.0272 (22/808); permute 0.0298 (23/773); planted 0.0185 (13/702)
- **bh** decompose precision 0.638 [0.456, 0.820], recall 0.697 [0.569, 0.824], accepted 210.7; nulls: complete 0.1768 (131/741); relabel 0.1833 (55/300); swap 0.1337 (108/808); permute 0.0996 (77/773); planted 0.0655 (46/702)
- **knockoff** decompose precision 0.669 [0.585, 0.754], recall 0.676 [0.587, 0.766], accepted 194.3; nulls: 
- **decoy** decompose precision 0.749 [0.554, 0.944], recall 0.692 [0.605, 0.779], accepted 178.3; nulls: complete 0.0391 (29/741); relabel 0.0967 (29/300); swap 0.1200 (97/808); permute 0.1100 (85/773); planted 0.0840 (59/702)

Pool (mean over seeds): amie.auc 0.636; amie.recall_at_0.8 0.272; complex.auc 0.653; complex.recall_at_0.8 0.249; correlate.auc 0.984; correlate.recall_at_0.8 0.963; decompose.auc 0.989; decompose.recall_at_0.8 0.975; lre.auc 0.767; lre.recall_at_0.8 0.513; prior.auc 0.506; prior.recall_at_0.8 0.005; rotate.auc 0.670; rotate.recall_at_0.8 0.289; transe.auc 0.707; transe.recall_at_0.8 0.311
