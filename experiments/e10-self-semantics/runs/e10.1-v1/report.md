# E10.1 — self-learned semantics (H-H)

Seeds [11, 22, 33]; CPU single-threaded jobs; world `wordnet`. Means over seeds with 95% t-intervals in brackets. Gold labels and audit observations are used only by evaluation; every learning-side decision uses held-out self-tests.

## H-H refutation clauses

| Clause | Result |
|---|---|
| (i) erased edges recovered above matched random candidates | {'0.1': False, '0.3': False, '0.5': True, '0.7': False} |
| (ii) blank slots align with hidden relations above chance (ARI − permuted, additive) | {'absent': False, 'collapsed': True} |
| (ii′) same, per-seed permutation test (Holm over seeds; added after the first run, reported alongside) | {'absent': True, 'collapsed': True} |
| (iii) self-acceptance beats random acceptance at the matched rate (edges) | {'erasure-0.3': False, 'erasure-0.5': False} |
| (iii) self-acceptance beats random acceptance at the matched rate (slots, pooled) | False |
| (iv) new-word frame inference beats nearest-neighbour frames (F1) | {'1': False, '2': False, '4': False, '6': False} |

## (a) Learnability ablation (30% erased, 5% spurious prior edges, noisy priors)

Contrasts against everything learnable with L2-to-prior (paired over seeds):

| Contrast | Δ held-out-obs fit (train concepts) | Δ test-concept fit (zero-shot) | Δ recovery F1 | Δ recovery AUC |
|---|---|---|---|---|
| atomics=fixed − all l2 | -0.1177 [-0.1260, -0.1093] | 0.0219 [-0.0798, 0.1235] | 0.032 [-0.040, 0.103] | -0.064 [-0.171, 0.044] |
| atomics=free − all l2 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| only atomics l2 − all fixed | 0.3597 [0.3417, 0.3777] | -0.0423 [-0.0714, -0.0132] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| relations=fixed − all l2 | -0.0044 [-0.0162, 0.0073] | 0.0017 [-0.0290, 0.0323] | 0.015 [-0.068, 0.099] | -0.003 [-0.042, 0.035] |
| relations=free − all l2 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| only relations l2 − all fixed | 0.1551 [0.1387, 0.1716] | -0.0188 [-0.0252, -0.0123] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| mapping=fixed − all l2 | -0.0275 [-0.0371, -0.0179] | -0.0160 [-0.0586, 0.0267] | -0.087 [-0.133, -0.040] | -0.023 [-0.061, 0.016] |
| mapping=free − all l2 | -0.0013 [-0.0055, 0.0029] | -0.0018 [-0.0755, 0.0720] | -0.262 [-0.321, -0.203] | -0.032 [-0.072, 0.008] |
| only mapping l2 − all fixed | 0.3102 [0.2817, 0.3386] | -0.0220 [-0.0405, -0.0034] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| frames=fixed − all l2 | -0.0438 [-0.0560, -0.0316] | 0.0063 [-0.0256, 0.0382] | -0.315 [-0.389, -0.242] | -0.083 [-0.165, 0.000] |
| frames=free − all l2 | -0.0025 [-0.0061, 0.0011] | 0.0082 [-0.0154, 0.0317] | 0.097 [0.022, 0.172] | 0.032 [0.003, 0.062] |
| only frames l2 − all fixed | 0.0876 [0.0842, 0.0911] | -0.0056 [-0.0533, 0.0422] | 0.375 [0.324, 0.425] | 0.012 [-0.024, 0.048] |
| all free − all l2 | -0.0008 [-0.0031, 0.0014] | 0.0096 [-0.0633, 0.0825] | -0.022 [-0.086, 0.042] | 0.002 [-0.052, 0.056] |
| all fixed − all l2 | -0.4877 [-0.5127, -0.4628] | 0.0551 [-0.0059, 0.1161] | -0.315 [-0.389, -0.242] | -0.083 [-0.165, 0.000] |

Selected grid cells:

| atomics / relations / mapping / frames | val fit | test fit | recovery F1 | AUC | atomic cos | spurious removed |
|---|---:|---:|---:|---:|---:|---:|
| l2 / l2 / l2 / l2 | 0.7350 [0.7322, 0.7377] | 0.0941 [0.0458, 0.1424] | 0.315 [0.242, 0.389] | 0.583 [0.500, 0.665] | n/a | 0.000 [0.000, 0.000] |
| free / free / free / free | 0.7341 [0.7300, 0.7382] | 0.1037 [0.0660, 0.1414] | 0.293 [0.204, 0.382] | 0.584 [0.535, 0.634] | n/a | 0.108 [0.041, 0.176] |
| fixed / fixed / fixed / fixed | 0.2472 [0.2241, 0.2703] | 0.1492 [0.1312, 0.1672] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | n/a | 0.000 [0.000, 0.000] |
| fixed / fixed / fixed / l2 | 0.3348 [0.3135, 0.3561] | 0.1437 [0.0927, 0.1946] | 0.375 [0.324, 0.425] | 0.512 [0.476, 0.548] | n/a | 0.000 [0.000, 0.000] |
| l2 / l2 / l2 / fixed | 0.6912 [0.6765, 0.7059] | 0.1004 [0.0574, 0.1434] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | n/a | 0.001 [-0.003, 0.004] |
| fixed / l2 / l2 / l2 | 0.6173 [0.6116, 0.6229] | 0.1160 [0.0527, 0.1792] | 0.347 [0.320, 0.374] | 0.519 [0.488, 0.550] | n/a | 0.023 [0.010, 0.037] |
| l2 / fixed / l2 / l2 | 0.7305 [0.7198, 0.7412] | 0.0958 [0.0603, 0.1312] | 0.330 [0.269, 0.392] | 0.579 [0.516, 0.643] | n/a | 0.000 [0.000, 0.000] |
| l2 / l2 / fixed / l2 | 0.7075 [0.6964, 0.7185] | 0.0781 [0.0575, 0.0987] | 0.229 [0.187, 0.270] | 0.560 [0.509, 0.611] | n/a | 0.000 [0.000, 0.000] |
| l2 / l2 / l2 / free | 0.7324 [0.7291, 0.7358] | 0.1023 [0.0393, 0.1653] | 0.412 [0.318, 0.507] | 0.615 [0.513, 0.717] | n/a | 0.000 [0.000, 0.000] |

Best test-concept fit in the full 3⁴ grid: `fixed|fixed|fixed|fixed`.

## (b) Erasure & recovery

| Erased (principal fixed) | Precision | Recall | F1 | AUC | R-precision | Random (prevalence / AUC 0.5) | Filler-frequency AUC / R-prec | Test fit |
|---|---|---|---|---|---|---|---|---|
| 10% (90%) | 0.441 [0.313, 0.569] | 0.308 [0.222, 0.395] | 0.362 [0.265, 0.460] | 0.608 [0.476, 0.740] | 0.448 [0.294, 0.603] | 0.333 [0.333, 0.333] / 0.5 | 0.399 [0.300, 0.497] / 0.329 [0.224, 0.433] | 0.1137 [0.0694, 0.1580] |
| 30% (70%) | 0.442 [0.309, 0.576] | 0.268 [0.184, 0.352] | 0.334 [0.234, 0.433] | 0.589 [0.527, 0.651] | 0.428 [0.333, 0.522] | 0.333 [0.333, 0.334] / 0.5 | 0.429 [0.408, 0.450] / 0.315 [0.297, 0.334] | 0.0979 [0.0672, 0.1286] |
| 50% (50%) | 0.408 [0.392, 0.423] | 0.155 [0.116, 0.193] | 0.224 [0.183, 0.265] | 0.561 [0.545, 0.577] | 0.403 [0.384, 0.422] | 0.333 [0.333, 0.333] / 0.5 | 0.453 [0.430, 0.475] / 0.297 [0.290, 0.304] | 0.1168 [0.0625, 0.1710] |
| 70% (30%) | 0.373 [0.271, 0.475] | 0.062 [0.050, 0.074] | 0.107 [0.088, 0.126] | 0.530 [0.511, 0.549] | 0.367 [0.342, 0.393] | 0.333 [0.333, 0.334] / 0.5 | 0.475 [0.435, 0.515] / 0.287 [0.230, 0.344] | 0.0894 [0.0395, 0.1392] |

## (c) Blank-relation discovery — hidden relations absent

| Method | ARI (gold edges) | ARI − permuted | permutation p (Holm, max over seeds) | mean best Jaccard | detection P | detection R | accepted / proposed slots | test fit |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.502 [0.360, 0.645] | 0.502 [0.360, 0.644] | 0.015 | 0.495 [0.382, 0.609] | 0.814 [0.735, 0.892] | 0.648 [0.555, 0.742] | n/a / n/a | 0.1058 [0.0739, 0.1377] |
| additive | 0.036 [-0.010, 0.081] | 0.036 [-0.010, 0.082] | 0.020 | 0.164 [0.144, 0.184] | 0.584 [0.520, 0.648] | 0.562 [0.516, 0.608] | 1.3 [-0.1, 2.8] / 2.3 [0.9, 3.8] | 0.1108 [0.0536, 0.1680] |
| additive_norule | 0.036 [-0.010, 0.081] | 0.036 [-0.010, 0.082] | 0.020 | 0.164 [0.144, 0.184] | 0.584 [0.520, 0.648] | 0.562 [0.516, 0.608] | 1.3 [-0.1, 2.8] / 2.3 [0.9, 3.8] | 0.1108 [0.0536, 0.1680] |
| all_at_once | -0.010 [-0.035, 0.015] | -0.011 [-0.035, 0.013] | 1.000 | 0.104 [0.079, 0.129] | 0.618 [0.579, 0.657] | 0.424 [0.342, 0.505] | 5.0 [5.0, 5.0] / 5.0 [5.0, 5.0] | 0.1017 [0.0490, 0.1545] |
| m3 | 0.031 [-0.055, 0.116] | 0.031 [-0.055, 0.117] | 0.095 | 0.155 [0.110, 0.200] | 0.592 [0.520, 0.663] | 0.540 [0.457, 0.623] | n/a / n/a | 0.1032 [0.0624, 0.1440] |
| stem_cell | -0.002 [-0.019, 0.015] | -0.003 [-0.020, 0.014] | 1.000 | 0.099 [0.083, 0.115] | 0.590 [0.510, 0.670] | 0.455 [0.380, 0.530] | n/a / n/a | 0.1113 [0.0727, 0.1499] |

Per hidden relation, best Jaccard of a discovered cluster (candidate level):

| Method | similar_to | antonym | instance_hypernym |
|---|---|---|---|
| oracle | 0.753 [0.665, 0.842] | 0.412 [0.328, 0.496] | 0.321 [0.132, 0.509] |
| additive | 0.302 [0.232, 0.373] | 0.142 [0.129, 0.155] | 0.047 [0.025, 0.070] |
| additive_norule | 0.302 [0.232, 0.373] | 0.142 [0.129, 0.155] | 0.047 [0.025, 0.070] |
| all_at_once | 0.148 [0.094, 0.202] | 0.091 [0.059, 0.122] | 0.073 [0.021, 0.125] |
| m3 | 0.276 [0.099, 0.454] | 0.132 [0.077, 0.187] | 0.057 [-0.015, 0.130] |
| stem_cell | 0.140 [0.080, 0.199] | 0.092 [0.056, 0.128] | 0.066 [0.027, 0.104] |

Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive: instance_hypernym 0.044 [0.022, 0.067]; antonym 0.112 [0.098, 0.126]; similar_to 0.219 [0.169, 0.270].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive_norule: instance_hypernym 0.044 [0.022, 0.067]; antonym 0.112 [0.098, 0.126]; similar_to 0.219 [0.169, 0.270].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), all_at_once: instance_hypernym 0.055 [0.013, 0.096]; antonym 0.055 [0.033, 0.077]; similar_to 0.088 [0.048, 0.127].

- additive − all_at_once: ARI 0.045 [-0.021, 0.112], mean best Jaccard 0.060 [0.021, 0.098], test fit 0.0090 [-0.0072, 0.0253]
- additive − m3: ARI 0.005 [-0.060, 0.070], mean best Jaccard 0.009 [-0.018, 0.035], test fit 0.0076 [-0.0645, 0.0796]
- additive − stem_cell: ARI 0.038 [-0.025, 0.100], mean best Jaccard 0.065 [0.037, 0.092], test fit -0.0006 [-0.0284, 0.0273]
- additive − additive_norule: n/a
- rule enforcement: relation recovery (additive − additive_norule): 0.000 [0.000, 0.000]

**Riddle-style interpretation (E10.6, absent).** 19 accepted slots, 1 adopted a structural rule; 1 of those rules are true of the matched hidden relation, 0 are its designed property. Rule predictions' gold precision n/a. Hypotheses scored per slot (full space): 92.5.

| Matched hidden relation | slots | adopted | true | designed | rules adopted |
|---|---:|---:|---:|---:|---|
| similar_to | 13 | 0 | 0 | 0 |  |
| antonym | 4 | 1 | 1 | 0 | functional×1 |
| instance_hypernym | 2 | 0 | 0 | 0 |  |

Accuracy of the adopted hypothesis vs number of candidate hypotheses (one true + m−1 false):

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| slots | 19 | 19 | 19 | 19 | 19 | 19 |

**Human-likeness (additive, absent).** Over 5 slots: purity of a slot's members w.r.t. its final hidden relation rises from 0.207 [-0.028, 0.441] (first committed members) to 0.516 [0.171, 0.861]; hidden relations present 2.00 [0.24, 3.76] → 2.20 [0.84, 3.56]; members 219.8 [-28.8, 468.4] (peak 220.0 [-28.2, 468.2]) → 184.4 [-23.5, 392.3]. Slots that overextended then refined: 4.

Soft hypothesis of a newly opened slot (mass-weighted shares of the pairs it holds; steps since opening):

| step | target relation | other hidden relations | distractors | effective size |
|---:|---:|---:|---:|---:|
| 0 | 0.23 | 0.24 | 0.53 | 472 |
| 25 | 0.24 | 0.25 | 0.51 | 345 |
| 50 | 0.22 | 0.28 | 0.50 | 309 |
| 100 | 0.20 | 0.30 | 0.50 | 278 |
| 200 | 0.20 | 0.31 | 0.49 | 245 |
| 400 | 0.24 | 0.28 | 0.48 | 214 |

Slot self-acceptance (absent; additive, no-rule and all-at-once runs pooled): accuracy 0.345 (Wilson 0.20–0.53, n = 29) vs random at the matched rate 0.369 (p = 0.684); accept-all 0.276; wrong-acceptance rate 0.739.

## (c) Blank-relation discovery — hidden relations collapsed

| Method | ARI (gold edges) | ARI − permuted | permutation p (Holm, max over seeds) | mean best Jaccard | detection P | detection R | accepted / proposed slots | test fit |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.953 [0.939, 0.967] | 0.953 [0.938, 0.969] | 0.015 | 0.992 [0.990, 0.995] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a / n/a | 0.1361 [0.0405, 0.2316] |
| additive | 0.330 [0.251, 0.409] | 0.330 [0.250, 0.410] | 0.015 | 0.259 [0.239, 0.279] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.0 [1.0, 1.0] / 2.0 [2.0, 2.0] | 0.1384 [0.0465, 0.2302] |
| additive_norule | 0.330 [0.251, 0.409] | 0.330 [0.250, 0.410] | 0.015 | 0.259 [0.239, 0.279] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.0 [1.0, 1.0] / 2.0 [2.0, 2.0] | 0.1384 [0.0465, 0.2302] |
| all_at_once | 0.018 [-0.019, 0.055] | 0.018 [-0.019, 0.055] | 0.139 | 0.099 [0.089, 0.108] | 1.000 [1.000, 1.000] | 0.772 [0.711, 0.833] | 5.0 [5.0, 5.0] / 5.0 [5.0, 5.0] | 0.1370 [0.0489, 0.2251] |
| m3 | 0.623 [0.595, 0.652] | 0.623 [0.595, 0.652] | 0.015 | 0.328 [0.326, 0.330] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a / n/a | 0.1074 [0.0726, 0.1422] |
| stem_cell | 0.037 [0.011, 0.062] | 0.037 [0.012, 0.062] | 0.015 | 0.141 [0.103, 0.180] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a / n/a | 0.1443 [0.0654, 0.2232] |

Per hidden relation, best Jaccard of a discovered cluster (candidate level):

| Method | substance_holonym | part_holonym | part_meronym | substance_meronym | member_meronym | member_holonym |
|---|---|---|---|---|---|---|
| oracle | 1.000 [1.000, 1.000] | 0.954 [0.938, 0.971] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| additive | 0.070 [0.047, 0.093] | 0.496 [0.477, 0.516] | 0.624 [0.536, 0.711] | 0.060 [0.027, 0.093] | 0.080 [0.050, 0.109] | 0.224 [0.180, 0.268] |
| additive_norule | 0.070 [0.047, 0.093] | 0.496 [0.477, 0.516] | 0.624 [0.536, 0.711] | 0.060 [0.027, 0.093] | 0.080 [0.050, 0.109] | 0.224 [0.180, 0.268] |
| all_at_once | 0.046 [0.036, 0.057] | 0.148 [0.089, 0.207] | 0.142 [0.129, 0.154] | 0.057 [0.032, 0.082] | 0.091 [0.054, 0.128] | 0.107 [0.042, 0.172] |
| m3 | 0.078 [0.068, 0.088] | 0.657 [0.624, 0.690] | 0.860 [0.828, 0.892] | 0.054 [0.045, 0.064] | 0.086 [0.061, 0.110] | 0.234 [0.213, 0.254] |
| stem_cell | 0.111 [0.011, 0.210] | 0.198 [0.116, 0.279] | 0.269 [0.256, 0.282] | 0.061 [0.027, 0.096] | 0.073 [0.047, 0.099] | 0.136 [0.052, 0.220] |

Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive: part_meronym 0.234 [0.137, 0.330]; member_meronym 0.053 [0.022, 0.084]; substance_meronym 0.022 [-0.013, 0.057]; part_holonym 0.191 [0.167, 0.215]; member_holonym 0.074 [0.031, 0.118]; substance_holonym 0.038 [0.019, 0.056].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive_norule: part_meronym 0.234 [0.137, 0.330]; member_meronym 0.053 [0.022, 0.084]; substance_meronym 0.022 [-0.013, 0.057]; part_holonym 0.191 [0.167, 0.215]; member_holonym 0.074 [0.031, 0.118]; substance_holonym 0.038 [0.019, 0.056].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), all_at_once: part_meronym 0.142 [0.129, 0.154]; member_meronym 0.075 [0.019, 0.130]; substance_meronym 0.045 [0.010, 0.079]; part_holonym 0.141 [0.061, 0.220]; member_holonym 0.090 [0.047, 0.133]; substance_holonym 0.046 [0.036, 0.057].

- additive − all_at_once: ARI 0.312 [0.261, 0.363], mean best Jaccard 0.160 [0.140, 0.180], test fit 0.0014 [-0.0048, 0.0075]
- additive − m3: ARI -0.293 [-0.401, -0.186], mean best Jaccard -0.069 [-0.089, -0.049], test fit 0.0310 [-0.0275, 0.0894]
- additive − stem_cell: ARI 0.293 [0.234, 0.352], mean best Jaccard 0.118 [0.093, 0.142], test fit -0.0059 [-0.0236, 0.0117]
- additive − additive_norule: n/a
- rule enforcement: relation recovery (additive − additive_norule): 0.000 [0.000, 0.000]

**Riddle-style interpretation (E10.6, collapsed).** 18 accepted slots, 6 adopted a structural rule; 3 of those rules are true of the matched hidden relation, 0 are its designed property. Rule predictions' gold precision n/a. Hypotheses scored per slot (full space): 110.4.

| Matched hidden relation | slots | adopted | true | designed | rules adopted |
|---|---:|---:|---:|---:|---|
| part_meronym | 13 | 4 | 3 | 0 | antisymmetric×3, sub_relation_of:meronym×1 |
| member_holonym | 1 | 0 | 0 | 0 |  |
| part_holonym | 4 | 2 | 0 | 0 | sub_relation_of:holonym×2 |

Accuracy of the adopted hypothesis vs number of candidate hypotheses (one true + m−1 false):

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| slots | 7 | 7 | 7 | 7 | 7 | 7 |

**Human-likeness (additive, collapsed).** Over 3 slots: purity of a slot's members w.r.t. its final hidden relation rises from 0.449 [0.219, 0.678] (first committed members) to 0.451 [0.225, 0.677]; hidden relations present 6.00 [6.00, 6.00] → 6.00 [6.00, 6.00]; members 208.0 [167.0, 249.0] (peak 208.0 [167.0, 249.0]) → 160.0 [99.3, 220.7]. Slots that overextended then refined: 0.

Soft hypothesis of a newly opened slot (mass-weighted shares of the pairs it holds; steps since opening):

| step | target relation | other hidden relations | distractors | effective size |
|---:|---:|---:|---:|---:|
| 0 | 0.43 | 0.57 | 0.00 | 511 |
| 25 | 0.43 | 0.57 | 0.00 | 501 |
| 50 | 0.43 | 0.57 | 0.00 | 487 |
| 100 | 0.43 | 0.57 | 0.00 | 424 |
| 200 | 0.43 | 0.57 | 0.00 | 306 |
| 400 | 0.43 | 0.57 | 0.00 | 257 |

Slot self-acceptance (collapsed; additive, no-rule and all-at-once runs pooled): accuracy 0.222 (Wilson 0.11–0.41, n = 27) vs random at the matched rate 0.222 (p = 0.574); accept-all 0.000; wrong-acceptance rate 1.000.

## (d) Self-tested acceptance of edge hypotheses (held-out observations; audit never read)

| Source | n | accept rate | accuracy (Wilson) | random @ matched rate | accept-all | in-sample (confirmation-bias control) | acc − random by seed | audit Δ accepted / rejected |
|---|---:|---:|---|---:|---:|---|---|---|
| erasure-0.3 | 2340 | 0.79 | 0.459 (0.44–0.48) | 0.449 | 0.411 | 0.463 (accepts 0.70) | 0.010 [-0.035, 0.056] | +0.3293 / +0.1053 |
| erasure-0.5 | 3697 | 0.81 | 0.431 (0.42–0.45) | 0.433 | 0.394 | 0.436 (accepts 0.76) | -0.002 [-0.019, 0.014] | +0.3259 / +0.1282 |

## (e) New-word frame inference (fast mapping)

| k | discovered F1 | known-only F1 | oracle-dictionary F1 | nearest-neighbour F1 | random F1 | discovered fit | NN fit | context-mean fit | gold-frame fit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.001 [-0.002, 0.004] | 0.001 [-0.002, 0.004] | n/a | 0.249 [0.246, 0.252] | 0.000 [0.000, 0.000] | 0.4343 [0.3990, 0.4697] | 0.3969 [0.2924, 0.5015] | 0.5142 [0.4793, 0.5491] | 0.1194 [0.0404, 0.1984] |
| 2 | 0.003 [0.002, 0.003] | 0.003 [0.002, 0.003] | n/a | 0.268 [0.230, 0.306] | 0.000 [0.000, 0.000] | 0.5004 [0.4640, 0.5368] | 0.4201 [0.3227, 0.5175] | 0.5936 [0.5589, 0.6282] | 0.1194 [0.0404, 0.1984] |
| 4 | 0.007 [-0.000, 0.013] | 0.007 [-0.000, 0.014] | n/a | 0.267 [0.221, 0.314] | 0.000 [0.000, 0.000] | 0.5506 [0.5270, 0.5741] | 0.4392 [0.3668, 0.5116] | 0.6543 [0.6345, 0.6742] | 0.1194 [0.0404, 0.1984] |
| 6 | 0.006 [0.000, 0.012] | 0.007 [0.001, 0.013] | n/a | 0.274 [0.222, 0.325] | 0.000 [0.000, 0.000] | 0.5632 [0.5412, 0.5852] | 0.4559 [0.3471, 0.5646] | 0.6669 [0.6464, 0.6874] | 0.1194 [0.0404, 0.1984] |

- k = 1: discovered − nearest_neighbour (F1) -0.248 [-0.252, -0.244]; discovered − known_only (F1) -0.000 [-0.002, 0.001]; discovered − random (F1) 0.001 [-0.002, 0.004]; discovered − nearest_neighbour (fit) 0.037 [-0.044, 0.118]; discovered − context_mean (fit) -0.080 [-0.085, -0.075]
- k = 2: discovered − nearest_neighbour (F1) -0.265 [-0.302, -0.227]; discovered − known_only (F1) 0.000 [0.000, 0.000]; discovered − random (F1) 0.003 [0.002, 0.003]; discovered − nearest_neighbour (fit) 0.080 [-0.017, 0.178]; discovered − context_mean (fit) -0.093 [-0.099, -0.088]
- k = 4: discovered − nearest_neighbour (F1) -0.261 [-0.303, -0.218]; discovered − known_only (F1) -0.000 [-0.002, 0.001]; discovered − random (F1) 0.007 [-0.000, 0.013]; discovered − nearest_neighbour (fit) 0.111 [0.055, 0.167]; discovered − context_mean (fit) -0.104 [-0.108, -0.100]
- k = 6: discovered − nearest_neighbour (F1) -0.267 [-0.322, -0.212]; discovered − known_only (F1) -0.000 [-0.002, 0.001]; discovered − random (F1) 0.006 [0.000, 0.012]; discovered − nearest_neighbour (fit) 0.107 [0.013, 0.202]; discovered − context_mean (fit) -0.104 [-0.111, -0.097]

Discovered-relation F1 by hierarchy depth at the largest k (exploratory 'basic-level' readout): depth 0: 0.003 [-0.010, 0.015]; depth 1: 0.015 [-0.050, 0.080]; depth 10: 0.019 [-0.062, 0.100]; depth 11: 0.000 [0.000, 0.000]; depth 12: 0.000 [0.000, 0.000]; depth 13: 0.000 [0.000, 0.000]; depth 16: 0.000; depth 2: 0.000 [0.000, 0.000]; depth 3: 0.022 [-0.036, 0.081]; depth 4: 0.000 [0.000, 0.000]; depth 5: 0.000 [0.000, 0.000]; depth 6: 0.002 [-0.006, 0.010]; depth 7: 0.000 [0.000, 0.000]; depth 8: 0.012 [-0.017, 0.042]; depth 9: 0.024 [-0.020, 0.067].

## E10.4 Seed ontology → recovered logical ontology

| Start | edge F1 (all framed) | P / R | candidate AUC | candidate-level relation Jaccard (mean) | relations recovered (J ≥ 0.5, of 7) | accepted slots | multi-hop is-a acc (gold) | transitivity located_in (gold) | inverse part (gold) | symmetry similar (gold) | test fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| core | 0.156 [0.147, 0.165] | 0.32 [0.31, 0.33] / 0.10 [0.10, 0.11] | 0.521 [0.486, 0.557] | 0.009 [0.006, 0.011] | 0.0 [0.0, 0.0] | 2.3 [0.9, 3.8] | 0.61 [0.58, 0.64] (0.95 [0.94, 0.96]) | n/a (n/a) | 0.00 [0.00, 0.00] (0.01 [0.00, 0.02]) | n/a (0.03 [0.02, 0.05]) | 0.0171 [0.0074, 0.0269] |
| curated30 | 0.445 [0.441, 0.448] | 0.77 [0.75, 0.78] / 0.31 [0.31, 0.32] | 0.536 [0.519, 0.553] | 0.019 [0.016, 0.022] | 0.0 [0.0, 0.0] | 0.7 [-0.8, 2.1] | 0.55 [0.54, 0.56] (0.95 [0.94, 0.96]) | n/a (n/a) | 0.00 [-0.00, 0.01] (0.01 [0.00, 0.02]) | 0.01 [-0.02, 0.03] (0.03 [0.02, 0.05]) | 0.0548 [-0.0139, 0.1236] |
| full | 1.000 [0.999, 1.001] | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | n/a | n/a | 16.0 [16.0, 16.0] | 0.0 [0.0, 0.0] | 0.95 [0.94, 0.96] (0.95 [0.94, 0.96]) | n/a (n/a) | 0.01 [0.00, 0.02] (0.01 [0.00, 0.02]) | 0.03 [0.02, 0.05] (0.03 [0.02, 0.05]) | 0.1239 [0.0580, 0.1897] |

## E10.5 Dreaming (offline self-revision) after injected corruptions

| Dream every (steps) | passes | wrong edges repaired | correct edges damaged | merged relations split (ARI) | wrong slot repaired | wrong slot reopened/removed | correct slots disturbed | audit fit (train concepts) | applied revisions |
|---|---:|---|---|---|---|---|---|---|---|
| off | 0 | 0.00 [-0.00, 0.01] | 0.000 [-0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.6204 [0.6102, 0.6306] | {} |
| 300 | 2 | 0.12 [0.10, 0.13] | 0.115 [0.104, 0.126] | -0.04 [-0.05, -0.03] | 0.08 [-0.03, 0.18] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.6152 [0.5999, 0.6305] | {'remove_edges': 3464, 'relabel_edges': 2133, 'split': 4, 'reopen': 1} |
| 600 | 1 | 0.07 [0.05, 0.08] | 0.083 [0.070, 0.095] | -0.01 [-0.07, 0.04] | 0.04 [-0.05, 0.14] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.6176 [0.6111, 0.6241] | {'remove_edges': 2431, 'relabel_edges': 1063, 'split': 3} |

Before any revision (right after injection): audit fit 0.4986 [0.4967, 0.5005].

## E10.8 Continual additive learning (hidden relations arrive one per stage)

| Condition | relations matched (Jaccard ≥ 0.5 on offered candidate pairs) by stage | final | final Jaccard on offered pairs | final Jaccard over all framed heads (incl. rule closure) | retention drop (arrival − final) | final val fit |
|---|---|---|---|---|---|---|
| additive | 0.00 [0.00, 0.00] → 0.00 [0.00, 0.00] → 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | antonym 0.27 [0.25, 0.30]; similar_to 0.16 [0.09, 0.22]; instance_hypernym 0.08 [0.06, 0.09] | antonym 0.18 [0.18, 0.18]; similar_to 0.10 [0.04, 0.15]; instance_hypernym 0.05 [0.03, 0.07] | antonym 0.06 [0.01, 0.11]; similar_to -0.02 [-0.10, 0.06]; instance_hypernym 0.00 [0.00, 0.00] | 0.7182 [0.7149, 0.7215] |
| additive_dream | 0.00 [0.00, 0.00] → 0.00 [0.00, 0.00] → 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | antonym 0.31 [0.23, 0.38]; similar_to 0.36 [0.30, 0.43]; instance_hypernym 0.07 [-0.07, 0.21] | antonym 0.18 [0.16, 0.20]; similar_to 0.23 [0.16, 0.29]; instance_hypernym 0.05 [-0.03, 0.13] | antonym 0.03 [-0.01, 0.07]; similar_to -0.02 [-0.12, 0.09]; instance_hypernym 0.00 [0.00, 0.00] | 0.7290 [0.7261, 0.7319] |
| all_at_once | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | antonym 0.08 [0.06, 0.10]; similar_to 0.15 [0.14, 0.15]; instance_hypernym 0.07 [0.02, 0.12] | antonym 0.05 [0.03, 0.07]; similar_to 0.09 [0.07, 0.10]; instance_hypernym 0.05 [0.01, 0.09] | antonym 0.00 [0.00, 0.00]; similar_to 0.00 [0.00, 0.00]; instance_hypernym 0.00 [0.00, 0.00] | 0.7215 [0.7160, 0.7269] |
| plastic | 0.00 [0.00, 0.00] → 0.00 [0.00, 0.00] → 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | antonym 0.15 [0.07, 0.22]; similar_to 0.17 [0.08, 0.26]; instance_hypernym 0.08 [-0.04, 0.21] | antonym 0.10 [0.06, 0.14]; similar_to 0.10 [0.04, 0.16]; instance_hypernym 0.06 [-0.01, 0.12] | antonym 0.19 [0.09, 0.28]; similar_to -0.01 [-0.06, 0.04]; instance_hypernym 0.00 [0.00, 0.00] | 0.7195 [0.7168, 0.7222] |

CPU seconds by part (sum over jobs): a 5527, b 1690, c 7644, seed 2856, dream 5305, continual 4705.
