# E10.0 — self-learned semantics (H-H)

Seeds [101, 202, 303]; CPU single-threaded jobs; world `synthetic`. Means over seeds with 95% t-intervals in brackets. Gold labels and audit observations are used only by evaluation; every learning-side decision uses held-out self-tests.

## H-H refutation clauses

| Clause | Result |
|---|---|
| (i) erased edges recovered above matched random candidates | {'0.1': True, '0.3': True, '0.5': True, '0.7': True} |
| (ii) blank slots align with hidden relations above chance (ARI − permuted, additive) | {'absent': False, 'collapsed': True} |
| (ii′) same, per-seed permutation test (Holm over seeds; added after the first run, reported alongside) | {'absent': True, 'collapsed': True} |
| (iii) self-acceptance beats random acceptance at the matched rate (edges) | {'erasure-0.3': True, 'erasure-0.5': True} |
| (iii) self-acceptance beats random acceptance at the matched rate (slots, pooled) | True |
| (iv) new-word frame inference beats nearest-neighbour frames (F1) | {'1': True, '2': True, '4': True, '8': True} |

## (a) Learnability ablation (30% erased, 5% spurious prior edges, noisy priors)

Contrasts against everything learnable with L2-to-prior (paired over seeds):

| Contrast | Δ held-out-obs fit (train concepts) | Δ test-concept fit (zero-shot) | Δ recovery F1 | Δ recovery AUC |
|---|---|---|---|---|
| atomics=fixed − all l2 | -0.1016 [-0.1129, -0.0903] | -0.0648 [-0.0894, -0.0402] | -0.007 [-0.051, 0.037] | -0.009 [-0.023, 0.006] |
| atomics=free − all l2 | 0.0191 [0.0153, 0.0229] | -0.3190 [-0.3697, -0.2683] | -0.817 [-0.838, -0.796] | -0.445 [-0.485, -0.405] |
| only atomics l2 − all fixed | 0.2138 [0.1985, 0.2291] | 0.0786 [0.0687, 0.0885] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| relations=fixed − all l2 | -0.0570 [-0.0669, -0.0471] | -0.0788 [-0.1193, -0.0382] | -0.026 [-0.047, -0.006] | -0.020 [-0.037, -0.003] |
| relations=free − all l2 | 0.0012 [0.0002, 0.0023] | 0.0001 [-0.0003, 0.0006] | 0.006 [-0.011, 0.023] | -0.002 [-0.008, 0.004] |
| only relations l2 − all fixed | 0.1219 [0.1170, 0.1268] | 0.1001 [0.0673, 0.1330] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| mapping=fixed − all l2 | -0.0125 [-0.0160, -0.0091] | -0.0167 [-0.0244, -0.0089] | 0.010 [-0.013, 0.034] | 0.002 [-0.006, 0.009] |
| mapping=free − all l2 | 0.0053 [0.0043, 0.0063] | -0.0803 [-0.1298, -0.0309] | -0.691 [-0.915, -0.466] | -0.027 [-0.040, -0.015] |
| only mapping l2 − all fixed | 0.0302 [0.0277, 0.0326] | 0.0108 [-0.0008, 0.0224] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| frames=fixed − all l2 | -0.0925 [-0.1026, -0.0825] | -0.0224 [-0.0259, -0.0190] | -0.840 [-0.860, -0.821] | -0.486 [-0.492, -0.480] |
| frames=free − all l2 | 0.0024 [0.0011, 0.0036] | -0.0001 [-0.0017, 0.0014] | 0.105 [0.086, 0.124] | 0.005 [-0.008, 0.018] |
| only frames l2 − all fixed | 0.1252 [0.1055, 0.1449] | 0.0000 [0.0000, 0.0000] | 0.787 [0.726, 0.847] | 0.437 [0.423, 0.452] |
| all free − all l2 | 0.0192 [0.0157, 0.0227] | -0.1209 [-0.1296, -0.1123] | -0.588 [-0.646, -0.531] | -0.060 [-0.063, -0.058] |
| all fixed − all l2 | -0.3856 [-0.4030, -0.3681] | -0.1850 [-0.2050, -0.1649] | -0.840 [-0.860, -0.821] | -0.486 [-0.492, -0.480] |

Selected grid cells:

| atomics / relations / mapping / frames | val fit | test fit | recovery F1 | AUC | atomic cos | spurious removed |
|---|---:|---:|---:|---:|---:|---:|
| l2 / l2 / l2 / l2 | 0.8033 [0.7986, 0.8079] | 0.6148 [0.5672, 0.6623] | 0.840 [0.821, 0.860] | 0.986 [0.980, 0.992] | 0.827 [0.825, 0.829] | 0.097 [-0.009, 0.204] |
| free / free / free / free | 0.8225 [0.8196, 0.8254] | 0.4939 [0.4388, 0.5489] | 0.252 [0.213, 0.291] | 0.926 [0.919, 0.932] | 0.570 [0.551, 0.588] | 0.121 [0.058, 0.185] |
| fixed / fixed / fixed / fixed | 0.4177 [0.4038, 0.4316] | 0.4298 [0.3789, 0.4807] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.800 [0.800, 0.800] | 0.000 [0.000, 0.000] |
| fixed / fixed / fixed / l2 | 0.5429 [0.5266, 0.5592] | 0.4298 [0.3789, 0.4807] | 0.787 [0.726, 0.847] | 0.937 [0.923, 0.952] | 0.800 [0.800, 0.800] | 0.000 [0.000, 0.000] |
| l2 / l2 / l2 / fixed | 0.7107 [0.7046, 0.7168] | 0.5923 [0.5476, 0.6371] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.818 [0.817, 0.820] | 0.170 [0.006, 0.334] |
| fixed / l2 / l2 / l2 | 0.7017 [0.6927, 0.7107] | 0.5500 [0.5225, 0.5774] | 0.833 [0.785, 0.881] | 0.978 [0.966, 0.989] | 0.800 [0.800, 0.800] | 0.154 [-0.077, 0.385] |
| l2 / fixed / l2 / l2 | 0.7463 [0.7362, 0.7564] | 0.5360 [0.4678, 0.6043] | 0.814 [0.773, 0.854] | 0.966 [0.949, 0.983] | 0.817 [0.815, 0.820] | 0.089 [0.051, 0.126] |
| l2 / l2 / fixed / l2 | 0.7907 [0.7861, 0.7954] | 0.5981 [0.5457, 0.6505] | 0.850 [0.814, 0.887] | 0.988 [0.975, 1.001] | 0.827 [0.826, 0.828] | 0.000 [0.000, 0.000] |
| l2 / l2 / l2 / free | 0.8056 [0.8013, 0.8099] | 0.6146 [0.5682, 0.6610] | 0.945 [0.922, 0.968] | 0.991 [0.978, 1.003] | 0.827 [0.825, 0.829] | 0.057 [-0.069, 0.183] |
| l2 / free / l2 / free | 0.8068 [0.8031, 0.8104] | 0.6151 [0.5681, 0.6621] | 0.940 [0.902, 0.977] | 0.989 [0.969, 1.008] | 0.827 [0.825, 0.829] | 0.057 [-0.069, 0.183] |

Best test-concept fit in the full 3⁴ grid: `l2|free|l2|free`.

## (b) Erasure & recovery

| Erased (principal fixed) | Precision | Recall | F1 | AUC | R-precision | Random (prevalence / AUC 0.5) | Filler-frequency AUC / R-prec | Test fit |
|---|---|---|---|---|---|---|---|---|
| 10% (90%) | 0.995 [0.974, 1.016] | 0.825 [0.692, 0.958] | 0.902 [0.817, 0.986] | 0.989 [0.988, 0.991] | 0.965 [0.929, 1.000] | 0.333 [0.333, 0.333] / 0.5 | 0.519 [0.414, 0.624] / 0.376 [0.229, 0.522] | 0.7290 [0.6973, 0.7606] |
| 30% (70%) | 0.998 [0.990, 1.006] | 0.710 [0.696, 0.724] | 0.830 [0.819, 0.840] | 0.987 [0.981, 0.992] | 0.959 [0.940, 0.978] | 0.333 [0.333, 0.333] / 0.5 | 0.515 [0.460, 0.570] / 0.370 [0.302, 0.438] | 0.6257 [0.5578, 0.6937] |
| 50% (50%) | 1.000 [1.000, 1.000] | 0.588 [0.564, 0.611] | 0.740 [0.722, 0.759] | 0.986 [0.985, 0.988] | 0.951 [0.938, 0.964] | 0.333 [0.333, 0.333] / 0.5 | 0.504 [0.491, 0.517] / 0.360 [0.325, 0.396] | 0.5311 [0.4918, 0.5705] |
| 70% (30%) | 1.000 [1.000, 1.000] | 0.381 [0.250, 0.512] | 0.550 [0.411, 0.690] | 0.984 [0.973, 0.996] | 0.929 [0.912, 0.946] | 0.333 [0.333, 0.333] / 0.5 | 0.519 [0.493, 0.545] / 0.371 [0.359, 0.383] | 0.3935 [0.2766, 0.5104] |

## (c) Blank-relation discovery — hidden relations absent

| Method | ARI (gold edges) | ARI − permuted | permutation p (Holm, max over seeds) | mean best Jaccard | detection P | detection R | accepted / proposed slots | test fit |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.807 [0.683, 0.931] | 0.809 [0.682, 0.935] | 0.015 | 0.839 [0.721, 0.957] | 0.962 [0.912, 1.013] | 0.877 [0.844, 0.911] | n/a / n/a | 0.6844 [0.6549, 0.7140] |
| additive | 0.386 [-0.005, 0.778] | 0.388 [-0.005, 0.782] | 0.015 | 0.441 [0.302, 0.580] | 0.874 [0.788, 0.961] | 0.781 [0.646, 0.917] | 3.3 [1.9, 4.8] / 4.3 [2.9, 5.8] | 0.6991 [0.6937, 0.7044] |
| additive_norule | 0.385 [-0.023, 0.793] | 0.387 [-0.023, 0.796] | 0.015 | 0.416 [0.303, 0.529] | 0.861 [0.841, 0.881] | 0.788 [0.676, 0.901] | 3.7 [0.8, 6.5] / 4.3 [2.9, 5.8] | 0.6833 [0.6506, 0.7160] |
| all_at_once | 0.273 [0.143, 0.403] | 0.274 [0.142, 0.405] | 0.015 | 0.330 [0.134, 0.527] | 0.887 [0.875, 0.899] | 0.646 [0.521, 0.770] | 5.0 [5.0, 5.0] / 5.0 [5.0, 5.0] | 0.6904 [0.6808, 0.7000] |
| m3 | 0.188 [-0.180, 0.556] | 0.189 [-0.180, 0.558] | 0.015 | 0.236 [0.081, 0.392] | 0.877 [0.842, 0.913] | 0.575 [0.386, 0.765] | n/a / n/a | 0.6699 [0.6352, 0.7045] |
| stem_cell | 0.140 [-0.059, 0.339] | 0.139 [-0.061, 0.339] | 0.015 | 0.215 [0.197, 0.233] | 0.885 [0.840, 0.931] | 0.632 [0.549, 0.715] | n/a / n/a | 0.6803 [0.6490, 0.7116] |

Per hidden relation, best Jaccard of a discovered cluster (candidate level):

| Method | has_part | located_in | similar_to |
|---|---|---|---|
| oracle | 0.795 [0.588, 1.002] | 0.858 [0.517, 1.199] | 0.864 [0.653, 1.075] |
| additive | 0.272 [-0.246, 0.789] | 0.437 [-0.023, 0.897] | 0.615 [0.338, 0.891] |
| additive_norule | 0.252 [-0.217, 0.721] | 0.380 [-0.283, 1.043] | 0.616 [0.338, 0.893] |
| all_at_once | 0.164 [-0.128, 0.456] | 0.392 [-0.072, 0.856] | 0.435 [0.314, 0.557] |
| m3 | 0.064 [0.013, 0.115] | 0.239 [-0.018, 0.496] | 0.406 [-0.127, 0.939] |
| stem_cell | 0.074 [0.013, 0.136] | 0.246 [0.068, 0.423] | 0.325 [0.064, 0.587] |

Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive: has_part 0.618 [-0.609, 1.845]; similar_to 0.490 [0.323, 0.656]; located_in 0.239 [-0.039, 0.517].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive_norule: has_part 0.119 [-0.038, 0.277]; similar_to 0.344 [0.150, 0.539]; located_in 0.235 [-0.107, 0.576].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), all_at_once: has_part 0.591 [-0.643, 1.825]; similar_to 0.394 [0.259, 0.528]; located_in 0.191 [-0.043, 0.424].

- additive − all_at_once: ARI 0.113 [-0.408, 0.634], mean best Jaccard 0.111 [-0.224, 0.446], test fit 0.0087 [-0.0060, 0.0233]
- additive − m3: ARI 0.198 [-0.189, 0.586], mean best Jaccard 0.205 [0.069, 0.341], test fit 0.0292 [-0.0046, 0.0630]
- additive − stem_cell: ARI 0.247 [-0.080, 0.573], mean best Jaccard 0.226 [0.105, 0.347], test fit 0.0188 [-0.0133, 0.0509]
- additive − additive_norule: n/a
- rule enforcement: relation recovery (additive − additive_norule): 0.216 [-0.190, 0.623]

**Riddle-style interpretation (E10.6, absent).** 25 accepted slots, 14 adopted a structural rule; 13 of those rules are true of the matched hidden relation, 12 are its designed property. Rule predictions' gold precision 0.866 [0.772, 0.959]. Hypotheses scored per slot (full space): 38.9.

| Matched hidden relation | slots | adopted | true | designed | rules adopted |
|---|---:|---:|---:|---:|---|
| similar_to | 10 | 8 | 8 | 7 | symmetric×7, inverse_of:similar_to×1 |
| located_in | 8 | 1 | 0 | 0 | functional×1 |
| has_part | 7 | 5 | 5 | 5 | inverse_of:part_of×5 |

Accuracy of the adopted hypothesis vs number of candidate hypotheses (one true + m−1 false):

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.57 | 0.55 | 0.58 | 0.56 | 0.58 | 0.39 |
| slots | 25 | 25 | 25 | 25 | 25 | 13 |

**Human-likeness (additive, absent).** Over 13 slots: purity of a slot's members w.r.t. its final hidden relation rises from 0.430 [0.216, 0.643] (first committed members) to 0.547 [0.303, 0.792]; hidden relations present 1.54 [0.77, 2.30] → 1.31 [0.59, 2.02]; members 25.5 [3.3, 47.8] (peak 33.8 [8.7, 58.8]) → 31.4 [8.3, 54.5]. Slots that overextended then refined: 8.

Soft hypothesis of a newly opened slot (mass-weighted shares of the pairs it holds; steps since opening):

| step | target relation | other hidden relations | distractors | effective size |
|---:|---:|---:|---:|---:|
| 0 | 0.15 | 0.15 | 0.70 | 172 |
| 25 | 0.43 | 0.20 | 0.37 | 46 |
| 50 | 0.43 | 0.12 | 0.45 | 36 |
| 100 | 0.50 | 0.08 | 0.42 | 34 |
| 200 | 0.56 | 0.06 | 0.39 | 31 |
| 400 | 0.56 | 0.05 | 0.39 | 31 |
| 700 | 0.56 | 0.05 | 0.39 | 31 |

Slot self-acceptance (absent; additive, no-rule and all-at-once runs pooled): accuracy 0.805 (Wilson 0.66–0.90, n = 41) vs random at the matched rate 0.675 (p = 0.008); accept-all 0.732; wrong-acceptance rate 0.194.

## (c) Blank-relation discovery — hidden relations collapsed

| Method | ARI (gold edges) | ARI − permuted | permutation p (Holm, max over seeds) | mean best Jaccard | detection P | detection R | accepted / proposed slots | test fit |
|---|---|---|---|---|---|---|---|---|
| oracle | 1.000 [1.000, 1.000] | 1.000 [0.998, 1.002] | 0.015 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a / n/a | 0.7602 [0.7409, 0.7795] |
| additive | 0.416 [0.361, 0.471] | 0.416 [0.362, 0.470] | 0.015 | 0.653 [0.589, 0.717] | 1.000 [1.000, 1.000] | 0.690 [0.666, 0.713] | 2.0 [2.0, 2.0] / 3.0 [3.0, 3.0] | 0.7499 [0.7347, 0.7651] |
| additive_norule | 0.381 [0.201, 0.562] | 0.381 [0.201, 0.560] | 0.015 | 0.652 [0.573, 0.731] | 1.000 [1.000, 1.000] | 0.794 [0.349, 1.238] | 1.7 [0.2, 3.1] / 2.7 [1.2, 4.1] | 0.7491 [0.7378, 0.7603] |
| all_at_once | 0.190 [0.022, 0.357] | 0.189 [0.020, 0.359] | 0.015 | 0.392 [0.260, 0.524] | 1.000 [1.000, 1.000] | 0.583 [0.539, 0.628] | 3.0 [3.0, 3.0] / 5.0 [5.0, 5.0] | 0.7446 [0.7256, 0.7637] |
| m3 | 0.363 [0.239, 0.486] | 0.362 [0.238, 0.486] | 0.015 | 0.631 [0.526, 0.736] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a / n/a | 0.7172 [0.6936, 0.7407] |
| stem_cell | 0.133 [-0.028, 0.293] | 0.133 [-0.026, 0.292] | 0.015 | 0.299 [0.019, 0.579] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | n/a / n/a | 0.7426 [0.7072, 0.7781] |

Per hidden relation, best Jaccard of a discovered cluster (candidate level):

| Method | part_of | member_of |
|---|---|---|
| oracle | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| additive | 0.670 [0.643, 0.698] | 0.636 [0.516, 0.756] |
| additive_norule | 0.670 [0.643, 0.698] | 0.634 [0.494, 0.774] |
| all_at_once | 0.397 [0.250, 0.544] | 0.387 [0.144, 0.630] |
| m3 | 0.632 [0.572, 0.693] | 0.630 [0.479, 0.780] |
| stem_cell | 0.327 [-0.002, 0.657] | 0.270 [0.008, 0.532] |

Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive: part_of 0.966 [0.935, 0.997]; member_of 0.051 [-0.020, 0.121].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive_norule: part_of 0.670 [0.643, 0.698]; member_of 0.019 [-0.062, 0.100].
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), all_at_once: part_of 0.914 [0.823, 1.005]; member_of 0.379 [0.131, 0.626].

- additive − all_at_once: ARI 0.226 [0.095, 0.358], mean best Jaccard 0.261 [0.159, 0.364], test fit 0.0052 [-0.0043, 0.0147]
- additive − m3: ARI 0.053 [-0.077, 0.183], mean best Jaccard 0.022 [-0.081, 0.126], test fit 0.0327 [0.0184, 0.0470]
- additive − stem_cell: ARI 0.283 [0.154, 0.412], mean best Jaccard 0.354 [0.104, 0.605], test fit 0.0072 [-0.0150, 0.0294]
- additive − additive_norule: n/a
- rule enforcement: relation recovery (additive − additive_norule): 0.164 [0.090, 0.237]

**Riddle-style interpretation (E10.6, collapsed).** 15 accepted slots, 7 adopted a structural rule; 7 of those rules are true of the matched hidden relation, 7 are its designed property. Rule predictions' gold precision 1.000 [1.000, 1.000]. Hypotheses scored per slot (full space): 52.1.

| Matched hidden relation | slots | adopted | true | designed | rules adopted |
|---|---:|---:|---:|---:|---|
| part_of | 7 | 6 | 6 | 6 | inverse_of:has_part×6 |
| member_of | 8 | 1 | 1 | 1 | antisymmetric×1 |

Accuracy of the adopted hypothesis vs number of candidate hypotheses (one true + m−1 false):

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.14 | 0.16 | 0.14 | 0.14 | 0.14 | 0.14 |
| slots | 15 | 15 | 15 | 15 | 15 | 15 |

**Human-likeness (additive, collapsed).** Over 8 slots: purity of a slot's members w.r.t. its final hidden relation rises from 0.811 [0.578, 1.044] (first committed members) to 0.958 [0.860, 1.057]; hidden relations present 1.50 [1.05, 1.95] → 1.25 [0.86, 1.64]; members 5.6 [0.7, 10.5] (peak 24.0 [-0.7, 48.7]) → 24.0 [-0.7, 48.7]. Slots that overextended then refined: 3.

Soft hypothesis of a newly opened slot (mass-weighted shares of the pairs it holds; steps since opening):

| step | target relation | other hidden relations | distractors | effective size |
|---:|---:|---:|---:|---:|
| 0 | 0.50 | 0.50 | 0.00 | 83 |
| 25 | 0.51 | 0.49 | 0.00 | 78 |
| 50 | 0.53 | 0.47 | 0.00 | 74 |
| 100 | 0.58 | 0.42 | 0.00 | 68 |
| 200 | 0.63 | 0.37 | 0.00 | 61 |
| 400 | 0.64 | 0.36 | 0.00 | 59 |
| 700 | 0.64 | 0.36 | 0.00 | 58 |

Slot self-acceptance (collapsed; additive, no-rule and all-at-once runs pooled): accuracy 0.688 (Wilson 0.51–0.82, n = 32) vs random at the matched rate 0.594 (p = 0.169); accept-all 0.875; wrong-acceptance rate 0.050.

## (d) Self-tested acceptance of edge hypotheses (held-out observations; audit never read)

| Source | n | accept rate | accuracy (Wilson) | random @ matched rate | accept-all | in-sample (confirmation-bias control) | acc − random by seed | audit Δ accepted / rejected |
|---|---:|---:|---|---:|---:|---|---|---|
| erasure-0.3 | 840 | 0.92 | 0.927 (0.91–0.94) | 0.824 | 0.890 | 0.898 (accepts 0.99) | 0.103 [0.082, 0.124] | +0.0991 / +0.0066 |
| erasure-0.5 | 1368 | 0.92 | 0.925 (0.91–0.94) | 0.825 | 0.889 | 0.901 (accepts 0.98) | 0.098 [0.032, 0.164] | +0.0995 / +0.0070 |

## (e) New-word frame inference (fast mapping)

| k | discovered F1 | known-only F1 | oracle-dictionary F1 | nearest-neighbour F1 | random F1 | discovered fit | NN fit | context-mean fit | gold-frame fit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.417 [0.329, 0.506] | 0.403 [0.270, 0.535] | 0.739 [0.705, 0.773] | 0.344 [0.240, 0.447] | 0.000 [0.000, 0.000] | 0.6300 [0.5703, 0.6897] | 0.4234 [0.3422, 0.5046] | 0.7368 [0.7267, 0.7470] | 0.7119 [0.6549, 0.7690] |
| 2 | 0.543 [0.483, 0.604] | 0.501 [0.273, 0.728] | 0.923 [0.855, 0.991] | 0.361 [0.276, 0.445] | 0.000 [0.000, 0.000] | 0.6795 [0.6646, 0.6945] | 0.4376 [0.3741, 0.5011] | 0.7915 [0.7791, 0.8038] | 0.7119 [0.6549, 0.7690] |
| 4 | 0.639 [0.542, 0.736] | 0.565 [0.366, 0.765] | 0.952 [0.926, 0.978] | 0.363 [0.268, 0.458] | 0.002 [-0.007, 0.011] | 0.7102 [0.6801, 0.7403] | 0.4346 [0.3700, 0.4993] | 0.8241 [0.8143, 0.8339] | 0.7119 [0.6549, 0.7690] |
| 8 | 0.685 [0.598, 0.773] | 0.585 [0.332, 0.838] | 0.971 [0.939, 1.003] | 0.361 [0.278, 0.445] | 0.001 [-0.003, 0.006] | 0.7264 [0.7011, 0.7517] | 0.4383 [0.3697, 0.5068] | 0.8405 [0.8307, 0.8503] | 0.7119 [0.6549, 0.7690] |

- k = 1: discovered − nearest_neighbour (F1) 0.074 [0.007, 0.141]; discovered − known_only (F1) 0.014 [-0.039, 0.068]; discovered − random (F1) 0.417 [0.329, 0.506]; discovered − nearest_neighbour (fit) 0.207 [0.181, 0.232]; discovered − context_mean (fit) -0.107 [-0.157, -0.057]
- k = 2: discovered − nearest_neighbour (F1) 0.183 [0.116, 0.249]; discovered − known_only (F1) 0.043 [-0.127, 0.212]; discovered − random (F1) 0.543 [0.483, 0.604]; discovered − nearest_neighbour (fit) 0.242 [0.193, 0.291]; discovered − context_mean (fit) -0.112 [-0.115, -0.109]
- k = 4: discovered − nearest_neighbour (F1) 0.276 [0.213, 0.339]; discovered − known_only (F1) 0.074 [-0.029, 0.176]; discovered − random (F1) 0.637 [0.532, 0.742]; discovered − nearest_neighbour (fit) 0.276 [0.241, 0.310]; discovered − context_mean (fit) -0.114 [-0.135, -0.093]
- k = 8: discovered − nearest_neighbour (F1) 0.324 [0.240, 0.408]; discovered − known_only (F1) 0.100 [-0.070, 0.271]; discovered − random (F1) 0.684 [0.601, 0.768]; discovered − nearest_neighbour (fit) 0.288 [0.232, 0.344]; discovered − context_mean (fit) -0.114 [-0.130, -0.098]

Discovered-relation F1 by hierarchy depth at the largest k (exploratory 'basic-level' readout): depth 0: 1.000 [1.000, 1.000]; depth 1: 0.910 [0.658, 1.161]; depth 2: 0.786 [0.668, 0.904]; depth 3: 0.646 [0.533, 0.759].

## E10.4 Seed ontology → recovered logical ontology

| Start | edge F1 (all framed) | P / R | candidate AUC | candidate-level relation Jaccard (mean) | relations recovered (J ≥ 0.5, of 7) | accepted slots | multi-hop is-a acc (gold) | transitivity located_in (gold) | inverse part (gold) | symmetry similar (gold) | test fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| empty | 0.126 [0.083, 0.169] | 0.40 [0.32, 0.48] / 0.07 [0.05, 0.10] | 0.622 [0.557, 0.687] | 0.066 [0.030, 0.102] | 0.0 [0.0, 0.0] | 7.7 [6.2, 9.1] | 0.50 [0.50, 0.50] (0.90 [0.87, 0.93]) | 0.00 (1.00 [1.00, 1.00]) | 0.00 [0.00, 0.00] (0.79 [0.66, 0.91]) | 0.08 [-0.27, 0.43] (0.86 [0.83, 0.88]) | 0.0018 [-0.0058, 0.0093] |
| core | 0.254 [0.216, 0.291] | 0.93 [0.84, 1.02] / 0.15 [0.12, 0.17] | 0.680 [0.564, 0.796] | 0.026 [-0.009, 0.062] | 0.0 [0.0, 0.0] | 0.7 [-0.8, 2.1] | 0.56 [0.52, 0.59] (0.90 [0.87, 0.93]) | n/a (1.00 [1.00, 1.00]) | 0.00 [0.00, 0.00] (0.79 [0.66, 0.91]) | 0.00 (0.86 [0.83, 0.88]) | 0.1731 [0.0862, 0.2601] |
| noisy | 0.579 [0.520, 0.638] | 0.84 [0.81, 0.87] / 0.44 [0.37, 0.52] | 0.761 [0.693, 0.829] | 0.292 [0.088, 0.495] | 0.7 [-0.8, 2.1] | 0.7 [-0.8, 2.1] | 0.58 [0.52, 0.63] (0.90 [0.87, 0.93]) | 0.23 [-0.63, 1.10] (1.00 [1.00, 1.00]) | 0.14 [0.10, 0.19] (0.79 [0.66, 0.91]) | 0.20 [0.03, 0.37] (0.86 [0.83, 0.88]) | 0.3164 [0.2574, 0.3755] |
| curated30 | 0.547 [0.415, 0.679] | 0.96 [0.90, 1.02] / 0.38 [0.25, 0.52] | 0.750 [0.667, 0.833] | 0.177 [-0.137, 0.491] | 0.7 [-0.8, 2.1] | 0.0 [0.0, 0.0] | 0.56 [0.48, 0.64] (0.90 [0.87, 0.93]) | 0.50 [-0.74, 1.74] (1.00 [1.00, 1.00]) | 0.14 [0.11, 0.16] (0.79 [0.66, 0.91]) | 0.24 [0.15, 0.33] (0.86 [0.83, 0.88]) | 0.3140 [0.2693, 0.3586] |
| full | 1.000 [1.000, 1.000] | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] | n/a | n/a | 7.0 [7.0, 7.0] | 0.0 [0.0, 0.0] | 0.90 [0.87, 0.93] (0.90 [0.87, 0.93]) | 1.00 [1.00, 1.00] (1.00 [1.00, 1.00]) | 0.79 [0.66, 0.91] (0.79 [0.66, 0.91]) | 0.86 [0.83, 0.88] (0.86 [0.83, 0.88]) | 0.7611 [0.7440, 0.7783] |

## E10.5 Dreaming (offline self-revision) after injected corruptions

| Dream every (steps) | passes | wrong edges repaired | correct edges damaged | merged relations split (ARI) | wrong slot repaired | wrong slot reopened/removed | correct slots disturbed | audit fit (train concepts) | applied revisions |
|---|---:|---|---|---|---|---|---|---|---|
| revisit only, every 300 | 4 | 0.00 [-0.01, 0.02] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.78 [0.58, 0.98] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.7817 [0.7688, 0.7947] | {'reopened': 3} |
| off | 0 | 0.00 [-0.01, 0.02] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.7765 [0.7632, 0.7898] | {} |
| 300 | 4 | 0.53 [0.45, 0.61] | 0.003 [-0.001, 0.006] | 0.76 [0.51, 1.01] | 0.89 [0.71, 1.07] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.7967 [0.7900, 0.8034] | {'remove_edges': 210, 'relabel_edges': 131, 'split': 10} |
| 600 | 2 | 0.51 [0.43, 0.60] | 0.003 [-0.001, 0.007] | 0.77 [0.39, 1.14] | 0.88 [0.85, 0.92] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.7955 [0.7866, 0.8045] | {'remove_edges': 202, 'relabel_edges': 94, 'split': 6} |
| 1200 | 1 | 0.46 [0.33, 0.60] | 0.002 [-0.003, 0.008] | 0.52 [-0.73, 1.77] | 0.52 [-0.23, 1.28] | 0.33 [-1.10, 1.77] | 0.00 [0.00, 0.00] | 0.7912 [0.7809, 0.8015] | {'remove_edges': 183, 'relabel_edges': 67, 'split': 3} |

Before any revision (right after injection): audit fit 0.7211 [0.7156, 0.7267].

## E10.7 Data variety vs self-confirmation bias

| Sources per concept (8 non-audit observations) | proposals | wrong self-acceptance rate | by seed | artifact proposals | artifact acceptance | audit Δ of accepted artifacts | in-sample acceptance |
|---|---:|---|---|---:|---:|---|---|
| 1 | 1365 | 0.439 | 0.439 [0.436, 0.441] | 534 | 1.000 | -0.0593 | 0.990 |
| 2 | 1747 | 0.181 | 0.181 [0.160, 0.201] | 924 | 0.139 | -0.0126 | 0.521 |
| 4 | 1606 | 0.140 | 0.140 [0.118, 0.163] | 804 | 0.139 | 0.0151 | 0.535 |
| 8 | 1709 | 0.165 | 0.164 [0.086, 0.243] | 943 | 0.169 | 0.0290 | 0.538 |

## E10.8 Continual additive learning (hidden relations arrive one per stage)

| Condition | relations matched (Jaccard ≥ 0.5 on offered candidate pairs) by stage | final | final Jaccard on offered pairs | final Jaccard over all framed heads (incl. rule closure) | retention drop (arrival − final) | final val fit |
|---|---|---|---|---|---|---|
| additive | 1.00 [1.00, 1.00] → 1.67 [0.23, 3.10] → 1.67 [0.23, 3.10] | 1.67 [0.23, 3.10] | similar_to 0.88 [0.76, 1.00]; has_part 0.52 [-0.30, 1.34]; located_in 0.13 [-0.29, 0.55] | similar_to 0.64 [0.52, 0.77]; has_part 0.53 [-0.45, 1.52]; located_in 0.06 [-0.12, 0.24] | similar_to -0.00 [-0.03, 0.03]; has_part -0.00 [-0.09, 0.08]; located_in 0.00 [0.00, 0.00] | 0.7875 [0.7721, 0.8028] |
| additive_dream | 1.00 [1.00, 1.00] → 1.00 [1.00, 1.00] → 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | similar_to 0.87 [0.75, 0.98]; has_part 0.36 [0.16, 0.56]; located_in 0.04 [-0.06, 0.13] | similar_to 0.63 [0.51, 0.76]; has_part 0.40 [-0.17, 0.97]; located_in 0.02 [-0.03, 0.06] | similar_to 0.02 [-0.04, 0.07]; has_part -0.03 [-0.08, 0.02]; located_in 0.00 [0.00, 0.00] | 0.7896 [0.7731, 0.8060] |
| all_at_once | 2.00 [-0.48, 4.48] | 2.00 [-0.48, 4.48] | similar_to 0.58 [0.33, 0.82]; has_part 0.58 [-0.60, 1.77]; located_in 0.39 [-0.08, 0.86] | similar_to 0.39 [0.26, 0.53]; has_part 0.59 [-0.64, 1.83]; located_in 0.19 [-0.04, 0.42] | similar_to 0.00 [0.00, 0.00]; has_part 0.00 [0.00, 0.00]; located_in 0.00 [0.00, 0.00] | 0.7878 [0.7745, 0.8012] |
| plastic | 1.00 [1.00, 1.00] → 1.00 [-1.48, 3.48] → 1.67 [0.23, 3.10] | 1.67 [0.23, 3.10] | similar_to 0.70 [0.22, 1.17]; has_part 0.30 [-0.14, 0.74]; located_in 0.62 [0.08, 1.16] | similar_to 0.37 [0.14, 0.60]; has_part 0.14 [-0.02, 0.29]; located_in 0.30 [0.08, 0.51] | similar_to 0.14 [-0.14, 0.41]; has_part -0.01 [-0.08, 0.06]; located_in 0.00 [0.00, 0.00] | 0.7942 [0.7903, 0.7981] |

CPU seconds by part (sum over jobs): a 1003, b 63, c 528, seed 254, dream 437, variety 69, continual 193.
