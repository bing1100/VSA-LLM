# E10.0-baselines — baselines for claim D (WP-PQ2)

Seeds [101, 202, 303]; dev seeds [7, 8, 9] set the operator-axiom thresholds only (RotatE 0.621, HRR roles 0.198, tuned AMIE 0.611). World and scenarios: `experiments/e10-self-semantics/e10-synthetic.yaml`. Means over seeds with 95% t-intervals in brackets. Baselines see only asserted edges (and, for the learnable ontology, training and validation observations); gold is read by the evaluation only.

## D-B2 Erased-edge recovery on the E10.0 (b) candidate pools

Methods are fit on 90% of the asserted edges; F1 uses a threshold chosen on the other 10% plus distractors. The learnable ontology's row (and the filler-frequency row) is its committed E10.0 run `experiments/e10-self-semantics/runs/e10.0-v3/metrics.jsonl` (same pools; pool sizes match), whose F1 uses its fixed mass ≥ 0.5 rule. With one erased edge per two distractors, accepting every candidate gives F1 = 0.5, so AUC and R-precision are the primary comparison.

| Erased | Method | AUC | R-precision | F1 | ΔAUC vs learnable | ΔF1 vs learnable |
|---|---|---|---|---|---|---|
| 10% | learnable ontology | 0.989 [0.988, 0.991] | 0.965 [0.929, 1.000] | 0.902 [0.817, 0.986] | n/a | n/a |
| 10% | prior_corr | 0.909 [0.903, 0.915] | 0.765 [0.702, 0.829] | 0.756 [0.714, 0.799] | -0.081 [-0.086, -0.075] | -0.145 [-0.213, -0.077] |
| 10% | amie | 0.769 [0.761, 0.778] | 0.627 [0.579, 0.676] | 0.690 [0.674, 0.706] | -0.220 [-0.228, -0.213] | -0.212 [-0.289, -0.134] |
| 10% | transe | 0.761 [0.738, 0.784] | 0.621 [0.506, 0.735] | 0.602 [0.567, 0.637] | -0.229 [-0.253, -0.205] | -0.300 [-0.372, -0.227] |
| 10% | rotate | 0.718 [0.628, 0.807] | 0.571 [0.500, 0.641] | 0.562 [0.467, 0.657] | -0.272 [-0.362, -0.182] | -0.340 [-0.351, -0.329] |
| 10% | complex | 0.710 [0.688, 0.732] | 0.521 [0.481, 0.561] | 0.522 [0.443, 0.601] | -0.280 [-0.304, -0.255] | -0.380 [-0.511, -0.248] |
| 10% | itere | 0.722 [0.680, 0.763] | 0.562 [0.528, 0.596] | 0.576 [0.546, 0.605] | -0.268 [-0.309, -0.227] | -0.326 [-0.387, -0.265] |
| 10% | filler-frequency ranking | 0.519 [0.414, 0.624] | 0.376 [0.229, 0.522] | — | | |
| 30% | learnable ontology | 0.987 [0.981, 0.992] | 0.959 [0.940, 0.978] | 0.830 [0.819, 0.840] | n/a | n/a |
| 30% | prior_corr | 0.916 [0.902, 0.930] | 0.771 [0.739, 0.803] | 0.771 [0.763, 0.780] | -0.070 [-0.087, -0.054] | -0.058 [-0.077, -0.040] |
| 30% | amie | 0.694 [0.673, 0.715] | 0.533 [0.518, 0.547] | 0.555 [0.493, 0.616] | -0.292 [-0.309, -0.275] | -0.275 [-0.333, -0.217] |
| 30% | transe | 0.688 [0.646, 0.730] | 0.542 [0.477, 0.606] | 0.544 [0.470, 0.617] | -0.299 [-0.338, -0.260] | -0.286 [-0.351, -0.221] |
| 30% | rotate | 0.672 [0.618, 0.727] | 0.512 [0.391, 0.632] | 0.539 [0.499, 0.579] | -0.314 [-0.363, -0.265] | -0.291 [-0.334, -0.248] |
| 30% | complex | 0.663 [0.623, 0.703] | 0.498 [0.453, 0.543] | 0.512 [0.478, 0.546] | -0.324 [-0.363, -0.284] | -0.317 [-0.361, -0.273] |
| 30% | itere | 0.661 [0.641, 0.682] | 0.502 [0.468, 0.536] | 0.478 [0.424, 0.533] | -0.325 [-0.348, -0.302] | -0.351 [-0.400, -0.303] |
| 30% | filler-frequency ranking | 0.515 [0.460, 0.570] | 0.370 [0.302, 0.438] | — | | |
| 50% | learnable ontology | 0.986 [0.985, 0.988] | 0.951 [0.938, 0.964] | 0.740 [0.722, 0.759] | n/a | n/a |
| 50% | prior_corr | 0.917 [0.901, 0.933] | 0.775 [0.760, 0.789] | 0.760 [0.692, 0.828] | -0.070 [-0.087, -0.053] | 0.020 [-0.062, 0.101] |
| 50% | amie | 0.616 [0.581, 0.651] | 0.447 [0.411, 0.483] | 0.500 [0.500, 0.500] | -0.370 [-0.406, -0.335] | -0.240 [-0.259, -0.222] |
| 50% | transe | 0.623 [0.577, 0.669] | 0.465 [0.397, 0.533] | 0.453 [0.305, 0.600] | -0.363 [-0.410, -0.316] | -0.288 [-0.436, -0.139] |
| 50% | rotate | 0.593 [0.571, 0.615] | 0.428 [0.372, 0.484] | 0.493 [0.473, 0.514] | -0.393 [-0.414, -0.372] | -0.247 [-0.254, -0.240] |
| 50% | complex | 0.597 [0.538, 0.656] | 0.445 [0.368, 0.522] | 0.491 [0.462, 0.521] | -0.389 [-0.449, -0.329] | -0.249 [-0.296, -0.202] |
| 50% | itere | 0.585 [0.512, 0.659] | 0.432 [0.344, 0.519] | 0.478 [0.418, 0.539] | -0.401 [-0.475, -0.326] | -0.262 [-0.334, -0.189] |
| 50% | filler-frequency ranking | 0.504 [0.491, 0.517] | 0.360 [0.325, 0.396] | — | | |
| 70% | learnable ontology | 0.984 [0.973, 0.996] | 0.929 [0.912, 0.946] | 0.550 [0.411, 0.690] | n/a | n/a |
| 70% | prior_corr | 0.913 [0.896, 0.931] | 0.775 [0.757, 0.793] | 0.766 [0.760, 0.772] | -0.071 [-0.078, -0.064] | 0.215 [0.081, 0.350] |
| 70% | amie | 0.557 [0.537, 0.578] | 0.390 [0.366, 0.414] | 0.500 [0.500, 0.500] | -0.427 [-0.451, -0.403] | -0.050 [-0.190, 0.089] |
| 70% | transe | 0.560 [0.507, 0.614] | 0.398 [0.327, 0.469] | 0.465 [0.444, 0.485] | -0.424 [-0.482, -0.366] | -0.086 [-0.243, 0.071] |
| 70% | rotate | 0.538 [0.493, 0.583] | 0.373 [0.327, 0.419] | 0.457 [0.397, 0.516] | -0.446 [-0.502, -0.390] | -0.094 [-0.185, -0.002] |
| 70% | complex | 0.548 [0.496, 0.601] | 0.372 [0.326, 0.418] | 0.490 [0.448, 0.533] | -0.436 [-0.488, -0.384] | -0.060 [-0.211, 0.090] |
| 70% | itere | 0.545 [0.477, 0.613] | 0.376 [0.280, 0.472] | 0.479 [0.429, 0.530] | -0.439 [-0.516, -0.363] | -0.071 [-0.169, 0.026] |
| 70% | filler-frequency ranking | 0.519 [0.493, 0.545] | 0.371 [0.359, 0.383] | — | | |

## D-B1 Horn axioms on the 30%-erasure graph (vs axioms with confidence ≥ 0.9 on the complete gold graph)

Gold axioms (union over seeds): has_part⁻¹ ⇒ part_of, located_in ∧ located_in ⇒ located_in, part_of⁻¹ ⇒ has_part, similar_to ∧ is_a ⇒ is_a, similar_to⁻¹ ⇒ similar_to, similar_to⁻¹ ∧ is_a ⇒ is_a.

| Reading | precision | recall | F1 | AUC of the score | axioms accepted | recall: symmetric / inverse / transitive / chain |
|---|---|---|---|---|---|---|
| AMIE (defaults: PCA ≥ 0.1, HC ≥ 0.01) | 0.451 [0.240, 0.662] | 1.000 [1.000, 1.000] | 0.618 [0.410, 0.826] | 0.999 [0.998, 1.001] | 13.7 [6.5, 20.8] | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] |
| AMIE (default filters and PCA ≥ 0.611, dev-chosen) | 0.841 [0.807, 0.875] | 0.889 [0.650, 1.128] | 0.863 [0.735, 0.992] | 0.999 [0.998, 1.001] | 6.3 [4.9, 7.8] | 1.00 [1.00, 1.00] / 1.00 [1.00, 1.00] / 0.33 [-1.10, 1.77] / 1.00 [1.00, 1.00] |
| IterE-style RotatE phases (≥ 0.621, dev-chosen) | 1.000 [1.000, 1.000] | 0.333 [-0.081, 0.747] | 0.484 [0.010, 0.959] | 0.863 [0.667, 1.059] | 2.0 [-0.5, 4.5] | 0.67 [-0.77, 2.10] / 0.67 [-0.77, 2.10] / 0.00 [0.00, 0.00] / 0.00 [0.00, 0.00] |
| learnable ontology's HRR roles (≥ 0.198, dev-chosen) | 0.009 [-0.028, 0.046] | 0.167 [-0.551, 0.884] | 0.016 [-0.054, 0.087] | 0.449 [-0.193, 1.090] | 80.0 [-17.7, 177.7] | 0.00 [0.00, 0.00] / 0.00 [0.00, 0.00] / 0.33 [-1.10, 1.77] / 0.33 [-1.10, 1.77] |

Closure of the accepted axioms over the observed graph (one application; `amie_dev` = support ≥ 2 and PCA ≥ the dev threshold; `rotate` = supported axioms above the dev threshold, at most 50, as injected by the IterE loop):

| Axioms from | axioms | closure triples | recall of erased edges | precision vs gold |
|---|---|---|---|---|
| amie_default | 13.7 [6.5, 20.8] | 877.7 [839.6, 915.7] | 0.437 [0.407, 0.467] | 0.250 [0.221, 0.279] |
| amie_dev | 6.3 [4.9, 7.8] | 147.7 [129.4, 166.0] | 0.270 [0.212, 0.328] | 0.944 [0.924, 0.963] |
| rotate | 2.0 [-0.5, 4.5] | 62.7 [-5.5, 130.8] | 0.121 [0.000, 0.242] | 1.000 [1.000, 1.000] |

## D-B5 Edge self-test: re-used validation split vs fresh split vs multiplicity correction

`reused` = E10.0 (lower bound > 0 on the one validation split); `fresh` = a fresh simulated draw of the same size per proposal; `ttest` = one-sided t-test per proposal (α 0.025) on the re-used split; `holm` = Holm over all proposals of a run. `null` = nothing erased: every proposal is a distractor.

| Scenario | proposals | variant | acceptance | accuracy | wrong among accepted | false acceptance (of non-gold) | recall |
|---|---:|---|---|---|---|---|---|
| erasure 30% | 840 | reused | 0.915 [0.889, 0.942] | 0.927 [0.915, 0.939] | 0.053 [0.051, 0.056] | 0.446 [0.387, 0.505] | 0.973 [0.958, 0.988] |
| erasure 30% | 840 | fresh | 0.933 [0.905, 0.962] | 0.924 [0.909, 0.938] | 0.064 [0.061, 0.066] | 0.544 [0.476, 0.613] | 0.981 [0.965, 0.997] |
| erasure 30% | 840 | ttest | 0.824 [0.803, 0.845] | 0.884 [0.869, 0.898] | 0.030 [0.018, 0.043] | 0.228 [0.164, 0.293] | 0.897 [0.878, 0.917] |
| erasure 30% | 840 | holm | 0.043 [0.006, 0.081] | 0.153 [0.119, 0.187] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.049 [0.007, 0.090] |
| erasure 50% | 1368 | reused | 0.918 [0.851, 0.985] | 0.925 [0.891, 0.959] | 0.056 [0.037, 0.075] | 0.476 [0.341, 0.611] | 0.974 [0.946, 1.002] |
| erasure 50% | 1368 | fresh | 0.921 [0.888, 0.953] | 0.928 [0.868, 0.988] | 0.056 [-0.003, 0.114] | 0.459 [0.173, 0.745] | 0.977 [0.969, 0.985] |
| erasure 50% | 1368 | ttest | 0.824 [0.767, 0.881] | 0.879 [0.849, 0.910] | 0.033 [0.009, 0.058] | 0.248 [0.144, 0.352] | 0.895 [0.874, 0.916] |
| erasure 50% | 1368 | holm | 0.031 [-0.003, 0.065] | 0.142 [0.090, 0.193] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.035 [-0.002, 0.073] |
| null (no erasure) | 92 | reused | 0.450 [0.342, 0.558] | 0.550 [0.442, 0.658] | 1.000 [1.000, 1.000] | 0.450 [0.342, 0.558] | n/a |
| null (no erasure) | 92 | fresh | 0.400 [0.352, 0.447] | 0.600 [0.553, 0.648] | 1.000 [1.000, 1.000] | 0.400 [0.352, 0.447] | n/a |
| null (no erasure) | 92 | ttest | 0.281 [0.112, 0.450] | 0.719 [0.550, 0.888] | 1.000 [1.000, 1.000] | 0.281 [0.112, 0.450] | n/a |
| null (no erasure) | 92 | holm | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | n/a | 0.000 [0.000, 0.000] | n/a |

## D-B4 / D-B5 Blank-slot discovery: real, null and fresh-holdout worlds

Null worlds have no hidden relation (`null_distractors`: every relation asserted, only distractors offered; `null_permuted`: the absent scenario with the offered pairs' tails permuted), so every accepted slot is a false discovery. In the real worlds a slot counts as false if fewer than half of its pairs are offered gold pairs of a hidden relation.

| World / holdout | runs | proposed slots | accepted slots | false accepted slots | runs with ≥ 1 false acceptance | AMIE rules on the offered pairs |
|---|---:|---|---|---|---:|---|
| absent/reused | 3 | 4.33 [2.90, 5.77] | 3.33 [1.90, 4.77] | 0.67 [-2.20, 3.54] | 1 | 3.0 [3.0, 3.0] |
| absent/fresh | 3 | 4.33 [2.90, 5.77] | 3.33 [1.90, 4.77] | 0.67 [-2.20, 3.54] | 1 | 3.0 [3.0, 3.0] |
| collapsed/reused | 3 | 3.00 [3.00, 3.00] | 2.00 [2.00, 2.00] | 0.00 [0.00, 0.00] | 0 | 4.0 [4.0, 4.0] |
| null_distractors/reused | 3 | 3.67 [2.23, 5.10] | 2.67 [1.23, 4.10] | 2.67 [1.23, 4.10] | 3 | 0.0 [0.0, 0.0] |
| null_distractors/fresh | 3 | 4.00 [4.00, 4.00] | 3.00 [3.00, 3.00] | 3.00 [3.00, 3.00] | 3 | 0.0 [0.0, 0.0] |
| null_permuted/reused | 3 | 5.00 [5.00, 5.00] | 4.33 [2.90, 5.77] | 4.33 [2.90, 5.77] | 3 | 0.0 [0.0, 0.0] |
| null_permuted/fresh | 3 | 5.00 [5.00, 5.00] | 4.33 [2.90, 5.77] | 4.33 [2.90, 5.77] | 3 | 0.0 [0.0, 0.0] |

Explaining the accepted slots (pooled over seeds): adopted / true of the slot's relation / its designed property. `riddle` = E10.6 held-out hypothesis tests; `amie` = best AMIE rule on the captured pairs; `rotate` = IterE-style axiom from a RotatE fit; `hrr` = axiom read off the slot's own learned operator.

| World / holdout | accepted slots | riddle | AMIE | RotatE | HRR roles |
|---|---:|---|---|---|---|
| absent/reused | 10 | 5 / 5 / 5 | 7 / 5 / 5 | 0 / 0 / 0 | 9 / 0 / 0 |
| absent/fresh | 10 | 6 / 5 / 5 | 7 / 5 / 5 | 0 / 0 / 0 | 9 / 0 / 0 |
| collapsed/reused | 6 | 3 / 3 / 3 | 6 / 3 / 3 | 3 / 3 / 3 | 6 / 0 / 0 |
| null_distractors/reused | 8 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 7 / 0 / 0 |
| null_distractors/fresh | 9 | 1 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 8 / 0 / 0 |
| null_permuted/reused | 13 | 2 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 9 / 0 / 0 |
| null_permuted/fresh | 13 | 2 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 9 / 0 / 0 |

Relation completion with the adopted rule (Jaccard over all framed heads, best accepted slot per hidden relation; riddle = E10.0 crystallization with rule enforcement, AMIE = captured pairs ∪ the AMIE rule's predictions):

| World / holdout | relation | riddle | AMIE |
|---|---|---|---|
| absent/reused | has_part | 0.618 [-0.609, 1.845] | 0.602 [-0.694, 1.898] |
| absent/reused | similar_to | 0.490 [0.323, 0.656] | 0.487 [0.296, 0.677] |
| absent/reused | located_in | 0.239 [-0.039, 0.517] | 0.150 [0.048, 0.253] |
| absent/fresh | has_part | 0.618 [-0.609, 1.845] | 0.602 [-0.694, 1.898] |
| absent/fresh | similar_to | 0.490 [0.323, 0.656] | 0.487 [0.296, 0.677] |
| absent/fresh | located_in | 0.239 [-0.039, 0.517] | 0.150 [0.048, 0.253] |
| collapsed/reused | part_of | 0.966 [0.935, 0.997] | 0.966 [0.935, 0.997] |
| collapsed/reused | member_of | 0.051 [-0.020, 0.121] | 0.044 [-0.054, 0.142] |

Gold precision of the adopted rule's predictions beyond the captured pairs (mean over accepted slots with an adopted rule and a matched relation): absent/reused: riddle 0.823 [0.569, 1.076], AMIE 0.601 [0.200, 1.002]; absent/fresh: riddle 0.823 [0.569, 1.076], AMIE 0.601 [0.200, 1.002]; collapsed/reused: riddle 1.000 [1.000, 1.000], AMIE 1.000 [1.000, 1.000].

CPU seconds by part (sum over jobs): axioms-dev 34, edges-dev 18, recovery 3727, axioms 44, edges 53, discovery 1664; wall 1244 s.
