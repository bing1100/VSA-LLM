# E10.1-smoke — self-learned semantics (H-H)

Seeds [7]; CPU single-threaded jobs; world `wordnet`. Means over seeds with 95% t-intervals in brackets. Gold labels and audit observations are used only by evaluation; every learning-side decision uses held-out self-tests.

## H-H refutation clauses

| Clause | Result |
|---|---|
| (i) erased edges recovered above matched random candidates | {'0.3': False} |
| (ii) blank slots align with hidden relations above chance (ARI − permuted, additive) | {'absent': False, 'collapsed': False} |
| (ii′) same, per-seed permutation test (Holm over seeds; added after the first run, reported alongside) | {'absent': True, 'collapsed': True} |
| (iii) self-acceptance beats random acceptance at the matched rate (edges) | {'erasure-0.3': False} |
| (iii) self-acceptance beats random acceptance at the matched rate (slots, pooled) | False |
| (iv) new-word frame inference beats nearest-neighbour frames (F1) | {'1': False, '6': False} |

## (a) Learnability ablation (30% erased, 5% spurious prior edges, noisy priors)

Contrasts against everything learnable with L2-to-prior (paired over seeds):

| Contrast | Δ held-out-obs fit (train concepts) | Δ test-concept fit (zero-shot) | Δ recovery F1 | Δ recovery AUC |
|---|---|---|---|---|
| atomics=fixed − all l2 | -0.0151 | -0.0095 | 0.155 | -0.031 |
| atomics=free − all l2 | n/a | n/a | n/a | n/a |
| only atomics l2 − all fixed | n/a | n/a | n/a | n/a |
| relations=fixed − all l2 | n/a | n/a | n/a | n/a |
| relations=free − all l2 | n/a | n/a | n/a | n/a |
| only relations l2 − all fixed | n/a | n/a | n/a | n/a |
| mapping=fixed − all l2 | n/a | n/a | n/a | n/a |
| mapping=free − all l2 | n/a | n/a | n/a | n/a |
| only mapping l2 − all fixed | n/a | n/a | n/a | n/a |
| frames=fixed − all l2 | n/a | n/a | n/a | n/a |
| frames=free − all l2 | n/a | n/a | n/a | n/a |
| only frames l2 − all fixed | n/a | n/a | n/a | n/a |
| all free − all l2 | 0.0001 | -0.0096 | -0.036 | 0.034 |
| all fixed − all l2 | -0.1125 | -0.0632 | -0.036 | -0.050 |

Selected grid cells:

| atomics / relations / mapping / frames | val fit | test fit | recovery F1 | AUC | atomic cos | spurious removed |
|---|---:|---:|---:|---:|---:|---:|
| l2 / l2 / l2 / l2 | 0.7862 | 0.1018 | 0.036 | 0.550 | n/a | 0.000 |
| free / free / free / free | 0.7863 | 0.0922 | 0.000 | 0.583 | n/a | 0.000 |
| fixed / fixed / fixed / fixed | 0.6738 | 0.0386 | 0.000 | 0.500 | n/a | 0.000 |
| fixed / l2 / l2 / l2 | 0.7711 | 0.0923 | 0.190 | 0.519 | n/a | 0.000 |

Best test-concept fit in the full 3⁴ grid: `l2|l2|l2|l2`.

## (b) Erasure & recovery

| Erased (principal fixed) | Precision | Recall | F1 | AUC | R-precision | Random (prevalence / AUC 0.5) | Filler-frequency AUC / R-prec | Test fit |
|---|---|---|---|---|---|---|---|---|
| 30% (70%) | n/a | 0.000 | 0.000 | 0.580 | 0.455 | 0.335 / 0.5 | 0.238 / 0.036 | 0.1213 |

## (c) Blank-relation discovery — hidden relations absent

| Method | ARI (gold edges) | ARI − permuted | permutation p (Holm, max over seeds) | mean best Jaccard | detection P | detection R | accepted / proposed slots | test fit |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.643 | 0.641 | 0.005 | 0.482 | 0.957 | 0.595 | n/a / n/a | 0.1041 |
| additive | 0.212 | 0.211 | 0.010 | 0.150 | 0.548 | 0.459 | 1.0 / 2.0 | 0.1130 |
| all_at_once | 0.078 | 0.082 | 0.055 | 0.067 | 0.333 | 0.027 | 1.0 / 5.0 | 0.1024 |
| m3 | 0.212 | 0.211 | 0.010 | 0.158 | 0.586 | 0.459 | n/a / n/a | 0.1203 |
| stem_cell | 0.063 | 0.067 | 0.144 | 0.167 | 0.586 | 0.459 | n/a / n/a | 0.1014 |

Per hidden relation, best Jaccard of a discovered cluster (candidate level):

| Method | similar_to | antonym | instance_hypernym |
|---|---|---|---|
| oracle | 0.864 | 0.083 | 0.500 |
| additive | 0.395 | 0.024 | 0.030 |
| all_at_once | 0.000 | 0.000 | 0.200 |
| m3 | 0.417 | 0.025 | 0.032 |
| stem_cell | 0.250 | 0.000 | 0.250 |

Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive: instance_hypernym 0.030; antonym 0.020; similar_to 0.283.
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), all_at_once: instance_hypernym 0.250; antonym 0.000; similar_to 0.000.

- additive − all_at_once: ARI 0.133, mean best Jaccard 0.083, test fit 0.0106
- additive − m3: ARI 0.000, mean best Jaccard -0.008, test fit -0.0074
- additive − stem_cell: ARI 0.148, mean best Jaccard -0.017, test fit 0.0115

**Riddle-style interpretation (E10.6, absent).** 2 accepted slots, 1 adopted a structural rule; 1 of those rules are true of the matched hidden relation, 0 are its designed property. Rule predictions' gold precision n/a. Hypotheses scored per slot (full space): 43.0.

| Matched hidden relation | slots | adopted | true | designed | rules adopted |
|---|---:|---:|---:|---:|---|
| similar_to | 1 | 1 | 1 | 0 | one_to_one×1 |
| instance_hypernym | 1 | 0 | 0 | 0 |  |

Accuracy of the adopted hypothesis vs number of candidate hypotheses (one true + m−1 false):

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| slots | 2 | 2 | 2 | 2 | 2 | 2 |

**Human-likeness (additive, absent).** Over 1 slots: purity of a slot's members w.r.t. its final hidden relation rises from 0.381 (first committed members) to 0.469; hidden relations present 3.00 → 3.00; members 42.0 (peak 42.0) → 32.0. Slots that overextended then refined: 1.

Soft hypothesis of a newly opened slot (mass-weighted shares of the pairs it holds; steps since opening):

| step | target relation | other hidden relations | distractors | effective size |
|---:|---:|---:|---:|---:|
| 0 | 0.30 | 0.20 | 0.50 | 73 |
| 25 | 0.33 | 0.18 | 0.49 | 65 |
| 50 | 0.35 | 0.16 | 0.49 | 59 |
| 100 | 0.36 | 0.16 | 0.48 | 56 |

Slot self-acceptance (absent; additive, no-rule and all-at-once runs pooled): accuracy 0.714 (Wilson 0.36–0.92, n = 7) vs random at the matched rate 0.469 (p = 0.158); accept-all 0.571; wrong-acceptance rate 0.000.

## (c) Blank-relation discovery — hidden relations collapsed

| Method | ARI (gold edges) | ARI − permuted | permutation p (Holm, max over seeds) | mean best Jaccard | detection P | detection R | accepted / proposed slots | test fit |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.883 | 0.889 | 0.005 | 0.976 | 1.000 | 1.000 | n/a / n/a | 0.1150 |
| additive | 0.305 | 0.304 | 0.005 | 0.266 | 1.000 | 1.000 | 1.0 / 2.0 | 0.1265 |
| all_at_once | -0.003 | -0.003 | 0.478 | 0.191 | 1.000 | 0.596 | 4.0 / 5.0 | 0.1147 |
| m3 | 0.345 | 0.347 | 0.005 | 0.285 | 1.000 | 1.000 | n/a / n/a | 0.1425 |
| stem_cell | 0.115 | 0.118 | 0.035 | 0.284 | 1.000 | 1.000 | n/a / n/a | 0.1143 |

Per hidden relation, best Jaccard of a discovered cluster (candidate level):

| Method | part_holonym | part_meronym | substance_meronym | member_meronym | member_holonym | substance_holonym |
|---|---|---|---|---|---|---|
| oracle | 0.857 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| additive | 0.529 | 0.649 | 0.069 | 0.069 | 0.214 | 0.067 |
| all_at_once | 0.143 | 0.176 | 0.333 | 0.125 | 0.167 | 0.200 |
| m3 | 0.667 | 0.632 | 0.103 | 0.083 | 0.167 | 0.056 |
| stem_cell | 0.250 | 0.410 | 0.333 | 0.333 | 0.333 | 0.045 |

Relation recovery over all framed heads (accepted slots incl. rule-implied edges), additive: part_meronym 0.231; member_meronym 0.059; substance_meronym 0.059; part_holonym 0.115; member_holonym 0.000; substance_holonym 0.067.
Relation recovery over all framed heads (accepted slots incl. rule-implied edges), all_at_once: part_meronym 0.147; member_meronym 0.125; substance_meronym 0.333; part_holonym 0.105; member_holonym 0.000; substance_holonym 0.200.

- additive − all_at_once: ARI 0.307, mean best Jaccard 0.075, test fit 0.0118
- additive − m3: ARI -0.041, mean best Jaccard -0.018, test fit -0.0160
- additive − stem_cell: ARI 0.189, mean best Jaccard -0.018, test fit 0.0122

**Riddle-style interpretation (E10.6, collapsed).** 5 accepted slots, 1 adopted a structural rule; 0 of those rules are true of the matched hidden relation, 0 are its designed property. Rule predictions' gold precision n/a. Hypotheses scored per slot (full space): 45.2.

| Matched hidden relation | slots | adopted | true | designed | rules adopted |
|---|---:|---:|---:|---:|---|
| part_meronym | 3 | 1 | 0 | 0 | sub_relation_of:meronym×1 |
| substance_meronym | 1 | 0 | 0 | 0 |  |
| substance_holonym | 1 | 0 | 0 | 0 |  |

Accuracy of the adopted hypothesis vs number of candidate hypotheses (one true + m−1 false):

| m | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| slots | 1 | 1 | 1 | 1 | 1 | 1 |

**Human-likeness (additive, collapsed).** Over 1 slots: purity of a slot's members w.r.t. its final hidden relation rises from 0.588 (first committed members) to 0.600; hidden relations present 5.00 → 5.00; members 17.0 (peak 17.0) → 15.0. Slots that overextended then refined: 0.

Soft hypothesis of a newly opened slot (mass-weighted shares of the pairs it holds; steps since opening):

| step | target relation | other hidden relations | distractors | effective size |
|---:|---:|---:|---:|---:|
| 0 | 0.61 | 0.39 | 0.00 | 57 |
| 25 | 0.60 | 0.40 | 0.00 | 56 |
| 50 | 0.60 | 0.40 | 0.00 | 55 |
| 100 | 0.60 | 0.40 | 0.00 | 45 |

Slot self-acceptance (collapsed; additive, no-rule and all-at-once runs pooled): accuracy 0.429 (Wilson 0.16–0.75, n = 7) vs random at the matched rate 0.347 (p = 0.457); accept-all 0.143; wrong-acceptance rate 0.800.

## (d) Self-tested acceptance of edge hypotheses (held-out observations; audit never read)

| Source | n | accept rate | accuracy (Wilson) | random @ matched rate | accept-all | in-sample (confirmation-bias control) | acc − random by seed | audit Δ accepted / rejected |
|---|---:|---:|---|---:|---:|---|---|---|
| erasure-0.3 | 76 | 0.70 | 0.526 (0.42–0.63) | 0.453 | 0.382 | 0.526 (accepts 0.67) | 0.073 | +0.0987 / +0.0028 |

## (e) New-word frame inference (fast mapping)

| k | discovered F1 | known-only F1 | oracle-dictionary F1 | nearest-neighbour F1 | random F1 | discovered fit | NN fit | context-mean fit | gold-frame fit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.006 | 0.010 | n/a | 0.139 | 0.000 | 0.4372 | 0.3842 | 0.5036 | 0.0269 |
| 6 | 0.007 | 0.000 | n/a | 0.196 | 0.000 | 0.5921 | 0.5260 | 0.7191 | 0.0269 |

- k = 1: discovered − nearest_neighbour (F1) -0.133; discovered − known_only (F1) -0.004; discovered − random (F1) 0.006; discovered − nearest_neighbour (fit) 0.053; discovered − context_mean (fit) -0.066
- k = 6: discovered − nearest_neighbour (F1) -0.189; discovered − known_only (F1) 0.007; discovered − random (F1) 0.007; discovered − nearest_neighbour (fit) 0.066; discovered − context_mean (fit) -0.127

Discovered-relation F1 by hierarchy depth at the largest k (exploratory 'basic-level' readout): depth 0: 0.042; depth 1: 0.000; depth 10: 0.000; depth 11: 0.000; depth 2: 0.000; depth 3: 0.000; depth 4: 0.000; depth 5: 0.000; depth 6: 0.000; depth 7: 0.000; depth 8: 0.000; depth 9: 0.000.

## E10.4 Seed ontology → recovered logical ontology

| Start | edge F1 (all framed) | P / R | candidate AUC | candidate-level relation Jaccard (mean) | relations recovered (J ≥ 0.5, of 7) | accepted slots | multi-hop is-a acc (gold) | transitivity located_in (gold) | inverse part (gold) | symmetry similar (gold) | test fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| core | 0.143 | 0.31 / 0.09 | 0.476 | 0.005 | 0.0 | 1.0 | 0.50 (1.00) | n/a (n/a) | 0.00 (0.00) | n/a (0.00) | 0.0220 |
| full | 1.000 | 1.00 / 1.00 | n/a | n/a | 16.0 | 0.0 | 1.00 (1.00) | n/a (n/a) | 0.00 (0.00) | 0.00 (0.00) | 0.1488 |

## E10.5 Dreaming (offline self-revision) after injected corruptions

| Dream every (steps) | passes | wrong edges repaired | correct edges damaged | merged relations split (ARI) | wrong slot repaired | wrong slot reopened/removed | correct slots disturbed | audit fit (train concepts) | applied revisions |
|---|---:|---|---|---|---|---|---|---|---|
| off | 0 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.6568 | {} |
| 200 | 1 | 0.11 | 0.117 | -0.13 | 0.09 | 0.00 | 0.00 | 0.6559 | {'remove_edges': 112, 'relabel_edges': 99, 'split': 1} |

Before any revision (right after injection): audit fit 0.5797 [0.5797, 0.5797].

## E10.8 Continual additive learning (hidden relations arrive one per stage)

| Condition | relations matched (Jaccard ≥ 0.5 on offered candidate pairs) by stage | final | final Jaccard on offered pairs | final Jaccard over all framed heads (incl. rule closure) | retention drop (arrival − final) | final val fit |
|---|---|---|---|---|---|---|
| additive | 0.00 → 0.00 → 0.00 | 0.00 | antonym 0.13; similar_to 0.09; instance_hypernym 0.00 | antonym 0.08; similar_to 0.05; instance_hypernym 0.00 | antonym 0.01; similar_to 0.00; instance_hypernym 0.00 | 0.7859 |
| all_at_once | 0.00 | 0.00 | antonym 0.00; similar_to 0.00; instance_hypernym 0.00 | antonym 0.00; similar_to 0.00; instance_hypernym 0.00 | antonym 0.00; similar_to 0.00; instance_hypernym 0.00 | 0.6953 |

CPU seconds by part (sum over jobs): a 18, b 11, c 115, seed 22, dream 44, continual 30.
