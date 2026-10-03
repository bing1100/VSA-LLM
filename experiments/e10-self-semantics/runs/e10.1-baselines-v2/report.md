# E10.1-baselines — baselines for claim D (WP-PQ2)

Seeds [11, 22, 33]; dev seeds [7] set the operator-axiom thresholds only (RotatE -0.183, HRR roles 0.879, tuned AMIE 0.889). World and scenarios: `experiments/e10-self-semantics/e10-wordnet.yaml`. Means over seeds with 95% t-intervals in brackets. Baselines see only asserted edges (and, for the learnable ontology, training and validation observations); gold is read by the evaluation only.

## D-B2 Erased-edge recovery on the E10.1 (b) candidate pools

Methods are fit on 90% of the asserted edges; F1 uses a threshold chosen on the other 10% plus distractors. The learnable ontology's row (and the filler-frequency row) is its committed E10.1 run `experiments/e10-self-semantics/runs/e10.1-v1/metrics.jsonl` (same pools; pool sizes match), whose F1 uses its fixed mass ≥ 0.5 rule. With one erased edge per two distractors, accepting every candidate gives F1 = 0.5, so AUC and R-precision are the primary comparison.

| Erased | Method | AUC | R-precision | F1 | ΔAUC vs learnable | ΔF1 vs learnable |
|---|---|---|---|---|---|---|
| 30% | learnable ontology | 0.589 [0.527, 0.651] | 0.428 [0.333, 0.522] | 0.334 [0.234, 0.433] | n/a | n/a |
| 30% | amie | 0.508 [0.504, 0.512] | 0.341 [0.339, 0.343] | 0.500 [0.500, 0.500] | -0.082 [-0.148, -0.016] | 0.166 [0.067, 0.266] |
| 30% | transe | 0.588 [0.554, 0.622] | 0.496 [0.460, 0.531] | 0.512 [0.479, 0.546] | -0.002 [-0.085, 0.081] | 0.179 [0.046, 0.312] |
| 30% | rotate | 0.601 [0.569, 0.633] | 0.468 [0.420, 0.517] | 0.485 [0.454, 0.515] | 0.012 [-0.073, 0.097] | 0.151 [0.022, 0.280] |
| 30% | complex | 0.628 [0.618, 0.638] | 0.453 [0.433, 0.474] | 0.437 [0.418, 0.455] | 0.038 [-0.031, 0.108] | 0.103 [0.015, 0.191] |
| 30% | itere | 0.601 [0.558, 0.644] | 0.490 [0.440, 0.539] | 0.510 [0.460, 0.559] | 0.011 [-0.089, 0.111] | 0.176 [0.027, 0.325] |
| 30% | filler-frequency ranking | 0.429 [0.408, 0.450] | 0.315 [0.297, 0.334] | — | | |
| 50% | learnable ontology | 0.561 [0.545, 0.577] | 0.403 [0.384, 0.422] | 0.224 [0.183, 0.265] | n/a | n/a |
| 50% | amie | 0.507 [0.504, 0.510] | 0.341 [0.336, 0.346] | 0.500 [0.500, 0.500] | -0.054 [-0.072, -0.036] | 0.276 [0.235, 0.317] |
| 50% | transe | 0.570 [0.536, 0.603] | 0.446 [0.439, 0.454] | 0.466 [0.453, 0.479] | 0.009 [-0.037, 0.054] | 0.242 [0.199, 0.285] |
| 50% | rotate | 0.582 [0.569, 0.596] | 0.428 [0.396, 0.461] | 0.435 [0.396, 0.474] | 0.021 [-0.003, 0.046] | 0.211 [0.164, 0.258] |
| 50% | complex | 0.605 [0.565, 0.644] | 0.422 [0.390, 0.453] | 0.407 [0.351, 0.462] | 0.044 [-0.009, 0.096] | 0.182 [0.092, 0.273] |
| 50% | itere | 0.577 [0.542, 0.612] | 0.436 [0.404, 0.467] | 0.464 [0.428, 0.499] | 0.016 [-0.020, 0.052] | 0.240 [0.208, 0.271] |
| 50% | filler-frequency ranking | 0.453 [0.430, 0.475] | 0.297 [0.290, 0.304] | — | | |

## D-B1 Horn axioms on the 30%-erasure graph (vs axioms with confidence ≥ 0.9 on the complete gold graph)

Gold axioms (union over seeds): antonym ∧ antonym ⇒ hypernym, antonym ∧ attribute ⇒ attribute, antonym ∧ lexname ⇒ lexname, antonym ∧ pos ⇒ pos, entailment ∧ pos ⇒ pos, hypernym ∧ lexname ⇒ lexname, hypernym ∧ pos ⇒ pos, part_holonym ∧ lexname ⇒ lexname, part_holonym ∧ pos ⇒ pos, part_meronym ∧ pos ⇒ pos, similar_to ∧ lexname ⇒ lexname.

| Reading | precision | recall | F1 | AUC of the score | axioms accepted | recall: symmetric / inverse / transitive / chain |
|---|---|---|---|---|---|---|
| AMIE (defaults: PCA ≥ 0.1, HC ≥ 0.01) | 0.130 [0.088, 0.172] | 0.273 [0.273, 0.273] | 0.176 [0.137, 0.215] | 0.999 [0.998, 0.999] | 23.3 [15.3, 31.3] | n/a / n/a / n/a / 0.27 [0.27, 0.27] |
| AMIE (default filters and PCA ≥ 0.889, dev-chosen) | 0.393 [0.316, 0.470] | 0.273 [0.273, 0.273] | 0.322 [0.296, 0.347] | 0.999 [0.998, 0.999] | 7.7 [6.2, 9.1] | n/a / n/a / n/a / 0.27 [0.27, 0.27] |
| IterE-style RotatE phases (≥ -0.183, dev-chosen) | 0.001 [0.001, 0.001] | 0.970 [0.839, 1.100] | 0.001 [0.001, 0.002] | 0.569 [0.463, 0.676] | 15727.7 [15546.9, 15908.5] | n/a / n/a / n/a / 0.97 [0.84, 1.10] |

Closure of the accepted axioms over the observed graph (one application; `amie_dev` = support ≥ 2 and PCA ≥ the dev threshold; `rotate` = supported axioms above the dev threshold, at most 50, as injected by the IterE loop):

| Axioms from | axioms | closure triples | recall of erased edges | precision vs gold |
|---|---|---|---|---|
| amie_default | 23.3 [15.3, 31.3] | 5832.0 [3592.6, 8071.4] | 0.021 [0.012, 0.031] | 0.007 [0.006, 0.008] |
| amie_dev | 29.7 [26.8, 32.5] | 50.3 [25.4, 75.2] | 0.001 [-0.001, 0.004] | 0.548 [0.455, 0.641] |
| rotate | 50.0 [50.0, 50.0] | 7685.3 [7231.4, 8139.3] | 0.031 [0.007, 0.056] | 0.008 [0.006, 0.009] |

CPU seconds by part (sum over jobs): axioms-dev 51, edges-dev 170, recovery 5252, axioms 115; wall 764 s.
