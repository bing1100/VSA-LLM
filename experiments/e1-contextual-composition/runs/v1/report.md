# E1 — contextual composition against frozen contextual anchors

Host gpt2, target layer 12 (separability {"6": 0.086, "9": 0.086, "12": 0.092}); 690 lemmas, 45470 sentences, seeds [11, 22, 33].

| Learner | Split | Variance explained | Context MRR | Cosine | Sense acc. | MFS acc. |
|---|---|---:|---:|---:|---:|---:|
| free_sense_p2 | seen_lemmas_heldout_sentences | -0.2391 | 0.2995 | 0.6045 | — | — |
| m0_lemma | heldout_lemmas | -3.5213 | 0.0720 | 0.2189 | — | — |
| m0_lemma | seen_lemmas_heldout_sentences | -2.7274 | 0.2263 | 0.5032 | — | — |
| m0_sense_oracle | heldout_lemmas | -3.5124 | 0.0749 | 0.2485 | — | — |
| m0_sense_oracle | seen_lemmas_heldout_sentences | -3.2101 | 0.2466 | 0.4398 | — | — |
| m1_p0 | heldout_lemmas | -3.4878 | 0.0720 | 0.2218 | 0.2382 | 0.4920 |
| m1_p0 | seen_lemmas_heldout_sentences | -2.7857 | 0.2263 | 0.5026 | 0.2232 | 0.4721 |
| m1_p1 | heldout_lemmas | -3.5825 | 0.1033 | 0.2934 | 0.2342 | 0.4920 |
| m1_p1 | seen_lemmas_heldout_sentences | -3.1589 | 0.3260 | 0.5631 | 0.2159 | 0.4721 |
| m1_p2_diagonal | heldout_lemmas | -2.7987 | 0.1221 | 0.4511 | 0.2493 | 0.4920 |
| m1_p2_diagonal | seen_lemmas_heldout_sentences | -2.1691 | 0.3836 | 0.6458 | 0.2242 | 0.4721 |
| m1_p2_random_fixed_hrr | heldout_lemmas | -3.1075 | 0.1220 | 0.4444 | 0.2467 | 0.4920 |
| m1_p2_random_fixed_hrr | seen_lemmas_heldout_sentences | -2.4913 | 0.3858 | 0.6470 | 0.2232 | 0.4721 |
| m1_p2_untyped | heldout_lemmas | -3.1065 | 0.1245 | 0.4487 | 0.2475 | 0.4920 |
| m1_p2_untyped | seen_lemmas_heldout_sentences | -2.4112 | 0.3814 | 0.6439 | 0.2179 | 0.4721 |
| m1_p2 | heldout_lemmas | -2.8332 | 0.1227 | 0.4452 | 0.2410 | 0.4920 |
| m1_p2 | seen_lemmas_heldout_sentences | -2.3386 | 0.3868 | 0.6469 | 0.2224 | 0.4721 |

Primary `m1_p2` − `m0_lemma` on held-out lemmas: variance explained +0.6881 [+0.4170, +0.9592], context MRR +0.0507 [+0.0489, +0.0526]; sense accuracy − MFS -0.2510 [-0.2901, -0.2119].
**E1 gate: FAIL.**
