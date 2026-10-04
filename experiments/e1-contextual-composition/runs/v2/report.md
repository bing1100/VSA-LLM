# E1 — contextual composition against frozen contextual anchors

Host gpt2, target layer 12 (separability {"6": 0.086, "9": 0.086, "12": 0.092}); 690 lemmas, 45470 sentences, seeds [11, 22, 33].

| Learner | Split | Variance explained | Context MRR | Cosine | Sense acc. | MFS acc. |
|---|---|---:|---:|---:|---:|---:|
| free_sense_p2 | seen_lemmas_heldout_sentences | -0.2391 | 0.2995 | 0.6045 | — | — |
| m0_lemma | heldout_lemmas | -3.5213 | 0.0720 | 0.2189 | — | — |
| m0_lemma | seen_lemmas_heldout_sentences | -2.7274 | 0.2262 | 0.5032 | — | — |
| m0_sense_oracle | heldout_lemmas | -3.5124 | 0.0749 | 0.2485 | — | — |
| m0_sense_oracle | seen_lemmas_heldout_sentences | -3.2101 | 0.2466 | 0.4398 | — | — |
| m1_p0 | heldout_lemmas | -3.4878 | 0.0720 | 0.2218 | 0.2382 | 0.4920 |
| m1_p0 | seen_lemmas_heldout_sentences | -2.7857 | 0.2263 | 0.5026 | 0.2232 | 0.4721 |
| m1_p1 | heldout_lemmas | -3.5964 | 0.1034 | 0.2928 | 0.2339 | 0.4920 |
| m1_p1 | seen_lemmas_heldout_sentences | -3.1612 | 0.3268 | 0.5637 | 0.2161 | 0.4721 |
| m1_p2_diagonal | heldout_lemmas | -2.8019 | 0.1231 | 0.4484 | 0.2510 | 0.4920 |
| m1_p2_diagonal | seen_lemmas_heldout_sentences | -2.1534 | 0.3814 | 0.6432 | 0.2215 | 0.4721 |
| m1_p2_random_fixed_hrr | heldout_lemmas | -3.1223 | 0.1224 | 0.4371 | 0.2396 | 0.4920 |
| m1_p2_random_fixed_hrr | seen_lemmas_heldout_sentences | -2.5123 | 0.3852 | 0.6486 | 0.2209 | 0.4721 |
| m1_p2_untyped | heldout_lemmas | -3.1110 | 0.1210 | 0.4441 | 0.2521 | 0.4920 |
| m1_p2_untyped | seen_lemmas_heldout_sentences | -2.3861 | 0.3834 | 0.6459 | 0.2211 | 0.4721 |
| m1_p2 | heldout_lemmas | -2.8613 | 0.1230 | 0.4451 | 0.2477 | 0.4920 |
| m1_p2 | seen_lemmas_heldout_sentences | -2.3247 | 0.3888 | 0.6513 | 0.2219 | 0.4721 |

Primary `m1_p2` − `m0_lemma` on held-out lemmas: variance explained +0.6600 [+0.3564, +0.9637], context MRR +0.0510 [+0.0496, +0.0523]; sense accuracy − MFS -0.2443 [-0.2872, -0.2015].
**E1 gate: FAIL.**
