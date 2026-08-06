# Experiment 01 — GPT-2 / WordNet matrix-only pilot

Concepts: **1200**; host revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`; WordNet: **3.0**.

| Method | Held-out row cosine | Held-out kNN overlap |
|---|---:|---:|
| typed_hrr | 0.331 | 0.107 |
| typed_map | 0.417 | 0.114 |
| additive | 0.442 | 0.139 |
| shuffled_structure | 0.260 | 0.042 |
| nearest_recipe | 0.325 | 0.099 |
| train_mean | 0.529 | 0.043 |

Typed-HRR kNN gain over **additive**: **-0.032**.
Typed-HRR row-cosine gain over **train_mean**: **-0.197**.
Pilot gate: **FAIL**.

## Protocol

- Frozen GPT-2 input rows at the pinned revision above; the host is never optimized.
- 1,200 lowercase, alphabetic, single-token noun lemmas that map to exactly one WordNet noun synset; one deterministic lemma per synset.
- Recipe: lexicographer class plus four deterministic hypernym ancestors.
- Three 960/240 splits. Entire identical-recipe groups stay in one partition, while every held-out atomic ancestor appears somewhere in training. `splits.csv` records all assignments.
- The target row, aliases, and exact recipe twins never enter the training partition.

## Interpretation

This is a **negative result for typed HRR row synthesis**. Typed HRR loses to the strongest structural control, untyped additive composition, by **0.032 kNN overlap**, and it loses to the mean-row cosine control by **0.197**. MAP improves on HRR but still trails additive composition. Ontology information is nevertheless useful: additive composition reaches 0.139 kNN overlap versus 0.042 after structure shuffling.

The mean baseline's 0.529 row cosine but 0.043 kNN overlap exposes strong common-direction anisotropy; raw cosine alone is not a defensible promotion endpoint. The structural signal is larger among high-degree concepts (additive 0.202 kNN) but typed HRR still does not win there.

Earlier diagnostic runs allowed exact recipe twins across partitions and overstated all structure-aware methods. The group-disjoint correction materially reduced performance and is now a mandatory protocol invariant.

## Decision

**Do not advance this factorizer to experiment 02.** No insertion, tokenizer mutation, or claims of zero-gradient acquisition are justified by this result.

The next discriminating experiment-01 study should test centered/residual host geometry and sense-specific contextual anchors, add equal-budget low-rank/graph-projection baselines, and use relational neighborhoods rather than a single nested hypernym path. Keep additive composition as the control to beat; do not spend compute merely scaling HRR dimensions or steps.
