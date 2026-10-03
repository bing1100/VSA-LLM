# R10 — Self-learned semantics (H-H): E10.0 synthetic, E10.1 set-up

**Date:** 2026-10-02. **Status:** E10.0 complete (synthetic planted ontology, CPU, seeds 101/202/303, 81 jobs, 648 s wall on 4 single-threaded workers, 0.71 CPU-h). E10.1 (WordNet, frozen GPT-2 anchors) implemented, smoke-tested on a 300-concept slice, and ready to queue. E10.2/E10.3 come after G3 and E9/E7. **Run folders:** `experiments/e10-self-semantics/runs/e10.0-v3/` (E10.0; full tables in its `report.md`, rows in `metrics.jsonl`) and `runs/e10.1-smoke/` (pipeline check only, not a result). **Code:** `vsa_embed.learnable_ontology`, `vsa_embed.ontology_hypotheses`, `vsa_embed.self_test`, `vsa_embed.ontology_discovery`, `vsa_embed.frame_inference`, `vsa_embed.synthetic_ontology`, `vsa_embed.experiments.e10_self_semantics` (+ `e10_report`, `e10_wordnet`).

**H-H** (execution.md §E10): a model whose ontology mapping is learnable, regularised toward a curated prior, can (i) recover erased parts, (ii) discover relations it was never given through blank slots, interpret them and consolidate them additively, (iii) accept or reject its own hypotheses by self-constructed held-out tests, and (iv) map a new word zero-shot through the relations it has disentangled. **Principle (author, 2026-10-02):** learning is self-supervised. No gold label, gold pair or audit observation enters the learning loop. Gold is used only by the evaluation code, and the audit split only for the confirmation-bias control.

## 1. What was built

- **Learnable-ontology composer** (`LearnableOntologyComposer`, a `FrameComposer` subclass). Every concept has asserted edges (the curated prior, mass prior 1) and candidate edges (mass prior 0). Edge masses are learnable, with L2-to-prior on asserted masses (the *mapping*), and L1 plus projection to [0, 2] on candidate masses (the *frames*). A candidate at mass 0 contributes nothing. The first gradient on its mass is minus its first-order utility (formulation §5.2), so L1 acts as a utility threshold.
- **Open edges and blank slots.** An open edge carries a soft assignment over relation options: seed relations and/or blank slots. A blank slot is a vector-per-relation operator with no prior (unit norm); edges are bound with the π-mixed role. Assignments are free per-edge logits (softmax), or for the stem-cell baseline a router on the filler (sparsemax). An entropy prior pushes commitment.
- **Slot operations.** `crystallize` freezes the operator and hard-assigns committed edges as asserted edges; `absorb` adds late members; `reopen` reverses crystallization; `add_blank_slot`; `add_edges` adds rule-implied edges.
- **Learnability switches.** Per component {atomics, relations, mapping, frames} × {fixed, l2, free}.
- **Equivalence.** With no candidates, masses fixed at 1 and no relation scaling, the composer equals `FrameComposer` in bundle, salience and attentive (incl. hybrid) modes to ≤ 3e-8 (tested to 1e-6). Existing modules are untouched, so recorded runs replay unchanged.
- **Interpretation, self-supervised.** A slot's captured pairs give *structural* signatures: symmetry, antisymmetry, transitivity, functionality, injectivity, overlap with the inverse of / composition of / containment in the model's own relations. The riddle step (E10.6) generates structural hypotheses from the captured pairs (in-sample). Each hypothesis is tested by its predictions on unseen pairs (should / should-not hold), scored as held-out utility against matched random-tail controls with a cluster-bootstrap lower bound. The best passing hypothesis is adopted and *enforced* at crystallization: symmetric / transitive / inverse / composition closures are added as edges, functional / one-to-one violations pruned. Relation names never enter. The description is template text, with a `namer` hook for E10.3 (host-written natural-language names).
- **Self-tests.**
  - Edge test: removal/addition effect on the concept's validation observations, bootstrap over observations.
  - Slot test: (should hold) utility of its edges, cluster-bootstrapped over heads, **and** (should not hold) beating the same heads with random tails under the same operator.
  - In every case: accept iff the lower bound is > 0.
  - Structural isolation: the learning loop only receives a `LearningContext` (training observations + validation observations). A test runs the whole curriculum with all audit observations set to NaN and gets bit-identical decisions, masses and slot operators.
- **Procedures.**
  - Additive curriculum: open one slot → train → absorb late members into earlier relations → interpret → self-test → crystallize with rule, or close; stop after a rejection.
  - All slots at once, at the same step budget.
  - Dreaming pass (E10.5): remove or relabel asserted edges; split, merge, reopen or remove relations. Each proposal is refit against an equally long control refit and accepted only if held-out fit improves (lower bound > 0).
  - Utility-threshold revisit: reopen a crystallized slot if removing it improves held-out fit with lower bound > 0.
  - New-word frame inference: non-negative OMP over (relation option × filler) dictionaries, with everything frozen.
- **Baselines.**
  - M3 splitting: candidates typed under one "unknown" relation that M3 may split, with the E0 permutation null, parent fallback and consolidation.
  - The 01d stem-cell pool: 20 slots, filler router, sparsemax.
  - Oracle labels; random candidates (prevalence, AUC 0.5); filler-frequency ranking.
  - Random acceptance at the matched rate; in-sample acceptance (the confirmation-bias control); nearest-neighbour and random frames; the oracle (teacher) dictionary.

## 2. E10.0 world and protocol

The planted world (`make_ontology_world`):

- **Concepts and atomics.** 304 concepts in a 4-level is_a tree. 344 atomics: one identity atomic per concept plus 40 attributes.
- **Seven relations with designed logic.**

  | Relation | Planted logic |
  |---|---|
  | is_a | functional, antisymmetric |
  | has_attribute | inherited down the tree |
  | part_of | functional |
  | has_part | inverse of part_of |
  | member_of | functional |
  | similar_to | symmetric sibling pairs |
  | located_in | transitive closure of a place tree |

  ≈ 1,330 edges in total.
- **Teacher.** HRR, d = 64, per-relation salience × per-edge jitter.
- **Observations.** Each concept is seen through 12 noisy observations (cos ≈ 0.86 to its clean target): train 0–3, validation (self-test) 4–7, audit 8–11.
- **Priors.** Atomic and relation priors are noisy truth (cos 0.8), a stand-in for pretrained vectors.
- **Splits.** Training 61%; validation, test (composition-disjoint, zero-shot) and new-word concepts 13% each.
- **Defaults.** Every component learnable with L2-to-prior; L1 0.02; full-batch Adam (lr 0.02); 800 steps per slot round.
- **Hyper-parameter selection.** Chosen on dev seeds 7–9 only: L1 strength, round length, refine on/off, absorb.

## 3. Results (E10.0, 3 seeds; mean [95% t-interval over seeds])

### 3.1 (a) Which components need to be learnable (30% erased, 5% spurious prior edges, noisy priors; full 3⁴ grid)

Reference cell, everything learnable with L2-to-prior (`l2/l2/l2/l2`):

| Readout | Value |
|---|---|
| held-out-observation fit of training concepts | 0.803 [0.799, 0.808] |
| zero-shot fit of test concepts | 0.615 [0.567, 0.662] |
| recovery F1 of erased edges | 0.840 |
| recovery AUC | 0.986 |

Paired differences to that reference:

| Change | Δ held-out-obs fit | Δ test-concept fit (zero-shot) | Δ recovery F1 |
|---|---|---|---|
| atomics fixed | −0.102 [−0.113, −0.090] | −0.065 [−0.089, −0.040] | −0.007 (n.s.) |
| atomics free (no L2) | +0.019 [+0.015, +0.023] | **−0.319 [−0.370, −0.268]** | −0.817 [−0.838, −0.796] |
| relations fixed | −0.057 [−0.067, −0.047] | −0.079 [−0.119, −0.038] | −0.026 [−0.047, −0.006] |
| relations free | +0.001 | +0.000 | +0.006 (n.s.) |
| mapping (asserted masses + relation scales) fixed | −0.013 [−0.016, −0.009] | −0.017 [−0.024, −0.009] | +0.010 (n.s.) |
| mapping free | +0.005 | −0.080 [−0.130, −0.031] | −0.691 [−0.915, −0.466] |
| frames (candidates) fixed | −0.093 [−0.103, −0.083] | −0.022 [−0.026, −0.019] | −0.840 (no recovery possible) |
| frames free (no L1) | +0.002 | −0.000 | +0.105 [+0.086, +0.124] |
| everything free | +0.019 [+0.016, +0.023] | **−0.121 [−0.130, −0.112]** | −0.588 [−0.646, −0.531] |
| everything fixed | −0.386 [−0.403, −0.368] | −0.185 [−0.205, −0.165] | −0.840 |

Every component pays for itself when it is learnable *with* an L2 pull to the prior. Leave-one-out costs most for atomics (−0.10 held-out fit) and relations (−0.06 held-out, −0.08 zero-shot). Removing the prior pull is the expensive mistake for transfer. Free atomics drift to cosine 0.57 with the truth (prior 0.80; L2 keeps them at 0.83), which slightly improves the fit to training concepts but costs −0.32 in zero-shot composition of unseen concepts. A free mapping lets asserted masses absorb the scale and kills recovery. This is the author's L2-regularisation idea, confirmed at the synthetic level: L2-to-prior beats both freezing and free learning. The best zero-shot cell of the 81 (`l2/free/l2/free`, 0.6151) ties the reference. One exception: with *typed* candidates (relation known), L1 on frames is not needed (free frames +0.10 F1). The fit alone pushes wrong candidates to 0. Spurious *asserted* edges survive training: only 10% fall below mass 0.5 under the L2-to-1 mapping prior. Removing them needs the explicit revision pass (E10.5): dreaming removes about half of injected wrong edges.

### 3.2 (b) Erasure & recovery (fixed principal portion = 1 − erased)

| Erased (fixed) | Precision | Recall | F1 | AUC | R-precision | Random | Filler-frequency AUC / R-prec | Test-concept fit |
|---|---|---|---|---|---|---|---|---|
| 10% (90%) | 0.995 | 0.825 [0.692, 0.958] | 0.902 [0.817, 0.986] | 0.989 [0.988, 0.991] | 0.965 | 0.333 / 0.5 | 0.519 / 0.376 | 0.729 |
| 30% (70%) | 0.998 | 0.710 [0.696, 0.724] | 0.830 [0.819, 0.840] | 0.987 [0.981, 0.992] | 0.959 | 0.333 / 0.5 | 0.515 / 0.370 | 0.626 |
| 50% (50%) | 1.000 | 0.588 [0.564, 0.611] | 0.740 [0.722, 0.759] | 0.986 [0.985, 0.988] | 0.951 | 0.333 / 0.5 | 0.504 / 0.360 | 0.531 |
| 70% (30%) | 1.000 | 0.381 [0.250, 0.512] | 0.550 [0.411, 0.690] | 0.984 [0.973, 0.996] | 0.929 | 0.333 / 0.5 | 0.519 / 0.371 | 0.394 |

Candidates are the erased edge plus two distractors per erased edge: a random filler of the same relation, or the same filler under another relation. Ranking is nearly perfect at every erasure level (AUC ≈ 0.98–0.99 vs 0.5; R-precision 0.93–0.97 vs 0.33 random and ≈ 0.37 for the ontology-statistics baseline). Committed recovery (mass ≥ 0.5) is conservative: precision ≈ 1.0, recall falls from 0.83 to 0.38 as the fixed portion shrinks. Zero-shot fit of test concepts falls with erasure because their own frames are erased too and they have no candidates.

### 3.3 (c) Blank-relation discovery

*Hidden relations absent:* has_part, similar_to and located_in are removed. 70% of their training-head edges are offered as open candidates among as many same-type distractors.

| Method | ARI on hidden-relation edges | permutation p (Holm, max over seeds) | mean best Jaccard | detection P / R | accepted / proposed slots | test-concept fit |
|---|---|---|---|---|---|---|
| oracle labels | 0.807 [0.683, 0.931] | 0.015 | 0.839 | 0.96 / 0.88 | — | 0.684 |
| **additive curriculum** | 0.386 [−0.005, 0.778] | 0.015 | **0.441 [0.302, 0.580]** | 0.87 / 0.78 | 3.3 / 4.3 | **0.699** |
| additive, no rule enforcement | 0.385 | 0.015 | 0.416 | 0.86 / 0.79 | 3.7 / 4.3 | 0.683 |
| all slots at once (5, same steps) | 0.273 [0.143, 0.403] | 0.015 | 0.330 | 0.89 / 0.65 | 5.0 / 5.0 | 0.690 |
| M3 splitting of an "unknown" relation | 0.188 [−0.180, 0.556] | — | 0.236 | 0.88 / 0.58 | — | 0.670 |
| stem-cell pool (20, router) | 0.140 [−0.059, 0.339] | — | 0.215 | 0.89 / 0.63 | — | 0.680 |

- Additive minus M3: mean best Jaccard +0.205 [+0.069, +0.341].
- Additive minus stem cell: +0.226 [+0.105, +0.347].
- Additive minus all at once: +0.111 [−0.224, +0.446] (n.s.).
- Per seed, the additive ARI is 0.44 / 0.51 / 0.21. Every seed beats the permuted-label null (p = 0.005, the floor at 200 permutations).
- The across-seed t-interval of ARI − permuted (pre-written clause (ii)) narrowly includes 0, because ARI varies between seeds.
- Per relation (best Jaccard, candidate level), additive: similar_to 0.62, located_in 0.44, has_part 0.27. Oracle: 0.86 / 0.86 / 0.80.
- **Logical rules complete relations.** Counting the rule-implied edges added at crystallisation, additive recovers has_part at Jaccard 0.62 over *all* framed heads (no-rule: 0.12) and similar_to at 0.49 (no-rule 0.34). Relation recovery with minus without rules: +0.22 [−0.19, +0.62]; zero-shot test fit +0.016.

*Collapsed sub-types:* part_of and member_of are merged into one `meronym` label (the E3/M3-native setting).

| Method | ARI | mean best Jaccard | accepted slots | test fit |
|---|---|---|---|---|
| oracle | 1.000 | 1.000 | — | 0.760 |
| **additive** | **0.416 [0.361, 0.471]** | **0.653 [0.589, 0.717]** | 2.0 / 3.0 | 0.750 |
| additive, no rules | 0.381 | 0.652 | 1.7 / 2.7 | 0.749 |
| all at once | 0.190 [0.022, 0.357] | 0.392 | 3.0 / 5.0 | 0.745 |
| M3 | 0.363 [0.239, 0.486] | 0.631 | — | 0.717 |
| stem cell | 0.133 | 0.299 | — | 0.743 |

- Additive minus all at once: ARI +0.226 [+0.095, +0.358].
- Additive vs M3: equivalent on alignment (+0.05 [−0.08, +0.18]), and better zero-shot fit (+0.033 [+0.018, +0.047]).
- The part_of slot adopts `inverse_of:has_part` in 6 of 7 slots and is then rule-completed (part_of recovery 0.97 over framed heads vs 0.67 without rules; rules +0.16 [+0.09, +0.24]).

**Riddle-style interpretation (E10.6).** No names are used: hypotheses are structural and refer only to the model's own relations.

- Absent mode: 25 accepted slots. 14 adopted a rule; 13 of those are true of the matched hidden relation and 12 are its designed property:
  - symmetric ×7 for similar_to;
  - inverse_of:part_of ×5 for has_part;
  - one wrong adoption, `functional` on a located_in fragment.
- located_in fragments never accumulate enough 2-paths for `transitive` to be generated in-sample. Rule predictions have gold precision 0.87 [0.77, 0.96].
- Accuracy of the adopted hypothesis when the designed one competes with m − 1 false ones is flat at 0.55–0.58 for m = 1…16 and drops to 0.39 at m = 32. The remaining error is the designed hypothesis *not passing*, not being out-competed.
- Collapsed mode: the curve is 0.14, because member_of's designed properties (functional, antisymmetric) only make "should not hold" predictions, which rarely clear the bar.

**Human-likeness: fast commitment, transient overextension.**
- Over the first 25 steps after a blank slot opens (soft mass shares, absent mode, the over-inclusive start is set by the 0.1 initial mass), the target relation's share rises 0.15 → 0.43 while other hidden relations *rise* from 0.15 to 0.20. They then fall to 0.05 by step 400, and the effective size shrinks from 172 to 31 pairs.
- Committed members: purity w.r.t. the final relation rises 0.43 → 0.55 and the number of hidden relations present falls 1.54 → 1.31. 8 of 13 slots overextended and then refined.
- located_in and has_part are frequently lumped early. Fragmentation of a weak relation (located_in, lowest teacher weight on some seeds) into several accepted slots is the main failure mode; duplicates are not merged in (c).

### 3.4 (d) Self-tested acceptance (held-out observations only; audit split never read — tested)

| Hypotheses | n | accept rate | self-test accuracy | random at matched rate | accept all | in-sample test (confirmation-bias control) | accuracy − random, by seed | audit Δfit accepted / rejected |
|---|---:|---:|---|---:|---:|---|---|---|
| erased-edge proposals, 30% erasure | 840 | 0.92 | 0.927 (0.91–0.94) | 0.824 | 0.890 | 0.898 (accepts 99%) | +0.103 [+0.082, +0.124] | +0.099 / +0.007 |
| erased-edge proposals, 50% erasure | 1,368 | 0.92 | 0.925 (0.91–0.94) | 0.825 | 0.889 | 0.901 (accepts 98%) | +0.098 [+0.032, +0.164] | +0.100 / +0.007 |
| slot (relation) hypotheses, absent mode | 41 | 0.88 | 0.805 (0.66–0.90) | 0.675 (p = 0.008) | 0.732 | — | — | — |
| slot hypotheses, collapsed mode | 32 | 0.63 | 0.688 (0.51–0.82) | 0.594 (p = 0.17) | 0.875 | — | — | — |

The held-out self-test beats random acceptance at its own rate, accept-all, and the in-sample test, which accepts nearly everything (confirmation bias). On the hidden audit split, edges it accepts help (+0.10 cosine) and edges it rejects do not (+0.007). For relation hypotheses it is better than chance in the absent setting but not significantly so for collapsed sub-types (gold verdict: precision ≥ 0.5 and purity ≥ 0.7).

### 3.5 (e) New-word frame inference (fast mapping; everything frozen; non-negative OMP over known + crystallized relations × all atomics)

| k observations | discovered relations F1 | seed relations only | oracle dictionary (teacher vectors) | nearest-neighbour frame | random frame | zero-shot fit of the inferred frame | NN-frame fit | mean of the k observations |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.417 [0.329, 0.506] | 0.403 | 0.739 | 0.344 | 0.000 | 0.630 | 0.423 | 0.737 |
| 2 | 0.543 [0.483, 0.604] | 0.501 | 0.923 | 0.361 | 0.000 | 0.680 | 0.438 | 0.792 |
| 4 | 0.639 [0.542, 0.736] | 0.565 | 0.952 | 0.363 | 0.002 | 0.710 | 0.435 | 0.824 |
| 8 | 0.685 [0.598, 0.773] | 0.585 | 0.971 | 0.361 | 0.001 | 0.726 | 0.438 | 0.841 |

- Frame F1 vs nearest-neighbour frames: +0.07 [+0.01, +0.14] at k = 1, rising to +0.32 [+0.24, +0.41] at k = 8. This is a fast-mapping curve: most of the gain comes in the first 2–4 exposures.
- The relations discovered by the curriculum add +0.01 to +0.10 F1 over seed relations alone (n.s. at 3 seeds).
- The gap to the oracle dictionary (0.97 at k = 8) shows that the learned atomics (cosine ≈ 0.83 to truth), not the inference, limit identifiability.
- Zero-shot composition of the inferred frame (uniform masses) reaches 0.73 at k = 8, close to composing the gold frame with the learned dictionary (0.71). It stays below simply averaging the k observations (0.84): the frame denoises less than averaging, but it is a *symbolic* entry usable by the rest of the ontology.
- "Basic-level" readout (exploratory): F1 by depth 1.00 / 0.91 / 0.79 / 0.65 for depths 0–3. Shallower concepts have smaller frames; this is not a perceptual basic-level effect.

### 3.6 E10.4 Seed ontology → recovered logical ontology

The additive curriculum (≤ 9 slots, patience 2, settle phase) is run from five starting ontologies. "Noisy" is the stand-in for a host-authored seed: 30% of the curated edges plus 20% wrong edges. Relation labels are matched to gold after the fact.

| Start | edge F1 (framed heads) | P / R | candidate AUC | candidate-level relation Jaccard | relations recovered (J ≥ 0.5, of 7) | accepted slots | multi-hop is-a accuracy | inverse consistency part_of/has_part | symmetry similar_to |
|---|---|---|---|---|---|---|---|---|---|
| empty | 0.126 [0.083, 0.169] | 0.40 / 0.07 | 0.62 | 0.07 | 0 | 7.7 | 0.50 | 0.00 | 0.08 |
| core (top 2 levels × is_a, part_of, has_attribute) | 0.254 [0.216, 0.291] | 0.93 / 0.15 | 0.68 | 0.03 | 0 | 0.7 | 0.56 | 0.00 | 0.00 |
| noisy (30% + 20% wrong) | 0.579 [0.520, 0.638] | 0.84 / 0.44 | 0.76 | 0.29 | 0.7 | 0.7 | 0.58 | 0.14 | 0.20 |
| curated 30% | 0.547 [0.415, 0.679] | 0.96 / 0.38 | 0.75 | 0.18 | 0.7 | 0.0 | 0.56 | 0.14 | 0.24 |
| full curated (ceiling) | 1.000 | 1.00 / 1.00 | — | — | 7 | 0 | 0.90 | 0.79 | 0.86 |

**This is a negative result.** In this setting the self-supervised loop cannot build the logical ontology from nothing:

- From empty it accepts about 8 slots, but they are fragments (candidate-level relation Jaccard 0.07), and every reasoning probe is at chance.
- A core seed that covers only the top of the hierarchy does not propagate.
- A partial seed (30%, with or without 20% wrong edges) roughly doubles edge F1 (0.55–0.58), but complete relations and multi-hop reasoning stay far below the curated ceiling: multi-hop is-a 0.56–0.58 vs 0.90.

The learnable ontology behaves as a *refiner and extender of a substantial seed* (sections 3.1–3.3), not as a from-scratch ontology learner at this evidence level (4 training observations per concept, d = 64). Gold probe values are restricted to framed heads (validation/test/training), hence below 1.

### 3.7 E10.5 Dreaming (offline self-revision) after injected corruptions

Start from an oracle-consolidated ontology (hidden relations trained in their own slots and crystallized), then inject three corruptions: 10% wrong asserted edges, two crystallized relations merged into one slot, and a slot crystallized on 40 random distractor pairs with a random operator. 1,200 further training steps follow ("off" gets the dream passes' refit steps as plain training).

| Revision | passes | wrong edges repaired | correct edges damaged | merged relations split (ARI) | wrong slot repaired | correct slots disturbed | audit fit, training concepts |
|---|---:|---|---|---|---|---|---|
| off | 0 | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.777 [0.763, 0.790] |
| utility revisit only (every 300 steps) | 4 | 0.00 | 0.000 | 0.00 | 0.78 [0.58, 0.98] | 0.00 | 0.782 |
| dreaming every 1,200 steps | 1 | 0.46 [0.33, 0.60] | 0.002 | 0.52 [−0.73, 1.77] | 0.52 | 0.00 | 0.791 |
| dreaming every 600 steps | 2 | 0.51 [0.43, 0.60] | 0.003 | 0.77 [0.39, 1.14] | 0.88 [0.85, 0.92] | 0.00 | 0.796 [0.787, 0.805] |
| dreaming every 300 steps | 4 | 0.53 [0.45, 0.61] | 0.003 | 0.76 [0.51, 1.01] | 0.89 [0.71, 1.07] | 0.00 | 0.797 [0.790, 0.803] |

Right after injection the audit fit is 0.721. Training alone repairs nothing structural. The held-out-tested revisions:

- remove about half the wrong edges while damaging ≤ 0.3% of correct ones;
- split the merged relation back into its parts (ARI ≈ 0.77 with ≥ 2 passes);
- dissolve the wrongly crystallized relation (0.88–0.89; the utility-threshold revisit alone gets 0.78);
- never disturb a correct crystallized relation.

More frequent passes help up to 2 passes; 4 add little.

### 3.8 E10.7 Data variety vs self-confirmation bias

Same 8 non-audit observations per concept, drawn from V sources. Each source adds its own artifact association (a fixed (relation, attribute) pair) to every observation it produces, and the artifact is offered as a candidate edge. Audit observations come from fresh sources.

| Sources per concept | wrong self-acceptance (accepted ∧ not gold / accepted) | artifact acceptance | audit Δfit of accepted artifacts | in-sample acceptance |
|---|---|---|---|---|
| 1 | **0.439 [0.436, 0.441]** | **1.00** | −0.059 | 0.99 |
| 2 | 0.181 [0.160, 0.201] | 0.14 | −0.013 | 0.52 |
| 4 | 0.140 [0.118, 0.163] | 0.14 | +0.015 | 0.54 |
| 8 | 0.164 [0.086, 0.243] | 0.17 | +0.029 | 0.54 |

With a single source, the held-out observations share the training observations' artifact. The self-test confirms every artifact, and 44% of what it accepts is false; the audit on fresh sources shows the accepted artifacts *hurt* (−0.06). Any variety (≥ 2 sources) cuts wrong acceptance to 14–18% (the remaining errors are mostly same-filler/other-relation distractors). This is the confirmation-bias failure the author anticipated, and source variety is what fixes it.

### 3.9 E10.8 Continual additive learning (hidden relations arrive one per stage: similar_to → has_part → located_in)

Each stage releases the candidates of one hidden relation (with their distractors) and runs one slot round. Matching is scored on the offered candidate pairs; the framed-head Jaccard additionally credits rule-implied edges.

| Condition | relations matched by stage (J ≥ 0.5) | final Jaccard on offered pairs: similar_to / has_part / located_in | final Jaccard over all framed heads | retention drop (arrival − final) |
|---|---|---|---|---|
| additive, crystallize each stage | 1.00 → 1.67 → 1.67 | 0.88 / 0.52 / 0.13 | 0.64 / 0.53 / 0.06 | ≈ 0 for every relation |
| additive + dreaming between stages | 1.00 → 1.00 → 1.00 | 0.87 / 0.36 / 0.04 | 0.63 / 0.40 / 0.02 | ≈ 0 |
| plastic (new slot per stage, never crystallized) | 1.00 → 1.00 → 1.67 | 0.70 / 0.30 / 0.62 | 0.37 / 0.14 / 0.30 | similar_to 0.14 [−0.14, 0.41] |
| all at once (all evidence from the start) | 2.00 [−0.48, 4.48] | 0.58 / 0.58 / 0.39 | 0.39 / 0.59 / 0.19 | — |

- **Crystallization retains.** Earlier relations do not degrade (drop ≈ 0); the plastic slots drift, though not significantly. Rule closure makes the crystallized relations the most complete over all heads (similar_to 0.64 vs 0.37 plastic).
- **It does not grow more.** It does not end with more matched relations than plastic or all at once, and the last-arriving located_in is mostly missed (fragments).
- **Dreaming between stages hurt here** (has_part 0.52 → 0.36). Its revisions on freshly crystallized, still-incomplete relations trade alignment for fit, so E10.5's repair benefit does not transfer to this growth setting.

## 4. Read: does H-H hold at the synthetic level?

| Clause (refuted if …) | E10.0 result | Read |
|---|---|---|
| (i) erased edges not recovered above matched random candidates | AUC 0.98–0.99 vs 0.5 and precision ≈ 1.0 vs 0.33 prevalence at every erasure level (10–70%) | **supported**, strongly |
| (ii) blank slots do not align with hidden relations above chance | above the permutation null in every seed and both modes (Holm p = 0.015); the across-seed t-interval narrowly includes 0 in the absent mode (seed spread 0.21–0.51); collapsed 0.416 [0.361, 0.471] | **supported, weakly**: about half the oracle's alignment |
| (iii) self-acceptance not better than random acceptance at the same rate | edges +0.10 accuracy over random at matched rate (CIs exclude 0); relation hypotheses 0.80 vs 0.68 (p = 0.008, absent), 0.69 vs 0.59 (p = 0.17, collapsed) | **supported for edges and new relations**, not for sub-type splits |
| (iv) new-word frame inference no better than nearest-neighbour frames | +0.07 (k = 1) to +0.32 (k = 8) F1, CIs exclude 0 at every k | **supported** |

No refutation clause is met, so H-H survives E10.0. The effects split as follows.

**Strong and clean:**
- Recovering erased mapping connections (b).
- The case for *learnable + L2-to-prior* over fixed or free components (a). This is the author's L2 idea, and the prior pull matters most for zero-shot transfer.
- Held-out self-tests vs self-consistency: the in-sample test accepts 98–99% (d).
- Repair of injected errors by self-tested revision passes ("dreaming", E10.5).
- Source variety as the guard against self-confirmation (E10.7).
- Fast mapping of new words (e).

**Partial:**
- Blank-relation discovery (c). It works for relations with many edges and a positive-predicting logical property: similar_to is found and named *symmetric*; has_part is found and named *inverse of part_of*, then completed by that rule to Jaccard 0.62 over all heads. It reaches about half the oracle's alignment, fragments a weak relation (located_in) into several accepted duplicates, and essentially never names a transitive relation.
- The additive curriculum beats M3 and the stem-cell pool in the absent setting. It ties M3 on M3's native task (collapsed sub-types) while also yielding names and rules, and beats all-at-once significantly only in the collapsed setting.
- The overextension → refinement trajectory appears in the soft hypothesis: other relations' share rises at first, then falls. The over-inclusive *start* is an initialization choice; the narrowing is learned.

**Negative:**
- Building the logical ontology from an empty or core seed (E10.4).
- An advantage of sequential growth over plastic or all-at-once learning (E10.8; crystallization does retain).
- Dreaming between growth stages (E10.8).
- The held-out-tested pre-consolidation split ("refine"): on dev seeds it over-splits single relations (mean best Jaccard 0.15–0.34 vs 0.42–0.58 without), because held-out *observations* of the same concepts cannot tell one relation with noisy fillers from two relations.

**Caveat that bounds everything above.** The world was built for the mechanism: an HRR teacher, the learner's own family, vector priors at cosine 0.8, clean candidate pools. E10.0 is an upper bound on what E10.1 (frozen GPT-2 anchors, where E2/E3 fits were poor) can show. The E10.1 smoke slice (300 concepts, 200-step rounds, dev seed 7; a pipeline check, not evidence) shows weak signal:

- erasure-recovery AUC 0.58, R-precision 0.46 vs 0.34 prevalence;
- blank-slot ARI 0.21 (additive and M3) vs 0.64 for the oracle.

## 5. Deviations from the work-package text

1. **Fixed principal portion = 1 − erasure rate.** The two axes in the request are one axis. The fixed part is the asserted edges kept; the erased part is offered as candidates.
2. **Scale-free composition in E10.** Seed-relation operators are unit-normalised and relation log-scales mean-centred (`scale_free`); slot operators are unit-norm and have no scale. Without this, L1 on candidate masses is evaded by shrinking all asserted contributions (observed on dev seeds: relation scales fell to ≈ 0.4). The default (`scale_free=False`) keeps exact `FrameComposer` equivalence.
3. **Initial candidate mass of 0.1** when a blank slot opens. A random operator over zero-mass candidates gets no informative gradient: an over-inclusive start that L1 then prunes.
4. **Mass floor and zero rows.**
   - A mass floor `relu(1 − Σ_e m_e)²` applies only to concepts made solely of candidates (the empty seed).
   - Zero rows are left out of the fit; composition at `z = 0` gives ~1e13 gradients that freeze Adam.
5. **Refine split off by default.** The held-out-tested pre-consolidation split is implemented (`refine`) but disabled (dev seeds); refinement is left to competition (all-at-once) and to dreaming.
6. **Absorb.** Crystallized slots remain options for later candidates and absorb late members at round ends; dev seeds showed no effect.
7. **Dreaming accepts with lower bound > 0 against an equally long control refit.** Merges also need improvement; no merge was ever accepted.
8. **Slot gold verdicts are scored on the offered candidate universe** (precision ≥ 0.5 and purity ≥ 0.7). E10.4/E10.8 relation matching likewise; framed-head Jaccards are reported alongside. The first two E10.0 runs (v1, v2; uncommitted) used framed-head gold for E10.8, which no non-rule condition can reach; fixed in d51bf54.
9. **Clause (ii) has two readings.** The pre-written across-seed t-interval is kept; the per-seed permutation reading (Holm) was added after the first run and is labelled so.
10. **Baseline adaptations.**
    - M3 on absent relations: candidates are typed under one "unknown" relation that M3 may split.
    - Stem-cell pool: routes on the filler only, because candidates share one relation label.
11. **E10.1 learning runs on CPU.** The composer code is CPU-tested; only anchor extraction uses the GPU. E10.1 also uses 3,000 of the 6,000 E2 concepts, d = 128, 15 learnability configurations instead of the 3⁴ grid, and no E10.7 (no observation sources on WordNet).
12. **Synthetic "pretrained-authored" seed.** It is the noisy partial seed (stand-in). The E10.1 loader for an E7-authored ontology is a hook to add when E7 outputs exist (`seed_scenario` takes any asserted edge list).

## 6. Open decisions

| # | Decision | Default proposed |
|---|---|---|
| 1 | Duplicate relations: fragments of one relation are each accepted (they are real structure) | add a self-tested *merge-into-existing* step at crystallization (non-inferiority on held-out fit); measure on E10.1 |
| 2 | Split/refine criterion: held-out observations of the same concepts cannot reject over-splitting | test splits on held-out *concepts* (rule-implied pairs, validation concepts) or add an MDL charge per relation |
| 3 | L1 strength is domain-dependent (0.02 synthetic, 0.002 WordNet smoke) | scale L1 to a quantile of initial candidate utilities, so it transfers to E10.2 |
| 4 | E10.1 placement: CPU (~9–10 CPU-h, 2.5–3 h wall at 4 workers) vs adding device support for a GPU run (~0.5 GPU-h) | run on CPU after the GPU anchor job; device support only if E10.2 needs it |
| 5 | Negative-only structural hypotheses (functional, antisymmetric, one-to-one) rarely pass because their predictions are weak | keep in the hypothesis space but report them separately; E10.3 natural-language hypotheses may carry them |
| 6 | Dreaming between growth stages hurt in E10.8 | dream only over consolidated relations older than one stage, or with a stricter acceptance (lower bound > margin) |
| 7 | Transitive relations fragment before `transitive` can be generated in-sample | generate hypotheses over the union of a new slot and its most similar crystallized slot |

## 7. E10.1 (WordNet on frozen GPT-2 anchors): commands and cost

- **World.** The E2/E3 WordNet selection (`selection_seed` 5, single-token-aligned synsets). Eight prompt templates per concept give eight anchors: train 0–3, self-test 4–5, audit 6–7. These are centred and normalised.
- **Hidden relations.**
  - Absent: instance_hypernym, antonym, similar_to.
  - Collapsed: {part, member, substance}_meronym → meronym, and the same for the holonyms.
  - Erasure: over hypernym and the meronyms.
- **E10.4 core seed.** WordNet top levels (min_depth ≤ 4) with hypernym and part_meronym.
- **Configs.** `experiments/e10-self-semantics/e10-wordnet.yaml`, plus the smoke config `e10-wordnet-smoke.yaml`; the smoke run folder is `runs/e10.1-smoke`.

```bash
cd /home/bhux/workplace/VSA-LLM            # after merging this branch
PY=~/anaconda3/envs/vsa-repro/bin/python
# 1) GPU (queued, priority 35): multi-template anchors for the 6,000 E2 concepts (≈ 2 min, < 2 GB)
$PY -m vsa_embed.jobqueue add --name e10.1-wordnet-anchors --priority 35 --no-resume -- \
  $PY -m vsa_embed.experiments.e10_wordnet anchors --config experiments/e10-self-semantics/e10-wordnet.yaml \
  --output /home/bhux/data/vsa-llm/e10/wordnet-anchors-v1.pt
# 2) CPU (outside the GPU queue; 4 single-threaded workers, ≈ 2.5–3 h wall, ≈ 9–10 CPU-h)
nohup env CUDA_VISIBLE_DEVICES= $PY -m vsa_embed.experiments.e10_self_semantics \
  --config experiments/e10-self-semantics/e10-wordnet.yaml --output experiments/e10-self-semantics/runs/e10.1-v1 \
  > /home/bhux/data/vsa-llm/logs/e10.1-v1.log 2>&1 &
```

If everything should go through the queue, step 2 can be added with `--priority 36 --no-resume` (the same command after `--`). It then holds the queue for 2.5–3 h without using the GPU.

**Estimate.**
- GPU: ≈ 0.05 GPU-h, versus the plan's ≈ 5 GPU-h, which assumed GPU training. Adding device support to the composer would move step 2 onto the GPU at ≈ 0.5 GPU-h.
- CPU: from measured 0.10–0.12 s per full-batch step at 3,000 concepts (d = 128, one thread) and ≈ 90k steps per seed.

## 8. Reproduce E10.0

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES= $PY -m vsa_embed.experiments.e10_self_semantics \
  --config experiments/e10-self-semantics/e10-synthetic.yaml --output experiments/e10-self-semantics/runs/e10.0-v3
```

Recorded at commit d51bf54 (clean tree): 81 jobs, 4 workers × 1 thread, 648 s wall, 0.71 CPU-h. Every job seeds its own generators and runs single-threaded, so reruns replay exactly.


## E10.1 — WordNet on frozen GPT-2 anchors (run `experiments/e10-self-semantics/runs/e10.1-v1`, 3 seeds, 3,000 concepts, d = 128, CPU)

**Verdict: most of H-H does not transfer from the synthetic world to real WordNet relations on frozen anchors.** Refutation clauses: (i) erased-edge recovery above matched random candidates only at 50% erasure (AUC 0.53–0.61 vs 0.5); (ii) additive blank-slot alignment above chance only for collapsed relations (ARI 0.33; per-seed permutation also passes for absent relations, ARI 0.04); (iii) self-acceptance is **not** better than random at the matched rate (edges 0.46 vs 0.45; slots 0.35 vs 0.37); (iv) new-word frame inference is far **below** nearest-neighbour frames (F1 ≈ 0.01 vs 0.25), although the composed rows fit the targets better than nearest-neighbour rows (0.55 vs 0.44 at k = 4) and worse than the context mean (0.65).

| Readout | Synthetic (E10.0) | WordNet / frozen GPT-2 (E10.1) |
|---|---|---|
| L2-to-prior beats free and fixed (zero-shot fit) | yes | no — everything fixed has the best zero-shot fit (0.149 vs 0.094); learning fits training concepts (0.74 vs 0.25) but does not transfer |
| erased-edge recovery AUC (30% erased) | 0.987 | 0.589 |
| additive vs all-at-once (collapsed, ARI) | +0.23 | +0.31 [0.26, 0.36] |
| additive vs M3 splitting (collapsed, ARI) | tie | −0.29 (M3 better: 0.62) |
| riddle rules adopted / true | 13 of 14 true | 7 adopted of 37 slots, 4 true |
| self-acceptance vs random | +0.10 | ≈ 0 |
| new-word frame F1 at k = 4 vs nearest neighbour | 0.64 vs 0.36 | 0.007 vs 0.267 |
| dreaming repairs injected wrong edges | ≈ 50%, ≤ 0.3% damage | 7–12%, with 8–12% damage to correct edges |
| continual: additive + dreaming vs all-at-once (similar_to Jaccard) | — | 0.36 vs 0.15 (best E10.1 condition) |
| seed ontology: multi-hop is-a from a core / 30% seed | chance / 0.56–0.58 | 0.61 / 0.55 (full ontology 0.95) |

**Interpretation.** The frozen-anchor target is the regime the program already refuted for VSA row fitting (01a–01c: composed rows explain little of a frozen host's contextual variance; E2 test MRR 0.03–0.08). E10.1 shows the same limit for self-learning: the anchors carry too little relational signal for the learner to recover, discover or verify ontology structure, so self-tests are uninformative (≈ random) and inference collapses to nothing. The positives that survive are structural rather than signal-driven: one-relation-at-a-time (additive) discovery beats all-at-once, and dreaming between stages helps continual discovery of `similar_to`.

**Implication.** The synthetic success (E10.0) shows the mechanisms work when the training signal reflects the ontology; E10.1 shows frozen GPT-2 anchors do not provide that signal. The decisive test is **E10.2 (joint language-model training)**, where the signal is the LM loss on linked text; the S0 pilot already showed that a jointly trained channel improves held-out-concept similarity (CARD-660 0.10 → 0.33), i.e. that this regime carries ontology signal. E10.2 is scheduled after the E9/E4 recipe is fixed; its pre-registered readouts are the same refutation clauses on the LM's own strata and probes.


## E10 baselines for claim D (WP-PQ2, 2026-10-03)

**Why.** The 2026-10 novelty check (`manuscript/novelty-check-2026-10.md` §5.5) lists the baselines a reviewer will ask for before any claim D. This section runs four of them on the same worlds, scenarios, candidate pools and seeds as E10.0:
- D-B1, property identification: AMIE-style Horn-rule mining and IterE-style axioms read off relation operators;
- D-B2, erased-edge recovery: KG link prediction (TransE, RotatE, ComplEx, an IterE-style loop) and AMIE rule closure;
- D-B4, a null world: no hidden relation exists, so every accepted slot and every adopted rule is a false discovery;
- D-B5, validation reuse: a fresh validation split per test, and a multiplicity correction.

Not run here: clustering / TransG / IRM discovery baselines (D-B3), dreaming ablations (D-B6), resonator / Lasso frame inference (D-B7), L2-SP / retrofitting sweeps (D-B8) and the WN18RR benchmark (D-B9).

**Runs and code.**
- `experiments/e10-self-semantics/runs/e10.0-baselines-v2/`: E10.0 world, seeds 101/202/303. Dev seeds 7–9 are used only to choose the thresholds of the operator readings (RotatE 0.621, HRR roles 0.198) and of a tuned AMIE variant (PCA confidence 0.611). 51 jobs on 2 single-threaded workers: 20.7 min wall (the `-v1` run 19.7 min), at most 0.7 CPU-h, ≤ 1 GB per process.
- `experiments/e10-self-semantics/runs/e10.1-baselines-v2/`: E10.1 WordNet world, seeds 11/22/33, dev seed 7; graph-only parts (recovery and axioms) at 30% and 50% erasure. 11 jobs on 2 single-threaded workers: 12.7 min wall on a shared CPU (the `-v1` run 9.5 min), at most 0.43 CPU-h, ≤ 2.3 GB per process.
- Code: `vsa_embed.kg_baselines` (rule mining, KGE models, operator axioms), `vsa_embed.experiments.e10_baselines` (runner), and an opt-in report section (`python -m vsa_embed.experiments.e10_report --baselines RUN_DIR`). Configs: `e10-baselines.yaml` and `e10-baselines-wordnet.yaml`, merged over the E10.0 / E10.1 configs.
- **Provenance.** The numbers were first produced from an uncommitted working tree (folders `-v1`, manifests `git_dirty: true`, not committed). The committed `-v2` folders re-run the same commands from a clean tree at `d6edfc6` (the code was added in `0a38784`). Every job is seeded and single-threaded. Comparison of `-v2` with `-v1`, row by row without timing fields: E10.1, all 35 rows identical; E10.0, all 111 rows identical except the RotatE-axiom reading of discovered slots in 21 discovery rows (86 slot scores, one adoption), which `-v1` had computed before the final RotatE settings. Every number in this section is that of `-v2` (the `-v1` report showed 3 / 2 / 2 instead of 3 / 3 / 3 in the collapsed RotatE cell of §B.2).

**Methods** (same pools, seeds and observations as E10.0; gold is read only by the evaluation):
- AMIE-style mining (Galárraga et al., WWW 2013) covers closed rules with one or two body atoms, either of which may be inverted. Kinds: symmetric `r⁻¹ ⇒ r`, inverse `p⁻¹ ⇒ q`, sub-property `p ⇒ q`, and chains `a ∧ b ⇒ s` (transitivity included). Each rule gets support, head coverage, standard confidence and PCA confidence. AMIE's defaults are kept: head coverage ≥ 0.01 and PCA confidence ≥ 0.1, plus support ≥ 2.
- KG embeddings, full-batch with uniformly corrupted heads or tails:
  - TransE (Bordes et al., NIPS 2013);
  - RotatE (Sun et al., ICLR 2019, arXiv:1902.10197), with self-adversarial negatives;
  - ComplEx (Trouillon et al., ICML 2016, arXiv:1606.06357).

  All use dimension 64, 400 epochs and 32 negatives. A dev-seed sweep of dimension, epochs, learning rate and margin moved validation AUC only within 0.60–0.72.
- IterE-style (Zhang et al., WWW 2019, arXiv:1903.08948): read an axiom off the relation operators. For RotatE, an axiom holds when the phases compose: the mean cosine of the summed signed body phases minus the head phase. RotatE's rotation is unitary HRR binding in the Fourier domain. For the learnable ontology's own HRR role vectors the score is `cos(a′ ⊛ b′, s)`, with the involution `r*` for an inverted atom; symmetry is then `cos(r*, r)`. The IterE loop trains RotatE, adds the closure of the accepted axioms as training triples, and retrains it once.
- `prior_corr` is a no-learning control for recovery, added because KG completion turned out to be the wrong competitor. It scores a candidate `(h, r, a)` by the cosine of `h`'s mean training observation with the candidate bound from the *priors*, `r_prior ⊛ a_prior`; the learnable ontology starts from the same priors.
- Recovery thresholds (for F1) are chosen on 10% of the asserted edges, held out with distractors drawn by the scenario's own rule. The method is fit on the other 90%.
- "Gold axioms" are the rules with standard confidence ≥ 0.9 (support ≥ 3) on the complete gold graph:
  - `similar_to⁻¹ ⇒ similar_to`;
  - `part_of⁻¹ ⇒ has_part` and `has_part⁻¹ ⇒ part_of`;
  - `located_in ∘ located_in ⇒ located_in`;
  - the two sibling chains `similar_to(⁻¹) ∘ is_a ⇒ is_a`.
- Fresh validation data is **simulated**. New views of each concept are drawn exactly like the world's observations from its clean target, from a separate generator. They stand for a new split of the same size, which in practice would need more data.

### B.1 Erased-edge recovery (D-B2): graph-only completion is far below; a no-learning observation readout gets most of the way

AUC on the E10.0 (b) candidate pools; the learnable ontology's numbers are its committed E10.0 run, on the same pools (sizes checked). Means over 3 seeds [95% t-interval].

| Erased | learnable ontology | `prior_corr` (no learning) | AMIE closure | TransE | RotatE | ComplEx | IterE-style |
|---|---|---|---|---|---|---|---|
| 10% | 0.989 [0.988, 0.991] | 0.909 [0.903, 0.915] | 0.769 [0.761, 0.778] | 0.761 [0.738, 0.784] | 0.718 [0.628, 0.807] | 0.710 [0.688, 0.732] | 0.722 [0.680, 0.763] |
| 30% | 0.987 [0.981, 0.992] | 0.916 [0.902, 0.930] | 0.694 [0.673, 0.715] | 0.688 [0.646, 0.730] | 0.672 [0.618, 0.727] | 0.663 [0.623, 0.703] | 0.661 [0.641, 0.682] |
| 50% | 0.986 [0.985, 0.988] | 0.917 [0.901, 0.933] | 0.616 [0.581, 0.651] | 0.623 [0.577, 0.669] | 0.593 [0.571, 0.615] | 0.597 [0.538, 0.656] | 0.585 [0.512, 0.659] |
| 70% | 0.984 [0.973, 0.996] | 0.913 [0.896, 0.931] | 0.557 [0.537, 0.578] | 0.560 [0.507, 0.614] | 0.538 [0.493, 0.583] | 0.548 [0.496, 0.601] | 0.545 [0.477, 0.613] |

- Every graph-only method is below the learnable ontology at every erasure level. ΔAUC runs from −0.22 to −0.45, and every CI excludes 0. Graph-only AUC also falls toward chance as more of the graph is erased. AMIE closure and TransE are the best graph-only methods (0.77 at 10% erased). The IterE loop adds nothing over RotatE, because RotatE cannot represent transitivity or the sibling chains (§B.2).
- The no-learning `prior_corr` reaches 0.91–0.92 AUC at every erasure level, against 0.98–0.99 for the learnable ontology (ΔAUC −0.07 to −0.08, CI excludes 0). At 50–70% erasure its F1 at a validation-chosen threshold (0.76–0.77) equals or beats the learnable ontology's committed F1 (0.74 / 0.55). The learnable ontology's F1 uses its fixed mass ≥ 0.5 rule, which loses recall as erasure grows.
- **Reading.** "Erased-edge recovery, AUC ≈ 0.99" is not something any KG-completion method gets in this world; graph regularities give 0.55–0.77. But most of it does not need the learning loop either. In this world every observation is composed by an HRR teacher from all of a concept's gold edges, erased ones included. So a single correlation of the observations with prior-bound candidates already ranks erased edges at 0.91. The learnable ontology's own contribution is the step from 0.91 to 0.99: it learns better atomics and relations under the L2 pull, and better ranks than one correlation. Erasure recovery should therefore be described as "decoding the edges present in the training signal", not as inference from ontology structure. In a world whose observations do not contain the erased edges, the graph-only rows are the relevant comparison.

### B.2 Property identification from pairs vs from operators (D-B1)

Horn axioms on the 30%-erasure asserted graph against the gold axioms; 3 seeds.

| Reading | precision | recall | F1 | AUC of the score | axioms accepted |
|---|---|---|---|---|---|
| AMIE, defaults (PCA ≥ 0.1, HC ≥ 0.01) | 0.45 [0.24, 0.66] | **1.00** [1.00, 1.00] | 0.62 [0.41, 0.83] | **0.999** | 13.7 |
| AMIE, default filters and PCA ≥ 0.611 (dev-chosen) | 0.84 [0.81, 0.88] | 0.89 [0.65, 1.13] | **0.86** [0.74, 0.99] | 0.999 | 6.3 |
| IterE-style, RotatE phases ≥ 0.621 (dev-chosen) | **1.00** [1.00, 1.00] | 0.33 [−0.08, 0.75] | 0.48 [0.01, 0.96] | 0.86 [0.67, 1.06] | 2.0 |
| the learnable ontology's own HRR role vectors ≥ 0.198 (dev-chosen) | 0.01 | 0.17 | 0.02 | **0.45** (chance) | 80.0 |

- **AMIE recovers every planted Horn property from the observed pairs alone**: symmetry, both inverse directions, transitivity and the sibling chains. Its confidence ranks gold axioms almost perfectly (AUC 0.999). One application of the dev-tuned rules recovers 27% of the erased edges at 0.94 gold precision.
- RotatE's operators carry the symmetric and inverse axioms when they hold (precision 1.0). They cannot express transitivity or chains, so recall is 0.33.
- **The learnable ontology's HRR role vectors carry no axiom information** (AUC 0.45, at chance). This is expected: in a composer, a role binds fillers into the head's row; it does not map heads to tails. The logical properties in E10.0 come from testing hypotheses on *pairs* (the riddle step), not from the operators.

The same accepted slots of the additive curriculum (E10.0 (c), re-run here with identical settings), explained four ways. Entries are adopted / true of the slot's best-matching relation / that relation's designed property, pooled over 3 seeds:

| Scenario | accepted slots | riddle (E10.6) | AMIE on the captured pairs | RotatE axiom | slot's own HRR operator |
|---|---:|---|---|---|---|
| absent (has_part, similar_to, located_in hidden) | 10 | 5 / 5 / 5 | 7 / 5 / 5 | 0 / 0 / 0 | 9 / 0 / 0 |
| collapsed (part_of + member_of merged) | 6 | 3 / 3 / 3 | 6 / 3 / 3 | 3 / 3 / 3 | 6 / 0 / 0 |

- AMIE on the captured pairs finds exactly the riddle step's correct identifications: `symmetric` for similar_to slots, `inverse_of:part_of` for has_part, `inverse_of:has_part` for part_of. It also adopts more wrong ones:
  - 2 of 7 in the absent world: `inverse_of:part_of` on two mixed located_in slots;
  - 3 of 6 in the collapsed world: `sub_relation_of:meronym`, which is true by construction but uninformative.

  With the re-used split the riddle step adopts nothing wrong on accepted slots of the real worlds.
- Rule completion is the same: has_part 0.62 vs 0.60, similar_to 0.49 vs 0.49, part_of 0.97 vs 0.97 (Jaccard over all framed heads). The riddle step is ahead only on located_in fragments (0.24 vs 0.15).
- The riddle's extra predictions are more precise (0.82 vs 0.60 gold precision in the absent world).
- **Reading.** At this level the held-out hypothesis test is **not better than standard rule mining at identifying properties**. It is more conservative: it has equal true identifications and fewer false adoptions. Neither method names located_in fragments transitive.

### B.3 Null worlds (D-B4): the slot self-test does not control false discoveries

The additive curriculum, run exactly as E10.0 (c), on worlds with no hidden relation:
- `null_distractors`: every relation asserted, and only distractor pairs offered, as many as in the absent world;
- `null_permuted`: the absent world with the offered pairs' tails permuted, and no gold pair left.

| World | proposed slots / run | accepted slots / run | false accepted slots / run | runs with ≥ 1 false acceptance | rules adopted on accepted slots: riddle / AMIE | AMIE rules involving the offered pairs |
|---|---|---|---|---:|---|---|
| absent (real) | 4.33 | 3.33 [1.90, 4.77] | 0.67 | 1 of 3 | 5 / 7 | 3.0 (all reflect hidden relations) |
| null_distractors | 3.67 | **2.67** [1.23, 4.10] | **2.67** | **3 of 3** | 1 / 0 | 0 |
| null_permuted | 5.00 | **4.33** [2.90, 5.77] | **4.33** | **3 of 3** | 2 / 0 | 0 |

- **In both null worlds the curriculum accepts 2.7–4.3 slots per run, all false.** Every run has at least one false discovery, and the acceptance rate (73–87% of proposed slots) is as high as in the real world (77%).
- The slot self-test (utility lower bound > 0 *and* beating random tails under the same operator) therefore does not distinguish a real hidden relation from a bundle of random pairs.
- The likely mechanism is co-adaptation. Both tests ablate or add edges in a model *trained with* the slot's members, so the rest of the model has adapted to them, while the random-tail control was never trained in.
- The riddle step adopts a structural rule on 3 of 21 null slots (`one_to_one`, `antisymmetric`: weak, negative-only properties). AMIE adopts none, and mining the offered pairs as a pseudo-relation yields **no** rule in either null world (3 true rules in the real world).
- **Reading.** Clause (iii) of H-H ("self-acceptance better than random acceptance") was supported in E10.0 only *relative to* random acceptance at the matched rate in the real world. The null world shows that the acceptance itself has no false-discovery control. E10.0's "about half the oracle's alignment" for discovered relations must be read together with ≈ 3–4 accepted false slots per run in a world with nothing to discover. A merge test, a refit-based acceptance (compare against an equally long refit *without* the slot, as dreaming does), or an MDL charge per slot is needed before any discovery claim (R10 §6, open decisions 1–2).

### B.4 Validation reuse (D-B5): reuse is not what inflates acceptance

Edge self-test of E10.0 (d): accept a proposed edge if removing it lowers held-out fit (lower bound > 0). Variants:
- `reused`: one validation split for every proposal, as in E10.0;
- `fresh`: a fresh simulated split per proposal;
- `ttest`: a one-sided t-test per proposal;
- `holm`: Holm over all proposals of a run.

| Scenario | proposals | variant | acceptance | wrong among accepted | false acceptance of non-gold proposals | recall of gold |
|---|---:|---|---|---|---|---|
| erasure 30% | 840 | reused | 0.915 | 0.053 | 0.446 | 0.973 |
| | | fresh | 0.933 | 0.064 | 0.544 | 0.981 |
| | | ttest | 0.824 | 0.030 | 0.228 | 0.897 |
| | | holm | 0.043 | 0.000 | 0.000 | 0.049 |
| erasure 50% | 1,368 | reused | 0.918 | 0.056 | 0.476 | 0.974 |
| | | fresh | 0.921 | 0.056 | 0.459 | 0.977 |
| null (nothing erased) | 92 | reused | 0.450 | 1.000 | 0.450 | — |
| | | fresh | 0.400 | 1.000 | 0.400 | — |
| | | ttest | 0.281 | 1.000 | 0.281 | — |
| | | holm | 0.000 | — | 0.000 | — |

The `reused` row reproduces E10.0's committed (d) numbers exactly (840 proposals, acceptance 0.92, accuracy 0.927).

- A fresh split changes nothing material: acceptance 0.92–0.93 vs 0.92 in the erasure worlds, and 0.40 vs 0.45 in the null world. The slot curriculum with a fresh validation split per round accepts the same slots as with the re-used split in the absent world (3.33 / run in both), and 2.67 → 3.00 and 4.33 → 4.33 in the null worlds.
- So adaptive reuse of one split is not the problem. The edge test accepts **40–45% of the distractor edges that training proposes** even on fresh data, the same co-adaptation effect as for slots.
- Most proposals in the real worlds are real edges, so precision stays high (5–6% wrong among accepted). But "false acceptance among non-gold proposals" is 45–54%.
- Holm over all proposals (with 4 observations per test) is far too conservative: it accepts 3–4% of proposals. A plain t-test roughly halves the false acceptances (0.23 vs 0.45 at 30% erasure; 0.28 vs 0.45 in the null world) at a recall cost of 0.08.

### B.5 E10.1 (WordNet, frozen GPT-2 anchors): graph-only baselines

Graph-only parts only: recovery at 30% / 50% erasure on the E10.1 (b) pools and the axiom readings. Seeds 11/22/33; the learnable ontology's numbers are from the committed `runs/e10.1-v1`, on the same pools (sizes checked). The composer-based parts (edge self-tests, slot discovery, null worlds) were not repeated: E10.1 already found the self-tests at chance on WordNet. `prior_corr` needs observations in the composer's space, so it applies to the synthetic world only (the WordNet anchors are 768-d GPT-2 states).

| Erased | learnable ontology (E10.1) | AMIE closure | TransE | RotatE | ComplEx | IterE-style |
|---|---|---|---|---|---|---|
| 30% | 0.589 [0.527, 0.651] | 0.508 [0.504, 0.512] | 0.588 [0.554, 0.622] | 0.601 [0.569, 0.633] | 0.628 [0.618, 0.638] | 0.601 [0.558, 0.644] |
| 50% | 0.561 [0.545, 0.577] | 0.507 [0.504, 0.510] | 0.570 [0.536, 0.603] | 0.582 [0.569, 0.596] | 0.605 [0.565, 0.644] | 0.577 [0.542, 0.612] |

- On WordNet, every KG-embedding baseline matches or numerically exceeds the learnable ontology. ComplEx gives ΔAUC +0.04 [−0.03, +0.11] at 30% and +0.04 [−0.01, +0.10] at 50%; the TransE, RotatE and IterE-style CIs also include 0.
- AMIE's rule closure is at chance (0.51): erased hypernym and meronym edges are rarely implied by short Horn rules within 3,000 concepts.
- All three models' F1 (KGE 0.41–0.51, AMIE 0.50) is above the learnable ontology's committed F1 (0.33 / 0.22). The accept-all F1 is 0.5 at this prevalence, so the learnable ontology's fixed mass threshold is below the trivial F1 there.
- Gold axioms on the WordNet slice are category-inheritance chains (`hypernym ∘ lexname ⇒ lexname`, `… ∘ pos ⇒ pos`) and antonym chains. AMIE ranks them at AUC 0.999. With default filters it recovers only 27% of them: the head-coverage filter excludes rules into the large lexname / pos relations.
- The RotatE threshold chosen on the single dev seed is degenerate (−0.18): it accepts ≈ 15.7k axioms at precision 0.001. The IterE loop therefore injects only the 50 best supported axioms, and its recovery equals plain RotatE.
- **Reading.** This confirms the E10.1 verdict from the baseline side. On frozen-anchor WordNet the learnable ontology's recovery is no better than graph-only KG completion, so it carries no signal beyond what the graph provides.

### B.6 What the baselines mean for claim D

| Sub-claim of D | Baseline outcome | Wording consequence |
|---|---|---|
| Recovers erased edges (AUC ≈ 0.99) | KG completion: 0.55–0.77; a no-learning prior × observation correlation: 0.91–0.92 | Recovery is **decoding edges that are present in the training signal**; learning adds +0.07 AUC over one correlation. Not "inference"; not specific to a learnable ontology against KGE in the sense that KGE has no access to that signal |
| Identifies symmetric / inverse relations without labels | AMIE on the same captured pairs finds the same correct properties (and more false ones); AMIE on the observed graph identifies every planted Horn axiom (AUC 0.999) | "Property-hypothesis testing on held-out pairs **matches standard rule mining** in identification, with fewer false adoptions" — not a new capability |
| Self-tested acceptance | Null worlds: 2.7–4.3 false accepted slots per run, every run; fresh splits do not change acceptance; the edge test accepts 40–45% of proposed distractors | **No false-discovery control.** Clause (iii) holds only relative to random acceptance. A null-calibrated acceptance (refit-based or MDL) is a prerequisite for any discovery claim |
| The learnable operators encode logic | HRR role readings at chance | Do not claim that the learned operators are logical; the logic is in the hypothesis tests over pairs |

**Overall.** Recommended wording for D (novelty check §5.6) should be narrowed further. Say:
- "recovers erased edges present in its training signal";
- "identifies symmetric and inverse relations from captured pairs as well as rule mining";
- "its acceptance test is not calibrated against a null world".

Claim D stays out of the abstract. The next E10 step, before E10.2, is a null-calibrated acceptance test, with this null world as its pre-registered check.

**Deviations.**
1. `prior_corr` was added after the first run showed that every graph-only method was far below. It uses the learner's own priors and training observations only, with no gold.
2. The E10.0 baseline run was executed twice: before and after adding `prior_corr`. All shared rows are identical.
3. Operator-reading thresholds are F1-optimal on the dev seeds' gold axioms. This is an evaluation-side calibration; IterE itself uses a fixed hyper-parameter.
4. AMIE support ≥ 2 replaces AMIE 3's absolute 100 (a small graph).
5. Fresh validation splits are simulated from the clean targets (synthetic only).
6. E10.1 runs graph-only parts with a smaller KGE budget (150 epochs, 16 negatives; ≈ 10.7k entities). Its first attempt (200 epochs, 32 negatives) was stopped at ~12 min, at ≈ 4–5 GB per worker, above this work package's memory limit; that run left no folder.

**Reproduce.**

```bash
PY=~/anaconda3/envs/vsa-repro/bin/python
PYTHONPATH=src CUDA_VISIBLE_DEVICES= $PY -m vsa_embed.experiments.e10_baselines \
  --config experiments/e10-self-semantics/e10-baselines.yaml --output experiments/e10-self-semantics/runs/e10.0-baselines-v2 --workers 2
PYTHONPATH=src CUDA_VISIBLE_DEVICES= $PY -m vsa_embed.experiments.e10_baselines \
  --config experiments/e10-self-semantics/e10-baselines-wordnet.yaml --output experiments/e10-self-semantics/runs/e10.1-baselines-v2 --workers 2
PYTHONPATH=src $PY -m vsa_embed.experiments.e10_report --baselines experiments/e10-self-semantics/runs/e10.0-baselines-v2
```
