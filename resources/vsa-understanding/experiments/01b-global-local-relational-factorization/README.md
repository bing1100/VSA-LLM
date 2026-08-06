# Experiment 01b — Global–Local Relational Factorization

**Status:** immediate next experiment. **Depends on:** experiment 00 algebra tests and experiment 01a's leakage-resistant negative pilot. **Blocks:** experiment 02 insertion.

## Motivation from experiment 01a

The first real GPT-2/WordNet pilot found that true ontology structure was useful, but a simple untyped additive model preserved held-out neighborhoods better than typed MAP or HRR. The pilot encoded one nested hypernym path, gave every edge equal weight, and asked one fixed binding algebra to explain most of a token row. That result rejects the narrow factorizer, not the broader idea of compositional relational structure.

The immediate question is now:

> Can frozen-host concept geometry be explained efficiently by globally shared relation semantics, sparsely weighted local edges, and a restricted concept-specific residual—and what relation parameterization gives the best bias–capacity–data tradeoff?

The scientific object is **shared relational factorization**. HRR is one candidate inductive bias, not the premise.

## Global–local model

For target concept \(i\), construct a relational core

\[
z_i = \sum_{(j,r)\in N(i)} \alpha_{ijr}\,T_r(c_j),
\]

then map it to the host representation:

\[
\hat e_i = \mu + Uq_i + Pz_i + D(text_i) + B\rho_i.
\]

- \(c_j\): reusable concept atomic.
- \(T_r\): globally shared representation of relation \(r\).
- \(\alpha_{ijr}\): sparse local edge strength or representational salience.
- \(\mu+Uq_i\): host mean/low-rank nuisance geometry.
- \(P\): global host-alignment map.
- \(D(text_i)\): optional definition-derived zero-shot residual.
- \(B\rho_i\): restricted learned concept-specific correction.

Logical truth and representational salience are not the same variable. A true edge may make little contribution to host geometry; a salient textual association may matter strongly without being taxonomically defining. Preserve source truth/confidence separately from learned \(\alpha\).

## Core hypotheses

1. **Global sharing:** one relation representation reused over many pairs transfers better than edge-specific transforms.
2. **Weighted participation:** sparse learned edge weights outperform treating every ontology edge as binary and equally important.
3. **Structured shortcut:** relational initialization reduces the local parameters, examples, or gradient steps needed for mastery.
4. **Bias–capacity frontier:** fixed vector binding is sample-efficient when aligned with the data-generating relation; learned operators improve asymptotic fit under mismatch but require more data and regularization.
5. **Rigidity diagnostic:** a persistent held-out error floor that more optimization/data cannot close, but a controlled higher-capacity family can close, is evidence of insufficient representational capacity rather than failed optimization.

## Relation representation ladder

Compare a spectrum, not a binary HRR-versus-matrix choice.

| Level | Representation | Global shared object | Purpose |
|---|---|---|---|
| R0 | \(T_r(c)=c\) | none | weighted additive control |
| R1 | \(T_r(c)=bind(q_r,c)\) | relation vector \(q_r\), fixed operator | HRR, unitary HRR, MAP, fixed orthogonal binding |
| R2 | \(T_r(c)=F(c,q_r)\) | compact relation vector plus one shared learned interpreter | tests whether vectors suffice when interpretation is adaptable |
| R3 | \(T_r(c)=R_rc\) | constrained operator | diagonal, circulant, orthogonal, block, low-rank ranks 4–32 |
| R4 | \(T_{r,d}=R_r^{global}+\Delta R_{r,d}\) | global core plus penalized domain modulation | controlled contextual variation |
| R5 | \(T_{ijr}\) | edge-specific transform | memorization upper bound; never a promotion candidate |

R1 supplies the strongest compositional prior and least relation-specific capacity. R3 shifts capacity into the relation itself. R2 tests an important middle ground: ordinary relation vectors retain compact compositional identities while a shared interpreter learns how relation vectors act. R4 is allowed only after global models are understood.

## What “operator versus vector” means

A relation vector with HRR or MAP already induces an operator: binding by a fixed vector is a circulant or diagonal linear transform. A learned matrix does not introduce an entirely different idea; it relaxes the structural constraint on that transform. The relevant questions are therefore:

1. Is the fixed operator family aligned with real relational geometry?
2. How much data and optimization does mismatch cost?
3. Does the family have an irreducible approximation ceiling?
4. At what capacity does flexibility stop improving transfer and start enabling memorization?

Do not infer these answers from one endpoint. Measure learning curves and asymptotes.

## Stage A — controlled teacher × learner identifiability

Generate typed sparse graphs from teachers using:

- fixed HRR/MAP relation vectors;
- diagonal, orthogonal, and low-rank operators;
- global core plus small domain modulation;
- weighted edges, restricted residuals, nuisance low-rank host geometry, and noise.

Fit every learner family to every teacher family. Sweep edges per relation and report:

- held-out node/edge reconstruction and relation-query accuracy;
- recovery of relation operators up to valid symmetry/alignment;
- edge-weight rank correlation and support recovery;
- residual allocation error;
- data, steps, and FLOPs to reach fixed quality;
- train–test gap and asymptotic error floor.

This matrix separates aligned inductive bias, recoverable mismatch, insufficient capacity, and excessive capacity.

## Stage B — fixed heterogeneous WordNet graph

Use genuine relation types rather than ancestor-depth pseudo-roles:

- hypernym/hyponym and instance hypernym;
- member/substance/part meronymy and inverses;
- antonymy;
- attributes;
- derivationally related forms where token/sense alignment is defensible.

Begin with fixed candidate edges and one deterministic sense-specific concept anchor per node. Learn relation representations, scalar edge salience, host alignment, and residuals. Compare raw token rows, mean/PC-removed rows, and contextual sense anchors as separate targets.

## Stage C — residual and adaptation frontier

Sweep concept-specific residual dimensions `0, 4, 8, 16, 32, 64, full` and held-out adaptation evidence `0, 1, 2, 4, 8, 16` examples. Compare graph-only, definition-only, graph+definition, and equal-budget unstructured models.

The main outcome is a rate–distortion/sample-efficiency curve:

> At fixed behavior, how many concept-specific parameters, examples, steps, and FLOPs does relational initialization save?

Pure zero-shot composition is one endpoint. Few-shot residual mastery is an equally important success mode.

## Stage D — local graph refinement

Only after fixed-edge results are interpretable, permit sparse reweighting, pruning, relation relabeling, and a bounded number of evidence-backed candidate additions. Keep the source graph authoritative and version every learned overlay. Do not begin with unrestricted all-pairs graph discovery.

Use alternating optimization:

1. initialize candidate graph and atomics;
2. fit globally shared relation transforms and host alignment;
3. fit sparse local salience weights;
4. fit restricted residuals;
5. revise candidate edges from systematic residuals;
6. evaluate held-out transfer and repeat only under a fixed budget.

## Stage E — frozen-host behavioral confirmation

Promising candidates must be evaluated through contexts, not promoted from row geometry:

- definition matching and relation QA;
- cloze rank/MRR and alias-aware generation;
- paraphrase and counterfactual consistency;
- hidden-state/logit distillation;
- unrelated-text KL/perplexity and old-token retention.

## Splits and leakage controls

Report separately:

- edge-group disjoint;
- node and alias disjoint;
- connected-component disjoint;
- relation-instance disjoint;
- relation-family disjoint;
- ontology/domain-family disjoint;
- composition disjoint;
- atomic disjoint, with definition induction allowed only in its declared track.

Exact relational twins and inverse-edge duplicates remain in one partition. A global relation can appear in training for node/edge transfer, but relation-disjoint evaluation must hold out the entire relation family and any explicit inverse or paraphrase.

## Bias–capacity–data matrix

Cross:

- relation family/capacity;
- logarithmic training edges per relation;
- residual dimension;
- uniform versus scalar sparse edge weighting;
- clean, missing, spurious, and mislabeled graphs;
- raw, residualized, contextual, and behavioral host targets;
- matched parameters, matched optimization steps, and matched FLOPs.

Report both equal-budget comparisons and the full capacity frontier. “Operator beats vector” is not meaningful if it only means “more parameters beat fewer parameters.”

## Baselines

- unweighted and weighted additive aggregation;
- nearest relational recipe;
- graph embeddings plus matched projection: TransE, ComplEx, RotatE, HolE, R-GCN;
- equal-budget low-rank/unconstrained factorization;
- definition encoder and graph+definition fusion;
- shuffled relation labels, shuffled endpoints, and random fixed binding;
- edge-specific R5 training upper bound;
- mean/low-rank host predictor.

## Primary endpoints

1. **Synthetic:** held-out relational reconstruction/query quality integrated over the data-budget curve.
2. **Real host:** reduction in residual parameters or adaptation examples required to reach a predeclared contextual/behavioral target relative to weighted additive and equal-budget graph projection.

Secondary endpoints include kNN overlap after nuisance removal, pairwise geometry, edge AUC, residual energy, convergence cost, relation/edge recovery, and calibration/locality.

## Acceptance gate

A candidate advances to experiment 02 only if it:

1. beats weighted additive and the strongest equal-budget graph/unstructured baseline on held-out contextual or behavioral transfer;
2. shifts the adaptation curve left—fewer residual parameters, examples, steps, or FLOPs at fixed quality;
3. uses globally shared relations that outperform edge-specific memorization on held-out transfer;
4. passes recipe/edge/alias leakage audits and reproduces over at least three splits;
5. transfers to a second relation or ontology family;
6. preserves frozen-host locality;
7. emits auditable relation parameters, edge salience, residual allocation, uncertainty, and provenance.

Geometry alone can nominate a candidate but cannot pass the behavioral gate.

## Failure interpretation

- **R0 remains best:** graph region matters, but typed transformations add no transferable information.
- **R1 learns fast then plateaus below R3:** useful fixed bias with insufficient capacity; consider a constrained operator or structured residual.
- **R3/R4 wins only in-domain:** flexibility is fitting ontology/domain idiosyncrasies rather than global relations.
- **R5 wins training but loses transfer:** expected memorization; confirms the value of global sharing.
- **All relational models lose to definition encoder:** graph is incomplete or host semantics are primarily contextual/lexical.
- **Good row geometry, no behavior:** target is wrong; move the objective to hidden states/logits or a sidecar interface.
- **Residual remains large but adaptation is faster:** still a useful structured initialization; report honestly as few-shot rather than zero-shot learning.

## Deliverables and toolkit increment

- `RelationTransform` implementations for fixed binding, shared interpreter, diagonal, orthogonal, and low-rank operators;
- `EdgeWeightModel` with uniform and sparse scalar modes;
- restricted `StructuredResidual` bases;
- teacher × learner benchmark and bias–capacity reports;
- weighted `CompositionGraph` overlay with truth, salience, confidence, and provenance kept distinct;
- `ResidualBudgetCurve` and convergence-to-threshold evaluator;
- a candidate global–local factorizer or a documented negative frontier.

## Implemented Stage-A runner

The synthetic teacher × learner gate is runnable through the local package:

```bash
python -m vsa_embed.experiments.global_local_relations \
  --config experiments/01b-global-local-relational-factorization/smoke.yaml \
  --output experiments/01b-global-local-relational-factorization/runs/smoke
```

Use `reduced.yaml` for the three-seed HRR/diagonal/low-rank matrix. The runner materializes one maximum-budget world per `(teacher, residual, seed)`, takes relation-balanced nested prefixes for lower budgets, and preserves one fixed held-out set. Learned salience and residuals are globally shared feature functions; there are no edge- or concept-specific lookup parameters.

## First reduced Stage-A result (2026-07-17)

The corrected 216-condition matrix (`3 seeds × 3 teachers × 4 learners × 3 data budgets × 2 residual strata`) passed the synthetic gate. At 512 edges/relation, each teacher was best recovered by its aligned learner:

| Teacher | Additive MSE | HRR MSE | Diagonal MSE | Low-rank MSE | Winner |
|---|---:|---:|---:|---:|---|
| HRR | 0.01741 | **0.00014** | 0.01085 | 0.00925 | HRR |
| Diagonal | 0.00224 | 0.00221 | **0.00016** | 0.00111 | diagonal |
| Low-rank | 0.04140 | 0.04022 | 0.04029 | **0.00022** | low-rank |

The means pool residual dimensions 0 and 8. Aligned HRR and diagonal models were already near the noise floor with 32 edges/relation. The rank-8 low-rank learner exhibited the expected higher-data regime: mean MSE fell from 0.00813 to 0.00013 without a residual, and from 0.01214 to 0.00026 with an 8-dimensional transferable residual, between 32 and 128 edges/relation. At their winning conditions, relation-query accuracy was 0.96–1.00 and edge-weight correlation was approximately 0.998–1.000.

This supports three narrow conclusions: the implementations are identifiable; fixed structured transforms can be strongly sample-efficient when matched; and additional operator capacity has a measurable data cost. It does **not** show that real ontology relations follow any teacher family, that embedding-row reconstruction changes behavior, or that VSA structure beats equal-budget graph/definition models. Those remain Stage B/C questions.

During review, an initially identity-initialized diagonal teacher was found to be degenerate and was replaced with a non-identity relation-specific teacher. A regression test now prevents non-additive synthetic teachers from collapsing to the additive control. The valid artifacts are under `experiments/01b-global-local-relational-factorization/runs/reduced/`.

## Stage B implementation and first real-host result (2026-07-17)

Stage B replaces ancestor-depth recipes with a fixed heterogeneous WordNet graph containing seven genuine relation types: hypernym, instance hypernym, member/substance/part meronym, antonym, and attribute. A deterministic alphabetic single-token lemma anchors each synset. Polysemy is handled by a sense-conditioned target rather than pretending one static token row represents one synset:

```text
Definition: <WordNet gloss> Term: <aligned lemma>
```

The final-token hidden state of frozen GPT-2 revision `607a30d...` is the contextual sense anchor. Static input rows are retained as a control. Raw and train-endpoint-centered targets are evaluated separately. The experiment predicts a target anchor from a source anchor and globally shared relation transform, using both edge-disjoint and strict node-disjoint splits. Crossing edges are discarded in the node-disjoint track. Symmetric antonym reversals share one split group.

The reduced matrix used 560 edges—80 per relation—and 798 unique synset nodes. It crossed three split seeds, four target tracks, two split modes, and nine methods, producing 1,728 aggregate and relation-stratified rows. Methods were additive identity, relation mean, relation offset, HRR/circulant, diagonal, rank-8 low-rank, and capacity-matched shuffled-label controls for each structured family.

### Primary result: strict node-disjoint contextual retrieval

| Method | Mean target MRR |
|---|---:|
| additive identity | 0.2520 |
| relation mean | 0.1151 |
| relation offset | 0.2097 |
| **HRR** | **0.2713** |
| diagonal | 0.2652 |
| low-rank rank 8 | 0.2076 |
| shuffled HRR | 0.2516 |
| shuffled diagonal | 0.2342 |
| shuffled low-rank | 0.1436 |

HRR improved MRR over additive by **+0.0194** and beat both additive and shuffled HRR on all three seeds:

| Seed | Additive | HRR | Shuffled HRR | HRR − additive | HRR − shuffled |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.2269 | 0.2431 | 0.2223 | +0.0162 | +0.0207 |
| 22 | 0.2887 | 0.3055 | 0.2893 | +0.0168 | +0.0163 |
| 33 | 0.2403 | 0.2654 | 0.2431 | +0.0251 | +0.0223 |

Relation-query accuracy for HRR was 0.216–0.275 across seeds, above the seven-way chance rate of 0.143 and above additive. The gain was concentrated in instance hypernym and member/substance meronym retrieval. Direct hypernym and part-meronym MRR did not improve, while antonym and attribute were mixed. This argues against treating all WordNet predicates as one homogeneous operator class.

### Split scientific verdict

- **Retrieval gate: PASS.** The compact HRR/circulant bias improved exact-target ranking on unseen source and target synsets, exceeded its capacity-matched shuffled-label control, and reproduced over all three splits.
- **Reconstruction gate: FAIL.** HRR's primary cosine was 0.028 lower than additive identity. It predicts a more useful ranking direction without reconstructing the exact contextual anchor.
- **Static-row claim: unsupported.** On static centered node-disjoint targets, additive remained best. The signal is sense-conditioned, not evidence that GPT-2's vocabulary table is HRR-factorized.
- **Higher-capacity claim: unsupported.** Rank-8 low-rank operators underperformed; the result is not explained by adding parameters.
- **Embedding insertion: not yet authorized.** Geometry/retrieval can nominate HRR for behavioral validation, but exact row replacement would be contradicted by the cosine result.

The next experiment should therefore be a narrow Stage C validation rather than immediate insertion: use the learned HRR transform as (1) a frozen relation-scoring sidecar and (2) an initialization for a small residual adapter, then measure relation QA/cloze MRR, definition matching, unrelated-text KL, and examples/steps to criterion. Compare against additive, shuffled HRR, definition-only, and equal-parameter adapters. Promote to experiment 02 only if HRR shifts the behavioral adaptation curve left while preserving locality.

Implementation and artifacts:

- `src/vsa_embed/wordnet_relations.py`: aligned graph, provenance-bearing nodes/edges, strict splits, frozen-host anchors;
- `src/vsa_embed/real_relations.py`: global relation models and retrieval metrics;
- `src/vsa_embed/experiments/wordnet_relations.py`: auditable YAML runner and paired gates;
- `experiments/01b-global-local-relational-factorization/stage-b-{smoke,reduced}.yaml`;
- `experiments/01b-global-local-relational-factorization/runs/stage-b-reduced/`.
