# Experiments 01c–01d — Reconstruction Rescue and Developmental Relation Discovery

**Status:** 01c falsification program complete; fixed global HRR rejected for one-hop frozen-host transfer; 01d and insertion remain blocked. **Depends on:** the Stage-B WordNet/GPT-2 result in [01b](../01b-global-local-relational-factorization/). **Blocks:** experiment 02 embedding insertion. **Core rule:** repair and validate the relation operator on a fixed graph before allowing the graph and relation inventory to become latent.

## Executive decision

Experiment 01b produced a real but split result on strict node-disjoint contextual anchors:

| Model | Cosine | Target MRR | Main behavior |
|---|---:|---:|---|
| identity | 0.3063 | 0.2520 | preserves source similarity |
| **relation offset** | **0.3645** | 0.2097 | best reconstruction, weak ranking |
| pure HRR | 0.2782 | **0.2713** | best ranking, damages reconstruction |
| diagonal | 0.2916 | 0.2652 | intermediate |
| low-rank | 0.3337 | 0.2076 | flexible fit with overfitting risk |

The result does not say that structured reconstruction is impossible. It says that the **retrieval-selected pure HRR operator fails the joint reconstruction requirement**. Ranking and reconstruction currently have different winners.

The next program has two sequential scientific questions:

1. **01c — Reconstruction rescue:** with known relation labels and fixed graph edges, can an identity-preserving, basis-adapted VSA operator jointly improve reconstruction and retrieval?
2. **01d — Developmental discovery:** after the operator is validated, can initially generic relation “stem cells” differentiate into stable subrelations and induce a sparse, useful ontology?

Do not optimize operators, graph edges, relation identities, host basis, and unrestricted concept residuals simultaneously. Such a fit would be non-identifiable.

## Why pure HRR fails the reconstruction gate

The current model is

\[
\hat y=\operatorname{norm}(x\circledast r),
\]

where one source anchor and one global relation vector are expected to reconstruct the exact target contextual anchor. This makes several false or untested assumptions.

### The relation is not a function

WordNet edges are often one-to-many. A whole has several parts; a concept can have multiple hypernyms; the label `part_meronym` does not identify which part. Thus

\[
\operatorname{Var}(Y\mid X,R)>0.
\]

No deterministic operator can reconstruct target information absent from `(source, relation)`. The correct object may be a target distribution, a relation-conditioned region, the predictable target residual, or a multi-neighbor reconstruction.

### Pure binding destroys useful identity

Identity already has cosine 0.3063 because related concepts share semantics. A random-initialized circulant transform rotates away this useful component. The offset result suggests that most movement is a small correction rather than complete replacement.

### The host basis is arbitrary

Pure HRR assumes that a relation is circulant in GPT-2 coordinates:

\[
A_r\approx C(r).
\]

A more plausible hypothesis is

\[
A_r\approx P^{-1}C(r)P,
\]

for a globally shared, regularized host-to-VSA basis `P`.

### The target contains unavailable information

The target hidden state contains lemma identity, gloss content, syntax, prompt-template effects, morphology, frequency, and broad semantics in addition to relation-predictable structure. One edge cannot infer the target's unseen gloss. The target should be decomposed as

\[
y=y_{source/lexical}+y_{relation}+y_{target\ specific}+\epsilon.
\]

### Relation labels are heterogeneous

Taxonomy, meronymy, antonymy, and attributes have different directionality, symmetry, composition, and domain/range behavior. One rigid operator family is unlikely to fit all seven labels.

### Data are sparse relative to the representation

The primary node-disjoint splits contain roughly 20–25 training edges per relation in 768 dimensions. Low-rank training loss falls near 0.064 while held-out cosine remains modest, showing a train–test gap. Pure HRR remains near 0.62 training loss, indicating both approximation bias and limited evidence.

### The gate and objective are misaligned

Training used MSE plus cosine, while the successful endpoint was MRR. The gate then selected HRR by retrieval and asked the same model to beat identity in cosine. Future selection must use a validation Pareto frontier and an untouched test set.

## Combined model hypothesis

The host representation is better modeled as retained source/lexical content, a relation translation, a smaller VSA correction in a learned basis, evidence from multiple graph neighbors, and an explicitly budgeted irreducible residual:

\[
\hat y_j=\operatorname{norm}\left[
D(t_j)+B\rho_j+
\sum_{i\in N(j)}\sum_{k=1}^{K}a_{ijk}
P^T\left(Px_i+b_k+\beta_{ijk}(Px_i\circledast r_k)\right)
\right].
\]

- `D(t_j)`: optional definition evidence, declared unavailable in graph-only zero-shot tracks;
- `Bρ_j`: restricted concept-specific residual;
- `P`: shared host-to-VSA basis;
- `b_k`: relation translation;
- `r_k`: compact HRR relation atomic;
- `β`: scalar or input-conditioned structured gate;
- `a_ijk`: sparse source–target–expert assignment.

The full model is a destination, not the first experiment. Each term must earn inclusion through ablation.

---

# Experiment 01c — Reconstruction Rescue on a Fixed Graph

## Goal and falsifiable hypothesis

**Goal:** determine whether an identity-preserving and basis-adapted VSA relation operator can jointly improve exact reconstruction and target retrieval on unseen nodes when relation labels and graph edges are known.

**Hypothesis:** most target geometry is retained source/lexical structure; relation offsets explain a broad displacement; gated residual HRR contributes a smaller relation-discriminative correction in an adapted basis.

## Operator ladder

Implement and compare:

\[
\begin{aligned}
R0:&\quad T(x)=x \\
R1:&\quad T_r(x)=x+b_r \\
R2:&\quad T_r(x)=x+\beta_r(x\circledast r) \\
R3:&\quad T_r(x)=x+b_r+\beta_r(x\circledast r) \\
R4:&\quad T_r(x)=x+b_r+\sigma(w_r^Tx+c_r)(x\circledast r) \\
R5:&\quad T_r(x)=P^T[Px+b_r+\beta_r(Px\circledast r)].
\end{aligned}
\]

Initialize `b=0`, `β=0` or very small, and `P` as identity/orthogonal. The model begins as identity; relational corrections must improve held-out validation to survive. Compare fixed PCA whitening, trainable diagonal whitening, block-orthogonal/Householder, butterfly, and low-rank `I+UVᵀ` bases. Every learned basis requires an equal-parameter non-HRR control.

## Target decomposition and reliability audit

Before selecting an operator, cross:

- multiple prompt templates and gloss paraphrases;
- final token versus span pooling;
- several host layers;
- full contextual target;
- contextual minus projected static row;
- target minus a train-only source predictor `g(x)`;
- train-only whitened/top-PC-removed target;
- relation-sensitive hidden-state or logit effects.

Report paraphrase reliability, template sensitivity, source predictability, relation predictability, and an oracle estimate of conditional variance. Fit all transforms using training nodes only.

The primary residual endpoint is

\[
R^2_{rel}=1-\frac{\sum\lVert y-\hat y_{rel}\rVert^2}
{\sum\lVert y-g(x)\rVert^2}.
\]

The full target remains a required secondary endpoint; residualization must not hide poor host-space behavior.

## Objective

Use a validation-selected multi-objective loss:

\[
L=\lambda_{cos}L_{cos}+\lambda_{mse}L_{mse}
+\lambda_{rank}L_{InfoNCE}+\lambda_{rel}L_{relation}
+\lambda_{id}L_{identity}+\lambda_{alg}L_{algebra}
+\lambda_{complexity}L_{complexity}.
\]

Report the Pareto frontier rather than collapsing metrics post hoc. The identity term protects retained semantics but must not be strong enough to make learning a no-op.

## 01c stages

### 01c-A — Synthetic recovery

Generate teachers with identity plus offset, residual HRR, basis-transformed HRR, relation subtypes, one-to-many stochastic targets, concept residuals, missing edges, and noise. Require aligned learner recovery, a near-zero learned HRR gate for offset-only teachers, nonzero gate recovery for HRR teachers, and basis recovery up to valid symmetries.

### 01c-B — Target reliability audit

Select the representation target using development splits. Reject targets dominated by prompt/template variation or with negligible relation-predictable variance.

### 01c-C — Fixed-graph rescue matrix

Compare identity, relation mean, offset, pure HRR, residual HRR, gated residual HRR, offset plus residual HRR, shared-basis offset plus residual HRR, and equal-parameter low-rank residual models. Include capacity-matched shuffled-label controls for every trainable family.

Use edge-disjoint, strict node-disjoint, and connected-component-disjoint splits where feasible; nested data budgets; at least three development seeds plus two untouched confirmation seeds; and a second ontology/domain before promotion.

### Primary diagnostics

- train/test cosine and MSE;
- relation-residual `R²`;
- target MRR and Recall@K;
- relation-query accuracy;
- per-relation and relation-family results;
- source-retention versus relation-shift decomposition;
- learned gate magnitude and basis spectrum;
- train–test gap and data-to-threshold curves;
- paraphrase/template stability;
- parameters, FLOPs, and wall time.

## 01c acceptance gate

One **validation-selected** candidate must:

1. beat relation offset—not merely identity—by at least `Δcosine = +0.02` on the untouched node-disjoint test;
2. achieve positive relation-residual `R²`;
3. beat its capacity-matched shuffled-label and equal-parameter non-HRR controls;
4. preserve target MRR within a predeclared noninferiority margin and preferably improve it by `≥0.01`;
5. have a paired confidence interval above zero or win every predeclared split;
6. improve more than one relation family rather than exploit one easy predicate;
7. reproduce on a second ontology/domain;
8. remain stable across target paraphrases/templates;
9. pass frozen-host behavioral locality before insertion.

If 01c fails, do not use reconstruction to train latent ontology discovery. A ranking-only sidecar remains possible, but direct embedding insertion remains blocked.

## Implementation log — 2026-07-17

The first executable 01c milestone reuses the audited WordNet/GPT-2 Stage-B runner and adds four identity-initialized operators: residual HRR, offset plus residual HRR, input-gated offset plus residual HRR, and diagonal-basis offset plus residual HRR. The implementation records HRR coefficient, correction norm, offset norm, and basis drift; includes shuffled-relation controls; and uses a joint reconstruction/retrieval summary. Smoke configurations are explicitly `promotion_eligible: false`.

### Smoke 01c.0 — unconstrained residual

On 168 WordNet edges (24 per relation), one node-disjoint seed, and centered contextual GPT-2 anchors, the best candidate was offset plus residual HRR:

- cosine `0.2736` versus offset `0.3440` (`−0.0704`);
- target MRR `0.3820` versus pure HRR `0.3753` (`+0.0067`);
- learned correction norm `1.20` despite unit-norm source anchors; basis variants reached correction norm `2.28` and large basis drift.

This failed reconstruction and showed that an unconstrained “residual” can stop being residual and overfit the sparse graph.

### Smoke 01c.1 — bounded, regularized residual

Roles and HRR corrections were normalized, the correction coefficient was bounded to `±0.15`, and correction/offset/basis penalties were added. The best candidate became input-gated offset plus residual HRR:

- cosine `0.3482` versus offset `0.3445` (`+0.0036`);
- target MRR `0.3824` versus pure HRR `0.3879` (`−0.0056`, inside the exploratory `−0.02` margin);
- Recall@10 `0.6829` versus pure HRR `0.5854`;
- low-rank remained the strongest cosine model (`0.3764`) but did not establish a VSA-specific gain.

This is a promising diagnostic, not a pass: it is one seed, the gain is below the predeclared promotion threshold `+0.02`, and the selected gated candidate did not yet have its own shuffled-label control in that run. Matched shuffled controls for every candidate and a three-seed development configuration are now declared. **01d and embedding insertion remain blocked.**

Auditable artifacts:

- `experiments/01c-developmental-relation-discovery/runs/smoke/` — unconstrained failure;
- `experiments/01c-developmental-relation-discovery/runs/smoke-bounded/` — bounded diagnostic;
- `experiments/01c-developmental-relation-discovery/development.yaml` — next non-promotional development run.

### Development 01c.2 — three seeds and larger graph

The declared development run completed on 560 edges (80 per relation), three seeds (`11`, `29`, `47`), edge-disjoint and strict node-disjoint splits, twelve methods/controls, and 250 optimization steps. It produced 576 aggregate/per-relation result rows with the complete run contract in `experiments/01c-developmental-relation-discovery/runs/development/`.

On the primary node-disjoint condition:

| Method | Cosine | Target MRR | Recall@10 |
|---|---:|---:|---:|
| offset | 0.3488 | 0.2013 | 0.3587 |
| pure HRR | 0.2592 | 0.2437 | 0.3778 |
| offset + residual HRR | **0.3515** | 0.2189 | 0.3709 |
| gated offset + residual HRR | 0.3513 | 0.2213 | 0.3685 |
| basis + offset + residual HRR | 0.3502 | **0.2585** | **0.4087** |
| low-rank | 0.3130 | 0.2018 | 0.3873 |

The cosine-selected `offset_residual_hrr` candidate gained only `+0.0027` cosine over offset and lost `−0.0248` MRR versus pure HRR. It beat its shuffled control in all three seeds, but beat offset on cosine in only two seeds and met the retrieval margin in only one. The exploratory gate therefore failed.

The more interesting development result is the **basis model on the Pareto frontier**: relative to offset it gained `+0.0014` cosine, and relative to pure HRR it gained `+0.0147` MRR and `+0.0309` Recall@10. It also dominated both offset and pure HRR on the edge-disjoint aggregate. However, its reconstruction gain was not seed-stable: it clearly improved seed 11, regressed seed 29, and slightly regressed seed 47. This is hypothesis-generating evidence, not confirmation.

Per-relation analysis shows heterogeneous structure:

- basis HRR improved both endpoints for `attribute`, `hypernym`, and `member_meronym`, and strongly improved retrieval for `instance_hypernym`;
- it regressed `part_meronym` and modestly regressed `substance_meronym`;
- offset/gated residual models improved taxonomy retrieval but lost substantial MRR on part and substance meronymy.

This supports relation-family-specific operators or gates rather than one universal transform. It also confirms that a cosine-only automatic selector is inadequate: 01c selection must first filter candidates by the retrieval noninferiority constraint and then optimize reconstruction on a validation split.

One implementation issue was exposed by diagnostics. The basis model bounded its correction to `0.15` in encoded coordinates, but inverse diagonal scaling amplified the decoded correction to mean norm `0.4365`. Before the next run, enforce the cap in host/output coordinates or constrain the basis condition number. Also add a validation/confirmation split and evaluate the basis candidate against an equal-parameter non-HRR basis control. **No 01c promotion occurred; 01d and insertion remain blocked.**

### Development 01c.3 — controlled basis falsification

The basis confound was corrected by normalizing and capping the residual **after host-space decoding** and bounding each diagonal log-scale. A capacity-matched non-HRR control replaced circular convolution with elementwise diagonal interaction while retaining exactly the same `11,527` parameters, offset, basis, correction budget, optimizer, and data. Candidate selection was also changed to enforce retrieval noninferiority before reconstruction ranking.

On three new development seeds (`13`, `31`, `53`):

| Method | Cosine | Target MRR | Recall@10 |
|---|---:|---:|---:|
| offset | 0.3219 | 0.1858 | 0.3508 |
| pure HRR | 0.2480 | **0.2407** | 0.3529 |
| corrected basis + offset + HRR | 0.3244 | 0.2021 | 0.3602 |
| corrected basis + offset + diagonal control | **0.3274** | 0.2011 | **0.3716** |

The corrected HRR candidate beat offset reconstruction in two of three seeds by a mean `+0.0025` cosine and beat its shuffled control in all three, but lost `−0.0386` MRR to pure HRR. It failed the `−0.02` retrieval margin in **all three seeds**, so no candidate was retrieval-eligible and the two untouched confirmation seeds (`71`, `89`) were not opened.

The equal-parameter diagonal control slightly exceeded HRR in aggregate cosine (`+0.0030`) and Recall@10 (`+0.0114`). Therefore the small general reconstruction improvement is attributable to a learned relation-specific bounded correction, not specifically to HRR circular convolution. HRR retained a localized advantage over the diagonal control for `hypernym` and `instance_hypernym` retrieval (`+0.0204` and `+0.0174` MRR), but lost on most other relation families. This is insufficient for the universal fixed-graph rescue claim.

The corrected diagnostics behaved as intended: mean host-space HRR correction norm was `0.1368`, below the `0.15` cap, and basis drift was small. Thus the failure is no longer explained by decoded correction amplification.

**Decision:** reject the universal exact-target reconstruction formulation for 01c in its current form. Do not spend confirmation seeds, start 01d, or insert embeddings. Preserve the ranking-only HRR signal and move to a narrower next experiment: taxonomy-only relation-conditioned retrieval and predictable-residual/distributional targets, with exact target reconstruction retained only as a secondary endpoint. Artifacts are in `experiments/01c-developmental-relation-discovery/runs/basis-confirmation-development/`.

### Development 01c.4 — taxonomy-only distributional retrieval

The narrower test retained only `hypernym` and `instance_hypernym`, used a multi-positive contrastive objective over all training targets sharing `(source, relation)`, and evaluated any corresponding held-out target as correct. Exact-target MRR and cosine were secondary diagnostics. Three new node-disjoint development seeds (`17`, `37`, `59`) compared corrected basis HRR with the exactly matched diagonal control and shuffled relations.

| Method | Distribution MRR | Distribution R@10 | Exact MRR | Cosine |
|---|---:|---:|---:|---:|
| offset | 0.2387 | 0.5785 | 0.2374 | 0.0824 |
| pure HRR | 0.1890 | 0.4378 | 0.1861 | 0.0424 |
| basis + offset + HRR | 0.2317 | 0.5683 | 0.2303 | 0.1136 |
| basis + offset + diagonal control | **0.2387** | **0.5847** | **0.2373** | **0.1167** |
| shuffled basis HRR | 0.1263 | 0.2958 | 0.1257 | 0.0280 |

HRR beat its shuffled control in all three seeds by `+0.1054` distribution MRR and exceeded offset cosine by `+0.0312`, showing that true relation labels and learned corrections matter. However, it lost `−0.0070` distribution MRR to the equal-parameter diagonal operator, winning only seed `59`; the predeclared HRR-specific gate therefore failed.

The relation audit did not rescue the claim. For ordinary hypernyms, basis HRR reached `0.2609` distribution MRR versus `0.2672` for diagonal control. For instance hypernyms it reached `0.2209` versus `0.2296`. The diagonal control also had lower training loss, slightly better cosine, and the same bounded correction budget. Circular convolution again failed to add value beyond generic elementwise relation conditioning.

This run also exposed a benchmark-design limitation: the mean number of valid held-out targets per `(source, relation)` was only `1.023` (`1.000` for hypernym and `1.043` for instance hypernym). The implemented multi-positive method is valid, but this sampled WordNet graph scarcely tests one-to-many uncertainty. A genuine distributional experiment must preserve source-centric neighborhoods during graph collection and splitting, then evaluate set retrieval, calibration, and coverage rather than treating mostly singleton edges as distributions.

**Decision:** reject fixed global circular-convolution HRR as the privileged taxonomy operator in frozen GPT-2 coordinates. Do not proceed to 01d or insertion from this branch. The next justified study is an operator-neutral, source-centric hierarchy benchmark with full sibling/parent neighborhoods and predictable-residual targets. If VSA structure is revisited, test whether its value lies in explicit compositional path algebra or bundled graph memory—not one-hop host-space transfer. Artifacts are in `experiments/01c-developmental-relation-discovery/runs/taxonomy-distribution-development/`.

### Development 01c.5 — complete source-centric parent neighborhoods

The benchmark-design concern from 01c.4 was resolved before testing another model. Instead of sampling individual edges, 01c.5 sampled `(source, relation)` queries and retained **every aligned parent**. It contains 149 queries over 385 nodes and 301 edges: 120 hypernym queries, all 29 eligible instance-hypernym queries, 146 two-parent sets, and three three-parent sets. Each seed held out 37 complete queries; no query was fragmented across train and test.

Evaluation occurs once per query rather than once per edge. Primary endpoints are best-valid-parent MRR and recall of the complete parent set at 10. Secondary endpoints are any-parent Hit@10, negative log probability mass assigned to the parent set, and cosine to the parent centroid—the predictable first moment of the set-valued target.

| Method | Query MRR | Set R@10 | Hit@10 | Set NLL ↓ | Centroid cosine |
|---|---:|---:|---:|---:|---:|
| offset | 0.3282 | 0.4009 | 0.6757 | 3.0249 | 0.1718 |
| low-rank rank-4 | **0.4386** | **0.4640** | **0.7027** | 3.4548 | **0.2590** |
| basis + offset + HRR | 0.3411 | 0.4099 | 0.6847 | 3.0816 | 0.1944 |
| basis + offset + diagonal control | 0.3722 | 0.4234 | 0.6937 | **2.9450** | 0.2106 |
| shuffled basis HRR | 0.2413 | 0.3108 | 0.5315 | 3.9214 | 0.0974 |

The corrected HRR model again used exactly the same 3,842 parameters as its diagonal control. It beat shuffled relation neighborhoods in every seed by `+0.0998` query MRR, but lost to diagonal conditioning by `−0.0311` MRR and `−0.0135` set Recall@10. It also had worse set-mass NLL (`3.0816` versus `2.9450`) and lower centroid cosine (`0.1944` versus `0.2106`). HRR lost against diagonal in **all three seeds**. The predeclared development gate failed.

The operator-neutral baselines reveal a meaningful capacity/calibration trade-off rather than an HRR effect. Rank-4 low-rank transfer is best on query MRR, complete-set recall, any-parent Hit@10, and parent-centroid reconstruction, but has worse set-mass NLL than offset or diagonal conditioning. The exactly matched diagonal model is the best calibrated. This suggests separating a **set retrieval head** from a **probability-calibration head** and controlling flexible-model capacity, rather than forcing one point operator to represent a multimodal target.

**Stopping decision:** after matched controls, two taxonomy-only studies, genuine multi-parent neighborhoods, three fresh seeds each, and aligned ranking/set/reconstruction metrics, fixed global circular convolution is rejected for one-hop transfer in frozen GPT-2 space. Do not tune this branch further, open confirmation seeds, start latent relation discovery, or insert rows from it. The next VSA-specific test must exercise a capability that VSA algebra actually predicts—multi-hop role composition/inversion or bundled graph-memory querying—against symbolic traversal, TransE/RotatE-style operators, diagonal/MAP binding, and matched neural baselines. Artifacts are in `experiments/01c-developmental-relation-discovery/runs/source-neighborhood-development/`.

---

# Experiment 01d — Developmental Relation Discovery

## Goal and stem-cell hypothesis

**Goal:** determine whether initially generic relation experts can differentiate into stable, sparse, interpretable relation subtypes and discover graph structure without memorizing target anchors.

Initialize a bank of weakly differentiated relation “stem cells.” Each receives a small perturbation around an identity-preserving residual operator, competes for edges, specializes under sparse routing, and is later merged, pruned, or split according to held-out functional behavior.

Different initialization alone is insufficient. Without explicit routing and a developmental schedule, likely outcomes are expert collapse, winner-take-all routing, dead experts, arbitrary fragmentation, and seed-specific ontologies.

## Relation experts and routing

For expert `k`,

\[
T_k(x)=P^T[Px+b_k+\beta_k(Px\circledast r_k)].
\]

Use inference-available source/context features in a sparse router:

\[
p(k\mid x,c)=\operatorname{entmax}(g_k(x,c)/\tau).
\]

Target-conditioned routing may be reported only as an oracle upper bound. It cannot define zero-shot inference.

Test three levels:

1. one expert per known relation;
2. two to four latent subexperts under each known broad relation family;
3. 20 unlabeled global stem cells for carte-blanche discovery.

Separate global relation identity from domain modulation:

\[
T(x,r,d)=T_r^{global}(x)+\Delta T_{r,d}(x).
\]

This tests whether “animal is-a” is a new relation or a bounded domain-specific variant of shared taxonomy.

## Sparse graph variables

L2 controls magnitude but does not select edges. Use entmax/sparsemax, hard-concrete L0 gates, top-k budgets, group lasso, or proximal L1 updates. Store independently:

- edge existence;
- expert assignment;
- representational salience;
- truth confidence;
- evidence and provenance.

Never equate a geometrically useful edge weight with ontological truth.

Avoid materializing an unrestricted `O(N²K)` graph. Build candidates from semantic neighbors, type compatibility, known ontology edges, lexical/co-occurrence evidence, extracted propositions, and fixed random negatives. Permit only a budgeted exploration set outside these candidates. Key–query retrieval should reduce the active graph to approximately `O(NMK)`.

## Developmental schedule

| Phase | Routing | Diversity | Sparsity | Concept residual |
|---|---|---|---|---|
| warm-up | soft and balanced | modest | weak | zero/minimal |
| differentiation | temperature annealed | strong functional | increasing | minimal |
| commitment | entmax/top-k | maintained | strong | restricted |
| consolidation | near-hard | merge-tested | fixed budget | calibrated |
| adaptation | ontology fixed | stable | fixed | small residual allowed |

Use early load balancing and minimum utilization, functional diversity penalties, a bounded dead-expert reset window, multiple restarts, and explicit split/merge operations. Load balancing must decay; forcing every expert to remain active can manufacture meaningless relations.

## Alternating optimization

Joint graph/operator learning is non-identifiable. Use:

1. freeze anchors and candidate edges;
2. learn soft assignments with simple residual operators;
3. freeze assignments and fit operators;
4. update sparse edge salience;
5. permit a bounded set of edge changes;
6. consolidate duplicate/dead experts;
7. introduce a restricted concept residual only after structure stabilizes;
8. evaluate untouched nodes/components before another cycle.

Use minimum-description-length penalties over active experts, edges, residual dimensions, and parameter bits.

## Functional consolidation

Do not merge experts using parameter-vector cosine alone. Compute functional distance

\[
D_{kl}=\mathbb E_{x\sim\mathcal D}[1-\cos(T_k(x),T_l(x))]
\]

and combine it with routing overlap, held-out prediction agreement, composition/inverse signatures, downstream behavior, and the validation cost of a tentative merge. Split an expert only when its held-out residuals contain stable, reproducible multimodal structure.

## Relation algebra

Apply constraints only to justified families:

- inverse consistency;
- antonym symmetry/involution;
- hierarchy direction, composition, and selected transitivity;
- domain/range compatibility;
- acyclicity for hierarchical experts;
- noncommutative composition where expected.

These constraints distinguish relations from predictive clusters, but incorrect universal algebra can impose false ontology structure.

## 01d stages

### 01d-A — Synthetic stem-cell identifiability

Generate worlds with 4, 8, 12, and 20 true relations; global-plus-domain modulation; redundant relations; severe class imbalance; one-to-many edges; missing/spurious edges; and known inverse/composition rules. Initialize 20 experts and evaluate effective relation count, edge assignment, operator recovery, domain/relation separation, and duplicate pruning.

Ablate load balancing, diversity, sparsity, annealing, functional pruning, ontology seeding, and fixed-`K` versus split/merge.

### 01d-B — Masked WordNet recovery

Use WordNet labels only for evaluation in subtype-hidden, domain-hidden, relation-family-hidden, edge-hidden, and component-hidden tracks. Report adjusted/normalized mutual information, Hungarian-matched F1, edge average precision, functional recovery, and cross-seed stability.

### 01d-C — Initialization comparison

Run:

- **carte blanche:** random/equivariant experts without ontology labels;
- **weakly seeded:** broad relation priors that may move, split, merge, or die;
- **anchored refinement:** trusted edges fixed or strongly regularized while free experts model missing structure.

This separates ontology discovery from confirmation bias.

### 01d-D — Sparse graph induction

Learn edge existence, expert assignment, salience, and calibrated confidence on the bounded candidate graph. Withhold concept-specific residuals during initial discovery; later measure the reduction in residual bits achieved by the induced ontology.

### 01d-E — Interpretation

For each expert, emit a relation card containing high-responsibility edges, counterexamples, source/target type distributions, domain specialization, known-label enrichment, inverse/composition partners, functional neighbors, seed stability, held-out utility, proposed natural-language labels, and uncertainty. Unstable or artifact-driven experts remain unnamed.

### 01d-F — Frozen-host behavioral validation

Use the discovered structure as a relation-scoring sidecar, retrieval augmentation, or initialization for a small residual adapter before attempting row insertion. Measure relation QA/cloze, definition matching, compositional queries, examples/steps to criterion, unrelated-text KL/perplexity, old-token retention, and graph-edit reversibility.

## 01d acceptance gate

A developmental model advances only if it:

1. improves held-out reconstruction over offset and equal-budget low-rank controls while preserving retrieval;
2. produces a materially sparse graph and shorter description than dense assignments;
3. recovers masked known structure above shuffled controls;
4. yields functionally stable experts across seeds after permutation matching;
5. survives merge tests and does not depend on target-conditioned inference;
6. produces coherent held-out relation cards or explicitly leaves experts unlabeled;
7. reduces the concept-specific residual budget;
8. improves frozen-host behavior with acceptable locality;
9. reproduces on a second ontology/domain.

Training reconstruction alone is never sufficient.

---

# Controls, implementation, and artifacts

## Required baselines and falsification controls

- identity, relation mean, and relation offset;
- pure HRR and equal-parameter low-rank/non-HRR adapters;
- nearest-neighbor graph without relation experts;
- supervised WordNet relation reference;
- shuffled labels, endpoints, and contextual targets;
- frozen random experts;
- edge-specific memorization upper bound;
- definition-only and graph-plus-definition tracks;
- seeded versus unseeded experts;
- no-sparsity, no-diversity, and no-algebra ablations;
- parameter-, step-, and FLOP-matched comparisons.

## Proposed implementation modules

```text
src/vsa_embed/
├── residual_relations.py       # residual/gated/shared-basis transforms
├── relation_routing.py         # entmax, top-k, hard-concrete, balancing
├── graph_discovery.py          # candidates, overlays, alternating optimization
├── relation_interpretation.py  # expert matching, distance, relation cards
└── experiments/
    ├── reconstruction_rescue.py
    └── developmental_relations.py
```

Core interfaces should include:

```python
class ResidualRelationTransform: ...
class GatedResidualHRR: ...
class SharedBasisRelationTransform: ...
class RelationRouter: ...
class FunctionalDiversityLoss: ...
class CandidateGenerator: ...
class SparseGraphOverlay: ...
class ExpertSplitMergeController: ...
class FunctionalExpertDistance: ...
class ExpertAligner: ...
class RelationCardBuilder: ...
```

Routing APIs must identify which features are available at inference. Graph records must preserve source truth, learned salience, confidence, and provenance separately.

## Tests

Unit and regression tests must verify:

- residual HRR is exactly identity at initialization;
- `β=0` removes the HRR contribution;
- basis encode/decode preserves vectors initially;
- sparse routers produce the declared zeros/budget;
- hard-concrete evaluation is deterministic;
- target-only features cannot enter inference routing;
- train-only preprocessing never sees test nodes;
- inverse/composition losses behave on fixtures;
- functional distance recognizes equivalent operators;
- duplicate experts merge and multimodal experts split on synthetic worlds;
- sparse edges and known relation counts are recoverable;
- shuffled worlds fail the scientific gates;
- every reported gate recomputes from raw metrics.

## Run artifacts

In addition to the shared run contract, persist:

- exact node/component splits and target preprocessing state;
- model/data revisions and target prompts;
- checkpoints and operator/basis spectra;
- expert parameter and functional trajectories;
- routing utilization and entropy over time;
- graph snapshots and edge provenance;
- split/merge/prune event log;
- expert alignment across seeds;
- per-relation/per-expert predictions;
- relation cards;
- machine-readable Pareto and acceptance decisions.

## Decision sequence

```text
01b retrieval signal / reconstruction split
  └─> 01c-A synthetic operator recovery
       └─> 01c-B target reliability audit
            └─> 01c-C fixed-graph joint rescue
                 ├─ fail ─> ranking sidecar only; insertion blocked
                 └─ pass ─> 01d-A synthetic stem-cell recovery
                              └─> 01d-B masked WordNet recovery
                                   └─> 01d-C seeded vs unseeded
                                        └─> 01d-D sparse graph induction
                                             └─> 01d-E interpretation
                                                  └─> 01d-F behavior/locality
                                                       └─> 02 insertion
```

## First implementation slice

The first runnable matrix is deliberately narrow:

- identity, offset, pure HRR, residual HRR, offset plus residual HRR, gated residual HRR, shared-basis residual HRR, and equal-parameter low-rank;
- full contextual, source-residual, and train-whitened residual targets;
- strict node-disjoint splits with three development and two confirmation seeds;
- shuffled-label controls;
- joint cosine/residual-`R²`/MRR reporting.

Only after one model jointly beats the offset reconstruction baseline and preserves the HRR retrieval signal should the 20-expert developmental learner be implemented.
