# Experiments, evaluation, and roadmap

> **Runnable-seed companion:** [`experiments/README.md`](experiments/README.md) reorganizes the proposals by beneficial-learning dependency and provides one detailed folder plus `seed.yaml` per experiment. [`experiments/toolkit-architecture.md`](experiments/toolkit-architecture.md) specifies how their outputs become a reusable PyTorch/Hugging Face attachment library.

## Core hypotheses

**H1 — global–local factorization:** globally shared relation semantics, sparsely weighted local graph participation, and a restricted concept residual can provide a transferable shortcut into behaviorally important pretrained geometry.

**H1b — bias–capacity tradeoff:** fixed-binding relation vectors learn efficiently when their induced operator matches relational geometry; constrained learned operators reduce mismatch error when needed, but excessive relation/local capacity harms transfer.

**H2 — fast concept insertion:** held-out nodes synthesized from source-trained global relations outperform lexical and graph baselines at zero update or require materially less restricted residual adaptation in a frozen causal LLM.

**H3 — structure use:** explicit binding/unbinding training improves compositional and multi-hop generalization, not merely nearest-neighbor similarity.

**H4 — online learning:** text-to-VSA memory improves persistent QA/reasoning over matched GraphRAG cost while retaining provenance and rollback.

**H5 — latent prediction:** VSA-JEPA improves relation/entity-disjoint transfer over non-VSA latent prediction and token reconstruction.

## Datasets

### Controlled

- Synthetic typed worlds with known compositional grammar, aliases, distractors, contradictions, and temporal updates.
- CLUTRR-style relational chains and family graphs.
- Generated novel words/concepts whose occurrence can be guaranteed absent from pretraining prompts.

Controlled data is essential to prove algebraic generalization and detect leakage.

### General language

- **WordNet:** taxonomy, meronymy, lexical senses; manageable first OCVF target.
- **ConceptNet:** broad multilingual commonsense, but noisy relation quality.
- **Wikidata:** rich typed/temporal graph; use timestamped dumps and bounded subsets.
- **DBpedia/Wikipedia:** definitions plus graph; high contamination risk for pretrained models.
- **ATOMIC:** event commonsense and role-like relations.
- Domain ontologies: ChEBI/GO/MeSH/UMLS where licenses permit; legal/product/tool schemas for temporal novelty.

### Dynamic knowledge

Use facts and entities created after the model's documented pretraining cutoff, while recognizing that exact training corpora may be unknown. Stronger tests use synthetic names plus real relational patterns and private held-out documents.

## Split taxonomy

Report all separately:

1. **No direct gradient:** node exists in graph but receives no target-token examples.
2. **Node-disjoint:** target node and aliases absent from fitting episodes.
3. **Component-disjoint:** connected subgraphs held out, preventing neighbor leakage.
4. **Relation-disjoint:** relation type unseen; tests schema meta-learning.
5. **Compositional-disjoint:** atomics seen, exact combination unseen.
6. **Atomic-disjoint:** some components unseen; hardest and likely needs definition induction.
7. **Temporal:** future graph updates only.
8. **No parameter update:** new concept created after model/interface training.

“Zero-shot” without this taxonomy is ambiguous.

## Baselines

### Embedding initialization

- random matched mean/norm;
- average/sum of surface subword rows;
- definition subword average;
- contextual definition encoder;
- nearest-neighbor weighted interpolation;
- WECHSEL/FOCUS-like lexical alignment;
- CoLLEGe-style concept generator;
- graph methods: TransE, ComplEx, RotatE, HolE, R-GCN plus projection;
- unconstrained matrix factorization with equal parameter count;
- HRR with random/frozen atomics;
- OCVF without residual, without graph loss, and full OCVF.

### Knowledge addition

- prompt-only definition;
- vector RAG;
- GraphRAG/HippoRAG-style retrieval;
- LoRA/fine-tuning;
- ROME/MEMIT/SERAC-class editing where model scale permits;
- exact symbolic graph query.

### JEPA

- masked language modeling/causal prediction;
- graph autoencoder and link prediction;
- contrastive graph/text encoder;
- same JEPA architecture with unconstrained latent vectors;
- deterministic VSA target without predictor (shortcut control).

## Metrics

### Geometry

- row cosine/MSE and norm error;
- kNN overlap, trustworthiness, rank correlation;
- relation analogy/unbinding accuracy;
- ontology edge vs non-edge retrieval AUC;
- residual energy and variance explained;
- frequency/degree predictability from embeddings.

### Frozen-model behavior

- new-token cloze and next-token rank/MRR;
- generation exact match plus alias-aware scoring;
- definition/attribute/relation QA under paraphrase;
- multi-hop and counterfactual reasoning;
- perplexity/KL on unrelated calibration text (locality);
- old-token benchmark retention;
- calibration/Brier/ECE and abstention.

### Continual memory

- write latency, query latency, bytes per fact;
- sequential retention and interference curves;
- contradiction/temporal query accuracy;
- provenance precision and evidence faithfulness;
- deletion/rollback success;
- capacity vs graph size and bundle degree.

### Statistical practice

Use multiple model seeds and split seeds, bootstrap confidence intervals, paired tests over identical examples, effect sizes, and correction for multiple comparisons. Pre-register the primary endpoint. Do not select only favorable frequency bins.

## Empirical update after experiment 00 milestone 2

The first 36-condition study changes the program from **“select an algebra, then scale it”** to **“learn an operating envelope, route by load, and verify exact facts.”** Across 39,360 queries, only 15/36 conditions jointly met ECE ≤5% and held-out selective accuracy ≥95%. Dimension helped strongly, candidate correlation reduced retrieval quality/coverage, heavy-tailed degree was the dominant failure regime, and real HRR, unitary HRR, and MAP were too close/inconsistent to justify a universal winner.

These results are not a reason to stop experiments 01–05: embedding factorization and typed graph construction do not require a global superposed memory to be reliable. They **are** a reason to block memory infrastructure (06), meta-learned automatic writes (07), VSA-JEPA targets (08), and consolidation (09) from assuming that one flat bundle or one global confidence threshold is safe.

## Empirical update after experiment 01a

The leakage-resistant GPT-2/WordNet pilot found real structural signal, but rejected the narrow fixed-path typed-HRR model. On held-out recipe groups, untyped addition achieved 0.139 kNN overlap versus 0.114 for MAP and 0.107 for HRR; shuffled structure fell to 0.042. Raw cosine was strongly confounded by the host mean direction. The result implies:

1. ontology neighborhood is informative, but ancestor depth is not an adequate relation schema;
2. relation binding must be treated as an ablation, not assumed correct;
3. binary equal-weight paths should be replaced by heterogeneous global relations and sparse local salience;
4. the target should include residualized/contextual geometry and frozen-host behavior;
5. success may be a large reduction in residual learning rather than perfect zero-update reconstruction.

### Immediate insertion: experiment 01b — global–local relational factorization

Run [experiment 01b](experiments/01b-global-local-relational-factorization/) before experiment 02. It compares:

- weighted addition;
- ordinary relation vectors interpreted by fixed HRR/MAP/orthogonal binding;
- compact relation vectors interpreted by one shared learned function;
- diagonal, circulant, orthogonal, block, and low-rank global relation operators;
- penalized global-plus-domain modulation;
- edge-specific operators only as a memorization upper bound.

A binding vector already induces a constrained linear operator: HRR is circulant and MAP is diagonal. Learned operators relax that constraint. The study therefore measures the **bias–capacity–data frontier**: learning curves, data/steps/FLOPs to fixed quality, asymptotic error, transfer, and residual budget. A rigid family that learns quickly but plateaus is distinguished from a flexible family that learns slowly but transfers, and from excessive capacity that merely memorizes.

Start with a synthetic teacher × learner identifiability matrix, then a fixed heterogeneous WordNet graph, residual/adaptation sweeps, bounded graph refinement, and finally behavioral confirmation. Global relation parameters, domain modifiers, local edge salience, and concept residuals must be reported separately.

**01b gate:** advance only if a globally shared relation model beats weighted addition and equal-budget graph/unstructured controls on held-out contextual or behavioral transfer, shifts the adaptation curve left, passes edge/relation-family leakage controls, transfers to a second relation/ontology family, and preserves locality.

### Immediate insertion: experiment 00b — paired sharding and risk control

Run 00b before promoting a backend and before experiment 06:

1. Materialize each graph workload once (candidate vectors, target IDs, roles, degrees, queries, and noise) and replay it across backends using common random numbers. The milestone-2 fixed-degree contrasts are clean, but variable-degree correlation/dimension aggregates used different sampled workloads and remain directional rather than causal.
2. Replace the equicorrelation-only stressor with empirical covariance spectra and nearest-neighbor distributions from WordNet-aligned host embeddings; keep equicorrelation as a controlled diagnostic.
3. Compare flat memory against typed, relation, community, degree-capped, and temporal shards at **equal bytes and candidate-recall targets**. Include exact sparse lookup and random-projection sketches.
4. Fit degree/load-aware confidence using observable features only: top-1/top-2 gap, shard load, candidate-set size, relation type, and correlation/anisotropy proxy. Compare Platt scaling with isotonic and risk-controlling/conformal selection.
5. Evaluate calibration under graph-distribution shift, not just new random seeds. Report risk–coverage curves and lower confidence bounds, rather than a single selected threshold.
6. Benchmark D=512 as the minimum serious candidate from the current study, but include D=256 and higher/equal-byte sharded alternatives. Do not treat 512 as promoted until latency and bytes are measured.

**00b gate:** promote an *operating envelope*, not one universal algebra. For each supported `(backend, dimension, shard policy, load band)`, require ≥95% candidate recall, ECE ≤5%, a one-sided 95% upper confidence bound on accepted error ≤5%, predeclared minimum coverage, and a Pareto advantage over exact/vector retrieval. Outside the envelope, abstain or use exact fallback.

### Consequences for experiments 01–09

| Experiment | Decision after 00 | Required change |
|---|---|---|
| **01a fixed-path factorization** | **Gate failed; retain as negative result** | Do not scale the ancestor-depth HRR recipe. Preserve its leakage controls and additive/shuffled baselines. |
| **01b global–local factorization** | **Stage B split result** | HRR improved strict node-disjoint target ranking over additive and shuffled HRR, but lost reconstruction cosine to identity/offset. Preserve the signal; do not insert rows yet. |
| **01c reconstruction rescue** | **Proceed now, in parallel with 00b** | Test identity-preserving residual HRR, relation offsets, shared host-to-VSA bases, predictable-residual targets, and joint reconstruction/retrieval selection on a fixed graph. |
| **01d developmental discovery** | **Blocked on 01c** | Only after operator validation, test sparse stem-cell relation specialization, masked ontology recovery, split/merge consolidation, and carte-blanche graph induction. |
| **02 insertion** | **Proceed only from a passing 01c/01d candidate** | Report pure zero-update behavior and residual-only few-shot adaptation curves. An inserted row must carry provenance, relation family/capacity, composition load, salience, residual budget, and uncertainty. |
| **03 query/readout** | **Raise priority; run after the first 01/02 candidate** | Cross depth with bundle load and degree. Train cleanup/readout on multiple backends and include degree/load features. Require gains over exact execution, random binding, and unstructured vectors on matched workloads; test out-of-envelope fallback explicitly. |
| **04 multilingual** | **Defer expansion** | Run only a small diagnostic after 02/03 pass. Multilingual anisotropy adds another correlation shift, so require language-held-out calibration and language-specific versus shared sharding controls. |
| **05 text compiler** | **Proceed in parallel, graph-first** | Compiler/frame accuracy, evidence, contradiction handling, and rollback can advance independently. Keep the exact graph authoritative; report VSA caching as a separate optional layer until 00b passes. |
| **06 episodic memory** | **Blocked on 00b** | Make VSA a candidate router/cache, never sole storage. Pre-shard hubs, preserve exact frame IDs, use conservative top-k plus fallback, and make candidate recall (before verification) the primary safety endpoint. Stress heavy-tail, tenant, temporal, and correction workloads. |
| **07 meta-learner** | **Delay automatic writes** | First learn atomics/rows under supervised evaluation only. Predict operating-envelope class, uncertainty, and “allocate/shard/abstain” decisions—not merely a vector. Automatic writes require calibration under held-out ontology-family shift. |
| **08 VSA-JEPA** | **Delay full-scale training** | Use only validated, shard-local structured targets from 01/03/05. Match target dimension/bytes and compare VSA, MAP, random-binding, and unconstrained JEPA. Stratify transfer by target load; otherwise JEPA may learn capacity artifacts rather than relations. |
| **09 consolidation** | **Remain last** | Consolidate only exact-verified, in-envelope, provenance-stable memories. Never distill low-confidence/high-load retrievals; keep the canonical graph and adapter rollback registry. |

### Revised near-term critical path

```text
00 milestone 2
 ├─> 00b paired workloads + sharding + risk control ──────────────> 06 memory
 ├─> 01a negative pilot ─> 01b split result ─> 01c operator rescue ─> 01d discovery
 │                                                                    └─> 02 insertion ─> 03 readout
 └─> 05 exact text-to-frame graph ───────────────────────────────────────────┤
                                                                  ├─> 07 meta-learning
                                                                  ├─> 08 VSA-JEPA
                                                                  └─> 09 consolidation
```

This preserves beneficial parallelism: 01 and 05 can generate useful evidence while 00b resolves memory reliability. It avoids spending substantial compute on 06–09 using an unsafe flat-memory assumption.

## Global–local factorization experiment design

### Phase 0: algebra and implementation

- Verify FFT HRR equals naive circular convolution and gradients match.
- Fix/test row normalization versus historical column normalization.
- Simulate retrieval capacity by dimension, degree, nesting, and noise.

### Phase 1a: completed fixed-path pilot

- Freeze GPT-2 rows and use leakage-safe WordNet recipe groups.
- Record the negative typed-HRR result and additive structural signal.
- Do not promote this factorizer to insertion.

### Phase 1b: relation and residual frontier

- Model: 0.5–1.5B open-weight causal LM.
- Graph: heterogeneous WordNet relations, then a second ontology family.
- First identify vector/operator behavior on synthetic teachers.
- Freeze host; jointly fit global relation representations, sparse local salience, projector, and restricted residual.
- Sweep relation capacity, edge data, residual dimension, and adaptation examples.
- Measure residualized geometry, contextual anchors, and base behavior before testing new nodes.

**Gate:** proceed only if globally shared structure preserves locality, improves held-out contextual/behavioral transfer beyond weighted addition/equal-size unstructured factorization, and reduces local adaptation cost.

### Phase 1c: reconstruction rescue

- Preserve source semantics with identity-initialized offset plus gated residual HRR.
- Test a shared regularized host-to-VSA basis and equal-capacity non-HRR controls.
- Audit prompt/layer/pooling targets and measure relation-explainable residual `R²`.
- Select from a validation Pareto frontier; keep confirmation nodes/components untouched.

**Gate:** one model must beat relation offset on held-out cosine, achieve positive relation-residual `R²`, preserve retrieval, beat shuffled/equal-budget controls, and reproduce across relation families and a second ontology.

### Phase 1d: developmental relation discovery

- Initialize generic relation stem cells only after 1c passes.
- Differentiate experts through annealed sparse routing, functional diversity, and staged split/merge consolidation.
- Compare carte-blanche, weakly seeded, and anchored ontology tracks.
- Recover masked known relations before claiming novel ontology discovery.

**Gate:** require sparse stable experts, masked-ontology recovery, functional cross-seed agreement, reduced concept-residual budget, coherent relation cards, and frozen-host behavioral/locality gains.

### Phase 2: strict unseen-node insertion

- Reserve 1,000 IDs.
- Hold out nodes before global–local factorization.
- Construct rows from graph only; optionally run a separate graph+definition track.
- Evaluate input comprehension and tied-head generation.

**Primary endpoint:** average normalized gain over strongest lexical/definition baseline on held-out behavioral tasks, with less than a predeclared locality degradation.

### Phase 3: learn to use algebra

Train only projector/gates/LoRA on source-node relation queries and composition tasks. Test deeper chains and unseen combinations. Probe whether the model can unbind the requested role rather than retrieve by lexical association.

### Phase 4: online text-to-graph memory

Feed documents sequentially. Compare exact GraphRAG, vector RAG, VSA cache + exact verification, and hybrid. Include corrections/deletions and adversarial conflicting sources.

### Phase 5: VSA-JEPA

Only after robust graph representations are established. Hide connected components or future events. Compare matched predictors with/without VSA constraints.

## Critical ablations

- weighted addition vs fixed-binding relation vectors vs shared relation-vector interpreter vs diagonal/circulant/orthogonal/block/low-rank relation operators;
- global operator vs penalized domain modulation vs edge-specific memorization upper bound;
- equal-parameter/equal-FLOP comparison and unconstrained capacity frontier;
- data/steps/FLOPs to quality threshold and asymptotic error floor;
- relation direction and explicit inverse vectors;
- one-hop vs path/recursive compositions;
- description only, graph only, both;
- fixed random atomics vs pretrained-row-distilled atomics;
- no residual, definition-derived, sparse-basis, gated, and low-rank residual over explicit dimensional budgets;
- row normalization, degree normalization, uniform versus sparse learned edge salience;
- source truth/evidence confidence versus representational salience versus query-time attention;
- tied vs untied output head;
- projection linear vs MLP vs orthogonal map;
- exact graph verification on/off;
- memory shards and dimensions.

## Failure interpretation

- **Good geometry, poor behavior:** embedding rows are not sufficient; train an interface or use sidecar attention.
- **Input success, output failure:** tied-head/logit calibration or lexicalization issue.
- **Seen nodes work, held-out fail:** relation/local capacity is memorizing; atomics may not capture transferable factors, or ontology/split may be wrong.
- **Fixed vectors learn quickly then plateau:** aligned bias with insufficient capacity; test the smallest constrained operator that closes the held-out gap.
- **Operators fit training but not new relation families:** flexibility is absorbing domain idiosyncrasy rather than learning a global relation.
- **Structure only reduces adaptation cost:** report a successful few-shot shortcut, not perfect zero-shot mastery.
- **High-degree failure:** capacity/normalization issue; shard memory.
- **GraphRAG dominates:** use VSA only as compression/router, not primary memory.
- **Random composition matches HRR:** gains may be parameter sharing/regularization rather than symbolic semantics.

## Roadmap and deliverables

### 0–2 months: reproducible core

- Extract VSA library from BERTHA, add tests/benchmarks, typed graph schema, and sparse batch generation.
- Reproduce a small rare/unseen-code result if assets permit.
- Publish exact zero-shot split definitions.
- Add paired workload manifests, degree/load-aware risk–coverage evaluation, and sharding benchmarks (00b).

### 2–5 months: OCVF proof of concept

- Synthetic relation teacher × learner matrix, heterogeneous WordNet + small causal LLM.
- Global/local factorization, residual-budget and data-to-threshold curves, then reserved-token benchmark.
- Paper-worthy result if relational initialization shifts the strict held-out adaptation curve left over strong generated-row/graph baselines, even if a small residual remains necessary.
- In parallel, complete the graph-first portion of experiment 05 without claiming a VSA-memory advantage.

### 5–9 months: behavioral interface

- Gated residual/LoRA and explicit unbinding curriculum.
- Extend to Wikidata/scientific ontology and multilingual track.

### 9–15 months: persistent graph memory

- Text-to-frame compiler, confidence/provenance, exact graph fallback, sequential benchmark.

### 12–24 months: VSA-JEPA and consolidation

- Leakage-resistant latent relational prediction.
- Selective memory-to-LoRA consolidation with rollback.

## Decision rule

The program should continue only if it demonstrates at least one advantage unavailable from simpler methods: (a) better strict held-out concept behavior, (b) materially lower memory/latency at matched accuracy, (c) reliable algebraic compositional generalization, or (d) superior auditability/rollback with competitive quality.
