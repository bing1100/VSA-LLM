# Research proposals

## Rating scheme

Novelty is calibrated against work known at the 2026-07-17 cutoff, not a patentability claim.

- **High:** the specific combination/objective appears substantially underexplored.
- **Medium:** meaningful extension or synthesis of active lines.
- **Low:** useful engineering or validation, but close analogues exist.
- **Impact:** potential scientific/practical value if the hypothesis succeeds.

## Portfolio summary

| Proposal | Core gap | Novelty | Potential impact | First cost |
|---|---|---:|---:|---:|
| 1. Ontology-Constrained VSA Factorization (OCVF) | Existing rows are rich but unstructured; random atomics ignore them | **High** | **Very high** | Medium |
| 2. Zero-gradient compositional vocabulary insertion | New concepts normally need examples/optimization | Med–high | High | Low–medium |
| 3. Text-to-VSA compiler and self-growing graph | Reading does not become durable, inspectable structure | Med–high | **Very high** | Medium–high |
| 4. Amortized VSA meta-learner | Graph construction and embedding fitting are per-domain | Medium | High | Medium |
| 5. Provenance-aware VSA episodic memory | RAG is verbose; weight edits are hard to audit/undo | Medium | High | Medium |
| 6. Relational VSA-JEPA | Language SSL predicts tokens, not explicit latent relations | **High**, speculative | **Very high** | High |
| 7. Bidirectional VSA query/readout training | Structured inputs do not ensure explicit algebra use | Medium | High | Medium |
| 8. Multilingual ontology pivot | Cross-lingual rows lack explicit shared concept identity | Medium | High | Medium |
| 9. Hierarchical capacity-aware memory | Flat superposition degrades at realistic graph scale | Medium | High | Medium |
| 10. Continual VSA-to-weights consolidation | External memory and parametric memory remain disconnected | Med–high | High | High |

## 1. Ontology-Constrained VSA Factorization (OCVF)

### Idea

Given a pretrained embedding table `E`, an ontology graph `G`, and VSA composition function `f_G`, recover atomic vectors `A`, relation vectors `R`, an alignment map `P`, and residual rows `U`:

```text
ê_i = gate_i · P f_G(i; A,R) + (1-gate_i) · U_i
```

Optimize on ontology-linked existing tokens/entities:

```text
L = λrow Lcos(ê_i,E_i)
  + λnbr Lneighborhood(ê,E)
  + λbeh KL[p_frozen(.|x; E) || p_frozen(.|x; Ê)]
  + λgraph Lrelation
  + λreg Lcapacity/residual
```

Then construct a held-out/new concept `j` by `P f_G(j; A,R)` without fitting `U_j` or exposing token occurrences.

### Novelty

**High for the complete inverse-plus-forward formulation.** CoLLEGe and vocabulary-transfer methods generate rows from definitions/lexical anchors; HRR-BERT learns atomics from task data; HolE and KG embeddings learn graph scoring spaces. The proposed distinction is to *factor an existing LLM embedding manifold through a shared ontology algebra* and use the recovered factors for forward synthesis. Do not claim that generated embeddings or graph embeddings themselves are novel.

### Impact

- Turns a pretrained vocabulary into a reusable atomic “concept chemistry.”
- Preserves years of linguistic pretraining while making part of the space interpretable and editable.
- Enables new-node initialization, cross-domain transfer, and ontology diagnostics.
- Could compress large entity vocabularies if residuals are sparse/low-rank.

### Falsification

Reject the hypothesis if OCVF cannot beat mean-of-subwords/definition generation on held-out concept behavior at matched parameters, or if reconstructed rows materially degrade base-model perplexity/locality.

## 2. Zero-gradient compositional vocabulary insertion

### Idea

Reserve token IDs in an open-weight causal LLM. For a new concept, create both input and tied output rows from known atomic definitions/relations, optionally calibrating only a global projection learned beforehand. No new-concept gradient step is allowed.

Example:

```text
QUOKKA = IS_A ⊛ MARSUPIAL + HABITAT ⊛ AUSTRALIA
       + APPEARANCE ⊛ (SMALL + FURRY)
```

Evaluate whether the frozen model can understand `<QUOKKA>`, choose it as an answer, and generate it in suitable contexts.

### Novelty and impact

**Medium–high novelty:** zero-shot rows are active research, but explicit relation-role composition plus tied-output evaluation is less established. **High impact:** rapid domain vocabulary updates, rare scientific entities, tools, products, legislation, and personal concepts without full fine-tuning.

### Falsification

The model must outperform lexical initialization under truly held-out nodes and not merely copy definitions from context. Input-only success is insufficient; test output probability and paraphrase generalization.

## 3. Text-to-VSA compiler and self-growing knowledge graph

### Idea

After reading a document, an LLM proposes normalized frames:

```json
{"subject":"X", "relation":"inhibits", "object":"Y",
 "time":"2026", "polarity":"positive", "confidence":0.84,
 "evidence":"doc:line-span"}
```

Entity resolution maps known nodes to atomics and allocates new ones. A schema validator checks types/cardinality; contradiction detection compares existing claims. Approved frames update a canonical temporal property graph and a VSA cache. The LLM can later retrieve/unbind the structure or receive its projected vector.

### Novelty and impact

LLM graph extraction is established; **medium–high novelty** lies in closed-loop compilation into a differentiable, reversible VSA cache that can synthesize model-compatible concept embeddings. **Very high impact** if robust: inspectable lifelong learning, compact memory, deletion, and provenance.

### Critical safeguard

The VSA is a cache, not authoritative storage. Exact triples/evidence remain canonical. New claims enter a quarantine layer until corroborated; confidence, time, source, and negation are explicit roles.

### Falsification

Compare with storing the same frames in ordinary GraphRAG. If VSA adds no quality/latency/memory benefit, or errors accumulate under sequential updates, prefer the simpler graph.

## 4. Amortized VSA meta-learner

### Idea

Train a hypernetwork `H` across many episodic ontologies. It receives definitions, examples, neighboring frames, and the host LLM's embedding anchors, then predicts:

- entity and relation atomics;
- composition gates/weights;
- uncertainty and required new atomic capacity;
- an alignment residual.

Episodes hide nodes and relations, so `H` learns *how to create a VSA representation*, not a fixed graph.

### Novelty and impact

**Medium novelty:** embedding hypernetworks/meta-learning exist, but ontology-algebra induction with explicit compositional output and host-manifold alignment is a distinct synthesis. **High impact:** fast adaptation to a new scientific schema or organization without rebuilding atomics manually.

### Falsification

Test on ontologies and relation labels unseen during meta-training. If it only memorizes common schema templates or loses to a frozen definition encoder, it is not meta-learning the intended structure.

## 5. Provenance-aware VSA episodic memory

### Idea

Encode each episode with bound roles for entity, relation, value, source, time, confidence, and episode ID. Use hierarchical bundles and cleanup indices. Retrieve algebraically, then verify against exact stored frames before generation.

### Novelty and impact

Associative VSA memory is foundational and LLM long-term memory is active; novelty is **medium** for a provenance/temporal VSA memory explicitly integrated with an LLM and benchmarked against GraphRAG/SERAC. Impact is **high** for compact, reversible, auditable memory with constant-width summaries.

### Falsification

Measure retrieval degradation with load and conflicting episodes. If exact vector/graph retrieval is more accurate at acceptable cost, VSA should serve only as a routing sketch.

## 6. Relational VSA-JEPA

### Idea

Mask a node, edge set, event, or future graph state. A context encoder reads surrounding text/subgraph; a predictor receives a query role (e.g. `CAUSE_OF`, future time); a slow target encoder produces a VSA-structured target. Predict target latent structure, not tokens:

```text
z_context, q_relation → predictor → ẑ_target
                                  ≈ stopgrad(z_target)
```

Regularize variance/covariance and require successful unbinding/link prediction to prevent collapse.

### Novelty and impact

**High but speculative novelty:** JEPA has expanded across modalities, while direct VSA-structured relational JEPA for language/knowledge graphs appears sparse. **Very high potential impact:** a world model whose latent state has explicit compositional operations, supports abstract prediction, and avoids reconstructing irrelevant surface text.

### Falsification

Compare against masked LM, graph autoencoder, contrastive graph encoder, and identical JEPA without VSA constraints. Split by connected components/time and remove identity/definition shortcuts. If VSA only improves probe access but not transfer/reasoning, narrow the claim.

## 7. Bidirectional VSA query/readout training

### Idea

Train the LLM to explicitly execute binding/unbinding: given a concept vector and relation query, recover the filler; given text, emit a composition; given two compositions, detect shared factors. Add auxiliary cleanup and cycle-consistency losses.

### Novelty and impact

**Medium novelty:** VSA query operations are established, but current HRR-BERT only supplies vectors. Demonstrating a transformer can reliably manipulate the algebra would materially strengthen the neuro-symbolic claim. Impact: interpretable intermediate queries and better multi-hop composition.

## 8. Multilingual ontology pivot

### Idea

Use language-independent concept atomics plus language-specific lexical/phonological residuals:

```text
embedding(surface, lang) = concept_VSA + language_adapter(surface,lang)
```

New language rows inherit ontology semantics even when lexical overlap is low.

### Novelty and impact

Cross-lingual lexical initialization is established; **medium novelty** comes from explicit shared relational composition and zero-shot graph-node synthesis. Impact is high for low-resource languages and multilingual terminology systems.

## 9. Hierarchical capacity-aware VSA memory

### Idea

Estimate per-bundle signal-to-noise and allocate multiple sketches by community/type/time. Route queries to sparse submemories; use exact graph fallback when confidence is low. Learn unitary relation vectors and degree-aware normalization.

### Novelty and impact

**Medium novelty/engineering contribution.** Capacity management exists in associative memory, but realistic LLM graph integration needs rigorous scaling laws. Impact: determines whether the approach survives millions of facts rather than demos.

## 10. Continual VSA-to-weights consolidation

### Idea

Keep new knowledge in VSA episodic memory first. Periodically distill well-supported, frequently used structures into a small adapter/LoRA while preserving outputs on a locality buffer. Retain the graph/VSA recipe for rollback.

### Novelty and impact

**Medium–high novelty** for staged symbolic-to-parametric consolidation with reversible provenance; related memory/editing/distillation components exist. Impact: combines immediate updates with efficient long-term inference.

### Falsification

Compare against keeping all facts in retrieval and ordinary sequential LoRA. Require edit success, multi-hop portability, locality, retention, rollback, and compute advantages.

## Recommended prioritization

1. **OCVF + zero-gradient insertion** — sharp hypothesis, feasible with a frozen small LLM, directly extends this code.
2. **Explicit VSA query/readout** — verifies that structure is usable rather than decorative.
3. **Text-to-VSA graph with exact fallback** — practical persistent learning.
4. **Meta-learner and multilingual pivot** — scale across schemas/languages after basic validity.
5. **VSA-JEPA and consolidation** — highest upside, but only after leakage/capacity/interface issues are understood.

Detailed, dependency-ordered experiment seeds are available in [`experiments/`](experiments/README.md). That program starts with proposal 9's capacity work because it supplies the algebra and operating envelope needed by proposal 1, then proceeds through factorization, insertion, readout, graph induction, memory, meta-learning, JEPA, and consolidation. The resulting artifacts are designed to accumulate into a reusable [VSA embedding toolkit](experiments/toolkit-architecture.md).
