# Literature review: VSA embeddings, new concepts, memory, graphs, and JEPA

## Scope and method

Searches were run on OpenAlex and arXiv on 2026-07-17 using combinations of *vector symbolic/hyperdimensional*, *transformer*, *concept embedding generation*, *vocabulary expansion/tokenizer transfer*, *knowledge editing/lifelong memory*, *knowledge graph construction/ontology*, and *JEPA/language*. Citation chains from the HRR-BERT paper were also followed. This is a focused narrative review, not a PRISMA systematic review. Recent preprints are evidence of activity, not settled results.

## 1. VSA and HRR

VSAs represent symbols with high-dimensional distributed vectors and provide binding, bundling, permutation, similarity, and approximate inverse operations. Plate's HRR uses circular convolution; later surveys organize many binary, bipolar, real, and complex variants. Their key appeal here is that symbolic compositionality remains in a fixed-width differentiable vector.

Related progress:

- **HolE** uses circular correlation for knowledge-graph link prediction, establishing a close relation between holographic algebra and graph embeddings.
- Modern VSA/HDC surveys emphasize robustness, efficient hardware, and encoding design, while also highlighting capacity, cleanup, and compositional-depth limits.
- **Generalized HRR** (2024 preprint; 2026 IEEE TAI record) expands the algebra and should be compared with ordinary circular convolution.
- **GPT-2 Through the Lens of VSAs** (2024 preprint) and **Attention as Binding** (2025 preprint) analyze transformer computation through VSA concepts. These support conceptual compatibility but do not establish ontology-generated vocabulary rows.
- RESOLVE and VSA4VQA show renewed interest in object/relational VSA processing beyond toy symbolic tasks.

**Gap:** current work rarely starts from a strong pretrained LLM, recovers a reusable symbolic atomic dictionary from its embedding geometry, and then tests algebraic creation of truly unseen language concepts.

## 2. Knowledge-infused and graph representations

Knowledge-graph embeddings (TransE-family, ComplEx, RotatE, HolE), relational GNNs, and ontology-informed learning provide strong baselines. They usually optimize entity/relation scores or contextual representations; they do not necessarily expose a compositional token row compatible with a frozen LLM's input/output space.

GraphRAG and HippoRAG externalize knowledge and retrieve relevant subgraphs. Language Models are Open Knowledge Graphs and subsequent extraction work show that LMs can produce triples, while recent LLM pipelines construct and align ontologies. External graph memory is editable and attributable but incurs retrieval, serialization, and context costs. Parametric embeddings are fast but difficult to update and audit.

**Opportunity:** use a VSA as a compact algebraic graph memory and as a bridge into the LLM, while retaining a canonical symbolic graph for exact truth, provenance, and deletion. The VSA must not be the sole source of truth because superposition is lossy.

## 3. New vocabulary and generated concept embeddings

Vocabulary adaptation is a close prior-art boundary:

- **WECHSEL** initializes target-language subword rows from cross-lingual lexical similarity.
- **FOCUS**-style vocabulary replacement aligns new tokens to semantically similar shared tokens.
- **Zero-shot cross-lingual alignment for embedding initialization** and 2025–2026 tokenizer-transplant/distillation work preserve model geometry while replacing vocabularies.
- **CoLLEGe: Concept Embedding Generation for LLMs** (2024 preprint) trains a generator that produces a new concept embedding from a definition and a few demonstrations.
- 2025–2026 preprints explore attention-aware token distillation, orthogonal matching pursuit, model-aware tokenizer transfer, concept tokens, lexical grounding, and geometry-preserving vocabulary expansion.

These methods make “generate new token embeddings” alone **not novel**. Most, however, generate each row from lexical/contextual evidence rather than force a shared, queryable ontology algebra whose atomics can be reused to construct future nodes.

**Specific gap for OCVF:** jointly solve an inverse problem—fit atomics so ontology compositions reconstruct a pretrained embedding manifold—and a forward problem—compose held-out nodes from those atomics. A residual channel and neighborhood-preservation loss are essential because ontology structure cannot explain all pretrained semantics.

## 4. Model editing, continual learning, and memory

ROME and MEMIT edit factual associations in transformer weights; MEND learns an editor; SERAC routes edited cases to an explicit memory. Surveys show persistent tensions among edit success, paraphrase generalization, locality, portability to related facts, sequential stability, and rollback. MQuAKE demonstrates that successful single-hop edits may fail multi-hop reasoning.

Retrieval and long-term agent memories avoid destructive weight changes but depend on retrieval quality and consume context. Continual fine-tuning risks catastrophic forgetting. Recent 2026 work continues null-space, low-rank, lifelong, and inference-time editing, indicating the problem remains open.

A VSA sidecar offers fast addition and approximate relational lookup, but it does not automatically alter the model's behavior. A learned interface, retrieval gate, or limited consolidation step is needed. Claims of “the LLM learned the fact” should distinguish external accessible memory from changed parametric behavior.

## 5. Meta-learning and concept induction

Meta-learning learns an update rule or amortized inference process across tasks. In-context learning can induce temporary task behavior without weight updates, but context disappears and internal representations are not reliably inspectable. Definition encoders and hypernetworks can amortize new embedding creation.

The proposed **text-to-VSA compiler** is a structured meta-learner:

```text
documents/examples → typed propositions + confidence → atomic assignments/compositions → concept vector → LLM interface
```

The hard research questions are entity resolution, relation typing, contradictions, uncertainty, and whether induced structures improve downstream behavior beyond simply retaining the source text in RAG. The LLM should propose a graph; validators and evidence stores should approve it. Self-generated structure should never be accepted as ground truth without provenance.

## 6. JEPA relationship

JEPA predicts representations of missing target regions from context rather than reconstructing every observation. I-JEPA and V-JEPA demonstrate this most clearly in vision/video; audio and multimodal variants followed. JEPA is an objective/architecture family, not a particular embedding-construction algorithm.

VSA and JEPA are complementary:

- VSA specifies a **structured latent target** and algebra.
- JEPA supplies a **predict-in-latent-space objective**.
- A target encoder can map observed text/subgraphs to a VSA-like concept state.
- A predictor receives context plus relation/query roles and predicts the held-out node, edge bundle, or future graph state.

This could avoid token-level reconstruction and prioritize predictable semantic structure. However, an HRR target built deterministically from the same ontology can make the task trivial. Valid tests must hide entities/edges/evidence, prevent identity shortcuts, and compare to ordinary graph autoencoders, contrastive learning, masked language modeling, and latent predictive baselines without VSA structure.

## 7. Where the field is and is not

| Claim | Assessment at cutoff |
|---|---|
| VSA can encode structured graphs and support approximate queries | Established, with known capacity/noise limits. |
| Structured token embeddings can help rare/unseen medical codes | Supported by HRR-BERT; external replication and broader domains needed. |
| New LLM token rows can be generated from lexical/definition evidence | Active and increasingly established. |
| An ontology can be distilled into pretrained LLM-compatible atomics | Plausible but not found as a mature standard method; key proposed gap. |
| An LLM can read text and reliably write new persistent VSA knowledge | Research hypothesis; extraction is feasible, reliable autonomous consolidation is unsolved. |
| VSA-JEPA improves language/world modeling | Research hypothesis; direct evidence is sparse. |
| External VSA memory equals parametric learning | False; accessibility and behavioral integration must be measured separately. |

## Research gaps worth targeting

1. Strictly evaluated zero-gradient concept insertion into tied autoregressive LLMs.
2. Ontology-constrained factorization that preserves pretrained embedding neighborhoods.
3. Bidirectional translation between text, canonical graph, VSA memory, and LLM hidden states.
4. Continual updates with contradiction handling, provenance, deletion, and temporal scope.
5. Capacity-aware hierarchical VSA memory with exact graph fallback.
6. Tests of explicit unbinding and compositional generation—not only similarity.
7. JEPA-style relational latent prediction under leakage-resistant graph splits.
