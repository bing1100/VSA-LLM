# Integration with pretrained open-weight LLMs

## Design principle

Do not assume an HRR composed in arbitrary coordinates is meaningful to a pretrained transformer. The pretrained embedding space has norms, anisotropy, neighborhoods, and tied-output behavior learned jointly with every layer. A VSA component therefore needs **alignment**, **gating**, or an **adapter**.

## Integration patterns

| Pattern | Core weights | Adds tokens? | Generation support | Best use |
|---|---|---:|---:|---|
| Offline row initialization | Frozen/optional tuning | Yes | Yes if output row added/tied | Cheapest proof of concept |
| Reserved-token replacement | Frozen | Uses pre-reserved IDs | Yes | Truly zero-gradient insertion |
| Span soft token | Frozen | No tokenizer change | Indirect | Multiword entities |
| Gated residual VSA embedding | Frozen + small gate/projector | Optional | Yes | Preserve pretrained geometry |
| VSA cross-attention sidecar | Small adapter | No | Yes through hidden states | Dynamic memory/facts |
| LoRA consolidation | LoRA only | Optional | Yes | Frequently used stable knowledge |

## 1. Offline row initialization

Extend the tokenizer and embedding matrix. Construct a VSA vector and map it through `P: R^d_vsa → R^d_model`. For tied models, resize once and assign the same new row to input and LM-head weights. Calibrate row norm and logit scale to existing tokens.

Advantages: minimal architecture changes and compatible with Hugging Face models. Limitation: the tokenizer/model checkpoint changes; some inference engines need re-export. Without alignment, syntactically valid code can still be behaviorally meaningless.

## 2. Reserved-token replacement

Before deployment, reserve `<CONCEPT_0001>...<CONCEPT_N>`. At runtime replace an unused row with a synthesized row and update the concept registry. This is the cleanest test of no-gradient insertion and avoids tokenizer surgery.

Controls:

- random matched-norm row;
- mean of definition subwords;
- contextual definition embedding projected to the table;
- nearest-neighbor weighted row;
- CoLLEGe-like generator;
- graph embedding plus learned projection;
- OCVF composition.

## 3. Span-level concept embeddings

Map a recognized multi-token mention to one VSA soft token or add a residual to each span token. This avoids exploding the vocabulary and handles entities dynamically. It requires an entity linker and careful position/caching logic. Generation can use a lexicalizer that maps concept IDs back to names, but this is not equivalent to the model generating a native new token.

## 4. Gated residual composition

Recommended practical form:

```text
e_i = LayerNorm(e_pretrained_i + α_i P vsa_i)
```

or

```text
e_i = g_i P vsa_i + (1-g_i)e_pretrained_i.
```

Learn `P` and gates while freezing the transformer. New concepts have no pretrained row, so initialize their residual from definition/subword anchors and use uncertainty-aware gates. Relation-specific projections or a small MLP can address mismatch between HRR and LLM coordinates.

## 5. VSA memory sidecar

Keep the tokenizer unchanged. A retriever/unbinder returns concept/fact vectors; a small cross-attention block injects them at selected layers. This is closest to RAG but passes compact latent structures instead of text. Always return provenance IDs so the final answer can cite exact source frames.

Use late layers first to reduce disruption. Train with memory-dropout so the model does not blindly trust the sidecar. Include an abstain/no-memory gate and exact graph verification.

## 6. OCVF fitting recipe

### Align ontology nodes to model anchors

- Single-token terms: use their table rows.
- Multi-token concepts: average is weak; use contextual mention representations, output-distribution matching, or learn a concept anchor across templates.
- Polysemous terms: create sense-specific nodes and infer contextual gates rather than forcing one row.

### Optimize manifold preservation

Row cosine alone is insufficient. Preserve:

- row norm distribution;
- local k-nearest-neighbor rankings;
- pairwise similarities on sampled anchors;
- logits and hidden states on a calibration corpus;
- output probabilities for tied embeddings.

Fit a structured component plus low-rank/sparse residual. Evaluate the explained variance as an ontology-quality diagnostic: relations that consistently require large residuals may be missing, overly broad, or misaligned.

### Synthesize a new node

1. Entity-link definition terms and neighbors.
2. Construct typed relation-target bindings.
3. Apply degree-aware weighted bundling.
4. Project through `P`.
5. Add a lexical anchor residual if allowed by the zero-shot protocol.
6. Match norm/logit statistics.
7. Insert into a reserved row and tied head.
8. Record the exact recipe and sources for rollback.

## Training regimes

### A. Strict zero-gradient insertion

Train OCVF/projector only on source concepts. At evaluation, no parameters are updated for held-out nodes. This is the strongest claim.

### B. Parameter-efficient alignment

Train projector, gates, and LoRA on generic composition tasks, never on target nodes. This tests whether the host model can learn to consume VSA structure.

### C. Few-shot adaptation

Allow a few target examples and compare sample efficiency. Useful, but label it few-shot rather than zero-shot.

### D. Full joint pretraining

Regenerate structured rows during training, as BERTHA does. This is scientifically useful but least relevant to retrofitting existing open-weight LLMs.

## Causal-LM concerns

1. **Tied weights:** a row useful as input may have poor output calibration; test both directions.
2. **Tokenizer segmentation:** adding a concept token changes sequence length and training distribution.
3. **Chat templates:** reserve tokens outside special/control ranges.
4. **Quantization:** update rows before quantization or maintain a higher-precision embedding overlay.
5. **KV caches:** dynamic row changes do not affect already cached occurrences; version memory/checkpoints.
6. **Serving:** vLLM/llama.cpp-style runtimes may need adapter or embedding override support.
7. **Security:** external graphs are an injection surface; validate schemas, sources, and tenant boundaries.

## General language beyond medical codes

Medical codes are unusually favorable: discrete tokens, curated ontology, and explicit mappings. General language requires:

- mention/entity linking and sense disambiguation;
- multiword concepts and morphology;
- events with participants, time, modality, and negation;
- graded/prototype categories rather than only taxonomic edges;
- lexical, episodic, and world knowledge kept distinct;
- uncertain/conflicting facts and source authority;
- canonical concept IDs separate from surface strings.

Start with bounded domains (taxonomy-rich science, tools/APIs, products, law) before unrestricted language.
