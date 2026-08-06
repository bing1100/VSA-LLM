# VSA embedding toolkit architecture

## Goal

Build a model-agnostic Python library that can attach structured VSA controls to:

1. any `torch.nn.Embedding` or raw embedding matrix;
2. Hugging Face encoder, decoder, or encoder-decoder models;
3. external graph/memory systems; and
4. custom PyTorch modules through small protocols rather than model forks.

Working package name: **`vsa_embed`**.

## Design principles

- Algebra, graph semantics, host-model integration, and evaluation are separate layers.
- Core depends only on PyTorch; Hugging Face support is optional.
- Existing model weights remain immutable by default.
- Input/output tying is explicit and tested.
- Every generated vector is traceable to a composition recipe and source frames.
- Exact graph storage is authoritative; VSA is an index/cache/representation.
- Dynamic updates are versioned and reversible.

## Proposed package

```text
vsa_embed/
├── algebra/
│   ├── base.py               # BindingAlgebra protocol
│   ├── hrr.py                # FFT circular convolution/correlation
│   ├── complex_hrr.py        # unitary/complex variants
│   └── bundle.py             # weighted normalization and capacity estimates
├── graph/
│   ├── schema.py             # Entity, Relation, Frame, Provenance
│   ├── compose.py            # graph -> recipes -> vectors
│   ├── relations.py          # vector/shared-interpreter/operator transforms
│   ├── weights.py            # edge truth, salience, and sparse local gates
│   ├── registry.py           # stable typed atomic IDs
│   └── store.py              # exact graph adapter protocol
├── embeddings/
│   ├── generator.py          # sparse/batched concept generation
│   ├── factorizer.py         # OCVF fitting
│   ├── alignment.py          # linear/orthogonal/MLP projections
│   ├── residual.py           # additive/concat/gated residuals
│   └── overlay.py            # non-destructive row overlays
├── integrations/
│   ├── torch.py              # nn.Embedding wrapper/hooks
│   ├── transformers.py       # tokenizer resize/reserved IDs/tied heads
│   ├── peft.py               # LoRA adapters and consolidation
│   └── serving.py            # export/version metadata
├── memory/
│   ├── associative.py        # cleanup and unbinding
│   ├── sharded.py            # capacity-aware stores
│   ├── temporal.py           # validity/conflict/tombstone handling
│   └── verify.py             # exact frame fallback
├── induction/
│   ├── text_to_frame.py      # pluggable extraction interface
│   ├── entity_linking.py
│   └── meta_generator.py
├── objectives/
│   ├── geometry.py
│   ├── behavioral.py
│   ├── query.py
│   └── jepa.py
├── evaluation/
│   ├── splits.py
│   ├── leakage.py
│   ├── metrics.py
│   └── baselines.py
├── artifacts/                # versioned serialization schemas
└── cli/                      # compose, factorize, attach, evaluate, inspect
```

## Stable public interfaces

```python
class BindingAlgebra(Protocol):
    def bind(self, role: Tensor, filler: Tensor) -> Tensor: ...
    def unbind(self, bound: Tensor, role: Tensor) -> Tensor: ...
    def bundle(self, xs: Tensor, weights: Tensor | None = None) -> Tensor: ...

class ConceptGenerator(Protocol):
    def generate(self, concept_ids: Sequence[str]) -> Tensor: ...
    def explain(self, concept_id: str) -> CompositionTrace: ...

class RelationTransform(Protocol):
    def apply(self, relation_ids: Tensor, concept_vectors: Tensor, context=None) -> Tensor: ...
    def complexity(self) -> dict[str, int | float]: ...

class EdgeWeightModel(Protocol):
    def weights(self, edges, concept_state=None) -> Tensor: ...

class StructuredResidual(Protocol):
    def generate(self, concept_ids, definitions=None, budget: int = 0) -> Tensor: ...

class HostAdapter(Protocol):
    def attach(self, generator: ConceptGenerator) -> AttachmentHandle: ...
    def detach(self, handle: AttachmentHandle) -> None: ...
```

The first production adapter should wrap `nn.Embedding` without mutating it:

```text
effective_row(id) = base_row(id) + gate(id) * aligned_vsa(id)
```

An `EmbeddingOverlay` maps reserved/virtual IDs to generated rows and supplies an optional tied output projection. No architecture-specific BERTHA fork should be required.

## User-facing workflows

```bash
vsa-embed graph import --format wordnet --out graph/
vsa-embed factorize --model Qwen/Qwen2.5-0.5B --graph graph/ --out factors/
vsa-embed attach --model ... --factors factors/ --mode reserved-token
vsa-embed concept add --definition concept.yaml --snapshot memory-v2/
vsa-embed evaluate --experiment zero-gradient --config seed.yaml
vsa-embed inspect --concept quokka --show-recipe --neighbors 20
```

## Experiment-to-module accumulation

| Experiment | Toolkit increment |
|---|---|
| 00 | algebra protocols, vector stores, composition traces, capacity simulator |
| 01a | fixed-path factorizer, alignment losses, pretrained-matrix adapters, leakage-safe split manifests |
| 01b | relation-transform ladder, weighted graph overlays, restricted residuals, bias–capacity evaluator |
| 02 | PyTorch/HF overlays, tied-head handling, token registry, rollback |
| 03 | unbinding, cleanup memory, query objectives/adapters |
| 04 | language-specific residual adapters and multilingual registries |
| 05 | frame schema, extraction/entity-linking interfaces, exact graph adapters |
| 06 | sharded temporal memory, provenance, verifier, tombstones |
| 07 | meta-generator/hypernetwork and uncertainty contract |
| 08 | JEPA objectives/target encoders |
| 09 | PEFT consolidation, locality guards, reversible adapter registry |

## Milestone releases

### v0.1 — algebra and standalone embeddings

Pure PyTorch HRR, graph composer, tests, sparse generation, serialization, capacity benchmark.

### v0.2 — pretrained embedding control

Global–local factorization, raw matrix/`nn.Embedding` adapters, vector/operator relation transforms, weighted graph overlays, restricted residuals, and geometry/adaptation-frontier evaluators.

### v0.3 — Hugging Face attachment

Reserved tokens, tied/untied heads, causal/MLM support, tokenizer registry, behavioral evaluator.

### v0.4 — queryable graph memory

Text/frame ingestion, exact graph fallback, sharded associative memory, provenance and deletion.

### v0.5 — learning extensions

Meta-generator, JEPA objective, multilingual adapters, and LoRA consolidation.

## Quality gates for the library

- CPU/CUDA numerical equivalence and gradient tests.
- Naive-vs-FFT algebra property tests.
- Save/load and version-migration tests.
- No base-weight mutation unless explicitly requested.
- Tied-head invariants for supported Hugging Face models.
- DDP/FSDP/`torch.compile` smoke tests.
- Benchmarks for latency, memory, and sparse-vs-full generation.
- Golden behavioral tests on a tiny synthetic ontology.
- Security tests for malformed schemas, graph poisoning boundaries, and tenant isolation.

## What should remain optional

LLM extraction providers, graph databases, Transformers, PEFT, Triton kernels, and model-serving integrations should be extras. The core algebra/composition package must remain usable with arbitrary embeddings and ordinary PyTorch.
