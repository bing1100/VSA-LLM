# Methodology improvements

## Priority 0: make the current result reproducible

1. Extract parser/composition code into a model-independent package with a typed `ConceptFrame` schema.
2. Pin dependencies, remove absolute paths, document licensed assets and exact preprocessing checksums.
3. Add unit tests for parser ordering, FFT/naive convolution equality, gradients, padding/identity, checkpoint round trips, and tied decoding.
4. Resolve the apparent column-vs-row normalization mismatch in `vsa_utils.py`; preserve a compatibility flag for historical checkpoints.
5. Add deterministic seeds, experiment manifests, confidence intervals, and one command for each reported table.

## Improve the representation

### Preserve role, direction, and scope

- Give every directed relation an explicit inverse or learn a constrained inverse.
- Represent subject/object roles rather than relying only on outgoing adjacency.
- Keep SNOMED role groups and generalize them to event frames.
- Add bound roles for time, location, polarity, modality, confidence, source, and evidence ID.
- Distinguish lexical aliases/definitions from asserted world facts.

### Replace raw sums with calibrated bundling

Use weighted, degree-normalized bundles:

```text
c_i = Normalize(Σ_k w_ik · r_k ⊛ a_k)
```

Weights can reflect relation reliability, information content, recency, provenance, or an attention gate. Report performance both with and without weights to ensure gains are not merely frequency injection.

### Preserve paths without unlimited noise

Compare:

- recursive path binding with depth markers/permutations;
- separate one-hop and multi-hop channels;
- hierarchical community bundles;
- relation-specific subspaces;
- exact graph retrieval followed by local VSA composition.

Avoid placing an entire large graph into one vector. Maintain signal-to-noise estimates and abstain when cleanup confidence is low.

### Upgrade the algebra

Benchmark ordinary HRR against complex/unitary vectors, generalized HRR, MAP/BSC variants, tensor-product approximations, and modern KG embeddings. Unitary relation vectors improve stable inversion. Cleanup memory and explicit negative samples are required for genuine query evaluation.

## Preserve pretrained knowledge

The current BERTHA setup learns atomics from random initialization during pretraining. For open-weight LLMs:

1. fit atomics to pretrained rows (OCVF);
2. retain gated residuals;
3. preserve neighborhoods and output behavior, not only row cosine;
4. calibrate norms/logits for tied heads;
5. use sense-specific nodes and contextual gates for polysemy.

An unconstrained residual is not a methodological failure. It is a necessary channel for syntax, frequency, morphology, pragmatics, and ontology omissions. The scientific question is how much transferable semantics the structured channel explains.

## Improve computational efficiency

### Sparse generation

Generate only unique concept rows used by a batch, plus sampled output rows where the objective permits. Cache frozen compositions and invalidate descendants when atomics change. For full softmax, cache FFT atomics and fuse scatter accumulation.

### Better training systems

- Store composition schedules as CSR/COO sparse structures.
- Replace Python loops with segmented reductions or custom Triton/CUDA kernels if profiling warrants.
- Avoid assigning module parameters during forward; expose generated weights as tensors with explicit state semantics.
- Test `torch.compile`, DDP/FSDP, mixed precision, gradient checkpointing, and quantized embedding overlays.
- Track wall time, peak memory, and recomputation overhead against ordinary embeddings.

## Improve “zero-shot” methodology

1. Publish exact entity, relation, component, atomic, and temporal splits.
2. Remove aliases/definitions from training corpora or state when they are allowed side information.
3. Detect graph paths from source to target that trivialize held-out tests.
4. Include new-token generation, not only representation similarity or downstream classification.
5. Compare equal parameter and equal compute budgets.
6. Separate zero **direct update** from zero **parameter update**.
7. Use synthetic/private concepts to reduce unknown pretraining contamination.

## Improve knowledge-graph construction

Use a dual representation:

- **Canonical graph:** exact typed frames, temporal validity, evidence, versioning, access control.
- **VSA cache:** lossy fast retrieval/composition with references back to canonical frame IDs.

Pipeline:

```text
text → extraction → entity resolution → schema validation
     → contradiction/temporal check → quarantine/approval
     → canonical graph → incremental VSA cache
```

Never “guess the HRR” directly without first producing an inspectable graph/frame. Require confidence calibration and abstention. Support truth maintenance: corrections should retract old frames and rebuild affected shards rather than superpose both claims invisibly.

## Improve meta-learning

- Meta-train across ontologies, not random node splits in one ontology.
- Hide relation labels and graph components to test schema induction.
- Predict uncertainty and whether a new atomic is needed.
- Penalize unnecessary atomics (minimum-description-length pressure).
- Use cycle consistency: text → frame/VSA → recovered frame/text attributes.
- Compare against retaining the source document in RAG; persistence alone is not a win.

## Improve JEPA methodology

Avoid deterministic shortcut targets. The context must lack the target node/edge evidence, and the target encoder must not expose IDs available to the predictor. Use EMA target encoders and anti-collapse regularization. Require downstream relational transfer and explicit query performance, not only latent loss.

Potential objective:

```text
L = Llatent(ẑ,z_target)
  + λunbind Lunbind
  + λvar Lvariance
  + λgraph Lnegative-edge
  + λtext Lcross-modal-alignment
```

Compare against identical latent prediction with unconstrained vectors to isolate VSA contribution.

## Safety, governance, and failure handling

- Bind source, timestamp, jurisdiction, confidence, and tenant to every memory.
- Do not let low-confidence extracted facts become global atomics automatically.
- Treat ontology poisoning as a model-supply-chain risk.
- Maintain append-only audit logs and reversible memory versions.
- Detect collisions and unexpected similarity between unrelated private concepts.
- Provide “forget” operations for exact graph records, VSA shards, cached rows, and consolidated adapters.
- In medical/legal use, ontology structure can be outdated or contested; expose provenance and human review.

## Stronger scientific claims

Prefer:

> “Ontology-compositional parameter sharing improves held-out-node behavior under a no-target-gradient protocol.”

over:

> “The model understands unseen concepts.”

Prefer:

> “The LLM can access a newly written external VSA memory through a trained interface.”

over:

> “The LLM updated its own knowledge.”

Only claim symbolic reasoning after controlled binding/unbinding, depth generalization, distractor resistance, and comparison to non-semantic random compositions.
