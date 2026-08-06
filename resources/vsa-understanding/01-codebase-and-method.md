# Codebase and method understanding

## Repository map

| Path | Role |
|---|---|
| `../vsa-paper.md` | 2024 HRR-BERT paper and reported MIMIC-IV/eICU/ROOD results. |
| `../models-main/medair_models/bertha/vsa_utils.py` | Ontology parser, atomic-to-concept map, FFT HRR generator, and embedding wrapper. |
| `../models-main/medair_models/bertha/modelling_bertha.py` | BERTHA integration and tied MLM decoding. |
| `../models-main/medair_models/bertha/configuration_bertha.py` | VSA, normalization, freezing, and residual configuration. |
| `../bertha_experiments-main/` | Pretraining, fine-tuning, OOD and hyperparameter scripts. |
| `../bert-on-mimic-iv-main/` | Earlier ICD tokenizer/data/BERT experiments, mainly notebooks. |
| `../models-main/medair_models/qwen2/` | Qwen2 fork with SSP encodings for continuous time/numeric values; not yet an ontology VSA language embedding system. |

The snapshots lack complete data-building scripts, pinned modern environments, tests, and usable README instructions. Several scripts contain machine-specific `/root/data` paths. Reproduction therefore requires reconstructing preprocessing and obtaining licensed SNOMED CT mappings.

## Concept representation

For concept `i`, the paper constructs

```text
c_i = Σ_(r,a)∈G_i  r ⊛ a
```

where `a` is an atomic target/word vector, `r` is a relation vector, `⊛` is HRR circular convolution (binding), and summation is bundling. Descriptions are represented as a description-role vector bound to a bundle of word atomics. Optional SNOMED role groups add another binding level.

Circular convolution is efficiently computed in the Fourier domain:

```text
FFT(r ⊛ a) = FFT(r) ⊙ FFT(a)
```

This keeps dimensionality fixed and is differentiable. Approximate unbinding uses circular correlation/inversion; cleanup requires nearest-neighbor or associative-memory lookup.

## End-to-end data flow

1. **Vocabulary and graph parsing.** `VsaDataParser` enumerates description words, SNOMED concepts, relationships, groups, and non-VSA special tokens.
2. **Composition selection.** `generate_icd_to_ids_mapping()` supports semantic pointers, description pointers, or both; group handling can ignore groups, keep group zero, keep only `isA`, or bind group roles.
3. **Vectorized schedule.** `build_av_cv_mapping()` reorganizes the ragged graph by relation and multiplicity into source/destination index lists.
4. **Atomic parameters.** `FastCVGen` creates one trainable matrix `avs` rather than one independent parameter per ICD code.
5. **Fourier composition.** Each forward pass performs one `rfft`, accumulates bound terms into concept rows, and performs `irfft`.
6. **Optional residual.** `use_cls='add'` adds an unconstrained concept row; `'cat'` concatenates structured and unconstrained halves.
7. **Transformer lookup.** `VsaEmbedding.forward()` regenerates the vocabulary matrix when VSA parameters remain trainable, then uses normal `F.embedding` lookup.
8. **Tied prediction.** BERTHA passes the current generated embedding matrix to a functional MLM decoder, so input and output semantics remain tied.

This is an elegant implementation of a hypergraph factorization: many concept rows share a smaller atomic dictionary, and gradients from any observed concept update all other concepts that reuse those atomics.

## What “zero-shot” means here

The ROOD experiment removes 32 ICD codes from pretraining/fine-tuning. An unstructured row for such a code stays near initialization. An HRR row changes indirectly when atomics shared with observed codes are updated. Thus the method supports **zero-direct-update concept transfer**.

It does *not* yet show:

- adding a token after training;
- learning an ontology from free text;
- importing facts into an existing general LLM;
- generating the unseen code in an autoregressive model;
- robust zero-shot relations whose atomic components are themselves unseen.

These distinctions should be explicit in future papers.

## Strengths

- **Compositional parameter sharing:** rare/unseen concepts inherit updates.
- **Explicit provenance:** each vector has a known symbolic recipe.
- **Differentiable and GPU-compatible:** no detached graph preprocessing during learning.
- **Fixed dimensionality:** unlike tensor products, nesting does not grow width.
- **Controllable inductive bias:** semantic structure and frequency/residual channels can be separated.
- **Tied-head compatibility:** particularly important for language generation.
- **Reported evidence:** strong ROOD-unseen gains and improved physician-rated neighbors for rare codes, alongside modest conventional-task gains.

## Limitations and likely technical issues

### Representation

1. **Flat one-hop bags lose topology.** Bundling direct relation-target pairs does not uniquely preserve paths, cardinality, logical restrictions, temporal validity, negation, or evidence.
2. **Direction and inverse semantics are underdeveloped.** Typed outgoing edges alone do not guarantee inverse-query behavior.
3. **Superposition noise and collisions grow with degree.** High-degree concepts can have poorer signal-to-noise ratios.
4. **Unweighted sums conflate evidence.** Common and reliable relations are treated like noisy aliases; vector norm can encode graph degree.
5. **Rare atomic bottleneck.** A held-out concept made mostly from held-out atomics receives little benefit—the paper correctly notes this.
6. **Ontology incompleteness becomes model bias.** HRRBase cannot represent task-relevant residual semantics absent from SNOMED.
7. **Polysemy and context are missing.** One static code vector cannot represent contextual senses.

### Implementation

1. `FastCVGen.forward()` reconstructs the **entire vocabulary matrix on every forward pass**, even when a batch uses few IDs.
2. Python loops and indexed in-place accumulation limit compiler/distributed efficiency.
3. `norm = torch.linalg.norm(cvs, dim=0)` normalizes **columns**, whereas the documentation says each concept vector should have unit L2 norm; row-wise normalization would normally use `dim=1, keepdim=True`. This requires a regression test before changing because historical results may depend on it.
4. Atomic initialization validation uses strict `torch.isclose` checks on global mean/std, which may reject valid checkpoints.
5. Identity handling via index `-1` is clever but brittle for serialization and mapping validation.
6. Assignment to `self.weight` during forward and functional tied decoding deserve tests under DDP, compilation, checkpoint save/load, and gradient accumulation.
7. No unit tests verify convolution, unbinding, parser invariants, padding gradients, generated-row order, or equivalence to a naive implementation.

### Evaluation

- The pure HRR model has much worse MLM accuracy unless frequency or residual information is added.
- The six-patient “entirely unseen” ROOD subset is very small despite striking results.
- Definition words and ontology neighbors may leak semantic information by design; evaluation must distinguish legitimate side information from split leakage.
- t-SNE is qualitative and unstable; use quantitative frequency predictability and semantic-neighborhood metrics.
- General-language claims require autoregressive generation and concept-in-context tests, not medical code classification alone.

## Architectural interpretation

HRR-BERT is best viewed as a **structured embedding generator**, not a complete neuro-symbolic reasoner. The transformer is not explicitly trained to unbind role fillers, execute graph queries, or update a graph from text. The next research phase should make these operations observable and testable, while preserving an unconstrained residual channel for linguistic detail.
