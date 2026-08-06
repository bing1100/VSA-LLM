# VSA–LLM research dossier

**Search cutoff:** 2026-07-17. **Status:** research synthesis; 2025–2026 items are often preprints and should be rechecked before publication.

This directory explains the HRR/BERTHA code in `../`, reviews adjacent research, and proposes a program for ontology-compositional embeddings in open-weight LLMs.

## Bottom line

The existing work demonstrates a real and useful effect: a token never updated by gradient descent can still acquire a useful representation because its vector is recomputed from atomic components shared with observed concepts. That is **structural transfer**, not ordinary prompt-based zero-shot learning.

The most promising next step is not to replace every pretrained token embedding with random HRR atomics. It is to **distill an open-weight LLM's existing embedding geometry into ontology-constrained atomics**, retain a gated residual for information the ontology cannot explain, and use the learned atomics to synthesize embeddings for new concepts. This combines:

1. the pretrained model's linguistic geometry;
2. explicit, editable ontology structure;
3. algebraic construction of unseen concepts; and
4. compatibility with tied input/output embeddings.

We call this proposal **Ontology-Constrained VSA Factorization (OCVF)**. Its strongest defensible novelty is the combination of *inverse VSA fitting to a pretrained embedding table* and *forward zero-shot synthesis from the recovered atomic dictionary*. Definition-based concept embedding, lexical vocabulary initialization, knowledge-graph embedding, and VSA composition each exist separately.

## Recommended first experiment

Use a 0.5–3B parameter Qwen/Llama/Gemma model, WordNet or a Wikidata subset, and 1,000 reserved concept tokens.

1. Freeze the transformer.
2. Fit HRR atomics and a small projection so reconstructed embeddings match existing entity/word rows and their local neighborhoods.
3. Hold out complete ontology nodes during fitting.
4. Build their rows only from relation, neighbor, and definition atomics.
5. Compare against random, mean-of-subwords, definition-encoder, WECHSEL/FOCUS-like lexical initialization, graph embeddings, and CoLLEGe-style generated embeddings.
6. Test both comprehension and generation, with strict no-occurrence and temporal/entity-disjoint splits.

A positive result requires more than cosine similarity: the frozen model must use and generate the new concept correctly in paraphrases, relation queries, and multi-hop tasks without harming old-token likelihood.

## Documents

- [01-codebase-and-method.md](01-codebase-and-method.md) — repository map, equations, gradient flow, limitations.
- [02-literature-review.md](02-literature-review.md) — VSA, vocabulary expansion, editing/memory, graphs, meta-learning, and JEPA.
- [03-research-proposals.md](03-research-proposals.md) — proposals with novelty, impact, feasibility, and falsification criteria.
- [04-open-weight-llm-integration.md](04-open-weight-llm-integration.md) — practical integration patterns for causal open-weight models.
- [05-experiments-and-roadmap.md](05-experiments-and-roadmap.md) — datasets, splits, baselines, metrics, ablations, and milestones.
- [06-methodology-improvements.md](06-methodology-improvements.md) — concrete improvements to the current HRR method.
- [references.md](references.md) — curated references and search notes.
- [experiments/README.md](experiments/README.md) — all ten proposals reordered by beneficial-learning dependencies, with one seed folder per experiment.
- [experiments/toolkit-architecture.md](experiments/toolkit-architecture.md) — cumulative `vsa_embed` architecture for arbitrary PyTorch embeddings and Hugging Face models.

## From experiments to a toolkit

The expanded [experiment program](experiments/README.md) starts with algebra/capacity, then builds factorization, frozen-model overlays, query/readout, graph induction, persistent memory, meta-learning, JEPA, and consolidation. Each successful stage must contribute a typed, tested component to the model-agnostic [`vsa_embed` toolkit](experiments/toolkit-architecture.md), rather than creating a new model fork.

## Claims discipline

- **Implemented:** directly supported by this repository or cited work.
- **Supported hypothesis:** combines established components but has not been demonstrated here.
- **Research proposal:** novel design requiring experiment.
- “Zero-shot” always states what is held out: token occurrences, entity nodes, relations, ontology components, or gradient updates.
