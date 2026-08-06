# Experiment 00 — Algebra correctness and capacity frontier

**Original proposal:** hierarchical capacity-aware memory (9). **Role:** foundation for every later experiment.

## Question and hypothesis

Which VSA algebra, normalization, dimension, and sharding policy preserve recoverable typed structure at realistic graph degrees and depths? We hypothesize that unitary/complex HRR plus weighted degree normalization and community sharding will dominate the current flat real-HRR sum at equal memory.

## Inputs and dependencies

- No learned host model is required.
- Synthetic typed graphs with controlled degree, path depth, noise, contradictions, and temporal versions.
- Optional samples of WordNet/SNOMED/Wikidata degree distributions.

## Stages

1. Implement naive and FFT bind/unbind; verify forward and gradient equivalence.
2. Compare real HRR, unitary/complex HRR, generalized HRR, MAP/BSC, and random non-semantic bindings.
3. Sweep dimension, bundle size, degree skew, depth, correlated atomics, noise, and dtype.
4. Test nearest-neighbor cleanup, learned cleanup, and exact-index fallback.
5. Compare flat, typed, community, and temporal shards at equal bytes.
6. Fit empirical capacity/confidence models that predict when to abstain.

## Baselines and metrics

Baselines: exact sparse graph, dense vector index, flat HRR, random projection sketch. Metrics: filler top-1/MRR, false retrieval, cosine margin, path-query accuracy, bytes/fact, write/query latency, degradation curves, and confidence calibration.

## Acceptance gate

Promote one default backend only if it reaches ≥95% one-hop recovery at the predeclared target load, calibrated abstention error ≤5%, and demonstrates a memory/latency advantage over exact retrieval in at least one relevant regime. Otherwise expose multiple backends and document the operating envelope.

## Outputs and toolkit increment

- `BindingAlgebra`, `AtomicStore`, `CompositionTrace`, cleanup API.
- Property/gradient tests and capacity benchmark suite.
- `capacity_model.json` used by experiments 05–09 for sharding and abstention.

## Main risks

Synthetic isotropic atomics may overestimate real correlated embeddings; include correlation sweeps. Do not select dimension after seeing only favorable graph degrees.

## Implementation status

An executable implementation now lives at [`../../../../experiments/00-capacity-and-algebra/`](../../../../experiments/00-capacity-and-algebra/) with the reusable package under [`../../../../src/vsa_embed/`](../../../../src/vsa_embed/). It covers real/unitary HRR, MAP control, exact cleanup retrieval, controlled correlated atomics, fixed/Poisson/heavy-tailed degrees, held-out confidence calibration, and core correctness tests. In the 39,360-observation milestone-2 run, 29/36 conditions met ECE ≤5%, but only 17/36 transferred ≥95% selective accuracy to held-out seeds. Heavy-tailed degree was the major failure regime, so no universal backend is promoted. Generalized HRR, path depth, sharding, and learned cleanup remain deferred rather than silently ignored.
