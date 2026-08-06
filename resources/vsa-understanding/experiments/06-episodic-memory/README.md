# Experiment 06 — Provenance-aware VSA episodic memory

**Original proposal:** 5. **Depends on:** a passing paired sharding/risk-control milestone (00b) and validated frames (05); query adapter (03) improves LLM access. Experiment 00 milestone 2 is not sufficient to start a flat-memory implementation.

## Question and hypothesis

Can hierarchical VSA sketches serve as a compact, fast routing and associative layer for temporal/provenance-rich LLM memory while exact frames guarantee correctness? We expect VSA+verification to reduce retrieval latency or memory versus GraphRAG at matched answer quality in some load regimes.

## Memory design

- Bind entity, relation, value, source, time, confidence, polarity, and frame ID.
- Shard by type/community/time according to experiment 00's capacity model.
- Treat hubs exceeding the supported load envelope as mandatory split/exact-fallback cases.
- Retrieve candidate frame IDs by unbinding/cleanup.
- Verify candidates against the canonical graph before generation.
- Use tombstones and snapshot versions for corrections/deletion.

## Workloads

Sequential personal/agent memory, temporal world updates, scientific claims, contradictory sources, multi-tenant isolation, and high-degree entities. Sweep facts from 1k to 10M where infrastructure permits.

## Baselines

Exact graph indices, dense vector retrieval, BM25, vector RAG, GraphRAG/HippoRAG-like memory, flat VSA without verification, and oracle retrieval.

## Metrics

Candidate recall@k before verification; final answer/temporal accuracy; provenance precision; bytes/fact; write/query p50/p95; capacity/interference curves; deletion/rollback; tenant leakage; calibration/abstention.

## Acceptance gate

Pass only if hybrid VSA memory has a Pareto advantage (quality vs latency or bytes) over exact/vector/GraphRAG baselines in a predeclared workload, **candidate recall meets the 00b risk guarantee in every supported degree band and under graph shift**, and final verified precision and rollback meet requirements. If not, retain VSA only as an experimental embedding representation, not memory infrastructure.

## Outputs and toolkit increment

`ShardedAssociativeMemory`, capacity-aware router, cleanup index, `ExactVerifier`, temporal snapshots, tombstones, memory benchmark harness.

## Risks

Approximate retrieval can omit the truth before verification; maintain conservative top-k and fallback. Superposition can leak similarity across tenants; physically separate stores/keys and test attacks.
