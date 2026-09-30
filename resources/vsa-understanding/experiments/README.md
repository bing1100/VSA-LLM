# Dependency-ordered experiment program

These folders convert the ten proposals into experiment seeds. The order maximizes **beneficial learning**: each stage should produce validated artifacts, interfaces, or negative results that reduce risk for later stages.

## Learning order and dependency graph

| Order | Experiment | Original proposal | Why it comes here | Primary reusable artifact |
|---:|---|---:|---|---|
| 00 | [capacity-and-algebra](00-capacity-and-algebra/) | 9 | Establishes correct algebra, scaling limits, and APIs before model experiments | `VSABackend`, capacity curves |
| 00b | paired sharding and risk control (next milestone of 00) | 9 | Current results reject a universal flat backend; establishes workload-paired operating envelopes before memory use | sharding policy, risk–coverage model |
| 01a | [ontology-factorization](01-ontology-factorization/) | 1 | Completed fixed-path test: real graph signal exists, but typed HRR was dominated by addition | negative-result protocol, leakage-safe matrix pilot |
| 01b | [global-local-relational-factorization](01b-global-local-relational-factorization/) | 1 | Learns global relations, local edge salience, and restricted residuals across a vector-to-operator capacity frontier | `RelationTransform`, weighted graph overlay, residual curves |
| 01c–d | [reconstruction rescue and developmental relation discovery](01c-developmental-relation-discovery/) | 1, 3 | Repairs the split retrieval/reconstruction result before testing sparse latent relation specialization and ontology induction | residual/shared-basis operators, sparse router, relation cards |
| 02 | [zero-gradient-insertion](02-zero-gradient-insertion/) | 2 | Tests pure composition and fast residual adaptation only after a passing 01c/01d candidate | `EmbeddingOverlay`, token registry |
| 03 | [query-and-readout](03-query-and-readout/) | 7 | Determines whether models can use binding/unbinding, not just accept rows | query adapters and cleanup heads |
| 04 | [multilingual-pivot](04-multilingual-pivot/) | 8 | Reuses aligned concept atomics and insertion/readout protocols | language adapters |
| 05 | [text-to-vsa-compiler](05-text-to-vsa-compiler/) | 3 | Converts reading into validated structures using established schemas/algebra | `TextToFrame`, graph writer |
| 06 | [episodic-memory](06-episodic-memory/) | 5 | Requires capacity results and compiler output; tests persistent retrieval | sharded VSA memory + verifier |
| 07 | [amortized-meta-learner](07-amortized-meta-learner/) | 4 | Learns to automate factorization/compiler choices after their targets are known | atomic/gate hypernetwork |
| 08 | [relational-vsa-jepa](08-relational-vsa-jepa/) | 6 | Builds on valid structured targets, readout tests, and leakage controls | latent predictor objective |
| 09 | [memory-consolidation](09-memory-consolidation/) | 10 | Requires a working memory and behavioral interface before distillation | reversible LoRA consolidator |

```text
00 algebra/capacity
 ├─> 00b paired/sharded envelope ─────────────────────────────> 06 episodic memory
 ├─> 01a fixed-path pilot ─> 01b global/local relations ─> 01c operator rescue
 │                                                        └─> 01d relation discovery
 │                                                             └─> 02 insertion ─> 03 query/readout ─> 04 multilingual
 │                                                                                 └─> 07 meta-learner
 └─> 05 text compiler ──────────────────────> 06 episodic memory
                          │                  ├─> 08 VSA-JEPA
                          └──────────────────┴─> 09 consolidation
```

## Current status after experiment 01b Stage B

- **Status correction (2026-09-30):** 01c is complete and **refuted** (fixed global HRR adds nothing over a capacity-fair diagonal operator for one-hop frozen-host transfer), and the 01b retrieval gain does not survive the corrected evaluation protocol; see the errata in the 01b and 01c READMEs. The program continues in [`../../plan-improvement/`](../../plan-improvement/proposal.md). *(Superseded line: "Proceed now: 01c fixed-graph reconstruction rescue; continue 00b and the exact graph/compiler portion of 05 in parallel.")*
- **Recorded split result (superseded, see 01b errata):** 01b Stage B found a seed-consistent HRR retrieval gain, but pure HRR lost reconstruction cosine to identity and relation offset. Direct insertion is not authorized. Under protocol 2 the retrieval gain is +0.0023 (CI [−0.064, +0.069]).
- **01d is replaced** by the M3 developmental dictionary (split statistic, permutation null, routing) of [`../../plan-improvement/formulation.md`](../../plan-improvement/formulation.md) §3, implemented in `src/vsa_embed/developmental.py` and tested first on synthetic teachers (E0.2).
- **Proceed conditionally (historical):** 01d only after a passing 01c operator/target; 02 only after joint reconstruction, discovery, behavior, and locality gates; 03 after a candidate sidecar/overlay.
- **Defer:** broad multilingual expansion (04) until insertion/readout work; VSA-backed memory (06) until 00b.
- **Keep downstream-gated:** 07–09 must not consume the current flat-memory calibrators as if they were promoted artifacts.

The 00b study must replay identical materialized workloads across methods. Its output is a set of supported load/shard envelopes and exact-fallback rules, not a single default algebra. See [`../05-experiments-and-roadmap.md`](../05-experiments-and-roadmap.md) for the revised gates and experiment-by-experiment consequences.

## Common execution contract

Each folder contains:

- `README.md`: hypothesis, dependencies, protocol, baselines, metrics, risks, and acceptance gate.
- `seed.yaml`: implementation-neutral starting configuration. It is a design seed, not guaranteed executable until the toolkit CLI exists.

All future runs should materialize:

```text
runs/<experiment>/<run_id>/
├── resolved_config.yaml
├── manifest.json             # git SHA, package/model/data hashes, hardware
├── metrics.jsonl             # append-only scalar/event records
├── artifacts/                # typed artifacts with schema version
├── predictions.parquet       # per-example outputs
└── report.md                 # generated summary and acceptance decision
```

## Shared scientific rules

1. Freeze and hash splits before training; label the zero-shot axis precisely.
2. Use at least three model/split seeds for a promoted result.
3. Match trainable parameters and compute where possible.
4. Evaluate model behavior, not embedding cosine alone.
5. Include random-composition controls to isolate semantic structure from parameter sharing.
6. Preserve exact graph/evidence records; VSA bundles are lossy caches.
7. A failed acceptance gate is useful. Later stages must not silently assume a failed artifact works.
8. Compare relation representations on equal-parameter/equal-compute budgets **and** over an explicit capacity frontier.
9. Keep source truth/confidence separate from learned edge salience; a geometrically useful weight is not a truth probability.
10. Report residual parameters, examples, steps, and FLOPs needed to reach fixed quality; zero-shot is one point on an adaptation curve.

## Artifact contracts

| Artifact | Required fields |
|---|---|
| `AtomicStore` | schema version, IDs/types, vectors, algebra, dimension, seed, normalization |
| `CompositionGraph` | typed directed frames, source, time, confidence, split labels |
| `WeightedGraphOverlay` | source-edge ID, learned salience, relation transform ID, uncertainty, regularization state, version |
| `RelationTransform` | family, global relation IDs, parameter count/rank, inverse policy, host/graph hashes |
| `ResidualBudgetCurve` | residual dimension, evidence count, steps/FLOPs, quality, locality, uncertainty |
| `AlignmentMap` | source/target dimensions, transform, calibration statistics, host-model hash |
| `EmbeddingOverlay` | token/concept registry, input rows, output rows/tie policy, rollback version |
| `MemorySnapshot` | shards, capacity estimates, frame references, tombstones, provenance |
| `EvaluationBundle` | prompts, target IDs/aliases, contamination policy, per-example results |

## Promotion policy

- **Sandbox:** one seed; verifies code and direction only.
- **Candidate:** ≥3 seeds, all baselines, no known leakage, acceptance gate met.
- **Promoted:** reproduced on a second model or ontology and packaged behind a stable toolkit interface.
- **Deprecated:** fails gate or is dominated by a simpler method; retain report and artifacts for learning.

See [toolkit-architecture.md](toolkit-architecture.md) for the cumulative library design.
