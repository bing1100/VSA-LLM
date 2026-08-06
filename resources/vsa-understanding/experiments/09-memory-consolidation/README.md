# Experiment 09 — Continual VSA-to-weights consolidation

**Original proposal:** 10. **Depends on:** working memory (06), behavioral interface (02–03), and provenance/approval (05). It is last because premature consolidation makes errors hard to undo.

## Question and hypothesis

Can mature, frequently used, well-supported VSA memories be selectively distilled into LoRA/adapters for faster inference while preserving locality, provenance, and rollback? Staged episodic-to-parametric consolidation should beat sequential LoRA on forgetting and beat retrieval-only memory on frequent-query latency.

## Policy

Only frames above support, age, consistency, and usage thresholds are eligible. Create versioned adapter batches with source frame IDs. Distill behavior from memory-assisted teacher outputs plus explicit relation/query losses and a locality replay buffer. Consolidate globally supported relation semantics separately from concept-local residuals: repeated local facts must not silently rewrite a global relation operator. Never delete canonical frames after consolidation.

## Comparisons

Retrieval-only VSA/GraphRAG, sequential LoRA, joint periodic LoRA, ROME/MEMIT/SERAC-class methods where feasible, full fine-tuning, and no update.

## Workloads

Sequential batches with new facts, corrections, contradictions, compositional consequences, rare/frequent query distributions, deletion requests, and domain shifts.

## Metrics

Edit success, paraphrase/multi-hop portability, locality KL, old-memory retention, sequential interference, latency/token, memory footprint, provenance traceability, adapter rollback, and deletion behavior.

## Acceptance gate

Pass if consolidation reduces frequent-query latency/cost versus retrieval-only at matched quality and significantly improves retention/locality over sequential LoRA, with exact adapter rollback and recoverable source provenance. If retrieval remains competitive, do not consolidate.

## Outputs and toolkit increment

`ConsolidationPolicy`, PEFT adapter trainer, locality guard, adapter/frame registry, reversible deployment handle, and continual-learning benchmark.

## Risks

Errors become parametric and deletion becomes difficult. Keep small versioned adapters, source manifests, canary evaluation, and automatic rollback. Do not consolidate disputed or tenant-private knowledge into shared weights.
