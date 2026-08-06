# Experiment 05 — Text-to-VSA compiler and self-growing graph

**Original proposal:** 3. **Depends on:** schema/algebra from 00; behavioral readout from 03 is preferred.

## Question and hypothesis

Can an LLM convert reading material into typed, evidence-backed frames that safely update a canonical graph and VSA cache, improving later questions about genuinely new concepts? A validated frame-first pipeline should be more accurate, auditable, and reversible than directly asking an LLM to “guess a vector.”

## Pipeline under test

```text
document → candidate entities/frames → entity resolution → schema validation
         → contradiction/temporal checks → quarantine/approval
         → exact graph commit → VSA cache update → behavioral query
```

Frames include subject, relation, object/value, polarity, modality, valid time, truth/evidence confidence, source, and evidence span. The exact graph remains authoritative. A separate learned overlay may attach host-specific representational salience, but salience must never overwrite evidence confidence or be interpreted as truth probability.

## Datasets and episodes

- Synthetic documents with complete known truth and controlled contradictions.
- Time-sliced scientific/news or bounded domain documents.
- New private/synthetic entities to prevent pretraining recall.
- Sequential corrections, aliases, and deliberate malicious statements.

## Baselines

LLM direct answer, vector RAG, direct triple extraction + exact graph, GraphRAG, direct text-to-vector encoder, and unvalidated LLM-to-VSA composition.

## Metrics

Frame/entity F1, schema validity, evidence entailment, contradiction/temporal accuracy, graph precision/recall, downstream QA/multi-hop score, provenance precision, rollback success, and write latency.

## Acceptance gate

Pass if the compiler+graph improves downstream persistent QA over prompt-only/direct-vector methods and matches or exceeds ordinary graph extraction quality, with ≥99% exact rollback and every accepted fact linked to evidence. VSA must add measurable latency/memory/query benefit over exact GraphRAG to claim more than graph construction.

## Outputs and toolkit increment

`Frame`/`Provenance` schemas, `TextToFrame` provider protocol, candidate relation labels/weights with evidence, entity linker, schema validator, quarantine/approval API, graph-store adapter, and incremental composition invalidation. Compiler output supplies candidates to 01b graph refinement; 01b salience remains a versioned derived overlay.

## Risks and safeguards

Extraction hallucinations can become durable. Default to quarantine, calibrated confidence, source trust policies, tenant isolation, and explicit abstention. Never overwrite conflicting facts; version valid-time claims.
