# Experiment 07 — Amortized VSA meta-learner

**Original proposal:** 4. **Depends on:** global–local factorization targets (01b), insertion/readout tests (02–03), and preferably compiler frames (05).

## Question and hypothesis

Can a hypernetwork predict reusable relation representations, sparse edge salience, residual budget, uncertainty, and alignment for a new ontology/concept? It should also predict when a fixed-binding vector is sufficient, when a higher-capacity operator is warranted, and when to abstain rather than silently allocate edge-specific capacity.

## Episode construction

Sample ontology/domain episodes. Support set provides source concepts and host anchors; query set holds nodes, combinations, relation labels, or whole graph components. Hard evaluation holds out an ontology family and renames relation labels to prevent memorization.

## Predicted outputs

- entity/relation atomics or updates to an atomic dictionary;
- relation family/capacity, composition salience, and restricted projection residual;
- uncertainty, abstain decision, and “allocate new atomic” decision;
- optional candidate frame schema from text.

## Baselines

Per-domain OCVF optimization, frozen definition encoder, CoLLEGe-like generator, MAML-style adaptation, nearest anchors, graph neural encoder, and fixed random atomics.

## Metrics

Held-out-ontology insertion/readout behavior, adaptation examples/steps/time, geometry, relation generalization, uncertainty calibration, atomic-allocation precision, and host locality.

## Acceptance gate

Pass if the meta-learner reaches a predeclared fraction (e.g. 95%) of per-domain 01b quality with at least 10× lower adaptation compute and beats definition generation on unseen ontology families. It must select capacity without exceeding a predeclared excess-parameter/overfit rate; relation-renaming performance must remain above baseline.

## Outputs and toolkit increment

`MetaConceptGenerator`, episode sampler, uncertainty/allocation contract, schema-generalization benchmark, and cached adaptation artifacts.

## Risks

Ontology collections may share labels/content and leak templates. Deduplicate entities, use family-level splits, relation renaming, and synthetic novel schemas. Calibration is mandatory before automatic memory writes.
