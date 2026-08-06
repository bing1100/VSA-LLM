# Experiment 03 — Bidirectional VSA query and readout

**Original proposal:** 7. **Depends on:** algebra/capacity (00), a validated 01b relation representation, and attachment/evaluation (02).

## Question and hypothesis

Does explicitly training a small interface to bind/unbind improve systematic relation and depth generalization beyond merely injecting structured rows? We hypothesize a frozen host plus projector/LoRA and query losses will execute unseen relation–filler combinations more reliably than standard QA fine-tuning.

## Tasks

1. `composition + role query → filler` with cleanup.
2. `typed frames → composition` and cycle recovery.
3. `text → typed frame/VSA` on controlled examples.
4. Shared-factor detection and analogical transfer.
5. Multi-hop paths with train depths 1–2 and test depths 3–6.
6. Counterfactual role swaps and hard distractors.

## Controls

Same interface with random fixed binding, weighted addition, concatenation, ordinary graph vectors, standard supervised QA, exact graph executor, and structured rows without query loss. Compare vector binding and the promoted operator family at matched capacity. Split by entities, edge groups, compositions, components, and relation families.

## Metrics

Primary: exact filler retrieval on unseen compositions/deeper paths. Also MRR, frame F1, cycle consistency, distractor robustness, calibration, and base-task locality.

## Acceptance gate

Pass if explicit VSA training significantly beats both standard QA and random-binding controls on compositional/depth generalization, not only in-distribution examples. Require interpretable failure curves versus depth/load.

## Outputs and toolkit increment

`QueryAdapter`, cleanup-head interface, query objective, cycle evaluator, and compositional task generator. These make later memory/JEPA vectors behaviorally consumable.

## Risks

The model may learn task templates rather than algebra. Use unseen symbols, relation renaming, and larger combinatorial test sets. Compare to an exact executor to quantify avoidable error.
