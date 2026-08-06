# Experiment 08 — Relational VSA-JEPA

**Original proposal:** 6. **Depends on:** validated capacity and relation targets (00, 01b), readout tasks (03), and graph episodes (05); meta-generator (07) is optional.

## Question and hypothesis

Does predicting masked/future **structured latent states** improve entity-, component-, relation-, and temporal-transfer over token reconstruction or an unconstrained latent JEPA? VSA structure should help only if its relational algebra provides a useful inductive bias.

## Architecture

- Context encoder reads partial text/subgraph/event history.
- Query role specifies missing relation, node, event participant, or future interval.
- EMA target encoder represents hidden evidence as a VSA state.
- Predictor estimates the target latent; variance/covariance and negative-edge losses prevent collapse.
- Unbinding/readout tasks test whether the predicted state preserves structure.

## Leakage controls

Hide target IDs, aliases, edges, and direct definition spans from context. Split connected components/time. Compare a deterministic VSA target, learned target, and identical-dimensional unconstrained target. Detect representation collapse and identity shortcuts.

## Baselines

Causal/masked LM, graph autoencoder/link prediction, contrastive text-graph encoder, non-VSA JEPA, and exact symbolic inference. Compare fixed-binding vectors, constrained operators, and unconstrained latents at matched dimension, parameters, and target-prediction compute so JEPA does not merely reward a higher-capacity target.

## Metrics

Latent prediction alone is diagnostic. Primary: downstream held-out relation/node/future-event behavior after matched pretraining compute. Also unbinding, link prediction, linear probes, few-shot transfer, collapse statistics, compute, and robustness.

## Acceptance gate

Pass if VSA-JEPA significantly exceeds non-VSA JEPA and token/graph objectives on at least two disjoint transfer axes, with matched architecture/compute and no shortcut evidence. Otherwise treat JEPA and VSA as independently useful rather than synergistic.

## Outputs and toolkit increment

`RelationalJEPAObjective`, target encoder/predictor interfaces, EMA utilities, collapse monitors, graph masking/split generators, and latent-readout evaluator.

## Risks

Deterministic graph composition can make prediction trivial; target information must genuinely be absent. JEPA can collapse or learn nuisance IDs; enforce variance and use unseen symbols/components.
