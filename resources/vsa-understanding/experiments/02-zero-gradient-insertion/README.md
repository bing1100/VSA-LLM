# Experiment 02 — Zero-gradient compositional vocabulary insertion

**Original proposal:** 2. **Depends on:** a passing global–local candidate from 01b and attachment invariants from toolkit v0.2. The failed 01a factorizer is not eligible.

## Question and hypothesis

Can a frozen causal LLM comprehend and generate a newly inserted concept token initialized from globally learned relations and locally weighted evidence? Test both pure zero-update composition and whether that initialization sharply reduces the residual data/optimization needed for mastery.

## Strict protocol

1. Reserve token IDs before evaluation or use a versioned embedding overlay.
2. Hold target nodes and aliases out of factorizer/interface training.
3. Synthesize the row from permitted graph components only; run graph+definition as a separate track.
4. Set both input and tied output behavior explicitly.
5. Do not place the definition in the evaluation prompt.
6. Test synthetic aliases/private concepts to control pretraining contamination.
7. After the pure zero-update endpoint, adapt only a predeclared restricted residual over `1, 2, 4, 8, 16` examples; never refit global relations on target concepts.

## Evaluations

- Input: property selection, entailment, contextual use, and paraphrase consistency.
- Output: next-token rank, constrained and free generation, alias lexicalization.
- Composition: novel combinations and distractor relations.
- Locality: token KL/perplexity and standard tasks on unrelated text.
- Calibration: confidence and abstention for under-specified concepts.

## Baselines

Matched random, surface/definition means, contextual definition encoder, nearest-anchor/WECHSEL-FOCUS interpolation, CoLLEGe-like generator, HolE/RotatE/R-GCN projection, and a few-shot row optimized with the same examples (upper reference, not zero-shot).

## Acceptance gate

Co-primary endpoints are pure zero-update behavior and examples/parameters/steps to a fixed mastery threshold. Pass if the candidate beats the strongest zero-update baseline **or** significantly shifts the residual-adaptation curve left versus definition and graph-projection initialization, while output rank improves and unrelated perplexity changes <0.5%. Label the result zero-shot or few-shot precisely. Reproduce on a second model before toolkit promotion.

## Outputs and toolkit increment

Non-destructive `EmbeddingOverlay`, reserved-token registry, tied/untied output adapters, rollback handle, and standardized new-concept evaluator.

## Risks

Success may come from lexical anchors rather than VSA structure; use graph-only and shuffled-relation controls. Added tokens alter segmentation; include virtual span overlays as a secondary track.
