# Experiment 04 — Multilingual ontology pivot

**Original proposal:** 8. **Depends on:** concept factors/insertion (01–02); query adapter (03) is preferred.

## Question and hypothesis

Can language-independent concept atomics plus small language-specific lexical residuals initialize low-resource vocabulary and transfer relation behavior? The shared ontology channel should help when surface overlap is weak, while residuals preserve morphology and syntax.

## Design

Use aligned concept IDs from WordNet/Wikidata/BabelNet-style resources. Represent each surface form as shared concept VSA + language adapter. Hold out one language's concept rows or an entire low-resource language during adapter training. Evaluate graph-only and graph+parallel-lexicon tracks.

## Comparisons

WECHSEL, FOCUS, zero-shot cross-lingual alignment, subword averaging, multilingual baseline model, definition translation, and shared concept vectors without VSA relations.

## Metrics

Cross-lingual concept retrieval, new-token cloze/generation, relation QA, lexical choice, morphology-sensitive tasks, and high-resource retention. Primary: held-out-language concept behavior.

## Acceptance gate

Pass if ontology VSA improves low-resource behavior over the strongest lexical initialization at equal adapter parameters without degrading high-resource behavior >1%. Demonstrate gains in low lexical-overlap bins to support the structural explanation.

## Outputs and toolkit increment

Multilingual concept/surface registry, `LanguageResidualAdapter`, cross-lingual evaluator, and language-independent composition artifacts.

## Risks

Ontology mappings and translations are noisy; audit by confidence. A multilingual pretrained model may already solve the task, masking contribution; include weaker and monolingual hosts if feasible.
