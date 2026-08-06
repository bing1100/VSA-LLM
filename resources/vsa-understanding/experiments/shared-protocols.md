# Shared protocols

## Dataset and split registry

Every dataset adapter must emit canonical concept IDs, aliases, definitions, typed edges, provenance, timestamps, source truth/confidence, and split tags. Split types are: `direct_update`, `edge_group`, `node`, `component`, `relation_instance`, `relation_family`, `ontology_family`, `composition`, `atomic`, and `temporal`. A target may belong to several; results are stratified rather than pooled blindly. Exact relational twins, inverse edges, aliases, and paraphrased relation labels must be grouped where their separation would leak the target.

## Host-model protocol

- Record exact model revision, tokenizer hash, tie-word-embedding setting, dtype, quantization, and chat template.
- Evaluate the untouched model before attaching any VSA component.
- Keep a fixed locality corpus and compare token-level KL, perplexity, and benchmark retention.
- Test input comprehension and output generation separately.

## Baseline registry

Reusable baseline names:

```text
random_norm, surface_mean, definition_mean, definition_encoder,
nearest_anchor, wechsel_focus, concept_generator,
transe, complex, rotate, hole, rgcn,
weighted_additive, low_rank_relation, edge_specific_upper_bound,
vector_rag, graph_rag, lora, exact_graph
```

Each experiment declares applicable entries and uses the same evaluator/config schema.

## Statistical protocol

- Predeclare one primary metric and one locality/safety constraint.
- Bootstrap paired 95% confidence intervals over examples.
- For model comparisons, use paired tests and report effect size.
- Repeat promoted experiments over at least three seeds and two split seeds.
- Report all attempted hyperparameter budgets, not only the winner.
- For relation families, report both equal-parameter/equal-compute comparisons and a capacity frontier.
- Report data, examples, steps, and FLOPs to a fixed target quality; a final endpoint cannot distinguish helpful bias from slow optimization.
- Sweep restricted concept-residual capacity and publish the full rate–distortion curve.

## Leakage audit

1. Search target names/aliases in train data and prompts.
2. Traverse graph paths across split boundaries.
3. Check whether definitions state the answer verbatim.
4. For pretrained-model contamination, use synthetic aliases/private facts or post-cutoff data where possible.
5. Store an allowlist of side information explicitly permitted by each track.
6. Keep all members of an exact edge/recipe equivalence class in one partition.
7. For relation-family holdout, also withhold explicit inverses, aliases, textual paraphrases, and domain-specific renamings.
8. Verify that edge-specific parameters cannot cross node, edge-group, or component boundaries.

## Global–local relation protocol

- A promoted relation representation is globally shared across its training instances; edge-specific operators are diagnostic upper bounds only.
- Store source truth, evidence confidence, learned representational salience, and query-time attention as distinct fields.
- Compare fixed-binding relation vectors, compact vectors with a shared interpreter, and constrained learned operators before allowing domain modulation.
- Diagnose rigidity with learning curves and asymptotic error: more data/steps failing to close a gap that a controlled higher-capacity family closes indicates representational undercapacity.
- Penalize local edge weights and residuals so a model cannot silently become an unconstrained embedding-table factorization.
- Report global, domain, edge, and concept-specific trainable parameters separately.

## Common acceptance semantics

An experiment passes only when its primary metric improves over the strongest applicable baseline with confidence intervals excluding zero **and** its locality/capacity constraint remains within the predeclared bound. A geometry-only gain cannot promote a behavioral component.
