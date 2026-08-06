# Experiment 01 — Ontology-Constrained VSA Factorization

**Original proposal:** OCVF (1). **Depends on:** experiment 00 algebra/API and capacity recommendations.

## Current status — synthetic sandbox passed

The executable bootstrap is in `experiments/01-ontology-factorization/`. Across three seeds, strict composition-disjoint synthesis achieved **0.897 held-out kNN overlap**, versus **0.221** for the strongest control (bag of factors), a **+0.676** gain. Shuffled structure scored 0.070, nearest recipe 0.214, and train mean 0.077. All held-out target rows are excluded from optimization, and tests verify that altering a held-out row cannot alter learned parameters.

This is a **sandbox result only** because the target matrix was produced by a hidden HRR teacher. It validates the implementation and the ability to learn shared atomics; it is not evidence that pretrained LLM rows have this structure. Experiment 02 remains gated on the WordNet + open-weight matrix milestone below.

**Control-design lesson:** values must reuse the same identity space across roles. If each role receives disjoint value IDs, an ostensibly untyped bag baseline can infer role from the ID and is not a valid structure-removal control.

## Experiment 01a: real GPT-2 / WordNet pilot — gate failed

A matrix-only pilot now exists in `experiments/01-ontology-factorization/runs/wordnet-gpt2-pilot/`. It uses 1,200 monosemous single-token noun lemmas, frozen GPT-2 input rows, four-level WordNet ancestor recipes, and three recipe-group-disjoint 960/240 splits with global atomic support.

| Method | Held-out row cosine | Held-out kNN overlap |
|---|---:|---:|
| typed HRR | 0.331 | 0.107 |
| typed MAP | 0.417 | 0.114 |
| **untyped additive** | **0.442** | **0.139** |
| shuffled structure | 0.260 | 0.042 |
| nearest recipe | 0.325 | 0.099 |
| train mean | 0.529 | 0.043 |

The result rejects the present typed-HRR factorization hypothesis: HRR is **−0.032** behind the strongest structural control on neighborhood overlap and **−0.197** behind the mean baseline on raw cosine. Structure still matters—additive composition substantially beats shuffled structure—but role binding is not the source of the gain. Strong mean-direction anisotropy also makes raw cosine misleading.

**Decision:** experiment 02 remains blocked. The immediate successor is [experiment 01b — global–local relational factorization](../01b-global-local-relational-factorization/). It replaces ancestor-depth pseudo-roles with globally shared semantic relations, sparse local edge salience, restricted concept residuals, and a controlled vector-to-operator capacity frontier. Exact recipe/edge groups must never cross a split; earlier diagnostic runs showed that allowing twins materially inflates results.

The 01a finding motivates—not pre-answers—the operator question. HRR/MAP relation vectors induce constrained circulant/diagonal operators. A learned operator relaxes this bias and may fit more realistic relations, but can demand more data and memorize. Experiment 01b therefore compares learning curves, data/compute to threshold, and asymptotic error under matched budgets rather than declaring one parameterization intrinsically superior.

## Question and hypothesis

Can an ontology-composed atomic dictionary reconstruct behaviorally relevant geometry from a pretrained embedding table and synthesize useful held-out nodes? We expect a structured component plus low-rank/gated residual to preserve neighborhoods and host behavior better than pure HRR, while transferring better than unconstrained factorization to node-disjoint concepts.

## Tracks

1. **Matrix-only:** raw embedding rows and WordNet-linked single tokens.
2. **Context anchor:** sense-specific concept anchors estimated from templates/corpora.
3. **Behavior-preserving:** add host-logit/hidden-state distillation on calibration text.

Hold out nodes/components before fitting. The target rows, aliases, and target-specific residuals are forbidden in the strict track.

## Model and data seed

- Qwen2.5-0.5B or comparably small tied causal LM; second model for promotion.
- WordNet 10k–100k concepts, later bounded Wikidata/scientific ontology.
- Existing-token anchors, typed/inverse edges, definition track reported separately.

## Objective ablations

Row cosine/MSE; local-neighborhood ranking; pairwise geometry; relation loss; calibration-logit KL; residual sparsity/rank; orthogonal vs linear vs MLP alignment; real vs unitary HRR.

## Baselines

Unconstrained low-rank matrix factorization, graph embeddings plus projection, random/frozen HRR atomics, definition encoder, nearest anchors, and no-factorization original table.

## Metrics and acceptance gate

Primary: held-out-node behavioral proxy or neighborhood retrieval versus strongest equal-budget baseline. Secondary: kNN overlap, relation retrieval, explained variance, residual energy, calibration KL, and base perplexity.

Pass if structured factorization significantly improves node-disjoint transfer over unstructured factorization/graph projection while base-model calibration perplexity changes by <1% for an overlay reconstruction. Geometry-only reconstruction is not enough for final promotion but can feed experiment 02 as a candidate.

## Outputs and toolkit increment

`OntologyFactorizer`, `AlignmentMap`, structured/residual checkpoint, factorization report, and reusable anchor extractor. Experiment 02 consumes these artifacts.

Implemented artifacts include the differentiable `OntologyFactorizer`, strict-index fitting API, nearest-recipe and structure-destroying controls, immutable GPT-2/WordNet matrix adapter, recipe-group split manifests, geometry metrics, versioned checkpoints/configs, and synthetic/real reports. Further fixed-path scaling is deprecated in favor of experiment 01b's heterogeneous relation and residual-frontier design.

## Risks

Tokenizer rows are not clean semantic concepts; polysemy and frequency dominate. Keep a sense-specific contextual track and interpret high residual energy as useful evidence, not force it away.
