# E5.4 zero-shot insertion — c3_synthetic — C2 seed 1 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-360M-full-C2-s1` (channel `free`); items `experiments/e5-explainability/items/c3-synthetic-smollm2-v1` (400 concepts, 1743 prompt items; contamination-free on pretrained hosts: **True**). Linked concepts: 400 of 400 ({'heldout': 400}).

`own` is the run's own row (composition: the structure-only candidate; C2: the mean-vector fallback; C0: no channel).

## Structure-only sources

| source | property | entailment | paraphrase | semantic_mrr | semantic_r10 | after_loss |
|---|---:|---:|---:|---:|---:|---:|
| own | 0.1911 | 0.4974 | 0.4742 | — | — | — |
| none | 0.1916 | 0.4974 | 0.4721 | — | — | — |
| random | 0.1906 | 0.4942 | 0.4721 | — | — | — |
| mean_row | 0.1927 | 0.4981 | 0.4690 | — | — | — |
| surface_mean | 0.1901 | 0.4961 | 0.4690 | — | — | — |
| graph_projection | 0.1885 | 0.4987 | 0.4700 | — | — | — |

## `own` − baseline (linked concepts; paired bootstrap; Holm within family and test)

| test | baseline | family | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---|---:|---:|---|
| property | none | structure | -0.0005 [-0.0052, +0.0036] | 968 | 1.0000 | no |
| property | random | structure | +0.0005 [-0.0036, +0.0052] | 968 | 1.0000 | no |
| property | mean_row | structure | -0.0015 [-0.0057, +0.0026] | 968 | 1.0000 | no |
| property | surface_mean | structure | +0.0010 [-0.0031, +0.0052] | 968 | 1.0000 | no |
| property | graph_projection | structure | +0.0026 [-0.0015, +0.0067] | 968 | 1.0000 | no |
| entailment | none | structure | +0.0000 [-0.0045, +0.0045] | 775 | 1.0000 | no |
| entailment | random | structure | +0.0032 [-0.0019, +0.0084] | 775 | 1.0000 | no |
| entailment | mean_row | structure | -0.0006 [-0.0045, +0.0032] | 775 | 1.0000 | no |
| entailment | surface_mean | structure | +0.0013 [-0.0039, +0.0071] | 775 | 1.0000 | no |
| entailment | graph_projection | structure | -0.0013 [-0.0058, +0.0032] | 775 | 1.0000 | no |
| paraphrase | none | structure | +0.0021 [-0.0072, +0.0114] | 968 | 1.0000 | no |
| paraphrase | random | structure | +0.0021 [-0.0083, +0.0134] | 968 | 1.0000 | no |
| paraphrase | mean_row | structure | +0.0052 [-0.0031, +0.0145] | 968 | 1.0000 | no |
| paraphrase | surface_mean | structure | +0.0052 [-0.0062, +0.0176] | 968 | 1.0000 | no |
| paraphrase | graph_projection | structure | +0.0041 [-0.0042, +0.0134] | 968 | 1.0000 | no |

Chance: property 0.200, entailment 0.500.

Property/entailment: accuracy averaged over template paraphrases (PMI against the null surface); paraphrase: argmax agreement across paraphrases; semantic rank among all entries' rows; after_loss in nats (lower is better).
