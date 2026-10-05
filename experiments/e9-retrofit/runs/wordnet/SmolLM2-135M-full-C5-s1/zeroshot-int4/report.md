# E5.4 zero-shot insertion — c3_synthetic — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train, int4-A)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C5-s1` (channel `compose`); items `experiments/e5-explainability/items/c3-synthetic-smollm2-v1` (400 concepts, 1743 prompt items; contamination-free on pretrained hosts: **True**). Linked concepts: 400 of 400 ({'heldout': 400}).

`own` is the run's own row (composition: the structure-only candidate; C2: the mean-vector fallback; C0: no channel).

## Structure-only sources

| source | property | entailment | paraphrase | semantic_mrr | semantic_r10 | after_loss |
|---|---:|---:|---:|---:|---:|---:|
| own | 0.1875 | 0.4735 | 0.5155 | — | — | — |
| none | 0.1870 | 0.4723 | 0.5155 | — | — | — |
| random | 0.1911 | 0.4774 | 0.5083 | — | — | — |
| mean_row | 0.1870 | 0.4723 | 0.5083 | — | — | — |
| surface_mean | 0.1885 | 0.4723 | 0.5165 | — | — | — |
| graph_projection | 0.1865 | 0.4723 | 0.5083 | — | — | — |

## `own` − baseline (linked concepts; paired bootstrap; Holm within family and test)

| test | baseline | family | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---|---:|---:|---|
| property | none | structure | +0.0005 [-0.0031, +0.0041] | 968 | 1.0000 | no |
| property | random | structure | -0.0036 [-0.0077, +0.0005] | 968 | 0.5697 | no |
| property | mean_row | structure | +0.0005 [-0.0031, +0.0041] | 968 | 1.0000 | no |
| property | surface_mean | structure | -0.0010 [-0.0057, +0.0036] | 968 | 1.0000 | no |
| property | graph_projection | structure | +0.0010 [-0.0026, +0.0052] | 968 | 1.0000 | no |
| entailment | none | structure | +0.0013 [-0.0032, +0.0058] | 775 | 1.0000 | no |
| entailment | random | structure | -0.0039 [-0.0090, +0.0013] | 775 | 0.7696 | no |
| entailment | mean_row | structure | +0.0013 [-0.0039, +0.0065] | 775 | 1.0000 | no |
| entailment | surface_mean | structure | +0.0013 [-0.0039, +0.0065] | 775 | 1.0000 | no |
| entailment | graph_projection | structure | +0.0013 [-0.0039, +0.0065] | 775 | 1.0000 | no |
| paraphrase | none | structure | +0.0000 [-0.0093, +0.0103] | 968 | 1.0000 | no |
| paraphrase | random | structure | +0.0072 [-0.0021, +0.0176] | 968 | 0.9395 | no |
| paraphrase | mean_row | structure | +0.0072 [-0.0031, +0.0186] | 968 | 0.9395 | no |
| paraphrase | surface_mean | structure | -0.0010 [-0.0124, +0.0114] | 968 | 1.0000 | no |
| paraphrase | graph_projection | structure | +0.0072 [-0.0031, +0.0186] | 968 | 0.9395 | no |

Chance: property 0.200, entailment 0.500.

Property/entailment: accuracy averaged over template paraphrases (PMI against the null surface); paraphrase: argmax agreement across paraphrases; semantic rank among all entries' rows; after_loss in nats (lower is better).
