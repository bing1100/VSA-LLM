# E5.4 zero-shot insertion — c3_synthetic — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train, int4-A)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-360M-full-C5-s1` (channel `compose`); items `experiments/e5-explainability/items/c3-synthetic-smollm2-v1` (400 concepts, 1743 prompt items; contamination-free on pretrained hosts: **True**). Linked concepts: 400 of 400 ({'heldout': 400}).

`own` is the run's own row (composition: the structure-only candidate; C2: the mean-vector fallback; C0: no channel).

## Structure-only sources

| source | property | entailment | paraphrase | semantic_mrr | semantic_r10 | after_loss |
|---|---:|---:|---:|---:|---:|---:|
| own | 0.1906 | 0.4942 | 0.4700 | — | — | — |
| none | 0.1896 | 0.4923 | 0.4721 | — | — | — |
| random | 0.1875 | 0.4923 | 0.4566 | — | — | — |
| mean_row | 0.1865 | 0.4942 | 0.4659 | — | — | — |
| surface_mean | 0.1932 | 0.4897 | 0.4638 | — | — | — |
| graph_projection | 0.1921 | 0.4968 | 0.4721 | — | — | — |

## `own` − baseline (linked concepts; paired bootstrap; Holm within family and test)

| test | baseline | family | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---|---:|---:|---|
| property | none | structure | +0.0010 [-0.0026, +0.0052] | 968 | 1.0000 | no |
| property | random | structure | +0.0031 [-0.0021, +0.0083] | 968 | 1.0000 | no |
| property | mean_row | structure | +0.0041 [-0.0005, +0.0093] | 968 | 0.4548 | no |
| property | surface_mean | structure | -0.0026 [-0.0088, +0.0036] | 968 | 1.0000 | no |
| property | graph_projection | structure | -0.0015 [-0.0057, +0.0026] | 968 | 1.0000 | no |
| entailment | none | structure | +0.0019 [-0.0026, +0.0065] | 775 | 1.0000 | no |
| entailment | random | structure | +0.0019 [-0.0032, +0.0077] | 775 | 1.0000 | no |
| entailment | mean_row | structure | +0.0000 [-0.0052, +0.0058] | 775 | 1.0000 | no |
| entailment | surface_mean | structure | +0.0045 [-0.0026, +0.0123] | 775 | 1.0000 | no |
| entailment | graph_projection | structure | -0.0026 [-0.0077, +0.0026] | 775 | 1.0000 | no |
| paraphrase | none | structure | -0.0021 [-0.0134, +0.0093] | 968 | 1.0000 | no |
| paraphrase | random | structure | +0.0134 [-0.0010, +0.0269] | 968 | 0.3698 | no |
| paraphrase | mean_row | structure | +0.0041 [-0.0083, +0.0165] | 968 | 1.0000 | no |
| paraphrase | surface_mean | structure | +0.0062 [-0.0103, +0.0238] | 968 | 1.0000 | no |
| paraphrase | graph_projection | structure | -0.0021 [-0.0145, +0.0103] | 968 | 1.0000 | no |

Chance: property 0.200, entailment 0.500.

Property/entailment: accuracy averaged over template paraphrases (PMI against the null surface); paraphrase: argmax agreement across paraphrases; semantic rank among all entries' rows; after_loss in nats (lower is better).
