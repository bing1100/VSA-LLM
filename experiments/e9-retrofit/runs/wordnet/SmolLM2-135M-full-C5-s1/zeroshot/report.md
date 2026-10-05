# E5.4 zero-shot insertion — c3_synthetic — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C5-s1` (channel `compose`); items `experiments/e5-explainability/items/c3-synthetic-smollm2-v1` (400 concepts, 1743 prompt items; contamination-free on pretrained hosts: **True**). Linked concepts: 400 of 400 ({'heldout': 400}).

`own` is the run's own row (composition: the structure-only candidate; C2: the mean-vector fallback; C0: no channel).

## Structure-only sources

| source | property | entailment | paraphrase | semantic_mrr | semantic_r10 | after_loss |
|---|---:|---:|---:|---:|---:|---:|
| own | 0.1890 | 0.4400 | 0.4463 | — | — | — |
| none | 0.1901 | 0.4413 | 0.4483 | — | — | — |
| random | 0.1885 | 0.4387 | 0.4463 | — | — | — |
| mean_row | 0.1890 | 0.4374 | 0.4463 | — | — | — |
| surface_mean | 0.1885 | 0.4368 | 0.4421 | — | — | — |
| graph_projection | 0.1890 | 0.4400 | 0.4432 | — | — | — |

## `own` − baseline (linked concepts; paired bootstrap; Holm within family and test)

| test | baseline | family | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---|---:|---:|---|
| property | none | structure | -0.0010 [-0.0052, +0.0036] | 968 | 1.0000 | no |
| property | random | structure | +0.0005 [-0.0052, +0.0057] | 968 | 1.0000 | no |
| property | mean_row | structure | +0.0000 [-0.0046, +0.0046] | 968 | 1.0000 | no |
| property | surface_mean | structure | +0.0005 [-0.0046, +0.0057] | 968 | 1.0000 | no |
| property | graph_projection | structure | +0.0000 [-0.0036, +0.0041] | 968 | 1.0000 | no |
| entailment | none | structure | -0.0013 [-0.0052, +0.0026] | 775 | 1.0000 | no |
| entailment | random | structure | +0.0013 [-0.0039, +0.0065] | 775 | 1.0000 | no |
| entailment | mean_row | structure | +0.0026 [-0.0020, +0.0077] | 775 | 1.0000 | no |
| entailment | surface_mean | structure | +0.0032 [-0.0013, +0.0077] | 775 | 0.9895 | no |
| entailment | graph_projection | structure | +0.0000 [-0.0058, +0.0052] | 775 | 1.0000 | no |
| paraphrase | none | structure | -0.0021 [-0.0135, +0.0103] | 968 | 1.0000 | no |
| paraphrase | random | structure | +0.0000 [-0.0124, +0.0124] | 968 | 1.0000 | no |
| paraphrase | mean_row | structure | +0.0000 [-0.0114, +0.0114] | 968 | 1.0000 | no |
| paraphrase | surface_mean | structure | +0.0041 [-0.0062, +0.0145] | 968 | 1.0000 | no |
| paraphrase | graph_projection | structure | +0.0031 [-0.0072, +0.0134] | 968 | 1.0000 | no |

Chance: property 0.200, entailment 0.500.

Property/entailment: accuracy averaged over template paraphrases (PMI against the null surface); paraphrase: argmax agreement across paraphrases; semantic rank among all entries' rows; after_loss in nats (lower is better).
