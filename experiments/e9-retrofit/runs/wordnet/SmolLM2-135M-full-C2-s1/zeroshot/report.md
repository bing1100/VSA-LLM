# E5.4 zero-shot insertion — c3_synthetic — C2 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-full-C2-s1` (channel `free`); items `experiments/e5-explainability/items/c3-synthetic-smollm2-v1` (400 concepts, 1743 prompt items; contamination-free on pretrained hosts: **True**). Linked concepts: 400 of 400 ({'heldout': 400}).

`own` is the run's own row (composition: the structure-only candidate; C2: the mean-vector fallback; C0: no channel).

## Structure-only sources

| source | property | entailment | paraphrase | semantic_mrr | semantic_r10 | after_loss |
|---|---:|---:|---:|---:|---:|---:|
| own | 0.1865 | 0.4368 | 0.4432 | — | — | — |
| none | 0.1885 | 0.4368 | 0.4401 | — | — | — |
| random | 0.1885 | 0.4374 | 0.4432 | — | — | — |
| mean_row | 0.1890 | 0.4381 | 0.4370 | — | — | — |
| surface_mean | 0.1865 | 0.4400 | 0.4360 | — | — | — |
| graph_projection | 0.1896 | 0.4406 | 0.4432 | — | — | — |

## `own` − baseline (linked concepts; paired bootstrap; Holm within family and test)

| test | baseline | family | difference [95% CI] | n | p (Holm) | own better |
|---|---|---|---|---:|---:|---|
| property | none | structure | -0.0021 [-0.0062, +0.0021] | 968 | 1.0000 | no |
| property | random | structure | -0.0021 [-0.0062, +0.0021] | 968 | 1.0000 | no |
| property | mean_row | structure | -0.0026 [-0.0062, +0.0010] | 968 | 0.8846 | no |
| property | surface_mean | structure | +0.0000 [-0.0046, +0.0046] | 968 | 1.0000 | no |
| property | graph_projection | structure | -0.0031 [-0.0072, +0.0010] | 968 | 0.8846 | no |
| entailment | none | structure | +0.0000 [-0.0045, +0.0045] | 775 | 1.0000 | no |
| entailment | random | structure | -0.0006 [-0.0058, +0.0045] | 775 | 1.0000 | no |
| entailment | mean_row | structure | -0.0013 [-0.0058, +0.0032] | 775 | 1.0000 | no |
| entailment | surface_mean | structure | -0.0032 [-0.0077, +0.0006] | 775 | 0.7196 | no |
| entailment | graph_projection | structure | -0.0039 [-0.0077, -0.0006] | 775 | 0.1649 | no |
| paraphrase | none | structure | +0.0031 [-0.0062, +0.0124] | 968 | 1.0000 | no |
| paraphrase | random | structure | +0.0000 [-0.0083, +0.0093] | 968 | 1.0000 | no |
| paraphrase | mean_row | structure | +0.0062 [-0.0031, +0.0155] | 968 | 0.9595 | no |
| paraphrase | surface_mean | structure | +0.0072 [-0.0031, +0.0165] | 968 | 0.9395 | no |
| paraphrase | graph_projection | structure | +0.0000 [-0.0083, +0.0083] | 968 | 1.0000 | no |

Chance: property 0.200, entailment 0.500.

Property/entailment: accuracy averaged over template paraphrases (PMI against the null surface); paraphrase: argmax agreement across paraphrases; semantic rank among all entries' rows; after_loss in nats (lower is better).
