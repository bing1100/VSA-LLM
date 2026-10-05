# E5.4 zero-shot insertion — c3_synthetic — P0 seed 1 (HuggingFaceTB/SmolLM2-135M/frozen, int4-A)

Run `experiments/e9-retrofit/runs/wordnet/SmolLM2-135M-frozen-P0-s1` (channel `none`); items `experiments/e5-explainability/items/c3-synthetic-smollm2-v1` (400 concepts, 1743 prompt items; contamination-free on pretrained hosts: **True**). Linked concepts: 400 of 400 ({'heldout': 400}).

`own` is the run's own row (composition: the structure-only candidate; C2: the mean-vector fallback; C0: no channel).

## Structure-only sources

| source | property | entailment | paraphrase | semantic_mrr | semantic_r10 | after_loss |
|---|---:|---:|---:|---:|---:|---:|
| own | 0.2102 | 0.4974 | 0.5093 | — | — | — |

Chance: property 0.200, entailment 0.500.

Property/entailment: accuracy averaged over template paraphrases (PMI against the null surface); paraphrase: argmax agreement across paraphrases; semantic rank among all entries' rows; after_loss in nats (lower is better).
