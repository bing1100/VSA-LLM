# t3-product-catalogues

Train tokens: **100,000,369** (domain 100,000,369, general 0; domain fraction 1.00); domain eval tokens: **3,000,190**; general eval tokens: 2,211,513.
Ontology: 5,895 concepts, 8,192 atomics, 7 relations, 28,894 aliases, 6,151 entries.
Holdout (presample): 182 concepts, 225 entries, sha256 `df7acfde37571d70…`; synthetic zero-shot concepts: 300 (sha256 `3f0e64f04acf60f4…`).

**Feasibility verdict: infeasible** (failed: ['rare_entries', 'rare_spans']). Held-out stratum and locality alone at ℓ_min = 2: powered with 1024 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 109 | 1,276 | 5 | 9 | 1 | 1 | 0.050 | 4,274 | no |
| 1 | 512 | 128 | 2,661 | 12 | 33 | 4 | 4 | 0.051 | 4,274 | no |
| 1 | 1024 | 153 | 5,327 | 38 | 64 | 4 | 5 | 0.050 | 4,274 | no |
| 2 | 256 | 98 | 365 | 6 | 10 | 1 | 1 | 0.030 | 4,273 | no |
| 2 | 512 | 120 | 746 | 13 | 34 | 4 | 4 | 0.031 | 4,273 | no |
| 2 | 1024 | 148 | 1,401 | 38 | 64 | 4 | 5 | 0.031 | 4,273 | no |
| 3 | 256 | 56 | 120 | 11 | 14 | 1 | 1 | 0.013 | 3,822 | no |
| 3 | 512 | 68 | 213 | 23 | 44 | 5 | 6 | 0.013 | 3,822 | no |
| 3 | 1024 | 90 | 397 | 65 | 95 | 5 | 7 | 0.013 | 3,822 | no |
| 4 | 256 | 21 | 31 | 13 | 18 | 1 | 1 | 0.004 | 2,784 | no |
| 4 | 512 | 29 | 54 | 21 | 31 | 3 | 3 | 0.004 | 2,784 | no |
| 4 | 1024 | 44 | 106 | 69 | 84 | 7 | 8 | 0.004 | 2,784 | no |

## Span cardinality (domain evaluation sample)

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,151 | 1,576 | 20,508 | 0.052 | 433 | 729 | 414 | 5.0 |
| 2 | 5,898 | 1,436 | 8,211 | 0.032 | 460 | 754 | 222 | 2.1 |
| 3 | 4,553 | 783 | 2,468 | 0.014 | 349 | 391 | 43 | 0.7 |
| 4 | 3,129 | 318 | 696 | 0.005 | 160 | 154 | 4 | 0.2 |

### gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,151 | 1,576 | 20,700 | 0.055 | 432 | 731 | 413 | 5.0 |
| 2 | 5,940 | 1,443 | 8,161 | 0.034 | 461 | 760 | 222 | 2.1 |
| 3 | 4,676 | 729 | 2,273 | 0.013 | 315 | 377 | 37 | 0.7 |
| 4 | 3,229 | 259 | 570 | 0.004 | 133 | 121 | 5 | 0.2 |

### Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,151 | 1,574 | 20,459 | 0.051 | 433 | 727 | 414 | 5.0 |
| 2 | 5,917 | 1,413 | 7,104 | 0.028 | 485 | 736 | 192 | 1.9 |
| 3 | 4,544 | 596 | 1,651 | 0.009 | 283 | 293 | 20 | 0.5 |
| 4 | 3,058 | 173 | 355 | 0.003 | 92 | 79 | 2 | 0.1 |

## Items

| file | rows | splits |
|---|---:|---|
| `synthetic_concepts.jsonl` | 300 | {'synthetic': 300} |
| `category_probe.jsonl` | 4,864 | {'heldout': 564, 'synthetic': 300, 'train': 4000} |
| `relevance_probe.jsonl` | 12,000 | {'heldout': 982, 'train': 5661, 'unlinked': 5357} |
| `zeroshot_entailment.jsonl` | 2,892 | {'heldout': 1092, 'synthetic': 1800} |
| `zeroshot_property.jsonl` | 4,338 | {'heldout': 1638, 'synthetic': 2700} |
