# t6-legal-regulatory

Train tokens: **100,000,476** (domain 99,633,919, general 366,557; domain fraction 1.00); domain eval tokens: **3,003,524**; general eval tokens: 2,211,513.
Ontology: 7,815 concepts, 5,120 atomics, 5 relations, 17,673 aliases, 7,841 entries.
Holdout (presample): 186 concepts, 189 entries, sha256 `8148fa06fe76929f…`; synthetic zero-shot concepts: 300 (sha256 `6f9243110d1c27ce…`).

**Feasibility verdict: infeasible** (failed: ['rare_spans']). Held-out stratum and locality alone at ℓ_min = 2: powered with 256 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 87 | 1,877 | 16 | 26 | 5 | 6 | 0.105 | 5,452 | no |
| 1 | 512 | 114 | 3,754 | 29 | 44 | 11 | 21 | 0.105 | 5,452 | no |
| 1 | 1024 | 148 | 7,405 | 59 | 99 | 14 | 56 | 0.105 | 5,452 | no |
| 2 | 256 | 59 | 1,314 | 18 | 31 | 5 | 6 | 0.063 | 5,355 | no |
| 2 | 512 | 86 | 2,610 | 31 | 50 | 13 | 24 | 0.063 | 5,355 | no |
| 2 | 1024 | 120 | 5,122 | 65 | 114 | 16 | 59 | 0.063 | 5,355 | no |
| 3 | 256 | 25 | 677 | 12 | 12 | 2 | 2 | 0.037 | 3,396 | no |
| 3 | 512 | 39 | 1,311 | 22 | 25 | 11 | 19 | 0.037 | 3,396 | no |
| 3 | 1024 | 54 | 2,498 | 56 | 75 | 17 | 44 | 0.037 | 3,396 | no |
| 4 | 256 | 9 | 589 | 10 | 28 | 0 | 0 | 0.029 | 2,067 | no |
| 4 | 512 | 16 | 1,126 | 23 | 62 | 7 | 11 | 0.028 | 2,067 | no |
| 4 | 1024 | 27 | 2,162 | 46 | 133 | 14 | 31 | 0.028 | 2,067 | no |

## Span cardinality (domain evaluation sample)

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,841 | 2,071 | 82,976 | 0.104 | 472 | 820 | 779 | 24.9 |
| 2 | 7,415 | 1,602 | 27,576 | 0.062 | 461 | 733 | 408 | 10.4 |
| 3 | 4,611 | 657 | 10,501 | 0.037 | 231 | 301 | 125 | 4.5 |
| 4 | 2,947 | 303 | 6,167 | 0.029 | 116 | 132 | 55 | 2.5 |

### gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,841 | 2,077 | 82,772 | 0.113 | 475 | 823 | 779 | 26.0 |
| 2 | 7,458 | 1,633 | 27,437 | 0.068 | 465 | 752 | 416 | 10.7 |
| 3 | 4,761 | 686 | 10,245 | 0.041 | 238 | 324 | 124 | 4.4 |
| 4 | 3,101 | 311 | 6,678 | 0.032 | 120 | 133 | 58 | 2.8 |

### Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7,841 | 2,072 | 83,085 | 0.106 | 468 | 825 | 779 | 25.3 |
| 2 | 7,459 | 1,601 | 26,764 | 0.063 | 458 | 727 | 416 | 10.0 |
| 3 | 4,651 | 640 | 9,427 | 0.036 | 218 | 302 | 120 | 3.7 |
| 4 | 2,725 | 268 | 6,367 | 0.028 | 91 | 122 | 55 | 2.6 |

## Items

| file | rows | splits |
|---|---:|---|
| `synthetic_concepts.jsonl` | 300 | {'synthetic': 300} |
| `defined_term_cloze.jsonl` | 4,000 | {'heldout': 96, 'train': 1494, 'unlinked': 2410} |
| `eurovoc_probe.jsonl` | 3,000 | {'heldout': 1014, 'train': 1986} |
| `zeroshot_entailment.jsonl` | 2,872 | {'heldout': 1072, 'synthetic': 1800} |
| `zeroshot_property.jsonl` | 4,308 | {'heldout': 1608, 'synthetic': 2700} |
