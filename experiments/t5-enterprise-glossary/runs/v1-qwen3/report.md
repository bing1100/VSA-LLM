# t5-enterprise-glossary-qwen3

Train tokens: **100,000,894** (domain 43,907,415, general 56,093,479; domain fraction 0.44); domain eval tokens: **2,759,329**; general eval tokens: 2,170,612.
Ontology: 4,200 concepts, 3,493 atomics, 23 relations, 4,319 aliases, 4,200 entries.
Holdout (fixed by the track): 360 concepts, 360 entries, sha256 `e7313dcece6d0b78…`; synthetic zero-shot concepts: 200 (sha256 `407388893336b53e…`).

**Feasibility verdict: feasible** at ℓ_min = 2 with 256 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 186 | 1,841 | 292 | 1,259 | 38 | 167 | 0.345 | 3,538 | yes |
| 1 | 512 | 255 | 3,410 | 461 | 2,423 | 60 | 317 | 0.346 | 3,538 | yes |
| 1 | 1024 | 326 | 6,864 | 600 | 4,850 | 81 | 676 | 0.347 | 3,538 | yes |
| 2 | 256 | 186 | 1,841 | 292 | 1,259 | 38 | 167 | 0.345 | 3,538 | yes |
| 2 | 512 | 255 | 3,410 | 461 | 2,423 | 60 | 317 | 0.346 | 3,538 | yes |
| 2 | 1024 | 326 | 6,864 | 600 | 4,850 | 81 | 676 | 0.347 | 3,538 | yes |
| 3 | 256 | 186 | 1,841 | 294 | 1,267 | 38 | 166 | 0.345 | 3,538 | yes |
| 3 | 512 | 255 | 3,409 | 463 | 2,433 | 60 | 316 | 0.346 | 3,538 | yes |
| 3 | 1024 | 326 | 6,861 | 602 | 4,869 | 81 | 674 | 0.346 | 3,538 | yes |
| 4 | 256 | 166 | 1,773 | 270 | 1,150 | 34 | 140 | 0.297 | 3,221 | yes |
| 4 | 512 | 228 | 3,258 | 429 | 2,220 | 57 | 287 | 0.296 | 3,221 | yes |
| 4 | 1024 | 294 | 6,542 | 557 | 4,388 | 78 | 614 | 0.296 | 3,221 | yes |

## Span cardinality (domain evaluation sample)

### Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,200 | 1,938 | 14,430 | 0.355 | 349 | 1,387 | 202 | 2.7 |
| 2 | 4,200 | 1,938 | 14,430 | 0.355 | 349 | 1,387 | 202 | 2.7 |
| 3 | 4,188 | 1,935 | 14,408 | 0.355 | 348 | 1,386 | 201 | 2.7 |
| 4 | 3,691 | 1,748 | 10,682 | 0.304 | 313 | 1,258 | 177 | 2.2 |

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,200 | 1,938 | 14,430 | 0.340 | 349 | 1,387 | 202 | 2.7 |
| 2 | 4,200 | 1,938 | 14,430 | 0.340 | 349 | 1,387 | 202 | 2.7 |
| 3 | 4,193 | 1,937 | 14,426 | 0.340 | 349 | 1,386 | 202 | 2.7 |
| 4 | 3,768 | 1,802 | 10,980 | 0.296 | 323 | 1,297 | 182 | 2.2 |

### Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,200 | 1,938 | 14,430 | 0.355 | 349 | 1,387 | 202 | 2.7 |
| 2 | 4,200 | 1,938 | 14,430 | 0.355 | 349 | 1,387 | 202 | 2.7 |
| 3 | 4,188 | 1,935 | 14,408 | 0.355 | 348 | 1,386 | 201 | 2.7 |
| 4 | 3,691 | 1,748 | 10,682 | 0.304 | 313 | 1,258 | 177 | 2.2 |

## Items

| file | rows | splits |
|---|---:|---|
| `synthetic_concepts.jsonl` | 200 | {'synthetic': 200} |
| `glossary_cloze.jsonl` | 5,432 | {'heldout': 2014, 'train': 3418} |
| `term_relation_probe.jsonl` | 4,200 | {'heldout': 360, 'synthetic': 200, 'train': 3640} |
| `zeroshot_entailment.jsonl` | 3,360 | {'heldout': 2160, 'synthetic': 1200} |
| `zeroshot_property.jsonl` | 4,698 | {'heldout': 3018, 'synthetic': 1680} |
