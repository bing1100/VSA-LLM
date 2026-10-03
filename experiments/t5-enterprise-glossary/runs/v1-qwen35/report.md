# t5-enterprise-glossary-qwen35

Train tokens: **100,002,971** (domain 45,481,329, general 54,521,642; domain fraction 0.45); domain eval tokens: **2,842,113**; general eval tokens: 2,192,203.
Ontology: 4,200 concepts, 3,493 atomics, 23 relations, 4,319 aliases, 4,200 entries.
Holdout (fixed by the track): 360 concepts, 360 entries, sha256 `e7313dcece6d0b78…`; synthetic zero-shot concepts: 200 (sha256 `407388893336b53e…`).

**Feasibility verdict: feasible** at ℓ_min = 2 with 256 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 186 | 1,687 | 296 | 1,274 | 31 | 142 | 0.326 | 3,538 | yes |
| 1 | 512 | 263 | 3,278 | 457 | 2,424 | 56 | 271 | 0.327 | 3,538 | yes |
| 1 | 1024 | 326 | 6,601 | 592 | 4,782 | 79 | 624 | 0.328 | 3,538 | yes |
| 2 | 256 | 186 | 1,687 | 296 | 1,274 | 31 | 142 | 0.326 | 3,538 | yes |
| 2 | 512 | 263 | 3,278 | 457 | 2,424 | 56 | 271 | 0.327 | 3,538 | yes |
| 2 | 1024 | 326 | 6,601 | 592 | 4,782 | 79 | 624 | 0.328 | 3,538 | yes |
| 3 | 256 | 186 | 1,686 | 301 | 1,282 | 31 | 141 | 0.326 | 3,536 | yes |
| 3 | 512 | 262 | 3,268 | 462 | 2,434 | 56 | 270 | 0.326 | 3,536 | yes |
| 3 | 1024 | 324 | 6,582 | 597 | 4,794 | 79 | 622 | 0.328 | 3,536 | yes |
| 4 | 256 | 164 | 1,598 | 272 | 1,145 | 30 | 129 | 0.276 | 3,202 | yes |
| 4 | 512 | 231 | 3,106 | 424 | 2,178 | 55 | 252 | 0.277 | 3,202 | yes |
| 4 | 1024 | 290 | 6,265 | 550 | 4,301 | 81 | 581 | 0.279 | 3,202 | yes |

## Span cardinality (domain evaluation sample)

### Qwen/Qwen3.5-0.8B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,200 | 1,938 | 14,430 | 0.336 | 349 | 1,387 | 202 | 2.7 |
| 2 | 4,200 | 1,938 | 14,430 | 0.336 | 349 | 1,387 | 202 | 2.7 |
| 3 | 4,177 | 1,931 | 14,387 | 0.336 | 347 | 1,384 | 200 | 2.7 |
| 4 | 3,632 | 1,733 | 10,576 | 0.285 | 312 | 1,248 | 173 | 2.1 |

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

## Items

| file | rows | splits |
|---|---:|---|
| `synthetic_concepts.jsonl` | 200 | {'synthetic': 200} |
| `glossary_cloze.jsonl` | 5,432 | {'heldout': 2014, 'train': 3418} |
| `term_relation_probe.jsonl` | 4,200 | {'heldout': 360, 'synthetic': 200, 'train': 3640} |
| `zeroshot_entailment.jsonl` | 3,360 | {'heldout': 2160, 'synthetic': 1200} |
| `zeroshot_property.jsonl` | 4,698 | {'heldout': 3018, 'synthetic': 1680} |
