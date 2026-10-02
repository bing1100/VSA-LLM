# t5-enterprise-glossary

Train tokens: **100,011,563** (domain 47,089,290, general 52,922,273; domain fraction 0.47); domain eval tokens: **2,969,121**; general eval tokens: 2,211,513.
Ontology: 4,200 concepts, 3,493 atomics, 23 relations, 4,319 aliases, 4,200 entries.
Holdout (fixed by the track): 360 concepts, 360 entries, sha256 `e7313dcece6d0b78…`; synthetic zero-shot concepts: 200 (sha256 `407388893336b53e…`).

**Feasibility verdict: feasible** at ℓ_min = 2 with 256 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 180 | 1,619 | 280 | 1,201 | 32 | 131 | 0.330 | 3,538 | yes |
| 1 | 512 | 253 | 3,140 | 434 | 2,261 | 56 | 282 | 0.331 | 3,538 | yes |
| 1 | 1024 | 327 | 6,362 | 586 | 4,623 | 81 | 624 | 0.334 | 3,538 | yes |
| 2 | 256 | 180 | 1,619 | 280 | 1,201 | 32 | 131 | 0.330 | 3,538 | yes |
| 2 | 512 | 253 | 3,140 | 434 | 2,261 | 56 | 282 | 0.331 | 3,538 | yes |
| 2 | 1024 | 327 | 6,362 | 586 | 4,623 | 81 | 624 | 0.334 | 3,538 | yes |
| 3 | 256 | 180 | 1,619 | 281 | 1,203 | 32 | 130 | 0.330 | 3,537 | yes |
| 3 | 512 | 253 | 3,140 | 437 | 2,269 | 56 | 281 | 0.331 | 3,537 | yes |
| 3 | 1024 | 327 | 6,360 | 589 | 4,642 | 81 | 622 | 0.333 | 3,537 | yes |
| 4 | 256 | 166 | 1,538 | 268 | 1,136 | 34 | 131 | 0.287 | 3,313 | yes |
| 4 | 512 | 235 | 3,001 | 419 | 2,147 | 58 | 282 | 0.288 | 3,313 | yes |
| 4 | 1024 | 306 | 6,115 | 557 | 4,338 | 84 | 613 | 0.291 | 3,313 | yes |

## Span cardinality (domain evaluation sample)

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,200 | 1,938 | 14,430 | 0.340 | 349 | 1,387 | 202 | 2.7 |
| 2 | 4,200 | 1,938 | 14,430 | 0.340 | 349 | 1,387 | 202 | 2.7 |
| 3 | 4,193 | 1,937 | 14,426 | 0.340 | 349 | 1,386 | 202 | 2.7 |
| 4 | 3,768 | 1,802 | 10,980 | 0.296 | 323 | 1,297 | 182 | 2.2 |

### gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,200 | 1,938 | 14,430 | 0.341 | 349 | 1,387 | 202 | 2.7 |
| 2 | 4,200 | 1,938 | 14,430 | 0.341 | 349 | 1,387 | 202 | 2.7 |
| 3 | 4,194 | 1,936 | 14,424 | 0.340 | 348 | 1,386 | 202 | 2.7 |
| 4 | 3,759 | 1,773 | 10,747 | 0.293 | 324 | 1,271 | 178 | 2.2 |

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
