# t4-chemistry

Train tokens: **100,000,929** (domain 50,518,619, general 49,482,310; domain fraction 0.51); domain eval tokens: **1,971,306**; general eval tokens: 2,211,513.
Ontology: 62,418 concepts, 8,192 atomics, 13 relations, 252,825 aliases, 67,545 entries.
Holdout (presample): 429 concepts, 551 entries, sha256 `b58e504f8b060726…`; synthetic zero-shot concepts: 299 (sha256 `629b185b544e1bdf…`).

**Feasibility verdict: feasible** at ℓ_min = 2 with 256 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 221 | 2,030 | 784 | 1,048 | 287 | 318 | 0.167 | 47,960 | yes |
| 1 | 512 | 287 | 3,939 | 1,492 | 2,018 | 544 | 613 | 0.168 | 47,960 | yes |
| 1 | 1024 | 342 | 8,055 | 2,922 | 4,102 | 1,105 | 1,245 | 0.168 | 47,960 | yes |
| 2 | 256 | 211 | 1,202 | 784 | 1,048 | 287 | 318 | 0.153 | 47,960 | yes |
| 2 | 512 | 275 | 2,369 | 1,493 | 2,019 | 544 | 613 | 0.153 | 47,960 | yes |
| 2 | 1024 | 335 | 4,800 | 2,923 | 4,103 | 1,105 | 1,245 | 0.153 | 47,960 | yes |
| 3 | 256 | 166 | 556 | 790 | 1,053 | 289 | 320 | 0.137 | 47,373 | no |
| 3 | 512 | 220 | 1,125 | 1,505 | 2,035 | 546 | 615 | 0.138 | 47,373 | yes |
| 3 | 1024 | 274 | 2,248 | 2,937 | 4,122 | 1,109 | 1,249 | 0.138 | 47,373 | yes |
| 4 | 256 | 103 | 303 | 746 | 987 | 282 | 314 | 0.117 | 44,359 | no |
| 4 | 512 | 146 | 604 | 1,441 | 1,939 | 538 | 609 | 0.118 | 44,359 | no |
| 4 | 1024 | 184 | 1,178 | 2,811 | 3,909 | 1,103 | 1,245 | 0.118 | 44,359 | yes |

## Span cardinality (domain evaluation sample)

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67,545 | 5,065 | 25,067 | 0.170 | 2,391 | 2,272 | 402 | 8.2 |
| 2 | 67,468 | 4,977 | 17,116 | 0.156 | 2,418 | 2,269 | 290 | 5.8 |
| 3 | 66,809 | 4,479 | 12,075 | 0.141 | 2,361 | 1,947 | 171 | 4.2 |
| 4 | 65,104 | 3,559 | 8,060 | 0.120 | 2,013 | 1,465 | 81 | 2.9 |

### gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67,545 | 5,079 | 25,221 | 0.181 | 2,394 | 2,282 | 403 | 8.2 |
| 2 | 67,498 | 5,015 | 18,456 | 0.169 | 2,402 | 2,286 | 327 | 6.1 |
| 3 | 67,024 | 4,644 | 13,745 | 0.155 | 2,372 | 2,050 | 222 | 4.8 |
| 4 | 65,612 | 3,832 | 9,412 | 0.133 | 2,098 | 1,619 | 115 | 3.3 |

### Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67,545 | 5,068 | 25,389 | 0.175 | 2,394 | 2,270 | 404 | 8.2 |
| 2 | 67,499 | 5,006 | 18,855 | 0.162 | 2,405 | 2,273 | 328 | 6.2 |
| 3 | 66,949 | 4,568 | 13,678 | 0.147 | 2,366 | 1,993 | 209 | 4.7 |
| 4 | 65,343 | 3,704 | 8,837 | 0.124 | 2,045 | 1,554 | 105 | 3.2 |

## Items

| file | rows | splits |
|---|---:|---|
| `synthetic_concepts.jsonl` | 299 | {'synthetic': 299} |
| `class_role_probe.jsonl` | 1,504 | {'heldout': 136, 'synthetic': 299, 'train': 1069} |
| `property_cloze.jsonl` | 3,712 | {'heldout': 1406, 'train': 2306} |
| `zeroshot_entailment.jsonl` | 2,970 | {'heldout': 1176, 'synthetic': 1794} |
| `zeroshot_property.jsonl` | 4,455 | {'heldout': 1764, 'synthetic': 2691} |
