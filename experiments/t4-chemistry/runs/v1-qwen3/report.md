# t4-chemistry-qwen3

Train tokens: **100,009,703** (domain 50,786,610, general 49,223,093; domain fraction 0.51); domain eval tokens: **1,972,937**; general eval tokens: 2,170,612.
Ontology: 62,418 concepts, 8,192 atomics, 13 relations, 252,825 aliases, 67,545 entries.
Holdout (frozen names (experiments/t4-chemistry/items/holdout_concepts.txt)): 429 concepts, 551 entries, sha256 `b58e504f8b060726…`; synthetic zero-shot concepts: 299 (sha256 `629b185b544e1bdf…`).

**Feasibility verdict: feasible** at ℓ_min = 2 with 256 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 219 | 2,197 | 876 | 1,158 | 298 | 327 | 0.181 | 47,964 | yes |
| 1 | 512 | 292 | 4,287 | 1,608 | 2,172 | 553 | 612 | 0.176 | 47,964 | yes |
| 1 | 1024 | 343 | 8,315 | 3,003 | 4,256 | 1,122 | 1,267 | 0.173 | 47,964 | yes |
| 2 | 256 | 214 | 1,581 | 877 | 1,159 | 298 | 327 | 0.168 | 47,963 | yes |
| 2 | 512 | 285 | 3,058 | 1,612 | 2,177 | 553 | 612 | 0.164 | 47,963 | yes |
| 2 | 1024 | 337 | 5,875 | 3,006 | 4,260 | 1,122 | 1,267 | 0.161 | 47,963 | yes |
| 3 | 256 | 180 | 911 | 879 | 1,155 | 305 | 334 | 0.153 | 47,474 | no |
| 3 | 512 | 241 | 1,805 | 1,628 | 2,187 | 557 | 616 | 0.149 | 47,474 | yes |
| 3 | 1024 | 286 | 3,438 | 3,030 | 4,283 | 1,133 | 1,278 | 0.146 | 47,474 | yes |
| 4 | 256 | 119 | 387 | 816 | 1,066 | 300 | 330 | 0.128 | 44,627 | no |
| 4 | 512 | 167 | 745 | 1,545 | 2,057 | 554 | 613 | 0.125 | 44,627 | no |
| 4 | 1024 | 196 | 1,427 | 2,867 | 4,002 | 1,123 | 1,269 | 0.123 | 44,627 | yes |

## Span cardinality (domain evaluation sample)

### Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67,545 | 5,068 | 25,389 | 0.175 | 2,394 | 2,270 | 404 | 8.2 |
| 2 | 67,499 | 5,006 | 18,855 | 0.162 | 2,405 | 2,273 | 328 | 6.2 |
| 3 | 66,949 | 4,568 | 13,678 | 0.147 | 2,366 | 1,993 | 209 | 4.7 |
| 4 | 65,343 | 3,704 | 8,837 | 0.124 | 2,045 | 1,554 | 105 | 3.2 |

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67,545 | 5,065 | 25,067 | 0.170 | 2,391 | 2,272 | 402 | 8.2 |
| 2 | 67,468 | 4,977 | 17,116 | 0.156 | 2,418 | 2,269 | 290 | 5.8 |
| 3 | 66,809 | 4,479 | 12,075 | 0.141 | 2,361 | 1,947 | 171 | 4.2 |
| 4 | 65,104 | 3,559 | 8,060 | 0.120 | 2,013 | 1,465 | 81 | 2.9 |

## Items

| file | rows | splits |
|---|---:|---|
| `synthetic_concepts.jsonl` | 299 | {'synthetic': 299} |
| `class_role_probe.jsonl` | 1,503 | {'heldout': 136, 'synthetic': 299, 'train': 1068} |
| `property_cloze.jsonl` | 3,646 | {'heldout': 1406, 'train': 2240} |
| `zeroshot_entailment.jsonl` | 2,970 | {'heldout': 1176, 'synthetic': 1794} |
| `zeroshot_property.jsonl` | 4,455 | {'heldout': 1764, 'synthetic': 2691} |
