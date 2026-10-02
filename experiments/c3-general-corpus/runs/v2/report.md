# C3 general corpus (FineWeb-Edu × WordNet)

Train tokens: **1,100,000,492**; eval tokens: **4,945,038**.
Entries: 101,500; held-out entries: 5,900 (holdout sha256 `7f2462ed8f6eae9b…`, 576,226 held-out spans in eval).

## Span cardinality (sample of documents)

### gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 101,500 | 31,143 | 1,027,232 | 0.525 | 8,717 | 13,039 | 9,387 | 191.2 |
| 2 | 90,931 | 24,549 | 101,739 | 0.109 | 10,683 | 12,128 | 1,738 | 23.9 |
| 3 | 55,554 | 8,093 | 22,678 | 0.036 | 4,520 | 3,246 | 327 | 5.3 |
| 4 | 30,812 | 2,115 | 5,863 | 0.012 | 1,326 | 711 | 78 | 1.3 |

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 101,500 | 31,116 | 1,053,544 | 0.528 | 8,695 | 13,061 | 9,360 | 192.2 |
| 2 | 90,371 | 24,543 | 117,304 | 0.119 | 11,350 | 11,773 | 1,420 | 26.7 |
| 3 | 53,755 | 7,501 | 23,076 | 0.036 | 4,457 | 2,750 | 294 | 5.5 |
| 4 | 28,838 | 1,808 | 4,741 | 0.010 | 1,223 | 528 | 57 | 1.2 |

### Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 101,500 | 31,172 | 1,059,789 | 0.544 | 8,726 | 13,062 | 9,384 | 194.5 |
| 2 | 90,588 | 24,433 | 125,499 | 0.130 | 10,629 | 12,016 | 1,788 | 28.5 |
| 3 | 52,135 | 7,527 | 24,006 | 0.038 | 4,239 | 2,988 | 300 | 5.6 |
| 4 | 27,283 | 1,785 | 4,709 | 0.010 | 1,104 | 616 | 65 | 1.1 |
