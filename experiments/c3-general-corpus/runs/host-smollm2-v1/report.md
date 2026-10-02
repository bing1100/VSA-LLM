# Host corpus `smollm2` (C3 linker and holdout under HuggingFaceTB/SmolLM2-135M)

Serves: HuggingFaceTB/SmolLM2-135M, HuggingFaceTB/SmolLM2-360M. Tokenizer sha256 `2225e8fb36485529…`, 49,152 ids → `uint16`; input normalization: None.

Train tokens: **130,000,011** (123,138 documents); eval tokens: **4,998,786** (5,000 documents; C3: 5000).
Held-out entries: 5,900 (holdout sha256 `7f2462ed8f6eae9b…`, 578,722 held-out spans in eval, 0 in train).

## Verification against C3

| Check | Value | Verified against |
|---|---|---|
| holdout_sha256 | `7f2462ed8f6eae9b…` | c3 ontology.pt, c3 summary.json |
| alias_table_sha256 | `c802b817351dc792…` | c3 ontology.pt, c3 summary.json, c3 eval/manifest.json |
| train_alias_table_sha256 | `08a0d3ddda273fa0…` | c3 summary.json, c3 train/manifest.json |
| C3 ontology.pt | identical atomic_names, concept_names, entry_concepts, frames, heldout_entries, relation_names | `f33b9cb6feffe260…` |

## Span cardinality, C3 sample (first C3 `cardinality_docs` documents)

Equal to C3's `cardinality.json` row for this tokenizer: **True**.

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 101,500 | 31,116 | 1,053,544 | 0.528 | 8,695 | 13,061 | 9,360 | 192.2 |
| 2 | 90,371 | 24,543 | 117,304 | 0.119 | 11,350 | 11,773 | 1,420 | 26.7 |
| 3 | 53,755 | 7,501 | 23,076 | 0.036 | 4,457 | 2,750 | 294 | 5.5 |
| 4 | 28,838 | 1,808 | 4,741 | 0.010 | 1,223 | 528 | 57 | 1.2 |

## Span cardinality, full eval corpus (strata by training frequency at ℓ ≥ 2)

| ℓ_min | spans | linked entries | covered-token fraction | held-out entries / spans | unseen | rare 1–9 | mid 10–99 | frequent ≥ 100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,559,946 | 41,467 | 0.530 | 4,364 / 578,722 | 390 / 130,385 | 3,394 / 86,542 | 21,317 / 995,140 | 12,002 / 769,157 |
| 2 | 286,050 | 37,830 | 0.120 | 3,897 / 30,382 | 347 / 497 | 2,898 / 4,721 | 18,853 / 50,706 | 11,835 / 199,744 |
| 3 | 56,330 | 13,780 | 0.036 | 1,195 / 4,526 | 218 / 323 | 1,530 / 2,463 | 7,182 / 16,605 | 3,655 / 32,413 |
| 4 | 11,614 | 3,673 | 0.010 | 267 / 807 | 107 / 145 | 557 / 872 | 1,844 / 3,851 | 898 / 5,939 |

## Span cardinality, full train corpus (strata by training frequency at ℓ ≥ 2)

| ℓ_min | spans | linked entries | covered-token fraction | held-out entries / spans | unseen | rare 1–9 | mid 10–99 | frequent ≥ 100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6,898,568 | 75,051 | 0.112 | 0 / 0 | 0 / 0 | 27,961 / 102,985 | 34,590 / 1,347,638 | 12,500 / 5,447,945 |
| 3 | 1,418,942 | 51,964 | 0.035 | 0 / 0 | 0 / 0 | 18,454 / 57,898 | 23,953 / 452,977 | 9,557 / 908,067 |
| 4 | 295,217 | 23,348 | 0.010 | 0 / 0 | 0 / 0 | 8,513 / 22,645 | 10,103 / 105,871 | 4,732 / 166,701 |
