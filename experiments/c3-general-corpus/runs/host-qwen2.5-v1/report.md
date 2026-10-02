# Host corpus `qwen2.5` (C3 linker and holdout under Qwen/Qwen2.5-0.5B)

Serves: Qwen/Qwen2.5-0.5B. Tokenizer sha256 `f884026e05f6dfff…`, 151,665 ids → `uint32`; input normalization: NFC.

Train tokens: **130,000,450** (125,523 documents); eval tokens: **4,895,335** (5,000 documents; C3: 5000).
Held-out entries: 5,900 (holdout sha256 `7f2462ed8f6eae9b…`, 581,929 held-out spans in eval, 0 in train).

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
| 1 | 101,500 | 31,172 | 1,059,789 | 0.544 | 8,726 | 13,062 | 9,384 | 194.5 |
| 2 | 90,588 | 24,433 | 125,499 | 0.130 | 10,629 | 12,016 | 1,788 | 28.5 |
| 3 | 52,135 | 7,527 | 24,006 | 0.038 | 4,239 | 2,988 | 300 | 5.6 |
| 4 | 27,283 | 1,785 | 4,709 | 0.010 | 1,104 | 616 | 65 | 1.1 |

## Span cardinality, full eval corpus (strata by training frequency at ℓ ≥ 2)

| ℓ_min | spans | linked entries | covered-token fraction | held-out entries / spans | unseen | rare 1–9 | mid 10–99 | frequent ≥ 100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,576,278 | 41,521 | 0.546 | 4,367 / 581,929 | 492 / 146,914 | 3,879 / 168,612 | 20,403 / 893,645 | 12,380 / 785,178 |
| 2 | 306,060 | 37,177 | 0.131 | 3,830 / 32,798 | 343 / 492 | 2,932 / 4,696 | 17,811 / 46,871 | 12,261 / 221,203 |
| 3 | 59,267 | 13,341 | 0.038 | 1,121 / 5,035 | 215 / 329 | 1,450 / 2,403 | 6,245 / 15,054 | 4,310 / 36,446 |
| 4 | 11,438 | 3,485 | 0.010 | 246 / 930 | 104 / 155 | 508 / 782 | 1,641 / 3,786 | 986 / 5,785 |

## Span cardinality, full train corpus (strata by training frequency at ℓ ≥ 2)

| ℓ_min | spans | linked entries | covered-token fraction | held-out entries / spans | unseen | rare 1–9 | mid 10–99 | frequent ≥ 100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 7,503,118 | 75,080 | 0.121 | 0 / 0 | 0 / 0 | 28,450 / 105,392 | 33,763 / 1,269,478 | 12,867 / 6,128,248 |
| 3 | 1,488,131 | 48,949 | 0.036 | 0 / 0 | 0 / 0 | 17,554 / 54,694 | 21,611 / 402,699 | 9,784 / 1,030,738 |
| 4 | 279,961 | 20,193 | 0.009 | 0 / 0 | 0 / 0 | 7,965 / 21,585 | 8,208 / 101,266 | 4,020 / 157,110 |
