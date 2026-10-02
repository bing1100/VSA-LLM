# t2-developer-tools

Train tokens: **300,004,714** (domain 150,000,089, general 150,004,625; domain fraction 0.50); domain eval tokens: **6,000,595**; general eval tokens: 4,945,038.
Ontology: 55,412 concepts, 8,192 atomics, 16 relations, 217,493 aliases, 55,227 entries.
Holdout (fixed by the track): 1,078 concepts, 1,078 entries, sha256 `75ee2217be97bd9d…`; synthetic zero-shot concepts: 600 (sha256 `d0e26c8f201b513d…`).

**Feasibility verdict: feasible** at ℓ_min = 2 with 1024 evaluation windows.

## Feasibility (domain evaluation windows; criteria in `feasibility.json`)

| ℓ_min | windows | held-out entries | held-out spans | rare (1–9) entries | rare spans | unseen entries | unseen spans | covered fraction | train entries | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 256 | 218 | 422 | 344 | 594 | 122 | 202 | 0.321 | 44,382 | no |
| 1 | 512 | 380 | 867 | 615 | 1,091 | 225 | 365 | 0.318 | 44,382 | no |
| 1 | 1024 | 575 | 1,866 | 1,026 | 2,115 | 465 | 715 | 0.315 | 44,382 | yes |
| 2 | 256 | 218 | 422 | 344 | 594 | 122 | 202 | 0.321 | 44,382 | no |
| 2 | 512 | 380 | 867 | 615 | 1,091 | 225 | 365 | 0.318 | 44,382 | no |
| 2 | 1024 | 575 | 1,866 | 1,026 | 2,115 | 465 | 715 | 0.315 | 44,382 | yes |
| 3 | 256 | 218 | 422 | 344 | 590 | 122 | 202 | 0.319 | 44,315 | no |
| 3 | 512 | 379 | 858 | 616 | 1,088 | 226 | 366 | 0.316 | 44,315 | no |
| 3 | 1024 | 575 | 1,854 | 1,028 | 2,112 | 464 | 711 | 0.313 | 44,315 | yes |
| 4 | 256 | 216 | 395 | 343 | 588 | 124 | 202 | 0.312 | 43,856 | no |
| 4 | 512 | 377 | 818 | 610 | 1,079 | 229 | 367 | 0.309 | 43,856 | no |
| 4 | 1024 | 572 | 1,795 | 1,016 | 2,091 | 469 | 711 | 0.306 | 43,856 | yes |

## Span cardinality (domain evaluation sample)

### gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55,227 | 6,973 | 36,530 | 0.312 | 2,535 | 3,421 | 1,017 | 9.6 |
| 2 | 55,227 | 6,973 | 36,530 | 0.312 | 2,535 | 3,421 | 1,017 | 9.6 |
| 3 | 55,190 | 6,941 | 35,711 | 0.310 | 2,529 | 3,409 | 1,003 | 9.4 |
| 4 | 54,656 | 6,827 | 33,545 | 0.303 | 2,515 | 3,349 | 963 | 8.9 |

### HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55,227 | 6,973 | 36,531 | 0.378 | 2,534 | 3,422 | 1,017 | 10.2 |
| 2 | 55,227 | 6,967 | 36,502 | 0.378 | 2,531 | 3,421 | 1,015 | 10.2 |
| 3 | 55,187 | 6,927 | 35,657 | 0.375 | 2,525 | 3,400 | 1,002 | 10.0 |
| 4 | 54,427 | 6,798 | 33,643 | 0.367 | 2,503 | 3,328 | 967 | 9.5 |

### Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55,227 | 6,971 | 36,528 | 0.383 | 2,533 | 3,421 | 1,017 | 10.3 |
| 2 | 55,227 | 6,961 | 36,489 | 0.383 | 2,529 | 3,417 | 1,015 | 10.3 |
| 3 | 54,394 | 6,853 | 35,361 | 0.380 | 2,493 | 3,364 | 996 | 10.1 |
| 4 | 52,591 | 6,726 | 32,872 | 0.368 | 2,470 | 3,293 | 963 | 9.4 |

## Items

| file | rows | splits |
|---|---:|---|
| `synthetic_concepts.jsonl` | 600 | {'synthetic': 600} |
| `doc_qa_cloze.jsonl` | 5,752 | {'heldout': 3788, 'train': 1964} |
| `property_probe.jsonl` | 2,578 | {'heldout': 1078, 'synthetic': 600, 'train': 900} |
| `signature_probe.jsonl` | 11,450 | {'heldout': 6382, 'train': 5068} |
| `zeroshot_entailment.jsonl` | 9,028 | {'heldout': 5798, 'synthetic': 3230} |
| `zeroshot_property.jsonl` | 11,589 | {'heldout': 7317, 'synthetic': 4272} |
## Feasibility against WP-T1's bar (whole domain evaluation split)

≥ 300 held-out entries with ≥ 5 occurrences and ≥ 2,000 held-out occurrences; ≥ 300 rare (training frequency 1–9) entries linked and ≥ 2,000 rare occurrences (`t1_open_corpus.feasibility_report`; details in `feasibility_strict.json`).

| ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | verdict | eval windows needed |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 6,000,595 | 10,932 | 677 | 12,410 | 2,903 | feasible | 3,072 |
| 2 | 6,000,595 | 10,932 | 677 | 12,410 | 2,903 | feasible | 3,072 |
| 3 | 6,000,595 | 10,893 | 673 | 12,388 | 2,907 | feasible | 3,072 |
| 4 | 6,000,595 | 10,495 | 660 | 12,206 | 2,839 | feasible | 3,072 |
## Alias normalization: `identifier` vs the default mode

The same 2,000 domain evaluation documents linked with both alias tables (gpt2, ℓ_min = 1; `linking_by_normalization.json`). The default mode turns `_` into a space, so a snake_case alias can never match code.

| mode | linked entries | span occurrences | covered-token fraction | linked concepts | of all concepts |
|---|---:|---:|---:|---:|---:|
| default | 3,754 | 27,077 | 0.212 | 3,793 | 0.068 |
| identifier | 6,973 | 36,530 | 0.312 | 7,096 | 0.128 |
