# Wikidata entities (Wikidata + FineWeb-Edu) — corpus build (T8)

Ontology: Wikidata entities of the general relation benchmarks (Entity Inferences / ECBD, LRE relations, BEAR, PopQA, TwoHopFact) and their frames' filler entities; frames from Wikidata truthy statements of a fixed property list (`ontologies/wikidata.py`). Corpus: FineWeb-Edu (the C3 sample-10BT shards) documents that mention a selected name, mixed 50/50 (tokens) with documents that mention none. Data: Wikidata (CC0); FineWeb-Edu (ODC-By 1.0). The hosts' pretraining likely contains FineWeb-Edu, so the text is not new to them; the holdout is the controlled part.

Ontology: 20,000 concepts, 88 relations, 8,192 filler atomics, 144,367 edges (99 empty frames; dropped {'filler_outside_dictionary': 69628, 'over_max_degree': 2139, 'filler_without_label': 2884}). Linker: 29,993 aliases over 20,000 entries.

| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |
|---|---:|---:|---|
| HuggingFaceTB/SmolLM2-360M | 29,993 | 29,986 | 7 / 2,539 / 6,347 / 7,463 / 13,637 |
| Qwen/Qwen3-0.6B-Base | 29,993 | 29,952 | 41 / 3,621 / 7,460 / 7,882 / 10,989 |

## Holdout (M1, ROOD)

1,680 held-out entries (1,645 chosen of 8,217 eligible with ≥ 20 screen occurrences, 35 added by the closure); sha256 `99312697ee24fd8b6d5abb5ff8ee4f8b34ed32622061b8153e78349dfb14291d`. ROOD eligibility: 13,590 entries (excluded: {'frame_filler': 3585, 'exclusion_cost': 2739, 'closure_not_eligible': 86}). Node-disjoint: no held-out entity fills any frame. Alias-disjoint: no held-out name is, or is a whole-word part of, a training alias. Document exclusion: every training document naming a held-out entity under any of its 5,305 Wikidata names is dropped (those documents feed `eval-entities`); the held-out entities' names occur in 127,761 training-side screen documents (an upper bound, ≈ 9.8% of the pool).

**Leakage audit** (the realized training documents re-read): held-out names found 0, training documents in `eval-entities` 0, held-out mentions with the full alias table 0 (129,325 documents) — **passed**. The full-table linker also matches 402 held-out aliases at the start of a longer word (e.g. Javan|ese, Tajik|istan, Matthew 2|4, glaucon|ite; the linker has no right word boundary) and 1 at offsets shifted by a character whose lowercase is longer (the linker lowercases the document and indexes it with the original offsets): neither is a mention, and the training table never links held-out aliases at all.

## Corpora

Evaluation side: documents whose id's sha256 bucket is < 1000 / 10,000, plus the ROOD documents; `eval-entities` keeps the first 5000 evaluation-side and 8000 ROOD documents. General evaluation: C3's first 5,000 documents. Calibration (chars per HuggingFaceTB/SmolLM2-360M token): entities 4.527, general 4.580.

| tokenizer | corpus | tokens | documents | spans | entity-text token share |
|---|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | eval | 10,186,948 | 7,453 | 76,243 | 0.509 |
| HuggingFaceTB/SmolLM2-360M | eval-entities | 28,777,699 | 13,000 | 264,024 | — |
| HuggingFaceTB/SmolLM2-360M | eval-general | 4,998,786 | 5,000 | 27,098 | — |
| HuggingFaceTB/SmolLM2-360M | train | 100,000,285 | 126,809 | 332,677 | 0.506 |
| Qwen/Qwen3-0.6B-Base | eval | 9,983,993 | 7,453 | 76,282 | 0.510 |
| Qwen/Qwen3-0.6B-Base | eval-entities | 28,204,793 | 13,000 | 264,027 | — |
| Qwen/Qwen3-0.6B-Base | eval-general | 4,895,335 | 5,000 | 27,058 | — |
| Qwen/Qwen3-0.6B-Base | train | 100,000,477 | 129,325 | 339,763 | 0.506 |

## Span cardinality (evaluation samples, full alias table)

### entities — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,000 | 5,711 | 39,131 | 0.021 | 2,309 | 2,709 | 693 |
| 2 | 19,996 | 5,632 | 26,664 | 0.018 | 2,362 | 2,771 | 499 |
| 3 | 18,370 | 4,075 | 14,896 | 0.013 | 1,988 | 1,871 | 216 |
| 4 | 14,504 | 2,219 | 6,257 | 0.007 | 1,203 | 944 | 72 |

### entities — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,000 | 5,709 | 39,216 | 0.021 | 2,323 | 2,698 | 688 |
| 2 | 19,975 | 5,616 | 27,754 | 0.018 | 2,367 | 2,714 | 535 |
| 3 | 17,718 | 3,917 | 14,964 | 0.012 | 1,898 | 1,779 | 240 |
| 4 | 13,143 | 2,038 | 5,847 | 0.006 | 1,101 | 870 | 67 |

### general — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,000 | 2,412 | 10,892 | 0.012 | 1,198 | 1,008 | 206 |
| 2 | 19,996 | 2,289 | 7,311 | 0.010 | 1,238 | 943 | 108 |
| 3 | 18,370 | 1,501 | 4,120 | 0.007 | 907 | 551 | 43 |
| 4 | 14,504 | 748 | 1,784 | 0.004 | 491 | 246 | 11 |

### general — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,000 | 2,406 | 10,878 | 0.012 | 1,189 | 1,011 | 206 |
| 2 | 19,975 | 2,278 | 7,403 | 0.010 | 1,203 | 959 | 116 |
| 3 | 17,718 | 1,479 | 4,095 | 0.007 | 878 | 560 | 41 |
| 4 | 13,143 | 702 | 1,652 | 0.004 | 463 | 229 | 10 |

### mixed — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,000 | 4,922 | 31,793 | 0.017 | 2,103 | 2,267 | 552 |
| 2 | 19,996 | 4,828 | 21,249 | 0.014 | 2,157 | 2,295 | 376 |
| 3 | 18,370 | 3,415 | 11,901 | 0.010 | 1,769 | 1,480 | 166 |
| 4 | 14,504 | 1,840 | 4,941 | 0.005 | 1,063 | 721 | 56 |

### mixed — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,000 | 4,913 | 31,786 | 0.017 | 2,100 | 2,264 | 549 |
| 2 | 19,975 | 4,800 | 21,948 | 0.015 | 2,124 | 2,267 | 409 |
| 3 | 17,718 | 3,320 | 11,982 | 0.010 | 1,695 | 1,443 | 182 |
| 4 | 13,143 | 1,699 | 4,566 | 0.005 | 984 | 664 | 51 |

## Feasibility per ℓ_min

Criterion (on `eval-entities`, per tokenizer, as T1-open / T7): ≥ 300 held-out entries with ≥ 5 occurrences and ≥ 2,000 held-out occurrences; ≥ 300 rare entries (training frequency 1–9) and ≥ 2,000 rare occurrences.

| tokenizer | ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | verdict | eval windows needed |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| HuggingFaceTB/SmolLM2-360M | 1 | 28,777,699 | 16,079 | 927 | 28,103 | 5,591 | feasible | 8,192 |
| HuggingFaceTB/SmolLM2-360M | 2 | 28,777,699 | 16,028 | 927 | 28,778 | 5,663 | feasible | 8,192 |
| HuggingFaceTB/SmolLM2-360M | 3 | 28,777,699 | 11,111 | 668 | 23,318 | 4,934 | feasible | 12,288 |
| HuggingFaceTB/SmolLM2-360M | 4 | 28,777,699 | 5,721 | 354 | 12,603 | 3,112 | feasible | 24,576 |
| Qwen/Qwen3-0.6B-Base | 1 | 28,204,793 | 16,192 | 926 | 27,823 | 5,597 | feasible | 8,192 |
| Qwen/Qwen3-0.6B-Base | 2 | 28,204,793 | 16,039 | 925 | 28,199 | 5,666 | feasible | 8,192 |
| Qwen/Qwen3-0.6B-Base | 3 | 28,204,793 | 10,377 | 621 | 22,130 | 4,781 | feasible | 16,384 |
| Qwen/Qwen3-0.6B-Base | 4 | 28,204,793 | 5,141 | 330 | 11,457 | 2,874 | feasible | 24,576 |
