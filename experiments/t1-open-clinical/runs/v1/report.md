# T1-open corpus — open-clinical (MeSH + PubMed)

Ontology: MeSH 2026 descriptors (frames from tree positions, pharmacological actions, see-also); corpus: PubMed 2026 baseline abstracts mixed 50/50 (tokens) with FineWeb-Edu (the C3 stream). Pretrained hosts may have seen PubMed; the newest baseline files were used to limit overlap.

Data: PubMed, courtesy of the U.S. National Library of Medicine (2026 baseline snapshot, not updated); MeSH 2026, U.S. National Library of Medicine.

Linker: 173,258 aliases over 30,915 entries (alias policy {'exclude_classes': ['2'], 'drop_lexical_tags': ['ABB', 'ACR'], 'drop_resolved_inverted': True}).

| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |
|---|---:|---:|---|
| gpt2 | 173,258 | 171,122 | 2,136 / 19,221 / 31,779 / 31,540 / 88,582 |
| HuggingFaceTB/SmolLM2-135M | 173,258 | 170,385 | 2,873 / 24,042 / 35,241 / 30,930 / 80,172 |
| Qwen/Qwen2.5-0.5B | 173,258 | 171,033 | 2,225 / 21,046 / 34,007 / 30,556 / 85,424 |

Holdout: 1,976 entries (702 chosen of 7,023 eligible, 1,274 added by the closure, 1,125 hub entries not eligible); sha256 `1c477afda9822b260596c046fcdd3ca7205d9a5c0e22b44892d0cc1870ddcd88`.

## Corpora

PubMed evaluation pool: PMIDs whose sha256 bucket is < 800 / 10,000 (never trained on); `eval-pubmed` uses all of them. General text: the C3 FineWeb-Edu stream, its first 5,000 documents (C3's evaluation documents) never trained on. Mixing calibration (chars per gpt2 token): PubMed 4.728, general 4.627.

| tokenizer | corpus | tokens | documents | spans | PubMed token share |
|---|---|---:|---:|---:|---:|
| gpt2 | eval | 9,895,896 | 18,205 | 915,674 | 0.500 |
| gpt2 | eval-pubmed | 14,622,326 | 39,252 | 1,490,914 | — |
| gpt2 | eval-general | 4,945,038 | 5,000 | 413,047 | — |
| gpt2 | train | 300,000,302 | 546,794 | 26,232,134 | 0.500 |
| HuggingFaceTB/SmolLM2-360M | eval | 9,983,671 | 18,205 | 903,178 | 0.499 |
| HuggingFaceTB/SmolLM2-360M | eval-pubmed | 14,739,829 | 39,252 | 1,464,979 | — |
| HuggingFaceTB/SmolLM2-360M | eval-general | 4,998,786 | 5,000 | 409,243 | — |
| HuggingFaceTB/SmolLM2-360M | train | 130,000,254 | 233,013 | 11,026,262 | 0.498 |

## Span cardinality (evaluation samples, full alias table)

### pubmed — gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 6,504 | 75,564 | 0.171 | 1,951 | 3,236 | 1,317 | 21.0 |
| 2 | 30,575 | 6,025 | 32,401 | 0.121 | 1,996 | 3,208 | 821 | 9.9 |
| 3 | 27,984 | 4,178 | 16,472 | 0.081 | 1,609 | 2,211 | 358 | 5.1 |
| 4 | 23,793 | 2,599 | 8,234 | 0.051 | 1,141 | 1,316 | 142 | 2.7 |

### pubmed — HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 6,504 | 74,152 | 0.149 | 1,947 | 3,249 | 1,308 | 20.6 |
| 2 | 30,458 | 5,855 | 27,083 | 0.093 | 2,040 | 3,162 | 653 | 8.5 |
| 3 | 27,068 | 3,531 | 11,424 | 0.055 | 1,502 | 1,832 | 197 | 3.7 |
| 4 | 22,006 | 1,959 | 5,329 | 0.032 | 913 | 984 | 62 | 1.8 |

### pubmed — Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 6,509 | 75,303 | 0.165 | 1,950 | 3,242 | 1,317 | 20.9 |
| 2 | 30,556 | 6,021 | 31,566 | 0.113 | 2,003 | 3,225 | 793 | 9.6 |
| 3 | 27,724 | 4,049 | 15,373 | 0.074 | 1,594 | 2,138 | 317 | 4.8 |
| 4 | 23,121 | 2,371 | 7,093 | 0.043 | 1,073 | 1,179 | 119 | 2.3 |

### general — gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 6,178 | 168,668 | 0.103 | 1,761 | 2,555 | 1,862 | 27.8 |
| 2 | 30,575 | 5,847 | 33,712 | 0.040 | 1,977 | 2,977 | 893 | 6.5 |
| 3 | 27,984 | 3,273 | 11,116 | 0.019 | 1,559 | 1,507 | 207 | 2.1 |
| 4 | 23,793 | 1,497 | 4,134 | 0.009 | 856 | 581 | 60 | 0.8 |

### general — HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 6,184 | 167,326 | 0.096 | 1,768 | 2,561 | 1,855 | 27.5 |
| 2 | 30,458 | 5,660 | 28,140 | 0.031 | 2,095 | 2,894 | 671 | 5.6 |
| 3 | 27,068 | 2,623 | 7,209 | 0.012 | 1,442 | 1,081 | 100 | 1.5 |
| 4 | 22,006 | 1,080 | 2,467 | 0.005 | 678 | 376 | 26 | 0.5 |

### general — Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 6,183 | 168,582 | 0.103 | 1,756 | 2,564 | 1,863 | 27.8 |
| 2 | 30,556 | 5,817 | 33,161 | 0.039 | 1,973 | 2,965 | 879 | 6.5 |
| 3 | 27,724 | 3,145 | 10,021 | 0.017 | 1,543 | 1,434 | 168 | 2.0 |
| 4 | 23,121 | 1,328 | 3,677 | 0.008 | 750 | 532 | 46 | 0.7 |

### mixed — gpt2

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 8,127 | 138,321 | 0.139 | 2,266 | 3,842 | 2,019 | 23.3 |
| 2 | 30,575 | 7,680 | 45,507 | 0.082 | 2,449 | 4,045 | 1,186 | 8.7 |
| 3 | 27,984 | 5,107 | 21,042 | 0.051 | 2,007 | 2,632 | 468 | 4.1 |
| 4 | 23,793 | 3,021 | 10,025 | 0.031 | 1,353 | 1,486 | 182 | 2.0 |

### mixed — HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 8,131 | 136,509 | 0.124 | 2,268 | 3,850 | 2,013 | 23.0 |
| 2 | 30,458 | 7,516 | 37,961 | 0.063 | 2,542 | 4,038 | 936 | 7.5 |
| 3 | 27,068 | 4,354 | 14,437 | 0.034 | 1,919 | 2,176 | 259 | 3.0 |
| 4 | 22,006 | 2,292 | 6,295 | 0.019 | 1,108 | 1,105 | 79 | 1.3 |

### mixed — Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 8,132 | 138,058 | 0.135 | 2,263 | 3,854 | 2,015 | 23.3 |
| 2 | 30,556 | 7,680 | 44,403 | 0.078 | 2,455 | 4,080 | 1,145 | 8.5 |
| 3 | 27,724 | 4,956 | 19,529 | 0.046 | 1,968 | 2,582 | 406 | 3.8 |
| 4 | 23,121 | 2,744 | 8,602 | 0.026 | 1,258 | 1,335 | 151 | 1.7 |

## Feasibility per ℓ_min

**Criterion** (on the `eval-pubmed` split, per tokenizer): ≥ 300 held-out entries with ≥ 5 occurrences and ≥ 2,000 held-out span occurrences; ≥ 300 distinct rare entries (training frequency 1–9, the trainer's `after_rare` stratum) linked and ≥ 2,000 rare-entry occurrences.

**Why these numbers.** E4 gate item 1 compares the held-out (and seen-rare) stratum loss between conditions with a paired bootstrap over examples, Holm-corrected over the condition grid. The unit that varies is the span occurrence (its 8 following tokens are strongly correlated), so the occurrence count sets the standard error: with a per-occurrence paired loss difference of SD ≈ 0.5–1 nat, 2,000 occurrences give SE ≈ 0.011–0.022 nat, i.e. a minimum detectable difference of ≈ 0.04–0.08 nat at 80% power after Holm over ≈ 4 comparisons — the size of effect a channel must show to matter. Occurrences cluster by concept, so at least 300 distinct concepts with ≥ 5 occurrences each keep a concept-cluster bootstrap stable and stop a few frequent concepts from carrying the stratum. The same two numbers apply to the rare stratum (gate item 1, second half).

Rare-stratum columns are shown only when the training corpus has its full budget (a slice cannot measure training frequencies). `eval windows needed` = the fewest evenly spread 1,024-token windows (`eval.windows` of the trainer) whose spans meet the measurable criteria; the trainer default is 1,024.

| tokenizer | ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | training frequencies | verdict (whole split) | eval windows needed | windows for whole split |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| gpt2 | 1 | 14,622,326 | 77,278 | 935 | 1,995 | 1,136 | measured | feasible for the held-out stratum only (rare stratum underpowered) | not reached | 14,278 |
| gpt2 | 2 | 14,622,326 | 50,299 | 916 | 1,995 | 1,136 | measured | feasible for the held-out stratum only (rare stratum underpowered) | not reached | 14,278 |
| gpt2 | 3 | 14,622,326 | 25,409 | 659 | 2,320 | 1,361 | measured | feasible | 14,278 | 14,278 |
| gpt2 | 4 | 14,622,326 | 14,570 | 419 | 2,327 | 1,402 | measured | feasible | 14,278 | 14,278 |
| HuggingFaceTB/SmolLM2-360M | 1 | 14,739,829 | 76,915 | 936 | 5,956 | 2,565 | measured | feasible | 6,144 | 14,393 |
| HuggingFaceTB/SmolLM2-360M | 2 | 14,739,829 | 46,325 | 908 | 5,965 | 2,570 | measured | feasible | 6,144 | 14,393 |
| HuggingFaceTB/SmolLM2-360M | 3 | 14,739,829 | 18,705 | 580 | 6,411 | 2,951 | measured | feasible | 6,144 | 14,393 |
| HuggingFaceTB/SmolLM2-360M | 4 | 14,739,829 | 9,645 | 338 | 5,483 | 2,733 | measured | feasible | 12,288 | 14,393 |
