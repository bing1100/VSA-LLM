# T1-open corpus — open-clinical (MeSH + PubMed)

Ontology: MeSH 2026 descriptors (frames from tree positions, pharmacological actions, see-also); corpus: PubMed 2026 baseline abstracts mixed 50/50 (tokens) with FineWeb-Edu (the C3 stream). Pretrained hosts may have seen PubMed; the newest baseline files were used to limit overlap.

Data: PubMed, courtesy of the U.S. National Library of Medicine (2026 baseline snapshot, not updated); MeSH 2026, U.S. National Library of Medicine.

Linker: 173,258 aliases over 30,915 entries (alias policy {'exclude_classes': ['2'], 'drop_lexical_tags': ['ABB', 'ACR'], 'drop_resolved_inverted': True}).

| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |
|---|---:|---:|---|
| gpt2 | 173,258 | 171,122 | 2,136 / 19,221 / 31,779 / 31,540 / 88,582 |
| HuggingFaceTB/SmolLM2-135M | 173,258 | 170,385 | 2,873 / 24,042 / 35,241 / 30,930 / 80,172 |
| Qwen/Qwen2.5-0.5B | 173,258 | 171,033 | 2,225 / 21,046 / 34,007 / 30,556 / 85,424 |

Holdout: 1,964 entries (702 chosen of 7,023 eligible, 1,262 added by the closure, 1,125 hub entries not eligible); sha256 `fd4244eae86a54d13bfaffa69b95848e1a7a002072d6ee048344af88c2a1cf29`.

## Corpora

PubMed evaluation pool: PMIDs whose sha256 bucket is < 800 / 10,000 (never trained on); `eval-pubmed` uses 15000 of them. General text: the C3 FineWeb-Edu stream, its first 5,000 documents (C3's evaluation documents) never trained on. Mixing calibration (chars per gpt2 token): PubMed 4.728, general 4.631.

| tokenizer | corpus | tokens | documents | spans | PubMed token share |
|---|---|---:|---:|---:|---:|
| gpt2 | eval | 1,500,003 | 2,747 | 138,369 | 0.500 |
| gpt2 | eval-pubmed | 5,616,075 | 15,000 | 568,802 | — |
| gpt2 | eval-general | 995,926 | 1,000 | 83,686 | — |
| gpt2 | train | 3,000,995 | 5,527 | 266,182 | 0.500 |
| HuggingFaceTB/SmolLM2-360M | eval | 566,768 | 1,053 | 51,690 | 0.502 |
| HuggingFaceTB/SmolLM2-360M | eval-pubmed | 752,566 | 2,000 | 74,152 | — |
| HuggingFaceTB/SmolLM2-360M | eval-general | 282,354 | 300 | 23,229 | — |
| HuggingFaceTB/SmolLM2-360M | train | 1,500,005 | 2,754 | 130,179 | 0.499 |

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
| 1 | 30,915 | 4,602 | 83,686 | 0.105 | 1,416 | 1,933 | 1,253 | 27.6 |
| 2 | 30,575 | 4,094 | 17,176 | 0.042 | 1,656 | 2,034 | 404 | 6.6 |
| 3 | 27,984 | 2,034 | 5,749 | 0.020 | 1,106 | 830 | 98 | 2.2 |
| 4 | 23,793 | 866 | 2,171 | 0.010 | 539 | 299 | 28 | 0.8 |

### general — HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 4,606 | 83,136 | 0.098 | 1,421 | 1,938 | 1,247 | 27.4 |
| 2 | 30,458 | 3,838 | 14,236 | 0.033 | 1,710 | 1,839 | 289 | 5.6 |
| 3 | 27,068 | 1,560 | 3,759 | 0.013 | 942 | 566 | 52 | 1.5 |
| 4 | 22,006 | 600 | 1,194 | 0.005 | 401 | 186 | 13 | 0.5 |

### general — Qwen/Qwen2.5-0.5B

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30,915 | 4,606 | 83,659 | 0.105 | 1,410 | 1,947 | 1,249 | 27.7 |
| 2 | 30,556 | 4,067 | 16,751 | 0.040 | 1,652 | 2,029 | 386 | 6.5 |
| 3 | 27,724 | 1,931 | 5,149 | 0.018 | 1,064 | 791 | 76 | 2.0 |
| 4 | 23,121 | 754 | 1,798 | 0.008 | 466 | 266 | 22 | 0.7 |

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
| gpt2 | 1 | 5,616,075 | 27,757 | 671 | — | — | slice (3,000,995 of 300,000,000 tokens) | held-out stratum feasible; rare stratum pending the full build | 2,048 | 5,483 |
| gpt2 | 2 | 5,616,075 | 18,386 | 645 | — | — | slice (3,000,995 of 300,000,000 tokens) | held-out stratum feasible; rare stratum pending the full build | 2,048 | 5,483 |
| gpt2 | 3 | 5,616,075 | 9,503 | 431 | — | — | slice (3,000,995 of 300,000,000 tokens) | held-out stratum feasible; rare stratum pending the full build | 3,072 | 5,483 |
| gpt2 | 4 | 5,616,075 | 5,691 | 284 | — | — | slice (3,000,995 of 300,000,000 tokens) | infeasible (held-out stratum) | not reached | 5,483 |
| HuggingFaceTB/SmolLM2-360M | 1 | 752,566 | 3,702 | 172 | — | — | slice (1,500,005 of 130,000,000 tokens) | infeasible (held-out stratum) | not reached | 733 |
| HuggingFaceTB/SmolLM2-360M | 2 | 752,566 | 2,238 | 132 | — | — | slice (1,500,005 of 130,000,000 tokens) | infeasible (held-out stratum) | not reached | 733 |
| HuggingFaceTB/SmolLM2-360M | 3 | 752,566 | 911 | 58 | — | — | slice (1,500,005 of 130,000,000 tokens) | infeasible (held-out stratum) | not reached | 733 |
| HuggingFaceTB/SmolLM2-360M | 4 | 752,566 | 518 | 24 | — | — | slice (1,500,005 of 130,000,000 tokens) | infeasible (held-out stratum) | not reached | 733 |
