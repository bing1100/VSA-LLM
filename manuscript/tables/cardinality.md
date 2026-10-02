# Span cardinality by ℓ_min, tokenizer and corpus

Linked-span statistics on each corpus's evaluation sample (formulation §4.1). `linked` = distinct ontology entries linked; `covered` = fraction of tokens inside linked spans; `/1k` = mean distinct linked entries per 1,024-token window; `≥10` = entries seen at least 10 times in the sample.

## C3 general — FineWeb-Edu × WordNet 3.0

Source `experiments/c3-general-corpus/runs/v2/cardinality.json`.

| Tokenizer | ℓ_min | Linkable | Linked | Spans | Covered | /1k | ≥10 | Sample tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 BPE | 1 | 101,500 | 31,143 | 1,027,232 | 0.525 | 191.2 | 9,387 | 2,041,081 |
| GPT-2 BPE | 2 | 90,931 | 24,549 | 101,739 | 0.109 | 23.9 | 1,738 | 2,041,081 |
| GPT-2 BPE | 3 | 55,554 | 8,093 | 22,678 | 0.036 | 5.3 | 327 | 2,041,081 |
| GPT-2 BPE | 4 | 30,812 | 2,115 | 5,863 | 0.012 | 1.3 | 78 | 2,041,081 |
| SmolLM2 | 1 | 101,500 | 31,116 | 1,053,544 | 0.528 | 192.2 | 9,360 | 2,062,417 |
| SmolLM2 | 2 | 90,371 | 24,543 | 117,304 | 0.119 | 26.7 | 1,420 | 2,062,417 |
| SmolLM2 | 3 | 53,755 | 7,501 | 23,076 | 0.036 | 5.5 | 294 | 2,062,417 |
| SmolLM2 | 4 | 28,838 | 1,808 | 4,741 | 0.010 | 1.2 | 57 | 2,062,417 |
| Qwen2.5 | 1 | 101,500 | 31,172 | 1,059,789 | 0.544 | 194.5 | 9,384 | 2,020,153 |
| Qwen2.5 | 2 | 90,588 | 24,433 | 125,499 | 0.130 | 28.5 | 1,788 | 2,020,153 |
| Qwen2.5 | 3 | 52,135 | 7,527 | 24,006 | 0.038 | 5.6 | 300 | 2,020,153 |
| Qwen2.5 | 4 | 27,283 | 1,785 | 4,709 | 0.010 | 1.1 | 65 | 2,020,153 |

## T1-open clinical — PubMed 2026 × MeSH 2026

Source `experiments/t1-open-clinical/runs/v1/cardinality_by_source.json`.

| Tokenizer | ℓ_min | Linkable | Linked | Spans | Covered | /1k | ≥10 | Sample tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 BPE | 1 | 30,915 | 6,504 | 75,564 | 0.171 | 21.0 | 1,317 | 747,533 |
| GPT-2 BPE | 2 | 30,575 | 6,025 | 32,401 | 0.121 | 9.9 | 821 | 747,533 |
| GPT-2 BPE | 3 | 27,984 | 4,178 | 16,472 | 0.081 | 5.1 | 358 | 747,533 |
| GPT-2 BPE | 4 | 23,793 | 2,599 | 8,234 | 0.051 | 2.7 | 142 | 747,533 |
| SmolLM2 | 1 | 30,915 | 6,504 | 74,152 | 0.149 | 20.6 | 1,308 | 750,566 |
| SmolLM2 | 2 | 30,458 | 5,855 | 27,083 | 0.093 | 8.5 | 653 | 750,566 |
| SmolLM2 | 3 | 27,068 | 3,531 | 11,424 | 0.055 | 3.7 | 197 | 750,566 |
| SmolLM2 | 4 | 22,006 | 1,959 | 5,329 | 0.032 | 1.8 | 62 | 750,566 |
| Qwen2.5 | 1 | 30,915 | 6,509 | 75,303 | 0.165 | 20.9 | 1,317 | 749,517 |
| Qwen2.5 | 2 | 30,556 | 6,021 | 31,566 | 0.113 | 9.6 | 793 | 749,517 |
| Qwen2.5 | 3 | 27,724 | 4,049 | 15,373 | 0.074 | 4.8 | 317 | 749,517 |
| Qwen2.5 | 4 | 23,121 | 2,371 | 7,093 | 0.043 | 2.3 | 119 | 749,517 |

## T3 product — ESCI products × Google/Shopify taxonomy

Source `experiments/t3-product-catalogues/runs/v1/cardinality.json`.

| Tokenizer | ℓ_min | Linkable | Linked | Spans | Covered | /1k | ≥10 | Sample tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 BPE | 1 | 6,151 | 1,576 | 20,700 | 0.055 | 5.0 | 413 | 560,959 |
| GPT-2 BPE | 2 | 5,940 | 1,443 | 8,161 | 0.034 | 2.1 | 222 | 560,959 |
| GPT-2 BPE | 3 | 4,676 | 729 | 2,273 | 0.013 | 0.7 | 37 | 560,959 |
| GPT-2 BPE | 4 | 3,229 | 259 | 570 | 0.004 | 0.2 | 5 | 560,959 |
| SmolLM2 | 1 | 6,151 | 1,576 | 20,508 | 0.052 | 5.0 | 414 | 599,661 |
| SmolLM2 | 2 | 5,898 | 1,436 | 8,211 | 0.032 | 2.1 | 222 | 599,661 |
| SmolLM2 | 3 | 4,553 | 783 | 2,468 | 0.014 | 0.7 | 43 | 599,661 |
| SmolLM2 | 4 | 3,129 | 318 | 696 | 0.005 | 0.2 | 4 | 599,661 |
| Qwen2.5 | 1 | 6,151 | 1,574 | 20,459 | 0.051 | 5.0 | 414 | 561,341 |
| Qwen2.5 | 2 | 5,917 | 1,413 | 7,104 | 0.028 | 1.9 | 192 | 561,341 |
| Qwen2.5 | 3 | 4,544 | 596 | 1,651 | 0.009 | 0.5 | 20 | 561,341 |
| Qwen2.5 | 4 | 3,058 | 173 | 355 | 0.003 | 0.1 | 2 | 561,341 |

## T4 chemistry — ChEBI text × ChEBI 255

Source `experiments/t4-chemistry/runs/v1/cardinality.json`.

| Tokenizer | ℓ_min | Linkable | Linked | Spans | Covered | /1k | ≥10 | Sample tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 BPE | 1 | 67,545 | 5,079 | 25,221 | 0.181 | 8.2 | 403 | 442,407 |
| GPT-2 BPE | 2 | 67,498 | 5,015 | 18,456 | 0.169 | 6.1 | 327 | 442,407 |
| GPT-2 BPE | 3 | 67,024 | 4,644 | 13,745 | 0.155 | 4.8 | 222 | 442,407 |
| GPT-2 BPE | 4 | 65,612 | 3,832 | 9,412 | 0.133 | 3.3 | 115 | 442,407 |
| SmolLM2 | 1 | 67,545 | 5,065 | 25,067 | 0.170 | 8.2 | 402 | 437,165 |
| SmolLM2 | 2 | 67,468 | 4,977 | 17,116 | 0.156 | 5.8 | 290 | 437,165 |
| SmolLM2 | 3 | 66,809 | 4,479 | 12,075 | 0.141 | 4.2 | 171 | 437,165 |
| SmolLM2 | 4 | 65,104 | 3,559 | 8,060 | 0.120 | 2.9 | 81 | 437,165 |
| Qwen2.5 | 1 | 67,545 | 5,068 | 25,389 | 0.175 | 8.2 | 404 | 436,934 |
| Qwen2.5 | 2 | 67,499 | 5,006 | 18,855 | 0.162 | 6.2 | 328 | 436,934 |
| Qwen2.5 | 3 | 66,949 | 4,568 | 13,678 | 0.147 | 4.7 | 209 | 436,934 |
| Qwen2.5 | 4 | 65,343 | 3,704 | 8,837 | 0.124 | 3.2 | 105 | 436,934 |

## T5 glossary — synthetic enterprise documents × glossary

Source `experiments/t5-enterprise-glossary/runs/v1/cardinality.json`.

| Tokenizer | ℓ_min | Linkable | Linked | Spans | Covered | /1k | ≥10 | Sample tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 BPE | 1 | 4,200 | 1,938 | 14,430 | 0.341 | 2.7 | 202 | 232,601 |
| GPT-2 BPE | 2 | 4,200 | 1,938 | 14,430 | 0.341 | 2.7 | 202 | 232,601 |
| GPT-2 BPE | 3 | 4,194 | 1,936 | 14,424 | 0.340 | 2.7 | 202 | 232,601 |
| GPT-2 BPE | 4 | 3,759 | 1,773 | 10,747 | 0.293 | 2.2 | 178 | 232,601 |
| SmolLM2 | 1 | 4,200 | 1,938 | 14,430 | 0.340 | 2.7 | 202 | 234,901 |
| SmolLM2 | 2 | 4,200 | 1,938 | 14,430 | 0.340 | 2.7 | 202 | 234,901 |
| SmolLM2 | 3 | 4,193 | 1,937 | 14,426 | 0.340 | 2.7 | 202 | 234,901 |
| SmolLM2 | 4 | 3,768 | 1,802 | 10,980 | 0.296 | 2.2 | 182 | 234,901 |
| Qwen2.5 | 1 | 4,200 | 1,938 | 14,430 | 0.355 | 2.7 | 202 | 218,418 |
| Qwen2.5 | 2 | 4,200 | 1,938 | 14,430 | 0.355 | 2.7 | 202 | 218,418 |
| Qwen2.5 | 3 | 4,188 | 1,935 | 14,408 | 0.355 | 2.7 | 201 | 218,418 |
| Qwen2.5 | 4 | 3,691 | 1,748 | 10,682 | 0.304 | 2.2 | 177 | 218,418 |

## T6 legal — EUR-Lex × EuroVoc 4.24

Source `experiments/t6-legal-regulatory/runs/v1/cardinality.json`.

| Tokenizer | ℓ_min | Linkable | Linked | Spans | Covered | /1k | ≥10 | Sample tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 BPE | 1 | 7,841 | 2,077 | 82,772 | 0.113 | 26.0 | 779 | 1,098,984 |
| GPT-2 BPE | 2 | 7,458 | 1,633 | 27,437 | 0.068 | 10.7 | 416 | 1,098,984 |
| GPT-2 BPE | 3 | 4,761 | 686 | 10,245 | 0.041 | 4.4 | 124 | 1,098,984 |
| GPT-2 BPE | 4 | 3,101 | 311 | 6,678 | 0.032 | 2.8 | 58 | 1,098,984 |
| SmolLM2 | 1 | 7,841 | 2,071 | 82,976 | 0.104 | 24.9 | 779 | 1,190,514 |
| SmolLM2 | 2 | 7,415 | 1,602 | 27,576 | 0.062 | 10.4 | 408 | 1,190,514 |
| SmolLM2 | 3 | 4,611 | 657 | 10,501 | 0.037 | 4.5 | 125 | 1,190,514 |
| SmolLM2 | 4 | 2,947 | 303 | 6,167 | 0.029 | 2.5 | 55 | 1,190,514 |
| Qwen2.5 | 1 | 7,841 | 2,072 | 83,085 | 0.106 | 25.3 | 779 | 1,147,280 |
| Qwen2.5 | 2 | 7,459 | 1,601 | 26,764 | 0.063 | 10.0 | 416 | 1,147,280 |
| Qwen2.5 | 3 | 4,651 | 640 | 9,427 | 0.036 | 3.7 | 120 | 1,147,280 |
| Qwen2.5 | 4 | 2,725 | 268 | 6,367 | 0.028 | 2.6 | 55 | 1,147,280 |

_Generated by `manuscript/figures/make_figures.py` from `experiments/c3-general-corpus/runs/v2/cardinality.json`, `experiments/t1-open-clinical/runs/v1/cardinality_by_source.json`, `experiments/t3-product-catalogues/runs/v1/cardinality.json`, `experiments/t4-chemistry/runs/v1/cardinality.json`, `experiments/t5-enterprise-glossary/runs/v1/cardinality.json`, `experiments/t6-legal-regulatory/runs/v1/cardinality.json`. Do not edit by hand._
