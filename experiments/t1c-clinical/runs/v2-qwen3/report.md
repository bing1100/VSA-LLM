# T1c corpus — clinical (SNOMED CT + MIMIC-III)

Ontology: SNOMED CT International Edition 2022-05-31 (RF2 snapshot; active concepts of the selected top-level hierarchies; frames from the active inferred relationships). Corpus: MIMIC-III 1.4 `NOTEEVENTS` (rows with `ISERROR` dropped), split by patient, mixed 50/50 (tokens) with FineWeb-Edu (the C3 stream). Licensed and credentialed sources: this report holds aggregates only; every derived table stays under the data root.

Concepts 310,698 ({'body_structure': 40443, 'clinical_finding': 117002, 'observable_entity': 10127, 'organism': 32590, 'procedure': 58730, 'product': 24681, 'substance': 27125}); 62 relations; 8,192 atomics; 6,343 concepts with a SNOMED text definition.

Linker: 496,997 aliases over 295,143 entries; alias policy {'drop_abbreviations': True, 'max_abbreviation_chars': 8, 'min_chars': 3, 'max_words': 8, 'function_words': 187}; decisions {'concepts_without_alias': 15831, 'descriptions': 829188, 'dropped_abbreviation': 1560, 'dropped_function_word': 4, 'dropped_long': 59502, 'dropped_short': 6, 'keep': 768116}.

| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |
|---|---:|---:|---|
| HuggingFaceTB/SmolLM2-135M | 496,997 | 495,279 | 1,718 / 20,257 / 46,231 / 65,623 / 363,168 |
| Qwen/Qwen3-0.6B-Base | 496,997 | 495,667 | 1,330 / 15,759 / 37,589 / 57,138 / 385,181 |

Notes: 2,083,180 rows; 886 `ISERROR` rows and 68 empty notes dropped; patients train / eval 41,501 / 4,645 (on both sides: 0); notes train / eval 1,881,250 / 200,976.

| category | train notes | eval notes | train characters |
|---|---:|---:|---:|
| Physician | 128,171 | 13,110 | 914,689,164 |
| Radiology | 471,670 | 50,609 | 819,945,011 |
| Nursing/other | 741,014 | 81,455 | 591,050,545 |
| Discharge summary | 53,855 | 5,797 | 518,294,657 |
| Nursing | 202,545 | 20,637 | 361,034,108 |
| Echo | 41,525 | 4,269 | 96,281,238 |
| ECG | 189,205 | 19,806 | 39,671,343 |
| Respiratory | 28,728 | 2,973 | 39,010,284 |
| Nutrition | 8,575 | 825 | 20,886,814 |
| Rehab Services | 4,965 | 443 | 15,471,952 |
| General | 7,510 | 726 | 11,666,785 |
| Social Work | 2,451 | 210 | 5,259,449 |
| Case Management | 858 | 95 | 950,235 |
| Consult | 86 | 12 | 529,377 |
| Pharmacy | 92 | 9 | 233,672 |

Holdout: 2,074 entries (479 chosen of 1,596 eligible, 1,595 added by the closure, 2,541 hub entries not eligible); sha256 `7369ec276b45465900cc3a9898381392f52fa99d47821a4c6ad80df5eae8cec5` (names in the data root, not committed).

## Corpora

Mixing calibration (chars per HuggingFaceTB/SmolLM2-360M token): notes 3.275, general 4.580; `eval-mimic` = the first 40000 evaluation-patient notes (shuffled order).

| tokenizer | corpus | tokens | documents | spans | note token share |
|---|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | eval | 10,001,725 | 13,926 | 502,230 | 0.501 |
| HuggingFaceTB/SmolLM2-360M | eval-mimic | 22,352,802 | 39,993 | 1,311,829 | — |
| HuggingFaceTB/SmolLM2-360M | eval-general | 4,998,786 | 5,000 | 207,825 | — |
| HuggingFaceTB/SmolLM2-360M | train | 130,000,026 | 176,299 | 6,252,156 | 0.500 |
| Qwen/Qwen3-0.6B-Base | eval | 9,882,742 | 13,949 | 507,294 | 0.505 |
| Qwen/Qwen3-0.6B-Base | eval-mimic | 22,182,128 | 40,000 | 1,318,485 | — |
| Qwen/Qwen3-0.6B-Base | eval-general | 4,895,335 | 5,000 | 210,743 | — |
| Qwen/Qwen3-0.6B-Base | train | 130,000,451 | 178,658 | 6,381,540 | 0.504 |

## Span cardinality (evaluation samples, full alias table)

### mimic — HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 295,143 | 3,983 | 64,005 | 0.114 | 1,307 | 1,742 | 934 | 19.7 |
| 2 | 294,424 | 3,702 | 33,544 | 0.089 | 1,312 | 1,709 | 681 | 11.2 |
| 3 | 285,653 | 2,756 | 17,999 | 0.062 | 1,098 | 1,276 | 382 | 6.1 |
| 4 | 263,874 | 1,750 | 8,884 | 0.038 | 772 | 791 | 187 | 3.1 |

### mimic — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 295,143 | 3,983 | 64,338 | 0.123 | 1,304 | 1,739 | 940 | 19.6 |
| 2 | 294,579 | 3,710 | 36,432 | 0.100 | 1,301 | 1,695 | 714 | 12.0 |
| 3 | 287,539 | 2,857 | 20,876 | 0.073 | 1,121 | 1,309 | 427 | 6.9 |
| 4 | 269,528 | 1,938 | 10,791 | 0.046 | 841 | 864 | 233 | 3.7 |

### general — HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 295,143 | 5,054 | 83,943 | 0.049 | 1,881 | 2,045 | 1,128 | 15.2 |
| 2 | 294,424 | 4,472 | 13,830 | 0.016 | 2,180 | 2,062 | 230 | 2.9 |
| 3 | 285,653 | 1,979 | 4,399 | 0.007 | 1,214 | 719 | 46 | 0.9 |
| 4 | 263,874 | 783 | 1,460 | 0.003 | 534 | 240 | 9 | 0.3 |

### general — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 295,143 | 5,067 | 84,911 | 0.053 | 1,883 | 2,056 | 1,128 | 15.5 |
| 2 | 294,579 | 4,618 | 16,623 | 0.020 | 2,117 | 2,159 | 342 | 3.4 |
| 3 | 287,539 | 2,306 | 6,072 | 0.010 | 1,290 | 928 | 88 | 1.2 |
| 4 | 269,528 | 1,024 | 2,436 | 0.005 | 647 | 351 | 26 | 0.5 |

### mixed — HuggingFaceTB/SmolLM2-135M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 295,143 | 6,346 | 108,266 | 0.082 | 2,211 | 2,628 | 1,507 | 17.9 |
| 2 | 294,424 | 5,831 | 41,122 | 0.053 | 2,400 | 2,610 | 821 | 7.9 |
| 3 | 285,653 | 3,689 | 20,491 | 0.035 | 1,678 | 1,598 | 413 | 4.0 |
| 4 | 263,874 | 2,119 | 9,664 | 0.021 | 1,033 | 890 | 196 | 2.0 |

### mixed — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 295,143 | 6,357 | 108,994 | 0.088 | 2,217 | 2,629 | 1,511 | 18.0 |
| 2 | 294,579 | 5,930 | 45,430 | 0.061 | 2,365 | 2,664 | 901 | 8.6 |
| 3 | 287,539 | 3,935 | 24,283 | 0.042 | 1,733 | 1,717 | 485 | 4.7 |
| 4 | 269,528 | 2,417 | 12,184 | 0.026 | 1,153 | 1,012 | 252 | 2.4 |

## Feasibility per stratum

Bar (decision 17, WP-T1's power argument): ≥ 2,000 span occurrences and ≥ 300 entries with ≥ 5 occurrences on `eval-mimic`; WP-C7's lower bar (≥ 1,000 occurrences, ≥ 50 entries) marks a stratum exploratory-feasible. Strata as the trainer's: held-out; unseen (training frequency 0, not held out); rare-seen (1–9); 3+-subtoken (any entry). `windows needed` = the fewest evenly spread 1024-token windows that meet the bar; E9 evaluates on 2,048.

| tokenizer | ℓ_min | stratum | occurrences (split) | entries | entries ≥ 5 | verdict (split) | occurrences / entries ≥ 5 in the E9 windows | verdict (E9 windows) | windows needed |
|---|---:|---|---:|---:|---:|---|---|---|---:|
| HuggingFaceTB/SmolLM2-360M | 2 | after_heldout | 40,690 | 442 | 347 | feasible | 3,638 / 161 (2,048 windows) | feasible (entries counted with ≥ 1 occurrence) | 12288 |
| HuggingFaceTB/SmolLM2-360M | 2 | after_unseen | 1,365 | 786 | 32 | exploratory-feasible | 117 / 1 (2,048 windows) | infeasible | not reached |
| HuggingFaceTB/SmolLM2-360M | 2 | after_rare_seen | 6,592 | 2,636 | 316 | feasible | 668 / 3 (2,048 windows) | infeasible | 21827 |
| HuggingFaceTB/SmolLM2-360M | 2 | after_len3plus | 377,116 | 8,901 | 4,242 | feasible | 35,564 / 1,198 (2,048 windows) | feasible | 1024 |
| HuggingFaceTB/SmolLM2-360M | 3 | after_heldout | 22,552 | 384 | 280 | feasible (entries counted with ≥ 1 occurrence) | 2,063 / 116 (2,048 windows) | exploratory-feasible | not reached |
| HuggingFaceTB/SmolLM2-360M | 3 | after_unseen | 1,460 | 877 | 34 | exploratory-feasible | 126 / 1 (2,048 windows) | infeasible | not reached |
| HuggingFaceTB/SmolLM2-360M | 3 | after_rare_seen | 6,633 | 2,746 | 311 | feasible | 665 / 2 (2,048 windows) | infeasible | 21827 |
| HuggingFaceTB/SmolLM2-360M | 3 | after_len3plus | 377,116 | 8,901 | 4,242 | feasible | 35,564 / 1,198 (2,048 windows) | feasible | 1024 |
| Qwen/Qwen3-0.6B-Base | 2 | after_heldout | 44,821 | 443 | 347 | feasible | 4,201 / 167 (2,048 windows) | feasible (entries counted with ≥ 1 occurrence) | 12288 |
| Qwen/Qwen3-0.6B-Base | 2 | after_unseen | 1,366 | 782 | 32 | exploratory-feasible | 132 / 1 (2,048 windows) | infeasible | not reached |
| Qwen/Qwen3-0.6B-Base | 2 | after_rare_seen | 6,525 | 2,625 | 313 | feasible | 626 / 3 (2,048 windows) | infeasible | 21661 |
| Qwen/Qwen3-0.6B-Base | 2 | after_len3plus | 433,547 | 9,105 | 4,397 | feasible | 40,833 / 1,269 (2,048 windows) | feasible | 1024 |
| Qwen/Qwen3-0.6B-Base | 3 | after_heldout | 25,704 | 392 | 284 | feasible (entries counted with ≥ 1 occurrence) | 2,426 / 128 (2,048 windows) | exploratory-feasible | not reached |
| Qwen/Qwen3-0.6B-Base | 3 | after_unseen | 1,461 | 849 | 32 | exploratory-feasible | 144 / 1 (2,048 windows) | infeasible | not reached |
| Qwen/Qwen3-0.6B-Base | 3 | after_rare_seen | 6,700 | 2,740 | 325 | feasible | 633 / 2 (2,048 windows) | infeasible | 21661 |
| Qwen/Qwen3-0.6B-Base | 3 | after_len3plus | 433,547 | 9,105 | 4,397 | feasible | 40,833 / 1,269 (2,048 windows) | feasible | 1024 |
