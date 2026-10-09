# T7-ROOD new-vocabulary biomedical, held-out documents excluded from training (MeSH SCR + PubMed 2025-26) — corpus build

Ontology: MeSH 2026 supplementary concept records and descriptors whose names are frequent in the training-side PubMed 2025–26 text and absent from a general-text sample (`mesh_novel`; frames: mapped heading, pharmacological action, record class, tree branches); corpus: the PubMed abstracts that mention a selected name, mixed 50/50 (tokens) with FineWeb-Edu (the C3 stream).

Data: PubMed, courtesy of the U.S. National Library of Medicine (2026 baseline snapshot, not updated); MeSH 2026, U.S. National Library of Medicine.

Linker: 11,141 aliases over 8,184 entries (alias policy None).

| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |
|---|---:|---:|---|
| HuggingFaceTB/SmolLM2-360M | 11,141 | 11,141 | 0 / 585 / 1,561 / 2,100 / 6,895 |

Holdout: 908 entries (876 chosen of 3,496 eligible, 32 added by the closure, 0 hub entries not eligible); sha256 `5e133b5f2587f3ad0931442579eabc51105214a26895ce2992b0c76241727e10`.

## Corpora

PubMed evaluation pool: PMIDs whose sha256 bucket is < 2000 / 10,000 (never trained on); `eval-pubmed` uses all of them. General text: the C3 FineWeb-Edu stream, its first 5,000 documents (C3's evaluation documents) never trained on. Mixing calibration (chars per HuggingFaceTB/SmolLM2-360M token): PubMed 4.443, general 4.580.

| tokenizer | corpus | tokens | documents | spans | PubMed token share |
|---|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | eval | 10,009,876 | 16,845 | 32,132 | 0.501 |
| HuggingFaceTB/SmolLM2-360M | eval-pubmed | 6,538,704 | 15,599 | 41,940 | — |
| HuggingFaceTB/SmolLM2-360M | eval-general | 4,998,786 | 5,000 | 169 | — |
| HuggingFaceTB/SmolLM2-360M | train | 57,268,470 | 96,112 | 141,455 | 0.495 |

## Span cardinality (evaluation samples, full alias table)

### pubmed — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 1,612 | 5,565 | 0.032 | 692 | 804 | 116 | 1.2 |
| 2 | 8,184 | 1,612 | 5,565 | 0.032 | 692 | 804 | 116 | 1.2 |
| 3 | 7,764 | 1,541 | 5,317 | 0.031 | 665 | 765 | 111 | 1.2 |
| 4 | 6,732 | 1,300 | 4,363 | 0.028 | 580 | 632 | 88 | 1.0 |

### general — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 26 | 68 | 0.000 | 16 | 10 | 0 | 0.0 |
| 2 | 8,184 | 26 | 68 | 0.000 | 16 | 10 | 0 | 0.0 |
| 3 | 7,764 | 24 | 63 | 0.000 | 15 | 9 | 0 | 0.0 |
| 4 | 6,732 | 21 | 53 | 0.000 | 14 | 7 | 0 | 0.0 |

### mixed — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 1,617 | 5,602 | 0.016 | 693 | 808 | 116 | 0.8 |
| 2 | 8,184 | 1,617 | 5,602 | 0.016 | 693 | 808 | 116 | 0.8 |
| 3 | 7,764 | 1,546 | 5,354 | 0.016 | 666 | 769 | 111 | 0.7 |
| 4 | 6,732 | 1,305 | 4,394 | 0.014 | 583 | 634 | 88 | 0.6 |

## Feasibility per ℓ_min

**Criterion** (on the `eval-pubmed` split, per tokenizer): ≥ 300 held-out entries with ≥ 5 occurrences and ≥ 2,000 held-out span occurrences; ≥ 300 distinct rare entries (training frequency 1–9, the trainer's `after_rare` stratum) linked and ≥ 2,000 rare-entry occurrences.

**Why these numbers.** E4 gate item 1 compares the held-out (and seen-rare) stratum loss between conditions with a paired bootstrap over examples, Holm-corrected over the condition grid. The unit that varies is the span occurrence (its 8 following tokens are strongly correlated), so the occurrence count sets the standard error: with a per-occurrence paired loss difference of SD ≈ 0.5–1 nat, 2,000 occurrences give SE ≈ 0.011–0.022 nat, i.e. a minimum detectable difference of ≈ 0.04–0.08 nat at 80% power after Holm over ≈ 4 comparisons — the size of effect a channel must show to matter. Occurrences cluster by concept, so at least 300 distinct concepts with ≥ 5 occurrences each keep a concept-cluster bootstrap stable and stop a few frequent concepts from carrying the stratum. The same two numbers apply to the rare stratum (gate item 1, second half).

Rare-stratum columns are shown only when the training corpus has its full budget (a slice cannot measure training frequencies). `eval windows needed` = the fewest evenly spread 1,024-token windows (`eval.windows` of the trainer) whose spans meet the measurable criteria; the trainer default is 4,096.

| tokenizer | ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | training frequencies | verdict (whole split) | eval windows needed | windows for whole split |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 1 | 6,538,704 | 8,786 | 411 | 4,804 | 1,705 | measured | feasible | 4,096 | 6,384 |
| HuggingFaceTB/SmolLM2-360M | 2 | 6,538,704 | 8,786 | 411 | 4,804 | 1,705 | measured | feasible | 4,096 | 6,384 |
| HuggingFaceTB/SmolLM2-360M | 3 | 6,538,704 | 8,534 | 390 | 4,760 | 1,710 | measured | feasible | 6,144 | 6,384 |
| HuggingFaceTB/SmolLM2-360M | 4 | 6,538,704 | 7,087 | 306 | 4,224 | 1,581 | measured | feasible | 6,144 | 6,384 |

## Held-out documents excluded from training (T7-ROOD, decision 63 H1)

Split `frequency`; names: `candidates` (4,302 names, 1,351 of them linked aliases; sha256 `e03b683f61add881…`). Matcher: mesh_novel.MentionCounter: case-insensitive whole alphanumeric tokens (word boundaries).

Training stream: 96,112 documents, 57,268,470 reference tokens (target 57268320; reached: True). Dropped: 15,530 mention-bearing abstracts (6,758,752 tokens), 0 general documents, 0 refill abstracts. Refill: 16,347 abstracts without a selected name (`top_up: domain`).

| tokenizer | documents | tokens | leaked documents | mention-bearing tokens | refill tokens | general tokens | shared with reference (docs) | reference only (docs, flagged) | this only (docs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 96,112 | 57,268,470 | 0 | 21,861,364 | 6,497,623 | 28,909,483 | 79,514 | 15,530 (15,530) | 16,586 |

Evaluation corpora against the reference build (byte-identical `tokens.bin` / `spans.npz`): HuggingFaceTB/SmolLM2-360M: eval yes, eval-pubmed yes, eval-general yes.

**Leakage audit:** 0 training documents contain a held-out name (every written document decoded and matched again).
