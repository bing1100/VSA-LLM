# T7 rounds for E13 (round 2 = MeSH SCRs introduced latest; their documents excluded from round 1) — corpus build

Ontology: MeSH 2026 supplementary concept records and descriptors whose names are frequent in the training-side PubMed 2025–26 text and absent from a general-text sample (`mesh_novel`; frames: mapped heading, pharmacological action, record class, tree branches); corpus: the PubMed abstracts that mention a selected name, mixed 50/50 (tokens) with FineWeb-Edu (the C3 stream).

Data: PubMed, courtesy of the U.S. National Library of Medicine (2026 baseline snapshot, not updated); MeSH 2026, U.S. National Library of Medicine.

Linker: 11,141 aliases over 8,184 entries (alias policy None).

| tokenizer | aliases | ≥ 2 subtokens | 1 / 2 / 3 / 4 / 5+ |
|---|---:|---:|---|
| HuggingFaceTB/SmolLM2-360M | 11,141 | 11,141 | 0 / 585 / 1,561 / 2,100 / 6,895 |
| Qwen/Qwen3-0.6B-Base | 11,141 | 11,141 | 0 / 711 / 1,582 / 2,088 / 6,760 |

Holdout: 2,470 entries (2,455 chosen of 8,184 eligible, 15 added by the closure, 0 hub entries not eligible); sha256 `03f35409729e29cbef50ec322061911951fdf06b566cbf16888db544ee7e856e`.

## Corpora

PubMed evaluation pool: PMIDs whose sha256 bucket is < 2000 / 10,000 (never trained on); `eval-pubmed` uses all of them. General text: the C3 FineWeb-Edu stream, its first 5,000 documents (C3's evaluation documents) never trained on. Mixing calibration (chars per HuggingFaceTB/SmolLM2-360M token): PubMed 4.443, general 4.580.

| tokenizer | corpus | tokens | documents | spans | PubMed token share |
|---|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | eval | 10,009,876 | 16,845 | 32,132 | 0.501 |
| HuggingFaceTB/SmolLM2-360M | eval-pubmed | 6,538,704 | 15,599 | 41,940 | — |
| HuggingFaceTB/SmolLM2-360M | eval-general | 4,998,786 | 5,000 | 169 | — |
| HuggingFaceTB/SmolLM2-360M | train | 57,268,379 | 96,437 | 108,417 | 0.492 |
| Qwen/Qwen3-0.6B-Base | eval | 9,903,747 | 16,845 | 32,159 | 0.506 |
| Qwen/Qwen3-0.6B-Base | eval-pubmed | 6,536,229 | 15,599 | 41,970 | — |
| Qwen/Qwen3-0.6B-Base | eval-general | 4,895,335 | 5,000 | 167 | — |
| Qwen/Qwen3-0.6B-Base | train | 56,624,295 | 96,437 | 108,571 | 0.497 |

## Span cardinality (evaluation samples, full alias table)

### pubmed — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 1,612 | 5,565 | 0.032 | 692 | 804 | 116 | 1.2 |
| 2 | 8,184 | 1,612 | 5,565 | 0.032 | 692 | 804 | 116 | 1.2 |
| 3 | 7,764 | 1,541 | 5,317 | 0.031 | 665 | 765 | 111 | 1.2 |
| 4 | 6,732 | 1,300 | 4,363 | 0.028 | 580 | 632 | 88 | 1.0 |

### pubmed — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 1,614 | 5,573 | 0.032 | 693 | 805 | 116 | 1.2 |
| 2 | 8,184 | 1,614 | 5,573 | 0.032 | 693 | 805 | 116 | 1.2 |
| 3 | 7,692 | 1,537 | 5,290 | 0.031 | 667 | 758 | 112 | 1.2 |
| 4 | 6,655 | 1,304 | 4,317 | 0.028 | 583 | 636 | 85 | 1.0 |

### general — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 26 | 68 | 0.000 | 16 | 10 | 0 | 0.0 |
| 2 | 8,184 | 26 | 68 | 0.000 | 16 | 10 | 0 | 0.0 |
| 3 | 7,764 | 24 | 63 | 0.000 | 15 | 9 | 0 | 0.0 |
| 4 | 6,732 | 21 | 53 | 0.000 | 14 | 7 | 0 | 0.0 |

### general — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 26 | 65 | 0.000 | 16 | 10 | 0 | 0.0 |
| 2 | 8,184 | 26 | 65 | 0.000 | 16 | 10 | 0 | 0.0 |
| 3 | 7,692 | 23 | 60 | 0.000 | 15 | 8 | 0 | 0.0 |
| 4 | 6,655 | 20 | 52 | 0.000 | 13 | 7 | 0 | 0.0 |

### mixed — HuggingFaceTB/SmolLM2-360M

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 1,617 | 5,602 | 0.016 | 693 | 808 | 116 | 0.8 |
| 2 | 8,184 | 1,617 | 5,602 | 0.016 | 693 | 808 | 116 | 0.8 |
| 3 | 7,764 | 1,546 | 5,354 | 0.016 | 666 | 769 | 111 | 0.7 |
| 4 | 6,732 | 1,305 | 4,394 | 0.014 | 583 | 634 | 88 | 0.6 |

### mixed — Qwen/Qwen3-0.6B-Base

| ℓ_min | linkable entries | linked entries | span occurrences | covered-token fraction | entries seen once | 2–9 | ≥ 10 | distinct entries / 1024 tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,184 | 1,619 | 5,610 | 0.016 | 694 | 809 | 116 | 0.8 |
| 2 | 8,184 | 1,619 | 5,610 | 0.016 | 694 | 809 | 116 | 0.8 |
| 3 | 7,692 | 1,542 | 5,327 | 0.016 | 668 | 762 | 112 | 0.7 |
| 4 | 6,655 | 1,308 | 4,348 | 0.014 | 584 | 639 | 85 | 0.6 |

## Feasibility per ℓ_min

**Criterion** (on the `eval-pubmed` split, per tokenizer): ≥ 300 held-out entries with ≥ 5 occurrences and ≥ 2,000 held-out span occurrences; ≥ 300 distinct rare entries (training frequency 1–9, the trainer's `after_rare` stratum) linked and ≥ 2,000 rare-entry occurrences.

**Why these numbers.** E4 gate item 1 compares the held-out (and seen-rare) stratum loss between conditions with a paired bootstrap over examples, Holm-corrected over the condition grid. The unit that varies is the span occurrence (its 8 following tokens are strongly correlated), so the occurrence count sets the standard error: with a per-occurrence paired loss difference of SD ≈ 0.5–1 nat, 2,000 occurrences give SE ≈ 0.011–0.022 nat, i.e. a minimum detectable difference of ≈ 0.04–0.08 nat at 80% power after Holm over ≈ 4 comparisons — the size of effect a channel must show to matter. Occurrences cluster by concept, so at least 300 distinct concepts with ≥ 5 occurrences each keep a concept-cluster bootstrap stable and stop a few frequent concepts from carrying the stratum. The same two numbers apply to the rare stratum (gate item 1, second half).

Rare-stratum columns are shown only when the training corpus has its full budget (a slice cannot measure training frequencies). `eval windows needed` = the fewest evenly spread 1,024-token windows (`eval.windows` of the trainer) whose spans meet the measurable criteria; the trainer default is 4,096.

| tokenizer | ℓ_min | eval tokens | held-out occ. | held-out entries ≥ 5 | rare occ. | rare entries | training frequencies | verdict (whole split) | eval windows needed | windows for whole split |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 1 | 6,538,704 | 16,346 | 706 | 3,476 | 1,255 | measured | feasible | 4,096 | 6,384 |
| HuggingFaceTB/SmolLM2-360M | 2 | 6,538,704 | 16,346 | 706 | 3,476 | 1,255 | measured | feasible | 4,096 | 6,384 |
| HuggingFaceTB/SmolLM2-360M | 3 | 6,538,704 | 16,260 | 700 | 3,463 | 1,269 | measured | feasible | 4,096 | 6,384 |
| HuggingFaceTB/SmolLM2-360M | 4 | 6,538,704 | 14,913 | 630 | 2,903 | 1,133 | measured | feasible | 6,144 | 6,384 |
| Qwen/Qwen3-0.6B-Base | 1 | 6,536,229 | 16,350 | 706 | 3,481 | 1,253 | measured | feasible | 4,096 | 6,382 |
| Qwen/Qwen3-0.6B-Base | 2 | 6,536,229 | 16,350 | 706 | 3,481 | 1,253 | measured | feasible | 4,096 | 6,382 |
| Qwen/Qwen3-0.6B-Base | 3 | 6,536,229 | 16,218 | 694 | 3,500 | 1,287 | measured | feasible | 4,096 | 6,382 |
| Qwen/Qwen3-0.6B-Base | 4 | 6,536,229 | 14,747 | 628 | 2,836 | 1,127 | measured | feasible | 6,144 | 6,382 |

## Held-out documents excluded from training (T7-ROOD, decision 63 H1)

Split `date`; names: `candidates` (6,422 candidate names and 3,222 linked aliases of 2,470 records: 6,416 matcher keys, sha256 `130e95fd986ecc6a…`). Matcher: mesh_novel.MentionCounter: case-insensitive whole alphanumeric tokens (word boundaries).

Training stream: 96,437 documents, 57,268,379 reference tokens (target 57268320; reached: True). Dropped: 24,882 mention-bearing abstracts (10,714,690 tokens), 0 general documents, 0 refill abstracts. Refill: 25,888 abstracts without a selected name (`top_up: domain`).

| tokenizer | documents | tokens | leaked documents | mention-bearing tokens | refill tokens | general tokens | shared with reference (docs) | reference only (docs, flagged) | this only (docs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 96,437 | 57,268,379 | 0 | 17,905,426 | 10,297,681 | 29,065,272 | 70,162 | 24,882 (24,882) | 26,256 |
| Qwen/Qwen3-0.6B-Base | 96,437 | 56,624,295 | 0 | 17,908,367 | 10,245,571 | 28,470,357 | 70,162 | 24,882 (24,882) | 26,256 |

Evaluation corpora against the reference build (identical `tokens.bin` bytes, span arrays and manifests): HuggingFaceTB/SmolLM2-360M: eval no, eval-pubmed no, eval-general no; Qwen/Qwen3-0.6B-Base: eval no, eval-pubmed no, eval-general no.

**Leakage audit:** 0 training documents contain a held-out name (every written document decoded and matched again).

## E13 rounds

Round 2 = entries introduced after 2013-01-26 (MeSH DateIntroduced; {'DateIntroduced': 8184}): 2,470 entries (2,455 by date, 15 by the closure), round 1 5,714.

| tokenizer | corpus | tokens | documents | spans | PubMed share |
|---|---|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | train-round2 | 21,326,939 | 34,547 | 83,223 | 0.502 |
| HuggingFaceTB/SmolLM2-360M | eval-round2 | 2,417,494 | 5,628 | 18,010 | — |
| HuggingFaceTB/SmolLM2-360M | eval-round1 | 4,121,210 | 9,971 | 23,930 | — |
| Qwen/Qwen3-0.6B-Base | train-round2 | 21,103,066 | 34,547 | 83,285 | 0.507 |
| Qwen/Qwen3-0.6B-Base | eval-round2 | 2,413,915 | 5,628 | 18,022 | — |
| Qwen/Qwen3-0.6B-Base | eval-round1 | 4,122,314 | 9,971 | 23,948 | — |

Round-2 feasibility on `eval-round2` (ℓ_min 2): HuggingFaceTB/SmolLM2-360M: 16,337 occurrences, 706 entries ≥ 5, feasible for the held-out stratum only (rare stratum underpowered), windows for the whole split 2,359; Qwen/Qwen3-0.6B-Base: 16,341 occurrences, 706 entries ≥ 5, feasible for the held-out stratum only (rare stratum underpowered), windows for the whole split 2,356.
