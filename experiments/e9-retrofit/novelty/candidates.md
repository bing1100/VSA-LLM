# E9 novelty screen — candidate tracks (decision 55)

P0 = SmolLM2-360M, evaluation only, 512 windows of each candidate's slice (calibration tracks: their full E9 P0 runs). Primary proxy: inside nats per linked span (orders T5 > T4 > T1 > WordNet at both host sizes; `calibration/`); `vs t4` = its ratio to T4's value. The general / domain rate is biased towards 0 for candidates whose domain text keeps only abstracts that mention a name (a denser domain), so it does not rank them against the calibration tracks; inside nats per span does not depend on density.

| track / candidate | source | licence | entries | occurrences | inside nats / span | vs t4 | inside loss | subtokens / span | general / domain | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| t5 | finished E9 track | — | — | — | 15.31 | 2.87 | 3.38 | 5.53 | 0.0000 | novelty -76.1%, gain -7.47% |
| t4 | finished E9 track | — | — | — | 5.34 | 1.00 | 1.64 | 5.09 | 0.0363 | novelty -48.3%, gain -1.17% |
| t1 | finished E9 track | — | — | — | 1.26 | 0.24 | 0.76 | 2.77 | 0.3834 | novelty -6.6%, gain -0.01% |
| wordnet | finished E9 track | — | — | — | 1.07 | 0.20 | 0.91 | 2.25 | 0.9089 | novelty -0.7%, gain +0.01% |
| T2 developer tools (built) | generated private libraries + real API docs (90% / 10%) | generated; PSF-2.0 / MIT docs | 611 held-out, 1,203 rare | 2,147 held-out, 3,224 rare | 25.80 | 4.84 | 4.13 | 7.29 | 0.0000 | excluded: templated synthetic text (decision 28); its novelty is the generated identifiers |
| T3 product (built) | Google/Shopify taxonomy + Amazon ESCI titles | MIT / Apache-2.0 | 148 held-out, 38 rare | 1,401 held-out, 64 rare | 1.82 | 0.34 | 1.32 | 2.41 | 0.1017 | low novelty (T1-like); rare stratum infeasible; needs a 50/50 relink |
| T6 legal (built) | EuroVoc 4.24 + MultiEURLEX (EUR-Lex) | EU reuse policy / CC BY-SA 4.0 | 120 held-out, 65 rare | 5,122 held-out, 114 rare | 1.63 | 0.30 | 0.90 | 2.96 | 0.3031 | low novelty (T1-like); rare stratum infeasible; needs a 50/50 relink |
| MeSH descriptors introduced 2024-26 | MeSH 2026 desc + PubMed 2025-26 | NLM terms (free; attribution) | 465 | ≈ 110,000 | 2.16 | 0.41 | 1.10 | 2.98 | 0.0110 | low novelty: new headings for established words |
| MeSH SCRs introduced 2022-26 | MeSH 2026 supp + PubMed 2025-26 | NLM terms (free; attribution) | 541 | 14,114 | 5.92 | 1.11 | 1.59 | 4.81 | 0.0016 | above T4, but too small for the held-out stratum; a recency sub-stratum of T7 |
| MeSH names absent from general text, SCR + descriptors (alias rule) | MeSH 2026 desc + supp + PubMed 2025-26 + FineWeb-Edu | NLM terms; ODC-By (FineWeb-Edu, counts only) | 10,075 | 227,344 | 4.91 | 0.92 | 1.34 | 4.72 | 0.0008 | superseded: a common concept could enter through a rare spelling |
| MeSH names absent from general text, SCR + descriptors | MeSH 2026 desc + supp + PubMed 2025-26 + FineWeb-Edu | NLM terms; ODC-By (counts only) | 7,112 | 147,391 | 5.43 | 1.02 | 1.43 | 4.88 | 0.0007 | T4 level; descriptors dilute it |
| MeSH names absent from general text, introduced 2015+ | MeSH 2026 desc + supp + PubMed 2025-26 + FineWeb-Edu | NLM terms; ODC-By (counts only) | 1,794 | 56,948 | 5.98 | 1.12 | 1.56 | 4.90 | 0.0000 | above T4; small (403 records with ≥ 25 occurrences) |
| MeSH SCR names absent from general text (T7) | MeSH 2026 supp + PubMed 2025-26 (+ 2026 update files) + FineWeb-Edu | NLM terms; ODC-By (counts only) | 5,124 | 88,001 | 6.59 | 1.23 | 1.66 | 5.04 | 0.0008 | recommended (T7): the most novel natural candidate of feasible size |
