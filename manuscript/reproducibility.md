# Reproducibility record

**Status:** 2026-10-02. This page covers results committed so far. Rows for pending blocks are left open, to be filled as each report lands.

Sources:
- [R0](../reports/R0-reproduction.md) and [gates.md](../resources/plan-improvement/gates.md);
- the run manifests (`experiments/*/runs/*/manifest.json`);
- `~/data/vsa-llm/DATA_SOURCES.md` and `SHA256SUMS`, summarized here.

The data documentation itself stays outside the repository, because it describes files that are never committed.

## 1. Hardware

| Item | Value |
|---|---|
| Experiment machine (from 2026-10-02) | `bhux-tiny`: AMD Ryzen 5 8600G (6 cores / 12 threads), 60 GiB RAM, NVIDIA GeForce RTX 3090 24 GB (driver 595.84), Linux 7.0.0-31 |
| RAM screen | `scripts/memcheck.py 16 4` and `16 8`: 0 bad words |
| Previous machine (to 2026-10-01) | Same GPU class; **faulty RAM**: 204 corrupted words in a 6 GiB screen and 2,122 memtester failures. Every committed result that later work relies on was re-run on `bhux-tiny` and compared cell by cell (§5). Partial E1/E2/E3/C3 outputs from that machine were discarded; their re-runs use new run IDs. |
| GPU sharing | One GPU job at a time through the local queue (B11). The GPU also drives the desktop (budget ≈ 22 GB). The B10 GPU latencies were measured while another job held the GPU, so they are qualitative only. |
| CPU threads | CPU experiments that must be bit-reproducible run single-threaded (`num_threads: 1`). Multithreaded CPU training is not bit-reproducible across processes on this machine (G0). |

## 2. Software environment

Conda env `vsa-repro`:
- Python 3.12.4
- torch 2.11.0+cu128, CUDA 12.8
- transformers 4.54.0, tokenizers 0.21.4, safetensors 0.5.3
- numpy 2.2.6, scipy 1.18.1, pyarrow 25.0.1
- nltk 3.8.1 (WordNet 3.0), PyYAML 6.0.2
- torchao 0.18.0 (its Hopper-only extensions fail to load on the 3090; this is harmless)
- figures only: matplotlib 3.11.2, pandas 3.0.6

Every run manifest records the git SHA, a dirty-tree flag, package versions, the device, the thread count, the argv and the config sha256 (schema version 2).

## 3. Data sources

All data were retrieved on 2026-10-02 into `~/data/vsa-llm/` and hashed twice, with both passes equal. Each hash was compared with the publisher's checksum where one exists (HF/GitHub LFS sha256, NLM md5 sidecars). Archives were tested (`gzip -t`, `unzip -t`) and every extracted member was checked against its CRC32. `sha256sum -c SHA256SUMS` re-verifies the full list (158 files). Corpus text, PubMed abstracts and MIMIC-derived material are **never committed**. Committed item files carry pointers (PMID, offsets, sentence hashes) or short excerpts allowed by the licence.

| Dataset (use) | Source and version | Count | sha256 (prefix) | Licence / note |
|---|---|---|---|---|
| FineWeb-Edu `sample/10BT` shards 000 and 002 (C3 general corpus) | HF `HuggingFaceFW/fineweb-edu` @ `87f09149` | 726,000 + 727,000 rows | `b1ba7b2ce4cb5ea6…`, `547ae182d132c9f0…` (equal to hub LFS and `c3.yaml` prefixes) | every row group decompressed without error |
| WordNet 3.0 (C3, E1–E3 ontology) | `nltk` 3.8.1 corpus | 117,659 concepts → 101,500 linker entries (C3) | C3 ontology `f33b9cb6feffe260…` | — |
| MeSH 2026 descriptors (E2, T1) | NLM `xmlmesh/desc2026.gz` (Last-Modified 2026-08-12) | 31,110 descriptors | gz `ccd4d0d33bebfd4c…`; XML `449792ed59ba4d60…` (same from `.zip`) | NLM terms |
| PubMed 2026 baseline, 20 newest files `pubmed26n1315`–`1334` (T1-open, T4 domain text) | `ftp.ncbi.nlm.nih.gov/pubmed/baseline` (README of 2026-01-30) | 574,989 citations; 486,765 English abstracts kept (≈ 181M GPT-2 tokens) | per-file md5 = NLM sidecars; sha256 in `SHA256SUMS` | "Data: PubMed, courtesy of the U.S. National Library of Medicine"; snapshot not updated; text never committed |
| PubMedQA `pqa_labeled` + official test ids (T1 task) | HF `qiaojin/PubMedQA` @ `9001f285`; GitHub `pubmedqa/pubmedqa` @ `1cbae8e9` | 1,000 items; 500 test ids | `3d56bd1abc118845…`; `939fe566f09017d1…` | MIT; PMIDs disjoint from the T1 files |
| LAMBADA (OpenAI) | HF `EleutherAI/lambada_openai` @ `900124bf` | 5,153 | `4aa8d02cd17c7191…` | — |
| WiC | pilehvar.github.io/wic | 5,428 / 638 / 1,400 | zip `f1a2fb67d903c5b9…` | — |
| CARD-660 | pilehvar.github.io/card-660 | 660 pairs | `0cfae4d9ee596c4d…` | — |
| Stanford Rare Words | nlp.stanford.edu/~lmthang | 2,034 pairs | `f38941b02b85cf20…` | — |
| WSD Evaluation Framework (SemCor; ALL) | lcl.uniroma1.it/wsdeval (2017-01-16) | SemCor 226,036; ALL 7,253 (equal to Raganato et al. 2017) | zip `d71c4d3e93d265cb…` | — |
| BLESS | GEMS 2011 attachment via the Wayback Machine (two captures, byte-identical) | 26,554 tuples, 200 concepts | zip `dd4ac24dce1bcdee…` | official host gone |
| HyperLex | GitHub `cambridgeltl/hyperlex` @ `ccb41619` | 2,616 pairs | zip `27ef061e2f711d8e…` | official host gone |
| Google Product Taxonomy 2021-09-21; Shopify taxonomy @ `3ef0220`; Amazon ESCI (T3) | google.com basepages; GitHub `Shopify/product-taxonomy`; `amazon-science/esci-data` | 5,595 categories; 14,606 Shopify categories; 1,814,924 products | `30039729880ec5ac…`; `c03450e89fb1d3be…`; products `25124442d064d64b…` (= LFS) | ESCI Apache-2.0; Shopify MIT; GS1 GPC and WDC not used |
| ChEBI release 255 (T4) | `ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo.gz` (2026-09-09) | 218,822 terms; 62,119 3-star | `c2765eb4599fbcbb…` | CC BY 4.0 per file header |
| EuroVoc 4.24 SKOS core; MultiEURLEX English (T6) | Publications Office Cellar; HF `coastalcph/multi_eurlex` @ `2020d035` | 7,515 descriptors; 55,000 / 5,000 / 5,000 acts | zip `91bdb24e833ba431…`; train `0cb56e8342ee870c…` | EU reuse policy; CC BY-SA 4.0; ≤ 600-character excerpts in items |
| T5 synthetic enterprise glossary | generated (seed 20261005) | 4,000 terms | ontology `326d01322016e373…` | contamination-free by construction |
| C6 synthetic private library (T2) | generated (benchmark v1) | 414 symbols; 2,950 / 2,018 docs | — | leakage audit: 0 held-out symbols in train |
| SNOMED CT, UMLS, MIMIC-IV | **not on this machine**; T1 runs as open-clinical (MeSH + PubMed) | — | — | swapped in only if the author supplies the files; MIMIC never committed |

## 4. Frozen holdouts (one per track; hashed before any run that could see them)

| Track | Held-out set | sha256 | Frozen at |
|---|---|---|---|
| C3 general (WordNet) | 4,105 concepts / 5,900 entries, node- and alias-disjoint | `7f2462ed8f6eae9b37de050d7f535a9c9908dcc572ea68481c6526c4f6b73ced` | C3 v2 @ `724219a`; re-verified identical in the SmolLM2 (`57e4e23`) and Qwen2.5 (`ee184d2`) host corpora, with 0 held-out spans in train |
| T1-open clinical (MeSH) | 1,976 entries (702 chosen + 1,274 closure; 1,125 hub entries ineligible) | `1c477afda9822b260596c046fcdd3ca7205d9a5c0e22b44892d0cc1870ddcd88` | T1 v1 @ `cfbb50e` (re-frozen from the full config before any training; the slice pin `fd4244ea…` is stale, see gates.md) |
| T3 product | 182 concepts / 225 entries | `df7acfde37571d70c0e8cdc269b5918d6d7d82dd809c6ef627dc20676645dc62` | T3 v1 @ `16176c2` |
| T3 synthetic zero-shot set | 300 concepts | `3f0e64f04acf60f4bcb0cb712d987753120720e6773863522d4e85c67b784c92` | same |
| T4 chemistry | 429 concepts / 551 entries | `b58e504f8b060726a6c5b910e494f473ff469b0dabcdbcb22b7c9ea04f39861b` | T4 v1 @ `16176c2` |
| T4 synthetic zero-shot set | 299 concepts | `629b185b544e1bdf6fe48f66bfee4419e52df442607ade913e977ef1e3e79985` | same |
| T5 enterprise glossary | 360 concepts / 360 entries (fixed by the generator) | `e7313dcece6d0b7893784a26c80b4b39205269e90e0781340163fefca850243f` | T5 v1 @ `16176c2` |
| T5 synthetic zero-shot set | 200 concepts | `407388893336b53e78776e111883f46162b10226d58439f21ec09a13c25211bc` | same |
| T6 legal | 186 concepts / 189 entries | `8148fa06fe76929fb18a3d6bed7694992a671704cd7b774aa23aa135a634d75d` | T6 v1 @ `16176c2` |
| T6 synthetic zero-shot set | 300 concepts | `6f9243110d1c27ceb511c5de81383308944b4a770f6b46d2c8d1d2170a0cdbcd` | same |
| T2 developer tools | 57 held-out symbols (C6 v1) | no hash in the summary (**open item**: hash before the T2 track corpus is built) | — |
| E2 | node-disjoint splits regenerated from seeds 11, 22, 33 (test 20%, validation 10%, 6,000 concepts per family, `selection_seed` 5) | determined by config + seed | E2 v2 @ `07a7315` |
| E0 | synthetic teachers, seeds 101, 202, 303; 80–100 held-out concepts per teacher | determined by config + seed | E0 @ `2cc5d0b`, `9986d72` |

## 5. Commit per result

| Result | Run folder | Commit (manifest) | Dirty | Device | Reproduced on `bhux-tiny` | Report |
|---|---|---|---|---|---|---|
| E0 development (D0.1–D0.3; G1) | `experiments/e0-synthetic-identifiability/runs/e0-development` | `2cc5d0b` | no | CPU, 1 thread | MATCH at 1e-12 (8,100 + 154 cells) | run `report.md`; gates.md G1 |
| D0.2 M3 default (parent fallback, synced parents) | `…/runs/d02-parent-sync` | `9986d72` | no | CPU, 1 thread | MATCH (576 + 115 cells) | run `report.md`; gates.md |
| D0.2 parent routing | `…/runs/d02-parent-routing` | `bf5f535` | no | CPU | MATCH | — |
| D0.2 parent routing v2 | `…/runs/d02-parent-routing-v2` | `01e8090` | **yes** | CPU | MATCH | — |
| E2 mapping × operator frontier | `experiments/e2-mapping-operator-frontier/runs/v2` | `07a7315` | no | RTX 3090 | run on `bhux-tiny` | run `report.md`; gates.md D2. **Note:** gates.md cites commit `57e4e23` for this run, but the manifest records `07a7315` (the run-start state). The manifest is authoritative; gates.md should be corrected |
| B6 host memory/throughput | `experiments/b6-host-memory/runs/3090-v1` | `2ad45e5` | **yes** | RTX 3090 | re-run at HEAD: identical memory, tokens/s within 1% | R0 |
| B8 probe validation | `experiments/b8-probe-validation/runs/v1` | `e2078a4` | **yes** | RTX 3090 | LAMBADA, WiC identical; Spearman within 0.0035 | R0 |
| B8 PTQ | `experiments/b8-probe-validation/runs/v1-ptq` | `8e83d64` | **yes** | RTX 3090 | re-run at HEAD: identical PPL | R0 |
| WP-probe validation | `experiments/wp-probe-validation/runs/v1` | `34a7909` | no | RTX 3090 | — | run `report.md` |
| B10 sparse vs full generation | `experiments/b10-benchmarks/generation.{md,json}` | `05073c3` | no | CPU (4 threads) + shared GPU | — | `generation.md` |
| C3 general corpus | `experiments/c3-general-corpus/runs/v2` | `724219a` | no | CPU | built on `bhux-tiny`; 0 skipped documents | run `report.md` |
| C3 host corpus SmolLM2 | `…/runs/host-smollm2-v1` | `57e4e23` | no | CPU | — | run `report.md` |
| C3 host corpus Qwen2.5 | `…/runs/host-qwen2.5-v1` | `ee184d2` | no | CPU | — | run `report.md` |
| T1-open corpus | `experiments/t1-open-clinical/runs/v1` | `cfbb50e` | no | CPU | — | run `report.md` |
| T3–T6 track corpora | `experiments/t{3,4,5,6}-*/runs/v1` | `16176c2` | no | CPU | — | run `report.md` per track |
| C5 judge calibration v1 / v2 | `experiments/c5-judge-calibration/v{1,2}` | v2 `de4e512` | no | Claude (`claude-opus-5-5`) | v2: 30/30, κ 0.926 (v1 30/30, κ 0.812) | R0; gates.md |
| C6 developer-tools benchmark | `experiments/c6-devtools-benchmark/v1` | (no manifest) | — | CPU | — | — |
| Frozen-host 01a–01c protocol-2 re-runs | `experiments/01b…/runs/stage-b-reduced-v2`, `experiments/01c…/runs/*-v2` | `f9d49d8`, `5762dca` | **yes**, except `01c…/taxonomy-distribution-development-v2` | CPU | — | proposal §2; audit.md |
| S0 sanity pilot | `experiments/e4-small-lm/…` | *pending* | | | | R1 |
| E1, E3 | *pending* (`e3…/runs/hrr-v1` is from the old machine and untrusted) | | | | | R2 |
| E4 D4.0–D4.9, D4.1 | *pending* | | | | | R3 |
| E4.4/E4.3/E4.5 | *pending* | | | | | R4 |
| E5 | *pending* | | | | | R5 |
| E7 | *pending* | | | | | R6 |
| E8 | *pending* | | | | | R7 |

Figures and tables in `manuscript/` are regenerated by `manuscript/figures/make_figures.py`. Its output is byte-deterministic (checked by two consecutive runs). `figures/sources.json` records the sha256 of every input file and the manifest commit of every run behind each figure and table.

## 6. Compute ledger (GPU-hours, RTX 3090)

Planned figures come from [execution.md](../resources/plan-improvement/execution.md). **Measured** wall clock is not stored in the committed run folders for the blocks done so far. It is in the B11 queue logs (`.jobs/`), which this package does not read. The "measured" column is therefore left for the orchestrator to fill from those logs. Numbers below are not estimates of what was spent.

| Order | Block | Planned GPU-h | Measured GPU-h | Status |
|---:|---|---:|---:|---|
| — | B6 host memory/throughput, B8 probes and PTQ, WP-probe validation (incl. R0 re-runs) | ≈ 1 (B8 reproduction) | *fill from logs* (WP-probe: probe time 103.5 s + 72.8 s per `results.json`) | done |
| — | C3 / host / T1 / T3–T6 corpus builds, E0, C5 | 0 (CPU) | 0 | done |
| 0 | B8 reproduction (runbook Step 3) | ≈ 1 | *fill* | done |
| 1 | S0 sanity pilot (13 runs) | ≈ 6 | *fill* | running |
| 2 | D2 E2 frontier (`runs/v2`) | ≈ 10–20 | *fill* | done |
| 3 | D1 E1 contextual (`runs/v1`) | ≈ 10–20 | | queued |
| 4 | D3 E3 developmental (`runs/v2`) | ≈ 10–20 | | queued |
| 5 | D4.0 shake-out | ≈ 10 | | todo |
| 6 | D4.8 early C0 baselines | ≈ 15 | | todo |
| 7 | D4.7 ℓ_min feasibility | ≈ 12 | | todo |
| 8 | D4.9 50M screen | ≈ 50 | | todo (after G2) |
| 9 | D4.1 125M convergence | ≈ 65 | | todo |
| 10 | D4.4 continued pretraining | ≈ 55 | | todo (after G3) |
| 11 | D4.3 quantization + compression | ≈ 10 | | todo |
| 12 | D5 | ≈ 30 | | todo |
| 13 | D4.5 T1-open | ≈ 25 | | todo |
| 14 | D7 self-authoring | ≈ 40 | | todo |
| 15 | D8 tracks T2–T6 | ≈ 45 | | todo |
| | **Committed total** | **≈ 400** | | |
| — | Escalation tiers X1–X4 | ≈ 190 / 700 / 350 / 600 | — | not committed; author approval at G6 |

Judge calls (C5, D5.3, D7.1) use Claude and no local GPU. Recorded spend: C5 v1 USD 2.60 (headless CLI); v2 used Claude Code subagents.

## 7. Known reproducibility caveats

- Two manifests (B6 `2ad45e5`, B8 PTQ `8e83d64`) record dirty trees whose fixes landed later. Their numbers match the code now on `main` (R0).
- D0.2 parent-routing-v2 (`01e8090`) and three of the four 01b/01c protocol-2 re-runs (`f9d49d8`, `5762dca`) also record dirty trees. The D0.2 run reproduces anyway; the protocol-2 re-runs were not part of R0.
- GPU training is not bit-deterministic (index-add backward uses atomics). GPU comparisons use tolerance 1e-3 or the per-report tolerances in R0.
- Individual LLM-judge verdicts are harness-specific (v1 headless CLI vs v2 subagents). Agreement statistics are comparable. Each study records its harness.
- The E2 commit cited in gates.md (`57e4e23`) differs from the manifest (`07a7315`); see §5.
