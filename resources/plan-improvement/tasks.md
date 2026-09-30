# Task list and schedule

Dependency-ordered plan for implementing and running the program in [proposal.md](proposal.md), [formulation.md](formulation.md), [experiments.md](experiments.md), with the fixes from [audit.md](audit.md) placed where they must land. The answers in [clarifications.md](clarifications.md) (2026-09-30) are applied throughout.

**Staffing and hardware.** Claude implements, runs and reports; the author reviews at the gates (G0–G6) and owns the decisions there. Everything runs on the local machine: one NVIDIA RTX 3090 (24 GB; also drives the desktop, so ≈ 22 GB usable), 12 CPU cores, 30 GB RAM, ≈ 125 GB free disk. Effort is given in person-day-equivalents (pd) as a size measure for implementation work; the **binding constraint is the single GPU** (≈ 3,200 GPU-hours in total, [experiments.md](experiments.md) §Compute), so the calendar follows the GPU queue. No external deadline.

**Repository policy.** Commit to `main` as work lands and push to `origin/main`; no feature branches or PRs. Every commit keeps the test suite green. Fixes to `src/vsa_embed` and its tests are allowed; fixed behaviour gets new option or family names so recorded runs still replay. `resources/models-main` (BERTHA) stays an untouched reference.

## How to read

- **ID:** A = audit fixes, B = infrastructure, C = data / access / literature / evaluation protocol, D = experiments (D0–D8 = E0–E8), W = documentation and write-up.
- **Depends on:** hard prerequisites. **Impacts:** what a task changes or invalidates downstream — the reason it sits where it does.
- **Status:** all tasks are `todo`. Update in place; do not renumber.

## Critical path

```text
A1 ─► A6 ─► D2 ──┐
B1 ─► B5 ─► B7 ─► B11 ─► D4.0 ─► D4.7 ─► D4.1 ─► G3 ─► D4.4 ─► D7 ─► W3
B1 ─► B4 ─► D0.2 ─► D3 ─┘                                 └─► D4.5 / D5 / D8
C3 (corpus + frozen holdout + cardinality) ─► D4.0
```

Anything that delays A1–A6 delays every WordNet experiment, because E1–E3 reuse the audited runners and metrics. Anything that delays B5 → B7 → B11 → C3 delays E4, the core claim. After G3 the GPU queue itself is the critical path.

## Decision gates

| Gate | When | Decision | Inputs |
|---|---|---|---|
| G0 | end of week 3 | freeze the M1/M3/M4 interfaces; audit fixes merged | A1–A5 on `main`, C1 literature note |
| G1 | end of week 7 | mechanisms recover planted structure → proceed to E1–E3; else fix | D0 |
| G2 | end of week 12 | operator family, mapping rank, growth policy, sizes of the matched controls, measured throughput; pre-register E4 | D1, D2, D3, C4 |
| G3 | end of week 18 | E4-125M gate → core claim confirmed or narrowed; order of the remaining queue | D4.1, D4.7 |
| G4 | end of week 25 | explainability and zero-shot claims (LLM-graded estimates) | D5, D4.5 |
| G5 | end of week 28 | self-authoring claim | D7 |
| G6 | end of week 32 | application-track claims; whether 350M runs all four conditions | D8 |

## Phase 0 — audit fixes (weeks 1–3)

Must land before any runner from `src/vsa_embed/experiments` is reused. Every fix adds a **new option or family name**; existing names keep their semantics so recorded runs stay reproducible.

| ID | Task | Depends on | Effort | Output / acceptance | Impacts |
|---|---|---|---|---|---|
| A1 | Evaluation validity (audit F1, F2, F5, F7): target-disjoint `neighborhood_disjoint` split with seen/unseen strata; `validation_fraction` in `split_edges`; candidates named in config, selection on validation only; shared candidate set passed to every metric; mid-rank ties + finite-prediction assertion; relation accuracy reported only for parametric models; rank-loss positives by target node | — | 4 | tests: neighbourhood targets disjoint, selection never reads test rows, tie rule, rank loss = 0 for a perfect predictor with duplicate targets | invalidates every recorded 01b/01c number → A6; changes the numbers quoted in proposal §2 |
| A2 | Control fairness (F8, F9): `diagonal_permuted_control` (non-commuting basis), `low_rank_matched` (rank chosen to match parameters), `hrr_identity` (unit role, spectrum ≈ 1); derangement shuffle; summarizers refuse a comparison without a shuffled control for each compared family | — | 3 | tests: basis changes the control output; parameter counts equal; ≥95% labels changed by the shuffle | E2's operator ladder must include both `hrr` and `hrr_identity` to separate family from initialization |
| A3 | Gates and provenance (F4, F10): `promotion_eligible` defaults to False everywhere (update the test that relies on the True default); manifest gains git SHA, dirty-tree flag, package versions, device, argv, config hash; defaults resolved into `resolved_config.yaml`; runner refuses to overwrite an existing run directory and refuses a dirty tree for promotion-eligible runs; summarizer prints which design-gate items are not implemented | — | 2 | test: manifest schema v2 complete; second run into the same directory fails | every later run folder; stale-run marking in A6 |
| A4 | Statistics helpers (F6): paired bootstrap CI over seeds/examples in every summary; Wilson CI for selective accuracy; abstention threshold with a lower confidence bound option | — | 2 | summaries carry `ci_low/ci_high`; 01b gain reported as [0.007, 0.032] | gate semantics in D1–D8 |
| A5 | Splits and nits (F3, F11, F12): log realized split fractions; option to subsample edge-disjoint training to node-disjoint size; group symmetric `attribute`; `token_disjoint` option and same-token edge flag; `LowRankRelation` identity init; `fork_rng` in `make_synthetic_relations`; unitary guard for zero bins; anchor truncation/padding assertions; console script for `hierarchy_neighborhoods`; `relation_models.pt` saves the primary split's models | — | 3 | tests for each item from the audit's test-gap list | B1 inherits the normalization and accumulation conventions |
| A6 | Re-run and re-document: re-run 01b Stage B and 01c.3–01c.5 with A1–A5 into **new run IDs** (device recorded; CPU where comparability with recorded CPU numbers matters); keep the old runs and add `stale: true` notes to stage-b-smoke, 01c smoke and 01c development; **append errata sections** to the dossier READMEs under `resources/vsa-understanding/` (01c.5 low-rank claim, 01b relation-accuracy claim, 798 → 966, 05 consequences table) and a one-line status correction in the program index ("proceed now: 01c" → refuted); refresh proposal §2 numbers | A1–A5 | 3 + ≈ 1 GPU-day | regenerated reports; errata sections | E2 baselines; anything that cites 01b/01c |

## Phase 1 — infrastructure (weeks 2–8)

| ID | Task | Depends on | Effort | Output / acceptance | Impacts |
|---|---|---|---|---|---|
| B1 | `compose.py` — `FrameComposer`: CSR frame schedule, batched bind via the relation families, segment-softmax weights (M1), induced/free/hybrid concept factors (M2), `index_add_` scatter, `explain(concept)`; reads recipes in the `OntologyFactorizer` format so the E0 harness is reused; the old `compose` path stays for reproducing 01a | A5 conventions | 6 | tests: `τ → ∞` equals uniform bundle to 1e-6; segment softmax equals a loop; no per-concept parameters in induced mode; changing a held-out target changes no parameter; row norm; duplicate destinations accumulate | every D task; interface must be frozen at G0 |
| B2 | Relation adjoints: `adjoint()` per family (correlation for `hrr`, same diagonal, transposed low-rank, `Qᵀ`, autograd fallback for `bounded_residual`); `hrr_identity` shared with A2 | A2 | 2 | test: `⟨T x, y⟩ = ⟨x, Tᵀ y⟩` for every family | B4 per-usage gradients, B12 utility |
| B3 | `context.py`: P1 causal local-context encoder (depthwise conv / mean over `E[x_{t−w..t}]`); P2 sidecar hook (frame as K/V, inject at layer `ℓ+1`) | B1 | 3 | test: P1 output at `t` unchanged by tokens `> t` | D1, D4 C5, D6 |
| B4 | `developmental.py` (M3): screening from Adam state (κ, α), usage-level EMAs for candidates only, between-usage scatter, top eigenvector, split gain, permutation null, split/merge/prune/allocate, budget and cooldown, cards log; sibling routing for unobserved usages | B1, B2 | 8 | tests: gain ≥ 0 and = 0 when all usages agree; per-usage gradients equal per-sample autograd; permutation null accepts ≤ 1% on pure noise; split duplicates optimizer state and frames; budget respected | D0.2, D3, D4 C6, D7 (allocation of authored fillers) |
| B5 | `span_channel.py` (M4): alias tables (WordNet lemmas first; SNOMED/UMLS, API, product, ChEBI, glossary, EuroVoc adapters plug in via C2/C6/C7), Aho–Corasick longest-match on **character offsets**, holdout list removal, per-tokenizer span table `(s, e, concept, confidence)` for GPT-2 BPE, SmolLM2 and Qwen2.5, `ℓ_min`, last-subtoken injection, gate, semantic-prediction head; **`SpanCardinalityReport`** per tokenizer × `ℓ_min ∈ {1, 2, 3, 4+}` (distinct concepts, occurrences, covered-token fraction, frequency histogram, implied channel cost) | B1 | 7 | tests: causal-leak test; held-out aliases never linked; span ends align with each tokenizer's offsets; cardinality counts equal a brute-force count on a sample | B7 dataloader format; C3 must re-link if the linker changes → freeze linker version before C3; D4.7 and every E8 feasibility verdict |
| B6 | `integrations/transformers.py`: `inputs_embeds` hook for HF causal LMs, frozen / LoRA host, tied-output helper, chunked cross-entropy (Qwen2.5's 151,936-token vocabulary); smoke on the cached GPT-2, SmolLM2-135M, Qwen2.5-0.5B and on SmolLM2-360M (download); peak-memory table on the 3090 per host × micro-batch | B5 | 4 | test: hooked forward equals unhooked forward when the channel is off; every host trains one step within 22 GB | D4.4, D4.5, D5.4, D7 |
| B7 | From-scratch LM training harness: GPT-2-small/medium-like configs (50M / 125M / 350M), GPT-2 BPE, bf16 autocast, fused AdamW, SDPA/flash attention, activation checkpointing ≥ 125M, gradient accumulation, channel on/off, matched-parameter control sizing, stratified loss logging, tokens-to-loss, run-folder contract, checkpoint every 30 min with bit-exact resume (data order, RNG, optimizer) | B5 | 8 | 50M dry run reproduces a known loss curve without the channel; kill-and-resume gives an identical loss curve; throughput overhead measured | D4; sizing of C1/C2 controls waits for G2 (γ from D3) |
| B8 | Evaluation harness: stratified loss (frequency bin × span length; inside vs after span), probes in **both prompting and linear-probe form** (WiC, all-words WSD, CARD-660, Rare Words, BLESS/HyperLex, LAMBADA), `torchao` weight-only INT8/INT4 PTQ (dictionary in FP16 or quantized), table-compression variant, faithfulness ablations | B7 checkpoint format | 8 | each probe validated on the untouched SmolLM2-135M; PTQ retains published perplexity within tolerance | D4.1–D4.5, D5, D8 |
| B9 | Synthetic teachers for E0: contextual teacher with context prototypes, planted polysemy / relation sub-types, rank-`k*` bilinear teacher; extends `ontology_factorization.py` and `make_synthetic_relations` | B1 | 3 | teachers seeded through generators, no global RNG side effects | D0 |
| B10 | CI and quality: property tests, `torch.compile` smoke (optional), latency/memory benchmarks for sparse vs full generation | B1–B5 | 3 | CI green; benchmark table in the repo | proposal's efficiency accounting |
| B11 | Local GPU job queue: one GPU job at a time with priorities (core > backfill), CPU jobs in parallel, automatic resume after interruption or reboot, per-job log and status file, disk-budget check before a job starts (checkpoints: last + final weights only) | B7 | 2 | a queued job survives a killed process and a reboot; queue status readable from one command | every GPU task from D1 on |
| B12 | `authoring.py` (M5): surprisal candidate discovery, constrained authoring template (closed relation vocabulary; open ablation), sample pooling, held-out utility `U_{j,e}` with bootstrap bounds, authoring cards, round driver with locality and drift audits, compute accounting for the compute-matched control | B4, B5, B6 | 7 | tests: `U_{j,e}` equals the finite-difference loss change for small `β`; authoring and validation contexts disjoint; gold-audit edges never read by acceptance | D7, D8 self-authoring runs |
| B13 | `judging.py`: LLM-judge harness — rubric files, blinding and randomized order/position, API judge (pinned model version) and local instruct judge (4-bit on the 3090), prompt paraphrases, verdict cache keyed by item/judge/prompt hash, Cohen's κ and calibration-set agreement; hard rule: items flagged `mimic_derived` go to the local judge only | — | 3 | test: a `mimic_derived` item cannot reach the API client; cached verdicts replay | D5.3, D7.1 |

## Phase C — data, access, literature, evaluation protocol (start week 1)

| ID | Task | Depends on | Effort / timing | Output | Impacts |
|---|---|---|---|---|---|
| C1 | External literature search (web search permitted) and positioning note: knowledge-injected LMs, multi-sense induction, growing networks, gradient-conflict statistics, attention-as-binding, factorized/compositional embeddings, multi-token word modelling, knowledge-capacity scaling, CoLLEGe; for M5: ontology learning from text, LM-based KG construction, self-training / self-improvement loops, model collapse | — | 5 pd, weeks 1–3 | `related-work.md` with claims the proposal may and may not make | may change M1/M3/M4/M5 before G0; blocks any novelty claim |
| C2 | Clinical data (access held): locate SNOMED CT, UMLS and MIMIC-IV / MIMIC-IV-Note locally, record release versions and data-use terms in the `LinkerManifest` (MIMIC-derived text never leaves the machine); SNOMED/UMLS ontology adapter (concepts, attributes, ICD-10/RxNorm/MeSH mappings); PubMed abstracts + PMC-OA subset; ICD-coding labels from MIMIC-IV; BLURB subset | B5 | 5 pd, weeks 2–8 | T1 corpus, ontology, cardinality table, frozen holdout | D4.5, D5.3, D5.4, D7 clinical reading corpus |
| C3 | General corpus and data preparation: FineWeb-Edu subset (1B / 2.5B tokens; 7B only if the optional 350M run is approved), tokenized while streaming with GPT-2 BPE plus 0.5B-token slices for the SmolLM2 and Qwen2.5 tokenizers; disk budget ≤ 60 GB for all token streams; corpus linked with the frozen B5 linker; concept frequency table; **cardinality tables**; **held-out concept list frozen and hashed** (shared by D1–D5, D7); SemCor / WiC / WSD / CARD-660 / Rare Words / BLESS / HyperLex / LAMBADA fetched | B5 (frozen linker) | 4 pd, weeks 4–7 | `LinkerManifest`, holdout hash, `SpanCardinalityReport`, dataset cards | D4.0 cannot start without it; changing the holdout later invalidates D1–D3 |
| C4 | Throughput benchmark on the 3090: tokens/s and peak memory for 50M/125M/350M and each host, with and without the channel → measured GPU-hour table replacing the 2.5e13 FLOP/s planning figure; re-order the queue | B7, B6 | 1 pd, week 7 | revised compute table in experiments.md | G2; whether 350M stays at 3 seeds |
| C5 | LLM-judge protocol (replaces rater recruiting): rubrics for neighbour relatedness, edge explanations, M3 cards and authoring cards; 40-item pre-registered sets per study; author-labelled calibration set (≈ 30 items); judge models pinned. Clinician study later with the identical items (W4) | B13 | 2 pd, weeks 8–10 | protocol file and calibration results | D5.3, D7.1 |
| C6 | Developer-tools track (T2) benchmark: synthetic private API generator (schemas → frames, documentation corpus, held-out symbols with contamination-free names) plus real-library schemas | B5 | 5 pd, weeks 8–14 | benchmark v1 with leakage audit and cardinality table | D5.4, D7 dev-tools reading corpus, D8.2 |
| C7 | Remaining application tracks: T3 product (Google product taxonomy / GS1 GPC; ESCI, WDC), T4 chemistry (ChEBI, PubChem synonyms and descriptions), T5 enterprise (synthetic private-glossary generator), T6 legal (EuroVoc, EUR-Lex defined terms); T7 multilingual (Open Multilingual Wordnet, FineWeb-2) prepared last. Per track: adapter, corpus, cardinality table and feasibility verdict, frozen holdout, track tasks | B5, B8 | 3 pd per track, weeks 14–28 | per-track data cards | D8.3–D8.7 |

## Phase D — experiments

### D0 — E0 synthetic identifiability (weeks 4–7, CPU)

| ID | Task | Depends on | Effort | Gate |
|---|---|---|---|---|
| D0.1 | Contextual composition on contextual and static teachers; learners M0, feature salience, M1-P0, M1 with `q`, free upper bound; composition-disjoint × held-out contexts | B1, B9 | 2 + CPU-days | M1 recovers where M0 cannot; no harm on the static teacher; `τ → ∞` test |
| D0.2 | Split detection on planted polysemy and relation sub-types; policies none / random / coherence-only / M3 / second-order (small) / oracle; false-split rate at convergence | B4, B9 | 3 + CPU-days | precision & recall ≥ 0.9, ARI ≥ 0.8, false splits ≤ 5% |
| D0.3 | Factored-mapping transfer with a rank-`k*` teacher; options (a)/(b)/(c); rank sweep | B1, B9 | 1 | (a) fails, (b)/(c) transfer; rank curves for D2 |
| G1 | Decision: proceed to D1–D3 or fix mechanisms | D0 | — | — |

### D1 — E1 contextual composition on frozen anchors (weeks 8–9, ≈ 60 GPU-h)

| ID | Task | Depends on | Effort |
|---|---|---|---|
| D1.1 | Extend `contextual_anchors` to sentence contexts (SemCor), layer audit, union frames per lemma; hosts GPT-2 and Qwen2.5-0.5B | A6, B1, B3, C3 | 2 |
| D1.2 | Runs: M0 per lemma, M0 per sense (oracle), M1 with P1/P2 `q`, operator ablation; node- and sentence-disjoint splits; 3 seeds | D1.1 | ≈ 60 GPU-h |
| D1.3 | Report with variance explained, MRR, sense alignment vs MFS; CIs | D1.2 | 1 |

### D2 — E2 mapping × operator frontier (week 10, ≈ 60 GPU-h)

| ID | Task | Depends on | Effort |
|---|---|---|---|
| D2.1 | Grid config: mapping {binary, salience, M2 induced k ∈ {4,8,16,32}, hybrid+δ, contextual} × operator {`hrr`, `hrr_identity`, `unitary_hrr`, `diagonal`, `low_rank_matched`, `bounded_residual`, `orthogonal`, `random_fixed`, `untyped`}; shuffled control per family | A1–A6, B1, B2 | 2 |
| D2.2 | Second ontology family adapter: SNOMED CT subset (from C2), MeSH as the open alternative | C2, C3 | 2 |
| D2.3 | Runs on 01b contextual anchors + D1 multi-context anchors; data-to-threshold curves; parameter groups | D2.1, D2.2 | ≈ 60 GPU-h |
| D2.4 | Report; select operator family and rank for E4 | D2.3 | 1 |

### D3 — E3 developmental recovery on WordNet (weeks 11–12, ≈ 100 GPU-h)

| ID | Task | Depends on | Effort |
|---|---|---|---|
| D3.1 | Sense-collapse and relation-collapse adapters; masked-ontology ground truth | C3 holdout, A5 | 2 |
| D3.2 | Policies: none, uniform enlargement, random splits, coherence-only, M3, minimal 01d entmax stem-cell pool (relations only) | B4 | 3 |
| D3.3 | Runs on the D2-selected configuration; 3 seeds; growth curves, ARI, false-split rate, cards | D3.1, D3.2, D2.4, D0.2 pass | ≈ 100 GPU-h |
| D3.4 | Report; choose growth policy and budget γ → sizes of the matched controls C1/C2 in E4 | D3.3 | 1 |
| G2 | Decision and **pre-registration** of E4 (`preregistration.md` in the E4 experiment folder: frozen configs, holdout hash, control sizes, gates, measured throughput), committed before D4.1 starts | D1–D3, C4 | 1 |

### D4 — E4 small-LM training, both regimes (weeks 8–20; 350M in weeks 33–36)

| ID | Task | Depends on | Effort / GPU | Notes |
|---|---|---|---|---|
| D4.0 | Harness shake-out at 50M with C0–C3 on 1B tokens; verify stratified metrics, tokens-to-loss, overhead, resume; freeze the training recipe | B7, B8, B11, C3 | 3 pd + ≈ 30 h, weeks 8–9 | results are not for the gate |
| D4.8 | Backfill: no-channel baselines C0 at 50M / 125M / 350M × 3 seeds and C0' on SmolLM2-135M / 360M / Qwen2.5-0.5B × 3 seeds, lowest queue priority | D4.0 (frozen recipe) | ≈ 310 h, weeks 9–13 and later gaps | independent of every design decision; a recipe change after D4.0 invalidates them |
| D4.7 | Span-cardinality feasibility: cardinality tables for all three tokenizers on the general corpus; 50M C0/C1/best at `ℓ_min ∈ {1, 2, 3}`, 3 seeds | C3, D4.0 | 1 pd + ≈ 30 h, week 13 | fixes `ℓ_min` per track; feeds E8 |
| D4.9 | 50M screen: C1–C7 + C8 operator ablation, 3 seeds | G2, D4.0 | 1 pd + ≈ 140 h, weeks 13–14 | decides whether C4/C7 go to 125M |
| D4.1 | 125M core: C1, C2, C3, C5, C6 (+ C4/C7 if D4.9 separates them) + C8 {`random_fixed`, `untyped`, `diagonal`}, 3 seeds, 2.5B tokens; C0 from D4.8; stratified metrics; held-out concept stratum | G2, D4.9, C4 | 4 pd + ≈ 510 h, weeks 14–18 | **primary endpoint of the program**; C8 drops to 2 seeds only if the queue slips, reported as such |
| G3 | Decision on the core claim | D4.1 | — | |
| D4.4 | Continued-pretraining track (E4.6): SmolLM2-135M (frozen vs LoRA), SmolLM2-360M, Qwen2.5-0.5B; C1, C2, best channel condition vs C0' (D4.8); 3 seeds, 0.5B tokens | B6, G3 | 3 pd + ≈ 225 h, weeks 19–20 | required track; also feeds D5.4 and D7 |
| D4.3 | Quantization (`torchao` INT8/INT4) and table-compression variant on D4.1 and D4.4 models (D4.2 later) | B8, D4.1 | 3 pd + ≈ 20 h, week 21 | |
| D4.5 | Clinical track T1 (E8): SNOMED CT / UMLS linker on PubMed + PMC-OA + MIMIC-IV notes; 125M from scratch C0, C1, C2, best × 3 seeds; SmolLM2-360M continued pretraining C0', C2, best × 3 seeds | C2, G3 | 4 pd + ≈ 350 h, weeks 22–24 | flagship application; unconditional |
| D4.2 | 350M top-4 conditions (C0 from D4.8), 3 seeds, 2.5B tokens; optional best vs C0 at 7B tokens only if approved at G6 | G3, G6 | 2 pd + ≈ 525 h, weeks 33–36 | last in the queue; after a failed G3 reduced to C0 vs best |
| D4.6 | E4 report with CIs, curves, parameter groups, overhead, cardinality tables, both regimes | D4.1–D4.5, D4.7 | 3 pd | interim version at G3 (W2), final after D4.2 |

### D5 — E5 explainability and zero-shot (weeks 21–25, ≈ 100 GPU-h)

| ID | Task | Depends on | Effort |
|---|---|---|---|
| D5.1 | Faithfulness: top-k / random / bottom-k edge ablations | B8, D4.1 | 2 |
| D5.2 | Sense alignment on jointly trained models | D4.1 | 1 |
| D5.3 | Rating study, **LLM-graded** (neighbours of rare SNOMED/ICD codes, edge explanations, M3 cards); pre-registered `n` and test; judge agreement and calibration reported; items and rubric kept for the later clinician study (W4) | C5, B13, D4.5, D3.3 | 3 |
| D5.4 | Zero-shot insertion via the span channel under the 02 protocol; T1 clinical and T2 dev-tools scenarios now, other tracks in D8; baselines incl. definition encoder and graph embedding + projection | C6, B6, D4.1, D4.4, D4.5 | 5 |
| D5.5 | Frequency disentanglement (quantitative version of the paper's t-SNE claim) | D4.1 | 1 |
| G4 | Decision on explainability and zero-shot claims (labelled LLM-graded estimates) | D5 | — |

### D6 — E6 deep injection and readout (week 37, ≈ 60 GPU-h)

| ID | Task | Depends on | Effort |
|---|---|---|---|
| D6.1 | P2 sidecar runs at `ℓ ∈ {L/4, L/2}` on the 125M best configuration | B3, D4.1 | 2 + ≈ 40 GPU-h |
| D6.2 | Relation readout head (attention with relation probe + unbinding + cleanup) vs linear probe; depth-2 compositions | D6.1 | 4 |
| D6.3 | Report against the 03-style gate | D6.2 | 1 |

### D7 — E7 self-authored ontologies (weeks 26–28, ≈ 350 GPU-h)

| ID | Task | Depends on | Effort / GPU |
|---|---|---|---|
| D7.0 | Reading corpora and hidden gold subgraphs: general (20% of WordNet concepts masked), dev-tools (C6 synthetic library), clinical (held-out SNOMED subgraph; PubMed + MIMIC notes, local only); teacher author (≈ 7B instruct, 4-bit) set up | B12, C2, C3, C6 | 3 pd |
| D7.1 | Authoring quality on SmolLM2-135M/360M, Qwen2.5-0.5B and the teacher: edge P/R/F1 vs gold, discovery recall, LLM-judged plausibility of non-gold edges; baselines Hearst patterns and random frames | D7.0, B13, C5 | 2 pd + ≈ 40 h |
| D7.2 | Self-improvement, rounds 0–3: gold → host, self → self, self without verification, teacher → host, random frames, compute-matched continued pretraining; SmolLM2-360M × 3 seeds, best two + compute-matched on Qwen2.5-0.5B | D7.1, D4.4 | 3 pd + ≈ 200 h |
| D7.3 | Cross-authoring: best authored ontology as the linker ontology for from-scratch 50M and 125M (curated vs authored vs none) | D7.2, D4.1 | 1 pd + ≈ 110 h |
| D7.4 | Report: authoring cards, per-round curves, compute accounting | D7.3 | 2 pd |
| G5 | Decision on the self-authoring claim | D7 | — |

### D8 — E8 application tracks (T1 = D4.5; T2–T6 weeks 29–32, ≈ 380 GPU-h)

Each track: cardinality table → feasibility verdict; from-scratch 50M C0, C1, C2, best × 3 seeds on a 50/50 domain/general mix; SmolLM2-135M continued pretraining C0', C2, best × 3 seeds; a self-authoring run where gold allows auditing; track tasks and the E5.4 zero-shot scenario.

| ID | Track | Depends on | Effort / GPU |
|---|---|---|---|
| D8.2 | T2 developer tools | C6, G3, D7 recipe | 2 pd + ≈ 80 h |
| D8.3 | T3 product catalogues | C7-T3, G3 | 2 pd + ≈ 80 h |
| D8.4 | T4 chemistry | C7-T4, G3 | 2 pd + ≈ 80 h |
| D8.5 | T5 enterprise glossaries (synthetic) | C7-T5, G3 | 2 pd + ≈ 60 h |
| D8.6 | T6 legal / regulatory | C7-T6, G3 | 2 pd + ≈ 80 h |
| D8.7 | T7 multilingual pivot (stretch, after D6) | C7-T7, D6 | 2 pd + ≈ 80 h, not in the total |
| D8.8 | Cross-track report: gain vs span cardinality and ontology depth, feasibility verdicts, which domains pay off | D4.5, D8.2–D8.6 | 2 pd |
| G6 | Decision on application claims; approve or skip the optional 350M × 7B run | D8.8 | — |

## Phase W — documentation and write-up

| ID | Task | Depends on | Effort |
|---|---|---|---|
| W1 | Update the dossier index and roadmap for the new program (status of 01c, M3 replacing 01d, links to this folder) | A6 | 1 |
| W2 | Interim report at G3 (methods, E0–E4-125M results, decision) | D4.1 | 3 |
| W3 | Paper draft: framing (C1), formulation, E0–E8 results, operator finding, applications; working name CG-VSA kept until here | G6, D4.2 | 10, weeks 36–42 |
| W4 | Clinician rating study on the D5.3/D7.1 item sets (replaces the LLM-graded estimates for any external claim) | D5.3, clinicians available | later, not scheduled |

## Week-by-week schedule

GPU capacity ≈ 140 GPU-h per week at 85% utilization; CPU work runs alongside.

```text
week   A / B / C (implementation, data)                   GPU queue (one job at a time)                 gate / W
 1     A1 A2 A3, B1, C1                                   —
 2     A4 A5, B1 B2 B9, C1 C2                             —
 3     A6, B1 done, B3 B4, C1 done, C2                    A6 re-runs                                    G0, W1
 4     B4 B5, C2, C3 prep; D0.1 D0.3 (CPU)                B5/B6 smoke
 5     B4 B5 done, B6, C2 C3; D0.1 D0.2 D0.3 (CPU)        B6 memory table
 6     B7 B11 B12 start, C2 C3 (holdout frozen); D0.2     B7 dry runs
 7     B7 B8 B11, C3 done, C4; D0 report                  C4 throughput benchmark                       G1
 8     B8 B10 B12, B13 C5 C6, C2 done; D1.1 D2.1 D2.2     D4.0 shake-out, D1.2
 9     B8 B12, C5 C6                                      D1.2, D4.0 done, D4.8 backfill
10     B8 done, B12 done, C5 C6                           D2.3, D4.8 backfill
11     C6; D3.1 D3.2                                      D3.3, D4.8 backfill
12     C6; D3.4, preregistration.md                       D3.3 done, D4.8 backfill                      G2
13     C6                                                 D4.7 ℓ_min feasibility, D4.9 50M screen
14     C6 done, C7-T3                                     D4.9 done, D4.1 125M core
15–17  C7-T3 T4                                           D4.1
18     C7-T4; W2                                          D4.1 done                                     G3, W2
19–20  C7-T5                                              D4.4 continued pretraining
21     C7-T5 T6                                           D4.3 quantization, D5.1 D5.2 D5.5
22–24  C7-T6, D7.0                                        D4.5 clinical track (T1)
25     D5.3 LLM grading, D5.4                             D5.4 GPU evals                                G4
26–28  C7-T7 prep                                         D7.1 D7.2 D7.3 self-authoring                 G5
29–32  D8 reports                                         D8.2–D8.6 tracks T2–T6                        G6
33–36                                                     D4.2 350M
37                                                        D6.1 D6.2
36–42  D4.6 final, D6.3, D8.8                             (D8.7 multilingual if run)                    W3
```

The GPU is under-used before week 13 by design work, so the design-independent baselines (D4.8) fill it; after week 13 it is saturated and every slip moves everything behind it. If D3 slips, D4.1 starts with C1–C3, C5 and C8, and C6 plus the matched-control sizes are appended.

## Rules that keep the phases from invalidating each other

1. **New names, not new semantics.** Fixed or added families, splits and metrics get new names (`hrr_identity`, `neighborhood_disjoint`, `low_rank_matched`). Recorded runs must still replay bit-for-bit with their old names.
2. **One holdout per track.** The held-out concept list (C3, and one per E8 track) is frozen and hashed before the first run that could see it. Changing it means re-running everything after it. M5 gold-audit subgraphs are part of the holdout.
3. **Freeze the linker before linking a corpus.** Re-linking is expensive; B5 is versioned and each `LinkerManifest` records the version, tokenizer and licence versions.
4. **Pre-register before D4.1.** Configs, holdout hash, control sizes, measured throughput and gates are written down at G2 in `preregistration.md` and committed; later changes are logged as deviations.
5. **Controls are sized last.** C1/C2 parameter counts depend on the dictionary size after D3 (γ), so they are set at G2, not at B7.
6. **Do not cite pre-A6 numbers.** Proposal §2 and the dossier are refreshed in A6/W1; the audit's changed conclusions (01c.5 low-rank, 01b relation accuracy) are corrected before any new report quotes them.
7. **Compute the operator ablation everywhere.** Every experiment report includes it; the operator question is answered by accumulation, not by a single run. If `random_fixed ≈ hrr`, reports say "compositional parameter sharing".
8. **Commit, then run.** Commit and push to `main` before any run whose results may be cited; manifests carry the SHA and a dirty-tree flag, and promotion-eligible runs refuse a dirty tree.
9. **MIMIC stays local.** No MIMIC-derived text in prompts to external APIs, in commits, or in published artifacts; the judge harness enforces it.
10. **Frozen recipe for backfill.** D4.8 baselines are valid only for the training recipe frozen at D4.0; any recipe change re-queues them.

## Schedule risks

| Risk | Effect | Mitigation |
|---|---|---|
| A1–A6 take longer than 3 weeks | delays D1–D3 and G2 | B-track continues; D4.0 does not need A |
| Measured 3090 throughput below the 2.5e13 FLOP/s plan | every GPU phase stretches | C4 measures it at week 7; cut order: optional 7B run → 350M to C0 vs best → C8 to 2 seeds (reported) → T7 → fewer E8 tracks at 50M only |
| GPU shared with the desktop; interruptions, reboots, thermal throttling | lost hours, silent slowdowns | 30-min checkpoints and bit-exact resume (B7, B11); throughput logged per job; long runs overnight-safe |
| Disk (≈ 125 GB free) fills with corpora and checkpoints | jobs fail mid-run | ≤ 60 GB token streams, raw text not kept, last + final checkpoints only, B11 checks free space before starting |
| D0.2 fails (M3 does not detect planted splits) | D3, E4-C6 and M5 allocation blocked | ship E4 without C6; M5 allocates without splitting; revisit the statistic with the second-order baseline |
| Small hosts author poor frames (D7.1) | E7 reduces to teacher → host | report authoring quality per host size; the teacher and cross-authoring conditions still answer whether authored ontologies help |
| LLM-judge scores disagree with each other or with the calibration set | D5.3/D7.1 estimates unreliable | report κ and calibration; mark results inconclusive rather than pick a judge; prioritise W4 |
| An application track's cardinality table cannot power its gate | wasted GPU on that track | feasibility verdict before any training in the track; report it as infeasible |
| Literature search finds a close prior for M1, M4 or M5 | claims narrow; possibly a baseline to add | do C1 before G0 so the change is cheap |
