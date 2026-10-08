# Execution plan — the full queue on `bhux-tiny` (from 2026-10-02)

Turns every remaining task in [queue.md](queue.md) / [tasks.md](tasks.md) into work packages, a GPU order and report deliverables, so the program can run end to end and end in reports that a manuscript can be built from. [runbook.md](runbook.md) Steps 0–5 (reproduction, blocker) come first; this file is Step 6 onwards. Status lives in [queue.md](queue.md); gate decisions in [gates.md](gates.md).

## Author decisions of 2026-10-02 (apply throughout)

1. **Clinical track: open substitute now.** SNOMED CT / UMLS / MIMIC-IV files are not on this machine. T1 runs as **T1-open**: MeSH 2026 descriptors (ontology, aliases incl. entry terms) + PubMed abstracts (corpus). Every T1 result is labelled "open-clinical (MeSH + PubMed)". SNOMED/UMLS/MIMIC are swapped in later if the files arrive; the code keeps the ontology adapter pluggable. D5.3's rare-code neighbour study uses rare MeSH descriptors; the MIMIC/ICD-coding task is not run (reported as not run).
2. **Sanity signal first, fewer GPU hours.** Before the committed ≈ 400 h, a small **S0 sanity pilot** (≈ 6 GPU-h) runs as soon as the C3 corpus exists, to see whether any signal appears at all. It is not a gate and is never pooled with gate runs. The committed plan then proceeds as written; escalation tiers X1–X4 still wait for the author at G6.
3. **Judge harness.** Calibration passes with both harnesses (v1 `claude -p`, κ 0.81; v2 Claude Code subagents, κ 0.93; both 30/30). Large studies (D5.3, D7.1, the E7 teacher author) use the lean headless `claude -p` runner of B13 (`judging.claude_cli_runner`, model pinned to `claude-opus-5-5`) from background scripts; small ones may use the subagent exchange. The harness is recorded in every study's manifest.

## GPU order (one job at a time via the B11 queue, `.jobs/`)

Priorities: lower number runs first. Estimates use the measured C4 throughput.

| Order | Block | Runs | GPU-h | Needs |
|---:|---|---|---:|---|
| 0 | B8 reproduction (runbook Step 3) | 2 | ≈ 1 | probes |
| 1 | **S0 sanity pilot**: 50M × 100M tokens, {C0, C1, C2, C3, C5} × seeds {1, 2}; SmolLM2-135M LoRA × 25M tokens, {C0', C2, C5} × seed 1 | 13 | ≈ 6 | C3, WP-host |
| 2 | D2 E2 frontier (`runs/v2`) | 1 job | ≈ 10–20 | MeSH |
| 3 | D1 E1 contextual (`runs/v1`) | 1 job | ≈ 10–20 | C3, WSD |
| 4 | D3 E3 developmental (`runs/v2`) | 1 job | ≈ 10–20 | D2 |
| 5 | D4.0 shake-out (≈ 8 short 50M runs) → freeze recipe | 8 | ≈ 10 | C3, WP-E4R |
| 6 | D4.8 early C0 baselines (50M × 300M, 125M × 500M; 3 seeds; lowest priority / backfill) | 6 | ≈ 15 | D4.0 |
| 7 | D4.7 ℓ_min feasibility (50M; C0, C1, best × ℓ_min {1, 2, 3} × 3) | 12–27 | ≈ 12 | D4.0 |
| — | **G2** pre-registration (`experiments/e4-small-lm/preregistration.md`) | — | — | D1–D3, D4.0 |
| 8 | D4.9 50M screen (C1–C7 + C1h, C3t, C1s; C8 operators; 3 seeds) | ≈ 50 | ≈ 50 | G2 |
| 9 | D4.1 125M convergence (C1, C2, best, C8 {random_fixed, untyped}; 3 seeds) | 15 | ≈ 65 | D4.9 |
| — | **G3** + W2 interim report | — | — | D4.1 |
| 10 | D4.4 continued pretraining (SmolLM2-135M/360M, Qwen2.5-0.5B × {C0', C1, C2, best} × 3; LoRA check) | ≈ 39 | ≈ 55 | G3, WP-host |
| 11 | D4.3 quantization + table compression (evals) | — | ≈ 10 | D4.1, D4.4, WP-Q |
| 12 | D5 (faithfulness, sense alignment, frequency, zero-shot) | — | ≈ 30 | D4.1, WP-E5 |
| 13 | D4.5 T1-open (50M from scratch 50/50 mix C0, C1, C2, best × 3; SmolLM2-360M CPT C0', C2, best × 3) | 18 | ≈ 25 | WP-T1, G3 |
| 14 | D7 self-authoring (authoring quality, round 0 → 1, cross-authoring at 50M) | — | ≈ 40 | D4.4, WP-E7 |
| 15 | D8.2–D8.6 tracks T2–T6 (SmolLM2-135M CPT ×3; T2 also 50M from scratch) | — | ≈ 45 | WP-C7, G3 |

Total committed ≈ 400 GPU-h, i.e. roughly 17 days of GPU if back to back.

## Implementation work packages

Each WP is built on its own branch/worktree by an agent, with tests, and merged into `main` before any run that uses it. GPU use while building is limited to short smoke tests (≤ 4 GB, ≤ 5 min) so the queue keeps the GPU.

| WP | Content | Feeds |
|---|---|---|
| **WP-E4R** | E4 analysis: aggregate run folders → stratified loss tables (per stratum, paired over seeds, bootstrap CIs, Holm), gate-item verdicts (E4 gate 1–4), data multiplier `k(L)` and power-law projections via `convergence.py`, escalation-rule verdicts, parameter groups and throughput, figures (loss curves per stratum, `k` vs tokens); `e4_plan` stages `pilot` and `shakeout` | S0, D4.x, R1, R3 |
| **WP-host** | Host corpora: the C3 linker/holdout re-linked per host tokenizer (SmolLM2, Qwen2.5) into train (≥ 130M tokens) / eval token streams with shared entry ids and the same holdout; per-host cardinality; config stages for CPT runs | S0, D4.4, D7, D8 |
| **WP-probe** | Probes for trained checkpoints (ChannelLM with the channel live during probing): prompting and linear forms; WiC, all-words WSD (Raganato ALL), CARD-660, Rare Words, BLESS, HyperLex, LAMBADA; held-out-concept probes; paired bootstrap between conditions | D4.1, D4.4, D5, D8 |
| **WP-Q** | D4.3: torchao INT8/INT4 PTQ of trained ChannelLMs (dictionary FP16 vs quantized); E4.5 table compression `P c + δ` at 4/2 bits vs equal-bytes INT4/INT2 table, ALBERT factorization, QR/hash embeddings, TT table | D4.3, R4 |
| **WP-E5** | D5.1 faithfulness (top-k / random / bottom-k edge ablations; comprehensiveness, sufficiency), D5.2 sense alignment, D5.5 frequency disentanglement, D5.4 zero-shot insertion (02 protocol: reserved synthetic concepts; property selection, entailment, paraphrase consistency, semantic-head rank; baselines matched random, surface mean, definition mean, definition encoder, à la carte, CoLLEGe-style, C2 fallback, graph embedding + projection), D5.3 item builders (neighbours of rare concepts, edge explanations, M3 cards) and the Fisher test | D5, R5 |
| **WP-E7** | M5 round driver on pretrained hosts (candidates → authored frames → held-out utility → linker + dictionary → train), Claude teacher author via `claude -p`, Hearst-pattern and OLLM-style baselines, random frames, compute-matched CPT, EntiGraph-style synthetic CPT, masked-WordNet gold audit, cross-authoring config | D7, R6 |
| **WP-T1** | T1-open data: PubMed baseline abstracts (enough for ≥ 150M GPT-2 tokens), MeSH linker (descriptor names + entry terms), frozen holdout (node- and alias-disjoint), 50/50 mix with general text, cardinality + feasibility verdict, track tasks (MeSH-tree category probe, PubMedQA/BioASQ-style cloze if obtainable), rare-descriptor neighbour items | D4.5, D5.3, R7 |
| **WP-C7** | T3 product (Google product taxonomy; product text corpus), T4 chemistry (ChEBI ontology; PubChem/ChEBI definitions), T5 synthetic private glossary generator, T6 legal (EuroVoc; EUR-Lex English): adapter → frames, aliases, corpus, holdout, cardinality, feasibility verdict, track probes; T2 from the C6 benchmark | D8, R7 |
| **WP-B10** | CI (pytest workflow file), property tests, sparse vs full generation latency/memory benchmark table | R-appendix |
| **WP-MS** | Manuscript package (below): figure/table generators reading committed run folders, claims ledger, related-work refresh (2025–26 preprint re-check) | manuscript |

## Reports (all under `reports/`, committed; each cites run folders and commits)

| Report | When | Content |
|---|---|---|
| **R0** reproduction | after runbook Step 5 | machine change, RAM screen, every re-run comparison, env versions |
| **R1** sanity pilot | after S0 | early-signal tables and curves, honest caveats (1–2 seeds, short budget, not a gate) |
| **R2** mechanisms (E0–E3) + G1/G2 | after D3 | E0 (G1), E1, E2 (operator/mapping choice), E3 (growth policy, γ), pre-registration summary |
| **R3** core claim (E4) = W2 at G3, final = D4.6 | after D4.1; final after D4.3–D4.5 | ℓ_min feasibility, 50M screen, 125M convergence, controls C1/C2/C1h/C3t, operator ablation C8, stratified losses, probes, `k(L)`, projections, escalation verdicts |
| **R4** pretrained hosts + quantization | after D4.4, D4.3 | CPT on three hosts, size trend of `k`, PTQ retention, equal-bytes compression |
| **R5** explainability + zero-shot (G4) | after D5 | faithfulness, sense alignment, frequency, LLM-graded study with κ and calibration, zero-shot insertion |
| **R6** self-authoring (G5) | after D7 | authoring quality by host, round-1 utility, loss gaps vs controls, cross-authoring |
| **R7** application tracks | after D8 | per-track cardinality and feasibility, results, cross-track table (D8.8) |
| **R8** evidence report + escalation request (W3, G6) | last | per question: evidence, `k`, projections, rule verdicts, GPU-h cost of X1–X4, ranked recommendation; implications |

## Manuscript package (`manuscript/`)

- `outline.md`: working title (CG-VSA), contribution list tied to the hypotheses H-A … H-G, venue-neutral structure.
- `claims.md`: claims ledger — each claim, its evidence (report + run folder + commit), the allowed wording from proposal §7, status (supported / refuted / inconclusive).
- `draft.md`: sections (abstract, introduction, related work, method M0–M5, experimental setup, results E0–E8, limitations, reproducibility) written from the reports; negative results kept.
- `figures/` and `tables/` with `make_figures.py` that regenerates every figure from committed run folders.
- `reproducibility.md`: hardware, env, data sources and hashes, holdout hashes, commit per result, compute ledger.
- Pre-submission checklist: re-check the two claims most exposed to 2025–26 preprints (usage-level momentum-conflict splitting; context attention over HRR-bound edges); clinician study (W4) needed before any external clinical claim; LLM-graded results labelled as estimates.

## Rules (unchanged from tasks.md, restated because they bind every WP)

New names not new semantics; one frozen holdout per track; freeze the linker before linking; pre-register before D4.1; controls sized at G2; operator ablation in every report; commit, then run; MIMIC never committed; frozen recipe for early baselines; no escalation without evidence and the author's approval.

## Open decisions log (resolve at the named gate; record the outcome in gates.md)

| # | Decision | Raised by | Resolve at | Default if nothing better |
|---|---|---|---|---|
| 1 | Channel scale on Qwen2.5 (embedding rows have norm ≈ 0.46 vs 3.1–3.7 for SmolLM2; the initial injection is ≈ 7× larger relative to Qwen rows; at step 0 C2 on frozen Qwen is +0.031 nats vs C0', C5 on SmolLM2 +0.0003) | WP-host | before D4.4 (G3) | new key that scales the channel to the host's mean embedding norm, applied to all hosts (a no-op-sized change for SmolLM2); `gate_bias` unchanged |
| 2 | Size of the LoRA-vs-frozen check on SmolLM2-135M | WP-host | G3 | {C0', C2, best} × 3 (9 runs) rather than the full grid |
| 3 | CPT tokens/step 131k (128 × 1024), channel lr 1e-3, LoRA lr 2e-4 | WP-host | after the cpt-pilot | keep unless the pilot curves look unstable |
| 4 | Frozen-host C0' has a flat curve, so k(L) vs C0' is undefined in frozen mode | WP-host | WP-E4R | report loss gaps at fixed budgets for frozen hosts; k(L) only for LoRA hosts |
| 5 | Gate item 1b "fewer channel parameters than C2": C2 is capacity-matched to C3, so the comparison is a coin flip (50M: C5 2.242M vs C2 2.245M; 125M: 2.310M vs 2.251M) | WP-E4R | G2 | judge 1b against the full-width C2f, and against C2 only with a ±5% parameter tolerance |
| 6 | Locality is one-sided (unlinked loss not worse than C0 by > 0.5%, upper 95% bound); "matches C2" on rare concepts is non-inferiority with a 0.5% margin | WP-E4R | G2 | as implemented in `e4_report` |
| 7 | Gate significance: Holm-adjusted p < 0.05 with 10,000 bootstrap resamples | WP-E4R | G2 | as implemented |
| 8 | Rare stratum is thin (≈ 250 rare targets at 512 eval windows) | WP-E4R | G2 | D4.1 uses 1,024 windows plus a dedicated rare/held-out concept eval set built from extra eval documents if the pilot CIs are too wide |
| 9 | BLESS endpoint: the `[h_x; h_y]` concat probe equals a relatum-only control (memorizes the relatum) | WP-probe | G2 | primary = interaction probe `[|h_x−h_y|; h_x⊙h_y]` and prompt AUC; concat and relatum-only reported as controls |
| 10 | Probe layer (final vs middle) | WP-probe | G2 | final layer (as B8) for every reported number; one middle-layer sensitivity run per stage |
| 11 | Per-item probe outputs (≈ 0.3 MB per checkpoint) | WP-probe | now | committed with each run folder (needed for paired tests); `*.pt` checkpoints are git-ignored |
| 12 | T1-open evaluation corpus and size | WP-T1 | before D4.5 | gate strata on `eval-pubmed` with ≥ 2,048 windows (3,072 if ℓ_min = 3); locality on `eval-general`; ℓ_min = 2 |
| 13 | T1-open rare stratum power (measured only by the full build) | WP-T1 | after the full T1 build | if underpowered, report the rare stratum as exploratory rather than re-freezing the holdout |
| 14 | T1-open CPT corpus mix (50/50 PubMed/general like from-scratch) | WP-T1 | before D4.5 | keep 50/50 (same recipe as from scratch; locality measurable) |
| 15 | T3 product and T6 legal: seen-rare stratum nearly empty at 100M tokens (rare entries rarely occur in eval) | WP-C7 | G2 / before D8 | run T3 and T6 on the held-out part of the gate plus locality and track tasks; report the rare stratum as infeasible (a result for the cross-track table); a rare-enriched eval set only if time allows |
| 16 | (resolved 2026-10-02: strata `after_unseen`, `after_rare_seen` added; from the CPT pilot on) Trainer stratum `after_rare` counts entries with training frequency 0 (never linked in training, not held out) as rare | WP-C7 | after the S0 pilot finishes (changing lm.py mid-pilot would split strata across pilot runs) | add new strata `after_unseen` (frequency 0, not held out) and `after_rare_seen` (1–9) to lm.py; keep `after_rare` unchanged for replay; gate item 1b uses `after_rare_seen` |
| 17 | Feasibility thresholds differ between WP-T1 (≥ 2,000 occurrences, ≥ 300 entries) and WP-C7 (≥ 1,000 spans, ≥ 50 entries) | WP-T1, WP-C7 | G2 | harmonize at G2 on WP-T1's power argument (≥ 2,000 occurrences, ≥ 300 entries) for gate claims; WP-C7's lower bar marks a track "exploratory-feasible" |
| 18 | T2 developer-tools track corpus | WP-C7 | resolved 2026-10-02 | built by WP-T2 (`experiments/t2-developer-tools/runs/v1`, `v1-gpt2`; holdout `75ee2217…`; feasible at ℓ_min 2; needs 3,072 eval windows for the strict bar) |
| 19 | ℓ_min = 1 linking: WordNet abbreviations collide with function words under case-insensitive matching (" as" → American Samoa, " or" → operating room, " at" → astatine, " are" → the area unit); the C3 `train-l1` slice carries these links | WP-Q | before D4.7 | the D4.7 ℓ_min = 1 arm uses a new linker version that drops all-caps abbreviation aliases (case-sensitive only) and single-token function-word aliases, relinked into a new `train-l1-clean` slice with the same holdout; ℓ ≥ 2 corpora unchanged (their spans are real multiword lemmas) |
| 20 | Literature re-check (manuscript/related-work-recheck-2026-10.md): add controls C1h-E (Engram-faithful hashed memory), C1m (subtoken-mean span), an alias stratum, a translation (`x + r`) operator in C8, a fixed-threshold gradient-conflict split rule in E3/E0.2, and E7 controls (verified notes, synthetic-QA CPT, verified edges as plain text); narrow the M3/M1/M4/M5 wording | WP-MS | G2 (before D4.9) | adopt C1m, the translation operator and the alias stratum (cheap); adopt C1h-E in the D4.9 screen if its parameter-matched implementation fits the run budget, else qualify H-A as "at matched parameters, injection site and gate"; E3 threshold baseline as a supplemental run; E7 controls passed to WP-E7 |
| 21 | OpenReview `tfyLS1cB5W` ("Encoding Ontologies with HRR for Transformers") may be the group's own earlier submission | WP-MS | resolved 2026-10-04 (author) | it is the group's own previous work: cite the published HRRBERT record (CEUR-WS) as the group's prior work, in the third person for double-blind review, and do not count the OpenReview submission as a separate precedent. T. Yu's 2025 thesis contains no HRR causal-LM variant (author), so there is no prior disclosure |
| 22 | Judge cost: 10 validation calls of the E5.3 rubrics cost $0.027–0.061 each (mean $0.039), about twice the planned $0.02; D5.3 needs ≈ 1,550 calls (rare and held-out neighbours ≈ 450 each, edges ≈ 390, cards ≈ 265) + ≈ 450 for the T1-open neighbour set | WP-E5 | D5.3 start | run as pre-registered (3 calls × 2 paraphrases) at ≈ $60–80 (≈ $40 at $0.02); `grade --dry-run` prints the uncached calls and `--max-new-calls` caps them |
| 23 | E5.3 neighbour space and rare set: neighbours in the model's `state` space (final-layer state of "The word <alias>", channel live) or `row` space; rare = training frequency 1–9 (the E4 rare stratum) instead of HRRBERT's "< 30 occurrences" | WP-E5 | D5.3 start | `state` primary, `row` secondary; frequency 1–9, frequency-weighted draw of 10 concepts × top-4 = 40 pairs per system |
| 24 | E5.1 ablation semantics: a removed edge is erased from the weighted sum with the other weights kept (= softmax renormalized over the rest); attention is not recomputed without the edge, and only occurrences with more than `k` edges are ablated | WP-E5 | G4 | as implemented; `remove_all` = channel off is checked on every run |
| 25 | E5.2 coverage: at ℓ_min = 2 only multi-subtoken polysemous aliases are linked (≈ 7,800 SemCor instances with GPT-2 BPE, mostly multiword or rare senses); common polysemous words are single tokens | WP-E5 | G4 | report the ℓ_min = 2 runs; add the D4.7 ℓ_min = 1 runs as a secondary set if they exist |
| 26 | E5.4 baseline protocol: maps fitted on 4,000 seen entries (frequency ≥ 10) against the run's own rows; baseline rows rescaled to the mean seen-row norm; a text-evidence baseline without evidence for a concept falls back to the mean row (coverage recorded); pretrained hosts are judged on `c3-synthetic` items built per host tokenizer | WP-E5 | G4 | as implemented and pre-registered in `experiments/e5-explainability/preregistration.md` |
| 27 | T2 linking of identifiers: `normalize_alias` turns `_` into a space, so snake_case C6 symbols never match their text (`` `foo_bar` ``); without a fix every C6 zero-shot concept is reported `unlinked` | WP-E5 | WP-C7 (before D8.2) | WP-C7 links T2 with an identifier-preserving alias normalization (new name, existing default unchanged) |
| 28 | T2 synthetic API text is templated: only ≈ 30M GPT-2 tokens of non-redundant domain text exist; the 50/50 from-scratch mix (150M domain tokens) is ≈ 90% repetitive synthetic text | WP-T2 | before D8.2 | keep the pre-registered 50/50 recipe for comparability with T1, and report the repetition statistic with the T2 results; a reduced-domain-share variant only if T2 shows an effect worth checking |
| 29 | E5.4 `c6_devtools` zero-shot scenario must link with the T2 identifier normalization (`alias_table.json`, `normalization="identifier"`) | WP-T2 / WP-E5 | before the first E5.4 T2 run | switch the scenario to load T2's `alias_table.json` via `load_alias_table` (follow-up fix) |
| 30 | E7.2 host mode and channel condition | WP-E7 | G3 | LoRA host (with a frozen host the text controls can only train the channel, which favours the channel conditions); channel condition = the G3 best |
| 31 | E7 masking: 20% after the shared-concept cascade (draw 11.9% → 20.2% of linked entries) | WP-E7 | now | as implemented |
| 32 | E7 compute-matched control counts discovery + authoring + verification FLOPs (≈ +27M tokens over 25M) | WP-E7 | now | as implemented (the stronger control) |
| 33 | E7 text controls budget: matched tokens; teacher edges go through the same verification; defaults min_validation 6, min_documents 3, 1,000 candidates | WP-E7 | now | as implemented |
| 34 | E7 reading slice: documents after the C3 GPT-2 training corpus (doc ≥ 1,070,634), disjoint from every C3/host training corpus (CPT runs sample windows over all 130M host tokens, so a token-offset slice would overlap D4.4) | WP-E7 | now | as implemented |
| 35 | E9 new-word frames: category-only frames (lexname, pos, hypernym; 45% of WordNet entries) always exist already (their category siblings), so they cannot be "new combinations"; donors are entries with ≥ 1 non-category templated edge (degree ≥ 4), whose degree distribution the new words follow | WP-E9 | before the E9 main runs | as implemented (`e9_ontology_edit`: category edges kept, other fillers resampled so no entry has the category edge with any resampled edge) |
| 36 | E9 recipe (host mode, gate bias) | WP-E9 | resolved 2026-10-02 (engagement check) | full fine-tuning, host lr 3e-5, gate bias 0, channel lr 1e-3 (`--channel-lr`), 50M tokens — the `e9_plan` defaults |
| 39 | E9 corpora: general WordNet words are not rare for SmolLM2 (C5 = C0′ in every check setting), so E9 runs on track vocabularies first — T5 glossary, T4 chemistry, T1-open, then WordNet as the negative control (`e9_plan --track`) | orchestrator | resolved 2026-10-02 | as implemented: per-track corpora, evaluation alias tables replayed from the track adapters (digest-checked; `~/data/vsa-llm/e9/alias-tables/`), dimension-3 items per track (`experiments/e9-retrofit/items/*-<track>-smollm2-v1`), WP-C7 zero-shot items for T5/T4 (`e9_tracks zeroshot`), general-text locality and quantization damage on `eval-general` (`e4_quant --eval-corpus`) |
| 40 | E9 track edits: T5 edits `owned_by` (else `area`, `status`; `is_a` is visible in many T5 names' head noun); T4 edits `has_functional_parent` (else `has_role`, `is_a`) with "same type" = any ChEBI entity (coarse); T1 edits the MeSH `parent`; T1 has no WP-C7 zero-shot items (dimension 3 = invented new words + edits only) | WP-E9 | before the T4/T1 runs | as implemented |
| 41 | E9 evaluation cost: full probes ≈ 6 min per 135M run (WSD ≈ 45%) under GPU contention, ≈ 2.5× for 360M; with bf16 + INT4 for 4 models × 2 hosts the evaluations cost about as much as seed-1 training | WP-E9 | before queueing | queue with `--int4-probes lambada,wic,card660,rare_words,bless,hyperlex` (INT4 without WSD) unless the GPU is idle |
| 37 | E9 zero-shot (E5.4) sources: structure-only (`own,none,random,mean_row,surface_mean,graph_projection`); text-evidence sources (definitions, à la carte, CoLLEGe) are not part of the E9 question | WP-E9 | before the E9 main runs | as implemented in the chained jobs |
| 38 | E9 quantized evaluations of probes, zero-shot and editing run at INT4 variant A only (channel FP16); loss strata (`e4_quant`) cover INT8 and INT4 × A/B | WP-E9 | after seed 1 | add variant-B evaluations (`--quantize-channel`) only if A shows a dimension-1/3 effect |
| 42 | Open decision 1 resolved for the Qwen3 E9 hosts: `channel.scale_to_host` scales the injected rows by `ρ · n̄_E / n̄_c` (host's mean input-embedding row norm over the channel's own mean row norm at initialization), ρ = 0.3 = SmolLM2-360M's unscaled C5 ratio (scale 0.99 there, i.e. no-op-sized). Unscaled C5 ratios: SmolLM2-135M 0.27, -360M 0.30, Qwen2.5-0.5B 2.33, Qwen3-0.6B 1.21, -1.7B 1.04, -4B 1.61 (scales at ρ = 0.3: 0.25, 0.29, 0.19); the recorded SmolLM2 runs stay unscaled | WP-Qwen | before the Qwen3 E9 block | Qwen3 C2 and C5 scaled (`e9_plan --channel-scale auto`); SmolLM2 seeds 2–3 stay unscaled for comparability with seed 1 (`--channel-scale on` would scale them) |
| 43 | Qwen3 tokenizer vs Qwen2.5: the same byte-level BPE, merges, normalizer (NFC) and pre-tokenizer, plus 4 added tokens (`<tool_response>`, `</tool_response>`, `<think>`, `</think>`; ids 151,665–151,668), so the fingerprints differ (Qwen3 `a63080b4…` for all three sizes, Qwen2.5 `f884026e…`) and the trainer refuses Qwen2.5 corpora for Qwen3 hosts | WP-Qwen | now | separate `qwen3` corpora per track (same ontology, alias table, holdout); natural text tokenizes identically, so Qwen2.5 corpora would only differ where those four strings occur |
| 44 | Qwen3-4B memory: an fp32 host is 16 GB, LoRA-64 adds ≈ 2 GB with AdamW state; the memory probe measures fp32 ± checkpointing and a bf16 host (`model.host_dtype`, LoRA adapters kept fp32) | WP-Qwen | after the probe | fp32 host if any setting fits the budget, else bf16 (applied to every model of the host, P0 included, so pairs stay within one host numerics) |
| 45 | Qwen3 E9 GPU cost (6N model calibrated on the SmolLM2-360M LoRA-64 check, before the probe): per 50M-token LoRA run 0.6B ≈ 1.8 h, 1.7B ≈ 6.3 h, 4B ≈ 16.6 h with checkpointing; trained runs C0′/C2/C5 per host ≈ 5.5 / 19 / 50 GPU-h plus evaluations (≈ 3–5 / 6–15 / 12–30 h: FLOP-scaled from SmolLM2-135M's ≈ 6 min per probe set, an upper bound because small-model probe time is partly fixed overhead) | WP-Qwen | before queueing 4B | 0.6B + 1.7B block first; 4B single seed later as its own stage, ideally at 25M tokens or {C0′, C5} only (author) |
| 46 | Qwen3 per-run evaluations: 151,936-entry logits make the default probe/zero-shot batches (32/64) too large for 1.7B/4B next to the host; INT4 probes without WSD (decision 41) | WP-Qwen | now | evaluation batch 16 / 8 / 4 (0.6B / 1.7B / 4B) and INT4 probes `lambada,wic,card660,rare_words,bless,hyperlex` by default for Qwen3 |
| 47 | Qwen3.5 LoRA targets: the default set reaches q/k/v/o of the 6 full-attention layers and every MLP (96 adapters) but none of the 18 Gated DeltaNet token mixers; the E9 "same trainable host parameters" rule needs the mixers adapted | WP-Qwen35 | now | `model.lora_targets: linear_attention` for both Qwen3.5 hosts (186 adapters: in_proj_qkv / z / a / b and out_proj added; r = 64 on the 16-row in_proj_a/b is a full-rank update of a tiny matrix); LoRA parameters 43.3M (0.8B) / 67.3M (2B) |
| 48 | Qwen3.5 kernels and numerics: the fast path (fla in bf16 with fp32 accumulation, causal-conv1d) and the PyTorch reference (fp32 delta rule) agree to ≈ 2% relative RMS in the final hidden state (gradient cosine 0.9999) | WP-Qwen35 | now | every Qwen3.5 training and evaluation job runs on CUDA, i.e. on the fast path (recorded per run in `linear_attention_kernels`); CPU (tests only) uses the reference |
| 49 | Qwen3.5 software stack: transformers 5.18 / tokenizers 0.23 / fla 0.5.2 / causal-conv1d 1.7.0 in a venv over the pinned env (torch, torchao identical); tokenizer fingerprints depend on the transformers version (SmolLM2 and Qwen3 differ between 4.54 and 5.18) | WP-Qwen35 | now | Qwen3.5 corpora are built and consumed in the venv only, other hosts' in the pinned env; Qwen3.5 results are reported as a separate host family (no pooling with SmolLM2/Qwen3 runs) |
| 50 | Qwen3.5 E9 cost before the probe (FLOP-scaled from the Qwen3 probe): 2B ≈ 3.0 h per 50M-token LoRA run (≈ 4.5 h if checkpointing is needed), 0.8B ≈ 2.1 h; block (T5, seed 1, P0/C0′/C2/C5, evaluations, quantization, report) ≈ 20–25 GPU-h | WP-Qwen35 | after the probe | queue the probe at priority 52, then `e9_plan --track t5 --hosts Qwen3.5-2B-Base Qwen3.5-0.8B-Base` (stage `t5-qwen35`, 52–55) |
| 51 | E4 core (≈ 143 GPU-h, G2 pre-registration): run now or after the E9 natural-text results | orchestrator | resolved 2026-10-04 (author) | wait: decide after the T4 chemistry and T1-open E9 results (the copy-concern test); the E9 queue runs first |
| 52 | E10 null worlds (E10 baselines accept 2.7–4.3 false slots per null-world run) vs the author's framing | orchestrator | resolved 2026-10-04 (author) | the claim is not that a slot proves a relation exists. Passive (gradient) learning comes first; the model then reflects on what a slot means (self-disentangling, property-hypothesis tests) and commits the best-fit explanation as an explicit weight write instead of waiting for gradients to confirm it. Question: does pairing passive learning with this active self-reflection make the model learn much faster? The null world becomes the **cost-of-commitment** control (does a wrong commitment hurt, and does later reflection retract it?), not a refutation. New experiment **E10.9** (CPU, synthetic, then E10.9b on the T5 joint LM if E10.9a is positive) |
| 53 | E10.9b (passive vs hybrid inside joint LM training on T5) | orchestrator | resolved 2026-10-04 (author: "queue E10.9b too once E10.9a is positive") | positive = E10.9a pre-registered primary met at ρ = 1 (AULC H − P > 0, 95% CI excluding 0) **and** H beats the passive compute control (H − P+compute > 0, CI excluding 0). If positive: implement E10.9b (reusing E10.9a's reflection step) and queue it at priority 55 (after the PQ1 controls and the Qwen3.5 probe, before Qwen3-4B) without asking again; mixed results (e.g. only rule-implied edges improve) go back to the author first |

## E9 — retrofit × quantization (author request 2026-10-02)

Three applicability questions on pretrained hosts, on one fixed test set: (1) does a VSA ontology channel trained jointly with a pretrained model improve it on long-token, rare and out-of-distribution words; (2) is the gap larger under weight quantization (quantization hurts these words more, and the channel recovers more of it); (3) after training, can the model learn new or changed words zero-shot by editing the ontology alone?

- **Hosts:** SmolLM2-360M (primary), SmolLM2-135M (size trend); Qwen2.5-0.5B after open decision 1 (channel scale).
- **Models:** P0 original host (evaluation only); C0′ continued training without the channel (same tokens, same trainable host parameters); C2 continued training + capacity-matched free per-concept table; C5 continued training + attentive VSA channel (WordNet ontology, C3 holdout).
- **Training:** an engagement check first (the S0 CPT pilot with frozen/LoRA-16 host barely moved the loss in 25M tokens): full fine-tuning (host lr ≈ 3e-5) vs LoRA r = 64, gate bias −2 vs 0, 10M tokens, C5 vs C0′. The setting where training moves the linked strata fixes the recipe; then 50M tokens, seed 1 first, then seeds 2–3.
- **Fixed test set:** the host eval corpus strata (`inside`, `after_len2`, `after_len3plus`, `after_rare_seen`, `after_unseen`, `after_heldout`, `unlinked`, `all`) with per-window losses; probe subsets by status (CARD-660, Rare Words, BLESS, WiC, WSD; channel_probes); contamination-free unseen words (E5.4 `c3-synthetic-smollm2-v1`, invented names); LAMBADA and general-text loss for locality.
- **Quantization:** every model at bf16, INT8, INT4 (torchao weight-only, output head FP), C5 with the channel in FP16 (variant A) and quantized (variant B) — `e4_quant`.
- **Readout:** (1) C5 − C0′ and C5 − C2 per stratum at bf16; (2) quantization damage per stratum (INT4 − bf16) for P0, C0′, C2, C5, and the retained gain (C5 − C0′ at INT4 vs at bf16) — difference-in-differences with paired window bootstraps.
- **Dimension 3 — zero-shot learning by editing the ontology (after training, no weight update):** (a) *new words*: add entries with invented names whose frames are new combinations of existing atomics and relations (e.g. a new tool that `is_a` container and `part_of` vehicle), link them in text and test property selection, entailment and paraphrase consistency, and loss on held-out contexts written for them; (b) *edited words*: change a relation–filler edge of an existing concept and test whether the model's behaviour moves to the edited fact (and only for that concept); baselines: the same new names with no frame (C0′/P0 see only subtokens), with a mean row (C2 fallback), and with random frames of equal degree. Run on the bf16 and INT4 versions of each trained model.
- **Budget:** ≈ 1 GPU-h check + ≈ 4 h seed 1 + ≈ 6 h seeds 2–3 + ≈ 1 h quantization/probes; counted against D4.4/D4.3. Report: `reports/R9-retrofit-quantization.md`.
- **Implementation (WP-E9):** `experiments/e9_plan.py` (configs under `experiments/e9-retrofit/configs/<stage>/`, runs under `experiments/e9-retrofit/runs/<stage>/`; per-run probes, E5.4 zero-shot and editing evaluations at bf16 and INT4 chained after training, then `e4_quant` and the report); `experiments/e9_ontology_edit.py` (dimension 3; items in `experiments/e9-retrofit/items/`); `experiments/e9_report.py` (R9); `--quantize int8|int4 [--quantize-channel]` on `channel_probes`, `e5_zeroshot` and the editing module; `channel_probes.load_run` reads trainable-only checkpoints. Decisions 35–41. After the engagement check (below, after E10):
`experiments/e9_tracks.py` (track corpora, alias-table replay, track lexicons, WP-C7 zero-shot evaluation),
`e9_plan --track t5|t4|t1|wordnet` (T5 first), `e4_quant --eval-corpus` (general text), `e9_report --quant-general`.

## E10 — self-learned semantics (author request 2026-10-02; new hypothesis H-H)

**H-H.** A model whose ontology mapping is learnable (regularized toward the curated prior) can (i) recover erased parts of the ontology from its training signal, (ii) discover relations it was never given through blank relation slots, interpret them and consolidate them additively (crystallize, then open a new blank slot), (iii) accept or reject its own hypotheses by self-constructed tests scored on held-out text, and (iv) map a new word zero-shot through the relations it has disentangled. *Refuted if* erased edges are not recovered above a matched random-candidate baseline, blank slots do not align with hidden relations above chance (ARI vs permuted), self-acceptance is no more accurate than accepting at random at the same rate, or new-word frame inference is no better than nearest-neighbour frames.

Ladder (each step has gold because relations are hidden on purpose):

| Step | Setting | Experiments | Baselines | Compute |
|---|---|---|---|---|
| E10.0 | synthetic planted ontology (E0 teacher machinery), CPU | (a) learnability ablation: {atomics, relation operators, concept→edge mapping, frames} × {fixed, learnable + L2-to-prior, learnable free} + leave-one-out; (b) erasure & recovery: keep 30–90% of edges fixed, erase the rest, learn soft edge weights over a candidate set with sparsity → recovery P/R vs erasure rate; (c) blank relations: hide one or more relation types, add K blank slots with learnable edge assignment → slot ↔ hidden-relation alignment (ARI / Jaccard), interpretation (match to relation labels / descriptions), crystallize (freeze, label) and add the next blank slot (additive curriculum) vs all slots at once; (d) self-tested acceptance: hypotheses scored by self-constructed held-out tests, accept/reject accuracy vs gold; (e) new-word frame inference from k contexts | M3 splitting; random candidates; random acceptance at matched rate; nearest-neighbour frames | CPU |
| E10.1 | WordNet on frozen anchors (E2/E3 tooling) | E10.0 (a)–(e) with real relations: erase hypernym/meronym edges; hide meronym sub-types and instance-hypernymy; blank slots | as above + E3 policies | ≈ 5 GPU-h |
| E10.2 | joint LM training (50M from scratch; SmolLM2 via E9) | learnable ontology with L2-to-prior during LM training; erasure recovery from the LM signal; blank slots; zero-shot frame inference for new words from contexts | fixed ontology (C5); C2 | ≈ 20–30 GPU-h, after G3/E9 |
| E10.3 | self-reflective loop on a pretrained host (builds on M5/E7) | host names/disentangles its own slots (prompted over each slot's top concept pairs), writes its own test statements, updates, accepts/rejects by held-out utility and hidden-gold audit; new words mapped through its own relations | teacher (Claude) naming; random acceptance; compute-matched CPT | ≈ 10 GPU-h + LLM calls, after E7 |

**Human-learning comparison** (an evaluation axis, claims kept modest): fast mapping from few exposures (Carey & Bartlett 1978) and syntactic bootstrapping (Gleitman 1990) ↔ new-word frame inference; assimilation/accommodation (Piaget) and complementary learning systems with consolidation (McClelland, McNaughton & O'Reilly 1995) ↔ fixed principal ontology + plastic regions + crystallization; structure mapping (Gentner 1983) and relational-structure discovery (Kemp et al. 2006; Kemp & Tenenbaum 2008) ↔ blank-slot discovery; testing effect / self-explanation (Chi et al. 1989) ↔ self-tested acceptance. Divergences to respect: humans start from core priors (Spelke & Kinzler 2007) rather than a complete curated ontology (so "erase most, recover" is the closer analogue); consolidation is revisable (conceptual change), so crystallized relations must be reopenable under strong contrary evidence; self-consistency is not truth, so acceptance is scored on held-out text and audited against hidden gold. Measurable signatures: learning curves with few exposures, overextension → refinement trajectories, basic-level advantage. (Citations to be verified in the manuscript related-work pass.)

**Additions of 2026-10-02 (author).** Principle: **learning is self-supervised — no labels or answers enter the learning loop; gold ontologies are used only for evaluation and hidden audits.**

| Step | Experiment | Conditions | Readout |
|---|---|---|---|
| E10.4 | seed ontology → recover the logical ontology | start from: empty · a small *core* seed (self/agent, object, living thing, action, cause, goal, learn; relations is-a, part-of, causes — WordNet unique beginners and their top levels in E10.1) · a seed authored by the pretrained host (E7 authoring) · 30% of the curated ontology · full curated (upper bound) | recovered relations/edges vs hidden gold; reasoning probes: multi-hop is-a ("is a beagle an animal?"), transitivity and inverse consistency of the recovered ontology |
| E10.5 | dreaming mode (offline self-revision) | off · on at several frequencies; each pass revisits every relation and mapping given the current full context and proposes revisions, accepted only if held-out fit improves; injected corruptions (wrong edges, mislabeled/merged relations) | repair rate of injected errors, wrongly crystallized relations reopened, damage to correct ones |
| E10.6 | riddle-style relation identification | per blank slot the model generates candidate hypotheses ("given these concept pairs, what is the relation?") and tests each by its predictions on held-out pairs/text; best explanation adopted; structural hypotheses (symmetric, transitive, functional, inverse-of-X, composition-of-X-and-Y) on synthetic/WordNet, natural-language hypotheses on the pretrained host | identifiability without labels (alignment with the hidden relation after the fact), hypothesis accuracy vs number of candidate hypotheses |
| E10.7 | data variety vs self-confirmation bias | number of domains/sources and contexts per concept at fixed tokens | wrong self-acceptance rate on a hidden audit set vs variety |
| E10.8 | continual additive learning | relations learned sequentially (crystallize → new blank) with dreaming between stages vs all at once | growth curve of the self-built ontology, retention of earlier relations |
| E10.9 | passive learning paired with active self-reflection (author 2026-10-04, decision 52) | **E10.9a** (synthetic, CPU, `e10_active_passive`): P passive · H = passive + a reflection round every K steps that commits the best held-out-tested explanation of a slot as an explicit write (crystallize + rule closure / pruning) · H+R (+ revision of commitments) · H-rand (random write of the same size) · H-AMIE (AMIE explanation) · P+compute · A-first (reflection before any passive learning); structure ρ ∈ {1, 0.5, 0.25, 0} (0 = the D-B4 null world: cost of commitment); budgets f ∈ {0.05 … 1}; 5 seeds. **E10.9b** (T5 joint LM, GPU, design below) | AULC of held-out test-concept fit over the budgets (H − P at ρ = 1), observations to criterion; null-world non-inferiority of H+R and retraction of false commitments. Pre-registered in R10 "E10.9"; E10.9b is queued only if E10.9a passes the gate of decision 53. **E10.9a result (`runs/e10.9-v1`, 2026-10-04): not faster at matched data.** AULC H − P = −0.003 [−0.019, +0.013]; observations to criterion 0.60, 3 of 5 seeds ≤ 0.5. At equal gradient data reflection adds +0.021 [+0.006, +0.035] test fit at f = 1, all of it from rule-implied edges on never-observed concepts, and the content matters (H − H-rand +0.030). In the null world the false commitments are inert (±0.002) but not retracted (0 of 19). **Gate not met; E10.9b not queued** |

Report: `reports/R10-self-learned-semantics.md`. E10.0/E10.1 start now (CPU / backfill GPU); E10.2/E10.3 after G3 and E9/E7.

**E10.9b — passive vs hybrid inside joint LM training on T5 (implementation-ready design; queue automatically iff the E10.9a gate passes, decision 53).**

*Gate.* AULC H − P > 0 at ρ = 1 with the 95% t-interval excluding 0, **and** AULC H − P+compute > 0 with its t-interval excluding 0 (R10 "E10.9 results" states whether both hold).

**Outcome (2026-10-04):** neither criterion holds. H − P = −0.003 [−0.019, +0.013]; H − P+compute = −0.003 [−0.019, +0.013]. **E10.9b is not queued.** The design below stays ready in case a revised E10.9a passes.

The likely lever, if E10.9a is revised: in E10.9a the self-tests' held-out split costs the hybrid as much gradient data as reflection gains (P − P-split +0.018 AULC). In LM training held-out windows are cheap relative to the corpus, so the E10.9b reflection split (1% of the stream) costs much less.

*Question.* The E10.9a comparison inside joint LM training. Some T5 relations are hidden from the channel ontology the LM reads, and blank slots are given. Reflection rounds write committed relations, with their rule closures, into that ontology. Does the hybrid reach the passive learner's held-out-term loss with fewer training tokens?

*Hosts and recipe.*
- The E9 T5 recipe: SmolLM2-135M first, then SmolLM2-360M; full fine-tuning at host lr 3e-5; channel lr 1e-3; gate bias 0.
- Attentive compose channel at d = 256, key 8, context window 8.
- 50M tokens: 762 steps × 65,536 tokens. micro 8 × accum 8 for 135M; micro 4 × accum 16 for 360M.
- Corpus `~/data/vsa-llm/tracks/t5-glossary/v1`, with the same batches in every arm (`sample_batch` seeds by `(data.seed, step, micro_step)`).

*Ontology.* Every arm, P included, uses a derived channel ontology:
- Two inverse relations are added: `owns` = owned_by⁻¹ and `has_part` = part_of⁻¹, headed by the filler term's entry. The documents already use the inverse wording ("{O} owns {s}").
- Three relations are **hidden**:
  - `owned_by`: 3,800 edges, functional, present on all 360 held-out heads; it can be restored by the closure `inverse_of:owns`;
  - `part_of`: 844 edges; closure `inverse_of:has_part`;
  - `depends_on`: 1,227 edges, antisymmetric, with 839 unclosed two-step paths. It has no true closure, so a wrong `transitive` adoption would show the cost of commitment.

  This mirrors E10.9a, where the gain mechanism is closure edges written for heads that no observation covers. T5 itself has no symmetric relation and no materialized inverse pairs, which is why the two inverses are derived.
- 70% of the hidden edges on training-term heads are offered as open candidates, among as many same-pool distractors. Held-out heads get no candidates: only a committed rule can restore their hidden edges.
- **5 blank slots**: one opens at 0 tokens and one after each reflection round.
- A **null variant** (Pn, Hn) permutes the offered pairs' tails (ρ = 0; `graded_scenario`'s draw).

*Schedule.*
- Reflection every **10M tokens** (≈ 152 steps): rounds at 10M, 20M, 30M and 40M; the last 10M are passive. This is E10.9a's K = T/5.
- Evaluation every **2.5M tokens** on the fixed 1,024 T5 eval windows (all strata), plus before and after each round.
- Reflection evidence comes from a **reflection split**: the last 1% of the training token stream, 256 fixed windows that mention linked entries.
  - H and HR never take gradient on it.
  - P and Pc train on it (matched data, as in E10.9a).
  - The E9 eval windows are never read by learning.

*Arms* (names without "-", because `e4_report.NAME` parses `<size>-<condition>-s<seed>`):

| Arm | What it is |
|---|---|
| P | passive: learnable ontology (L2-to-prior, open candidates, slots), no reflection |
| H | reflection rounds through `e10_active_passive.reflect` with LM-window evidence |
| HR | H + revision of commitments (refit-free removal-utility revisit on the reflection windows, accepted iff the cluster lower bound > 0) |
| Pc | P + extra training tokens per round equal to H's reflection forward tokens ÷ 3 (one training token ≈ 3 forward tokens), measured in H's run of the same seed |
| Pn, Hn | null variant (ρ = 0) |

*Seeds.*
- 135M: seeds 1, 2, 3 for P, H, HR and Pc; seeds 1–3 for Pn and Hn.
- 360M: seeds 1 and 2 for P, H and Pc, queued only if the 135M primary is positive.

*Pre-registered readouts.*
- **Primary:** the `after_heldout` loss along the token axis, as tokens-to-criterion: the smallest token count at which H's `after_heldout` loss is ≤ P's loss at 50M (linear interpolation between evaluations), as a ratio to 50M. Also the loss AULC over tokens H − P, paired over seeds and windows (`e4_report.paired_difference`).
- **"Much faster"** = ratio ≤ 0.5.
- **Secondary:**
  - `after_unseen` and `after_rare_seen`;
  - H − Pc;
  - Hn − Pn at 50M (non-inferiority margin +0.01 nats);
  - commitments, adopted rules and their gold verdicts against the full T5 ontology (evaluation only);
  - general-text loss (locality).

*Queue.*
- Training at **priority 55**, evaluations at 56, report at 57 (`jobqueue add --priority 55 …`; pending work currently sits at 33–61; lower numbers run first).
- `min_free_gb` 20.

*GPU-h estimate.* Measured cost per 50M-token training job is 0.52 h for 135M and 1.11 h for 360M (`.jobs/t5-SmolLM2-*-full-*`; execution.md "E9 T5" timings). The 0.35 / 0.76 h figures quoted earlier are not in the records.
- Reflection overhead ≈ 0.13 h per 135M run: ≤ 36M forward tokens at ≈ 80k tokens/s.
- Twenty extra evaluations: ≈ 4 min (135M) / 8 min (360M) per run.
- **135M:** 18 runs ≈ 12 GPU-h.
- **360M:** 6 runs ≈ 9 GPU-h.
- **Total ≈ 21 GPU-h.**

*Reusable from E10.9a* (no LM code in them):
- `e10_active_passive.reflect`. Its evidence is injected as `interpret=` / `revise=`; the control flow, commitment, rule enforcement and absorb are reused unchanged.
- `ReflectionPolicy`, `amie_explanation`, `enforce_random_matched`, `hypothesis_from_name`, `graded_scenario`'s permutation logic, `to_criterion` / `aulc` / `_paired`.
- `ontology_discovery.enforce_rule`, `own_relations`, `edge_pairs`, `tail_pool`.
- `ontology_hypotheses.generate_hypotheses`, `pair_signatures`, `StructuralHypothesis.predict`, `describe_pairs`.
- `self_test.cluster_lower`, `bootstrap_lower`, `adopt`, `Decision`, `HypothesisScore`.
- `learnable_ontology.LearnableOntologyComposer`, which supports attentive mode with slot keys: `add_blank_slot`, `crystallize`, `absorb`, `reopen`, `add_edges`, `penalty`, `project_`, `EdgeTable.build`.

*Not reusable as is.* `interpret_slot`, `slot_self_test`, `score_hypotheses` and `dream_pass` score edits as row deltas against target vectors (`self_test.edit_effects`; bundle mode only, `_check_bundle`). The LM has no targets, and attentive weights are not additive.

*Missing code.*
1. **`src/vsa_embed/lm_ontology.py`** (new):
   - `derive_inverses(ontology, {"owns": "owned_by", "has_part": "part_of"})`: entry heads come from `entry_concepts` / `atomic_names`.
   - `learnable_table(ontology, hidden, coverage, distractor_ratio, max_slots, seed, permute_fraction)` → `EdgeTable`. Factor the permutation out of `graded_scenario` into a shared helper.
   - `lm_context(...)`: a `NodeMap` over entries and atoms, `rule_heads` = all entries.
2. **`src/vsa_embed/lm_reflection.py`** (new):
   - `ReflectionWindows` (fixed windows with an entry → window index; from `data/corpus.py` windows).
   - `frame_edit_effects(model, composer, windows, edit_sets)`: per edit set, the per-window loss change, computed by temporarily adding or masking edges (hypothetical edges at the slot's median mass) and re-running the LM, without gradient under bf16 autocast, only on windows that mention the edited heads.
   - `interpret_slot_lm(composer, slot, ctx, settings)`: the same record schema as `interpret_slot`. It scores hypotheses as predicted-pair minus random-tail contrasts, cluster-bootstrapped over heads, and runs the slot test (utility of the members plus specificity against random tails).
   - `revisit_lm(composer, ctx, revision, *, seed, resamples, alpha)`: the same signature as `dream_pass`.
3. **`src/vsa_embed/training/lm.py`**, opt-in keys with unchanged defaults:
   - `channel.learnable_ontology` in `build_channel` (217–257): build a `LearnableOntologyComposer(mode="attentive")` from `learnable_table`.
   - Loss: add `penalty() / entries in batch` next to the delta penalty (632–633); call `project_()` after `optimizer.step()`.
   - `build_optimizer` (460–472): exempt `edge_mass`, `assignment_logits`, `slot_roles` and `slot_keys` from weight decay.
   - `channel.reflection: {every_tokens, arm, policy}`: a hook after `step += 1` (640–645) that runs `reflect(..., interpret=interpret_slot_lm, revise=revisit_lm)` or the Pc extra steps, then `add_blank_slot`. Parameters replaced by `add_edges` are swapped into the optimizer with `developmental._replace_parameter` (112–127).
   - `eval.every_tokens`, plus an evaluation before and after each round.
   - Reflection rows in `metrics.jsonl`.
   - Checkpoints save the composer's table and slot state, and **restore them unconditionally on resume.** Today the schedule is restored only with a tracker (578–579), so a changed edge count fails to load.
4. **`load_final` (756–778) and `channel_probes.load_run` (318–370)**: rebuild the learnable composer, or export the committed frames as a plain `FrameSchedule`.
5. **`src/vsa_embed/experiments/e10_lm_plan.py`** (new):
   - Configs per (host, arm, seed) from `e9_plan.run_config`. The derived ontology goes under `channel.learnable_ontology`, because `run_config` overwrites `data.ontology` at line 340.
   - Queue at 55 / 56 / 57.
   - A report with tokens-to-criterion and AULC on `after_heldout` / `after_unseen`, and commitment verdicts.
6. **Tests:**
   - `frame_edit_effects` against a direct recomputation on a tiny LM;
   - a tiny-LM end-to-end run with one reflection round;
   - isolation: learning decisions are unchanged when the eval corpus is replaced.

**E9 engagement check (2026-10-02, `runs/e9-check`, SmolLM2-360M, 10M tokens, WordNet general corpus).** Full fine-tuning (host lr 3e-5) and LoRA-64 (2e-4), gate bias 0 and −2: C5 ends identical to C0′ to four decimals in every stratum (all 2.5806, inside 0.916, after 2.470, held-out 2.516). The channel is open (gate ≈ 0.48, injection ≈ 15% of an embedding-row norm) but the host neutralizes it: a pretrained host already models general WordNet vocabulary (inside-span loss 0.92 nats), so it is not rare/OOD for the host. **Decision:** E9 recipe = full fine-tuning, host lr 3e-5, gate bias 0, 50M tokens; primary corpora are vocabularies genuinely new or rare for the host — T5 synthetic enterprise glossary (contamination-free) first, then T4 chemistry and T1-open — with WordNet general as the secondary/negative-control corpus (still measured for the quantization gap).

## E9 on recent Qwen hosts (author request 2026-10-03)

Stronger evidence for the paper: the E9 design (P0 / C0′ / C2 / C5 × bf16, INT8, INT4 × dimensions 1–3) on **Qwen3-1.7B-Base** (main) and **Qwen3-0.6B-Base** (size trend) with LoRA r = 64 + the VSA channel (channel scaled to the host's embedding-row norm, open decision 1), starting on T5. **Qwen3-4B-Base** single seed after a memory/throughput probe measures its cost. **Qwen3.5** (Feb 2026; `qwen3_5`, multimodal `Qwen3_5ForConditionalGeneration`) is not supported by the pinned transformers 4.54; a compatibility check runs in a separate environment, and Qwen3.5-2B-Base becomes the most-recent-model run if it works. Queue order: E9-T5 SmolLM2 block → Qwen memory probe → E9 Qwen blocks (priority 26) → the rest of the recipe sweep, E1, E3. Estimates (to be replaced by the probe): Qwen3-0.6B ≈ 1.2 h per 50M-token run, Qwen3-1.7B ≈ 3 h, Qwen3-4B ≈ 7 h; Qwen3-1.7B seed 1 ≈ 13 GPU-h with evaluations.

**E9 on Qwen3 (WP-Qwen, author request 2026-10-03).** Modern hosts are stronger paper evidence than SmolLM2, so E9 also runs on Qwen3-0.6B/1.7B/4B-Base (downloads and hashes: `~/data/vsa-llm/DATA_SOURCES.md` §13). The recipe is LoRA r = 64 at host lr 2e-4 plus the VSA channel, with channel lr 1e-3, gate bias 0, 50M tokens, 65,536 tokens per step and the same T5 test set. Open decisions 42–46 apply.

- **Hosts (`cpt_plan.HOSTS`).**
  - Widths are 1024, 2048 and 2560; every host has a tied 151,936-row embedding, and the chunked LM loss never builds full logits.
  - `add_lora` finds q/k/v/o/gate/up/down in each layer (7 per layer).
  - Qwen3-4B uses non-reentrant gradient checkpointing.
  - New opt-in keys: `channel.scale_to_host` (`host_scale_fraction` 0.3), `model.host_dtype` (bf16 host, float32 adapters) and `model.checkpoint_use_reentrant`. Their defaults leave every recorded run unchanged.
- **Corpora.**
  - **T5 (built):** `experiments/t5-enterprise-glossary/t5-qwen3.yaml` → `runs/v1-qwen3`, with data in `~/data/vsa-llm/tracks/t5-glossary/v1-qwen3`.
    - Sizes: 100.0M Qwen3 tokens (domain 43.9M, general 56.1M), eval 2.76M tokens, eval-general 2.17M; uint32 ids, NFC.
    - Unchanged from the SmolLM2 build: the generated documents (decompressed sha256), glossary, entries and frames, alias table `5cae33ad…`, holdout `e7313dce…` and zero-shot set `40738889…`. The training-frequency strata also match (662 unseen / 693 rare / 1,979 mid / 866 frequent).
    - Feasible at ℓ_min = 2 with 256 windows. At 1,024 windows: 326 held-out entries / 6,864 spans and 600 rare entries / 4,850 spans.
    - The WP-C7 items are byte-identical; only the frequency annotations of `term_relation_probe` differ. E9 therefore reads the committed zero-shot items for both host families.
    - Dimension-3 items: `experiments/e9-retrofit/items/{new-words,edits}-t5-qwen3-v1` (items identical to SmolLM2's; the Qwen3 checks pass).
  - **T4 (built):** `t4-qwen3.yaml` → `experiments/t4-chemistry/runs/v1-qwen3`. The holdout chosen by the SmolLM2 presample is read through the new `data.holdout_names` key and pinned to `b58e504f…`.
    - Unchanged from SmolLM2: the alias table `ab5351cc…`, the entries and frames, the 299 synthetic concepts and the documents.
    - Sizes: 100.0M tokens (domain 50.8M), eval 1.97M tokens.
    - The zero-shot items are identical. Dimension-3 items: `{new-words,edits}-t4-qwen3-v1`. The new words are identical; the edits differ in the seen half (284 of 597 concepts shared), because their seen entries are sampled by this tokenizer's training frequencies.
  - **T1-open:** `t1-qwen3.yaml`, a host relink `hosts/qwen3`. The GPT-2-presample holdout is pinned. Not built: several CPU-hours.
  - **WordNet:** `host_corpus --host qwen3`. Not built.
  - Commands for T1-open and WordNet are below.
- **Memory probe (`e9_memory`).** The real E9 C5 run (LoRA 64, `scale_to_host`, T5 Qwen3 batches, sequence 1024) at micro-batches 1/2/4/8.
  - Settings: fp32 for every host; with and without checkpointing for 1.7B and 4B; a bf16 host for 4B.
  - It writes `recommendation.json`, which `e9_plan` reads, choosing per host the fastest setting under the budget with an fp32 host.
  - It is one GPU job that needs the whole GPU (≈ 15 min). Queue it first; generate the E9 Qwen3 configs only after its `recommendation.json` exists.
- **E9 Qwen3 stage (`e9_plan`).** The family is chosen by the hosts.
  - Stage `<track>-qwen3`, priority 26: evaluations at 27, `e4_quant` at 28, report at 29.
  - Every model of a host shares that host's dtype and settings.
  - Evaluation jobs use batch 16/8/4 and INT4 probes without WSD.
  - Each block gets its own stage, because quant and report job names are idempotent per stage.

```bash
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
# 1. memory/throughput probe (one GPU job, whole GPU)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e9-qwen3-memory-probe --priority 26 --no-resume -- \
  $PY -m vsa_embed.experiments.e9_memory --output experiments/e9-retrofit/memory/qwen3-v1
# 2. after experiments/e9-retrofit/memory/qwen3-v1/recommendation.json exists: the 1.7B + 0.6B block (P0, C0′, C2, C5; seed 1)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t5 --hosts Qwen3-1.7B-Base Qwen3-0.6B-Base --host-mode lora --lora-rank 64 --queue
# 3. later: Qwen3-4B, single seed, its own stage (consider --tokens 25000000 or --models P0 C0p C5; decision 45)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t5 --hosts Qwen3-4B-Base --host-mode lora --lora-rank 64 --stage t5-qwen3-4b --queue
# T4 Qwen3 block (corpus and items built): as step 2 with --track t4 (stage t4-qwen3)
# corpora for the later tracks (CPU; WordNet and T1 ≥ 1 h with 3–4 workers; not run)
PYTHONPATH=src $PY -m vsa_embed.experiments.t1_open_corpus --config experiments/t1-open-clinical/t1-qwen3.yaml --output experiments/t1-open-clinical/runs/v1-qwen3
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_tracks items --track t1 --kind new --family qwen3 --output experiments/e9-retrofit/items/new-words-t1-qwen3-v1
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_tracks items --track t1 --kind edits --family qwen3 --output experiments/e9-retrofit/items/edits-t1-qwen3-v1
PYTHONPATH=src $PY -m vsa_embed.experiments.host_corpus --host qwen3 --c3-run experiments/c3-general-corpus/runs/v2 --workers 3
PYTHONPATH=src $PY -m vsa_embed.experiments.e5_zeroshot items --scenario c3_synthetic --ontology ~/data/vsa-llm/c3/wordnet-qwen3-v1/ontology.pt \
  --tokenizer Qwen/Qwen3-0.6B-Base --output experiments/e5-explainability/items/c3-synthetic-qwen3-v1
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_tracks items --track wordnet --kind new --family qwen3 \
  --reserved-names experiments/e5-explainability/items/c3-synthetic-qwen3-v1 --output experiments/e9-retrofit/items/new-words-qwen3-v1
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_tracks items --track wordnet --kind edits --family qwen3 --output experiments/e9-retrofit/items/edits-qwen3-v1
```

**Qwen3.5 feasibility (WP-Qwen, 2026-10-03; separate env only).**

- **Environment:** `~/venvs/vsa-qwen35`, a `venv --system-site-packages` over `vsa-repro` (torch 2.11.0+cu128 inherited) with transformers 5.18.0, huggingface_hub 1.33.0, tokenizers 0.23.2 and safetensors 0.8.0. `vsa-repro` itself is untouched.
- **Models:** Qwen3.5-0.8B-Base and Qwen3.5-2B-Base, commits in `DATA_SOURCES.md` §13.
- **Findings:**
  - **Loading.** `AutoModelForCausalLM` loads the `Qwen3_5ForConditionalGeneration` checkpoint text-only as `Qwen3_5ForCausalLM`. It strips the `model.language_model.` prefix and ignores `mtp.*` and `model.visual.*`, with no missing or unexpected keys. 2B-Base has 1.88B text parameters, 24 layers (18 Gated DeltaNet linear-attention, 6 gated full-attention), width 2048, a tied 248,320-row head and a mean embedding row norm of 0.68.
  - **ChannelLM.** `get_input_embeddings`, `base_model` (`Qwen3_5TextModel` with `inputs_embeds`) and `get_output_embeddings` work. The chunked loss equals the HF loss exactly (2.83256). A C5 channel with LoRA and non-reentrant checkpointing runs forward and backward (CPU).
  - **LoRA.** The default targets reach q/k/v/o only in the 6 full-attention layers, plus gate/up/down in all 24 MLPs. The linear-attention projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) are not reached, so 18 of 24 token mixers would stay frozen.
  - **Kernels.** `flash-linear-attention` and `causal-conv1d` are absent, so transformers falls back to its PyTorch reference implementation, which it describes as "correct but much slower".
  - **Tokenizer.** A new one: 248,077 ids, NFC, fingerprint `4d765ac5…`, not Qwen3's.
  - **Test suite.** 498 passed, 1 skipped under transformers 5.18 on CPU. The only API drift found is a deprecation warning for `torch_dtype`.
- **Needed for E9 on Qwen3.5-2B-Base:**
  1. Run its jobs with the venv's interpreter. `e9_plan` queues `sys.executable`, so call it with `~/venvs/vsa-qwen35/bin/python`.
  2. Install `flash-linear-attention` and `causal-conv1d` in the venv. The latter needs a CUDA build.
  3. Add an opt-in `model.lora_targets` key covering the Gated DeltaNet projections.
  4. Add a `Qwen3.5-2B-Base` host and a `qwen3_5` family: corpora (the T5 build takes ≈ 3 min), items, and `torch_dtype` → `dtype`.
  5. Measure memory. The 248k vocabulary doubles the chunked-loss logits; `loss_chunk` may need lowering.
- **Risks:**
  - Speed and memory of the fallback path if the kernels do not build.
  - Every result would come from a second transformers major version.
  - The channel's effect passes through recurrent state rather than attention in 3/4 of the layers.
  - The checkpoint is a natively multimodal base, used here text-only.
  - INT4 tinygemm has not been tried on the new layer shapes.

**E9 on Qwen3.5 (WP-Qwen35, 2026-10-03).** Qwen3.5-2B-Base (main) and Qwen3.5-0.8B-Base (size trend) run the Qwen3 E9 recipe
(LoRA r = 64 at host lr 2e-4, channel lr 1e-3, gate bias 0, `scale_to_host`, 50M tokens, 65,536 tokens per step, T5 first) in the
separate environment; the pinned `vsa-repro` is unchanged. Open decisions 47–50 apply.

- **Environment** (`experiments/e9-retrofit/env/`: lock file `qwen35-requirements.txt`, notes `README.md`):
  `~/venvs/vsa-qwen35` over `vsa-repro` (torch 2.11.0+cu128, Triton 3.6.0, torchao 0.18.0 inherited) with transformers 5.18.0,
  tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0, flash-linear-attention / fla-core 0.5.2 and causal-conv1d 1.7.0
  (built from the sdist for torch 2.11 / CUDA 12.8 / sm_86 with a build-only conda-forge nvcc 12.8.93 in `~/venvs/cuda-12.8`;
  no prebuilt wheel exists for torch 2.11).
- **Fast path.** transformers binds the Gated DeltaNet functions to fla / causal-conv1d at import, whatever the device (a CPU
  forward then fails); `integrations.linear_attention.install_device_dispatch` (applied by `build_model` to linear-attention
  hosts) sends CUDA tensors to the kernels and CPU tensors to the PyTorch reference and counts the calls. Every Qwen3.5 run
  prints and records `linear_attention_kernels` (bound implementations, calls per implementation, interpreter) in its manifest.
  A GPU smoke at the 0.8B host's layer shapes (`linear_attention_smoke`, `experiments/e9-retrofit/env/kernel-smoke-shapes-v1`)
  ran every Gated DeltaNet call on the fast path at 2.3× the reference speed (forward+backward) with matching gradients
  (cosine 0.9999); the memory probe measures the full-size speed-up (`train-reference` row). End to end on CPU, the real
  Qwen3.5-0.8B trains through `training.lm` (186 adapters) and reloads through `channel_probes.load_run` with INT8.
- **Hosts (`cpt_plan.HOSTS`).** Width 2048 / 1024, tied 248,320-row heads, a `python` field (`~/venvs/vsa-qwen35/bin/python`) so
  every job of these hosts — training, evaluations, `e4_quant`, report — runs in the venv and every other host's with the
  pinned interpreter (`cpt_plan.pinned_python`, also when planning from the venv). New opt-in keys, absent elsewhere:
  `model.lora_targets: linear_attention` (in_proj_qkv / z / a / b and out_proj besides q/k/v/o/gate/up/down: 8 adapters per
  Gated DeltaNet layer, 7 per full-attention layer, every token mixer covered; recorded as `lora_coverage`) and
  `model.loss_chunk: 1024`. These hosts load with `use_cache` off. INT8 and INT4 (tile-packed) PTQ quantize every Qwen3.5
  linear layer, including the 16-row `in_proj_a/b`.
- **T5 corpus (built, CPU ≈ 10 min):** `experiments/t5-enterprise-glossary/t5-qwen35.yaml` → `runs/v1-qwen35`, data in
  `~/data/vsa-llm/tracks/t5-glossary/v1-qwen35`. 100.0M Qwen3.5 tokens (domain 45.5M, general 54.5M), eval 2.84M, eval-general
  2.19M; uint32 ids, NFC, tokenizer fingerprint `4d765ac5…`. Identical to the SmolLM2 and Qwen3 builds (`runs/v1-qwen35/crosscheck.json`):
  the generated documents (decompressed sha256 `cc311dd2…` / `f00b4d43…`), alias table `5cae33ad…`, holdout `e7313dce…`,
  zero-shot set `40738889…`, entries, frames and held-out entries, and the training-frequency strata (102 unseen, 693 rare,
  1,979 mid, 866 frequent non-held-out entries). Feasible at ℓ_min = 2 with 256 windows; at 1,024 windows 326 held-out
  entries / 6,601 spans and 592 rare entries / 4,782 spans. WP-C7 items byte-identical except the frequency annotations of
  `term_relation_probe` (as for Qwen3). Dimension-3 items `experiments/e9-retrofit/items/{new-words,edits}-t5-qwen3_5-v1`:
  `items.jsonl` / `concepts.jsonl` byte-identical to the SmolLM2 and Qwen3 ones.
- **T4:** `experiments/t4-chemistry/t4-qwen35.yaml` (frozen SmolLM2 holdout `b58e504f…`), not built. T1-open and WordNet relinks
  for Qwen3.5 have roots in `e9_tracks.QWEN35_ROOTS` but no configs yet.
- **Memory probe** (`e9_memory --hosts Qwen3.5-0.8B-Base Qwen3.5-2B-Base`): fp32 host with and without checkpointing at
  micro-batches 1, 2, 4; kernel calls per row and one reference-path row per host; `recommendation.json` under
  `experiments/e9-retrofit/memory/qwen3_5-v1`, which `e9_plan` reads by default for this family.
- **E9 stage `t5-qwen35`**: priority 52 (evaluations 53, `e4_quant` 54, report 55); evaluation batches 16 (0.8B) / 8 (2B),
  INT4 probes without WSD. Estimates before the probe (FLOP-scaled from the measured Qwen3 probe): 2B ≈ 3.0 h per 50M-token
  run if micro-batch 2 fits without checkpointing, ≈ 4.5 h with it; 0.8B ≈ 2.1 h; the block (3 trained runs per host + P0,
  evaluations, quantization, report) ≈ 20–25 GPU-h; the probe ≈ 0.5 h.

```bash
PYQ=/home/bhux/venvs/vsa-qwen35/bin/python
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
# 1. Qwen3.5 memory/throughput probe (one GPU job, whole GPU; venv interpreter)
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name e9-qwen35-memory-probe --priority 52 --no-resume -- \
  $PYQ -m vsa_embed.experiments.e9_memory --output experiments/e9-retrofit/memory/qwen3_5-v1 --hosts Qwen3.5-0.8B-Base Qwen3.5-2B-Base
# 2. after experiments/e9-retrofit/memory/qwen3_5-v1/recommendation.json exists: the 2B + 0.8B T5 block (P0, C0′, C2, C5; seed 1):
#    59 jobs (8 training at 52, 48 evaluations at 53, 2 e4_quant at 54, report at 55), every one with $PYQ (either interpreter plans)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t5 --hosts Qwen3.5-2B-Base Qwen3.5-0.8B-Base --host-mode lora --lora-rank 64 --queue
# T4 Qwen3.5 corpus and items (CPU, venv; then step 2 with --track t4, stage t4-qwen35)
PYTHONPATH=src $PYQ -m vsa_embed.experiments.track_corpus --config experiments/t4-chemistry/t4-qwen35.yaml --output experiments/t4-chemistry/runs/v1-qwen35
PYTHONPATH=src $PYQ -m vsa_embed.experiments.e9_tracks items --track t4 --kind new --family qwen3_5 --output experiments/e9-retrofit/items/new-words-t4-qwen3_5-v1
PYTHONPATH=src $PYQ -m vsa_embed.experiments.e9_tracks items --track t4 --kind edits --family qwen3_5 --output experiments/e9-retrofit/items/edits-t4-qwen3_5-v1
```

## E9 paper-quality controls, dimensions 1–2 (WP-PQ1, 2026-10-03)

Response to `manuscript/novelty-check-2026-10.md` for claims A and B (§2.2 wording audit; §2.5 A-B1/A-B2/A-B4 same-site
and text baselines, A-B5 lexical overlap, A-B6 operator/specificity; §3.4 B-B2 channel off, B-B3 quantizers, B-B6 DiD
reporting, B-B7 quantized embedding; §6.5 item 1 KnowLA-style KGE). Dimension 3 (editing baselines), E10 and the
manuscript wording are WP-PQ2.

- **Arms** (`e9_plan --models …`; same recipe, test set, seeds and channel site/gate as C5; opt-in, defaults unchanged):

  | Arm | What changes vs C5 | Answers |
  |---|---|---|
  | `C5rf` | operator `random_fixed:unitary_hrr`: a fixed random orthogonal (unit-spectrum circulant) operator per relation, never trained | does *learning* the binding matter (claims rule 7: "compositional parameter sharing" if C5 ≈ C5rf) |
  | `C5ut` | operator `untyped`: no binding, the concept vector bundles its filler atomics (attention keys still see relation ids) | does relation binding matter at all |
  | `C5tr` | operator `translation` `x + t_r` (TransE-style, `relations.AFFINE_FAMILIES`; affine, so not a binding) | multiplicative binding vs additive relation offsets |
  | `C5sh` | `frames: shuffled`: each entry reads another entry's frame (a seeded derangement) | frame content vs a generic "domain term here" signal |
  | `C6m` | rows = subtoken mean of the pretrained host's input embeddings over the term's aliases (FVT / Hewitt), frozen | the default "add the term as a token" initializer, as a same-site zero-shot competitor (A-B1 i / A-B2) |
  | `C6d` | rows = frozen host's last-layer state, mean-pooled over the entry's verbalized frame ("It is a process. It is owned by the …"; name excluded) | a text encoder of the same frame information (A-B1 ii / A-B4) |
  | `C6g` | rows = TransE embedding (d 256, L1, self-adversarial negatives) of the entry in the track ontology | KnowLA / map-tuning per-concept KG vectors at our site (§6.5 item 1) |

  C6 rows go through a trained GELU MLP `source → h → d` with `h·(source + d)` matched to C5's dictionary-plus-projector
  budget (the C2 budget), at the C5 gate and site (`channel.mode: source`); tables are whitened per dimension over the
  non-held-out entries and unit-normalized, built by `e9_rowsource` into `~/data/vsa-llm/e9/row-sources/<track>-<family>/`
  (the verbalization table `verbalized.jsonl.gz` beside them records the encoded text). Held-out terms get rows from
  the same source (their aliases, frames and triples exist), so the C6 arms are fair zero-shot competitors, unlike C2.
- **Filler / non-filler split** (claim A's copy concern): `X_filler` = targets in the 8-token window after a span that
  belong to an occurrence (starting after the span) of a token sequence verbalizing a filler of *that span's* frame
  (aliases of a concept-filler, lexicon wording, value; article stripped; as written / lower / capitalized; ± leading
  space), `X_nonfiller` = the rest, for every after-span stratum (held-out, rare, unseen, 3+-subtoken, …). Filler tables
  per track and tokenizer (`e9_rescore fillers`, `~/data/vsa-llm/e9/filler-tables/`). The same masks are the trainer's
  opt-in `eval.filler_strata` and `e9_rescore`'s re-scoring of finished `final.pt` files on the run's own windows
  (`RUN/rescore/`; the old strata replay the run's `eval_windows.npz` exactly — tested). Share of filler tokens among the
  after-span targets (first 256 evaluation windows): T5 18.6% (`after`), 21.1% held-out, 22.0% rare-seen, 23.9% unseen;
  T4 3.7%, 0.8% held-out, 10.3% rare-seen, 28.9% unseen (T4's text is mostly PubMed abstracts and ChEBI definitions).
  Tables built: filler tables T5 and T4 (SmolLM2 tokenizer); row sources T5 (FVT and definition for both SmolLM2 hosts,
  TransE: raw training tail MRR 0.73, hits@10 1.00) and T4 FVT for both hosts; T4 definition and TransE tables are built
  by the queued jobs (minutes on the GPU).
- **Claim-B controls** (`e9_rescore` variants, all paired by window with the run's other variants): `ref-off` /
  `int4-A-off` (the channel switched off: what the C5-trained host weights do alone, B-B2); `int4-hqq`, `int4-nf4`
  (torchao algorithms, NF4 bit-exact with `NF4Tensor`), `int4-gptq` (act-order, static groups, 1% damping, block-
  sequential, 64 calibration windows of the run's in-domain training mix), `int4-awq` (activation-aware scale search, no
  clip search), `int4-rtn` (the same grid without calibration), all simulated as dequantized weights; `int4-A-emb` (the
  input-embedding table at 4 bits too; a tied head keeps 16 bits; `-embhead` quantizes both). torchao 0.18's own GPTQ /
  AWQ prototypes target the plain `Int4Tensor` layout (needs `mslk`, which has no usable release here) — not used.
- **Report** (`e9_report`, new sections when their inputs exist): operator ablation (`C5 − arm` and `arm − C0′` per
  stratum, Holm over arms), same-site row sources (the same, `after_heldout` first), filler / non-filler table (gain on
  each part, share of targets and of the gain on filler tokens), claim-B table (absolute nats in the four cells, gaps,
  `DiD = gap@q − gap@bf16`, `DiD off`, the channel's own DiD; paired window bootstraps; Holm over quantizers).
- **Job chaining.** Arms at priority P: training; at P + 1 the `pq` evaluations (track zero-shot and editing at bf16; C5sh
  without the `random_frame` source and the edit items, which do not apply to shuffled frames) and `e9_rescore` (`ref`,
  `int4-A`); no `e4_quant` (the rescoring has `int4-A`); missing C6 tables are built by a job at P queued before the
  training jobs (GPU: seconds for T5, minutes for T4). `--rescore all` also rescores P0 / C0′ / C2 / C5 with the
  `controls` variants. A batch with arms gets its own quant/report job names (`…-s1-2-3-pq-<hosts>`, e.g. `t5-report-s1-2-3-pq-SmolLM2-360M`), so it never collides with a
  queued base batch, and a later arm batch on another host (Tier 3) reports again.

**Decisions (WP-PQ1).**

| # | Decision | Taken |
|---|---|---|
| PQ1-a | C5rf's operator | `random_fixed:unitary_hrr` (exactly orthogonal, as asked), not E2/E4's `random_fixed:hrr` (random non-unitary roles); both are frozen HRR-family binding, so the comparison with E4 C8 stays close |
| PQ1-b | C5ut keeps relation-typed attention keys | as the program's `untyped` operator everywhere (only the binding is removed); a relation-blind bag is C3t-style and not run |
| PQ1-c | C6 projector | 2-layer GELU MLP matched to C5's dictionary + projector (C2's budget); the C6 table is a non-persistent buffer (re-read from its file, counted in deployment bytes) |
| PQ1-d | C6d text and layer | the verbalized frame with subject "It" (no name), last hidden layer, mean over tokens; natural definitions (T4 ChEBI, T1 MeSH scope notes) not used, so every track encodes exactly the frame C5 composes |
| PQ1-e | C6g graph | transductive TransE over all entries' frames (held-out included, as C5 reads their frames); an atomic naming a concept is that concept's entry |
| PQ1-f | the opt-in `eval.filler_strata` is not set by `e9_plan` | every run keeps identical strata lists (e4_quant / report pairing unchanged); the split comes from `e9_rescore` for all runs alike |
| PQ1-g | dimension 3 for the arms | bf16 only, no probes; C6 arms have no row for an invented word (new words read the mean source row = a no-information control) |
| PQ1-h | arms run in the base stages (`t5`, `t4`) | one report per track covers base runs, arms and rescoring; priority 51 (evaluations 52, e4_quant 53, report 54) |

**Smoke (2026-10-03, real T5 data, SmolLM2-135M, 2 training steps, 8 evaluation windows; CPU, plus ≤ 1.2 GB GPU
checks).** Every arm (C5rf, C5ut, C5tr, C5sh, C6m, C6d, C6g) plus P0 / C0′ / C2 / C5 trains and evaluates through the
plan's own configs; `e9_rescore` reproduces each run's evaluation windows exactly (max |Δ window sum| 0.0 on all 11
runs) and adds the filler strata (805 of 3,806 after-span targets are filler tokens); the simulated quantizers, channel
off and `-emb` run on the CPU, and torchao `int4-A` / `int4-A-off` / `int4-A-emb`, HQQ, NF4, GPTQ (22 s) and AWQ (16 s)
on CUDA; the report renders all four new sections. The GPU smoke caught one bug, fixed: calibration statistics computed
inside the forward's bf16 autocast were bf16 Gram matrices (not PSD); they are now float32 (tested), and the GPTQ inverse
is computed in float64 with escalating damping.

**Measured costs** (T5 seed-1 block, RTX 3090): training per 50M-token run SmolLM2-360M 68 min, 135M 31 min; one
evaluation pass over the 1,024 windows (≈ 1M targets) 24 s / 12 s at bf16, 66 s / 35 s with torchao INT4; per-run
zero-shot + editing at bf16 ≈ 3 / 1.7 min; the R9 report ≈ 22 min. Estimates: `e9_rescore` light (`ref`, `int4-A`)
≈ 2 / 1 min per run; controls (9 variants incl. GPTQ / AWQ calibration) ≈ 9 / 5 min per run.

**Budgeted plan** (GPU-h; ≥ 3 seeds on the decisive comparisons C5 vs C5rf / C5ut / C5sh / C6m / C6d / C6g on T5 and
T4 with SmolLM2-360M; C5tr rides along with 3 seeds, first to drop):

| Tier | Block | New jobs | GPU-h |
|---|---|---:|---:|
| 1 | T5, SmolLM2-360M: 7 arms × seeds 1–3 (training 21 × 1.14 h; `pq` evaluations + light rescoring ≈ 5 min each) + controls rescoring of P0 / C0′ / C2 / C5 seeds 1–3 (10 × ≈ 9 min) + quant re-analysis + report | ≈ 97 | ≈ 28 |
| 0b | controls rescoring of the T5 SmolLM2-135M base runs (10) and the T4 135M seed-1 runs (4) | 14 | ≈ 1 |
| 2 | T4, SmolLM2-360M: base C0′ / C2 / C5 seeds 2–3 (6 × ≈ 1.2 h + full evaluations + e4_quant) + 7 arms × seeds 1–3 (21 × ≈ 1.2 h) + tables (definition, TransE: minutes) + controls rescoring (10) + report | ≈ 141 | ≈ 39 |
| 3 (optional) | size trend: T5, SmolLM2-135M, 7 arms × seeds 1–3 (21 × 0.52 h + evaluations) | ≈ 85 | ≈ 12.5 |

Tiers 1, 0b and 2 ≈ 68 GPU-h; with Tier 3 ≈ 80 GPU-h. Dropping C5tr saves ≈ 3.7 GPU-h per track and host. Optional
claim-B rescoring of the Qwen3 T5 base runs (`e9_rescore queue --stage t5-qwen3 --priority 51 --models P0 C0p C2 C5`;
1.7B ≈ 0.5 h, 0.6B ≈ 0.2 h per run) ≈ 5 GPU-h for seeds 1–2.

```bash
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
# Tier 1, T5 (360M): arms × 3 seeds; P0/C0p/C2/C5 already exist or are queued (their jobs are skipped by
# name) and get the controls rescoring through --rescore all
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t5 --stage t5 --hosts SmolLM2-360M \
  --models P0 C0p C2 C5 C5rf C5ut C5tr C5sh C6m C6d C6g --seeds 1 2 3 --rescore all --priority 51 --queue
# Tier 0b: the T5 135M base runs (filler split and claim-B controls for the size trend)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_rescore queue --stage t5 --priority 51 --models P0 C0p C2 C5
# Tier 2, T4 (360M): base seeds 2–3, arms × 3 seeds, rescoring of every base run (seed 1 is queued at 35–38)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t4 --stage t4 --hosts SmolLM2-360M \
  --models P0 C0p C2 C5 C5rf C5ut C5tr C5sh C6m C6d C6g --seeds 1 2 3 --rescore all --priority 51 --queue
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_rescore queue --stage t4 --priority 51 --models P0 C0p C2 C5
# Tier 3 (optional): T5, SmolLM2-135M arms
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t5 --stage t5 --hosts SmolLM2-135M \
  --models C5rf C5ut C5tr C5sh C6m C6d C6g --seeds 1 2 3 --priority 51 --queue
```

## 3-day GPU block (author request 2026-10-03, ≈ 72 GPU-h, measured costs)

Measured: SmolLM2 E9 block (360M + 135M, one track, one seed, with evaluations) ≈ 6.5 GPU-h; Qwen3 training per 50M-token LoRA run 0.6B 1.7 h, 1.7B 3.2 h, 4B 11.4 h (memory probe `experiments/e9-retrofit/memory/qwen3-v1`).

| Order | Block | Priority | GPU-h |
|---|---|---|---:|
| running | Qwen3-1.7B + 0.6B on T5, seed 1 (`t5-qwen3`) | 26–29 | ≈ 18 |
| 0 | E3 (D3, developmental WordNet) | 30 | ≈ 2 |
| 1 | SmolLM2 on T5, seeds 2–3 | 31–34 | ≈ 13 |
| 2 | SmolLM2 on T4 chemistry, seed 1 | 35–38 | ≈ 6.5 |
| 3 | SmolLM2 on T1-open (PubMed/MeSH), seed 1 | 39–42 | ≈ 7 |
| 4 | SmolLM2 on WordNet general (negative control), seed 1 | 43–46 | ≈ 6.5 |
| 5 | Qwen3-1.7B + 0.6B on T5, seed 2 | 47–50 | ≈ 18 |
| 6 | E7 D7.1 authoring quality | 60 | ≈ 1 |

Not in this block (author decision pending): Qwen3-4B (≈ 11.4 h per run; a lite P0/C0′/C5 at 25M tokens ≈ 15 h), Qwen3.5 (separate transformers 5.18 environment), the committed E4 core (D4.0 shake-out → D4.8/D4.7/D4.9/D4.1) and E10.2, which follow once the recipe sweep is analyzed.

## After the novelty check (2026-10-03)

`manuscript/novelty-check-2026-10.md`: the method is novel with narrowed wording; claim C's general framing ("edit a symbolic store instead of weights") is pre-empted (Facts-as-Experts, KBLaM, REMEDI, LMLM) — the bound-edge editing mechanism is new; claim B's headline is not supported without controls; claim D is novel only as a protocol and stays out of the abstract until E10.2; avoid the bare name "semantic channel". Queued/being built in response:

- **WP-PQ1** (E9 dimensions 1–2): operator/specificity ablation of C5 (random_fixed, untyped, translation, shuffled frames); same-site row-source baselines (subtoken-mean / FVT, definition encoder, KG embedding; parameter-matched); filler vs non-filler loss split (the copy concern on templated T5 text) incl. rescoring finished runs; quantization controls (channel off at INT4, GPTQ/AWQ/HQQ/NF4 beyond RTN, quantized embeddings, difference-in-differences). Queued at priority 51 with ≥ 3 seeds on the decisive comparisons (T5 and T4, SmolLM2-360M).
- **WP-PQ2** (E9 dimension 3, E10, manuscript): in-context upper bound (verbalized frame / definition; IKE), ROME/MEMIT(/AlphaEdit) on the same edit items, frame-transplant and channel-off audits; E10 baselines (rule mining, IterE-style induction, KG link prediction, no-hidden-relation world, validation-reuse control); manuscript renaming and claim wording; `manuscript/paper-readiness.md` audit of every experiment.
- **Qwen blocks:** Qwen3-4B on T5 (priority 56, ≈ 34 GPU-h training + evaluations); Qwen3.5-2B/0.8B on T5 in the `~/venvs/vsa-qwen35` environment (probe at 52, block queued automatically after it, ≈ 20–25 GPU-h).

**Queue after WP-PQ1 (2026-10-03).** PQ tiers 1 (T5, SmolLM2-360M, 7 arms × 3 seeds + controls rescore), 0b (rescore of existing base runs) and 2 (T4, SmolLM2-360M: base seeds 2–3 + 7 arms × 3 seeds + rescore) queued at priority 51–54 (≈ 68 GPU-h); tier 3 (T5 135M arms) not queued. Total queued GPU time ≈ 190 h (≈ 8 days): remainder of the 3-day block (≈ 60 h, priorities ≤ 50) → PQ controls (≈ 68 h, 51–54) → Qwen3.5-2B/0.8B T5 (≈ 25 h, 52–55; probe first) → Qwen3-4B T5 (≈ 40 h, 56–59) → E7 D7.1 (60). Filler share of after-span targets: T5 19–24%, T4 1–10% (T4 is the clean test of the copy concern).

## Author decisions of 2026-10-07 (after the preliminary results report)

| # | Decision | Detail |
|---|---|---|
| 54 | E4 core → **reduced operator screen** (report option 2) | Stage `opscreen`: 50M from scratch on the C3 corpus, frozen D4.0 recipe (32,768 tokens/step, peak lr 2e-3, warmup 5M, cosine to 0.1×), 100M tokens, 1,024 evaluation windows with per-window losses; C0, C2, C5, C5@C5tr (`translation`), C5@C5ut (`untyped`), C5@C5rf (`random_fixed:unitary_hrr`) × seeds 1–3 (18 runs ≈ 8 GPU-h) + channel probes + `e4_report`; priority 51 (after the T4 block). Not D4.9 and not pooled with it. **Fixed before any run:** primary = C5 − arm relative loss on `after_heldout` and `after` (paired window bootstrap, seeds pooled, Holm over C5ut and C5rf); secondary = CARD-660 held-out Spearman; C5tr reported alongside. Reading: C5 not better than both C5ut and C5rf (a CI including 0 or favouring the arm) → "compositional parameter sharing" in every regime (rule 7); C5 better than both with CIs excluding 0 → a binding claim for from-scratch training only. The full E4 core stays unqueued |
| 55 | A natural track with **genuinely new vocabulary** for E9 | T2 is templated synthetic text (decision 28) and T3/T6 lack a rare stratum (decision 15), so candidate tracks are ranked by measured novelty to the host before one is built and queued |
| 56 | **Claim C power** | New-word and edit items grown to ≈ 700 per arm (items `-v2`, a superset of `-v1`) before the dimension-3 baselines run; the T5 dimension-3 evaluations are re-run on `-v2` |
| 57 | **Qwen3 seed 3** | Qwen3-1.7B/0.6B-Base on T5, seed 3 (C0′, C2, C5), priorities 55–58 (after the Qwen3.5 block) |
| 58 | **Clinical track from the credentialed files** in `data/` (SNOMED CT International 2022-05-31, UMLS 2022AB, MIMIC-III 1.4 `NOTEEVENTS`, MIMIC-IV 3.1) | Supersedes decision 1's open substitute where the licensed sources apply. DUA and licence rules: `data/` is git-ignored; nothing derived from MIMIC, SNOMED CT or UMLS is committed (derived tables under `~/data/vsa-llm/`); MIMIC text is never sent to an external service (no `claude -p` teacher or judge on MIMIC text; Claude Code sessions read only file names, sizes and column headers, never note contents) |
| 59 | **Read-to-learn** (one-shot vocabulary from reading definitions or a textbook glossary) | New experiment for the self-reflective-learning use case: read a definition → frame → compose the new term's vector, no gradient step; design and pre-registration before any run |

## T1c — the licensed clinical track (decision 58; WP-T1c, 2026-10-07)

SNOMED CT International 2022-05-31 (ontology) × MIMIC-III 1.4 `NOTEEVENTS` (corpus), labelled **"clinical (SNOMED CT +
MIMIC-III)"**; T1-open (MeSH × PubMed) stays as its open counterpart. Code: `ontologies/snomed.py`, `data/mimic.py`,
`data/clinical_inventory.py`, `experiments/t1c_corpus.py` (stages `inventory`, `extract`, `build`, `zeroshot-items`,
`definitions`), `experiments/t1c_icd.py` (proposal (a) coverage counts); configs `experiments/t1c-clinical/t1c.yaml`
and `t1c-qwen3.yaml`; track `t1c` in `e9_tracks` / `e9_plan` (`--track t1c`; `--dry-run` prints jobs and GPU-hours).
Tests: `tests/test_t1c_clinical.py` (synthetic RF2 / NOTEEVENTS fixtures only).

**Licence and DUA.** Everything derived from MIMIC, SNOMED CT or UMLS lives under `~/data/vsa-llm/t1c/` (mode 700):
the shuffled note shards, corpora and `ontology.pt`, the holdout lists, the evaluation alias table (`e9/t1c.json`) and
every item set (`items/`). Committed: code, configs (paths and hyper-parameters), aggregate run folders
(`experiments/t1c-clinical/runs/`) and `items-digests.json`. `.gitignore` keeps only `manifest.json`,
`metrics.jsonl`, `resolved_config.yaml` and `probes*.json` of any `runs/t1c*` E9 run folder. No note text is printed
or sent anywhere; files were scanned for SNOMED alias strings before committing.

**Build (`runs/v2`, Qwen3 relink `runs/v2-qwen3`; linker and holdout settings committed before linking).** The first
build (`runs/v1`, holdout `fa738430…` at fraction 0.10) is stale: the author re-froze the holdout at fraction 0.30 on
2026-10-07 (T1c-1) from the same pre-sample, before any run; documents, linker and alias policy are unchanged.

- Ontology: 310,698 active concepts of seven top-level hierarchies (clinical finding 117,002; procedure 58,730; body
  structure 40,443; organism 32,590; substance 27,125; product 24,681; observable entity 10,127); 62 relations
  (hierarchy, semantic tag, is-a and 59 inferred attribute types with ≥ 200 edges: finding site 89.7k edges,
  associated morphology 63.2k, method 60.2k, interprets 38.1k, direct procedure site 33.1k, …); 8,192 atomics; degree
  ≤ 16 (767 frames truncated).
- Linker (decision 19's lesson): FSN without its tag + synonyms accepted in the US/GB refsets; 1,560 bare acronyms,
  4 function words, 6 aliases < 3 characters and 59,502 descriptions > 8 words dropped → 496,997 aliases over 295,143
  entries (evaluation alias table `e0ab5fa3…`; its digest includes the holdout, v1 was `18dc1545…`). SmolLM2 subtokens
  per alias 1/2/3/4/5+: 1,718 / 20,257 / 46,231 / 65,623 / 363,168.
- Notes: 2,083,180 rows (file sha256 = PhysioNet's); 886 `ISERROR` and 68 empty dropped; patient split (sha256 bucket
  < 1,000/10,000 → evaluation): 41,501 / 4,645 patients, 1,881,250 / 200,976 notes, no patient on both sides.
- SmolLM2 corpora (the reference): `train` 130.0M tokens (note share 0.500, exact), `eval-mimic` 22.35M (first 40,000
  shuffled evaluation-patient notes), `eval-general` 5.0M (C3's evaluation documents: locality), `eval` 10.0M (mixed).
  Qwen3 relink built (`hosts/qwen3`, `runs/v2-qwen3`): train 130.0M (share 0.504), `eval-mimic` 22.2M.
- Holdout (T1-open's rule at fraction 0.30, T1c-1; pinned `7369ec276b45465900cc3a9898381392f52fa99d47821a4c6ad80df5eae8cec5`,
  superseding `fa738430…`): 479 entries chosen (30% of 1,596 eligible; 2,541 hub entries not eligible) + 1,595 by the
  closure = 2,074 entries (2,069 concepts); node- and alias-disjoint (asserted). Fraction 0.10 on the same pre-sample
  reproduces `fa738430…` exactly.

**Feasibility** (decision 17's bar: ≥ 2,000 occurrences and ≥ 300 entries with ≥ 5 occurrences; SmolLM2, ℓ_min 2;
Qwen3 within ±10%):

| stratum | whole `eval-mimic` (21,827 windows): occ. / entries / entries ≥ 5 | verdict | E9's 2,048 windows: occ. / entries ≥ 5 | verdict |
|---|---|---|---|---|
| held-out | 40,690 / 442 / 347 (v1: 21,294 / 208 / 153) | **feasible** | 3,638 / 161 | occurrences met, entries ≥ 5 not (12,288 windows needed) |
| unseen (freq 0) | 1,365 / 786 / 32 | exploratory-feasible | 117 / 1 | infeasible |
| rare-seen (1–9) | 6,592 / 2,636 / 316 | feasible | 668 / 3 | infeasible (the whole split is needed) |
| 3+-subtoken | 377,116 / 8,901 / 4,242 | feasible | 35,564 / 1,198 | feasible (1,024 windows suffice) |

So the held-out and rare-seen gate strata are read on the whole split (T1c-2: `e4_quant --windows 21827` on the final
checkpoints); the 2,048-window evaluations of the trainer stay the curves and the 3+-subtoken gate. The held-out stratum
is bounded by how many distinct held-out terms occur in notes (SNOMED's compound aliases make frequent terms
closure-heavy hubs); note-level enrichment concentrates the strata only 1.5×.

**Novelty to the host (P0, SmolLM2-360M frozen; `runs/t1c-novelty-v2`, 2,048 windows, 2 min, 3.2 GB; v1 in
`runs/t1c-novelty`, stale).**

| P0 | T5 | T4 | T1-open | WordNet | **T1c** |
|---|---:|---:|---:|---:|---:|
| loss inside a term | 3.38 | 1.64 | 0.76 | 0.91 | **1.38** |
| loss on unlinked text | 3.59 | 2.35 | 2.43 | 2.75 | **3.16** |
| inside ÷ unlinked | 0.94 | 0.70 | 0.31 | 0.33 | **0.44** |
| after held-out / unseen / rare-seen / 3+-subtoken | 3.35 / 3.44 / 3.33 / 3.19 | 2.40 / 3.50 / 3.31 / 2.69 | 2.41 / 2.46 / 2.39 / 2.33 | 2.54 / 2.66 / 2.64 / 2.48 | **3.34 / 3.23 / 3.24 / 3.03** |
| held-out zero-shot property / entailment accuracy | 0.32 / 0.56 | 0.52 / 0.69 | — | — | **0.76 / 0.88** |

The *text* of notes is new to the host (unlinked 3.16 nats vs 2.35–2.75 on the other natural tracks), the *terms*
much less: inside-term loss sits between T4 and T1-open/WordNet, and P0 already answers 76% of held-out frame
properties. Expected place on R9's ordering: T5 > T4 ≳ **T1c** > T1-open ≈ WordNet — a small channel gain at most;
C0′ will lower every stratum (format adaptation), so C5 − C0′ is the readout, not C0′ − P0.

**Items** (rebuilt on the v2 holdout; `~/data/vsa-llm/t1c/items/`, the v1 sets moved to
`~/data/vsa-llm/t1c/stale-holdout-fa738430/`; digests in `experiments/t1c-clinical/items-digests.json`): 300 new words
(decision 35: `is_a` / `hierarchy` / `semantic_tag` kept, other fillers resampled; 817 property, 817 statement, 600
entailment items; contamination-checked on the training text), 200 edits (finding site → causative agent → associated
morphology, decision 40's pattern; 100 held-out, 100 seen; same-hierarchy fillers), track zero-shot items on 600 of
2,039 eligible held-out concepts (1,020 property groups × 2 paraphrases, 1,020 entailment pairs). Smokes on the P0 run:
track zero-shot 80–94 s / 2.3 GB at batch 8, editing 183 s / 2.2 GB, channel probes 386 s / 4.8 GB (360M probes need
≈ 5 GB on this GPU); 8-step C2 / C5 training smokes on SmolLM2-135M ran on the CPU.

**Queue** (from the main checkout; not queued by WP-T1c; `e9_plan --dry-run`, measured per-run times). Seed 1 at
51–54 after T7 and E11 tier 1 (orchestrator's placement); the whole-split rescoring (T1c-2) at 53, after the training
(51) and evaluations (52), before the report (54):

```bash
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
# seed 1: SmolLM2-360M + 135M × P0, C0', C2, C5; evaluations at 52, e4_quant (eval-mimic, eval-general) at 53, R9 report at 54
# — 59 jobs, ≈ 8.2 GPU-h (training and P0 ≈ 5.2)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t1c --hosts SmolLM2-360M SmolLM2-135M --seeds 1 --priority 51 --queue
# T1c-2: whole-split bf16 + INT8 (variant A) rescoring of the 360M C0'/C2/C5 seed-1 runs (held-out and rare strata) — ≈ 1.5 GPU-h
PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name t1c-quant-full-s1-SmolLM2-360M --priority 53 --min-free-gb 5 --no-resume -- \
  $PY -m vsa_embed.experiments.e4_quant --runs experiments/e9-retrofit/runs/t1c/SmolLM2-360M-full-C0p-s1 \
  experiments/e9-retrofit/runs/t1c/SmolLM2-360M-full-C2-s1 experiments/e9-retrofit/runs/t1c/SmolLM2-360M-full-C5-s1 \
  --output experiments/e9-retrofit/quant-full/t1c --bits 8 --variants A --baseline "C0'" --references C2 --windows 21827 \
  --resume --title "E9 T1c on the whole eval-mimic split (seed 1)"
# later, seeds 2–3 (after seed 1 is read) — ≈ 15 GPU-h; extend the rescoring to s2/s3 with the same command (--resume)
```

**Decisions taken (WP-T1c).**

| # | Decision | Taken |
|---|---|---|
| T1c-a | Hierarchies | clinical finding, procedure, body structure, substance, product, organism, observable entity (fillers from other hierarchies still appear in frames) |
| T1c-b | Aliases | FSN without its semantic tag + US/GB-accepted synonyms; bare acronyms dropped (decision 19); ≤ 8 words; UMLS not used for aliases (SNOMED's own synonyms are rich; more strings would grow a 6.6M-node trie for little linking) |
| T1c-c | Reference tokenizer | SmolLM2 (no GPT-2 corpus; a from-scratch variant would add one as a host relink) |
| T1c-d | Patient split, note order | 10% of patients evaluation-only; notes shuffled by `ROW_ID` hash so stream prefixes sample every category |
| T1c-e | Exact source shares | `build_corpus` records skipped input documents (`skipped_document_indices`; kept out of committed summaries) |
| T1c-1 | Held-out stratum below decision 17's entries bar at fraction 0.10 | **author, 2026-10-07:** re-freeze at fraction 0.30 before any run (`7369ec27…`; v1 stale) |
| T1c-2 | Whole-split evaluation for held-out / rare | **author, 2026-10-07:** `e4_quant --windows 21827` on the 360M C0′ / C2 / C5 runs (queued with seed 1 at 53) |

**Open decisions for the author.**

| # | Decision | Default if nothing better |
|---|---|---|
| T1c-3 | C2 at 295k entries is matched to 8 dimensions per entry (T1-open ≈ 76) | as the protocol says; C2f only if C5 shows a gain |
| T1c-4 | 1,560 acronym synonyms dropped; notes are dense with abbreviations | keep; proposal (d) tests a case-sensitive acronym linker |
| T1c-5 | Qwen3 on T1c | relink built; Qwen3 E9 runs only if SmolLM2 shows a gain |
| T1c-6 | Dimension-3 item counts | v1 sizes (300 / 200); grow to ≈ 700 (decision 56) only if T1c continues past seed 1 |

**Proposed health experiments** (not built; costs on the shared RTX 3090).

- **(a) Rare and unseen ICD code assignment from discharge summaries** (`runs/icd-coverage-v1`). MIMIC-III
  discharge-summary admissions 47,612 / 5,114 (T1c patient split), 11.7 ICD-9 codes each; 6,918 codes, 5,862 mapped to
  SNOMED CT by NLM's ICD-9-CM map, 5,589 to a T1c concept with a frame. Under the patient split only 143 codes are
  unseen in training (111 framed evaluation occurrences): too few. Design: a pre-registered **code-level holdout**
  (≈ 400 framed codes of the 10–49 band, never trained as labels; text kept) scored on all admissions (≈ 10k positive
  occurrences). Frozen E9 hosts (P0, C0′, C5 of the T1c seed-1 block), chunk-pooled states, label-wise attention head;
  code vectors: learned (seen codes), composed from the mapped concept's frame by C5's composer (zero-shot), and
  baselines (code title through the host = definition encoder, TransE as C6g, random, frame verbalized in context).
  Readout: macro-AUC and recall@k on held-out codes, micro-F1 on frequent codes. Cost ≈ 1 GPU-h per host model for one
  pass over ≈ 150M note tokens (P0, C0′, C5 × 360M ≈ 3 h; 135M ≈ 1.5 h), heads ≈ 0.5 h: **≈ 5 GPU-h**, CPU ≈ 15 min,
  ≈ 1 agent-day. MIMIC-IV (ICD-10, structured only): 19,440 codes, 8,971 framed via the SNOMED→ICD-10-CM map, 539
  unseen (154 framed); a text task on ICD-10 needs MIMIC-IV-Note (not on this machine). The group's `HRR_Atomics/v5`
  (ICD-9/10→SNOMED maps for 14.7k / 95.1k codes, relation maps, 21.5k concept atomic vectors, 50 relations) and
  `mimic-iv_data/{ood,rood}` are the structured-code predecessor; reuse the NLM map files, not the pickled vector spaces.
  - **Built as T1c-F** (author request 2026-10-07: "measure the frequency bias improvement like in the original HRR
    paper with MIMIC and how ontologies improve frequency bias").
    - Pre-registration: `experiments/t1c-clinical/icd-frequency/preregistration.md`, committed before any GPU run.
    - Code: `vsa_embed.icd_coding` and `vsa_embed.experiments.t1c_icd_frequency`. Tests:
      `tests/test_icd_frequency.py`, synthetic only.
    - Design:
      - HRRBERT's 7 ln-frequency bins.
      - A frozen, hashed holdout of 382 framed codes (`5450b37b…`), stratified over bins [−12,−10), [−10,−8) and
        [−8,−6). Held-out codes are scored on all 52,722 admissions.
      - A label-wise attention head on frozen hosts. Every label parameter, the bias included, comes from the
        condition's code vector: free, composed (head-trained or the T1c C5 dictionary), TransE, title, random, GRAM, or
        composed + free.
    - Primary (P0-360M):
      - E1: held-out macro-AUC, composed − free.
      - E2: the slope of per-code AUC on ln f, with a rare-code non-inferiority condition.
      - Holm over the two; two-way bootstrap over admissions and codes.
    - CPU stages are done; their aggregates are in `experiments/t1c-clinical/icd-frequency/runs/`.
    - Smokes (§14 of the pre-registration) passed: the GPU smoke took 2.7 min with a peak of 1.5 GB.
    - Queue commands are in `experiments/t1c-clinical/icd-frequency/queue-commands.sh`; GPU-h are idle-GPU estimates.
      - Pass 1 (P0) at 55: ≈ 4.1 GPU-h.
      - Pass 2 (C0′ / C5 encoders and the C5 dictionary) at 60–61, after T1c seed 1 and WP-UB's 55–59 block: ≈ 9.8 GPU-h.
    - Complements WP-UB: frequency bias inside the LM there, in a downstream clinical labelling task here.
- **(b) E10.2 on SNOMED CT (claim D).** Needs E10.9b's learnable-ontology code path (designed above, not built;
  ≈ 2 agent-days) and a null-calibrated acceptance test (R10 B.6). b1: erase 30% of the *non-inherited* finding-site
  and causative-agent edges of training-term heads (an edge implied by a parent's edge would be recovered by graph
  closure alone), offer them among as many same-hierarchy distractors; arms P (C5 on the reduced ontology), L
  (learnable, L2-to-prior), Ln (null world: offered tails permuted) × 3 seeds on SmolLM2-135M; acceptance threshold =
  the 95th percentile of Ln's scores (pre-registered); baselines `prior_corr`, TransE/RotatE, AMIE on the remaining
  graph. b2: hide causative agent (16.8k edges) with 3 blank slots; slot ↔ relation alignment against Ln. Cost: 18 ×
  0.65 h (135M) + 4 × 1.4 h (360M confirmation) + evaluations ≈ **20 GPU-h**; baselines ≈ 2 CPU-h.
- **(c) One-shot learning from definitions** (`runs/definitions-v2`; another agent designs read-to-learn).
  SNOMED text definitions: 6,343 of 310,698 T1c concepts (2.0%), 75 of 2,069 held-out. UMLS MRDEF (English sources,
  through the SNOMED CT US atoms' CUIs; NCI 24.2k, MeSH 18.6k, SNOMED 6.6k, Orphanet 6.5k, CSP 6.0k, HPO 4.9k): 43,583
  T1c concepts (14.0%), 498 held-out (24.1%). Local readers only; check each source's restriction level (MRSAB SRL)
  before use. Cost: minutes of CPU, < 1 GPU-h per model.
- **(d) Acronym-aware linker** (T1c-4): case-sensitive matching for the dropped all-caps synonyms (new linker version,
  relink ≈ 10 min CPU, ½ agent-day); C0′ / C5 seed 1 on 360M ≈ 2.5 GPU-h. Abbreviations are the part of note
  vocabulary most likely new to the host.

## E11 — read-to-learn (decision 59; WP-E11, 2026-10-07)

Pre-registration: `experiments/e11-read-to-learn/preregistration.md` (committed before any run). A trained E9 C5 model reads a
term's definition once; a reader writes a frame over the existing atomics and relations; the channel composes the term's row;
the term is tested with the definition out of context and no gradient step. Readers: oracle, stated, typeprior, pattern,
**linker** (the model chooses each found concept's relation by the definition's likelihood and self-tests the edge), host
(E7 prompted extraction), teacher (optional, open licences only), random, none. Text routes on the same items: the definition
in context (IKE-style) and a compute-matched one-shot gradient update (lr chosen on 40 dev words). Sets: T5 invented words
(3 definition styles; primary `prose`), T5 held-out terms, T4 held-out ChEBI entities with their ChEBI definitions (natural),
T4 invented compounds, T1 held-out MeSH (negative control), OpenStax *Chemistry 2e* glossary (exploratory frame swap; the
book is CC BY-NC-SA 4.0, not CC BY). Primary endpoints: P1 T5 new words, 360M, property, linker − none; P2 T4 held-out ChEBI
terms, loss after the term, linker − none (Holm over both). Code `vsa_embed.read_to_learn`, `vsa_embed.experiments.e11_read_to_learn`;
queue commands from `e11_read_to_learn plan` (priority 61–63, evaluation only).

**E11 follow-ups of 2026-10-07** (preregistration §§13.6, 14, 15; committed before any run):
- **E11-M, many terms read once each.**
  - Design:
    - Sets: N = 25 … 400 invented T5 words (v2 pool, nested episodes, 250 generated multi-term passages) and 242 T4
      held-out ChEBI terms (natural windows with ≥ 2 read terms).
    - Routes: frames written at once; all definitions in context under 2,048 / 7,168 tokens (cached prefix); BM25
      retrieval (k = 1, 3); a sequential one-pass gradient update.
    - Primary M1: T5, 360M C5, N = 200, property, `frame:linker − context:B2048`.
  - Code `vsa_embed.experiments.e11_many`. Queue: `experiments/e11-read-to-learn/queue-commands-many.sh`, priority 53,
    ≈ 5.6 GPU-h now and ≈ 1.9 GPU-h after T4 seeds 2–3.
- **Dictionary sources.**
  - Wiktionary neologisms are open (CC BY-SA), but have no gold frames and almost no local occurrences; Wikidata lexemes
    give CC0 glosses only; Merriam-Webster, Wordnik and Oxford are not open.
  - The feasible natural source is the MeSH 2026 supplementary-record notes of T7's held-out records:
    `items/t7-heldout-smollm2-v1` (487 terms), which runs once T7 is trained.

## E9 new-vocabulary track T7 (decision 55; screened and built 2026-10-07)

**Novelty proxy** (`e9_novelty calibrate`, `experiments/e9-retrofit/novelty/calibration/`).

- **What was measured.** P0-only quantities on the four finished tracks, seed 1, against R9's novelty index
  (C0′ − P0, mean of the held-out, unseen, rare and 3+-subtoken strata) and against the channel gain.
- **Agreement with the novelty ordering.** Three P0-side proxies order T5 > T4 > T1 > WordNet exactly at both
  SmolLM2 sizes (Kendall τ = 1). An exact ordering of 4 tracks has p = 1/24.
  - Inside nats per linked span: 15.3 / 5.3 / 1.3 / 1.1 at 360M.
  - Subtokens per span: 5.5 / 5.1 / 2.8 / 2.3.
  - General-to-domain frequency of the vocabulary: 0.00 / 0.04 / 0.38 / 0.91.
- **The other two proxies.** P0 inside loss swaps the two null tracks (τ = 0.67). After-term / unlinked loss does not
  order the tracks at all.
- **Agreement with the gain.** The gain orders three levels: T5, then T4, then T1 = WordNet (n.s.). Every ordering
  proxy reaches the best attainable τ-b of 0.91 (p = 0.083).
- **Primary proxy:** inside nats per span. It is the host's own surprisal for spelling a term, and it does not depend on
  span density. General-to-domain frequency is biased toward 0 for mention-filtered corpora.

**Candidates** (SmolLM2-360M P0 slices, 512 windows; `experiments/e9-retrofit/novelty/candidates.md`).
Inside nats per span:

| Candidate | Nats / span | Verdict |
|---|---:|---|
| T2 | 25.8 | templated synthetic, excluded |
| T3 | 1.82 | low novelty, infeasible rare stratum |
| T6 | 1.63 | low novelty, infeasible rare stratum |
| MeSH descriptors introduced 2024–26 | 2.16 | new headings for established words |
| MeSH SCRs introduced 2022–26 | 5.92 | only 541 records |
| MeSH names absent from general text, SCR + descriptors | 5.43 | |
| Same, SCR only | 6.59 | chosen |

Three ideas were not built:
- New INNs: the WHO lists are CC BY-NC-SA, and the INNs arrive through the SCRs anyway.
- ChEBI entries by date: ChEBI's OBO has no creation dates.
- Wikidata / Wikipedia 2025 entities: dumps of 25–140 GB, and most names are composed of known words.

**T7** (`experiments/t7-new-vocabulary/`, `ontologies/mesh_novel.py`, built by `t1_open_corpus` with the `mesh_novel`
adapter):
- **Vocabulary.** MeSH 2026 SCRs (chemicals, diseases, organisms) with ≥ 5 training-side mentions and 0 mentions in
  300,000 documents of the C3 general stream, counted over all of a record's names.
- **Selection.** 8,184 records, 11,141 names, frozen as `screen/selection.tsv` (sha256 `1ead39e8…`).
- **Frames.** Mapped heading, pharmacological action, record class and branches, over 3,295 descriptor atomics.
- **Corpus.** PubMed abstracts that mention a selected name: the 2026 baseline slice plus the 80 newest 2026 update
  files, PMID ≥ 41,025,505, 94% published in 2026. They are mixed 50/50 with FineWeb-Edu.
- **Sizes.** `train` 57.3M tokens; `eval-pubmed` 6.5M tokens, PMID buckets < 2,000.
- **Holdout.** 908 entries, node- and alias-disjoint, sha256 `5e133b5f…`.
- **Feasibility at ℓ_min 2:** feasible with 4,096 evaluation windows.
  - Held-out: 8,786 occurrences, 411 entries with ≥ 5.
  - Rare: 4,337 occurrences, 1,577 entries.
- **Measured P0 novelty on `eval-pubmed`:** 6.49 nats per span, 1.21× T4.
- **Dimension 3.** `items/{new-words,edits}-t7-smollm2-v1`: 300 new words and 200 edits. The edited relation is the
  pharmacological action, else the mapped heading.
- **Alias table.** `~/data/vsa-llm/e9/alias-tables/t7.json`.

**Queue** (not queued). Configs are in `experiments/e9-retrofit/configs/t7/`. `--dry-run` lists every job. The
estimates use measured costs, plus T7's 3,072 extra evaluation windows per evaluation point. T7 is proposed at priority
70–73, after T1c's 62–69:

```bash
# SmolLM2-360M P0, C0′, C2, C5 × seeds 1–3: 53 jobs ≈ 15.4 GPU-h (≈ 1.26 h per trained run with its 4,096-window evaluations)
PYTHONPATH=src python -m vsa_embed.experiments.e9_plan --track t7 --hosts SmolLM2-360M --seeds 1 2 3 --priority 70 --queue
# optional, the size trend: SmolLM2-135M seed 1 (P0, C0′, C2, C5): 23 jobs ≈ 2.9 GPU-h
PYTHONPATH=src python -m vsa_embed.experiments.e9_plan --track t7 --hosts SmolLM2-135M --seeds 1 --priority 70 --queue
```

**Prediction from the proxy (written before any run).** T7 is slightly more novel than T4, by 1.21× on the proxy. Its
gain should therefore be at least T4's on the term strata: after unseen terms −2.3%, after 3+-subtoken terms −1.1%
(360M, seed 1). It should not reach T5's. A null T7 would break the novelty ordering. A T4-sized gain would replicate it
on a second natural track whose text holds no curated definitions.

## E9 frequency bias and understanding beyond copying (author request 2026-10-07; WP-UB)

Pre-registration: `experiments/e9-retrofit/preregistration-understanding.md`, committed before any evaluation of these
measures on a checkpoint (`9c6f6e2`), with dated amendments §10.1–10.4 (`6c784b8` and later).

**Why.** R9's copy-concern table shows that 86–90% of the T5 gain after rare and unseen terms falls on filler tokens.
The author's two points:
- Copying fillers partly measures **frequency bias**, so measure that improvement properly, in HRRBERT's tradition.
- Build items whose correct answer is **not** a filler, so copying cannot solve them.

| Family | What | Primary endpoint (T5, SmolLM2-360M, seeds 1–3) |
|---|---|---|
| A1 | loss per half-decade log-frequency bin (1–3, 4–9, 10–31, …, ≥ 3,163; the unions are exactly the trainer's 1–9 / 10–99 / ≥ 100 strata); slope and rare–frequent gap per model; window and term-cluster bootstraps | Δgap = gap(C5) − gap(C0′) |
| A2 | cross-validated ridge R² of log2 training frequency from concept representations (composed / free / source row, injected row, host subtoken mean, model-free references), entry bootstrap, t-SNE as in HRRBERT | R²(C2 free row) − R²(C5 composed vector) |
| B1 | understanding items whose answer is not a filler: two-hop, type affordance, paraphrased options, reverse (filler → term, candidates in context), same/different comparison, negation (pair consistency) | composite, held-out terms, C5 − C0′ (concepts × seeds crossed model) |
| B2 | strict non-copy stratum: no alias or content word within two hops of the frame, no relation wording | `after_heldout_strict`, C5 − C0′ |

**Code** (tests: `tests/test_e9_understanding.py`, toy glossary on the CPU):
- `e9_freqbias` (`exclusions`, `score`, `rows`, `queue`);
- `e9_understanding` (`items`, `evaluate`, `queue`);
- `e9_report --freqbias --understanding ITEMS [--item-seed]`.

**Inputs.**
- Items, committed: `experiments/e9-retrofit/items/understanding-{t5-smollm2,t5-qwen3,t4-smollm2,t7-smollm2}-v1`.
- Exclusion tables: `~/data/vsa-llm/e9/exclusion-tables/`; T1c's is under `~/data/vsa-llm/t1c/` and is built by its own job.

**Outputs.**
- A1/B2: `RUN/freqbias`. A2: `RUN/freqrows`. B1: `RUN/understanding-<track>-<family>-v1`. Predictions are gzipped, about
  4 MB per C5 run.
- T1c run folders keep aggregates only (`.gitignore`). T1c gets no understanding items and no t-SNE.

**Cost** (GPU-h at the queue's dedicated throughput, from the committed dimension-3 timings with the faster scorer; ±50%):

| Block | Priority | Runs | GPU-h |
|---|---|---|---:|
| U1 T5 SmolLM2-360M (P0, C0′, C2, C5, 7 arms × s1–3) + 135M (P0, C0′, C2, C5 × s1–3) | 55 (rows 56, CPU) | 41 | ≈ 3.2 (rescoring 0.65, items 2.6) |
| U2 T5 Qwen3-0.6B/1.7B (P0, C0′, C2, C5 × s1–2) | 56 (rows 57) | 14 | ≈ 3.4 (rescoring 0.8, items 2.6) |
| U3 T4 / T1 / WordNet seed 1 (SmolLM2-360M + 135M) | 57 (rows 58) | 24 | ≈ 0.65 |
| U4 once trained: T4 s2–3 and arms, T7, T1c | 58 (rows 59) | 27 + 14 + 20 | ≈ 2.3 |
| Reports (CPU lane) | 59 | — | 0 |

**Total ≈ 9.6 GPU-h**, plus about 20 CPU-h of row probes in the CPU lane (4–25 min per run).

The priorities follow the T4/T7/E11/T1c block at 51–54. Within the same priority, jobs queued earlier run first. The run
folders need not exist when the jobs are queued: U4 runs after its training at 51–54.

```bash
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
# (from the main checkout, after merging this branch)
# U1 — T5 SmolLM2 (primary): A1/B2 rescoring (GPU, 55) + A2 row probes (CPU lane, 56) + B1 items (GPU, 55; C5 own,none,random_frame)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage t5 --priority 55
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_understanding queue --stage t5 --priority 55 \
  --items experiments/e9-retrofit/items/understanding-t5-smollm2-v1
# U2 — T5 Qwen3 seeds 1–2 (seed 3 later with --seeds 3; C5 sources own,none, §10.4)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage t5-qwen3 --seeds 1 2 --priority 56
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_understanding queue --stage t5-qwen3 --seeds 1 2 --priority 56 \
  --items experiments/e9-retrofit/items/understanding-t5-qwen3-v1 --candidate-sources own,none
# U3 — finished natural tracks and the negative control, seed 1
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage t4 --seeds 1 --models P0 C0p C2 C5 --priority 57
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_understanding queue --stage t4 --seeds 1 --models P0 C0p C2 C5 --priority 57 \
  --items experiments/e9-retrofit/items/understanding-t4-smollm2-v1
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage t1 --priority 57
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage wordnet --priority 57
# U4 — once trained (queued at 51–54): every T4 config (the seed-1 jobs above are skipped by name), T7, T1c (aggregates only, no figure)
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage t4 --priority 58
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_understanding queue --stage t4 --priority 58 \
  --items experiments/e9-retrofit/items/understanding-t4-smollm2-v1
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage t7 --priority 58
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_understanding queue --stage t7 --priority 58 \
  --items experiments/e9-retrofit/items/understanding-t7-smollm2-v1
PYTHONPATH=src $PY -m vsa_embed.experiments.e9_freqbias queue --stage t1c --priority 58 --licensed
# Reports (CPU lane, 59): one per stage; T7 and T1c once U4 is done (T1c: --freqbias only)
for spec in "t5 understanding-t5-smollm2-v1" "t5-qwen3 understanding-t5-qwen3-v1" "t4 understanding-t4-smollm2-v1"; do
  set -- $spec
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name $1-report-understanding --priority 59 --lane cpu --min-free-gb 0 --no-resume -- \
    $PY -m vsa_embed.experiments.e9_report --runs experiments/e9-retrofit/runs/$1 --output experiments/e9-retrofit/report/$1-understanding \
    --freqbias --understanding $2 --item-seed --overwrite --title "E9 $1 — frequency bias and understanding beyond copying"
done
for stage in t1 wordnet; do
  PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name $stage-report-freqbias --priority 59 --lane cpu --min-free-gb 0 --no-resume -- \
    $PY -m vsa_embed.experiments.e9_report --runs experiments/e9-retrofit/runs/$stage --output experiments/e9-retrofit/report/$stage-freqbias \
    --freqbias --overwrite --title "E9 $stage — frequency bias"
done
```

Each `queue` command takes `--dry-run`, which prints the jobs without queueing them. On 2026-10-07 the dry run gave:
- U1: 41 GPU rescoring jobs, 41 CPU row-probe jobs and 41 GPU item jobs;
- U2: 14 + 14 + 14;
- U3: 8 + 8 per track, plus 8 T4 item jobs;
- U4: T4 35, T7 14, T1c 20 (each job name is unique, so already-queued jobs are skipped).

**Smoke** (labelled; pre-registration §9 addendum). On real checkpoints:
- the masks replay the trainer's strata exactly;
- a GPU composing run replays to 9 × 10⁻⁵ on the large strata, and a non-composing run bit-exactly;
- the row probe takes 3–4 min per 135M run on the CPU;
- the items link 695 of 695 concepts.

**Open decisions for the author.**

| # | Decision | Default |
|---|---|---|
| UB-1 | A1 primary on the absolute gap (nats), with the relative gap as the key secondary (reading (a) needs both) | as pre-registered |
| UB-2 | B1 held-out subset = the 360 held-out + 200 generator zero-shot terms (both composed zero-shot) | as pre-registered |
| UB-3 | Qwen3 C5 without the `random_frame` source (cost) | as §10.4; add it for seed 1 if B1's SmolLM2 reading hinges on it |
| UB-4 | T7 items use the canonical (lower-case) alias; T7 is not trained yet | rebuild as `-v2` with MeSH display names before T7's evaluation, if the author wants |
| UB-5 | Priorities 55–59 interleave with Qwen3 seed 3 / Qwen3-4B (55–59), behind them at equal priority | as proposed; use 54 for U1 to read A1/B1 before those |

## Author decisions of 2026-10-08 (after the operator screen)

| # | Decision | Detail |
|---|---|---|
| 60 | **Binding and unbinding program** | For next-token loss, composition matters but multiplicative binding does not: on T5 (SmolLM2-360M, 3 seeds) untyped bundling ties learned HRR (C5ut − C5 = +0.2%, n.s.), translation beats it (+4.8%*), and a fixed random operator loses 5.2%* on held-out terms (learning the operator helps in the pretrained regime); in decision 54's 50M screen every operator ties, fixed random included. The likely reason is low role ambiguity: a term's filler set almost identifies it (terms whose filler multiset collides once roles are dropped: T5 0%, T4 0.5%, T1c 0.1%, WordNet 2%, T7 0%; same filler under two roles within one frame: T4 12%, T1c 3.7%, T5 0%). Binding should matter where a role has to be read out. Three steps, each pre-registered before its runs: **(1)** a decodability probe on trained checkpoints (evaluation only): algebraic unbinding plus cleanup against the atomic dictionary for every composer arm, against learned linear decoders (LRE-style) for every model, by seen / held-out, frame size and role ambiguity, plus behavioural role-swap twins (new words with the same fillers in swapped roles); **(2)** an unbinding readout arm: a head predicts a role from the hidden state, unbinds it from a preceding linked concept's composed vector, cleans up and injects the filler (gated), with operator variants (learned HRR, learned unitary, translation, untyped as controls), T5 SmolLM2-360M × 3 seeds, a 50M from-scratch screen optional; **(3)** chained two-hop and reverse lookup by unbinding, and a capacity test of local (per-concept) against global holographic superposition, on T5, T4 and T1c (aggregates only), Qwen3 optional. Related work checked 2026-10-08: Hernandez et al. ICLR 2024 (relation decoding is often a linear map: implicit unbinding); Dhayalkar, AAAI-26 LSRLM workshop (attention as soft unbinding; binding/unbinding heads proposed, not tested); Dhanraj & Eliasmith EMNLP 2025 (HRR over a frozen host's hidden states for rule-based reasoning); Kumar, arXiv 2606.24948 (HRR + Hopfield KG memory: one hop works, zero-shot two-hop fails from global superposition capacity); Wang & Sun ICLR 2026 (the reversal curse as a binding problem; JEPA fix). Placement "decisive first": step 1 as soon as it is built, steps 2–3 before the Qwen blocks |
