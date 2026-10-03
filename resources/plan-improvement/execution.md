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
| 21 | OpenReview `tfyLS1cB5W` ("Encoding Ontologies with HRR for Transformers") may be the group's own earlier submission | WP-MS | ask the author | do not cite twice; confirm with the author |
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

Report: `reports/R10-self-learned-semantics.md`. E10.0/E10.1 start now (CPU / backfill GPU); E10.2/E10.3 after G3 and E9/E7.

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
