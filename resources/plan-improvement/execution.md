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
