# Manuscript outline (working)

**Status:** first-pass scaffold, 2026-10-02. Results E1, E3 and E4–E8 are pending. Every sentence that depends on them is a slot `[[Rn: …]]`, where `Rn` is the report that will fill it ([execution.md](../resources/plan-improvement/execution.md) "Reports"). Claims use the allowed wording of [proposal.md](../resources/plan-improvement/proposal.md) §7. The further narrowing proposed in [related-work-recheck-2026-10.md](related-work-recheck-2026-10.md) §6 is marked **(re-check)** wherever it would apply. The claims ledger is [claims.md](claims.md); the text is [draft.md](draft.md).

## Working title

**CG-VSA: compositional ontology rows for multi-token concepts in small causal language models**

Alternative titles, chosen by the outcome (§ "Framing decision" below):

| If … | Title |
|---|---|
| E4 gate passes and C8 separates `hrr` from `random_fixed` | CG-VSA: role–filler binding of ontology edges reduces the syntax → semantics tax in small causal LMs |
| E4 gate passes but `random_fixed ≈ hrr` (C8) | CG-VSA: compositional parameter sharing over ontology edges for multi-token concepts in small causal LMs |
| E4 item 2 fails (C1 ≈ channel) or C1h ≈ channel | Is it the ontology or the lookup? A controlled test of structured span channels in small causal LMs |
| Only E0–E2 and a negative E4 | What ontology composition does and does not buy small language models |

The placeholder name **CG-VSA** (contextual, growable VSA) is kept until the operator question is answered. "VSA" stays in the title only if C8 separates a binding operator from `random_fixed`. This is tasks.md rule 7 and proposal §10.

## Abstract skeleton (one paragraph; slots marked)

> Byte-pair tokenizers optimise compression, not meaning. A language model must therefore learn, from data and parameters alone, how the fragments of a multi-token word compose into a concept. We call this cost the *syntax → semantics tax*. We study whether a curated ontology can pay part of it. CG-VSA adds a span channel to a causal LM. Each concept's row is composed from its ontology relation–filler edges by vector-symbolic binding, with no per-concept parameters (re-check: "by fixed binding"). The row can be re-weighted by a causal context query (it reduces to the static HRR bundle as τ → ∞). The shared dictionary can grow through a first-order, usage-level split test read from optimizer state. The row is added at the last subtoken of a prefix-causally linked span. On planted structure the mechanisms are identifiable. The context query recovers context-dependent compositions that the static bundle cannot (+0.36 to +0.46 held-out cosine). Ontology-induced edge weights transfer to composition-disjoint concepts (0.976 vs 0.60–0.70 for free factors). The split test recovers every planted polysemous atomic and relation sub-type with ≤ 1% surviving false splits. On frozen GPT-2 anchors for WordNet and MeSH, relation labels carry signal, but no binding operator separates from a random fixed or untyped one. `[[R2: E1 contextual-variance and sense-alignment result]]` `[[R2: E3 sense/sub-type recovery vs uniform enlargement at matched parameters]]` In 50M- and 125M-parameter LMs trained from scratch on 1.1B-token FineWeb-Edu × WordNet streams, `[[R3: held-out-concept and seen-rare stratified loss vs the free per-concept table (C2), random span vectors (C1), hashed n-gram memory (C1h), shuffled frames (C1s); locality; probes; data multiplier k(L) with CI]]`. `[[R3: operator ablation C8 — binding vs random_fixed/untyped]]` `[[R4: continued pretraining of SmolLM2-135M/360M, Qwen2.5-0.5B; PTQ retention]]` `[[R5: faithfulness and LLM-graded explanation estimate; structure-only zero-shot rows]]` `[[R6: self-authored ontology frames vs compute-matched CPT, EntiGraph- and SPA-style controls]]` `[[R7: which of six application tracks are feasible and where the channel pays off]]` All runs fit a ≈ 400-GPU-hour budget on one RTX 3090; convergence-speed evidence decides `[[R8: escalation verdict]]`.

## Contributions, tied to hypotheses

| # | Contribution (allowed wording) | Hypothesis / question | Feeds from | Status now |
|---|---|---|---|---|
| 1 | A compositional, growable span channel for causal LMs: rows built from ontology relation–filler edges with no per-concept parameters, a prefix-causal last-subtoken rule, and a gate (re-check: "by fixed binding … added in place … during LM training"; ConceptFormer is the nearest prior work) | H-A (mechanism) | Method; B5 tests; E4 | method implemented and unit-tested; effect pending |
| 2 | A measurement of the syntax → semantics tax on ontology-linked multi-token strata. It tests whether structure beats random span vectors, a free per-concept table, shuffled frames and a hashed n-gram memory at matched parameters (re-check: "… injection site and gate", or add C1h-E) | **H-A** | E4 (D4.7, D4.9, D4.1), E4.6 | pending (`[[R3]]`) |
| 3 | A query-conditioned weighting of VSA-bound ontology edges with zero per-concept parameters that recovers the static bundle as τ → ∞ (re-check: contrast CokeBERT) | **H-C** | E0.1 (done), E1, E5.2, E4-C5 | synthetic: supported; real: pending |
| 4 | An ontology-induced, transferable rank-k edge weighting (M2) | H-A, H-E | E0.3 (done), E2 (done), E4-C4 | synthetic: supported; frozen anchors: non-inferior to salience, not superior to binary |
| 5 | A first-order, usage-level split test for shared dictionary vectors, read from optimizer state and calibrated by a permutation null (re-check: concede Parameter Differentiation, Wang & Zhang 2022; add a threshold baseline) | **H-D** | E0.2 (done), E3, E4-C6 | synthetic: supported; WordNet: pending |
| 6 | Quantization retention and equal-bytes compression against factorized, hashed, TT and PTQ tables | **H-B** | E4.4, E4.5 | pending |
| 7 | Structure-only zero-shot rows for unseen ontology nodes | **H-E** | E4.3, E5.4 | pending |
| 8 | Faithful edge-level explanations and an LLM-graded estimate of explanation quality (labelled as an estimate; the clinician study comes later) | **H-F** | E5.1, E5.3; C5 calibration (done) | judge calibrated; study pending |
| 9 | Self-authored ontology frames verified by held-out gradient utility and consumed through the channel (re-check: extend the "not novel" list; add unstructured-notes, SPA-style and verbalized-frames controls) | **H-G** | E7 | pending |
| 10 | The operator question: does the binding operator matter in the joint regime? Either answer is reported | cross-cutting | E2 (done), E1, E3, E4-C8, every report | frozen anchors: no separation from `random_fixed`/`untyped` |
| 11 | Negative and methodological results: frozen-host HRR row fitting is refuted after an audit (01a–01c); an evaluation audit with corrected statistics; bit-reproducibility across machines; feasibility (span cardinality) of six application tracks | — | proposal §2, audit, R0, C3/T1/C7 | done |

## Framing decision (taken when R3 lands; rules from proposal §8/§10 and experiments.md E4)

| E4 outcome | Framing of the paper |
|---|---|
| Items 1–3 pass, C8 `hrr` > `random_fixed` with CI | ontology structure **and** binding matter: full CG-VSA framing |
| Items 1–3 pass, `random_fixed ≈ hrr` | "**compositional parameter sharing** over ontology edges", not symbolic semantics (rule 7) |
| Item 2 fails (channel ≈ C1) | a **boundary-signal** result: span boundaries help, ontology content does not |
| Channel ≈ C1h (or C1h-E) | a **lookup** result: any extra span memory helps as much |
| Channel ≈ C1s (shuffled frames) | atom sharing, not ontology truth, explains the gain |
| Item 1 fails, item 2 passes | seen-concept efficiency result without a zero-shot claim |
| Locality fails | report the damage; no deployment claim |
| `k` lower CI ≤ 1.1 at the committed budget | "no measurable data-efficiency gain at ≤ 500M tokens"; projection reported; no escalation |

## Section structure

### (a) Full paper

| § | Section | Content | Feeds from |
|---|---|---|---|
| 1 | Introduction | the tax; why ontologies; why joint training (frozen-host refutation, HRRBERT); contributions; framing per the decision table | proposal §0–§1, 01a–01c, `[[R3]]` |
| 2 | Related work | knowledge-injected LMs and compositional entity rows; attention as binding; factorized tables; splitting and growth; multi-token words and lookup memories; zero-shot rows; self-authored knowledge; VSA in neural models | related-work.md + re-check |
| 3 | Method | M0 → M1 → M2 → M3; M4 span channel and causal rule; M5 loop; objective; parameter and byte accounting; sparse generation | formulation.md; B10 |
| 4 | Experimental setup | hardware; corpora and holdouts; tracks; conditions C0–C9 and controls; statistics; convergence and escalation; LLM judge | C3/T1/C7 reports, B6, C5, experiments.md §0 |
| 5 | Mechanism studies | 5.1 E0 synthetic identifiability; 5.2 E1 contextual composition on frozen anchors; 5.3 E2 mapping × operator frontier; 5.4 E3 developmental recovery on WordNet | E0 (done), E2 (done), `[[R2: E1, E3]]` |
| 6 | Core result: small LMs from scratch | ℓ_min feasibility; 50M screen; 125M convergence; stratified losses; controls; C8; probes; `k(L)`; projections; escalation verdicts | `[[R3]]` (D4.7, D4.9, D4.1) |
| 7 | Pretrained hosts, quantization and compression | CPT on three hosts; size trend of `k`; PTQ; equal-bytes compression | `[[R4]]` |
| 8 | Explanations and zero-shot | faithfulness; sense alignment; frequency disentanglement; LLM-graded estimates; zero-shot insertion | `[[R5]]` |
| 9 | Self-authored ontologies | authoring quality; round 0 → 1; controls; cross-authoring | `[[R6]]` |
| 10 | Application tracks | cardinality and feasibility (done); per-track results; cross-track table | C3/T1/C7 (done), `[[R7]]` |
| 11 | The operator question | accumulated operator ablation across E1–E8 | E2 (done), `[[R2–R7]]` |
| 12 | Limitations | open-clinical substitute; LLM-graded estimates; single-GPU budget; frozen-host refutation; flat frames; linker errors | — |
| 13 | Reproducibility | hardware, env, data hashes, holdouts, commits, compute ledger | [reproducibility.md](reproducibility.md), R0 |
| App. | A: frozen-host 01a–01c and the audit; B: E0 details; C: cardinality tables; D: sparse generation benchmark; E: judge protocol and calibration; F: E6 (if run); G: pre-registration and deviations | | |

### (b) Short "core result" paper (if only E4 lands)

**Working title:** "Does ontology structure pay the syntax → semantics tax? A controlled test in small causal LMs" (positive or negative). Target length: 8 pages plus appendix.

| § | Section | Content | Feeds from |
|---|---|---|---|
| 1 | Introduction | the tax; the question "structure or lookup?"; the answer in one sentence `[[R3]]` | `[[R3]]` |
| 2 | Related work (compressed) | entity channels (KALM, KnowBERT, Bootleg, ConceptFormer); lookup memories (Engram family); detokenization (Feucht, Kaplan); HRRBERT | related-work.md + re-check §4 |
| 3 | Method | M0/M3-static + M1 (C5) + M4 only; M2 as a sentence; M3 only if C6 is the best condition | formulation §0–§1, §4 |
| 4 | Setup | C3 corpus and holdout; strata; conditions C0, C1, C1s, C1h (+ C1h-E), C2/C2f, C3, C3t, C4, C5 (C6); C8 operators; statistics; `k(L)` | C3 report; G2 pre-registration |
| 5 | Results | 5.1 sanity: E0 in one figure (identifiability), E2 in one sentence (operators tie on frozen anchors); 5.2 feasibility (ℓ_min); 5.3 stratified losses with controls (main table); 5.4 operator ablation; 5.5 convergence speed and projections; 5.6 probes | E0, E2 (done); `[[R3]]` |
| 6 | Discussion | framing per the decision table; what a negative result rules out; escalation verdict | `[[R3]]`, `[[R8]]` |
| 7 | Limitations, reproducibility | | as (a) §12–13 |

The short paper is complete with E0, E2, C3 and E4 alone. E1/E3 enter only as one sentence each if they are done; E5–E8 are referenced as future work.

## Figures and tables (generated by `figures/make_figures.py` from committed run folders)

| Item | File | Status |
|---|---|---|
| Fig. E0: recovery curves (D0.1), rank transfer (D0.3), split-detection gate bars (D0.2) | `figures/e0_identifiability.{png,pdf}` | done |
| Fig. E2: mapping × operator heatmap of test MRR per family (± 95% CI) | `figures/e2_mapping_operator_heatmap.{png,pdf}` | done |
| Fig. E2b: paired contrasts (induced vs salience/binary; selected vs random_fixed/untyped; true vs shuffled labels) | `figures/e2_contrasts.{png,pdf}` | done |
| Fig. cardinality: covered fraction and distinct entries by ℓ_min × tokenizer × corpus | `figures/cardinality_by_lmin.{png,pdf}` | done |
| Fig. generation: sparse vs full channel-row cost | `figures/generation_sparse_vs_full.{png,pdf}` | done |
| Fig. E4: paired stratified loss differences (forest) per cohort | `figures/e4_<analysis>_<cohort>.{png,pdf}` | stub; appears when `experiments/e4-small-lm/analysis/*/summary.json` exists |
| Fig. E4: loss curves per stratum, `k` vs tokens | written by `e4_report` itself | pending |
| Tables: corpora and holdouts; cardinality; E0; E2; generation; E4 | `tables/*.md` | done except E4 (pending stub) |
| Pending tables: E1, E3, E4.4/E4.5, E5, E7, E8 cross-track | to add to `make_figures.py` when their run folders exist | pending |
