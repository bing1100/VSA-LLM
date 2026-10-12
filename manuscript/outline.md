# Manuscript outline (working)

**Status:** 2026-10-03, updated after the novelty check ([novelty-check-2026-10.md](novelty-check-2026-10.md)); first-pass scaffold 2026-10-02. Results E3 and E4–E8 are pending; the E9 dimension-3 baselines are implemented but not yet run, and the E10 baselines are in R10 (they turn E10 into a negative / methods result by the framing rule below). Every sentence that depends on them is a slot `[[Rn: …]]`, where `Rn` is the report that will fill it ([execution.md](../resources/plan-improvement/execution.md) "Reports"; R9 = E9, R10 = E10). Claims use the allowed wording of [proposal.md](../resources/plan-improvement/proposal.md) §7. The further narrowing proposed in [related-work-recheck-2026-10.md](related-work-recheck-2026-10.md) §6 is marked **(re-check)** wherever it would apply. The E9/E10 claims A–D use the novelty check's recommended wording (claims ledger section D). The claims ledger is [claims.md](claims.md); the text is [draft.md](draft.md).

**Update 2026-10-12.** This session (workplace-ba) owns the title, abstract and outline (author, 2026-10-12). E9 now has:
- T5, T4 and T7 at three seeds (360M);
- the pre-registered frame swap;
- the decision-64 controls.

The abstract (here and in [draft.md](draft.md)) is revised accordingly, and the outcomes of the E9 framing rules are marked below. The E9 dimension-3 baselines are queued (priority 60). The addendum of 2026-10-11 (`experiments/e9-retrofit/preregistration-addendum-2026-10-11.md`) adds in-context frames, C2w, C6gi on T7-ROOD, C5ld, T4 135M seeds 2–3, and the T7 claim-B controls (done).

**Terminology (novelty check §6.4).**
- **Ontology-composed span embedding**: the gated vector, bound by HRR from a concept's ontology frame over a shared, jointly trained dictionary, that is added to the input embedding at a linked span's last subtoken. "Span channel" names the module (code `span_channel`); never bare "semantic channel", "VSA channel", "concept channel" or "latent concept injection".
- **Ontology frame**: the set of a concept's relation–filler edges (defined at first use).
- **Joint fine-tuning into a pretrained host** instead of an undefined "retrofit".
- Abstract-level text says **offline held-out-tested revision** and **property-hypothesis testing**, not "dreaming" or "riddle-style"; the cognitive terms appear only in the Discussion's labelled analogy.

## Working title

**CG-VSA: ontology-composed span embeddings for multi-token concepts in small causal language models**

Alternative titles, chosen by the outcome (§ "Framing decision" below):

| If … | Title |
|---|---|
| E4 gate passes and C8 separates `hrr` from `random_fixed` | CG-VSA: role–filler binding of ontology edges reduces the syntax → semantics tax in small causal LMs |
| E4 gate passes but `random_fixed ≈ hrr` (C8) | CG-VSA: compositional parameter sharing over ontology edges for multi-token concepts in small causal LMs |
| E4 item 2 fails (C1 ≈ channel) or C1h ≈ channel | Is it the ontology or the lookup? A controlled test of ontology-composed span embeddings in small causal LMs |
| Only E0–E2 and a negative E4 | What ontology composition does and does not buy small language models |
| E4 pending, E9 controls pass (A-B1–A-B4, C-B1/C-B2) | CG-VSA: ontology-composed span embeddings for new terms in small pretrained causal LMs |

*2026-10-12:* the last row's condition (E9 controls A-B1–A-B4 pass) is not met on natural text. There a TransE row ties the composed embedding, the definition encoder is ahead after held-out terms, and the gain is frame-specific only on T5. So that title is off the table unless T7-ROOD and the addendum change it.

The placeholder name **CG-VSA** (contextual, growable VSA) is kept until the operator question is answered. "VSA" stays in the title only if C8 separates a binding operator from `random_fixed`. This is tasks.md rule 7 and proposal §10.

## Abstract (novelty check §8; slots marked)

Revised 2026-10-12, kept identical to [draft.md](draft.md). Claim B is not in it (a track- and quantizer-dependent measurement; B-B1 deferred by the author 2026-10-11). Three rules: results that need pending controls are slots; "semantic channel", "first", an undefined "retrofit", "knowledge editing via the ontology rather than weights" and "makes quantized models robust" are absent; "HRR" names the default operator, and the text says "compositional parameter sharing" unless the operator ablation separates `hrr` from `random_fixed`. E10 is omitted until E10.2.

> HRRBERT (Hu et al., 2024) composed SNOMED CT code embeddings from atomic and relation vectors with holographic reduced representations (HRR) and trained them jointly with an encoder over medical-code sequences, improving rare codes. We extend this to causal text language models. A deterministic, prefix-causal linker maps multi-token terms to ontology concepts. Each concept vector is bound from the concept's relation–filler edges (its ontology frame) over a shared, jointly trained dictionary, with no per-concept parameters, and can be re-weighted by a causal context query. This ontology-composed span embedding is added to the input embedding at the term's last subtoken, leaving the tokenizer unchanged. We train it from scratch (50M–125M parameters) and by joint fine-tuning into pretrained small hosts. `[[R3: from-scratch held-out and seen-rare stratified loss vs random span vectors, a free per-concept table, shuffled frames and a hashed n-gram memory, one sentence]]`
>
> On a contamination-free synthetic glossary, joint fine-tuning into SmolLM2-135M/360M lowers next-token loss after held-out terms, whose vectors are composed zero-shot from their frames, by 9–13% (three seeds). The term's own frame carries the gain: swapping in another term's frame costs 24%. On natural text the effect is far smaller, and it follows how new the vocabulary is to the host: 0.4–2.3% on chemistry (ChEBI/PubMed), about 0.5% after rare and multi-subtoken 2026 MeSH names, and nothing on MeSH terms and WordNet words the host already models. Text outside linked terms is unchanged. On natural text the composed vector helps terms absent from the training text, but not held-out terms the host has already read, unlinked, in it; after 2026 MeSH names of that kind it hurts (+0.5%). It also has no edge there over other per-term vectors. A TransE vector of the entry ties it, a definition encoder of the same frame does at least as well after held-out terms, and a code composed from shuffled frames keeps most of the gain `[[A: in-context frames, an inductively folded-in TransE and the clean held-out test T7-ROOD (addendum 2026-10-11)]]`. Nor is the binding operator the source of the gain: fixed random, untyped and translation operators match or beat learned HRR. We therefore describe the mechanism as compositional parameter sharing over ontology frames `[[A: free table at the composed width, C2w]]`. On the synthetic glossary, without any weight update, writing a frame for an invented word modestly improves property selection, and replacing one edge of a seen term's frame shifts predictions toward the new fact. On natural text neither effect depends on the frame `[[C: vs. ROME/MEMIT and in-context editing]]`.

The `[[R3]]` slot is an addition to the §8 text: the from-scratch core result belongs in the abstract once it exists. The earlier E0–E8 skeleton (mechanism identifiability, frozen anchors, R2–R8 slots) moves to the introduction's contribution list; E0's identifiability numbers can return to the abstract as one clause if space allows.

## Contributions, tied to hypotheses

| # | Contribution (allowed wording) | Hypothesis / question | Feeds from | Status now |
|---|---|---|---|---|
| 1 | Ontology-composed span embeddings for causal LMs: concept vectors bound from ontology relation–filler edges over a shared, jointly trained dictionary with no per-concept parameters, added by a gate at the last subtoken of a prefix-causally linked span (re-check: "by fixed binding … added in place … during LM training"; ConceptFormer and, for pretrained hosts, KnowLA are the nearest prior work; novelty check §6: "novel with narrowed wording") | H-A (mechanism) | Method; B5 tests; E4 | method implemented and unit-tested; effect pending |
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
| 12 | **Joint fine-tuning into pretrained hosts (claim A).** Ontology-composed span embeddings jointly fine-tuned into SmolLM2-135M/360M lower loss after held-out glossary terms, composed zero-shot, by 9–13% (three seeds). The own frame carries the gain: swapping in another term's frame costs 24%. On natural text the gain is 0.4–2.3% (T4 chemistry) and about 0.5% after rare and long 2026 MeSH names (T7); it is null on vocabulary the host already models (T1, WordNet), so the benefit follows the vocabulary's novelty. On natural text a TransE row ties it, a definition encoder is at least as good after held-out terms, and the vector is a net cost after held-out terms the host read unlinked in training (frame swap). Positioned against K-Tokeniser, map-tuning/PELT, Vocab Diet, KnowLA, the new-token initialization family (Mundra et al., FVT, Hewitt, Token Distillation) and the ECBD/Onoe et al. in-context-definition baseline | H-A, H-E (pretrained hosts) | E9 (R9) | T5/T4/T7 360M at 3 seeds; frame swap done; A-B1, A-B2, A-B4, A-B5 and A-B6 run: composition-specific only on T5. A-B3 (in-context frames), C2w and C6gi are queued (addendum); T7-ROOD and T8 are pending |
| 13 | **Quantization as a narrow measurement (claim B).** Under 4-bit weight-only PTQ the advantage is preserved on T5 (GPTQ keeps the held-out advantage fully) and partly lost on T4 (GPTQ too; the loss sits in the host weights). On T7 it grows only under round-to-nearest, an effect carried by the channel, and is kept but not enlarged under GPTQ. Framed against the published mechanism that 4-bit PTQ erases small fine-tuning deltas (Zhang et al. 2025; Abitante et al. 2026) and the long-tail literature (Hooker et al.; Marchisio et al.). A measurement, not a headline, and out of the abstract | H-B (pretrained hosts) | E9 (R9) | 3 seeds on T5/T4/T7; B-B2 and B-B3 run on all three; B-B1 (bit-matched side module) deferred (author, 2026-10-11); open anomaly: C2 under AWQ on T7 |
| 14 | **Ontology editing without weight updates (claim C).** On the synthetic glossary, a frame for an invented word modestly improves property selection over random-frame and no-frame controls. One replaced edge of a seen term's frame shifts predictions toward the new fact while preserving neighbourhood specificity; edits transfer to terms unseen in training only at 135M. On natural text (T4, T7) nothing depends on the frame: T7's entailment rise appears with a random frame too. Positioned as a new mechanism inside a pre-empted framing: Facts as Experts, KBLaM, LMLM, REMEDI and map-tuning already edit non-weight knowledge; ROME/MEMIT/AlphaEdit and IKE are the baselines | H-E (editing) | E9 (R9) | proof of concept on T5 (3 seeds); null on T4 and T7; C-B1–C-B6 queued (dimension-3 baselines, priority 60) |
| 15 | **A self-tested protocol for a learnable ontology (claim D, synthetic).** L2-to-prior learnable mapping; candidate edges; relation operators opened one at a time; property-hypothesis testing on held-out pairs against random-tail controls; offline held-out-tested revision. Recovers, discovers, identifies and repairs in a planted world; does not transfer to WordNet on frozen anchors. Precedents: IterE, IRM, OntExt, DreamCoder, L2-SP, Functional Retrofitting | H-H | E10 (R10) | E10.0 supported (synthetic); E10.1 negative; baselines D-B1–D-B5 being run (WP-PQ2); E10.2 pending; **after the E10 baselines (R10): AMIE matches the identifications and null worlds accept 2.7–4.3 false slots per run, so by the framing rule this becomes a negative / methods result, not a protocol claim**. **E10.9a (R10 "E10.9", author's passive + active self-reflection framing): not faster at matched data.** AULC H − P = −0.003 [−0.019, +0.013]; the E10.9b gate is not met. At equal gradient data, committed closure rules add +0.021 [+0.006, +0.035] zero-shot fit through edges for never-observed concepts. Null-world commitments are inert but not retracted. Reported as a methods result with the negative speed reading |

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

**E9/E10 framing rules** (novelty check §2–§5; claims ledger section D):

| E9/E10 outcome | Wording |
|---|---|
| New-token initialization (subtoken mean / encoded definition) or a same-site frame-text vector ≈ composed embedding on held-out terms (A-B1, A-B2) | "frame information at the term's position helps"; no composition-specific claim |
| In-context frame ≥ composed embedding and channel + in-context ≈ in-context (A-B3, C-B2) | the embedding is reported as a zero-context-token alternative with its token savings, not as better |
| Gain only on filler-verbalizing continuation tokens, or absent on T4/T1 (A-B5) | template copying in a synthetic glossary; no "domain terms" claim |
| Bit-matched unstructured module or channel-off INT4 shows the same growth (B-B1, B-B2) | "a separate module survives PTQ"; claim B dropped from the paper's contributions |
| ROME/MEMIT/IKE higher efficacy at equal specificity (C-B1, C-B2) | claim C stays a proof of concept of a zero-gradient, zero-context-token edit |
| AMIE/IterE/KGE match E10.0 or the null world accepts at the same rate (D-B1, D-B2, D-B4) | E10 reported as a negative/methods result; no protocol claim |
| E10.9a: AULC H − P at ρ = 1 includes 0, or H ≈ P+compute (**the 2026-10-04 outcome**) | no "learns faster" claim. The paper may say only that "at equal gradient data, committing held-out-tested closure rules writes edges for concepts no observation covers (+0.021 zero-shot fit, synthetic)", and that "in a null world wrong commitments were inert but not retracted". E10.9b not run |
| E10.9a positive (AULC H − P and H − P+compute CIs exclude 0) | narrowed claim: "passive learning paired with held-out-tested commitment of best-fit relation hypotheses reaches the passive learner's held-out fit with X× fewer observations in a planted world; wrong commitments in a null world cost ≤ … and are retracted …", then E10.9b decides the LM-scale wording |
| E10.2 negative | E10 stays a synthetic-only protocol in the appendix |

**Outcomes of the E9 rules so far (2026-10-12):**
- **A-B1 / A-B2 (same-site rows ≈ composed embedding on held-out terms):** triggered on natural text. On T4 TransE ties everywhere and the definition encoder is ahead after held-out terms; on T7 C5 is behind C5sh and C6d there. The natural-text wording is "frame information at the term's position helps", with no composition-specific claim. T5 keeps the composition-specific claim (C5 beats C6m / C6d / C6g by 10.8–15.3% after held-out terms). C6gi (inductive TransE) on T7-ROOD is the fair re-test.
- **A-B3 / C-B2 (in-context frame):** pending (addendum §3, priority 54.34; T7-ROOD 54.49716).
- **A-B5 (filler copying, or absent on T4/T1):** not triggered on natural text. Most of T4's gain is not on filler tokens (360M: 1 / 51 / 25 / 12% after held-out / unseen / rare / long terms), and T7's is ≈ 0%. T1 is null because the host already models MeSH, not because of copying. On T5, 75–90% of the rare and unseen gain is filler copying, so T5's sizes are an upper bound.
- **B-B1 / B-B2 (separate-module or channel-off growth):** partly triggered. With the channel off, T5's all-token robustness and T4's INT4 loss sit in the host weights. T7's RTN growth is the channel's, but it vanishes under GPTQ. Claim B stays a measurement (contribution 13), out of the abstract. B-B1 is deferred.
- **C-B1 / C-B2 (weight editors, IKE):** pending (priority 60).

## Section structure

### (a) Full paper

| § | Section | Content | Feeds from |
|---|---|---|---|
| 1 | Introduction | the tax; why ontologies; why joint training (frozen-host refutation, HRRBERT); contributions; framing per the decision table | proposal §0–§1, 01a–01c, `[[R3]]` |
| 2 | Related work | knowledge-injected LMs and compositional entity rows (incl. KnowLA, map-tuning, PELT, K-Tokeniser); attention as binding; factorized tables; splitting and growth; multi-token words and lookup memories (incl. the Engram Adapter); vocabulary expansion and new-entity knowledge (Mundra et al., FVT, Hewitt, Token Distillation, Vocab Diet, ECBD/Onoe et al., ALCUNA); knowledge editing outside the weights (Facts as Experts, KBLaM, LMLM, REMEDI; ROME, MEMIT, AlphaEdit, IKE); quantization and the long tail; learning ontology structure (IterE, IRM, OntExt, DreamCoder, L2-SP, reusable holdout); self-authored knowledge; VSA in neural models (incl. Dhanraj & Eliasmith, Dhayalkar) | related-work.md + re-check + novelty check |
| 3 | Method | M0 → M1 → M2 → M3; M4 ontology-composed span embeddings and the causal rule; M5 loop; objective; parameter and byte accounting; sparse generation | formulation.md; B10 |
| 4 | Experimental setup | hardware; corpora and holdouts; tracks; conditions C0–C9 and controls; E9 models P0/C0′/C2/C5; statistics; convergence and escalation; LLM judge | C3/T1/C7 reports, B6, C5, experiments.md §0, execution.md E9 |
| 5 | Mechanism studies | 5.1 E0 synthetic identifiability; 5.2 E1 contextual composition on frozen anchors; 5.3 E2 mapping × operator frontier; 5.4 E3 developmental recovery on WordNet | E0 (done), E1 (D1: gate fail), E2 (done), `[[R2: E3]]` |
| 6 | Core result: small LMs from scratch | ℓ_min feasibility; 50M screen; 125M convergence; stratified losses; controls; C8; probes; `k(L)`; projections; escalation verdicts | `[[R3]]` (D4.7, D4.9, D4.1) |
| 7 | Joint fine-tuning into pretrained hosts (E9) | 7.1 tracks and novelty: T5 synthetic glossary (upper bound), T4 chemistry, T7 2026 MeSH names, T7-ROOD (held-out documents removed: the clean zero-shot test), T8 Wikidata, T1 and WordNet (known vocabulary, nulls); the cross-track novelty index. 7.2 what the composed vector does: the frame swap (held-out, unseen, rare, linked sets) and the controls (operator arms, shuffled frames, same-site rows C6m/C6d/C6g and the inductive C6gi, the wide free table C2w, in-context frames, link dropout C5ld); the mechanism wording (compositional parameter sharing). 7.3 quantization as a narrow DiD measurement (claim B) with the channel-off and GPTQ/AWQ/HQQ/NF4 controls. 7.4 ontology editing (claim C) against ROME/MEMIT/AlphaEdit, IKE and the in-context frame, with frame transplant, channel-off audit and intra-entity locality. 7.5 hosts: SmolLM2 (full fine-tuning; 135M and 360M), Qwen3 and Qwen3.5 (LoRA; reported separately, not pooled); CPT on three hosts and equal-bytes compression `[[R4]]` | R9, addendum 2026-10-11, `[[R4]]` |
| 8 | Explanations and zero-shot | faithfulness; sense alignment; frequency disentanglement; LLM-graded estimates; zero-shot insertion | `[[R5]]` |
| 9 | Self-authored ontologies | authoring quality; round 0 → 1; controls; cross-authoring | `[[R6]]` |
| 10 | A learnable ontology (E10; synthetic protocol) | L2-to-prior learnability; erased-edge recovery vs KGE link prediction; relation discovery vs clustering; property-hypothesis testing vs AMIE/IterE; null-world false-discovery rate; reusable-holdout control; offline held-out-tested revision; WordNet negative result; passive vs passive + active self-reflection at matched data, with the null world as the cost-of-commitment control (E10.9a: not faster; equal-gradient-data closure gain; inert, unretracted wrong commitments); E10.2 | R10, `[[R10: baselines]]`, R10 "E10.9" |
| 11 | Application tracks | cardinality and feasibility (done); per-track results; cross-track table | C3/T1/C7 (done), `[[R7]]` |
| 12 | The operator question | accumulated operator ablation across E1–E9 | E1, E2 (done), `[[R2–R7]]`, E9 A-B6 |
| 13 | Discussion | what the embedding buys and where (new vs known vocabulary); what a negative E4 or E10.2 would rule out; **the human word-learning comparison as a labelled qualitative analogy** (fast mapping ↔ frame inference; consolidation ↔ crystallization; overextension → refinement; propose-but-verify (Trueswell, Medina, Hafri & Gleitman 2013) ↔ the E10.9 hybrid's committed hypotheses vs cross-situational accumulation (Yu & Smith 2007) ↔ the passive learner, citations unverified; E10.9a found that commitment pays only when it licenses predictions beyond the observed pairs, and that "verify" costs data the accumulator could use), with the disanalogies of novelty check §5.4 stated: overextension is relation-slot membership set by the 0.1 initial mass; the rising F1 over k = 1…8 resembles cross-situational learning more than single-exposure fast mapping; crystallization is parameter freezing, not interleaved replay; the depth curve is not a basic-level effect; and "dreaming" denotes generative replay or RL self-play elsewhere (Behrouz et al. 2026), so the revision passes keep the plain name outside the analogy | R10, `[[R3]]` |
| 14 | Limitations | open-clinical substitute; LLM-graded estimates; single-GPU budget; frozen-host refutation; flat frames; linker errors; E9: natural-text effects are small, T4/T7 held-out terms occur unlinked in the training text (T7-ROOD is the clean test), T4 135M at one seed until the addendum runs; claim B depends on track and quantizer; claim-D scope | — |
| 15 | Reproducibility | hardware, env, data hashes, holdouts, commits, compute ledger | [reproducibility.md](reproducibility.md), R0, [paper-readiness.md](paper-readiness.md) |
| App. | A: frozen-host 01a–01c and the audit; B: E0 details; C: cardinality tables; D: sparse generation benchmark; E: judge protocol and calibration; F: E6 (if run); G: pre-registration and deviations; H: E10 full tables | | |

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

The short paper is complete with E0, E2, C3 and E4 alone. E1/E3 enter only as one sentence each if they are done; E5–E8 are referenced as future work. E9 enters as one results subsection (claim A only, with A-B1–A-B4 and ≥ 3 seeds); claims B–D stay out of the short paper.

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
| E9 tables and figures (dimension 1 strata; DiD under INT8/INT4; new words and edits) | generated by `e9_report` (`experiments/e9-retrofit/report/<stage>/`); to wire into `make_figures.py` | T5, T4 and T7 at 3 seeds exist (`report/t5`, `report/t4`, `report/t7`, frame swap under `frameswap/`); dimension-3 baselines and the addendum pending |
| E9 dimension-3 baselines (in-context frame, IKE, ROME/MEMIT/AlphaEdit, transplant, channel-off, intra-entity locality) | `e9_report` dimension-3 baseline section (WP-PQ2) | pending |
| E10 tables (E10.0 synthetic, E10.1 WordNet) and the D-B baselines | `e10_report` (`experiments/e10-self-semantics/runs/*/report.md`; `e10_report --baselines runs/e10.0-baselines-v2`) | E10.0/E10.1 and D-B1, D-B2, D-B4, D-B5 exist |
