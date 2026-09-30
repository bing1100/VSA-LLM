# Research proposal: a contextual, growable VSA semantic channel for small language models

**Status:** research proposal, 2026-09-30. Source note: [idea.md](idea.md). Companion files: [formulation.md](formulation.md) (the math), [experiments.md](experiments.md) (protocols, gates, compute), [audit.md](audit.md) (state of the current implementation), [tasks.md](tasks.md) (dependency-ordered task list and schedule), [clarifications.md](clarifications.md) (questions and the answers of 2026-09-30, now applied throughout). Background: the dossier in [../vsa-understanding/](../vsa-understanding/README.md).

Claims discipline follows the dossier: **implemented** (in this repository or cited work), **supported hypothesis** (established components, not yet demonstrated here), **research proposal** (needs experiment). Everything in §3–§6 is a research proposal unless marked otherwise.

## 0. Bottom line

idea.md proposes to close the gap between a tokenizer that is optimal for compression and an embedding table that must learn meaning per token, by giving the model a **shared compositional semantic channel**: concept vectors built from atomic and relation vectors, learned in parallel with syntax, attached mainly to words the tokenizer fragments into several tokens. It adds four things the current code does not have: a **query** so that context changes the composition, a **learnable low-rank mapping**, a rule for **growing** the dictionary when gradients conflict, and the **span-level** attachment itself.

Two facts from the repository set the frame:

1. The frozen-host line (fit VSA composition to pretrained rows, experiments 01a → 01b → 01c) is **refuted** for circular convolution, and more firmly after the audit re-runs (protocol 2, 2026-09-30): relation labels carry signal (every structured family beats its derangement-shuffled control), but a capacity-fair diagonal operator matches or beats HRR in every re-run, and the recorded 01b HRR retrieval gain shrinks to +0.0023 (CI [−0.064, +0.069]) once the candidate is named in advance. The 01d growth stage was never implemented and its design had no split statistic.
2. The only regime in which VSA composition has shown a behavioral benefit is HRRBERT's **joint training** regime (atomics trained with the transformer): ROOD-unseen precision 83.5 vs 46.2, physician-rated neighbours 28/40 vs 4/40 for rare codes. That paper never tested parameter efficiency, data efficiency, small models or quantization.

So this proposal moves the program to the joint-training regime on small language models, treats the binding operator as an ablation rather than a premise, and makes idea.md's efficiency, explainability and zero-shot claims into pre-registered, stratified measurements. The primary experiment (E4) trains ≈ 125M-parameter LMs with and without the channel at matched parameters and tokens, with two controls that decide whether any gain is composition (vs a free per-concept table) or merely span-boundary information (vs random fixed span vectors).

Both regimes are tested: from-scratch training is the pre-registered core, and continued pretraining of pretrained small hosts (SmolLM2-135M/360M, Qwen2.5-0.5B) is a required second track. The pretrained regime opens a fifth mechanism: **self-authored ontologies** (M5, E7), where the model reads text, generalizes it into ontology frames, verifies them against held-out text, and uses them to improve its own weights. Every application area in §6 is explored (E8), with the clinical track as flagship now that SNOMED CT, UMLS and MIMIC access is in hand. All work runs on one local RTX 3090 under an evidence-first compute plan: a committed ≈ 400 GPU-hours of short runs measures how fast each condition converges (data multiplier, fitted loss curves, projections), and longer or larger runs are escalation tiers unlocked only when that evidence justifies their cost. Claude implements, the author reviews at the gates.

## 1. Problem and thesis (from idea.md)

BPE learns a segmentation that is good for compression, not for meaning. Every token then gets a free embedding, and the transformer must learn, from data and parameters alone, how multi-token fragments compose into concepts. Call the excess data and parameters spent on that the **syntax → semantics tax**. It is largest for rare, multi-token words and for small models, which is where idea.md expects the channel to help most.

Traceability from the note to this proposal:

| idea.md statement | Mechanism | Hypothesis | Experiment |
|---|---|---|---|
| semantic decomposition learned in parallel to syntax; bridges the syntax→semantics jump | M0/M4 channel trained jointly with the LM | H-A | E4 |
| composition "is similar to attention"; work the query in so context modifies the mapping | M1 attentive composition | H-C | E0.1, E1, E4 C5 |
| make the mapping learnable, LoRA/SVD-like `A × B` | M2 factored mapping | H-A, H-E | E0.3, E2, E4 C4 |
| learn the number of relations/atomics; split when momentum ≈ 0 but mean |gradient| is high | M3 developmental dictionary | H-D | E0.2, E3, E4 C6 |
| add VSA like a position embedding, only for multi-token words | M4 span channel, last-subtoken rule | H-A | E4 |
| fewer parameters and less data for the same performance | stratified tokens-to-loss, matched budgets | H-A | E4 |
| best for quantized and smaller models on complex vocabulary | size sweep, PTQ, table compression | H-B | E4.4, E4.5 |
| explainability through atomic mappings and ontologies | edge attention, cards, unbinding readout | H-F | E5.1–E5.3, E6 |
| zero-shot via new ontology definitions | composition with zero per-concept parameters | H-E | E4.3, E5.4 |
| (clarifications) a pretrained model reads text, writes its own ontology, and improves its own weights | M5 self-authoring loop with held-out-utility verification | H-G | E7 |
| (clarifications) explore every application area | per-track ontology, linker, cardinality table, from-scratch and pretrained runs | H-A, H-E, H-G | E8 |
| (clarifications) feasibility depends on how many words are ≥ `ℓ` subtokens | span-cardinality table per tokenizer × `ℓ_min` | H-A | E4.7 |
| (clarifications) spend few GPU hours; measure convergence speed and build the case for more | data multiplier, fitted curves and projections at every checkpoint; pre-registered escalation rule | H-A, H-B | E4 §0.13, escalation tiers X1–X4 |

## 2. Where the repository stands

| Item | State | Consequence for this proposal |
|---|---|---|
| `vsa_embed` library | implemented: real/unitary HRR, MAP, six relation-transform families, bounded residual relations, factorizer, calibration, WordNet adapters, 66 passing tests | reuse the algebra, transforms and split machinery; add composer, context, growth, span channel |
| 00 capacity | done: D=512 ≫ D=256; heavy-tailed degree is the failure regime; real HRR ≈ unitary ≈ MAP; no universal backend | cap or shard high-degree frames; do not build memory infrastructure |
| 01a fixed-path factorization | refuted: additive 0.139 kNN overlap > HRR 0.107 | role binding must be an ablation |
| 01b global–local relations | recorded: HRR MRR 0.271 > additive 0.252 (test-selected; CI [0.007, 0.032]). **Protocol-2 re-run** (HRR named in advance, mid-rank ties, derangement shuffles): HRR 0.228 vs additive 0.225 (+0.0023, CI [−0.064, +0.069], 1/3 wins); diagonal 0.254 best; HRR > shuffled HRR in 3/3 seeds; cosine −0.039 | relation labels matter; no evidence that circular convolution beats identity or diagonal; retrieval and reconstruction still disagree |
| 01c reconstruction rescue | refuted, and confirmed by protocol-2 re-runs with a capacity-fair rotated-diagonal control: 01c.3 HRR cosine gain over offset +0.004 (CI [−0.008, +0.017]), control ≥ HRR; 01c.4 HRR − control −0.001 (CI [−0.007, +0.004]); 01c.5 with parents held out: diagonal control 0.360 > rotated 0.341 > HRR 0.324, and the recorded rank-4 low-rank lead (0.439) was leakage — it drops to 0.338 and ties its shuffled control | frozen-host row insertion is off the table; bounded relation-conditioned corrections are the strongest one-hop operator so far |
| 01d developmental discovery | designed, not implemented; no split statistic | M3 fills the gap |
| HRRBERT paper | joint regime; ROOD-unseen and physician-rated gains; no efficiency/small-model tests; column-wise normalization bug; identical vectors for identical recipes | test the untested claims; carry the fixes |
| Literature review | no coverage of contextual VSA, splitting/growing dictionaries, low-rank mappings, multi-token word embeddings, quantized models | external search required before any novelty claim (§7) |

Three rules from the dossier are carried unchanged: exact recipe/edge groups never cross a split; a random-composition control isolates structure from parameter sharing; equal parameters and compute, reported by group.

## 3. Hypotheses

Each hypothesis names its measurement and what refutes it.

- **H-A (tax).** At matched non-embedding parameters and tokens, the channel lowers loss on tokens inside and after multi-token linked words, more so for rarer concepts and smaller models, and reaches a fixed loss on that stratum with fewer tokens. *Refuted if* the gain is not larger than that of random fixed span vectors (C1), or vanishes at 125M.
- **H-B (small and quantized).** The gain survives INT8/INT4 weight quantization, and the dictionary + composition replaces linked embedding rows at fewer bytes for equal quality. *Refuted if* retained gain under quantization is smaller than for the free-table control, or the compressed table loses more than the equal-bytes quantized table.
- **H-C (context).** Query-conditioned composition explains more contextual variance of polysemous words than static composition, and its edge attention aligns with gold senses above the most-frequent-sense baseline. *Refuted if* E1/E5.2 show no alignment or no variance gain.
- **H-D (growth).** Gradient-conflict splitting recovers collapsed senses and relation sub-types (ARI above random splits and uniform enlargement at matched parameters) with a false-split rate ≤ 5%. *Refuted if* uniform enlargement matches it per added parameter.
- **H-E (zero-shot).** Concepts absent from training (node-disjoint, alias-controlled) obtain useful behavior from composition alone, beating surface/definition means and the free-table fallback. *Refuted if* definition-mean initialization ties it.
- **H-F (explainability).** Edge attention is faithful (ablating top-weighted edges changes predictions more than random edges) and rated explanations improve over nearest-neighbour explanations (LLM-graded estimate first, clinicians later). *Refuted if* top-`k` ablation ≈ random ablation.
- **H-G (self-authoring).** A pretrained small LM that reads unannotated text can author ontology frames which, after held-out-utility verification and fed through its own channel, lower its loss on the authored concepts more than compute-matched continued pretraining on the same text and more than random frames, and recover hidden gold edges above pattern-based extraction. *Refuted if* self-authored ≈ compute-matched continued pretraining, or ≈ random frames at matched edge count.

A cross-cutting question, not a hypothesis: does the **binding operator** matter in the joint regime? Reported by the operator ablation in every experiment; either answer is publishable.

## 4. Method overview

```text
tokens  x_1 … x_T ──► E[x_t] + p_t ─────────────────────────────┐
                                                                 ├─► transformer ─► L_LM
linker  spans (s,e,concept j) ─► frame F_j = {(r_e, a_e)}        │
        atomics A, relations R ─► v_e = T_r(a_e)                  │
        query u_j(q_t) = W_Q c̄_j + W_C q_t   (q_t: local context) │
        weights w = |F_j|·softmax⟨u, k_e⟩/τ                       │
        c_j(q_t) = N(Σ w v_e)  ─► gate · P c_j  added at t = e ───┘
        (optional) predict c_j from h_{s−1}  ─► L_sem
growth  Adam state → coherence κ, activity α → candidates → usage-level split gain G* → split/merge/allocate
author  (pretrained host) read text → surprisal candidates → authored frames → held-out utility → linker + dictionary
```

- **M0** static bundle (existing), with row normalization and duplicate-safe accumulation.
- **M1** attentive composition: the frame is a key–value memory; uniform attention is bundling; a relation probe is unbinding; the context query re-weights edges. Zero per-concept parameters, so it applies to unseen concepts. `τ → ∞` recovers M0.
- **M2** factored mapping: rank-`k` weights on the ontology support with the concept factor induced from the frame (transferable) or free (upper bound).
- **M3** developmental dictionary: idea.md's screening statistic is Adam's per-coordinate signal-to-noise ratio; the confirmation statistic is a first-order split gain `Σ_u |⟨m^{(u)}, v⟩| − |Σ_u ⟨m^{(u)}, v⟩|` along the top between-usage direction, tested against a permutation null; children are offset along that direction and usages partitioned by sign, which is what makes identical copies drift apart.
- **M4** span channel: deterministic linker on character offsets (one linked corpus for every tokenizer), injection at the last subtoken (no leakage of future subtokens in causal LMs), a confidence gate, an optional semantic-prediction head, a table-compression variant, and a span-cardinality report per tokenizer × `ℓ_min`.
- **M5** self-authored ontologies (pretrained hosts): discover poorly modelled multi-token spans by excess surprisal, author frames under a constrained template, keep only edges whose first-order utility on held-out contexts is positive with confidence, integrate them into the linker and dictionary (M3 allocates and splits), train, audit against hidden gold, repeat.

Full definitions, cost model and limits: [formulation.md](formulation.md).

## 5. Experimental program overview

| Exp. | Question | Depends on | GPU cost (local 3090) | Gate (primary endpoint) |
|---|---|---|---|---|
| E0 | do M1/M2/M3 recover planted structure and do no harm without it? | — | CPU | contextual recovery; split precision/recall ≥ 0.9, false splits ≤ 5%; transfer of induced factors |
| E1 | does contextual composition explain contextual variance of polysemous words on a frozen host? | E0.1 | ≈ 20 h | variance explained and MRR over static, sense alignment > MFS |
| E2 | which mapping and operator transfer to node-disjoint concepts, at what data cost? | 01b code | ≈ 20 h | induced mapping ≥ feature salience; operator family chosen |
| E3 | does M3 recover collapsed WordNet senses and relation sub-types? | E0.2, E2 | ≈ 20 h | ARI > random/uniform growth at matched params |
| **E4** | does the channel reduce the tax (and learn faster) in 50M/125M LMs from scratch and in pretrained hosts; does it beat the free-table and random-vector controls; does it survive quantization; which `ℓ_min` is feasible? | E2, E3 | ≈ 190 h | held-out-concept loss < C2, all linked strata < C1, locality ≤ 0.5%, ≥2 probes improve |
| E5 | are explanations faithful and well rated; does zero-shot insertion work via the span channel? | E4 | ≈ 40 h (with quantization) | top-`k` ≫ random ablation; 02-protocol baselines beaten |
| E6 | can the model use the algebra (readout, depth) with mid-network injection? | E4, E5 | tier X4 | 03-style compositional gains |
| E7 | can a pretrained host author, verify and use its own ontology? | E4.6 | ≈ 40 h (round 1) | self-authored > compute-matched continued pretraining and random frames |
| E8 | do the gains carry into clinical, developer-tools, product, chemistry, enterprise and legal domains? | E4, E7 | ≈ 70 h | E4 gate on each track's held-out stratum + one track task |

≈ 400 committed GPU-hours on one RTX 3090 (≈ 3 weeks of GPU time), ≈ 22 weeks end to end, set by implementation; decision points after E0, after E2/E3, after E4-125M, after E5, after E7, and a final evidence report that requests escalation tiers (X1 125M at 2.5B tokens ≈ 190 h; X2 350M ≈ 700 h; X3 longer continued pretraining and multi-round self-authoring ≈ 350 h; X4 full tracks, E6, multilingual ≈ 600 h) only where the convergence evidence meets the pre-registered rule. Details: [experiments.md](experiments.md), schedule in [tasks.md](tasks.md).

## 6. Real-world applications

### Criteria

An application is a good fit when: (1) a curated ontology with decent coverage exists; (2) its terminology is multi-token and long-tailed; (3) new concepts arrive often, so zero-shot rows have value; (4) explanations or audit are demanded; (5) small or quantized models are how it is deployed; (6) evaluation can be made contamination-free with public data; (7) data and licences are accessible.

### Ranking

| Rank | Application | Ontology | Multi-token long tail | New-concept rate | Explainability demand | Small/quantized deployment | Clean eval | Access | Verdict |
|---:|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | Clinical coding and extraction (SNOMED CT, ICD, RxNorm, MeSH) | ●●● | ●●● | ●●● | ●●● | ●●● | ●● | ●●● | Highest fit and the team's prior pipeline; SNOMED CT, UMLS and MIMIC access is held (flagship, T1) |
| 2 | Developer tools: API, library and tool vocabularies for small code/agent models | ●● | ●●● | ●●● | ●● | ●●● | ●●● | ●●● | Fastest clean proof: identifiers are multi-token, type signatures and docs are frames, synthetic private APIs give contamination-free zero-shot tests, IDE-local models are small and quantized |
| 3 | Product catalogues and e-commerce search (GS1/Google taxonomy, attributes) | ●● | ●●● | ●●● | ●● | ●●● | ●● | ●● | Strong commercial case; public data thinner (ESCI, WDC) |
| 4 | Chemistry and materials (ChEBI, PubChem, IUPAC nomenclature) | ●●● | ●●● | ●●● | ●● | ●● | ●● | ●●● | Purest compositionality test: IUPAC names are compositional syntax and functional groups are natural atomics; narrower market |
| 5 | Enterprise glossaries and private jargon for on-prem assistants | ● | ●●● | ●● | ●●● | ●●● | ● | ●● | High practical value, no public benchmark; use the synthetic-private-concept protocol |
| 6 | Legal and regulatory (EuroVoc, LKIF, defined terms) | ●● | ●●● | ●● | ●●● | ●● | ●● | ●●● | Ontologies are weak; explanations matter |
| 7 | Multilingual low-resource pivot | ●● | ●● | ● | ● | ●● | ●● | ●● | Dossier experiment 04; defer |

Not recommended now: open-domain chat assistants (ontology coverage is poor and large models already pay the tax), VSA as memory infrastructure (blocked by experiment 00), and replacing the tokenizer (the channel sits on top of a fixed tokenizer by design).

### Decision (clarifications 2, 11)

Every area above is explored (E8), with a common recipe so the tracks are comparable: the general-language WordNet track is the scientific core of E4; **clinical** (SNOMED CT, UMLS, MIMIC; access confirmed) is the flagship and runs first; **developer tools** runs alongside it as the contamination-free demo (a small model that understands a new library from its schema without fine-tuning); product catalogues, chemistry, enterprise glossaries (synthetic private glossaries) and legal follow; multilingual is a stretch after E6. Each track also hosts a self-authoring run (E7 recipe) where a gold subgraph allows auditing. A track whose span-cardinality table cannot power its gate is reported as infeasible rather than run. The ranking above now orders the waves rather than selecting tracks.

## 7. What is new, and what may be claimed (after the literature search)

The external search (task C1) is in [related-work.md](related-work.md), with verified citations. It narrows every mechanism's novelty claim; the allowed wording is:

| Mechanism | Closest prior work | Allowed claim |
|---|---|---|
| M1 attentive composition | Dasigi et al. 2017; SE-WRL (Niu et al. 2017); KnowBERT; Schlag et al. 2021; Hrrformer | a query-conditioned weighting of VSA-bound ontology edges with zero per-concept parameters that recovers the static bundle as `τ → ∞` — not "first contextual composition" or "unifies attention and binding" |
| M2 factored mapping | LoRA; ALBERT; NodePiece; QR trick; TT-Rec | an ontology-induced, transferable rank-`k` edge weighting; compression only against equal-bytes factorized, hashed, TT and PTQ tables |
| M3 developmental dictionary | Splitting Steepest Descent; Firefly/GradMax; G-means; Linde–Buzo–Gray; AdaGram/NP-MSSG; PCGrad; GSNR | a first-order usage-level split test for shared dictionary vectors, read from optimizer state and calibrated by a permutation null; the ±ε offset and sign partition are credited to splitting / G-means / LBG |
| M4 span channel | KALM (a causal LM with entity input and entity prediction); Bootleg; UmlsBERT; ERNIE/KnowBERT/LUKE; Engram; Kaplan et al. 2025 | a compositional, growable span channel for causal LMs whose rows are built from ontology relation–filler edges, with a last-subtoken rule (independently supported by Feucht et al. 2024 and Kaplan et al. 2025) |
| M5 self-authoring | EntiGraph; Symbolic Knowledge Distillation; OLLM/LLMs4OL; STaR; TracIn/LESS; Gerstgrasser et al. | self-authored frames verified by held-out gradient utility (a TracIn/LESS-style inner product), consumed through the channel, beating compute-matched and EntiGraph-style controls |
| H-A tax | Feucht et al.; Kaplan et al.; Engram; Allen-Zhu & Li | measures the tax on ontology-linked multi-token strata and whether structure beats a hashed lookup memory |
| H-E zero-shot | CoLLEGe; Bahdanau et al. 2017; à la carte; BERTRAM | structure-only zero-shot rows for unseen ontology nodes; text-evidence generators reported separately |
| VSA in LMs | HRRBERT (Hu et al., KDD 2024 workshops, CEUR-WS 3894); Hrrformer | extends HRRBERT to causal text LMs, with the operator as an ablation |

Baselines added because of the search (in [experiments.md](experiments.md)): a hashed n-gram span memory at matched parameters (the sharpest test of "structure vs any extra lookup"), a type/relation-only composition without fillers, context attention over free sense vectors in E1, GRAM on the clinical track, equal-bytes compression tables, a G-means-style Anderson–Darling split test and NP-MSSG/AdaGram sense induction in E0.2/E3, à la carte and CoLLEGe-style generators for zero-shot, and EntiGraph-style synthetic continued pretraining plus OLLM/LLMs4OL prompting in E7. The two claims most exposed to a missed 2025–26 preprint (usage-level momentum-conflict splitting; context attention over HRR-bound edges) are re-checked before any submission.

## 8. Risks and failure interpretation

| Risk | Mitigation / reading |
|---|---|
| Gain comes from span-boundary information, not composition | C1 control; if C1 ≈ C5, report a boundary-signal result, not a VSA result |
| Gain comes from parameter sharing, not symbolic structure | operator ablation with `random_fixed` and `untyped`; claim accordingly |
| Linker errors and coverage; polysemy under-linked | confidence gate, union frames, coverage reported per corpus; E5.1 faithfulness catches harmful links |
| Leakage through aliases, definitions, or injection position | linker holdout, no definitions in prompts, causal-leak unit test |
| Splitting is unstable or grows without bound | permutation null, budget, cooldown, consolidation phase, false-split rate as a gated metric |
| Contamination of pretrained hosts | from-scratch models for the core claim; synthetic private concepts for zero-shot |
| Probe evaluations on small LMs are noisy | paired bootstrap over examples, several probes, pre-registered endpoints |
| Licences and data terms (SNOMED, UMLS, MIMIC) | access held; MIMIC-derived text may go to the Claude judge (confirmed) but is never committed or published; licence versions recorded in the linker manifest |
| Compute: one RTX 3090 (24 GB), shared with the desktop | ≈ 400 committed GPU-h of convergence-speed runs; larger runs only through the escalation rule; checkpoint/resume every 30 min; measured throughput replaces the planning figure before G2 |
| Short runs mislead: early-training gains can wash out | fitted curves with projected gaps and their CIs are part of the escalation rule; the report says whether a gain is early-only |
| Self-authoring reinforces its own errors | disjoint authoring/validation contexts, held-out-utility acceptance, replay and locality gate, hidden gold audits, compute-matched control |
| LLM-graded ratings are not expert ratings | Claude Code as judge with the model pinned, repeated calls and prompt paraphrases, self-agreement and calibration reported; claims labelled as estimates; identical items kept for the later clinician study |

Failure readings from the dossier still apply: good geometry with poor behavior means the interface is missing; seen concepts work but held-out fail means the mapping memorizes; structure that only reduces adaptation cost is a few-shot result, not zero-shot.

## 9. Implementation plan

New modules in `vsa_embed`, each with tests, building on what exists:

| Module | Contents | Builds on |
|---|---|---|
| `compose.py` | `FrameComposer`: CSR frame schedule, batched bind, segment-softmax weights, `index_add_` scatter; M0/M1/M2 switches; `explain(concept)` | `algebra.py`, `relations.py`, the `FastCVGen` schedule |
| `context.py` | P1 causal local-context encoder; P2 sidecar hook | — |
| `developmental.py` | statistics from Adam state, usage-level EMAs, split gain, permutation null, split/merge/allocate, cards, budget | `relations.py` adjoints |
| `span_channel.py` | linker (alias tables, holdout), injection with last-subtoken rule, gate, semantic head | — |
| `integrations/transformers.py` | `inputs_embeds` hook for HF causal LMs (GPT-2, SmolLM2, Qwen2.5); frozen/LoRA host; tied-output helper; chunked cross-entropy | — |
| `authoring.py` | M5: surprisal candidate discovery, constrained frame authoring, held-out utility with bootstrap bounds, authoring cards, round driver | `compose.py`, `developmental.py`, `span_channel.py` |
| `judging.py` | LLM-judge harness: rubric, blinding, randomization, headless Claude Code calls with a pinned model, verdict cache, agreement statistics | — |
| `convergence.py` | per-stratum checkpoint losses, data multipliers, curve fits with bootstrap projections, escalation-rule verdicts | — |
| `experiments/` | E0–E4 runners under the run-folder contract | existing runners |

Tests to add: utility `U_{j,e}` equals the finite-difference loss change for small `β`; authoring and validation contexts disjoint; `τ → ∞` equals uniform bundle; segment-softmax equals a loop; per-usage gradients equal autograd per-sample gradients; split gain is ≥ 0 and 0 when all usages agree; permutation null calibration on noise; no leakage from held-out rows; causal-leak test; row normalization; duplicate-index accumulation.

Housekeeping before new work, from the implementation audit in [audit.md](audit.md). Two recorded statements need correcting: the 01c.5 "rank-4 low-rank transfer is best" claim is a leakage artifact (test parents seen in training: low-rank MRR 0.570 seen vs 0.171 unseen), and the 01b "relation accuracy above additive" claim compares against a metric that degenerates to the hypernym share for parameter-free models. Before any E1–E3 run reuses these runners: target-disjoint neighbourhood splits, pre-named candidates with a validation partition, shared candidate sets and tie handling in every metric, the rank-loss positive fix, capacity-matched controls, `promotion_eligible=False` everywhere, and manifests with git SHA and versions. Also update the stale "proceed now: 01c" status in the dossier index and record 01c as refuted in the program README.

## 10. Decision rule

Continue past E4 only if the channel shows at least one advantage a simpler method does not: (a) better held-out-concept behavior than the free-table control at matched parameters, (b) materially fewer bytes at matched quality, (c) faithful, well-rated explanations without a quality loss (LLM-graded first, expert-validated before any external claim), or (d) self-authored ontologies that beat compute-matched continued pretraining (E7). If the operator ablation shows `random_fixed ≈ hrr`, the write-up reports compositional parameter sharing, not symbolic semantics (confirmed). Drop any mechanism whose ablation shows no effect; keep the operator finding whichever way it goes.
