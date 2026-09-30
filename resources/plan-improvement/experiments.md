# Experimental program

Nine experiments, dependency-ordered. E0–E3 are cheap and decide the design choices; E4 is the core claim, run in both regimes (from scratch = pre-registered core, continued pretraining of pretrained hosts = second required track); E5 evaluates the explainability and zero-shot claims; E6 is a stretch; E7 tests self-authored ontologies on pretrained hosts; E8 carries the channel into every application track. Everything runs on the local machine (§0.9). Mechanism names (M0–M5) and placements (P0–P2) refer to [formulation.md](formulation.md). Every experiment follows the dossier's [shared protocols](../vsa-understanding/experiments/shared-protocols.md) and run-folder contract, plus the additions in §0.

## 0. Methodology additions specific to this proposal

1. **Linker holdout.** Held-out concepts and *all* their aliases are removed from the linker during training. At evaluation they are linked and composed with zero update. Split labels: `node` + alias-controlled; a `component` subset for the strict track. Definitions never appear in prompts.
2. **Causal-leak test.** A unit test perturbs tokens after position `t` and asserts that no input to positions `≤ t` changes (injection rule M4 §4.2). Any injection scheme that fails this test is excluded from causal-LM runs.
3. **Frequency × span stratification.** Every LM metric is reported by concept-frequency bin (including the zero bin = held-out) and by span length (1, 2, 3+ subtokens), and separately for tokens inside spans and the 8 tokens after a span.
4. **Curves, not endpoints.** Tokens/steps/FLOPs to reach a fixed loss on the rare-concept stratum, plus loss at fixed budget. This is how "fewer parameters and less data" is measured.
5. **Operator ablation is mandatory** wherever a VSA effect is claimed: `{hrr, diagonal, low_rank, bounded_residual, random_fixed, untyped}`.
6. **Two channel controls.** *Random fixed span vectors* (matched parameters; isolates span-boundary information and extra capacity) and *free per-concept span vectors* (KnowBERT/LUKE-style; isolates composition and parameter sharing). A VSA condition that does not beat the first is a boundary detector; one that does not beat the second on held-out concepts has no compositional advantage.
7. **Parameter groups and bytes** reported separately: global / relation / atomic / concept-local / host adapter; plus throughput overhead.
8. **Statistics.** ≥3 seeds for candidates; paired bootstrap 95% CIs over examples; Holm correction across each condition grid; one pre-registered primary endpoint per experiment, stated in its gate. All attempted budgets reported. If compute forces two seeds (C8 only), the report says so.
9. **Local hardware.** One NVIDIA RTX 3090 (24 GB, bf16 ≈ 71 TFLOP/s dense peak; it also drives the desktop, so budget ≈ 22 GB), 12 CPU cores, 30 GB RAM, ≈ 125 GB free disk. Consequences: one GPU job at a time from a queue; CPU work (E0, linking, tokenization, statistics) runs alongside; every training job checkpoints at least every 30 min and resumes bit-exactly (data order and RNG state saved); bf16 autocast, fused AdamW, SDPA/flash attention, activation checkpointing above 125M, gradient accumulation to the target batch, chunked cross-entropy for large vocabularies (Qwen2.5: 151,936); token streams stored as `uint16`/`uint32` memmaps; raw web text is tokenized while streaming and not kept; checkpoints keep last + final weights only.
10. **Span cardinality.** Every linked corpus ships the cardinality table of formulation §4.1 (concepts, occurrences, covered-token fraction, frequency histogram, per tokenizer × `ℓ_min ∈ {1, 2, 3, 4+}`). It is computed before any training in that track and decides whether the track is feasible (enough distinct linked concepts in the held-out and rare strata to power the gate) — see E4.7.
11. **Probe protocol.** Both prompting-style (cloze/likelihood) and linear probes on frozen hidden states for every lexical probe; linear probes are the reported secondary metrics below 1B parameters, prompting results are reported alongside.
12. **LLM grading.** Where human judgement is needed (E5.3, E7 authored-frame quality without gold), an LLM judge grades for now; clinicians and other experts come later with the same items. Protocol: blinded, randomized order and position, rubric fixed before grading, each item graded by ≥2 judges (an API judge with a pinned model version and a local open instruct model on the 3090) with ≥2 prompt paraphrases; report inter-judge agreement (Cohen's κ) and agreement with a small author-labelled calibration set. Items containing MIMIC-derived text are graded **only by the local judge** (credentialed-data terms); ontology-only items (codes, preferred terms, neighbours, edges) may go to the API judge. Verdicts are cached with the judge ID and prompt hash so the later human study rates the identical items.

## E0 — Synthetic identifiability (CPU, weeks 4–7)

Extends the teacher/holdout machinery in [ontology_factorization.py](../../src/vsa_embed/experiments/ontology_factorization.py) and `make_synthetic_relations` in [global_local.py](../../src/vsa_embed/global_local.py). Purpose: show each mechanism recovers what it is designed to recover *when the structure exists*, and does no harm when it does not.

### E0.1 Contextual composition

- **Teacher.** Hidden dictionaries `A*, R*` (`d = 64`), concepts with degree 3–8, `K` context prototypes; edge weights `w*_{i,e}(q) = |F_i| softmax⟨W* q, k*_e⟩`; targets `y_{i,q} = N(Σ w* v*) + noise`. A second teacher with static weights (no context dependence) is the no-harm control.
- **Learners.** M0 static; 01b feature salience; M1-P0 (`q = ∅`); M1 with `q`; free per-(concept, context) weights (upper bound, cannot transfer).
- **Splits.** Composition-disjoint concepts × held-out contexts.
- **Metrics.** Held-out cosine/MSE and kNN overlap; correlation between learned `w` and `w*`.
- **Gate.** M1-with-`q` recovers held-out (concept × context) targets where M0 cannot (report the curve over teacher context-strength); on the static teacher M1 is not worse than M0; a unit test shows `τ → ∞` reproduces M0 to 1e-6.

### E0.2 Split detection

- **Teacher.** Dictionary with `K` atomics of which `K_p` are polysemous (two hidden sense vectors used by disjoint concept subsets); similarly `R_p` relations with two hidden sub-operators under one label. The student starts collapsed (one vector per label).
- **Policies.** None; random splits (matched count); coherence-only screening (idea.md, elementwise, no confirmation); coherence + eigen confirmation (M3 default); second-order splitting (Hessian-vector, small scale only); oracle.
- **Metrics.** Precision/recall of split decisions against the planted set; adjusted Rand index (ARI) between the recovered usage partition and true senses; final loss vs parameters added; **false-split rate on non-polysemous atomics after convergence** (this is the converged-vs-conflicted test of M3 §3.2).
- **Gate.** M3 default reaches ≥0.9 precision and recall at `d = 64`, ARI ≥ 0.8, false-split rate ≤ 5% under the permutation null. Coherence-only is reported alongside to quantify what the cheap screen alone buys.

### E0.3 Factored mapping transfer

- **Teacher.** Edge weights from a rank-`k*` bilinear form.
- **Learners.** M2 options (a) free, (b) induced, (c) hybrid; rank sweep `k ∈ {4, 8, 16, 32}`.
- **Gate.** (a) fails composition-disjoint transfer, (b)/(c) succeed; rank curves recorded for E2.

## E1 — Contextual composition against frozen contextual anchors (local GPU, weeks 8–9)

The cheapest real-data test of M1, before the expensive E4. Extends `contextual_anchors` in [wordnet_relations.py](../../src/vsa_embed/wordnet_relations.py), which currently uses one `"Definition: … Term: …"` template.

- **Hosts.** GPT-2 (existing adapter) and Qwen2.5-0.5B.
- **Data.** WordNet polysemous lemmas with ≥2 senses and ≥10 sense-tagged sentences each (SemCor; WiC for pairs). Targets: centered contextual hidden states at the lemma's last subtoken in each sentence, layer chosen by a small audit (reported).
- **Composition.** Union frame per lemma; `q` from P1 (causal mean of the sentence's preceding token embeddings) and P2 (host hidden state at an earlier layer).
- **Learners.** M0 one vector per lemma; M0 per sense with oracle sense (upper bound); M1 with `q`; operator ablation.
- **Splits.** Node-disjoint lemmas; sentence-disjoint contexts.
- **Metrics.** (i) contextual variance explained `1 − ‖ŷ − y‖² / ‖y − ȳ_lemma‖²` on held-out contexts of held-out lemmas; (ii) retrieval MRR (existing `retrieval_metrics`); (iii) sense accuracy of attention mass against gold senses vs the most-frequent-sense baseline.
- **Gate.** M1 beats M0 on (i) and (ii) with CIs excluding zero over 3 seeds on held-out lemmas, and (iii) beats most-frequent-sense.
- **Caveat.** 01c showed absolute reconstruction in frozen GPT-2 space is poor. E1's endpoints are relative to static composition; a pass is evidence for contextual composition, not for frozen-host row insertion.

## E2 — Mapping × operator frontier (local GPU, week 10)

Extends 01b Stage A/B ([global_local_relations.py](../../src/vsa_embed/experiments/global_local_relations.py), [wordnet_relations.py](../../src/vsa_embed/experiments/wordnet_relations.py)).

- **Grid.** Mapping ∈ {binary, 01b feature salience, M2 induced `k ∈ {4, 8, 16, 32}`, M2 hybrid + `δ`, M1 contextual} × operator ∈ {`hrr`, `unitary_hrr`, `diagonal`, `low_rank`, `bounded_residual`, `orthogonal`, `random_fixed`, `untyped`}.
- **Targets.** 01b's contextual centered anchors and E1's multi-context anchors; a second ontology family (MeSH or a ConceptNet subset) for the transfer requirement.
- **Endpoints.** Node-disjoint MRR and variance explained; data-to-threshold curves; parameter groups.
- **Gate.** M2-induced transfers to node-disjoint concepts at least as well as feature salience (which also has no per-edge parameters) and beats binary; hybrid `δ` improves seen but not held-out concepts (expected; report). The best operator family is carried to E4; if `random_fixed` ties it, that is recorded and E4's C8 decides.

## E3 — Developmental recovery on WordNet (local GPU, weeks 11–12)

Masked-ontology recovery, as the 01d design intended, now with the M3 statistic.

- **Setup.** Collapse senses: one atomic per polysemous lemma across its senses. Collapse relations: `{part, member, substance} meronym → meronym`, `{hypernym, instance hypernym} → hypernym`. Train the best E2 configuration (or the 50M E4 model) with M3 enabled.
- **Policies.** None; uniform enlargement (dictionary × (1 + γ), matched final parameters); random splits (matched count); coherence-only; M3 default; the 01d design's fixed pool of `K = 20` stem-cell experts with entmax routing, implemented minimally, for the relation case.
- **Metrics.** Sense recovery: ARI vs WordNet senses over split lemmas, split precision vs polysemy; relation-subtype recovery: ARI of the edge partition vs original labels; loss and held-out MRR per added parameter; growth curve; cross-seed agreement of partitions; card coherence (human-readable).
- **Gate.** M3 recovers collapsed senses and relations with ARI above random-split and uniform-enlargement at matched parameters (3 seeds, CIs excluding zero), false-split rate ≤ 5% on monosemous lemmas, and improves held-out metrics per added parameter over uniform enlargement.

## E4 — Small-LM joint training (core; weeks 8–20, 350M in weeks 33–36)

### E4.0 Setup

- **From-scratch models (pre-registered core).** Decoder-only, three sizes ≈ 50M / 125M / 350M parameters (GPT-2-small/medium-like configs), bf16, context 1024; tokenizer **GPT-2 BPE** (keeps the single-token lemma tooling and matches the cached GPT-2 host); corpus: a FineWeb-Edu subset. Token budgets are set for the local GPU (§Compute): 50M on 1B tokens (≈ 20 tokens/parameter), 125M on 2.5B (≈ 20), 350M on 2.5B (matched data to 125M, ≈ 7 tokens/parameter; this makes the 125M → 350M comparison a "more parameters at fixed data" contrast, which is what H-A asks). A 350M run on 7B tokens is optional and only for the final best condition vs C0.
- **Continued-pretraining track (required, E4.6).** SmolLM2-135M, SmolLM2-360M and Qwen2.5-0.5B, each with its own tokenizer; see E4.6.
- **Ontologies and linkers.** WordNet lemmas (general); the application tracks of E8 bring SNOMED CT / UMLS / MeSH, API schemas, product taxonomies, ChEBI, glossaries and EuroVoc. Linker restricted to ≥2-subtoken spans by default; single-token rare-word track as a variant; feasibility per `ℓ_min` in E4.7.
- **Held-out concepts.** 10% of linked concepts, node-disjoint, aliases removed from the linker during training; component-disjoint subset for the strict track.

### E4.1 Conditions

Matched non-embedding parameters, tokens, steps and schedule; channel parameters reported.

| Condition | Channel | Isolates |
|---|---|---|
| C0 | none | baseline |
| C1 | random fixed span vectors (matched params) | span-boundary information, extra capacity |
| C2 | free per-concept span vectors | composition vs a per-entity table |
| C3 | M0 static VSA, operator from E2 | structure |
| C4 | C3 + M2 factored mapping | learned mapping |
| C5 | C4 + M1 contextual (P1) | context |
| C6 | C5 + M3 developmental | growth |
| C7 | best of C3–C6 + semantic prediction head (M4 §4.3) | semantic target |
| C8 | best of C3–C7 × operator {`hrr`, `diagonal`, `low_rank`, `bounded_residual`, `random_fixed`, `untyped`} | operator |
| C9 | best of C3–C7 with P2 sidecar (optional, E6 preview) | placement |

### E4.2 Metrics

- LM loss: overall; on unlinked text (locality); stratified per §0.3; tokens-to-loss on the rare-concept stratum.
- Lexical semantics probes at the end of training, each in prompting and linear-probe form (§0.11): WiC; all-words WSD (SemEval 2007/2013/2015); rare-word similarity (CARD-660, Rare Words); hypernymy / lexical entailment (BLESS, HyperLex-style); LAMBADA for general retention.
- Application-track tasks: see E8.
- Throughput overhead, parameter groups, bytes.

### E4.3 Held-out concepts (H-E)

Stratified loss and probes on held-out concepts at zero update. C2 has no rows for them and falls back to a mean vector; that contrast is the point. The 02-style behavioral tests (property selection, paraphrase consistency) are in E5.4.

### E4.4 Quantization (H-B)

Post-training weight-only quantization of every final model at INT8 and INT4 with `torchao` (GPTQ/AWQ-class weight-only; runs on the 3090), dictionary and projector kept in FP16, plus a variant quantizing them too. Endpoint: retained gain of the channel conditions over C0/C2 under quantization.

### E4.5 Table compression (E4c)

Replace rows of linked single-token concepts by `P c + δ` at 4 and 2 bits; equal-bytes comparison against an INT4/INT2-quantized full table.

### E4.6 Continued-pretraining track (pretrained hosts)

The retrofit regime, required alongside the from-scratch core: it repeats the small-model claim on real pretrained hosts and is the regime in which self-authoring (E7) is possible.

- **Hosts.** SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B (all fit the 24 GB GPU with a frozen or LoRA host; Qwen2.5 needs chunked cross-entropy for its 151,936-token vocabulary). The channel is linked per host tokenizer (formulation §4.1), so the `ℓ ≥ 2` concept sets differ across hosts; the cardinality table is reported per host.
- **Data.** 0.5B tokens of the same FineWeb-Edu subset (held-out concepts removed from the linker as in E4.0).
- **Conditions.** C0' (continued pretraining without the channel, same tokens, same LoRA if any), C1, C2, best of C3–C6 from the 50M screen; host frozen vs LoRA (r = 16) on SmolLM2-135M only, then the better one on the larger hosts.
- **Endpoints.** As E4.2–E4.4, relative to C0'. Contamination caveat: pretrained hosts may know held-out concepts; the zero-shot claim on these hosts uses the synthetic private concepts of E5.4, not WordNet held-outs.

### E4.7 Span-cardinality feasibility

Before E4.1, compute the cardinality table (§0.10) for GPT-2 BPE, SmolLM2 and Qwen2.5 on the general corpus and on each E8 corpus. At 50M, run C0, C1 and the best channel condition at `ℓ_min ∈ {1, 2, 3}` (3 seeds). Report, per `ℓ_min`: linked concepts, covered-token fraction, channel overhead, and the gain over C1 in each `ℓ` stratum. This answers how many ≥`ℓ`-subtoken concepts a corpus needs before the channel is measurable, and fixes the per-track `ℓ_min` for E8.

### Gate (pre-registered primary endpoint)

On the 125M from-scratch model:

1. C5 or C6 vs C2: stratified loss on **held-out** concepts improves with CI excluding zero over 3 seeds, and on seen rare concepts matches C2 with fewer channel parameters.
2. C5/C6 vs C1: improvement on every linked stratum (otherwise the effect is span-boundary information).
3. Locality: unlinked-text loss within 0.5% of C0.
4. Probes: at least two of {WiC, rare-word similarity, WSD, domain task} improve with CI excluding zero.

C8 is reported whatever it shows; if `random_fixed ≈ hrr`, the claim becomes "compositional parameter sharing", not "symbolic semantics" (framing confirmed). The size trend (50M → 350M) and quantization retention are secondary endpoints.

### Compute (local RTX 3090)

Planning throughput: `6·N·D` FLOPs (from scratch; `≈ 4·N·D` with a frozen host, no weight gradients) at ≈ 35% MFU of the 3090's 71 TFLOP/s bf16 peak ≈ **2.5e13 FLOP/s**. Replaced by the measured tokens/s from the C4 benchmark before G2.

| Run | Tokens | FLOPs | Per run | Runs | GPU-hours |
|---|---:|---:|---:|---|---:|
| 50M from scratch | 1B | 3.0e17 | ≈ 3.3 h | shake-out (≈ 9) + screen C0–C7 × 3 + C8 × 3 + ℓ_min sweep | ≈ 200 |
| 125M from scratch | 2.5B | 1.9e18 | ≈ 21 h | C0, C1, C2, C3, C5, C6 × 3 (C4/C7 only if the 50M screen separates them) + C8 {`random_fixed`, `untyped`, `diagonal`} × 3 | ≈ 570 |
| 350M from scratch | 2.5B | 5.3e18 | ≈ 58 h | top-4 × 3 | ≈ 700 |
| SmolLM2-135M / 360M / Qwen2.5-0.5B, frozen host | 0.5B | 2.7e17 / 7.2e17 / 9.9e17 | ≈ 3 / 8 / 11 h | 4 conditions × 3 seeds each + LoRA check | ≈ 290 |
| E7 self-authoring | — | — | — | see E7 | ≈ 350 |
| E8 application tracks | — | — | — | see E8 | ≈ 730 |
| E1–E3, E5, E6, quantization evals | — | — | — | — | ≈ 400 |
| **Total** | | | | | **≈ 3,200** |

≈ 3,200 GPU-hours is ≈ 19 weeks of the single GPU at 100% and ≈ 23 weeks at a realistic 85%, so the GPU, not implementation, sets the calendar. Ordering follows value: the 125M core and the continued-pretraining track first, then E5, the clinical track, E7 and the other tracks, and the 350M runs last. No-channel baselines (C0 at every size, C0' on every host) depend on no design decision and run as backfill whenever the GPU would otherwise idle, once the training recipe is frozen at the end of D4.0. The optional 350M × 7B-token pair would add ≈ 330 h. Data loading and the channel add < 10% if generation is sparse (formulation §7).

## E5 — Explainability and zero-shot application evaluations (weeks 21–25)

On E4's best models.

- **E5.1 Faithfulness.** For linked spans, ablate the top-`k` attention-weighted edges vs random edges vs bottom-`k`; measure the change in loss and in probe predictions (comprehensiveness / sufficiency). Expect top-`k` ≫ random.
- **E5.2 Sense alignment.** Attention mass per sub-frame vs gold senses on the jointly trained model (E1 measured this on frozen anchors).
- **E5.3 Rating study (LLM-graded now, clinicians later).** Replicate the paper's neighbour study (28/40 vs 4/40 strongly related for rare codes) on the clinical track (E8-T1), adding ratings of edge-level explanations, M3 cards and E7 authoring cards (are discovered senses, relation sub-types and authored edges meaningful?). Graded by the LLM-judge protocol of §0.12 for an estimate; the identical item set, rubric and pre-registered `n` are kept for a later clinician study. One-tailed Fisher test as in the paper, reported with judge agreement; claims from this stage are labelled "LLM-graded estimate".
- **E5.4 Zero-shot insertion via the span channel.** Dossier experiment 02's strict protocol (reserved concepts, synthetic/private names against contamination, no definition in the prompt), using the span channel instead of tokenizer surgery. Tests: property selection, entailment, paraphrase consistency, and concept-level generation rank via the semantic head. Baselines: matched random, surface mean, definition mean, definition encoder, C2 fallback, graph embedding + projection. Scenarios, one per E8 track: new SNOMED CT / ICD / RxNorm concepts (T1), new API symbols from the synthetic private library (T2), new SKUs and product types (T3), new compounds by IUPAC name (T4), synthetic private glossary terms (T5), new EuroVoc descriptors (T6); on pretrained hosts only the synthetic/private scenarios count as contamination-free.
- **E5.5 Frequency disentanglement.** Frequency predictability from embeddings (the paper's t-SNE claim, made quantitative) for C0 vs channel conditions.

## E6 — Deep injection and readout (stretch, week 37; ties to dossier experiment 03)

P2 sidecar at layer `ℓ ∈ {L/4, L/2}`; a relation readout head that answers "filler of relation `r` for the concept at this span" through attention-with-relation-probe + unbinding + cleanup (formulation §1.2) vs a standard linear probe; depth-2 compositions unseen in training. Gate as in experiment 03: gains over standard QA and random-binding controls on compositional/depth generalization, not only in distribution.

## E7 — Self-authored ontologies on pretrained hosts (formulation M5; weeks 26–28)

Tests the clarifications' key question: can a pretrained small model read text, generalize it into ontology frames, and use them to improve its own weights?

- **Hosts.** SmolLM2-360M and Qwen2.5-0.5B as authors and consumers; SmolLM2-135M as a consumer only (its authoring quality is measured in E7.1 but not expected to suffice). Teacher author: a ≈ 7B instruct model, 4-bit, local.
- **Reading corpora.** (a) General: FineWeb-Edu shards with the WordNet ontology masked for a held-out 20% of concepts (their gold frames used only for audit). (b) Developer tools: documentation and usage corpus of the synthetic private library (E8-T2), whose schemas are the gold. (c) Clinical: PubMed abstracts and MIMIC-IV notes with a held-out SNOMED CT subgraph as gold (MIMIC text never leaves the machine).

### E7.1 Authoring quality

Masked-ontology recovery: the host reads `D_read` and authors frames for candidate concepts. Metrics: edge precision/recall/F1 against the hidden gold subgraph, relation-type accuracy, candidate discovery recall (fraction of hidden gold concepts found by the surprisal screen), LLM-judged plausibility of authored edges that have no gold counterpart (§0.12). Baselines: Hearst/lexico-syntactic pattern extraction, the teacher author, random frames with matched degree.

### E7.2 Self-improvement

Rounds `r = 0..3` of the M5 loop. Conditions (formulation §5.4): gold → host; self → self; self without verification; teacher → host; random frames → host; compute-matched continued pretraining. Three seeds on SmolLM2-360M; the best two conditions plus compute-matched control repeated on Qwen2.5-0.5B.

- **Primary endpoint.** Loss on held-out contexts of concepts that entered the ontology only through authoring (not in the curated ontology), and on masked gold concepts, relative to the compute-matched control.
- **Secondary.** Probes (E4.2), locality on general text, drift across rounds, authored-edge precision per round, growth of the dictionary through M3.

### E7.3 Cross-authoring into a from-scratch model

The ontology authored by the best E7.2 host is used as the linker ontology for a from-scratch 50M and 125M model (C5/C6 configuration), against the same model with the curated ontology and with no channel. Tests whether a pretrained model's self-generated ontology transfers to a model that could not have written it.

### Gate

Self → self beats the compute-matched control and random frames on the primary endpoint with CIs excluding zero (3 seeds), locality within 0.5%, and verification improves authored-edge precision over the no-verification condition. Refutation reading: if self ≈ compute-matched, the model gains nothing from structuring what it read beyond reading it again; if self ≈ random frames, the gain is capacity.

### Compute

Authoring and verification ≈ 20–40 GPU-h per host per round-set; training rounds on 0.25B tokens per round ≈ 2–5 h each; E7.2 ≈ 200 h, E7.3 ≈ 110 h (125M × 2 conditions × 3 seeds is the bulk), E7.1 ≈ 40 h. Total ≈ 350 GPU-hours.

## E8 — Application tracks (T1 weeks 22–24, T2–T6 weeks 29–32)

All application areas from [proposal.md](proposal.md) §6 are explored; the question for each is whether the channel's gains (E4) and self-authoring (E7) carry into that domain. Every track follows the same recipe, so tracks are comparable:

1. ontology adapter → frames, and alias tables → linker;
2. corpus assembly and the cardinality table (§0.10) → per-track `ℓ_min` and feasibility verdict;
3. held-out concepts frozen and hashed (node- and alias-disjoint), plus a synthetic/private-name set for contamination-free zero-shot tests;
4. from scratch: 50M (125M for T1) with C0, C1, C2 and the best channel condition, 3 seeds, on the domain corpus mixed 50/50 with general text;
5. continued pretraining: SmolLM2-135M (360M for T1) with C0', C2 and the best channel condition, 3 seeds;
6. self-authoring (E7 recipe) on the track's corpus where a gold subgraph allows auditing;
7. track tasks and the E5.4 zero-shot scenario.

| Track | Ontology / gold | Corpus | Track tasks | Scale | GPU-h |
|---|---|---|---|---|---:|
| T1 clinical (flagship; licences held) | SNOMED CT, UMLS (ICD-10, RxNorm, MeSH sources) | PubMed abstracts + PMC-OA subset; MIMIC-IV notes (local only) | BLURB subset (NER, relations, via probing), PubMedQA/BioASQ cloze, ICD coding on MIMIC-IV notes (probe), E5.3 neighbour study | 125M + SmolLM2-360M | ≈ 350 |
| T2 developer tools | API schemas and type signatures (synthetic private library + real Python/JS libraries) | docs, READMEs, code comments | symbol → signature/property probes, doc-QA cloze, new-symbol zero-shot | 50M + SmolLM2-135M | ≈ 80 |
| T3 product catalogues | Google product taxonomy, GS1 GPC, product attributes | Amazon ESCI, WDC product corpus | query–product relevance (probe), attribute prediction, new-SKU zero-shot | 50M + SmolLM2-135M | ≈ 80 |
| T4 chemistry | ChEBI (roles, functional groups), PubChem synonyms | PubChem descriptions, open chemistry abstracts | IUPAC-name → class/role probes, property cloze, new-compound zero-shot | 50M + SmolLM2-135M | ≈ 80 |
| T5 enterprise glossaries | synthetic private glossary generator (terms, definitions, relations; contamination-free by construction) | generated internal-style documents | glossary QA cloze, term-relation probes, new-term zero-shot | 50M + SmolLM2-135M | ≈ 60 |
| T6 legal / regulatory | EuroVoc, defined terms extracted from EUR-Lex | EUR-Lex (English) | EuroVoc descriptor classification (probe), defined-term cloze | 50M + SmolLM2-135M | ≈ 80 |
| T7 multilingual low-resource pivot | WordNet / Open Multilingual Wordnet | FineWeb-2 subsets in 1–2 low-resource languages | cross-lingual rare-word similarity, WiC-style tasks where available | 50M | stretch, not in the total |

T0 (general WordNet) is E4 itself. T1 is unconditional now that SNOMED CT, UMLS and MIMIC access is in hand; MIMIC-derived text stays on the local machine and is graded only by the local judge. Waves: T1 and T2 first (flagship and contamination-free demo), then T3–T6 in the order their cardinality tables look most promising; T7 after E6. A track whose cardinality table cannot power the gate is reported as infeasible at that `ℓ_min` rather than run.

**Gate per track.** The E4 gate items 1–3 on the track's held-out stratum and locality, and at least one track task improving with CI excluding zero. Tracks are reported individually and in one comparison table (gain vs span cardinality and ontology depth), which is itself a result: where does the channel pay off?

## Timeline and decision points

GPU-bound on the single local GPU (week numbers match [tasks.md](tasks.md)):

```text
weeks 1–7     audit fixes, infrastructure, E0 on CPU; GPU: audit re-runs, harness dry runs, throughput benchmark
weeks 8–12    E1, E2, E3 and the 50M shake-out; no-channel baselines as backfill → G2: pre-register E4
weeks 13–18   E4: 50M screen, ℓ_min feasibility, 125M core grid           → G3: core claim
weeks 19–20   E4.6 continued pretraining on SmolLM2-135M/360M, Qwen2.5-0.5B
weeks 21–25   quantization, E5.1/5.2/5.5; E8-T1 clinical; E5.3/E5.4    → G4
weeks 26–28   E7 self-authoring                                          → G5
weeks 29–32   E8-T2 … T6
weeks 33–37   E4 at 350M; E6
weeks 36–42   write-up
```

Stop rules: if E0 fails, the mechanisms are wrong in principle — fix before spending GPU time. If E4-125M fails gate item 2, the channel is a boundary detector; if it fails item 1 but passes item 2, report a seen-concept efficiency result without a zero-shot claim. E7 and E8 run whatever G3 shows, because self-authoring and the application tracks are separate questions, but a failed G3 moves them ahead of the 350M runs and drops the 350M runs to C0 vs the best condition.

## Artifacts

Run-folder contract of the dossier (`resolved_config.yaml`, `manifest.json`, `metrics.jsonl`, `report.md`, per-example predictions). New artifact types: `FrameComposerCheckpoint` (dictionary, keys, projector, temperature, operator family), `DevelopmentalLog` (cards), `LinkerManifest` (alias tables, holdout list, hash, linker version), `SpanCardinalityReport` (per tokenizer × `ℓ_min`), `StratifiedLMMetrics` (per-bin losses with CIs), `AuthoringLog` (authoring cards per round, accepted/rejected edges with utility CIs), `JudgeVerdicts` (item, judge ID and version, prompt hash, verdict). Held-out concept lists are hashed and frozen before any training.
