# Novelty check, 2026-10: claims A–D (E9, E10) and the method as a whole

**Date:** 2026-10-03.

**Purpose:** a pre-manuscript novelty check of the four new result claims and of the method as a whole, before more GPU time is committed. The four claims:
- **A.** E9 retrofit.
- **B.** E9 quantization.
- **C.** E9 ontology editing.
- **D.** E10 self-learned semantics.

**Builds on:** [related-work-recheck-2026-10.md](related-work-recheck-2026-10.md) (2026-10-02, mechanisms M1–M5), which this note does not repeat, and [related-work.md](../resources/plan-improvement/related-work.md) (2026-09-30).

**Inputs read:**
- [proposal.md](../resources/plan-improvement/proposal.md) §7 (allowed claims) and [formulation.md](../resources/plan-improvement/formulation.md);
- [claims.md](claims.md);
- [R1](../reports/R1-sanity-pilot.md), [R9](../reports/R9-retrofit-quantization.md) and [R10](../reports/R10-self-learned-semantics.md);
- the HRRBERT paper ([vsa-paper.md](../resources/vsa-paper.md)).

This note edits no other file. The orchestrator decides what to adopt into claims.md, the draft and the proposal.

**Method.** Five parallel search threads ran: one per claim A–D, plus one for the method as a whole. The last also re-checked the two "most exposed" claims of the 2026-10-02 note. The orchestrating check then re-opened every work used as a "closest work".

Search conditions:
- The session's WebSearch budget (200 calls) was already exhausted when the threads started. They searched through fallbacks instead:
  - the OpenAlex API (until its daily quota ran out);
  - Crossref, DataCite (arXiv DOIs) and Europe PMC;
  - the OpenReview search API (forum pages are bot-blocked);
  - the arXiv export API and arXiv website search (intermittently rate-limited, 429);
  - Hugging Face paper search;
  - Google Scholar, ACL Anthology and proceedings pages, fetched directly.
- About 650 queries and lookups were run in total (§10).

**Verification standard.**

| Mark | Meaning |
|---|---|
| **✓✓** | Re-opened for this note on 2026-10-03 by the orchestrating check (arXiv abs, ACL Anthology, proceedings or publisher page); title, authors and year matched |
| **✓** | Verified the same way by the search thread that found it |
| **(m)** | Metadata only (a Crossref, DataCite, OpenReview-API or PubMed record). Usable as a pointer; open the paper before quoting its content |

A venue is stated only where a proceedings page or the arXiv comment confirms it; otherwise it is "venue unverified". Snippet-only items are in §9 and must not be cited until checked.

**Limitations.**
- The degraded search tooling makes absence of a hit weaker evidence than usual.
- Springer, ACM DL, PNAS and OpenReview forum pages could not be opened.
- 2026 preprints appear quickly in this area (the Engram family, knowledge editing, vocabulary expansion). Re-run §1 and the closest-work checks within 30 days of submission.

---

## 0. Executive verdicts

| Claim | Verdict | Why, in one line | Closest prior work | Must-run before claiming |
|---|---|---|---|---|
| **A** retrofit: lower loss after new/held-out domain terms | **Novel with narrowed wording** | The combination is new. The free-table comparison is trivially won on held-out terms, and the T5 text is generated from the same ontology | K-Tokeniser; map-tuning; Vocab Diet; ECBD / Onoe et al. 2023; the vocabulary-initialization family (Mundra et al., FVT, Hewitt, Token Distillation); KnowLA (§6) | New-token initialization (subtoken mean, definition encoding); in-context verbalized frame; same-site text-encoder / KGE vector; lexical-overlap split; natural-text tracks; ≥ 3 seeds |
| **B** advantage grows under INT4 | **Novel only as a narrow, control-backed measurement; the proposed headline is not supported** | A published mechanism (4-bit PTQ erases small full-fine-tuning deltas, so knowledge in a separate module survives) predicts the pattern for *any* side module. R9's 2.8× is driven by unlinked text, where the channel is inactive. 360M is mixed. One seed, RTN only | Abitante et al. 2026; Zhang et al. ICLR 2025; Twagirayezu & Mitra 2026; Hooker et al.; Marchisio et al.; Kumar et al.; Catalan-Tatjer et al. | Bit-matched unstructured side module; channel off at INT4; GPTQ / AWQ / NF4; DiD reporting; revert diagnostics; ≥ 3 seeds |
| **C** zero-shot words and edits by ontology editing | **The general framing is pre-empted; the specific mechanism is novel with narrowed wording** | "Edit a symbolic store instead of weights" is Facts-as-Experts (2020/21), KBLaM, LMLM and REMEDI. Our version (one bound edge in an HRR-composed input row; invented words as new atom combinations) is new but modest and works only for seen terms | Facts as Experts; REMEDI; KBLaM / LMLM; map-tuning / PELT; Onoe et al. 2023 and Padmanabhan et al. 2023; ALCUNA | ROME / MEMIT / AlphaEdit and IKE on the same items; in-context frame upper bound; new-word embedding baselines; frame transplant; channel-off audit; intra-entity locality; ≥ 3 seeds |
| **D** self-learned semantics | **Novel only as a protocol; synthetic-scoped wording** | Every part has precedent (L2-SP; IterE property axioms with closure; IRM; OntExt; DreamCoder wake–sleep; co-training / confirmation bias). The held-out-vs-control acceptance protocol over unnamed relations inside a VSA channel is new. The evidence is synthetic, and the WordNet transfer is negative | IterE; IRM / structural form; OntExt; DreamCoder; L2-SP / Functional Retrofitting; open relation discovery | AMIE / IterE property baselines; KGE link prediction for recovery; clustering / TransG baselines for discovery; null-world false-discovery rate; reusable-holdout control; WN18RR hidden relations |
| **Method as a whole** (HRRBERT → causal LMs) | **Novel with narrowed wording** | No work combines HRR-bound ontology edges over a shared, jointly trained dictionary with prefix-causal last-subtoken injection and joint training (from scratch and retrofit). New neighbours rule out broad "first" phrasings | KnowLA (NAACL 2024; *not in the earlier notes*); Dhanraj & Eliasmith (EMNLP 2025); Dhayalkar (KnowFM 2026, VSA KG injection proposed, no experiments); Engram Adapter (Findings EMNLP 2026); ConceptFormer (WWW '26) | KnowLA-style per-concept KGE at our site and at KnowLA's; KGE-initialized jointly trained table; verbalized-frame text baseline; retrofit Engram adapter; parameter-matched LoRA plus an OOD-retention suite; injection-site ablation; linker-noise curve |

**Bottom line.** Nothing pre-empts the method or claims A, B or D outright. Claim C's *framing* is pre-empted. Every claim needs narrower wording, and each has a short list of baselines a reviewer will ask for first:
- **A, C:** in-context frame; new-token / definition initialization.
- **B:** a bit-matched side module plus GPTQ/AWQ.
- **C:** ROME / MEMIT / IKE.
- **D:** rule-mining and KGE baselines.
- **All:** ≥ 3 seeds.

The 2026-10-02 note's "most exposed" claims (M1, M3) remain unpre-empted (§1).

---

## 1. Re-verification of the two "most exposed" claims (2026-10-02 note)

| Claim | 2026-10-03 result | New nearest work | Change |
|---|---|---|---|
| **M1:** context attention over HRR/VSA-*bound* ontology edges with zero per-concept parameters; the τ → ∞ limit is the HRRBERT bundle | **Not pre-empted** | **KnowLA** (NAACL 2024, §6). Its context query (mean of the word's subtoken states) attends over *candidate entities* with frozen per-entity KGE vectors, not over one concept's bound edges, and it has no zero-per-concept parameterization. Dhayalkar's binding/unbinding heads (arXiv:2512.14709; KnowFM 2026) are unimplemented proposals | Add KnowLA next to CokeBERT and Dasigi et al. in the M1 contrast |
| **M3:** usage-level momentum/gradient-conflict splitting of shared dictionary vectors with a permutation null | **Not pre-empted** | MoME (Li, Sigal & Liao, arXiv:2609.15126, Sep 2026, ✓✓) remains the nearest: a fixed number M of gated slots per token, no split test. Lin et al. (arXiv:2601.11610) is confirmed as the POI-recommendation hypergraph paper that splits scalar parameters on scenario-gradient conflict, as the earlier note described | None. Reviewer point to pre-empt: Tao Lin (arXiv:2601.16531) reports that hash collisions act as a regularizer in Engram-style memories, so splitting shared vectors could remove a useful regularizer. Keep the E0/E3 random-split and no-growth controls |

**Search.** Seven arXiv website queries by the orchestrating check (on HRR/VSA + transformer / LLM / knowledge; gradient conflict + split; multi-sense + LM pretraining) and about 20 arXiv-API and Scholar queries by the method thread. No paper from 2025–2026 came closer than the ones above.

Also new, but not threats, are several VSA–LLM papers that do not compose ontology rows:
- Hyperdimensional Probe (arXiv:2509.25045, which decodes LLM states);
- Holographic Neural PCFG (arXiv:2607.08063);
- HoloByte (arXiv:2603.16917, tokenizer-free distillation);
- *Learning Holographic Reduced Representations with Clifford Variational Autoencoders* (2026).

These come from arXiv listings and OpenAlex records only; they were not opened.

---

## 2. Claim A — retrofit into pretrained hosts (E9, dimension 1)

### 2.1 Verdict: novel, with narrower wording

No paper found (through October 2026) does all of the following together:
- composes vectors for multi-token domain terms from ontology relation–filler frames;
- adds them at the last subtoken of a **pretrained decoder**, without changing the tokenizer, trained jointly with the host;
- reports next-token loss on text after **held-out, never-linked terms** against a free per-concept table.

Every ingredient has close precedent, though:
- ontology-informed vectors for new domain tokens (K-Tokeniser);
- KG vectors mapped into a pretrained model's input at mention spans (map-tuning, PELT);
- compositional vectors that extend a pretrained LLM to unseen words (Vocab Diet);
- the evaluation target itself, LM loss on text about unseen entities, with in-context definitions as a strong baseline (ECBD; Onoe et al. 2023).

The current comparison set (C0′, C2) is the weak point. A free per-concept table cannot help a held-out term **by construction**. The baselines that also give a held-out term a vector zero-shot are new-token initialization from subtokens or definitions, a text encoder of the verbalized frame, and the frame in context. None of them has been run.

### 2.2 Wording audit against R9 (one seed, SmolLM2, T5 synthetic glossary)

| Claim text | What R9 shows | Problem | Fix |
|---|---|---|---|
| "lowers loss after new/OOD domain terms by 6–12%" | Relative to C0′: held-out −7.5% (135M) / −11.7% (360M); 3+-subtoken −6.4% / −7.1%; unseen −3.4% / −5.5%; rare −3.1% / −5.6% | The 6–12% range omits the unseen and rare strata | "3–12% depending on stratum; 7.5–11.7% after held-out terms" |
| "beating a capacity-matched free per-concept embedding table" | Held-out: C2 is worse than C0′ by +2–5%, because it has no rows for these terms. Seen-rare: C5 − C2 = −7.1% (360M) / −7.8% (135M) | Beating C2 on held-out terms is trivially true | Claim the C2 comparison on **seen-rare** and 3+-subtoken strata. For held-out terms, compare against zero-shot initializers (§2.5) |
| "genuinely new/out-of-distribution domain terms" | The T5 text is **generated from the same ontology** whose frames the channel receives | The channel hands the model the generator's latent variables. The lexical-overlap confound of Onoe et al. 2023 applies: the gain may be copying of filler words that follow the term | Report T4 (ChEBI) and T1 (PubMed) natural text before claiming "domain terms". Split the loss into tokens that verbalize a frame filler vs. all other tokens |
| "no effect on vocabulary the host already knows" | The WordNet engagement check: C5 = C0′ to four decimals | It is a null, not a demonstrated invariance | "The host neutralizes the channel on vocabulary it already models (|Δ| < 10⁻⁴ nats)", with the minimum detectable effect stated |
| "(and Qwen3 next)" | Pending; LoRA r = 64 instead of full fine-tuning | Different host mode | Report Qwen3 separately; do not pool |

### 2.3 Closest works

| # | Citation | Verified | What it does | Overlap / precise difference | Verdict |
|---|---|---|---|---|---|
| A1 | Abul Hasan, Jinge Wu, Quang Ngoc Nguyen, et al. (10 authors). *Infusing clinical knowledge into tokenisers for language models* (K-Tokeniser). arXiv:2406.14312, 2024. https://arxiv.org/abs/2406.14312. Journal version: *Computers in Biology and Medicine*, 2025, DOI 10.1016/j.compbiomed.2025.110747 (Crossref record only) | ✓✓ (arXiv) | Initializes token representations from UMLS semantic types of domain concepts and picks among them by sentence context; no further pretraining | **Overlap:** an ontology supplies vectors for domain terms in a pretrained host. **Differences:** changes the tokenizer; semantic types only, not relation–filler frames bound from a shared dictionary; encoder models and downstream F1, not decoder LM loss; no held-out-term vs. free-table test | narrows wording; baseline (type-only) |
| A2 | Zhengyan Zhang, Zhiyuan Zeng, Yankai Lin, et al. (11 authors). *Plug-and-Play Knowledge Injection for Pre-trained Language Models* (map-tuning). ACL 2023. https://arxiv.org/abs/2305.17691, https://aclanthology.org/2023.acl-long.594/ | ✓✓ | Trains a mapping from TransE entity embeddings into the input space of a frozen PLM, at entity mention spans | **Overlap:** KG-derived vectors injected at mentions of a pretrained LM. **Differences:** TransE rows are free per-entity parameters, so there is no zero-shot composition; encoders, downstream tasks; the backbone is frozen rather than jointly trained | narrows wording; baseline (mapped KG-embedding at the same site) |
| A3 | Yuval Reif, Guy Kaplan, Roy Schwartz. *Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic*. Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.1618/ (arXiv:2510.17001) | ✓✓ | Represents surface forms as base-form embedding plus shared transformation vectors in pretrained LLMs; extends coverage to OOV forms | **Overlap:** shared relation-like vectors compose representations of unseen words in a pretrained LLM. **Differences:** morphological offsets, not ontology relation–filler binding; vocabulary reshaping, not a semantic channel; no held-out-concept LM-loss test | narrows wording (no broad "compositional vectors extend pretrained LLMs to unseen words" claim) |
| A4 | Yasumasa Onoe, Michael J. Q. Zhang, Eunsol Choi, Greg Durrett. *Entity Cloze By Date: What LMs Know About Unseen Entities*. Findings of NAACL 2022. https://arxiv.org/abs/2205.02832. With Yasumasa Onoe, Michael Zhang, Shankar Padmanabhan, Greg Durrett, Eunsol Choi. *Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge*. ACL 2023. https://arxiv.org/abs/2305.01651, https://aclanthology.org/2023.acl-long.300/ | ✓✓ (2023) / ✓ (2022) | Measure LM perplexity / cloze on text about entities unseen in pretraining. Definitions in context help in every setting. Fine-tuning on definitions helps mainly when target and injected facts overlap lexically | **Overlap:** the same evaluation target. They establish the in-context-definition baseline and the lexical-overlap confound. **Difference:** no embedding channel, no ontology | requires new baselines and controls |
| A5 | Nandini Mundra, Aditya Nanda Kishore Khandavally, Raj Dabre, Ratish Puduppully, Anoop Kunchukuttan, Mitesh M. Khapra. *An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models*. CoNLL 2024. https://arxiv.org/abs/2407.05841, https://aclanthology.org/2024.conll-1.8/. Same family: J. Hewitt, *Initializing New Word Embeddings for Pretrained Language Models*, blog, 6 Dec 2021, https://www.cs.columbia.edu/~johnhew/vocab-expansion.html (✓✓; blog, not peer-reviewed). L. Gee, A. Zugarini, L. Rigutini, P. Torroni, *Fast Vocabulary Transfer for Language Model Compression*, EMNLP 2022 Industry, https://aclanthology.org/2022.emnlp-industry.41/ (✓✓). K. Dobler, D. Elliott, G. de Melo, *Token Distillation: Attention-aware Input Embeddings for New Tokens*, ICLR 2026 (arXiv comment), https://arxiv.org/abs/2505.20133 (✓✓) | ✓✓ | Initialize new-token embeddings by convex-hull/mean, subtoken mean (FVT), or distillation from the original tokenization | **Overlap:** these also give a never-trained term a vector zero-shot and are the default answer to "why not add the term as a token?". **Differences:** they change segmentation; no ontology composition | **requires a new baseline (the most direct competitor for held-out terms)** |
| A6 | Ivan Vulić, Goran Glavaš, Nikola Mrkšić, Anna Korhonen. *Post-Specialisation: Retrofitting Vectors of Words Unseen in Lexical Resources*. NAACL 2018. https://arxiv.org/abs/1805.03228, https://aclanthology.org/N18-1048/. With Faruqui et al., *Retrofitting Word Vectors to Semantic Lexicons*, NAACL 2015, https://aclanthology.org/N15-1184/ | ✓✓ / ✓ | Learns a function that carries lexical-resource specialization to unseen words | **Term clash.** "Retrofitting" already means lexicon-based post-processing of embeddings, including extension to unseen words; it is also used for LLM tokenization (Feher, Vulić & Minixhofer, arXiv:2411.18553). **Difference:** static vectors, no LM | narrows wording (define "retrofit", or say "post-hoc joint fine-tuning into a pretrained host") |

### 2.4 Further relevant works (✓ = verified by the search thread; (m) = metadata record only)

- **Vocabulary adaptation and initialization**
  - OFA (Liu, Lin, Wang & Schütze, Findings NAACL 2024, arXiv:2311.08849 ✓); HyperOFA (ACL SRW 2025, arXiv:2504.21018 ✓); AdaptiVocab (arXiv:2503.19693 ✓, venue unverified)
  - Gold Panning in Vocabulary (EMNLP 2024 ✓); Teaching Old Tokenizers New Words (Findings EACL 2026, arXiv:2512.03989 ✓); Learning Faster with Better Tokens (Balde et al., ACL 2026, 2026.acl-long.2088 ✓; domain OOV terms in Llama-3.1/Qwen2.5)
  - AVocaDo (EMNLP 2021 ✓); adaptive tokenization (Sachidananda et al., SustaiNLP 2021 ✓)
  - In-Place Tokenizer Expansion (arXiv:2607.15232 ✓, preprint); Defragmenting Language Models (arXiv:2604.16656 ✓)
  - Anonymous ACL ARR Aug 2026 submission, *Balancing Semantic Priors and Learnability in Novel Concept Token Initialization* (OpenReview tQoM5v7slF; API record only): semantically informed initialization of a new concept token starts better but learns more slowly than mean initialization. Already covered: Concept Tokens (arXiv:2601.04465), ZeTT, WECHSEL, FOCUS.
- **OOV / new-word embeddings**
  - LOVE (ACL 2022 ✓); HiCE (ACL 2019 ✓); GRM (ACL 2023 ✓; OOV embeddings by message passing over a word-relation graph, a graph-composed OOV precedent)
  - Neologism Learning (Hewitt, Tafjord, Geirhos & Kim, arXiv:2510.08506 ✓✓; trains only a new word's embedding in a frozen LLM)
  - Minnow (Wang, Jiang, Linzen & Lake, EMNLP 2025 ✓)
- **KG / graph vectors into LLM inputs**
  - KG-Adapter (Findings ACL 2024 ✓); GraphToken (arXiv:2402.05862 ✓); Graph Neural Prompting (AAAI 2024 ✓)
  - KoPA (arXiv:2310.06671 ✓); SSQR (arXiv:2501.18119 ✓); ReaLM (arXiv:2510.09711 ✓); KGT (arXiv:2602.22698 ✓)
  - PELT (Ye et al., ACL 2022 short ✓)
- **Text encoders producing inductive entity vectors** (baseline family for §2.5 A-B4)
  - BLP (Daza, Cochez & Groth, WWW 2021, arXiv:2010.03496 ✓); BioLORD (Findings EMNLP 2022 ✓); xRAG (NeurIPS 2024 ✓); KEPLER (covered)
- **LM-loss precedents with KG conditioning**
  - KGLM (Logan et al. 2019, arXiv:1906.07241 ✓); Relational Memory-Augmented LMs (Liu, Yogatama & Blunsom, TACL 2022 ✓); KELM verbalized-KG corpus (NAACL 2021 ✓)
- **Evaluation design with invented or new terms**
  - ALCUNA (Yin, Huang & Wan, EMNLP 2023 ✓✓; artificial entities built by altering KG attributes, the same construction as T5 and our invented words)
  - NEO-BENCH (ACL 2024 ✓); NewTerm (NeurIPS 2024 D&B ✓); FictionalQA (ICLR 2026 ✓); WinoDict (EACL 2023 (m))
- **Locality**
  - LoRA Learns Less and Forgets Less (Biderman et al., TMLR 2024 ✓); Sparse Memory Finetuning (arXiv:2510.15103 ✓)
  - Memory Decoder (NeurIPS 2025 ✓; a plug-in domain memory that lowers domain perplexity, an optional extra domain baseline)
- **Clinical niche**
  - KEEP (PMLR 287, 2025 ✓); MedTok (covered); UmlsBERT, SapBERT (covered)

### 2.5 Required baselines and controls (E9, dimension 1)

| ID | Baseline / control | Definition | Answers |
|---|---|---|---|
| A-B1 | **New-token baselines** | Add one token per glossary term and initialize it three ways: (i) mean of the term's original subtokens (FVT/Hewitt); (ii) encoded verbalized frame (host mean or last hidden state over the frame text, or Token Distillation); (iii) type-only (mean of the is-a parent's row; K-Tokeniser-like). Then run the same 50M-token training. Held-out terms get the same zero-shot initialization and no gradient | "Why not just add the term as a token?" The direct zero-shot competitor for held-out terms |
| A-B2 | **Same-site vector from the frame text** | The (i)/(ii) vectors, gated and added at the **same last-subtoken site**, tokenizer unchanged, parameter-matched | Separates "HRR composition from a shared dictionary" from "the frame's information reaches the host at that position" |
| A-B3 | **In-context verbalized frame (upper bound / competitor)** | Prepend the verbalized frame or definition at the first mention of each linked term; score only continuation tokens (ECBD/Onoe protocol). Also channel + in-context. Report extra tokens and latency | How much of the achievable gain the channel captures with zero context tokens |
| A-B4 | **Text-encoder channel** | KEPLER/BLP/BioLORD/xRAG-style: a frozen sentence encoder (or host mean-pool) over the verbalized frame plus a trained projector, at the same site and gate, parameter-matched | The strongest zero-shot competitor for held-out terms; composition must match it at equal parameters |
| A-B5 | **Lexical-overlap split** | Score separately the continuation tokens that literally verbalize one of the term's frame fillers and all other tokens. Confirm on T4/T1 natural text | The Onoe 2023 confound; template copying vs. use of meaning |
| A-B6 | **Specificity inside E9** | Shuffled frames among held-out terms of the same type; equal-degree random frames; type-only frame; gate open with a zero vector; random fixed per-term vectors (C1); **operator ablation** (`hrr` vs `random_fixed`, `untyped`, `translation`) | Frame content vs. a generic "domain term here" signal, and whether binding matters in the retrofit regime (OP.1) |
| A-B7 | **Engram-style hashed memory (C1h-E)** at the same site, capacity-matched | As in the 2026-10-02 note | "Extra lookup capacity at the term position". Like C2, it should not help held-out terms, which strengthens the comparison |
| A-B8 | Statistics | ≥ 3 seeds; term-level cluster bootstrap; strata by term frequency and by whether a held-out term's **fillers** were seen in training | Seed variance, the compositional-generalization limit |

### 2.6 Recommended wording (A)

> Jointly fine-tuning an ontology-composed channel into pretrained small decoders (SmolLM2-135M/360M), which adds the composed vector at the last subtoken of linked multi-token terms without changing the tokenizer, lowers next-token loss after invented, contamination-free glossary terms by 3–12% relative to continued training without the channel. The largest reduction (7.5–11.7%) is after held-out terms whose vectors are composed zero-shot from their ontology frames. On seen rare terms it beats a capacity-matched free per-concept table by 7–8%. The host neutralizes the channel on vocabulary it already models, and unlinked text is unchanged.

Add "…and new-token initialization from subtoken means or encoded definitions, and approaches in-context frames" only after A-B1–A-B4. Say "synthetic glossary" until T4/T1 are in, and avoid an unqualified "retrofit".

---

## 3. Claim B — the channel's advantage grows under INT4 PTQ (E9, dimension 2)

### 3.1 Verdict: novel only as a narrow, control-backed measurement; the proposed headline is not supported

No paper was found that shows a trained, ontology-composed input channel *increasing* a quantized LM's advantage on rare, new or held-out multi-subtoken terms. But almost every supporting observation is already published:
- compression hurts the long tail and low-resource items;
- INT8 is near-lossless;
- domain-fine-tuned models are more INT4-fragile;
- training dynamics drive PTQ fragility.

A published mechanism also predicts the observed pattern for **any** separately parameterized module, structured or not. Full fine-tuning deltas are often too small to survive 4-bit PTQ, so INT4 pulls the fine-tuned host back toward its pre-fine-tuning state (Zhang et al., ICLR 2025; Abitante et al., IJCNN 2026). Knowledge stored in a separate module (here the channel) survives. This mechanism predicts the R9 pattern. INT4 costs the fine-tuned hosts +38% loss on the domain text, where fine-tuning added the knowledge, while every model, including the original host, loses the same ≈ +7.7% on general text.

R9's own numbers point the same way:
- **The 2.80× "all tokens" retention at 135M comes from unlinked domain text, where the channel is inactive.** The C5 − C0′ gap there goes from −0.000 nats/token (bf16) to −0.048 (INT4). The C5-trained host weights are more quantization-robust in general. This is not the channel recovering rare words.
- At 360M the held-out-term advantage **halves** (0.47×).
- One seed; one data-free quantizer (torchao `Int4WeightOnlyConfig`, group size 128, round-to-nearest).
- The tied input embedding and output head are never quantized (`src/vsa_embed/evaluation/quantization.py`, `quantize_weight_only`), so rare-token embedding rows are never stressed.
- The ratios divide two small differences and inflate where the bf16 advantage is small.

"A structured semantic channel makes small quantized models more robust on rare vocabulary" should **not** go in the abstract until controls B-B1–B-B4 below pass.

### 3.2 Closest works

| # | Citation | Verified | What it does | Overlap / precise difference | Verdict |
|---|---|---|---|---|---|
| B1 | João Vitor Boer Abitante, Joana Meneguzzo Pasquali, Luan Fonseca Garcia, Ewerton de Oliveira, Thomas da Silva Paula, Rodrigo C. Barros, Lucas S. Kupssinskü. *Quantization-Robust LLM Unlearning via Low-Rank Adaptation*. IJCNN 2026 (arXiv comment). https://arxiv.org/abs/2602.13151 | ✓✓ | Full-parameter fine-tuning "often induces parameter changes" too small to survive 4-bit PTQ; moving the update into LoRA adapters preserves it | **Same mechanism** as the likely explanation of E9: the no-channel host keeps all new-term knowledge in small full-fine-tuning deltas that INT4 erases. **Difference:** unlearning, not term acquisition; no structured channel | requires a new baseline (bit-matched separate module); narrows wording |
| B2 | Zhiwei Zhang, Fali Wang, Xiaomin Li, Zongyu Wu, Xianfeng Tang, Hui Liu, Qi He, Wenpeng Yin, Suhang Wang. *Catastrophic Failure of LLM Unlearning via Quantization*. ICLR 2025. https://arxiv.org/abs/2410.16454 | ✓✓ | 4-bit PTQ restores "forgotten" knowledge (21% retained in full precision → 83% after 4-bit): quantization snaps fine-tuned weights back toward the original | Explains why INT4 costs the fine-tuned hosts +38% on the domain text while general-text damage stays ≈ +7.7% for every model. **Difference:** unlearning; no external module | narrows wording (the fragility is expected, not a finding) |
| B3 | Leonard Twagirayezu, Prasenjit Mitra. *The Effect of Quantization on Clinical Benchmarks: Accuracy and Safety Across Model Families*. arXiv:2609.22216, Sep 2026 (venue unverified). https://arxiv.org/abs/2609.22216 | ✓✓ | GPTQ INT8 "universally safe"; INT4 substantial; clinically fine-tuned BioMistral-7B loses 19.7% on MedMCQA, "clinical fine-tuning does not confer compression robustness" | Pre-empts two sub-observations: INT8 is nearly free, and domain-fine-tuned models are more INT4-fragile. **Difference:** 7–8B models, QA accuracy, no channel | pre-empts those sub-claims only |
| B4 | Kelly Marchisio, Saurabh Dash, Hongyu Chen, Dennis Aumiller, Ahmet Üstün, Sara Hooker, Sebastian Ruder. *How Does Quantization Affect Multilingual LLMs?* Findings of EMNLP 2024. https://arxiv.org/abs/2407.03211. With Sara Hooker, Aaron Courville, Gregory Clark, Yann Dauphin, Andrea Frome. *What Do Compressed Deep Neural Networks Forget?* arXiv:1911.05248 (venue unverified) | ✓✓ | Compression disproportionately hurts the under-represented long tail; quantization hurts non-Latin and low-resource languages most; automatic metrics underestimate harm | The premise "quantization hurts rare items more" is established. Our contribution can only be "an external structured channel offsets part of it" | narrows wording |
| B5 | Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, et al. (9 authors). *Scaling Laws for Precision*. arXiv:2411.04330 (ICLR 2025 oral per OpenReview record). Albert Catalan-Tatjer, Niccolò Ajroldi, Jonas Geiping. *Training Dynamics Impact Post-Training Quantization Robustness*. arXiv:2510.06213 (ICLR 2026 poster per OpenReview record). Xu Ouyang et al. *Low-Bit Quantization Favors Undertrained LLMs*. ACL 2025, arXiv:2411.17691 | ✓✓ (Kumar, Catalan-Tatjer) / ✓ | PTQ degradation grows with training level and data, and depends on learning-rate schedule and other hyperparameters | The two arms differ in training dynamics (different gradients, lower final loss), so differential PTQ fragility can come from that rather than from channel structure | requires a new control (matched schedules, delta-norm diagnostics) |
| B6 | Sammy Nwaiwu, N. Jongsawat, A. Tungkasthan. *Compressed Causal Reasoning: Quantization and GraphRAG Effects on Interventional and Counterfactual Accuracy*. arXiv:2512.13725 (venue unverified). With Hoang et al., *Do Compressed LLMs Forget Knowledge?*, arXiv:2310.00867, and LLM-KICK (Jaiswal et al., ICLR 2024, arXiv:2310.01382) | ✓ | GraphRAG with a ground-truth graph partly offsets NF4 degradation on Llama-3-8B. Input-side augmentation (prompting) recovers knowledge displaced by compression | The nearest prior statement that structured external knowledge offsets quantization loss. **Difference:** knowledge is retrieved into context, not a trained embedding-side channel; not rare vocabulary | narrows wording (no "first to show structured knowledge mitigates quantization loss") |

### 3.3 Further relevant works

- **Long tail, low-resource, disparity**
  - Hooker et al., *Characterising Bias in Compressed Models* (arXiv:2010.03058 ✓); Ahia, Kreutzer & Hooker, *The Low-Resource Double Bind* (Findings EMNLP 2021 ✓)
  - Ogueji et al. (EMNLP 2022 ✓; sparsification can *help* low-resource languages, so the direction is not universal)
  - Marie & Fujita 2025 (arXiv:2508.20893 ✓; quantizer and calibration language matter); Soualhi 2026 (arXiv:2608.09941 ✓)
  - TARQ (arXiv:2605.27808 ✓; data-aware PTQ under-weights rare words, and tail-aware calibration fixes it)
  - Bellam & Kim 2025 (arXiv:2509.07222 ✓)
- **Knowledge recall under compression**
  - Namburi et al., *The Cost of Compression* (Findings EMNLP 2023 per arXiv comment ✓)
  - *Through a Compressed Lens* (TrustNLP@ACL 2026, arXiv:2505.13963 ✓; recall drops more in smaller models)
  - Li et al., *Evaluating Quantized LLMs* (ICML 2024 ✓); Dutta et al., *Accuracy is Not All You Need* (NeurIPS 2024 ✓; report flip rates)
- **Quantizers and outliers**
  - GPTQ (Frantar, Ashkboos, Hoefler & Alistarh, ICLR 2023, arXiv:2210.17323 ✓✓)
  - AWQ (Lin et al., MLSys 2024, arXiv:2306.00978 ✓✓)
  - ZeroQuant-V2 (arXiv:2303.08302 ✓; RTN vs GPTQ from 125M up); SpQR (arXiv:2306.03078 ✓); Super Weight (arXiv:2411.07191 ✓)
  - LLM.int8() (NeurIPS 2022 ✓); Puccetti et al. (Findings EMNLP 2022 ✓; outlier dimensions track token frequency)
  - Williams & Aletras (ACL 2024 ✓; calibration data)
- **Quantization-aware and compensation methods**
  - QLoRA (arXiv:2305.14314 ✓); LoftQ (arXiv:2310.08659 ✓); L4Q (ACL 2025 ✓); ParetoQ (NeurIPS 2025 ✓); EoRA (arXiv:2410.21271 ✓); RILQ (AAAI 2025 ✓)
- **Toolkit:** TorchAO (Or et al., CODEML@ICML 2025, arXiv:2507.16099 ✓); cite it as the method.
- **Engram follow-ups** (TF-Engram arXiv:2607.07388, Engram Adapter arXiv:2608.29327 ✓): no quantization-robustness study. The Engram Adapter could serve as the separate-module baseline.

### 3.4 Required controls (E9, dimension 2)

| ID | Control | Answers |
|---|---|---|
| **B-B1** | **Bit-matched unstructured separate module.** (a) C0′ plus an FP16 LoRA, or an FP16 free per-concept table, with the **same byte budget** as the channel, trained on the same schedule. (b) The existing C2 at the **same precision as the channel** in each variant (A: both FP16; B: both INT4 with the same group size). (c) The channel's bytes spent on host precision instead: AWQ/SpQR-style FP16 salient columns, or the most sensitive layer kept at INT8 | Structure, or merely "some of the update lives outside the INT4-quantized Linear weights" (B1, B2)? Only growth versus (a) and (b) supports a *structured* claim |
| **B-B2** | **Channel off at INT4.** Evaluate the INT4 C5 host with the gate forced to 0, against INT4 C0′ | Separates "the C5-trained host is more quantization-robust" (what the unlinked-text result suggests) from "the channel compensates" |
| **B-B3** | **Quantizer sweep.** GPTQ (g128, act-order), AWQ, bitsandbytes NF4 (or HQQ), at group sizes 32/64/128. Calibrate on generic text and on in-domain text that contains the rare terms (TARQ; Williams & Aletras) | Whether "advantage grows" is an artifact of data-free RTN |
| **B-B4** | **Revert diagnostics and matched dynamics.** Per arm: ‖W_ft − W_0‖; the fraction of delta entries below half the INT4 step of their group; INT4-host loss vs. FP original host on new terms. Use the same LR schedule, final LR and steps, and report pre-quantization loss. Add a LoRA pair (with and without channel) | The B1/B2/B5 mechanisms |
| B-B5 | QAT / QLoRA arm (torchao int4 QAT, or NF4 base + LoRA), with and without the channel | Does the advantage survive when the host is trained for quantization? If it vanishes, the effect is PTQ fragility, not channel robustness |
| B-B6 | **Reporting.** Absolute nats/token in all four cells (FP/INT4 × channel/no channel) per stratum, with n. Primary statistic: difference-in-differences DiD = (L_C0′,INT4 − L_C5,INT4) − (L_C0′,FP − L_C5,FP), with paired bootstrap CIs. Retention ratios only next to DiD | Ratio inflation |
| B-B7 | ≥ 3 fine-tuning seeds per arm (plus calibration seeds for GPTQ/AWQ); resolve 360M; a third host (Qwen3-0.6B); a variant that **also quantizes the tied embedding** (as GGUF Q4_K deployments do) | Generality; the deployment-realistic setting |

### 3.5 Recommended wording (B)

> Under INT4 weight-only round-to-nearest PTQ (embeddings and output head unquantized), the channel's loss advantage on rare, held-out and multi-subtoken domain terms in a fully fine-tuned SmolLM2-135M is preserved and, as a difference-in-differences, increases. This holds whether or not the channel itself is quantized. At 360M the result is mixed. Consistent with prior work, INT8 is nearly lossless and fine-tuned hosts are far more INT4-fragile than the original host.

Upgrade to "…more robust than a bit-matched unstructured module, under GPTQ and AWQ" only after B-B1–B-B3 pass with ≥ 3 seeds. Until then, keep claim B out of the abstract or give it one hedged clause.

---

## 4. Claim C — zero-shot vocabulary expansion and edits by ontology editing (E9, dimension 3)

### 4.1 Verdict: the general framing is pre-empted; the specific mechanism is novel with narrow wording

**Pre-empted.** "Change a pretrained LM's knowledge by editing a symbolic store it reads, with no gradient step" is established:
- **Facts as Experts** (2020/NAACL 2021) explicitly adds new facts and overwrites existing ones by editing its symbolic fact memory, without retraining;
- **KBLaM** ("dynamic updates without model fine-tuning or retraining"), **LMLM** ("explicit, editable, and verifiable knowledge bases"), and Larimar make the same claim;
- **REMEDI** inserts a fact by adding a vector to an entity's representation in a frozen LM;
- GRACE, SERAC and IKE edit outside the weights.

So "knowledge editing via the ontology rather than weights" cannot be claimed as a new paradigm.

**Not found anywhere**, as a combination:
- the edit is a single relation–filler edge change in a frame that is HRR-bound from a shared atom/relation dictionary;
- the composed vector is added at the input embedding at the last subtoken of an existing multi-token span: no new token, no retrieval or attention memory, no trained editor;
- new words are written as **new combinations of existing atoms**, tested against an equal-degree random-frame control.

**The evidence is modest.**
- New-word property selection: 0.249 vs 0.188, with chance 0.20. The C0′ baseline is below chance, so the probe sits near its floor. Entailment is n.s.
- Edits move 0.40 → 0.455 and work only for terms seen in training (held-out +0.04, n.s.). ROME/MEMIT report near-ceiling CounterFact efficacy on GPT-2 XL / GPT-J; their efficacy on SmolLM2 is unmeasured here. Reviewers will read our result as a proof of concept, not an editing method.

### 4.2 Closest works

| # | Citation | Verified | What it does | Overlap / precise difference | Verdict |
|---|---|---|---|---|---|
| C1 | Pat Verga, Haitian Sun, Livio Baldini Soares, William W. Cohen. *Facts as Experts: Adaptable and Interpretable Neural Memory over Symbolic Knowledge*. arXiv:2007.00849 (2020). Published as *Adaptable and Interpretable Neural Memory Over Symbolic Knowledge*, NAACL 2021, pp. 3678–3691. https://aclanthology.org/2021.naacl-main.288/ | ✓✓ | An LM with a symbolic fact memory read by key–value attention mid-network; "can be updated without re-training by manipulating its symbolic representations… add new facts and overwrite existing ones" | **Pre-empts the generic framing.** **Differences:** FaE stores facts as separate memory rows read by attention, trained for QA. Ours swaps one bound edge inside a superposed concept vector added to the input of a pretrained decoder. No invented-word setting | narrows wording (cite as the precedent for non-weight editing) |
| C2 | Evan Hernandez, Belinda Z. Li, Jacob Andreas. *Inspecting and Editing Knowledge Representations in Language Models* (REMEDI). COLM 2024 (arXiv comment). https://arxiv.org/abs/2304.00740 | ✓✓ | Learns to map a natural-language fact to an encoding that is added to the entity's hidden representation of a frozen LM, changing generation | **The closest mechanism:** a vector added at the entity's position, no weight update, specificity measured. **Differences:** a trained editor reads text and adds at an intermediate layer. Ours composes symbolically with no editor, at the input embedding, into a host trained to read it. No zero-shot invented words | narrows wording; **requires a baseline** (a REMEDI-style editor on the same items) |
| C3 | Xi Wang, Taketomo Isazawa, Liana Mikaelyan, James Hensman. *KBLaM: Knowledge Base augmented Language Model*. arXiv:2410.10450 (ICLR 2025 per OpenReview record). With Linxi Zhao, Sofian Zalouk, Christian K. Belardi, et al. (10 authors). *Pre-training Limited Memory Language Models with Internal and External Knowledge* (LMLM). arXiv:2505.15962 (ICLR 2026 poster per OpenReview record) | ✓✓ (arXiv) | KB triples become key–value knowledge tokens read by rectangular attention ("dynamic updates without fine-tuning or retraining"). LMLM externalizes facts to an editable database during pretraining of small models | Non-weight, editable symbolic knowledge in LMs, including small ones. **Differences:** lookup or attention over a store, vs. a composed input vector with no lookup step | narrows wording |
| C4 | Zhengyan Zhang et al. *Plug-and-Play Knowledge Injection* (map-tuning), ACL 2023 (see A2). Deming Ye, Yankai Lin, Peng Li, Maosong Sun, Zhiyuan Liu. *A Simple but Effective Pluggable Entity Lookup Table for Pre-trained Language Models* (PELT). ACL 2022 (short). https://aclanthology.org/2022.acl-short.57/ | ✓✓ / ✓ | Input-side entity or KG vectors plugged into a frozen PLM; the knowledge source can be swapped or extended without touching LM weights | **Overlap:** input-side structured vectors that can be swapped. **Differences:** vectors come from learned KGE + mapping or from corpus aggregation, not compositional binding; no single-edge edits or random-frame controls | narrows wording; baseline (inductive KGE vector + linear map) |
| C5 | Onoe et al., ACL 2023 (A4), and Shankar Padmanabhan, Yasumasa Onoe, Michael J. Q. Zhang, Greg Durrett, Eunsol Choi. *Propagating Knowledge Updates to LMs Through Distillation*. NeurIPS 2023 (arXiv comment). https://arxiv.org/abs/2306.09306 | ✓✓ / ✓ | For new entities, in-context definitions beat gradient injection. Context distillation beats fine-tuning and ROME/MEND-style editors on propagation | Same problem: new-entity and edit propagation. **Difference:** text definitions in context or via weights vs. a symbolic frame through an embedding channel | **requires baselines** (in-context frame upper bound; optionally distillation) |
| C6 | Xunjian Yin, Baizhou Huang, Xiaojun Wan. *ALCUNA: Large Language Models Meet New Knowledge*. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.87/ | ✓✓ | KnowGen creates artificial entities by altering the attributes and relations of existing ones | The same construction as our invented words. **Difference:** a benchmark with knowledge given in context, no mechanism | requires an evaluation alignment (cite for the construction; report an ALCUNA-style split) |

### 4.3 Further relevant works

- **Weight and side-memory editors (baselines)**
  - ROME (Meng, Bau, Andonian & Belinkov, NeurIPS 2022, arXiv:2202.05262 ✓)
  - MEMIT (Meng, Sen Sharma, Andonian, Belinkov & Bau, arXiv:2210.07229 ✓✓; ICLR 2023 per OpenReview record)
  - AlphaEdit (ICLR 2025, arXiv:2410.02355 ✓); MEND, PMET, WISE ✓
  - GRACE (NeurIPS 2023 ✓); SERAC (ICML 2022 ✓); Larimar (ICML 2024 ✓)
  - IKE (Zheng et al., EMNLP 2023, https://aclanthology.org/2023.emnlp-main.296/ ✓; arXiv:2305.12740 ✓✓)
- **KG-based editing**
  - GLAME (EMNLP 2024 ✓; KG-guided but still edits weights); KEDKG (AAAI 2025 ✓); OneEdit (LLM+KG@VLDB 2024 ✓)
  - Knowledge Graph Tuning (arXiv:2405.19686 ✓; venue unverified)
  - Niu 2026, *Edit Propagation Depends on an Addressable Memory* (arXiv:2606.20737 ✓). Edits propagate only when computation rereads an addressable memory, which is a useful frame for "edits work only for seen terms".
- **Evaluation**
  - RippleEdits (TACL 2024 ✓); MQuAKE (EMNLP 2023 ✓); KnowEdit/EasyEdit (arXiv:2401.01286 ✓)
  - ConceptEdit (Findings EMNLP 2024 ✓); COMPS (EACL 2023 ✓; property-inheritance minimal pairs for novel concepts, the natural format for property selection)
  - Duan et al. (NAACL 2025 ✓; same-subject interference, so intra-entity locality must be measured)
- **Older precedent**
  - Onoe & Durrett, interpretable typed entity vectors that can be modified post hoc by rules (Findings EMNLP 2020 ✓)
  - Raman et al. (arXiv:2010.12872 ✓): heavily perturbed KGs leave KG-augmented models' performance intact. This is a faithfulness warning, so use a frame-perturbation control.
- **New-word learning**
  - CoLLEGe (covered); Minnow (EMNLP 2025 ✓); Bahdanau et al. 2017 and Lampinen & McClelland 2017 (✓, venues unverified); WinoDict (m)
- **VSA/HDC knowledge editing:** no LM-editing work was found (searched on OpenReview and arXiv).

### 4.4 Required baselines and controls (E9, dimension 3)

| ID | Baseline / control | Answers |
|---|---|---|
| **C-B1** | **Weight editors on the identical edit items and host.** ROME, MEMIT, AlphaEdit (EasyEdit), plus FT-L / LoRA-FT, on SmolLM2 with and without the channel. Report CounterFact metrics (efficacy, paraphrase, neighborhood, fluency) and zsRE metrics. Also "ontology edit + ROME" | Whether ontology edits are competitive, complementary, or a proof of concept |
| **C-B2** | **In-context upper bound.** Verbalized frame or verbalized edited fact prepended; IKE with demonstrations. Run for both new words and edits; report the token cost | How much of the achievable gain the channel captures with zero context tokens (Onoe 2023 shows this baseline is strong) |
| **C-B3** | **New-word embedding baselines without gradient:** (a) mean-of-vocabulary new token (Hewitt); (b) mean of the verbalized frame's subword embeddings; (c) definition encoding (host final state over the verbalized frame, linearly mapped, at the same site; Bahdanau-style); (d) a CoLLEGe-style generator trained on the training terms and frames; (e) gradient upper bound: optimize only the new vector on frame-verbalized sentences for a fixed small budget (Neologism / Concept-Tokens style) | Whether symbolic composition beats cheap text-derived vectors |
| **C-B4** | **Composition controls.** Keep "own > random equal-degree > none ≈ mean row". Add: relation-shuffled frame (same fillers, permuted relations); filler-only additive frame; and a **frame transplant** (give seen term A the frame of seen term B: does the model then attribute B's properties to A?) | Whether binding matters, and whether the channel carries meaning or acts as an identifier. This is critical given the held-out-term failure |
| **C-B5** | **Edit validity.** Channel-off audit (gate 0 on edited seen terms); edit-to-random-filler null of the same magnitude; revert test; dose–response (1–3 edges) | Calibrates the +0.69 log-odds shift |
| C-B6 | **Locality and portability.** Neighbourhood specificity: unchanged vectors by construction, so this only measures spillover through context. **Intra-entity locality:** the term's other relations, since one edit changes the whole superposed vector. **Portability:** RippleEdits / MQuAKE 2-hop queries through the edited filler; alias subjects | Standard editing metrics |
| C-B7 | **Items and statistics.** ALCUNA/COMPS-style items; frequency- and length-matched options in permuted order; check that invented-word pieces do not leak the answer. ≥ 3 joint-training seeds; item × seed mixed-effects logistic regression; paired tests; Holm. **Power:** detecting Δ ≈ 0.06 at 0.19 base needs about 700 items per arm unpaired | Reviewers' statistical bar |

### 4.5 Recommended wording (C)

> After joint training, the model's behavior on a concept can be changed without any weight update by editing the frame from which its bound concept vector is composed. A frame written for an invented word improves property selection over equal-degree random-frame and no-frame controls (0.25 vs 0.19 at 360M; chance 0.20). Replacing one relation–filler edge of a term seen in training shifts predictions toward the new fact (+0.69 log-odds) while preserving neighbourhood specificity. Edits do not transfer to terms unseen in training.

Position this against Facts-as-Experts, KBLaM/LMLM, REMEDI and map-tuning as prior non-weight editing, and against in-context and ROME/MEMIT baselines. Do not say "knowledge editing via the ontology rather than weights" as a new paradigm, and do not say "first".

---

## 5. Claim D — self-learned semantics (E10)

### 5.1 Verdict: novel only as a protocol, with narrow, synthetic-scoped wording

**Not found:** no work combines, in one loop:
- (a) a learnable vector ontology inside a neural model, pulled toward a curated prior;
- (b) new relation types found in empty operator slots;
- (c) **label-free identification** of those relations by testing structural hypotheses (symmetric / inverse / transitive / functional / composition) on **held-out pairs against random-tail controls** with a bootstrap lower bound;
- (d) offline structural revision ("dreaming") accepted only by a held-out improvement over an equally long control refit.

**Already published, part by part:**

| Part | Prior work |
|---|---|
| (a) | L2-SP: L2 toward the pretrained starting point beats both free fine-tuning and freezing. Functional Retrofitting: relation functions learned together with entity embeddings, penalized toward the distributional prior |
| (b) | Open relation discovery; statistical predicate invention; NELL's OntExt; TransG/CTransR; the IRM |
| (c) | IterE: induces OWL property axioms (symmetric, inverse, transitive, …) from learned relation embeddings and injects the closure triples back into training. Statistical mining of property axioms. Possibilistic axiom testing |
| (d) | Wake–sleep; DreamCoder's abstraction sleep; 2026 "sleep" papers for LLMs; reduced-error pruning |
| Source variety vs. self-confirmation (E10.7) | Restates co-training's independent-views requirement and the pseudo-labelling confirmation-bias literature |

**The evidence bounds the claim.** Effects are strong only in a planted world built from the learner's own HRR family. On WordNet with frozen GPT-2 anchors, clauses (i)–(iv) mostly fail. There the **frozen** ontology has the best zero-shot fit, self-acceptance ≈ random, and new-word frame F1 is 0.007 vs 0.267 for nearest neighbour. E10.4 (building from an empty seed) is negative, and the joint-LM test (E10.2) is pending. No positive E10 claim belongs in the abstract before E10.2.

### 5.2 Closest works

| # | Citation | Verified | What it does | Overlap / precise difference | Verdict |
|---|---|---|---|---|---|
| D1 | Wen Zhang, Bibek Paudel, Liang Wang, Jiaoyan Chen, Hai Zhu, Wei Zhang, Abraham Bernstein, Huajun Chen. *Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning* (IterE). WWW 2019 (arXiv comment). https://arxiv.org/abs/1903.08948 | ✓✓ | Alternates KGE training with induction of 7 OWL2 property axiom types (reflexive, symmetric, transitive, equivalent, sub-property, inverse, chain). Axioms are scored from relation embeddings; inferred triples are injected back into training | **Overlap:** essentially our "identify a relation's logical property from the learned operator, then add closure edges" (riddle → crystallize with rules). **Differences:** named, given relations only (no blank-slot discovery); acceptance by an embedding-score threshold, not held-out prediction against random-tail controls; no prior pull, no LM | narrows wording; **requires a baseline** (IterE-style operator score, e.g. for HRR: symmetric iff the operator ≈ its involution) |
| D2 | Charles Kemp, Joshua B. Tenenbaum, Thomas L. Griffiths, Takeshi Yamada, Naonori Ueda. *Learning Systems of Concepts with an Infinite Relational Model*. AAAI 2006. https://cdn.aaai.org/AAAI/2006/AAAI06-061.pdf. With Kemp & Tenenbaum, *The discovery of structural form*, PNAS 2008 (PubMed 18669663) | ✓ | Nonparametric discovery of kinds and of which relations hold between them; selection among structural hypotheses by Bayesian model selection | **Overlap:** label-free discovery of relational systems and of structural forms. **Differences:** marginal likelihood on the same data, not held-out tests; hypotheses about entity organization, not relation properties; not inside a neural model | narrows wording; cite |
| D3 | Thahir P. Mohamed, Estevam R. Hruschka Jr., Tom M. Mitchell. *Discovering Relations between Noun Categories* (OntExt, NELL). EMNLP 2011. https://aclanthology.org/D11-1134/ | ✓ | Extends a seed ontology with new relations by co-clustering text contexts; a classifier judges validity | **Overlap:** growing a substantial seed ontology with new relation types; NELL's coupling is the classic guard against semantic drift. **Differences:** text-pattern clusters and a supervised validity check; no in-model operators, no property-based identification | narrows wording |
| D4 | Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Luc Cary, Lucas Morales, Luke Hewitt, Armando Solar-Lezama, Joshua B. Tenenbaum. *DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning*. arXiv:2006.08381 (2020); PLDI 2021 version as *DreamCoder: bootstrapping inductive program synthesis with wake-sleep library learning* (venue/title not opened here) | ✓✓ (arXiv) | Wake–sleep library learning: abstraction sleep grows and refactors the library under an MDL objective; dreaming trains on fantasies | **Overlap:** offline phases that restructure a learned symbolic library; also the term "dreaming". **Differences:** revisions accepted by compression, not by held-out improvement over a control refit; no relation split, merge or reopen | narrows wording (terminology); baseline (MDL acceptance) |
| D5 | Xuhong Li, Yves Grandvalet, Franck Davoine. *Explicit Inductive Bias for Transfer Learning with Convolutional Networks* (L2-SP). ICML 2018. https://arxiv.org/abs/1802.01483. Ben Lengerich, Andrew Maas, Christopher Potts. *Retrofitting Distributional Embeddings to Knowledge Graphs with Functional Relations*. COLING 2018. https://aclanthology.org/C18-1205/ | ✓ | L2 toward the starting point beats free and frozen. Relation functions learned with entity embeddings, penalized toward the prior | **Pre-empts the generic finding "L2-to-prior beats frozen and free".** **Difference:** ours applies it to a composed ontology mapping (edge masses, candidate frames, binding operators), evaluated by zero-shot composition | pre-empts the generic finding; narrows wording |
| D6 | Diego Marcheggiani, Ivan Titov, TACL 2016 (Q16-1017 ✓). Sha Li, Heng Ji, Jiawei Han, *Open Relation and Event Type Discovery with Type Abstraction*, EMNLP 2022 (✓). Hongyao Tu et al., *LLM-OREF*, EMNLP 2025 (https://aclanthology.org/2025.emnlp-main.459/ ✓) | ✓ | Label-free relation discovery from text, with LM-generated names | **Overlap:** discovering and interpreting new relation types. **Difference:** clustering mentions, naming rather than tested properties, no binding operators | narrows wording; baseline (clustering) |

### 5.3 Further relevant works

- **Axiom and property mining**
  - Völker & Niepert, *Statistical Schema Induction*, ESWC 2011 (m); Fleischhacker, Völker & Stuckenschmidt, *Mining RDF Data for Property Axioms*, 2012 (m)
  - Tettamanzi et al., possibilistic axiom testing, 2014/2017 (m). It accepts axioms by confirmations vs. counterexamples, the same logic as our should-hold / should-not-hold tests.
  - Potoniec 2020, *Learning OWL 2 Property Characteristics as an Explanation for an RNN* (m; content unchecked, possibly close to (c))
- **Rule mining**
  - AMIE / AMIE 3, AnyBURL, Neural LP, DRUM, RNNLogic, NCRL (m)
  - RotatE (m), which models symmetric, antisymmetric, inverse and composition patterns
- **Predicate invention and relation sub-types**
  - Kok & Domingos, *Statistical Predicate Invention*, ICML 2007 (m); TransG (ACL 2016 ✓); CTransR (AAAI 2015 ✓)
- **Joint learning of embeddings and logic, structure learning**
  - pLogicNet (NeurIPS 2019 ✓); KALE / RUGE (m); Franceschi et al., *Learning Discrete Structures for GNNs* (ICML 2019 ✓)
- **Ontology revision and schema induction with LMs**
  - Ji et al., *Ontology Revision based on PLMs* (arXiv:2310.18378 ✓); Evo-DKD (arXiv:2507.21438 ✓)
  - AutoSchemaKG (ACL 2026 ✓); EDC (EMNLP 2024 ✓)
- **Hypothesis generation and testing**
  - Hypothesis Search (ICLR 2024 ✓; verifies on *observed* examples, the contrast with our held-out testing); Qiu et al. (ICLR 2024 ✓); HypoGeniC (✓)
- **Self-confirmation and validation reuse**
  - Arazo et al., confirmation bias in pseudo-labelling (arXiv:1908.02983 ✓); Blum & Mitchell, co-training (m)
  - Dwork et al., *The reusable holdout*, Science 2015 (✓, PubMed). Repeated hypothesis tests on one validation split need adaptive-reuse control.
  - Quinlan, reduced-error pruning, 1987 (m)
- **Sleep and dreaming**
  - Hinton et al. 1995 (✓); Fachechi, Agliari & Barra, *Dreaming neural networks*, 2019 (✓); Tadros et al., Nat. Commun. 2022 (✓)
  - Sorrenti et al., *Wake-Sleep Consolidated Learning* (arXiv:2401.08623 ✓)
  - Behrouz, Hashemi, Javanmard & Mirrokni, *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories* (arXiv:2606.03979 ✓✓; dreaming = RL-generated synthetic data, no held-out acceptance); Auto-Dreamer (arXiv:2605.20616 ✓)
  - McClelland, McNaughton & O'Reilly 1995 and Kumaran, Hassabis & McClelland 2016 (✓)
- **VSA**
  - Resonator networks (Frady et al. 2020 ✓), a baseline for frame inference; NVSA (✓)
  - ROLE (Soulos et al., BlackboxNLP 2020 ✓): label-free discovery of TPR role slots
- **Word learning** (only if the human comparison is kept)
  - Carey & Bartlett 1978; Clark 1973; Rescorla 1980; Xu & Tenenbaum 2007; Fazly, Alishahi & Stevenson 2010
  - Horst & Samuelson 2008; McMurray, Horst & Samuelson 2012; Davis & Gaskell 2009
  - Ferreira Pinto & Xu 2021; Gandhi & Lake (NeurIPS 2020 ✓); Lake & Murphy 2023; Chang & Bergen (TACL ✓)
  - Frank 2026 (arXiv:2608.17120 ✓)

### 5.4 The human word-learning comparison

**Not defensible as written.** Expect pushback on five points:
1. **Overextension.** Children's overextension is word-to-referent extension in production. The model's "overextension" is relation-slot membership, and its over-inclusive start is set by the 0.1 initial mass, as R10 itself concedes.
2. **Fast mapping.** The rising F1 over k = 1…8 contexts resembles cross-situational learning more than single-exposure fast mapping. Fast mapping without slow consolidation is poorly retained.
3. **Consolidation.** Crystallization is parameter freezing. CLS-style consolidation is interleaved replay.
4. **Basic level.** The depth curve is not a basic-level effect (R10 says so itself).

**Recommendation:** keep the comparison as a labelled qualitative analogy in the Discussion, with the disanalogies stated. In the abstract, write "offline revision" and "property-hypothesis testing" instead of "dreaming" and "riddle-style".

### 5.5 Required baselines and controls (E10)

| ID | Control | Answers |
|---|---|---|
| D-B1 | **Property identification baselines** on the same captured pairs: AMIE 3 (PCA confidence), Fleischhacker-style axiom mining, and an IterE-style score read off the learned slot operator | Does held-out testing against random-tail controls beat standard confidence or embedding scores? |
| D-B2 | **Link-prediction baselines for erased-edge recovery:** RotatE / ComplEx / TransE and AnyBURL ranking the same candidate pools | Is AUC ≈ 0.99 specific to the learnable ontology, or would any KG-completion method get it in this world? |
| D-B3 | **Relation-discovery baselines at the same slot budget:** k-means / GMM on pair-difference or unbinding vectors; TransG / CTransR clustering; IRM on the candidate graph | Do blank-slot operators add anything over clustering pairs? |
| D-B4 | **Null-world false-discovery control:** plant no hidden relation, or permute candidate pairs; report accepted slots. E10.4 accepts about 7.7 fragment slots from an empty seed | Calibration of slot acceptance; motivates a merge test or an MDL charge |
| D-B5 | **Adaptive-holdout control:** fresh validation splits per pass, Thresholdout, or proposal counting with family-wise error | Whether the bootstrap lower bound > 0 stays calibrated over many proposals and dreaming passes |
| D-B6 | **Dreaming ablations:** greedy reduced-error pruning without scheduling; random proposals accepted at a matched rate; MDL acceptance (DreamCoder-style) | Is the held-out-vs-control criterion what does the work? |
| D-B7 | **Frame-inference baselines:** resonator-network decoding; least squares plus thresholding / Lasso; à-la-carte or CoLLEGe embedding inference followed by nearest-neighbour frame decoding | Is the "fast-mapping curve" more than generic sparse recovery? |
| D-B8 | **Prior-pull comparisons:** an L2-SP / EWC λ sweep; Faruqui retrofitting of the atomics; Functional Retrofitting. Report the WordNet reversal (frozen best) in the main text | Size of the L2-to-prior effect relative to known methods |
| D-B9 | **External benchmark:** hide symmetric relations (WN18RR `verb_group`, `similar_to`) and inverse pairs in WN18RR / FB15k-237; > 3 seeds (the absent-mode ARI interval includes 0); sweep the initial candidate mass (0.01 / 0.1 / 0.3) for the overextension claim | Comparability with published methods; initialization artefacts |

### 5.6 Recommended wording (D)

> We make the ontology mapping of the channel learnable under an L2 pull toward the curated prior. The model proposes structure edits: candidate edges, initially empty relation operators opened one at a time, and offline revision passes. Each edit is accepted only if it improves fit on held-out observations relative to a matched control. In a planted synthetic ontology this recovers erased edges, discovers hidden relations at about half of oracle alignment, identifies symmetric and inverse relations without labels by testing logical-property hypotheses on held-out pairs, and repairs injected errors. On WordNet with frozen GPT-2 anchors these effects largely do not transfer; the joint-LM test is pending.

---

## 6. The method as a whole: HRRBERT extended to causal LMs

### 6.1 Verdict: novel with narrowed wording

From 2024 to October 2026, no work was found that combines:
- concept vectors **bound by HRR/VSA from ontology relation–filler edges** over a shared, jointly trained dictionary, with no per-concept parameters;
- a causal context query over those edges;
- **gated additive injection at the last subtoken** of spans found by a deterministic, prefix-causal linker in a **causal LM**;
- **joint training** both from scratch and in pretrained hosts (full fine-tuning or LoRA);
- dictionary growth and self-authored ontologies.

Three works not on the earlier lists rule out broader phrasings:
- **KnowLA** (NAACL 2024) already injects linked KG-entity vectors into a decoder LLM's word subtokens with context attention and LoRA.
- **Dhanraj & Eliasmith** (EMNLP 2025) already add HRR vectors into a causal LLM's residual stream.
- **Dhayalkar** (KnowFM 2026) already *proposes* VSA "latent sub-graph injection" of role–filler bindings into foundation models, but runs no experiments.

### 6.2 Closest works (new since the 2026-10-02 note)

| # | Citation | Verified | What it does | Precise difference | Verdict |
|---|---|---|---|---|---|
| M-a | Xindi Luo, Zequn Sun, Jing Zhao, Zhe Zhao, Wei Hu. *KnowLA: Enhancing Parameter-efficient Finetuning with Knowledgeable Adaptation*. NAACL 2024, pp. 7153–7166. https://aclanthology.org/2024.naacl-long.396/ (arXiv:2403.14950) | ✓✓ | Words are linked to WordNet / ConceptNet / Wikidata entities. Frozen KGE vectors (TransE, RotatE, …) pass through a trainable mapping. A query (mean of the word's subtoken states) attends over candidate entities with a knowledge sentinel. The result is added residually, with a learned scale, to all of the word's subtokens at a mid layer of Llama-2-7B, trained with LoRA on instruction data | Per-entity frozen KGE table: no shared dictionary, no binding of edges, no zero-per-concept rows (unknown entities fall back to the sentinel). Attention is over candidates (disambiguation), not over one concept's edges. Injection is mid-layer into every subtoken from a pooled vector that includes later subtokens, so it is not prefix-causal. No from-scratch training, growth or self-authoring | **requires a new baseline; narrows wording.** The closest prior work for the retrofit setting |
| M-b | Varun Dhanraj, Chris Eliasmith. *Improving Rule-based Reasoning in LLMs using Neurosymbolic Representations*. EMNLP 2025, pp. 30577–30596 (arXiv journal-ref). https://arxiv.org/abs/2502.01657 | ✓✓ | Encodes LLaMA-3.1-8B hidden states into HRR/VSA vectors, computes rule-based arithmetic in VSA space, and merges the decoded result back into the hidden state; the LLM is frozen | No ontology and no dictionary of concepts. The VSA content is computed from hidden states, not composed from symbolic frames. Mid-layer, task-specific, frozen host | narrows wording (no "first HRR/VSA inside an LLM") |
| M-c | Sahil Rajesh Dhayalkar. *Overcoming the Impedance Mismatch: A Theoretical Roadmap for Fusing Foundation Models and Knowledge Graphs*. KnowFM 2026, pp. 78–89. https://aclanthology.org/2026.knowfm-1.6/. Also his *Attention as Binding*, arXiv:2512.14709 (already cited) | ✓✓ | A position paper. It advocates "Vector Symbolic Architectures for latent sub-graph injection" (role–filler bindings into intermediate layers at inference) and "Structured Residual Streams" | No implementation, linker, dictionary, training regime or results; inference-time, retrieved subgraphs | narrows wording. Cite it as a proposal that this work instantiates and tests |
| M-d | Jiayu Hou, Lei Wang. *When to Adapt: Conditional Memory Adapters for Retention-Preserving Domain Specialization* (Engram Adapter). Findings of EMNLP 2026 (arXiv comment). https://arxiv.org/abs/2608.29327. Related: TF-Engram (arXiv:2607.07388 ✓), Lngram (arXiv:2605.24869 ✓) | ✓✓ | Engram-style n-gram conditional memory as a post-hoc adapter on frozen Qwen3-4B/8B, with a gate. Better in-domain accuracy (MedMCQA) while retaining 99.4–100.1% of out-of-domain capability | Hashed or surface n-gram keys with free slot vectors; no ontology, composition or linker | **requires a new baseline** (a retrofit-setting C1h-E plus an OOD-retention suite) |
| M-e | Joel Barmettler, Abraham Bernstein, Luca Rossetto. *ConceptFormer*, now published as *ConceptFormer: Towards Graph-Native Grounding of Large Language Models via Latent Concept Injection*, Companion Proceedings of WWW '26, DOI 10.1145/3774905.3794653 | (m) for the venue (Crossref); arXiv:2504.07624 ✓✓ in the 2026-10-02 note | Already covered: KG-neighbourhood "concept vectors" for a frozen GPT-2 | New angle: the subtitle takes the phrase **"latent concept injection"**, so avoid it | terminology |
| M-f | Map-tuning (A2), PELT (C4), K-Tokeniser (A1) | ✓✓ / ✓ | Retrofit-mapping precedents | See §2.3 and §4.2 | narrows wording |

**Further works** (citation only):
- KG-Adapter (Findings ACL 2024 ✓);
- ALIGNed-LLM (arXiv:2507.13411 ✓, venue unverified);
- *Frozen Memory Is Not Enough* (arXiv:2608.17050; Engram tables need an aligned reader across backbones; arXiv-API record);
- Tao Lin's collision-free Engram study (arXiv:2601.16531 ✓; collisions act as a regularizer);
- Morand, Mothe & Piwowarski (BlackboxNLP 2025, arXiv:2510.09421; multi-token entity information goes beyond the last token, relevant to the injection-site ablation);
- Randhir Kumar (arXiv:2606.24948; HRR memory fails at zero-shot multi-hop composition, so avoid multi-hop claims);
- the TPR-interpretability papers already cited (Zhang & McCoy; McCoy et al. 2026);
- Johnson et al., *Embeddings of clinical codes enable knowledge-grounded AI in medicine*, npj Digital Medicine 2026 ((m), content unread; check for the clinical track).

### 6.3 Positioning as an extension of HRRBERT

- **The canonical record (✓✓, CEUR-WS index opened).** Bing Hu, Trevor Yu, Tia Tuinstra, Ryan Rezai, Harshit Bokadia, Rachel DiMaio, Thomas Fortin, Brian Vartian, Bryan Tripp. *Encoding Medical Ontologies With Holographic Reduced Representations for Transformers*. Joint proceedings of KiL 2024 and DL4KG 2024 (co-located with KDD 2024), CEUR-WS Vol-3894, paper 13, pp. 81–92. https://ceur-ws.org/Vol-3894/. Also on OpenReview as the KiL 2024 poster, forum `LN4zA2D8vd` (API record).
- **What is inherited and what is new.**
  - Inherited from HRRBERT: HRR binding of atomic and relation vectors following SNOMED CT; joint training of the atomics with a transformer; the rare/OOD-code motivation.
  - New:
    - causal text LMs instead of an encoder over code sequences;
    - a deterministic, prefix-causal span linker over multi-token words, with injection at the last subtoken and the tokenizer unchanged;
    - the context query (M1), the factored mapping (M2), dictionary growth (M3) and self-authoring (M5);
    - pretrained-host retrofit (E9);
    - a learnable ontology (E10);
    - the operator ablation, which HRRBERT did not have.
- **Follow-ups.** No third-party follow-up of HRRBERT was found. Google Scholar lists two citing works, both University of Waterloo theses from the group:
  - T. Yu 2025, *Encoding FHIR Medical Data for Transformers*;
  - B. Hu 2026, *Generative Synthetic Data Models for Pre-Clinical Drug Discovery*.

  Neither moves to causal text LMs. The author should check that T. Yu 2025 does not already contain an HRR causal-LM variant that would count as prior disclosure. **Resolved 2026-10-04 (author): it does not.**
- **The OpenReview forum `tfyLS1cB5W`** appears in Google Scholar as *Encoding Ontologies with Holographic Reduced Representations for Transformers* (BX Hu, T Yu, T Tuinstra, R Rezai, H Bokadia, R DiMaio, …). It looks like the group's earlier, general (not medical-only) ICLR 2024 submission. The forum is still bot-blocked, so it remains unverified. If it is public and non-anonymous, cite it as the group's own prior work, not as an independent precedent. **Resolved 2026-10-04 (author): it is the group's own previous work.** Cite the published HRRBERT record as the group's prior work (third person for double-blind review), not as an independent precedent.

### 6.4 Terminology

| Term | Problem | Recommendation |
|---|---|---|
| "semantic channel" | Established in semantic communications ("joint semantic-channel coding", "semantic channel equalization") and in Lu's semantic channel theory (arXiv:1803.08979, 1805.01288). Also used by 2026 ML papers (SemPSG arXiv:2609.36619; DSCH-Loss arXiv:2607.24567; a VLA "shared semantic channel", arXiv:2607.13597). These were seen in arXiv listings only | Do not use it bare in the title or abstract. Use "ontology-composed (HRR) span channel" or the working name CG-VSA, and qualify at first use |
| "concept channel" / "concept injection" / "latent concept injection" | "Sparse Concept Channels…" (arXiv:2607.20993); ConceptFormer's 2026 subtitle; Large Concept Models use "concept" for sentence embeddings | Say "ontology concept vector" |
| "frame" | Frame semantics; the *Frame Representation Hypothesis* (arXiv:2412.07334) uses "frames" for multi-token words | Define "ontology frame = the set of relation–filler edges of a concept" at first use |
| "retrofit" | Faruqui et al. 2015; post-specialisation; *Retrofitting LLMs with Dynamic Tokenization* | "joint fine-tuning into a pretrained host" or a defined "retrofit" |
| "token tax" | The fertility-based "Token Tax" (Lundin et al. 2025), already flagged | Keep "syntax → semantics tax" and the distinction |
| "dreaming", "riddle-style", "fast mapping", "overextension" | Loaded cognitive-science and ML terms (§5.4); "dreaming" means generative replay or RL self-play elsewhere (Behrouz et al. 2026) | Use "offline held-out-tested revision" and "property-hypothesis testing" in the abstract; keep the cognitive terms as a labelled analogy in the Discussion |

### 6.5 Baselines for the overall paper

These are beyond those already planned: C1, C1-matched, C2, C1s, C1h / C1h-E, C1m, C3t, the translation operator, and the operator ablation.

1. **KnowLA-style per-concept KGE.** Train TransE / RotatE on the same ontology, map with an MLP, and inject (a) at our site (last subtoken, gated input add) and (b) at KnowLA's site (mid layer, all subtokens, pooled with a sentinel). This tests whether composition beats standard per-concept KG vectors, especially for rare and held-out concepts, which a KGE cannot represent without retraining, and whether our site matters. Note that (b) leaks later-subtoken information in a causal LM.
2. **Jointly trained per-concept table initialized from KGE.** This separates composition from joint training; C2 is randomly initialized.
3. **Verbalized-frame text baseline.** Give the same frames as text (inline glosses in K-BERT / Dict-BERT style, or as extra CPT text) at matched tokens and FLOPs. Same as B3 of the 2026-10-02 note for E7, and as A-B3 / C-B2 here.
4. **Retrofit conditional memory** (Engram Adapter / TF-Engram style) at matched parameters, under the same full-FT and LoRA budgets.
5. **Parameter-matched LoRA without the channel**, with the LoRA rank raised to the channel's parameter count, plus an **OOD-retention suite** on general benchmarks.
6. **Injection-site ablation:** first vs. last vs. all subtokens; input vs. mid layer.
7. **Linker-noise curve:** wrong-concept links at controlled rates.
8. **Throughput and parameter overhead** against all of the above.

---

## 7. Consolidated required baselines and controls, prioritized

P0 = needed before the corresponding claim enters the abstract. P1 = needed for the full paper.

| Prio | Baseline / control | Claims | Cost class |
|---|---|---|---|
| P0 | **≥ 3 seeds** for every E9 arm (R9 is one seed) | A, B, C | training (queued for T5) |
| P0 | **In-context verbalized frame / definition** at first mention, scoring only continuation tokens. Report channel + in-context together, and the token cost | A, C | evaluation only |
| P0 | **New-token baselines:** subtoken-mean (FVT/Hewitt), encoded-definition (Token Distillation style) and type-only initialization, with continued training; held-out terms initialized zero-shot | A, C | 3 extra E9 arms per host |
| P0 | **Same-site vector from frame text or KGE:** verbalized-frame encoder + projector, or a KnowLA / map-tuning KGE + MLP, at the last-subtoken site, parameter-matched | A, C, method | one new channel mode + arms |
| P0 | **Operator and specificity ablation within E9:** `hrr` vs `random_fixed`, `untyped`, `translation`; shuffled frames; relation-shuffled frames; zero-vector gate | A, C, method (OP.1) | E9 arms |
| P0 | **Lexical-overlap split** of continuation tokens (filler-verbalizing vs. other), plus **T4/T1 natural-text tracks** | A | analysis only + queued runs |
| P0 | **Bit-matched unstructured side module** (FP16 LoRA or free table at equal bytes; C2 at the channel's precision) **and channel off at INT4** | B | training arms + evaluation |
| P0 | **GPTQ and AWQ (and NF4)** in addition to torchao RTN INT4; calibration on generic and in-domain text; also a variant that quantizes the tied embedding | B | evaluation (tooling) |
| P0 | **DiD reporting** with absolute nats/token in all four cells; ratios only alongside | B | analysis only |
| P0 | **ROME / MEMIT / AlphaEdit and IKE** on the identical edit items and hosts (EasyEdit), with CounterFact/zsRE metrics | C | evaluation (tooling) |
| P0 | **Frame transplant, channel-off audit, edit-to-random-filler null, intra-entity locality** | C | evaluation only |
| P1 | Retrofit Engram adapter / C1h-E at matched parameters; parameter-matched LoRA; OOD-retention suite | A, method | training arms |
| P1 | Revert diagnostics (delta norms vs. INT4 step); QAT / QLoRA arm | B | analysis + arms |
| P1 | CoLLEGe-style generator; definition-encoder new-word vectors; ALCUNA/COMPS-style items; power to about 700 items per arm | C | moderate |
| P1 | Injection-site ablation; linker-noise curve; KGE-initialized joint table | method | arms |
| P1 (before any D claim) | AMIE 3 / IterE-style property identification; RotatE / AnyBURL recovery; clustering / TransG / IRM discovery; null-world false-discovery rate; reusable holdout; dreaming vs. pruning / MDL / random acceptance; resonator / Lasso frame inference; WN18RR hidden relations | D | CPU mostly |

---

## 8. Recommended abstract wording

Three rules for the abstract:
- Results that need pending controls are shown as slots.
- "Semantic channel", "first", "retrofit" (undefined), "knowledge editing via the ontology rather than weights" and "makes quantized models robust" are deliberately absent.
- Use "HRR" for the default operator, and say "compositional parameter sharing" unless the operator ablation separates `hrr` from `random_fixed` (rule 7 of the claims ledger).

> HRRBERT (Hu et al., 2024) composed SNOMED CT code embeddings from atomic and relation vectors with holographic reduced representations (HRR) and trained them jointly with an encoder over medical-code sequences, improving rare codes. We extend this to causal text language models. A deterministic, prefix-causal linker maps multi-token terms to ontology concepts. Each concept vector is bound from the concept's relation–filler edges over a shared, jointly trained dictionary, with no per-concept parameters, and can be re-weighted by a causal context query. The vector is added to the input embedding at the term's last subtoken, leaving the tokenizer unchanged. We train it from scratch (50M–125M parameters) and by joint fine-tuning into pretrained small hosts. On a contamination-free synthetic glossary, joint fine-tuning into SmolLM2-135M/360M lowers next-token loss after held-out terms, whose vectors are composed zero-shot from their frames, by 7.5–11.7%. On seen rare terms it beats a capacity-matched free per-concept table, and it leaves text the host already models unchanged `[[A: vs. new-token initialization and in-context frames; seeds; T4/T1]]`. Without any weight update, writing a frame for an invented word modestly improves property selection, and replacing one edge of a seen term's frame shifts predictions toward the new fact `[[C: vs. ROME/MEMIT and in-context editing]]`. `[[B: one hedged clause on INT4 only if the bit-matched and GPTQ/AWQ controls pass]]` `[[operator ablation: HRR vs random fixed → "compositional parameter sharing" framing if they tie]]`

For E10, use at most one sentence, and only as a synthetic result (§5.6); omit it from the abstract until E10.2.

---

## 9. Unverified items (do not cite until checked)

- **HRRBERT-related**
  - ~~OpenReview `tfyLS1cB5W`~~ resolved 2026-10-04: the group's own previous work (author).
  - ~~T. Yu 2025 thesis~~ resolved 2026-10-04: no HRR/VSA causal-LM variant (author).
- **Claim A**
  - *Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension* (Joshi et al., 2026; OpenAlex only).
  - *Post-hoc Vocabulary Expansion for Embedding Models via Definition-Learning* (ACL ARR Jan 2026; title only).
  - TermGPT (CoRR 2025; title only).
  - Golden-Retriever (arXiv:2408.00798; snippet).
  - InfuserKI, StructTuning, *Concept-Aware Fine-Tuning* (snippets).
  - exBERT (Crossref only).
  - The anonymous ARR Aug 2026 *Balancing Semantic Priors and Learnability…* (OpenReview API record; anonymous).
  - Venues not confirmed: AdaptiVocab, KoPA, GraphToken, WinoDict, Ovadia et al., KGLM, Sparse Memory Finetuning, Feher et al. (ACL 2025 per OpenAlex), ReaLM (WWW 2026 per OpenAlex).
  - The search thread reported that Token Distillation is retitled "AweDist" on arXiv. The abs page opened here (v3, Mar 2026) still shows *Token Distillation: Attention-aware Input Embeddings For New Tokens*. Use the abs-page title.
- **Claim B**
  - HQQ (blog, not opened).
  - *Long-Tail Knowledge in LLMs: Taxonomy…* (arXiv:2602.16201).
  - *Emergent Specialization: Rare Token Neurons* (OpenReview 0WpeebY9Qi).
  - *Reliability Scaling Laws for Quantized LLMs* (arXiv:2607.10855).
  - QTALE, Recover-LoRA (2606.04238), Delta-CoMe (2406.08903), *The Undetected Damage of Quantization on Retrieval* (2609.24322).
  - Venues of SmoothQuant, QuIP# and Super Weight.
  - Not searched (from memory): Guan et al. 2019 (4-bit embedding-table PTQ); Berges et al. 2024 (*Memory Layers at Scale*); Springer et al. 2025 (*Overtrained LMs are harder to fine-tune*).
  - ICLR venues of Kumar et al. 2025 and Catalan-Tatjer et al. 2026 come from OpenReview records (the abs pages carry no venue comment).
- **Claim C**
  - Anonymous ICLR 2027 submissions (OpenReview API records only): *Auditing Fact Deletion in Language Models with Separable Memory*; Co-LMLM; KGHEdit; *Multi-Hop Knowledge Editing via Intermediate-Entity Representation Alignment*.
  - *Controlling the Scope of Knowledge Editing…* (TMLR under review).
  - Venues of KBLaM (ICLR 2025), LMLM (ICLR 2026), MEMIT (ICLR 2023) and CoLLEGe (COLM 2024): from OpenReview records.
  - Venues of KGT, KnowEdit, Raman et al. 2021, Bahdanau et al. 2017 and Lampinen & McClelland 2017.
  - MobiEdit (2506.13772) and *Selective Contextual Reasoning* (2503.05212): arXiv search records only.
  - DySK-Attn and Fact Grounded Attention: mechanism unverified.
  - Map-tuning: whether it evaluated swapping individual entity vectors. Read §4–5 before writing "they did not test edits".
  - REMEDI: the exact intervention layer (the abs says "hidden representations").
- **Claim D**
  - Völker & Niepert 2011, Fleischhacker et al. 2012 and Tettamanzi et al. 2014/2017: Springer pages blocked; axiom types and statistics unchecked.
  - Potoniec 2020: the abstract was not visible. It may be close to (c).
  - Kok & Domingos 2007; AMIE / AMIE 3 / AnyBURL / Neural LP / DRUM / RNNLogic / NCRL / RotatE: venues from metadata.
  - TransG / CTransR mechanism details: from memory.
  - 2026 sleep-for-LLM snippets (arXiv:2605.26099; SCM arXiv:2604.20943).
  - Granitzer et al., *Automated Ontology Learning and Validation Using Hypothesis Testing* (year missing).
  - Classic word-learning papers not found online: Carey & Bartlett 1978; Clark 1973; Bloom 2000.
  - Venues of EWC (PNAS 2017) and WSCL (TNNLS 2024).
  - DreamCoder's PLDI 2021 title and venue (the PLDI page was not opened).
- **Method as a whole**
  - EntiFA (KSEM 2025): whether it injects entity embeddings at entity tokens in LLMs.
  - GLA-LoRA (Neural Networks 2026; cites KnowLA).
  - KG-BiLM.
  - *Large Language Models Meet Neuro-Symbolic Computing* (OpenReview MHT8FAbcy3).
  - Schlegel et al., *Beyond Addition: HDC Binding for Transformers' Position Encoding* (IJCNN 2026).
  - Johnson et al. 2026 (npj Digit. Med.): content unread.
  - The "semantic channel" collision papers: arXiv listings only.

---

## 10. Search record

| Thread | Queries / lookups | Main sources | Query log (session scratch; not committed) |
|---|---|---|---|
| A retrofit | 128 logged | OpenAlex, Hugging Face papers, OpenReview API, Crossref, arXiv API, arXiv/ACL pages | `claimA_retrofit_x7q/querylog.txt` |
| B quantization | about 101 (92 logged) | DataCite (arXiv DOIs), OpenReview API, arXiv listing/API | `claimB_e9quant_zq7/qlog.txt`, `verified.tsv` |
| C editing | about 138 + 11 failed (429) | OpenAlex, Crossref, OpenReview API, arXiv API | `e9/log*.txt` |
| D self-learned semantics | about 150 (OpenAlex 50, Crossref 58, DataCite 17, Europe PMC 4, arXiv 7, ≈ 10 long-form variants) | as listed | `d/querylog.txt` |
| Method as a whole + M1/M3 re-check | about 125 (about 110 distinct) | arXiv API and web search, Google Scholar (incl. cited-by), ACL Anthology, Crossref | `nc/querylog.txt` |
| Orchestrating check | 7 arXiv website queries, 3 OpenAlex queries, about 40 verification fetches (every ✓✓ entry) | arXiv, ACL Anthology, CEUR-WS, Columbia | — |

Representative query families (the full lists are in the thread logs):
- **Vocabulary expansion and initialization:** FVT, Hewitt, Mundra, OFA, ZeTT, Token Distillation, AdaptiVocab; KG-initialized new tokens; OOV imputation (LOVE, HiCE, GRM).
- **Knowledge injection into pretrained decoders:** KG-Adapter, GraphToken, map-tuning, KnowLA trail; ontology/UMLS/SNOMED concept embeddings; invented/nonce-word and contamination-free evaluation (ECBD, ALCUNA, NewTerm, WinoDict).
- **Quantization and the long tail:** Hooker, Marchisio, Namburi, LLM-KICK; precision scaling laws; training dynamics and PTQ; unlearning vs. quantization; GPTQ/AWQ/SpQR/super weights; RAG/GraphRAG vs. quantization; Engram and quantization.
- **Knowledge editing:** ROME/MEMIT/AlphaEdit/GRACE/SERAC/IKE; KG editing (GLAME, KEDKG, OneEdit, KGT); editable symbolic memories (Facts as Experts, KBLaM, LMLM, Larimar, REMEDI); new-word learning (CoLLEGe, Minnow, Bahdanau, Lampinen & McClelland); VSA/HDC editing (none found).
- **Relation discovery and property mining:** IRM, OntExt, open relation discovery, AMIE, IterE, statistical schema induction.
- **Learning loops:** hypothesis testing; wake–sleep, DreamCoder and sleep-for-LLMs; fast mapping, overextension and CLS; resonator networks.
- **VSA/HRR in LLMs:** Dhanraj & Eliasmith, Dhayalkar, Hyperdimensional Probe, GHRR.
- **Neighbouring areas:** the Engram 2026 family; "semantic channel" and "concept channel" collisions; Large Concept Models and tokenizer-free models; gradient-conflict splitting.

**Absence-of-evidence caveat.** No hit was found for:
- VSA/HDC knowledge editing in LMs;
- "riddle" or 20-questions-style relation identification;
- VSA relation discovery;
- per-usage momentum-conflict splitting.

Given the degraded tooling, these are weak negatives. Keep keyword alerts on:
- HRR/VSA + language model + ontology;
- Engram/conditional memory + ontology/KG keys;
- knowledge editing + input embedding / KG composition;
- quantization + rare/long-tail vocabulary + side module.
