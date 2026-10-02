# Related-work re-check, 2025–2026 (pre-submission checklist item)

**Date:** 2026-10-02. **Scope:** the claims [proposal.md](../resources/plan-improvement/proposal.md) §7 marks as most exposed to a missed recent preprint, plus four further areas:

- (i) usage-level momentum- or gradient-conflict splitting of shared dictionary vectors (M3);
- (ii) context attention over HRR-bound ontology edges (M1), and VSA/HRR inside LMs;
- (iii) span-level knowledge channels for causal LMs (successors of KALM and Engram) and multi-token word modelling (M4, H-A);
- (iv) self-authored ontologies and self-improvement through synthetic structured knowledge (successors of EntiGraph) (M5, H-G).

This note **does not edit** [related-work.md](../resources/plan-improvement/related-work.md); the orchestrator decides what to adopt. Section 6 collects the proposed changes.

**Method.** Four parallel web searches, one per area, covered arXiv, the ACL Anthology, OpenReview, PMLR/NeurIPS/ICLR proceedings, Semantic Scholar and general web search, for January 2025 to October 2026. Each search also looked for older work that the 2026-09-30 search could have missed. Together they ran about 180 search queries and about 150 page fetches (the queries are listed in the per-area agent reports, summarized in §7).

**Verification standard.** A citation is **verified** only if the abs, proceedings or OpenReview page was opened and its title, authors and year matched.
- Entries marked **✓✓** were re-opened independently for this note on 2026-10-02, because they change a claim or a baseline.
- **✓** entries were verified by the search thread that found them.
- In addition, every arXiv identifier in this note was resolved through the arXiv API on 2026-10-02. Title and author lists were matched, and author names corrected where the threads had abbreviated them. The raw metadata is kept in the session scratch (`related-work-recheck-2026-10.md.arxiv.json`, not committed).
- Venues not confirmed on a proceedings page are marked "venue unverified".
- Anything seen only in search snippets is listed under "unverified" and must not be cited until checked.

**Limitations.**
- The shared web-search budget ran out near the end of three of the four threads.
- dblp and OpenReview pages were partly blocked by bot checks.
- Coverage of graph-walk synthetic-CPT variants and of 2025 decoder-LM entity injection is therefore thinner than the rest.
- As in the first search, absence of a hit is weak evidence.

## 1. Summary

| Claim (proposal §7 wording) | Pre-empted? | Closest new work | Forced change |
|---|---|---|---|
| **M3**: "a first-order usage-level split test for shared dictionary vectors, read from optimizer state, calibrated by a permutation null" | **No** | Parameter Differentiation (Wang & Zhang, AAAI 2022): duplicates shared parameters whose task gradients conflict and splits the tasks; Lin et al. 2026 (scalar-level, cosine < −0.5) | **Concede the general move** (gradient-conflict duplication of shared parameters with partitioned users) and credit it. New baseline: a threshold "differentiation" rule without the null. Optional: fixed-K gated slots (MoME). |
| **M1**: "a query-conditioned weighting of VSA-bound ontology edges with zero per-concept parameters that recovers the static bundle as τ → ∞" | **No** | CokeBERT (Su et al., AI Open 2021; *missed by the first search*): text-conditioned softmax attention over a KG entity's relation-keyed neighbour edges; MKGL (NeurIPS 2024): zero-per-entity-parameter KG-composed LLM input tokens, no query | Wording survives only through "VSA-bound" and "zero per-concept parameters". "Recovers a static aggregate at τ → ∞" is not distinctive by itself (CokeBERT ablates against mean pooling); **the limit being the HRRBERT bundle is.** New control: an additive/translation operator (`a_e + r_e`, TransE/CokeBERT-style) in the operator ablation. |
| **M4**: "a compositional, growable span channel for causal LMs whose rows are built from ontology relation–filler edges, with a last-subtoken rule" | **No** | ConceptFormer (Barmettler et al., 2025): a shared generator turns Wikidata neighbours and edges into concept vectors for a **frozen** GPT-2, appended as extra positions | Add "by fixed, parameter-free binding", "added in place at the last subtoken of a prefix-causally linked span" and "during LM training". Do not claim to be first to compose KG-edge vectors for causal LMs. |
| **H-A**: "measures the tax on ontology-linked multi-token strata and whether structure beats a hashed lookup memory" | **No** | The Engram family: Engram's own ablations; Memory Grafting, TN-gram and FactorEngram (2026) beat vanilla Engram | C1h as specified (one input-level hashed table) is a **weak** lookup baseline. Either add **C1h-E**, an Engram-faithful hashed memory, or qualify the claim as "at matched parameters, injection site and gate". Add C1m (subtoken-mean span vector) and an alias stratum. Rename nothing: C1s (shuffled frames) already isolates ontology truth. |
| **M5 / H-G**: "self-authored frames verified by held-out gradient utility, consumed through the channel, beating compute-matched and EntiGraph-style controls" | **No single work**; components are | Evontree (2025, self-authored ontology → self-distillation); Montessori-Instruct (ICLR 2025) and OptimSyn (2026) (synthetic data scored by first-order held-out influence); GraphGen and SoG (2025) (KG-/loss-targeted structured synthesis); Active Reading (ICLR 2026) and SEAL (NeurIPS 2025) (self-authored study data beats frontier-teacher data) | Extend "not novel" to: ontology-guided self-training, KG-/loss-targeted synthesis, influence-filtered synthetic data, and self-authors beating teachers. New E7 controls: self-authored **unstructured notes** through the same utility filter; **SPA / synthetic-QA CPT**; **verbalized-frames CPT** (same verified edges as text, no channel). |
| Multi-token word modelling (framing) | n/a | Dual-Route induction (COLM 2025) supports the last-subtoken rule; "The Token Tax" (2025) is a name collision | Cite as support. Distinguish "syntax → semantics tax" from the fertility-based "token tax". |

**Bottom line:** nothing pre-empts the narrowed §7 claims. Three wordings must be narrowed further (M3, M4, M5), and one wording needs an explicit contrast (M1, with CokeBERT). One experimental gap is serious enough to fix before G2: the lookup-memory control for H-A (C1h). Five new controls are recommended (§6.2).

## 2. (i) Splitting shared dictionary vectors on gradient conflict (M3)

No paper found combines all four of the following:
- per-usage first moments read from Adam state;
- the cancellation gain `G(v)` along the top between-usage eigenvector;
- a permutation-null acceptance test;
- Adam-SNR screening, applied to shared embedding or dictionary vectors inside an LM.

The general move, duplicating a shared parameter whose users' gradients conflict and partitioning the users, has precedent. That precedent is not in the current credit list.

| # | Citation | Verified | What it does | Closeness | Action |
|---|---|---|---|---|---|
| 1 | Qian Wang, Jiajun Zhang. *Parameter Differentiation based Multilingual Neural Machine Translation.* AAAI 2022 (pp. 11440–11448). arXiv:2112.13619. https://arxiv.org/abs/2112.13619 | ✓✓ (arXiv; AAAI page numbers from listings) | Shared units (layer, module, operation) "differentiate" into language-specific copies when inter-task gradient similarity on held-out data is negative; tasks are bipartitioned to minimise interference, every 8k steps; fixed threshold, no statistical test | **High**: same "split a shared parameter, partition its users by gradient conflict" idea, at task granularity, threshold-triggered | Credit; narrow wording; **baseline: threshold rule** (split when the maximum pairwise negative cosine of per-usage momenta exceeds τ, then bipartition) vs `G*` + null |
| 2 | Yuxi Lin, Yongkang Li, Jie Xing, Zipei Fan. *Multifaceted Scenario-Aware Hypergraph Learning for Next POI Recommendation.* arXiv:2601.11610 (v2 Mar 2026). https://arxiv.org/abs/2601.11610 | ✓✓ (abs + HTML) | Per scalar parameter, normalised per-scenario gradient cosines are computed every 10 batches; parameters with a pairwise similarity < −0.5 are split, and scenarios are clustered by gradient similarity | **High** on mechanism (hand-set threshold, no null, scenarios rather than usages) | Cite; supports the threshold baseline |
| 3 | Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, Xiao-Ming Wu. *Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning.* ICLR 2023. arXiv:2302.11289 | ✓ | Layer-wise conflict scores pick the most-conflicted layers, which become task-specific; retrain | Medium (layer/task level, offline) | Cite |
| 4 | Wooseong Jeong, Kuk-Jin Yoon. *Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning* (DTME-MTL). ICCV 2025. arXiv:2507.07485 | ✓ | Projects per-task gradients onto range and null spaces of the token covariance; null-space conflicts add task-specific tokens | Medium (conflict-triggered growth of vectors; no test) | Cite |
| 5 | Linghao Kong, Inimai Subramanian, Yonadav Shavit, Micah Adler, Dan Alistarh, Nir Shavit. *Expand Neurons, Not Parameters.* ICML 2026. arXiv:2510.04500 | ✓ | Splits every neuron into children with disjoint weight subsets after warm-up; **random splits recover most of the gain** | Medium: a reviewer will use it to argue that the test may not matter | Already controlled: the E0.2/E3 `random` policy (random vector, random direction, random usage partition, matched count). Keep it in E3 and report it next to M3 |
| 6 | Rupert Mitchell, Robin Menzenbach, Kristian Kersting, Martin Mundt. *Self-Expanding Neural Networks.* arXiv:2307.04526 | ✓ (arXiv; venue unverified) | A natural-gradient expansion score `gᵀF⁻¹g` decides when and where to add width or depth | Low–medium: with a diagonal Fisher this is close to Σ m̂²/v̂, i.e. the SNR screen | Cite; state the difference: usage-conditional cancellation vs aggregate gain |
| 7 | Lemeng Wu, Mao Ye, Qi Lei, Jason D. Lee, Qiang Liu. *Steepest Descent Neural Architecture Optimization: Escaping Local Optimum with Signed Neural Splitting.* arXiv:2003.10392 | ✓ | Signed neuron splitting (children with ± weights) | Medium on mechanics: `θ ± εv*` is a signed split | Add to the credit list next to Liu et al. 2019 |
| 8 | Muchen Li, Leonid Sigal, Renjie Liao. *MoME: Mixture-of-Memory Embeddings for Context-Aware Sparse Lookup.* arXiv:2609.15126 (Sep 2026) | ✓ | Each token's memory row becomes M gated slots during pretraining; polysemous tokens route to sense-specific slots | Medium–low (fixed M, no split test) | Optional E3 baseline: K gated copies per dictionary vector at matched parameters |
| 9 | Longrong Yang, Dong Shen, Chaoxiang Cai, et al. *Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model* (STGC). arXiv:2406.19905 | ✓ (arXiv; ICLR 2025 venue unverified) | Flags tokens whose gradient conflicts with an expert's mean gradient and re-routes them | Medium (usage-level conflict detection, no split) | Cite |
| 10 | Minsik Choi, Geewook Kim. *Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging* (MERIT). ICML 2026. arXiv:2606.01717 | ✓ | Partitions a data mixture along top PCA axes of dataset-level gradient conflict | Medium–low (splits data, not parameters) | Cite |
| 11 | Chaitanya Dwivedi, Binxuan Huang, Himanshu Gupta, et al. *Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts.* arXiv:2604.19835 | ✓ | Duplicates MoE experts mid-training, chosen by gradient-importance scores | Medium–low | Cite |
| 12 | Andrei Mircea, Supriyo Chakraborty, Nima Chitsazan, et al. *Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning.* ACL 2025. arXiv:2506.05447 | ✓ | Per-example gradients in LMs become systematically opposed ("zero-sum learning") | Low (motivation for "momentum ≈ 0 with high \|g\|") | Cite in the M3 motivation |

Optional or near-miss work, citation only:
- G³-LoRA, arXiv:2609.35189 ✓
- GradientSpace, arXiv:2512.06678 ✓
- Wang, Zaki, Kollias, Kalantzis, *Multi-Sense Embeddings for Language Models and Knowledge Distillation*, arXiv:2504.06036 ✓ (Findings of ACL 2025 per listing; venue unverified)
- Hesse et al., *Disentangling Polysemantic Channels in CNNs*, arXiv:2504.12939 ✓
- SSVQ, arXiv:2503.08668 ✓
- CDSP-MoE, arXiv:2512.20291 ✓
- DynMoE, arXiv:2405.14297 ✓

Unverified (snippets only): TransG (Xiao et al., ACL 2016; multiple vectors per KG relation, relevant to relation splitting) and Splitter (Epasto & Perozzi, WWW 2019).

**Proposed wording (M3):**
> Gradient-conflict-driven duplication of shared parameters has precedent at task and scenario level (Wang & Zhang 2022; Shi et al. 2023; Lin et al. 2026). We contribute a first-order, usage-level split test for shared dictionary vectors. It scores a candidate split by the cancellation gain of per-usage Adam momenta along the top between-usage eigenvector and accepts it only against a permutation null, after an Adam-SNR screen. The ±ε offset and sign partition are credited to splitting steepest descent / signed splitting, G-means and LBG.

## 3. (ii) Context attention over HRR-bound ontology edges (M1); VSA/HRR in LMs

No work attends with a context query over HRR- or VSA-*bound* ontology edges, with zero per-concept parameters and the HRR bundle as the τ → ∞ limit. The first search missed one older paper that must be contrasted (CokeBERT). No 2025–2026 follow-up of HRRBERT or by its authors was found.

| # | Citation | Verified | What it does | Closeness | Action |
|---|---|---|---|---|---|
| 1 | Yusheng Su, Xu Han, Zhengyan Zhang, Peng Li, Zhiyuan Liu, Yankai Lin, Jie Zhou, Maosong Sun. *CokeBERT: Contextual Knowledge Selection and Embedding towards Enhanced Pre-Trained Language Models.* AI Open 2021. arXiv:2009.13964. https://arxiv.org/abs/2009.13964 | ✓✓ | A semantic-driven GNN builds a mentioned entity's embedding by softmax attention over its KG neighbours (2 hops). The query comes from the whole input text; keys are relation-only; values are TransE-style translations (`n ± r`) of pretrained per-entity vectors. The result is fused into BERT (ERNIE-style). Ablations include mean pooling | **High** for "query-conditioned weighting of ontology edges"; differs in no binding (translation), per-entity vectors, a bidirectional sentence-level query, and a non-HRR static limit | **Contrast explicitly** in Related Work and Method §3.2; **new operator control** `translation` (`T_r(x) = x + r`, relation-only keys) in E1/E2/C8 |
| 2 | Lingbing Guo, Zhongpu Bo, Zhuo Chen, et al. *MKGL: Mastery of a Three-Word Language.* NeurIPS 2024 (spotlight). arXiv:2410.07526 | ✓ | New KG-entity token embeddings for an LLM are built from the LLM's text-token embeddings plus multi-layer PNA aggregation over KG neighbours. Inductive (no per-entity parameters); not conditioned on the sentence | Medium | Cite; do not claim the first zero-per-concept-parameter KG-derived LLM input embedding |
| 3 | Haochen Liu, Song Wang, Chen Chen, Jundong Li. *Question-Aware Knowledge Graph Prompting for Enhancing Large Language Models.* arXiv:2503.23523 | ✓ | The question embedding enters GNN neighbourhood aggregation; the pooled output is prepended as soft prompts | Medium (query-dependent edge weighting for LLMs, prefix-level, no binding) | Cite |
| 4 | Calvin Yeung, Zhuowen Zou, SungHeon Jeong, Wenjun Huang, Nathaniel D. Bastian, Mohsen Imani. *Generalized Holographic Reduced Representations.* arXiv:2405.09689 (v2 May 2026) | ✓ | Non-commutative GHRR binding; shows it "can implement a kind of attention"; replacing transformer attention with it helps a language-modelling task | Medium for VSA-in-LM positioning (VSA *inside* attention, not attention over bound edges) | Cite next to Hrrformer and Dhayalkar 2025 |
| 5 | Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, Partha Talukdar. *Composition-based Multi-Relational Graph Convolutional Networks* (CompGCN). ICLR 2020. arXiv:1911.03082 | ✓ | Composes neighbour and relation by subtraction, multiplication or circular correlation, then aggregates | Medium–low: the GNN precedent for "bind relation and filler per edge, then sum"; per-entity vectors, no text query | Cite next to HolE |
| 6 | Yezi Liu, William Youngwoo Chung, Hanning Chen, Calvin Yeung, Mohsen Imani. *Encoder-Free Knowledge-Graph Reasoning with LLMs via Hyperdimensional Path Retrieval* (PathHD). arXiv:2512.09369 | ✓ | KG relation paths as GHRR hypervectors, retrieved by blockwise cosine; an LLM adjudicates. Hypervectors never enter the model | Medium–low (VSA-bound KG relations used *with* an LLM, outside it) | Cite |
| 7 | Xiaorui Su, Shvat Messica, Yepeng Huang, et al. (Zitnik). *Multimodal Medical Code Tokenizer* (MedTok). ICML 2025. arXiv:2502.04397 | ✓ | Ontology- and graph-aware medical-code tokens (text + graph encoders, vector-quantized), plugged into EHR transformers | Medium–low: same clinical niche as HRRBERT; static, no VSA | Cite on the clinical track |
| 8 | Mohammad Mahmudul Alam, Alexander Oberle, Edward Raff, Stella Biderman, Tim Oates, James Holt. *A Walsh Hadamard Derived Linear Vector Symbolic Architecture* (HLB). NeurIPS 2024. arXiv:2410.22669 | ✓ | A differentiable linear binding operator designed for neural networks | Low–medium | Cite; optional extra operator |
| 9 | Ruitong Liu, Boxu Lin, Peize Li, et al. *Beyond Prefixes: Graph-as-Memory Cross-Attention for Knowledge Graph Completion with LLMs.* arXiv:2510.08966 | ✓ | LLM hidden states cross-attend to graph memory tokens in several layers | Medium–low (relevant to P2 / E6) | Cite in E6 |
| 10 | Salvador E. Barbosa. *Using Holographically Compressed Embeddings in Question Answering.* arXiv:2007.07287 (2020) | ✓ | HRR binds token embedding × POS × NE-type into one input embedding for a QA RNN | Low–medium (early static HRR role–filler input embedding) | Optional citation |

Analysis papers that support the motivation only (citation optional):
- Zhang Enyan & R. Thomas McCoy, *A Unifying Perspective on Language Model Representations: From Filler-Role Structure to Mechanistic Interpretability*, arXiv:2608.29034 ✓
- McCoy, Soulos, Linzen, Smolensky, *The Emergent Symbolic Structure of Artificial Neural Networks*, arXiv:2608.29530 ✓
- Erica Coppolillo, *Injecting Knowledge Graphs into Large Language Models*, arXiv:2505.07554 ✓ (static KGE vectors as input tokens to a frozen LLM)

No action needed:
- Olin-Ammentorp & Bazhenov, arXiv:2207.08953 ✓
- *Hypertokens*, arXiv:2507.00002 ✓
- Kumar, arXiv:2606.24948 ✓
- Dai, Heinzerling, Inui, arXiv:2604.19052 ✓

**Unverified; needs the author's input:** OpenReview forum `tfyLS1cB5W`, *Encoding Ontologies with Holographic Reduced Representations for Transformers*. Search snippets list the HRRBERT author list and call it an ICLR 2024 submission; the page was blocked. If it is the group's own earlier submission of HRRBERT, the manuscript should cite only the CEUR-WS record and not count it as separate prior work.

**Proposed wording (M1):**
> A query-conditioned weighting of VSA-bound ontology edges with zero per-concept parameters, whose τ → ∞ limit is the static HRR bundle of HRRBERT. Text-conditioned attention over ontology or KG neighbours predates us (Dasigi et al. 2017; CokeBERT, Su et al. 2021). What differs is that edge values are role–filler bindings rather than per-entity vectors or their translations, and that the query comes causally from preceding tokens.

## 4. (iii) Span-level knowledge channels for causal LMs; multi-token words (M4, H-A)

No 2025–2026 paper adds ontology-edge-composed vectors with no per-concept parameters, in place, at the last subtoken of a prefix-causally linked span, during causal-LM training. None compares structured span vectors against a parameter-matched hashed n-gram memory. The Engram line, however, has moved fast: a single input-level hashed table is now a weak representative of "lookup memory".

| # | Citation | Verified | What it does | Closeness | Action |
|---|---|---|---|---|---|
| 1 | Joel Barmettler, Abraham Bernstein, Luca Rossetto. *ConceptFormer: Towards Efficient Use of Knowledge-Graph Embeddings in Large Language Models.* arXiv:2504.07624 (Apr 2025). https://arxiv.org/abs/2504.07624 | ✓✓ | A generator trained with a **frozen** LLM (GPT-2) turns a Wikidata node's neighbourhood (neighbours and edges) into "concept vectors" in the LLM's embedding space, precomputed into a lookup table. Per the paper body (search thread), they are appended as extra input positions after the mention. Large Hit@10 gains in factual recall; no random-vector or hashed baseline | **High** on mechanism (KG-edge-composed vectors for a causal LM). Differs in: learned generator rather than fixed binding; appended positions rather than in-place addition at the last subtoken; frozen LM; gold mentions rather than a causal linker | **Narrow M4 wording** (§6.1); cite as the nearest prior work |
| 2 | Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models* (Engram). arXiv:2601.07372 (already cited; re-read for ablations) | ✓ | Ablations: context-aware gating and tokenizer compression are among the most critical parts; layer-2 insertion beats layer 1; it scales better than input-level n-gram averaging (Over-Encoding). Gate visualisations fire at the ends of multi-token named entities | **High for H-A**: a hashed memory already shows a qualitative "tax relief" on multi-token entities (not stratified by ontology links) | **Add C1h-E** or qualify H-A (§6.2) |
| 3 | Runxi Cheng, Yuchen Guan, Yongxian Wei, et al. *Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory.* arXiv:2605.20948 (May 2026) | ✓✓ | Memory values are frozen hidden states of a pretrained model on frequent n-grams (exact suffix match, hashed fallback); beats vanilla Engram (53.86 vs 52.43 at 2.8B) | High for the "structure vs lookup" question: content-bearing span memories beat hashed ones, so H-A's contribution must be about *ontology* structure, not "content beats hash" | Cite; optional control C1t (§6.2) |
| 4 | Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, Yuning Qiu, Qibin Zhao, Danilo Mandic. *Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs* (TN-gram). arXiv:2606.08347 | ✓ | CP-factorised n-gram memory with factors shared across n-grams; matches or beats Engram with far fewer parameters | Medium–high: learned *shared factorisation* across lookup rows helps, so a reviewer may attribute M4 gains to atom sharing rather than ontology truth | Answered by **C1s** (shuffled frames, same atoms) — keep it in D4.9 and report it with C1h |
| 5 | Bowen Yang, Jingbo Zhou, Qinghong Miao, Hua Wu. *FactorEngram: Factorized N-gram Memory with Basis-Level Gating for Language Models.* arXiv:2609.35578 (Sep 2026) | ✓✓ | Sparse coefficients over a shared basis with gating per basis vector; improves over Engram at 340M and 1B; mid-layer placement best | As #4 | As #4 |
| 6 | Da Yu, Edith Cohen, Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, et al. *Scaling Embedding Layers in Language Models* (SCONE). NeurIPS 2025 (arXiv comment: camera ready). arXiv:2502.01637 | ✓ | Contextualized embeddings of frequent n-grams added at the input, precomputed off-accelerator; a 1B model beats a 1.9B baseline | Medium (another non-ontological informative span vector) | Cite |
| 7 | Yuwen Qu, Wenhui Dong, Chenyang Si, Caifeng Shan. *NGM: A Plug-and-Play Training-Free Memory Module for LLMs.* arXiv:2605.16893 | ✓ | Causal mean of pretrained embeddings over the trailing n tokens, injected through a cosine gate | Medium: a zero-parameter, surface-compositional span vector | **New control C1m** (subtoken-mean span vector, same site and gate) |
| 8 | Hong Liu, Jiaqi Zhang, Chao Wang, et al. *Scaling Embeddings Outperforms Scaling Experts in Language Models* (LongCat-Flash-Lite). arXiv:2601.21204 | ✓ | Hashed n-gram input embeddings (> 30B embedding parameters in a 68.5B MoE) beat parameter-matched MoE | Medium (industrial evidence that hashed n-gram memory is a serious baseline) | Cite |
| 9 | Tao Lin. *A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory: A Controlled Study of Training Dynamics.* arXiv:2601.16531 | ✓ | Collision-free memory via minimal perfect hashing shows collisions are not the bottleneck; gate credit assignment is | Medium: pre-empts "C1h loses because of collisions" | Cite; supports a context-aware gate in C1h-E |
| 10 | Yuto Nishida, Naoki Shikoda, Yosuke Kishinami, et al. *Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms* (RedirectQA). arXiv:2604.21882 (ACL 2026 per arXiv comment) | ✓ | Fact recall varies across entity aliases (Wikipedia redirects) | Medium: ontology linking maps aliases to one row, hashing cannot | **Alias stratum**: evaluate spans whose surface form is a non-canonical alias |
| 11 | Jihoon Tack, Jack Lanchantin, Jane Yu, Andrew Cohen, Ilia Kulikov, et al. *LLM Pretraining with Continuous Concepts* (CoCoMix). arXiv:2502.08524 | ✓ | Predicts SAE concepts and interleaves continuous concept vectors into hidden states during pretraining; better sample efficiency | Medium (concept channel in causal pretraining; learned concepts, not ontological) | Cite |
| 12 | Xi Wang, Taketomo Isazawa, Liana Mikaelyan, James Hensman. *KBLaM: Knowledge Base augmented Language Model.* arXiv:2410.10450 (ICLR 2025 per listing; venue unverified) | ✓ | KB triples become key-value knowledge tokens read by rectangular attention in an 8B LLM | Medium–low | Cite |
| 13 | Sheridan Feucht, Eric Todd, Byron Wallace, David Bau. *The Dual-Route Model of Induction.* COLM 2025. arXiv:2504.03022 | ✓ | Concept-induction heads attend to the *ends* of multi-token words | Support for the last-subtoken rule | Cite next to Feucht et al. 2024 and Kaplan et al. 2025 |
| 14 | Qiwei Peng, Yekun Chai, Anders Søgaard. *Understanding Subword Compositionality of Large Language Models.* EMNLP 2025. arXiv:2508.17953 | ✓ | Five LLM families compose subwords into words in three distinct ways | Background for H-A | Cite |
| 15 | Masaki Sakata, Benjamin Heinzerling, Sho Yokoi, Takumi Ito, Kentaro Inui. *On Entity Identification in Language Models.* Findings of ACL 2025. arXiv:2506.02701 | ✓ | Entity identity sits in low-dimensional early-layer subspaces | Background | Cite |
| 16 | Daniela Gottesman, Alon Gilae-Dotan, Ido Cohen, et al. *LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations.* arXiv:2509.03405 (TACL 2026 per arXiv comment) | ✓ | Entity-annotated pretraining corpus plus 12 models with checkpoints | Possible resource for H-A strata | Cite |
| 17 | Jessica M. Lundin, Ada Zhang, Nihal Karim, et al. *The Token Tax: Systematic Bias in Multilingual Tokenization.* arXiv:2509.05486 (AfricaNLP 2026) | ✓ | Fertility (tokens per word) predicts lower accuracy | **Name collision** | Distinguish "syntax → semantics tax" (stratified loss on ontology-linked spans) from the fertility "token tax" |
| 18 | Sachin Pawar, Manoj Apte, Kshitij Jadhav, Girish Keshav Palshikar, Nitin Ramrakhiyani. *Broken Words, Broken Performance: Effect of Tokenization on Performance of LLMs.* arXiv:2512.21933 (IJCNLP-AACL 2025) | ✓ | Splitting natural words into several tokens hurts task performance | Background | Cite |
| 19 | Craig W. Schmidt, Varshini Reddy, Chris Tanner, Yuval Pinter. *Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier.* COLM 2025. arXiv:2504.00178 | ✓ | Superword tokenizer | Tokenizer-side reference | Cite next to SuperBPE |

Further verified hashed or lookup memories, citation only, low closeness:
- X-GRAM, arXiv:2604.21724
- Lngram, arXiv:2605.24869
- STEM, arXiv:2601.10639
- Qwen3.8-Next, arXiv:2608.30320
- KoRe, arXiv:2605.20170
- Limited Memory LMs, arXiv:2505.15962
- Pouransari et al., arXiv:2510.02375

Confirmed by the arXiv API on 2026-10-02 but not read beyond the abstract metadata (citation only): Huang et al., *Ultra-Sparse Memory Network* (UltraMem, arXiv:2411.12364, ICLR 2025 per comment); Jie et al., *Mixture of Lookup Experts* (arXiv:2503.15798, ICML 2025 per comment); LCM team et al., *Large Concept Models* (arXiv:2412.08821); Qu et al., *Dynamic Large Concept Models* (arXiv:2512.24617); Hwang, Wang & Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (H-Net, arXiv:2507.07955).

**Proposed wording (M4):**
> A compositional, growable span channel for causal LMs whose rows are built from ontology relation–filler edges **by fixed binding with no per-concept parameters**, **added in place at the last subtoken of a prefix-causally linked span during LM training**.

**H-A:** either add C1h-E, or qualify the claim as "…whether ontology structure beats a hashed lookup memory **at matched parameters, injection site and gate**". State that Engram's multi-token-entity evidence is qualitative and not stratified by ontology links.

## 5. (iv) Self-authored ontologies; self-improvement through synthetic structured knowledge (M5, H-G)

No work has a small LM author typed ontology frames for its own poorly modelled corpus spans, admit each edge only when a held-out gradient-utility lower bound is positive, and consume the frames through a non-text compositional channel with a hidden-gold audit. Several components are now published separately.

| # | Citation | Verified | What it does | Closeness | Action |
|---|---|---|---|---|---|
| 1 | Mingchen Tu, Zhiqiang Liu, Juan Li, Liangyurui Liu, Junjie Wang, Lei Liang, Wen Zhang. *Evontree: Ontology Rule-Guided Self-Evolution of Large Language Models.* arXiv:2510.26683 (v2 Mar 2026) | ✓✓ | Extracts domain ontology knowledge from the raw LLM, detects inconsistencies with two ontology rules, and reinforces the gap knowledge by self-distilled fine-tuning (8B medical models) | **High**: the closest "self-authored ontology → self-improvement". Differs in a rule verifier (not held-out utility), text SFT, parametric rather than corpus-span source, and 8B hosts | Narrow wording; optional alternative-verifier ablation (rule consistency) |
| 2 | Zihong Chen, Wanli Jiang, Jinzhe Li, Zhonghang Yuan, Huanjun Kong, Wanli Ouyang, et al. *GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation.* arXiv:2505.20416 | ✓ | Builds a KG from source text, finds the trainee's blind spots by calibration error on KG facts, generates QA for them | Medium–high (loss-targeted structured synthesis ≈ excess-surprisal selection) | Do not claim loss-targeted structured synthesis is new; cite |
| 3 | Shengjie Ma, Xuhui Jiang, Chengjin Xu, Cehao Yang, Liyu Zhang, Jian Guo. *Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of LLMs* (SoG). arXiv:2505.00979 (v3 Sep 2025; venue unverified) | ✓ | Cross-document entity graph, graph-walk sampling, LLM-generated text; beats EntiGraph, rephrasing and raw CPT | Medium (the structured EntiGraph successor) | Cite; control only if E7 adds multi-hop evaluation |
| 4 | Jessy Lin, Vincent-Pierre Berges, Xilun Chen, Wen-Tau Yih, Gargi Ghosh, Barlas Oğuz. *Learning Facts at Scale with Active Reading.* arXiv:2508.09494 (ICLR 2026 per mlanthology) | ✓✓ (arXiv) | The model writes its own study strategies per document and trains on them; an 8B model trained on its own data beats the same model trained on 70B-generated data; EntiGraph skipped as weaker than synthetic QA | **High** for "self-authored" | **New control B1** (self-authored unstructured notes) |
| 5 | Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, Pulkit Agrawal. *Self-Adapting Language Models* (SEAL). NeurIPS 2025. arXiv:2506.10943 | ✓ (the NeurIPS poster page lists 5 authors and omits Akyürek; take the list from the proceedings PDF) | The model writes "self-edits" (e.g. implications), fine-tunes on them; RL rewards edits that improve the updated model; self-generated data beat GPT-4.1 data on SQuAD | High (self-authored knowledge plus utility signal, text channel) | B1; "a self-author beating a frontier teacher" is not new |
| 6 | Kexian Tang, Jiani Wang, Shaowen Wang, Kaifeng Lyu. *SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection.* arXiv:2603.22213 (v2 Aug 2026) | ✓ | Tuned prompts plus large-scale augmentation match or beat EntiGraph, Active Reading, SoG, SEAL and PaST | Medium, but it makes an EntiGraph-only control look weak | **New control B2** (SPA / synthetic-QA CPT at matched tokens) |
| 7 | Yangjun Ruan, Neil Band, Chris J. Maddison, Tatsunori Hashimoto. *Reasoning to Learn from Latent Thoughts* (BoLT). arXiv:2503.18866 (venue unverified) | ✓ | The LM infers latent thoughts behind raw text and trains on them; EM bootstrapping improves a 1B model over rounds without a stronger teacher, beating matched raw data | Medium–high (self-annotation → train → repeat, against matched raw data) | Cite |
| 8 | Xiaochuan Li, Zichun Yu, Chenyan Xiong. *Montessori-Instruct: Generate Influential Training Data Tailored for Student Learning.* ICLR 2025 (per search thread; not in the arXiv comment). arXiv:2410.14208 | ✓ (arXiv API) | "Local data influence" (reference-set loss before minus after one step on a synthetic point); the teacher is DPO-trained toward high-influence data | **High** on the verification mechanism | Cite; wording (§6.1) |
| 9 | Zhiting Fan, Ruizhe Chen, Tianxiang Hu, et al. *Optimsyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation.* arXiv:2604.00536 (Apr 2026; venue unverified) | ✓✓ | Optimizer-aware gradient influence of each synthetic sample (LESS-like) guides RL over generation rubrics | High on mechanism | Cite; wording |
| 10 | Jiachen T. Wang, Prateek Mittal, Dawn Song, Ruoxi Jia. *Data Shapley in One Training Run.* ICLR 2025. arXiv:2406.11011 | ✓ | First-order term = per-step dot product of training and validation gradients, the form of `U_{j,e}` | Credit | Cite next to TracIn and LESS |
| 11 | Zhiqiang Liu, Chengtao Gan, Junjie Wang, et al. *OntoTune: Ontology-Driven Self-training for Aligning Large Language Models.* WWW 2025. DOI 10.1145/3696410.3714816. arXiv:2502.05478 | ✓ | Finds SNOMED CT concepts the LLM has not mastered, self-trains on them | Medium (external gold ontology) | Cite (clinical track) |
| 12 | Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, et al. *AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction.* arXiv:2510.15339 | ✓ | KG constructor rewarded by downstream RAG utility | Medium (utility-verified edges through a retrieval channel) | Cite |
| 13 | Babaei Giglou, D'Souza, Mihindukulasooriya, Auer. *LLMs4OL 2025 Overview: The 2nd LLMs for Ontology Learning Challenge.* Open Conference Proceedings 6 (ISWC 2025). DOI 10.52825/ocp.v6i.2913 | ✓ | Challenge overview | Low | Cite next to LLMs4OL 2023 |
| 14 | Zeyneb N. Kaya, Nick Rui. *Test-Time Meta-Adaptation with Self-Synthesis* (MASS). arXiv:2603.03524 (ICLR 2026 RSI workshop) | ✓ | Meta-gradients reward useful self-generated data | Medium–high | Cite |
| 15 | Tristan Thrush, Sung Min Park, Herman Brunborg, et al. *Synthetic Data for any Differentiable Target.* arXiv:2604.08423 | ✓ | Dataset policy gradient toward a differentiable target | Medium | Cite |
| 16 | Kazdan, Schaeffer, Dey, Gerstgrasser, Rafailov, Donoho, Koyejo. *Collapse or Thrive: Perils and Promises of Synthetic Data in a Self-Generating World.* ICML 2025, PMLR 267 | ✓ | Replace → collapse; accumulate → stable; fixed-size → slow degradation | Credit for the replay design | Cite; report each round's regime |
| 17 | Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian, Dacheng Tao. *A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops.* ICLR 2025. arXiv:2502.18865 | ✓ | A constant real-data share guarantees convergence ("recursive stability") | Credit | Cite |
| 18 | Bingji Yi, Qiyuan Liu, Yuwei Cheng, Haifeng Xu. *Escaping Model Collapse via Synthetic Data Verification: Near-term Improvements and Long-term Convergence.* arXiv:2510.16657 | ✓ | Verifier-guided retraining helps short-term but converges to the verifier's "knowledge center" | Medium–high: predicts that validation-anchored acceptance may plateau or bias toward the validation distribution | Cite; report gains per round and audit on data disjoint from validation |
| 19 | Sina Alemohammad, Li Chen, Richard G. Baraniuk, Zhangyang Wang. *Not All Synthetic Data Is Yours to Learn From.* arXiv:2605.31126 | ✓ | A model's own generated data is the most effective source; same-lineage beats a stronger, differently trained teacher | Medium (predicts the teacher-control outcome) | Cite |

Further verified citations, low closeness:
- Feng, Dohmatob, Yang, Charton, Kempe, *Beyond Model Collapse*, arXiv:2406.07515 (venue unverified)
- Dohmatob et al., *Strong Model Collapse*, arXiv:2410.04840
- Schaeffer et al., *Position: Model Collapse Does Not Mean What You Think*, arXiv:2503.03150
- Kang et al., *Demystifying Synthetic Data in LLM Pre-training*, EMNLP 2025
- Knowledge-Instruct, arXiv:2504.05571
- KGGen, arXiv:2502.09956
- Synthetic Bootstrapped Pretraining, arXiv:2509.15248


**Proposed wording (M5 / H-G):**
> Self-improvement from self-authored knowledge is not new in general. Neither are ontology-guided self-training (OntoTune, Evontree), KG- or loss-targeted synthetic data (GraphGen, SoG), filtering synthetic data by its first-order effect on held-out loss (Montessori-Instruct, OptimSyn; cf. TracIn, LESS, In-Run Data Shapley), or a small self-author beating a frontier teacher (SEAL, Active Reading). Our claim: a small LM authors typed ontology frames for its own poorly modelled corpus spans; each edge is admitted only if its held-out gradient utility has a bootstrap lower bound > 0; the frames are consumed through a non-textual compositional channel; and at matched compute this beats continued pretraining, EntiGraph- and SPA/QA-style synthetic CPT, self-authored unstructured notes, and the same verified edges given as text.

## 6. Proposed changes for the orchestrator (not applied)

### 6.1 Wording (proposal §7 / related-work.md summary table)

| Mechanism | Current allowed wording | Proposed |
|---|---|---|
| M1 | query-conditioned weighting of VSA-bound ontology edges with zero per-concept parameters that recovers the static bundle as τ → ∞ | … whose τ → ∞ limit is the static HRR bundle (HRRBERT); contrast CokeBERT and Dasigi et al. (text-conditioned attention over KG/ontology neighbours with per-entity vectors) |
| M3 | first-order usage-level split test … from optimizer state, calibrated by a permutation null | prefix with the concession to Parameter Differentiation / Recon / Lin et al. 2026; credit signed splitting (Wu et al.) next to Liu et al., G-means, LBG |
| M4 | compositional, growable span channel … with a last-subtoken rule | … rows built **by fixed binding with no per-concept parameters**, **added in place at the last subtoken of a prefix-causally linked span during LM training**; ConceptFormer is the nearest prior work |
| H-A | … whether structure beats a hashed lookup memory | … at matched parameters, injection site and gate (unless C1h-E is run); distinguish from the "token tax" |
| M5 | … beating compute-matched and EntiGraph-style controls | extended "not novel" list and controls of §5 |
| H-E | structure-only zero-shot rows for unseen ontology nodes | unchanged; add MKGL (inductive KG-composed LLM tokens) and ConceptFormer to the comparison set |

### 6.2 New baselines and controls

| Name (proposed) | Experiment | Definition | Answers | Cost |
|---|---|---|---|---|
| `translation` operator | E1, E2, E4-C8 | `T_r(x) = x + r_r` with relation-only keys (CokeBERT/TransE-style) | Is binding needed, or is typed attention enough? | one more operator family; negligible |
| **C1h-E** | E4 D4.9 (+ D4.1 if C1h is close to the best channel) | Engram-faithful hashed memory at matched parameters: multi-head hashing of 2/3-gram suffixes, tokenizer compression, context-aware gate, layer-2 insertion; span-restricted and dense variants | Does ontology structure beat a strong lookup memory? | implementation (new channel mode) + ≈ 6 runs at 50M |
| **C1m** | E4 D4.9 | Mean of the span's subtoken embeddings at the same site and gate (NGM-style, zero parameters) | Is span pooling alone the gain? | trivial |
| C1t (optional) | E4.6 (CPT) | LM-derived span vector (frozen teacher's hidden state at span end), same site and gate (Memory Grafting / SCONE-style) | Ontology content vs any informative span vector | moderate |
| Alias stratum | E4 eval | Spans whose surface form is a non-canonical alias of the concept | Where linking should beat hashing by construction | evaluation only |
| Threshold "differentiation" split | E0.2, E3 | Split when the maximum pairwise negative cosine of per-usage momenta > τ, then bipartition (no null) | What the permutation null adds over Wang & Zhang / Lin et al. | CPU (E0.2) + one E3 policy |
| K gated slots | E3 (optional) | K gated copies per dictionary vector at matched parameters (MoME-style) | Split test vs fixed multi-sense capacity | one E3 policy |
| **B1** self-authored unstructured notes | E7.2 | The same host writes free-text notes/implications for the same spans, through the same held-out-utility filter, at matched tokens (Active Reading / SEAL-style, no RL) | Structure + channel vs self-authorship + verification | ≈ one E7 condition |
| **B2** SPA / synthetic-QA CPT | E7.2 | Synthetic QA / SPA augmentation at matched tokens | Stronger synthetic-CPT control than EntiGraph alone | ≈ one E7 condition |
| **B3** verbalized-frames CPT | E7.2 | The same verified edges rendered as text, trained by ordinary CPT, no channel | Isolates the channel | ≈ one E7 condition |
| Rule-consistency verifier, random filter (optional) | E7.2 | Evontree-style rule verifier; random acceptance at matched rate | What gradient-utility verification buys | small |

Already covered by existing conditions (no change needed):
- **C1s** (shuffled frames, same atoms) answers the TN-gram / FactorEngram "shared atoms, not ontology" argument.
- The E0.2/E3 **random** split policy answers the *Expand Neurons, Not Parameters* "random splits suffice" argument.

### 6.3 Other items

- Ask the author about OpenReview `tfyLS1cB5W` (§3) to avoid double-citing HRRBERT.
- Keep the keyword alerts: (i) "gradient conflict" + split / duplicate + embedding / dictionary; (ii) VSA / HRR / hyperdimensional + language model + ontology; (iii) Engram follow-ups; (iv) self-authored KG / ontology + continued pretraining.
- Re-run this check within 30 days of submission.

## 7. Search record

Per-area query lists, about 45 queries each, were kept in the four search-thread reports of 2026-10-02. They included, among others:
- "gradient conflict split shared embedding", "neuron splitting gradient disagreement", "Adam second moment SNR growth criterion", "parameter differentiation follow-ups";
- "holographic reduced representations attention knowledge graph language model", "VSA transformer embedding ontology", "HRRBERT Hu Tripp", "dynamic knowledge context selection" (this one found CokeBERT);
- "Engram conditional memory follow-up 2026", "scaling n-gram embedding layer", "entity embeddings decoder-only KG injection", "multi-token words harder to learn", "Dual-Route induction";
- "synthetic continued pretraining knowledge graph", "SEAL self-edits", "Active Reading", "influence data selection self-generated synthetic held-out", "Collapse or Thrive", "LLMs4OL 2025".

Sources used:
- arXiv abs/HTML pages and the arXiv API
- Semantic Scholar (HRRBERT: 0 citations listed)
- PMLR, NeurIPS virtual pages, mlanthology
- tib-op.org (LLMs4OL 2025)

Blocked sources:
- dblp (bot wall)
- OpenReview (bot check)
