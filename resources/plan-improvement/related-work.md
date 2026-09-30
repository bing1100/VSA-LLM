# Related work and novelty positioning (task C1)

**Status:** positioning note, 2026-09-30. Scope: prior work for mechanisms M1–M5 and hypotheses H-A, H-E and H-G of [proposal.md](proposal.md) / [formulation.md](formulation.md). It adds to, and does not repeat, [../vsa-understanding/02-literature-review.md](../vsa-understanding/02-literature-review.md) and [references.md](../vsa-understanding/references.md).

**How citations were verified.** Every arXiv ID below was resolved against the arXiv API on 2026-09-30 and its title, authors and year were matched. Venues come from the arXiv comment field, ACL Anthology, NeurIPS/ICLR proceedings pages, CEUR-WS or publisher pages. A venue marked **†** is the commonly cited one and was *not* re-checked in this search. "et al." means more than two authors. This is a targeted search, not a systematic review. Absence of a hit is weak evidence, especially for 2025–2026 preprints.

**Corrections to the existing dossier.**
- `references.md` #27 gives the wrong authors for CoLLEGe. The correct entry is Teehan, Lake & Ren, *CoLLEGe: Concept Embedding Generation for Large Language Models*, COLM 2024, arXiv:2403.15362.
- The HRRBERT paper is published as Hu, Yu, Tuinstra, Rezai, Bokadia, DiMaio, Fortin, Vartian & Tripp, *Encoding Medical Ontologies With Holographic Reduced Representations for Transformers*, in the joint proceedings of KiL 2024 and DL4KG 2024 (co-located with KDD 2024), CEUR-WS Vol-3894, paper 13. Cite that record rather than the local manuscript.
- "Attention as Binding" (arXiv:2512.14709) is a single-author position paper (Dhayalkar, 2025). It argues a perspective and does not supply experimental evidence.

---

## 1. Knowledge-injected LMs with entity embeddings (closest prior art to M4)

**Key prior work.**
- **ERNIE** (Zhang et al., *ERNIE: Enhanced Language Representation with Informative Entities*, ACL 2019, arXiv:1905.07129) fuses pretrained TransE entity vectors into BERT at aligned mention tokens. The table has one vector per entity. It should not be confused with Baidu's ERNIE (Sun et al., arXiv:1904.09223), which uses entity-level masking and no entity table.
- **KnowBERT** (Peters et al., *Knowledge Enhanced Contextual Word Representations*, EMNLP 2019, arXiv:1909.04164) takes a mention-span representation from a middle layer, attends over candidate entity embeddings (including WordNet synsets), and adds the weighted entity vector back into the residual stream. This is context-conditioned attention over knowledge vectors, at the level of candidates rather than edges.
- **LUKE** (Yamada et al., EMNLP 2020, arXiv:2010.01057) treats entities as extra input tokens with a large per-entity table and entity-aware self-attention.
- **KEPLER** (Wang et al., TACL 2021†, arXiv:1911.06136) computes entity embeddings by encoding entity *descriptions* with the LM, trained jointly with a KE loss and MLM. Rows therefore need no per-entity parameters, but they are built from text, not from ontology edges.
- **K-BERT** (Liu et al., AAAI 2020†, arXiv:1909.07606) injects triples into the sentence as a tree, with soft positions and a visibility matrix.
- **E-BERT** (Poerner, Waltinger & Schütze, arXiv:1911.03681; Findings of EMNLP 2020†) aligns Wikipedia2Vec vectors to the wordpiece space. **Entities as Experts** (Févry et al., arXiv:2004.07202; EMNLP 2020†) and **Mention Memory/TOME** (de Jong et al., arXiv:2110.06176; ICLR 2022†) use per-entity or per-mention memories accessed from mention spans.
- **KALM** (Rosset et al., *Knowledge-Aware Language Model Pretraining*, arXiv:2007.00655, preprint) is a GPT-2 drop-in *causal* LM. It feeds entity signals at the input through an entity-extended tokenizer and adds an entity-prediction loss at the output. This is the closest analogue to M4 injection plus the §4.3 `L_sem` head, with a free per-entity table.
- Graph-joint pretraining: **K-Adapter** (arXiv:2002.01808) and **DRAGON** (Yasunaga et al., NeurIPS 2022, arXiv:2210.09338).
- Clinical: **UmlsBERT** (Michalopoulos et al., NAACL 2021, arXiv:2010.10391) adds a learned *semantic-group* embedding to the input embedding of UMLS-linked words, a type-level shared vector added like a position embedding. **SapBERT** (Liu et al., NAACL 2021, arXiv:2010.11784) and **KeBioLM** (Yuan et al., BioNLP 2021, arXiv:2104.10344) are further clinical examples.

**Prior work that already builds compositional entity rows.**
- **Bootleg** (Orr et al., *Bootleg: Chasing the Tail with Self-Supervised Named Entity Disambiguation*, CIDR 2021, arXiv:2010.10363) represents a candidate as `MLP([u_e, AddAttn(type embs), AddAttn(relation embs)])`. It masks the per-entity vector `u_e` with probability inversely related to popularity, and reports that type-only and relation-only variants reach 3.3× the prior state of the art on unseen entities using 1% of the space. That is a compositional, near-zero-per-entity row that targets the tail. Bootleg is an NED system, not an LM channel, and its relations are unbound relation *types*, not relation–filler pairs.
- **NodePiece** (Galkin et al., ICLR 2022, arXiv:2106.12144) builds any entity, including unseen ones, from a fixed vocabulary of anchor nodes plus relation context, with roughly 10× fewer parameters. It is explicitly framed as "subword tokenization for KGs". It works inside KG models, not LMs.
- **GRAM** (Choi et al., arXiv:1611.07012; KDD 2017†) represents a medical code as an attention-weighted combination of its ontology ancestors. It helps rarely observed diseases, the same claim HRRBERT makes, and the attention is conditioned on the concept rather than on the sentence.
- **HRRBERT** (above) composes SNOMED CT codes with HRR in a BERT over code sequences.

**Established.** Entity/concept signals injected into LMs improve knowledge probes and rare-entity behavior, including in causal LMs (KALM). Composing rows from types, relations, anchors or ancestors helps the tail and can remove per-entity parameters (Bootleg, NodePiece, GRAM, HRRBERT).

**Gap.** No work found combines all of the following:
- (a) rows built from *bound* relation–filler edges of a curated ontology, with an operator ablation;
- (b) injection into a causal LM trained on running text, jointly or by continued pretraining;
- (c) a leakage-safe last-subtoken rule;
- (d) growth of the dictionary;
- (e) matched-parameter comparison against a free per-concept table *and* against type/relation-only composition.

**Verdict: claim must be narrowed to** "a compositional, growable span channel whose concept rows are built from ontology relation–filler edges with zero per-concept parameters, for causal LMs." The proposal may not claim to be the first compositional or zero-parameter entity rows. Bootleg, NodePiece, GRAM, KEPLER and HRRBERT must be cited as such.

## 2. Contextual composition, attention as binding, TPRs and fast weights (M1)

**Key prior work.**
- Binding algebra: **Tensor product representations** (Smolensky, *Tensor product variable binding and the representation of symbolic structures in connectionist systems*, Artificial Intelligence 46(1–2):159–216, 1990). Also *Neurocompositional computing* (Smolensky et al., AI Magazine, arXiv:2205.01128).
- Transformer with TPR roles: **TP-Transformer** (Schlag, Smolensky et al., arXiv:1910.06611).
- Fast weights, the direct link between attention and binding: Schlag & Schmidhuber, *Learning to Reason with Third-Order Tensor Products* (arXiv:1811.12143, NeurIPS 2018†); Ba et al., *Using Fast Weights to Attend to the Recent Past* (arXiv:1610.06258); Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* (arXiv:2102.11174, ICML 2021†). Linear attention is outer-product binding plus query unbinding.
- Attention as associative memory: **Hopfield Networks is All You Need** (Ramsauer et al., arXiv:2008.02217, ICLR 2021†). Key–value memories: Sukhbaatar et al., NIPS 2015, arXiv:1503.08895; Miller et al., arXiv:1606.03126.
- HRR and attention: **Hrrformer** (Alam et al., *Recasting Self-Attention with Holographic Reduced Representations*, ICML 2023, arXiv:2305.19534). **Associative LSTM** (Danihelka et al., ICML 2016, arXiv:1602.03032). Dhayalkar 2025 (position, above).
- Context-conditioned composition from lexical knowledge, the closest to M1's *function*:
  - **Ontology-aware token embeddings** (Dasigi, Ammar, Dyer & Hovy, ACL 2017, arXiv:1705.02925): a token's embedding is a context-dependent distribution over its WordNet synsets and their hypernyms, trained jointly.
  - **SE-WRL** (Niu, Xie, Liu & Sun, *Improved Word Representation Learning with Sememes*, ACL 2017, P17-1187): the SAT model uses context words to attend over the sememe-composed senses of the target word.
  - **SDLM** (Gu et al., EMNLP 2018, arXiv:1810.12387): sememe experts inside an LM.
  - **Dynamic meta-embeddings** (Kiela et al., EMNLP 2018, arXiv:1804.07983).
  - **KnowBERT** (§1).

**Established.**
- "Uniform attention = superposition, query attention = soft unbinding" is a known reading. Fast-weight and linear-attention work proves it for outer-product binding, and Hrrformer implements attention with HRR.
- Context-dependent weighting of ontology or sememe components to form a token representation was shown in 2017, including sense selection.

**Gap.** No work found attends over *VSA-bound role–filler edges* of a concept's frame with a query that (i) contains no per-concept parameters, (ii) reduces exactly to the static HRR bundle at `τ → ∞`, and (iii) keeps the relation-probe unbinding readout as a special case in the same module.

**Verdict: claim must be narrowed to** "a query-conditioned re-weighting of bound ontology edges that contains the static VSA bundle as a limit and needs no per-concept parameters." Drop "first contextual VSA composition" and "unifying bundling, unbinding and attention" as novelty claims. The unification is a known observation and should be presented as design motivation, citing Smolensky, Schlag et al. 2021 and Ramsauer et al.

## 3. Factorized and compositional embeddings (M2 and the compression claim)

**Key prior work.**
- Low-rank adaptation and factorization: **LoRA** (Hu et al., arXiv:2106.09685, ICLR 2022†) and **ALBERT**'s factorized embedding `V×E · E×H` (Lan et al., arXiv:1909.11942, ICLR 2020†).
- Compositional tables for recommendation: the **quotient–remainder trick** (Shi et al., *Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems*, arXiv:1909.02107, KDD 2020†); **TT-Rec** (Yin et al., MLSys 2021, arXiv:2101.11714); **Tensorized embedding layers** (Hrinchuk et al., arXiv:1901.10787).
- Hashing: **feature hashing** (Weinberger et al., ICML 2009, arXiv:0902.2206); **hash embeddings** (Svenstrup, Hansen & Winther, arXiv:1709.03933, NIPS 2017†); **DHE**, a table-free embedding (Kang et al., KDD 2021, arXiv:2010.10784); **ROBE** (Desai et al., arXiv:2108.02191).
- Word-embedding compression: **Compositional code learning** (Shu & Nakayama, arXiv:1711.01068, ICLR 2018†) and **adaptive input representations** (Baevski & Auli, arXiv:1809.10853, ICLR 2019).
- Graph-side: NodePiece (§1).
- For LMs specifically, **T-FREE** (Deiseroth et al., EMNLP 2024, arXiv:2406.19223) embeds words through sparse character-trigram activations and reports more than 85% parameter reduction in the embedding and head layers.

**Established.**
- Low-rank or factorized weighting is standard.
- Composing rows from shared sub-tables gives large, well-characterized memory savings, from tens of × to about 1000×, with modest quality loss.
- Inductive rows for unseen items exist (NodePiece, DHE, T-FREE).

**Gap.** What remains new is the *ontology-supported* sparsity pattern, `W_{i,e} = ⟨u_i, k_e⟩ M_{i,e}`, with a concept factor *induced from the frame* so that it transfers to unseen nodes. It is useful only if it beats generic compressors at equal bytes.

**Verdict:**
- Factorization / "LoRA-style mapping" as such: **claim not novel**.
- M2: **claim must be narrowed to** "ontology-induced, transferable edge weighting."
- Compression (H-B, E4c): **claim must be narrowed to** "fewer bytes than the ALBERT-factorized, QR/hash-compositional, TT and PTQ tables at matched quality on the linked stratum." Without those baselines no compression claim is allowed.

## 4. Multi-sense induction, growing and splitting, gradient-conflict statistics (M3)

**Key prior work.**
- Multi-sense embeddings:
  - **MSSG/NP-MSSG** (Neelakantan, Shankar, Passos & McCallum, *Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space*, EMNLP 2014, arXiv:1504.06654) creates a new sense when a context is far from every existing sense cluster.
  - **AdaGram** (Bartunov, Kondrashkin, Osokin & Vetrov, *Breaking Sticks and Ambiguities with Adaptive Skip-gram*, arXiv:1502.07257, AISTATS 2016†) is a Dirichlet-process prior over senses.
  - Li & Jurafsky, *Do Multi-Sense Embeddings Improve Natural Language Understanding?* (arXiv:1506.01070, EMNLP 2015†) is an important caution: gains often vanish against capacity-matched single-sense baselines.
- Growing networks:
  - **Net2Net** (Chen, Goodfellow & Shlens, arXiv:1511.05641, ICLR 2016†): function-preserving widening by copying units.
  - **Splitting Steepest Descent** (Liu, Wu & Wang, NeurIPS 2019, arXiv:1910.02366): splits a neuron into copies offset along the minimum eigenvector of a "splitting matrix", using a second-order criterion.
  - **Firefly** (Wu, Liu, Stone & Liu, NeurIPS 2020, arXiv:2102.08574): grows networks by first-order Taylor selection over candidate units.
  - **GradMax** (Evci et al., ICLR 2022, arXiv:2201.05125): initializes new units to maximize gradient norm.
  - NORTH* (Maile et al., AutoML 2022, arXiv:2202.08539).
  - **Dynamically Expandable Networks** (Yoon et al., arXiv:1708.01547): split or duplicate units that drift.
  - **bert2BERT** (arXiv:2110.07143) and **Sparse Upcycling** (Komatsuzaki et al., arXiv:2212.05055): copying units or experts in LMs.
- Split tests from clustering and quantization:
  - **LBG** (Linde, Buzo & Gray, *An Algorithm for Vector Quantizer Design*, IEEE Trans. Communications 28(1):84–95, 1980): codebook growth by perturbing codewords and re-partitioning.
  - **G-means** (Hamerly & Elkan, *Learning the k in k-means*, NIPS 2003): splits a cluster when a statistical test on the data projected onto its principal split direction rejects the null.
  - **DP-means** (Kulis & Jordan, arXiv:1111.0352, ICML 2012).
- Gradient statistics:
  - **gradient noise scale** (McCandlish, Kaplan & Amodei, arXiv:1812.06162);
  - **PCGrad** (Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020, arXiv:2001.06782), which detects conflict as negative cosine between task gradients;
  - **GSNR** (Liu et al., ICLR 2020, arXiv:2001.07384), per-parameter gradient signal-to-noise;
  - **Coherent Gradients** (Chatterjee, ICLR 2020, arXiv:2002.10657).

**Established.**
- Splitting units into offset copies (Liu et al.; LBG) is known.
- Deciding *whether* to split with a statistical test along a principal direction (G-means) is known.
- Measuring gradient conflict between groups (PCGrad) and per-parameter gradient SNR (GSNR) is known.
- Sense inventories can be induced nonparametrically.

**Gap.**
- No work found splits a *shared dictionary vector* using per-usage *momentum* conflict, `G(v) = Σ_u |⟨m^{(u)}, v⟩| − |Σ_u ⟨m^{(u)}, v⟩|`, with a permutation null and usage partitioning, inside a jointly trained LM.
- Unlike Liu et al., M3's children serve *disjoint usages*, so the gain appears at first order. That makes it a gradient-space G-means rather than a neuron split.

**Verdict: claim must be narrowed to** "a first-order, usage-level split test for shared VSA dictionary vectors, read from optimizer state and calibrated by a permutation null." The ±ε offset and the partition-by-sign rule should be credited to LBG, G-means and Liu et al., not presented as new. The observation that "momentum ≈ 0 with high |g|" is Adam's per-coordinate SNR is presented correctly in formulation §3.2, and GSNR should be cited there.

## 5. Multi-token words and tokenizer-free modelling (framing of the "syntax → semantics tax")

**Key prior work.**
- Tokenizer-side remedies: **SuperBPE** (Liu et al., COLM 2025, arXiv:2503.13423) uses superword tokens and reports +4.0% average over BPE at 8B with 27% less inference compute. **T-FREE** (§3).
- Byte-level and pooled models: **MegaByte** (Yu et al., arXiv:2305.07185, NeurIPS 2023†); **Byte Latent Transformer** (Pagnoni et al., ACL 2025, arXiv:2412.09871); **dynamic token pooling** (Nawrot et al., ACL 2023, arXiv:2211.09761); **word-pooled tokenization** (Thawani et al., Findings EMNLP 2023, arXiv:2310.11628); **hierarchical byte/word transformers** (Neitemeier et al., arXiv:2501.10322).
- Adapting tokenization of pretrained models: **Zero-Shot Tokenizer Transfer** (Minixhofer, Ponti & Vulić, NeurIPS 2024, arXiv:2405.07883) and **dynamic tokenization** (Feher, Vulić & Minixhofer, arXiv:2411.18553, ACL 2025 per ACL Anthology).
- Input-side n-gram memories, the closest *architectural* competitors to M4:
  - **Over-Tokenized Transformer** (Huang et al., ICML 2025, arXiv:2501.16975): hashed multi-gram input vocabularies, with loss log-linear in input-vocabulary size.
  - **Engram** (Cheng et al., *Conditional Memory via Scalable Lookup*, arXiv:2601.07372, 2026): hashed suffix N-gram embeddings injected into the residual stream at chosen layers, with a hidden-state-conditioned gate, beating iso-parameter MoE. Its analysis says the memory "relieves the backbone's early layers from static reconstruction", which is the tax thesis stated for n-grams.
- Detokenization inside LMs:
  - **Token erasure** (Feucht, Atkinson, Wallace & Bau, arXiv:2406.20086, EMNLP 2024†): last-token representations of multi-token words and entities "erase" subtoken identity in early layers.
  - **From Tokens to Words** (Kaplan, Oren, Reif & Schwartz, ICLR 2025, arXiv:2410.05864): words are assembled at their *last* token in early and middle layers, and these representations can be fed back as new vocabulary without fine-tuning.
  - "Detokenization" neurons and stages: Gurnee et al. (arXiv:2305.01610) and Lad et al. (arXiv:2406.19384).

**Established.**
- LMs spend early-layer computation assembling multi-token words at the last subtoken. This independently supports M4's last-subtoken injection rule.
- Adding a lookup-based signal for multi-token units at the input helps at scale (Over-Tokenized, Engram).
- Changing the tokenizer also helps (SuperBPE, BLT).

**Gap.**
- The "tax" is not a new idea. What is new is to measure it per ontology-linked stratum and ask whether a *structured, compositional* signal beats an *unstructured* hashed n-gram signal at equal parameters.
- Engram and Over-Tokenized are exactly the "span-boundary/lookup" explanation that proposal §8 lists as the first risk.

**Verdict: claim must be narrowed to** "a measurement of the syntax → semantics tax on ontology-linked multi-token spans, and a test of whether ontology structure beats hashed n-gram lookup at matched parameters." The framing should cite Feucht et al., Kaplan et al. and Engram as prior evidence.

## 6. Knowledge capacity and rare-word learning in small LMs (H-A)

**Key prior work.**
- **Physics of Language Models 3.3** (Allen-Zhu & Li, *Knowledge Capacity Scaling Laws*, ICLR 2025, arXiv:2404.05405) reports about 2 bits of factual knowledge per parameter, holding under int8 quantization. Its abstract also reports that prepending a domain tag to training data increases capacity. Companion paper: Part 3.1, arXiv:2309.14316.
- Long-tail knowledge:
  - **LLMs struggle to learn long-tail knowledge** (Kandpal et al., ICML 2023, arXiv:2211.08411): QA accuracy tracks pretraining document counts.
  - **PopQA** (Mallen et al., ACL 2023, arXiv:2212.10511).
- Rare words:
  - **Word acquisition in neural LMs** (Chang & Bergen, TACL, arXiv:2110.02406): per-word learning curves by frequency.
  - **Representation degeneration** of rare-token embeddings (Gao et al., ICLR 2019, arXiv:1907.12009) and **FRAGE** (Gong et al., NeurIPS 2018, arXiv:1809.06858).
  - **Rare words in BERT / attentive mimicking** (Schick & Schütze, AAAI 2020, arXiv:1904.06707; NAACL 2019, arXiv:1904.01617) and **BERTRAM** (ACL 2020, arXiv:1910.07181).
  - Misra & Mahowald, *Language Models Learn Rare Phenomena from Less Rare Phenomena* (arXiv:2403.19827, 2024): LMs learn rare constructions from related, more frequent ones.

**Established.**
- Knowledge per parameter is bounded. Rare items are learned late, embedded poorly, and answered poorly.
- A metadata tag alone can raise capacity. This matters for C1 in E4: random span vectors are a tag and may help for that reason alone.

**Gap.** No work found tests whether an *externally structured* channel shifts the tokens-to-loss curve for rare *multi-token* concepts in small LMs at matched non-embedding parameters.

**Verdict: claim allowed**, provided C1 (random fixed span vectors) and an Engram/hashed-span control are both beaten. Allen-Zhu & Li's domain-tag result is a concrete reason to expect C1 to be non-trivial.

## 7. Zero-shot embeddings for new concepts (H-E)

**Key prior work.**
- **CoLLEGe** (Teehan, Lake & Ren, COLM 2024, arXiv:2403.15362): a meta-learned generator of new-concept embeddings from a few sentences or definitions.
- **Learning to compute word embeddings on the fly** (Bahdanau et al., arXiv:1706.00286): embeddings from dictionary definitions.
- **Definition modeling** (Noraset, Liang, Birnbaum & Downey, AAAI 2017, arXiv:1612.00394).
- **Mimick** (Pinter, Guthrie & Eisenstein, EMNLP 2017, arXiv:1707.06961).
- **High-risk learning** (Herbelot & Baroni, EMNLP 2017, arXiv:1707.06556).
- **À la carte embedding** (Khodak et al., ACL 2018, arXiv:1805.05388).
- Attentive mimicking and BERTRAM (§6); ZeTT (§5).
- The vocabulary-transfer methods already in `references.md` (WECHSEL, FOCUS, Token Distillation, Concept Tokens).
- On the graph side, NodePiece and Bootleg (§1) give inductive rows for unseen entities from structure alone.

**Established.** Generating useful rows for unseen words from definitions, contexts or subword strings is mature. Generating them from graph structure alone is established in KG models and NED.

**Gap.** What remains new is zero-parameter rows for unseen *ontology nodes*, using no definitions and no contexts, inside a causal LM, evaluated on node-disjoint, alias-controlled splits.

**Verdict: claim must be narrowed to** "structure-only zero-shot rows (no text evidence)." Head-to-head claims against text-evidence generators are allowed only on tracks where definitions exist and those baselines are run.

## 8. Self-authored ontologies (M5, H-G)

**Key prior work.**
- Classical ontology learning:
  - **Hearst patterns** (Hearst, *Automatic Acquisition of Hyponyms from Large Text Corpora*, COLING 1992, C92-2082);
  - **OntoLearn** (Navigli & Velardi, Computational Linguistics 30(2):151–179, 2004);
  - **Text2Onto** (Cimiano & Völker, NLDB 2005).
- LM-based KG and ontology construction:
  - **COMET** (Bosselut et al., ACL 2019, arXiv:1906.05317);
  - **Symbolic Knowledge Distillation** (West et al., NAACL 2022, arXiv:2110.07178): a large LM authors a commonsense KG, a trained critic filters it, and a much smaller student trained on it surpasses the teacher on that task;
  - **BertNet** (Hao et al., Findings ACL 2023, arXiv:2206.14268);
  - **LMCRAWL** (Cohen et al., Findings EACL 2023, arXiv:2301.12810);
  - **LLMs4OL** (Babaei Giglou, D'Souza & Auer, ISWC 2023, arXiv:2307.16648);
  - **OLLM** (Lo, Jiang, Li & Jamnik, *End-to-End Ontology Learning with Large Language Models*, NeurIPS 2024, arXiv:2410.23584).
- Self-improvement:
  - **STaR** (Zelikman et al., arXiv:2203.14465, NeurIPS 2022†);
  - **Self-Instruct** (Wang et al., ACL 2023, arXiv:2212.10560);
  - **Self-Rewarding LMs** (Yuan et al., ICML 2024, arXiv:2401.10020);
  - **ReST-EM** (Singh et al., TMLR, arXiv:2312.06585).
- Knowledge acquisition from a small corpus: **Synthetic continued pretraining / EntiGraph** (Yang, Band, Li, Candès & Hashimoto, ICLR 2025, arXiv:2409.07431) extracts entities from source documents and generates text relating them, then continues pretraining. This is the closest prior art to M5's goal, done with text rather than with frames.
- Utility scoring by gradient inner products: **TracIn** (Pruthi et al., NeurIPS 2020, arXiv:2002.08484) and **LESS** (Xia et al., ICML 2024, arXiv:2402.04333).
- Model collapse:
  - Shumailov et al., *AI models collapse when trained on recursively generated data*, Nature 631:755–759, 2024 (preprint arXiv:2305.17493);
  - Alemohammad et al., *Self-Consuming Generative Models Go MAD* (arXiv:2307.01850, ICLR 2024†);
  - Gerstgrasser et al., *Is Model Collapse Inevitable?* (arXiv:2404.01413, reported as COLM 2024 but not independently confirmed): accumulating real data alongside synthetic data avoids collapse.

**Established.**
- LMs can author KGs and ontologies, and a filtered, machine-authored KG can train a model that beats its teacher (Symbolic KD).
- Self-generated data can bootstrap reasoning (STaR, ReST-EM).
- Self-generated entity-relation text improves knowledge acquisition from small corpora (EntiGraph).
- First-order gradient inner products are a standard utility or influence score (TracIn, LESS).
- Replacing real data causes collapse, while accumulating it does not.

**Gap.** No work found has the *same* model author typed frames, accept edges by held-out first-order utility, feed them through a compositional *embedding channel* (not as text), and compare against compute-matched continued pretraining **and** against self-generated text (EntiGraph-style).

**Verdict: claim must be narrowed to** "self-authored ontology frames, verified by held-out gradient utility, consumed through a compositional channel." The utility test should be credited as a TracIn/LESS-style first-order score, and the replay design as following Gerstgrasser et al. "Self-improvement from self-authored knowledge" in general is **not novel**.

## 9. VSA/HRR in neural models more broadly (positioning only)

**Key prior work.**
- HRR in LMs and networks: HRRBERT (see corrections at the top); **Learning with HRR** (Ganesan et al., NeurIPS 2021, arXiv:2109.02157); **2D HRR for secure CNN deployment** (Alam et al., ICML 2022, arXiv:2206.05893); Hrrformer and Associative LSTM (§2); HolE (Nickel, Rosasco & Poggio, arXiv:1510.04935, AAAI 2016).
- Analyses and probes: **GPT-2 through the lens of VSAs** (Knittel et al., ATTRIB workshop at NeurIPS 2024, arXiv:2412.07947).
- Surveys: Kleyko et al. Part I (ACM Computing Surveys 55(6), arXiv:2111.06077) and Part II (already in `references.md`).

**Established.**
- HRR binding can be trained end to end in deep networks (Ganesan et al.; Hrrformer).
- HRR-composed ontology embeddings help rare codes in a joint-trained BERT (HRRBERT).

**Gap.** The listed papers either use HRR for efficiency or privacy (Hrrformer, 2D HRR) or for analysis (Knittel et al.). None uses it to compose token-level concept rows for *causal* LMs on text. HRRBERT did this for encoders over code sequences.

**Verdict: claim allowed:** "extends HRR-composed ontology embeddings (HRRBERT) from code-sequence encoders to causal text LMs, with the operator treated as an ablation."

---

## Summary

| Mechanism / hypothesis | Closest prior work | Allowed claim wording |
|---|---|---|
| M1 attentive composition | Dasigi et al. 2017; SE-WRL (Niu et al. 2017); KnowBERT; Schlag et al. 2021; Hrrformer | "query-conditioned weighting of VSA-bound ontology edges with zero per-concept parameters that recovers the static bundle as `τ → ∞`"; *not* "first contextual composition" or "unifies attention and binding" |
| M2 factored mapping | LoRA; ALBERT; NodePiece; QR trick; TT-Rec; DHE; T-FREE | "ontology-induced, transferable rank-`k` edge weighting"; compression only against equal-bytes factorized/hashed/TT/PTQ tables |
| M3 developmental dictionary | Splitting Steepest Descent; Firefly/GradMax; G-means; LBG; AdaGram/NP-MSSG; PCGrad; GSNR | "first-order usage-level split test for shared dictionary vectors from optimizer state, calibrated by a permutation null" |
| M4 span channel | KALM; Bootleg; UmlsBERT; ERNIE/KnowBERT/LUKE; Engram; Over-Tokenized; Kaplan et al. 2025 | "compositional, growable span channel for causal LMs with ontology-built rows and a last-subtoken rule"; not "first compositional entity rows" |
| M5 self-authoring | EntiGraph; Symbolic KD; OLLM/LLMs4OL; STaR; TracIn/LESS; Shumailov et al.; Gerstgrasser et al. | "self-authored frames verified by held-out gradient utility, consumed through the channel, beating compute-matched and EntiGraph-style controls" |
| H-A tax | Feucht et al.; Kaplan et al.; Engram; Allen-Zhu & Li 3.3; Kandpal et al. | "measures the tax on ontology-linked multi-token strata and whether structure beats hashed lookup" |
| H-E zero-shot | CoLLEGe; Bahdanau et al. 2017; à la carte; BERTRAM; NodePiece | "structure-only zero-shot rows for unseen ontology nodes" |
| VSA in LMs | HRRBERT; Hrrformer; Ganesan et al. 2021 | "extends HRRBERT to causal text LMs; operator is an ablation" |

## Baselines and changes the experiments should add

1. **Hashed n-gram / Engram-style span memory (E4, new control).** Give the same parameter budget to a hashed table keyed by the linked span's subtoken n-gram, injected at the last subtoken with the same gate. This separates "structured composition" from "any extra lookup memory for multi-token units" more sharply than C1 and C2. If it ties the channel, the result is a lookup result.
2. **Type/relation-only composition, no binding (E4, M1/M4 ablation).** Add a Bootleg/UmlsBERT-style row: additive attention over the concept's relation types and semantic types, without fillers or binding. It is cheaper than `untyped` and answers whether filler binding matters.
3. **Context attention over free sense vectors (E1, M1 baseline).** Dasigi- or KnowBERT-style: context attends over per-sense learned vectors. This isolates what bound-edge keys add over sense-level attention.
4. **GRAM (E8 clinical).** Ancestor-attention code embeddings are the standard clinical baseline for rare codes next to HRRBERT.
5. **Compression baselines (E4c/E4.5).** Add ALBERT-factorized embeddings, QR/hash embeddings, a TT or tensorized table, and INT4/INT8 PTQ of the free table, all at equal bytes.
6. **Split-test alternatives (E0.2/E3).**
   - G-means-style test: Anderson–Darling on the per-usage projections onto `v*`, alongside the permutation null.
   - Liu et al.'s second-order split, already planned.
   - NP-MSSG/AdaGram context-clustering sense induction as the E3 sense-recovery baseline.
   - Uniform enlargement, already planned.
7. **Text-evidence zero-shot generators (E4.3/E5.4, where definitions exist).** À la carte (cheap) and a CoLLEGe-style or definition-encoder generator, reported separately from the structure-only claim.
8. **M5 controls (E7).**
   - EntiGraph-style synthetic continued pretraining at matched tokens, as a second control next to compute-matched plain continued pretraining.
   - OLLM/LLMs4OL-style prompting as an authoring baseline for gold-edge recovery, next to Hearst patterns.
   - Replay framed as data *accumulation*, following Gerstgrasser et al.
9. **Tokenizer-side reference (optional, cited only).** A SuperBPE-tokenized from-scratch model is the obvious reviewer question. If it is not run, state why: the channel is designed to sit on a fixed tokenizer.

## Not verified or not found

- Venues marked † were not re-checked. Bootleg's per-type and per-relation "3.3× at 1% of the space" figure is quoted from the arXiv v3 text. Engram's injection layers and gate were taken from the arXiv v2 HTML.
- No paper was found that splits embedding or dictionary vectors using per-usage momentum conflict, or that attends over HRR-bound ontology edges with a context query. Both are the claims most exposed to a missed 2025–2026 preprint, so a keyword alert should be kept until submission.
