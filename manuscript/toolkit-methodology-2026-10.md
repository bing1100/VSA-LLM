# The read–learn–write toolkit: methodology updates and benchmarks

**Date:** 8 October 2026. **Status:** proposal for the author. Nothing in this document is queued.

**Sources:**
- the status document [status-2026-10-08.md](status-2026-10-08.md);
- the current designs (E7, E10, E11, E12, the binding program, T1c-F) in
  [execution.md](../resources/plan-improvement/execution.md) and the experiments' pre-registrations;
- a benchmark check made on 8 October 2026 against each paper's page, repository or data card. Items that could
  not be verified are marked *unverified*.

---

## 0. Summary

**The toolkit.** It has three tools on one shared store of composed concept vectors, plus a meta layer that calls them:

| Tool | What it does | VSA operation | Evidence today |
|---|---|---|---|
| **Read** (debind) | gets relations back out of a term's stored vector, for free, as a tool call | unbind + clean-up | **Strongest.** Stored relations decode at 0.99 MRR. Decoding them into the prompt lifts role-swap accuracy from 0.51 to 0.79 (seed-1 pilot; 3 seeds due tonight). T5 only |
| **Learn** (distil) | turns what passive training absorbed into explicit, checkable relations and concepts | decompose a learned vector into role–filler bindings; propose, test, commit | **Weakest.** Synthetic recovery works but rule mining (AMIE) matches it. The acceptance test is not calibrated. WordNet transfer fails. Active + passive is not faster (E10.9a) |
| **Write** (read-to-learn) | adds a never-seen word from its definition with no gradient step | compose a frame of new and existing atomics, insert the row | **Mixed.** Strong on T5 (held-out terms −9 to −13% loss; property test 0.19 → 0.26), null on real chemistry (T4). E11 (definition → frame) is queued |
| **Meta** | the model decides when to call the tools and judges their answers | tool calls over the three tools | Designed. The 3a agent pilot produces well-formed calls 82–84% of the time. 3b and 3c are pre-registered, but their code is not built |

**The cycle you described** is one experiment, proposed here as **E13**:
1. passive training from a seed ontology;
2. *learn* new relations and concepts;
3. *write* new words by reading;
4. *read* them for reasoning;
5. a second round of training that needs fewer tokens, also on a 4-bit model whose weights stay frozen.

E13 is the only experiment that tests the toolkit as a whole, and the only one that can support a training-efficiency
claim.

**What must change in the method:**
- **M1:** a real-world holdout in the HRRBERT style. Today the held-out terms of the real tracks still appear,
  unlinked, in the training text.
- **M2:** one store API for the three tools.
- **M3:** a null-calibrated acceptance test for *learn* and *write*.
- **M4:** a shared set of baselines.
- **M5:** a Wikidata-framed track, so the general benchmarks can be used at all.
- **M6:** a modern host on real text.

**Benchmarks** (section 3):
- **Read:** the fictitious reversal-curse set, the LRE relations, BEAR and BioLAMA.
- **Learn:** taxonomy expansion (TaxoExpan/TMN) and the SNOMED time-split ontology-enrichment set, plus a MeSH time-split
  we build (none exists).
- **Write:** Entity Inferences, COMPS-WUGS, ALCUNA and WNLaMPro, plus ICD-10-CM FY2027 codes, which no model has seen.
- **Meta:** PopQA, framed as "when to call the memory tool".

**Order** (section 6): two results this week decide how much of this is worth building:
- the E12 recall result at 3 seeds, due tonight;
- T7 at 3 seeds, due about 10–11 October.

**Build order:**
1. holdout data;
2. *write* benchmarks and E13;
3. *read* benchmarks;
4. the *learn* redesign, which carries the most risk.

**Rough cost:** 150–200 GPU-hours (≈ 6–8 days of the RTX 3090) on top of the current queue.

---

## 1. Where each tool stands, and the gap

### 1.1 Read (debind)

**Have:**
- the unbind/clean-up API (`relations.py`, `cleanup.py`);
- a recall tool with decode, chain and reverse (`self_query.py: RecallStore`);
- an agent toolbox (`e12_agent.py`);
- a training-time readout arm (`readout.py`, queued).

**Shown:**
- Decoding is near perfect: 0.99 MRR.
- Chained two-hop works through per-word vectors (0.88) but not through one global memory (0.07).
- HRR beats untyped bundling (+0.38) and translation (+0.19) on reverse lookup with ambiguous fillers.
- The model does not use the binding on its own: role-swap twins sit at chance.
- Decoding into context makes it usable (pilot: 0.79).

**Gaps:**
- Everything is on T5.
- No public benchmark has been run.
- The comparison that matters to a reviewer, the store against a plain knowledge-base lookup, has only one data point:
  the gold frame as text scores 0.86 against 0.79 for recall.

### 1.2 Learn (distil)

**Have:**
- edge acceptance by held-out utility (`authoring.py: edge_utility`, `self_test.py: accept_edges`);
- slot and property tests;
- a reflection loop (`e10_active_passive.py: reflect`);
- sparse frame inference from vectors (`frame_inference.py: omp_frame`, synthetic only);
- E7 self-authoring code, with 9 jobs pending at priority 60.

**Shown (synthetic):** erased-edge recovery AUC 0.98.

**Against it:**
- AMIE makes the same identifications.
- A no-learning readout reaches 0.91.
- Null worlds accept 2.7–4.3 false slots per run.
- On WordNet with frozen GPT-2 nothing transfers.
- The active + passive pairing is not faster (E10.9a). It does write useful rule-implied edges for concepts never
  observed (+0.021).

**Gaps:**
- No calibrated acceptance test.
- No joint-LM version: E10.2 is designed but not built.
- No real-data gold.
- The goal was framed as speed, which failed.

### 1.3 Write (read-to-learn)

**Have:**
- definition → frame readers (`read_to_learn.py`: linker, type prior, pattern, host, teacher);
- row insertion (`span_channel.add_entries`, `compose.add_concepts`);
- E11, pre-registered and queued:
  - P1: T5 property accuracy;
  - P2: loss after held-out T4 terms;
  - E11-M: frame against definition-in-context.

**Shown:**
- Held-out T5 terms composed from gold frames lower the loss after the term by 9–13% (that is "reading" with a perfect
  reader).
- The property test rises from 0.19 to 0.26.
- On T4 the property test does not move.

**Gaps:**
- No real reader result yet: E11 is queued.
- No public benchmark.
- The efficiency question has not been tested: does reading first make the next round of training cheaper?

### 1.4 Meta

**Have:** the E12 3a agent harness; the pilot produces well-formed tool calls 82–84% of the time.

**Designed:**
- 3b, learning from tool traces;
- 3c, calibrated self-critique with a corrupted-store null world.

**Gaps:**
- The 3b and 3c code (`e12_traces`, `e12_critique`) does not exist.
- No public benchmark.

---

## 2. Methodology updates that apply to every tool

### M1. A real-world holdout in the HRRBERT style ("ROOD-text")

**What HRRBERT did.**
- 32 ICD codes were held out of pre-training and fine-tuning entirely.
- All MIMIC-IV patients with any of those codes (≈ 30k) formed the test set.
- The unstructured model never saw the codes; HRRBERT composed them from the ontology.
- Result on disease prediction for these patients: 83.5 against 46.2 for the unstructured model.

**What we do now is weaker.**
- On T4, T1-open, T7 and T1c a held-out term's vector is untrained, but the term still occurs, unlinked, in the
  training text (`t4-chemistry/README.md:38`). The host learns the word from text; only the channel is zero-shot.
- Only T5 removes held-out terms from the training text completely.
- T1c-F holds its 382 codes out as *labels* only; the admissions stay in training.

**Update.** Add three real-world holdouts, each with a downstream task, not only loss:

| Holdout | Built from | What is never seen | Test |
|---|---|---|---|
| **H1. Document exclusion (T7-ROOD)** | T7: MeSH 2026 supplementary records in 2025–26 PubMed | every training document that mentions a held-out record (any alias) is dropped; the host's pretraining is unlikely to contain them, because the names are absent from 300k general-web documents | loss after the term; MeSH-heading and pharmacological-action placement; PubMed items mentioning the term |
| **H2. Code-system time split (T1c-ROOD)** | MIMIC-III (ICD-9) for training, MIMIC-IV ICD-10 admissions for testing; ICD-10 codes framed from SNOMED via the UMLS map | ICD-10-only codes never occur in the ICD-9-era training data | coding and disease prediction for admissions whose codes are all unseen (HRRBERT's "ROOD-unseen" protocol) |
| **H3. Post-release codes** | ICD-10-CM FY2027 new codes (effective 1 Oct 2026; public; 190–238 codes, sources differ) | created after every host's training data | place the code under its parent (Hits@k); match code to description (MedConceptsQA format) |

H2 also mirrors HRRBERT's second out-of-distribution test (eICU), because the coding system changes between training
and testing.

All MIMIC rules stay as they are:
- aggregates only;
- no note text reaches any external service or agent context;
- derived tables stay under `~/data/vsa-llm/`.

### M2. One store, three verbs

Today the three tools live in separate experiment modules. Proposal: a single `ConceptStore` with:
- `read(term, role | chain | reverse)`;
- `write(term, frame)`;
- `propose(evidence) → frames`;
- `accept(proposals, test)`.

Each verb is also exposed as a tool call for the meta layer. This is a refactor of existing code (`RecallStore`,
`RecallWriter`, `read_linker`, `accept_edges`), not new science. It is what makes the cycle in E13 possible.

### M3. A null-calibrated acceptance test

R10 found that the acceptance tests accept structure in worlds that have none. The fix:
- Every *learn* or *write* proposal is accepted only if it passes a test whose false-acceptance rate is measured on a
  null world: the same data with relation labels permuted, or with fillers swapped between terms.
- Pre-register a false-discovery ceiling of 5%.
- Candidates: refit-based held-out utility with Holm over proposals (it cut null-world edge acceptance from 40–45% to
  3–4% in E10), an MDL criterion, or knockoffs.
- Report precision on real gold next to the null-world rate.

### M4. Shared baselines

Every tool is compared with what a reviewer will ask for first:

| Tool | Baselines |
|---|---|
| all | the same information as text in the prompt (frame, definition or facts prepended), with its token cost |
| read | a symbolic lookup of the gold frame; a learned linear decoder on hidden states (LRE); untyped, translation and fixed-random operators |
| write | new-token initializations (subtoken mean, definition encoder); TransE rows; fine-tuning on the definition; weight editors (ROME, MEMIT, AlphaEdit); CoLLEGe-style generated embeddings (cited; reimplementation optional) |
| learn | AMIE; KG embeddings (TransE, RotatE, ComplEx) on the same candidate pools; LLM prompting (OLLM/LLMs4OL style); Hearst patterns |
| meta | always or never calling the tool; a fixed pipeline; adaptive retrieval in the Self-RAG style |
| 4-bit | QLoRA fine-tuning on the same text at matched tokens |

### M5. A Wikidata-framed track for the general benchmarks

Most public relation benchmarks (BEAR, LRE relations, reversal curse, TwoHopFact, PopQA, Entity Inferences) are built
on Wikidata entities. Our ontologies are MeSH, ChEBI and SNOMED.

To run those benchmarks, the store needs frames for their entities:
- **T8:** a Wikipedia / FineWeb-Edu track linked to Wikidata entities, with frames from Wikidata properties;
- a holdout that is node-disjoint and document-excluded, as in M1.

**Caution.** On famous entities the host already knows the facts. There, the read tool is in effect a knowledge-base
lookup, and the claim has to be about properties of the store, not knowledge coverage:
- size: fixed-width vectors, no per-concept parameters;
- zero-shot composition for held-out entities;
- reverse and chained queries in vector space.

The contamination-free sets (fictitious reversal pairs, ALCUNA, Fictional Knowledge, COMPS-WUGS) are where the claim
is clean.

### M6. A modern host on real text

Every Qwen result so far is on synthetic T5.
- SmolLM2-135M/360M can be scored on ranking (log-likelihood) benchmarks.
- Generation-heavy evaluations and the meta layer need Qwen3-1.7B.

Proposal: Qwen3-1.7B (LoRA) on T7 and on T8, seed 1, and seeds 2–3 if the effect is clear.

---

## 3. Benchmarks per tool

"Ranking" means scoring the answer options by log-likelihood, which works for small models. The benchmark check
confirmed every entry below unless marked otherwise.

### 3.1 Read: relational recall from the store

| Benchmark | What it tests | Why it fits | Pitfall |
|---|---|---|---|
| **Reversal curse, fictitious set** (Berglund et al., ICLR 2024; MIT) | learn "A is B", answer "B is A" | **Headline candidate.** HRR unbinding works both ways (reverse lookup 0.96 on T5), so "the store breaks the reversal curse" is a clean, contamination-free test. It needs the *write* tool to insert the fictitious people and works first | descriptions must be parsed into frames with new atomics (E11 reader) |
| **LRE relations** (Hernandez et al., ICLR 2024; MIT; 47 relations) | can a relation be decoded linearly from the subject | its faithfulness and causality metrics are the reviewers' vocabulary for "read out" and "edit"; our probe B already computes LRE-style decoders | Wikidata entities (needs M5) |
| **BEAR** (Wiland et al., Findings NAACL 2024; CC-BY-SA) | relational facts by ranking statements | made for causal LMs and reports small models | known entities: test store properties, not coverage |
| **TwoHopFact** (Yang et al., ACL 2024; CC-BY) | latent two-hop composition | compares directly with our chained-recall result | 135M will score low; report per host |
| **BioLAMA** (Sung et al., EMNLP 2021; CTD part public) | biomedical relation probing | domain match with T7/T1c | best published models reach only ≈ 18.5% Acc@5 |
| Real-world: T7-ROOD (H1) | recall of relations for new 2025–26 records | our own track, contamination-free | — |

**Recommended set:**
- the fictitious reversal set as the headline;
- LRE relations;
- BEAR;
- T7-ROOD.

### 3.2 Learn: turning passive learning into explicit structure

**Redesign.** The goal changes from "learns faster" (refuted in E10.9a) to "makes explicit, correct and useful what
passive training absorbed". The method is **decompose, then verify**, with three proposal sources:
1. **Decompose learned vectors.** Express a passively trained vector (a free-table row, or a hidden state read
   through the LRE decoder) as a sparse sum of role–filler bindings over the dictionary (OMP / resonator / Lasso;
   `frame_inference.py`). Probe B shows the host's hidden states carry relation information (0.57 MRR against 0.78
   for the store), so there is something to extract.
2. **Rule closure.** Rules adopted after property tests imply new edges. E10.9a showed this writes useful edges for
   concepts no observation covers.
3. **Self-authoring.** The host writes frames for terms it read (E7).

Each proposal passes the M3 test before it is written.

**Three things it should learn,** with gold:

| Learns | Gold | Measured by |
|---|---|---|
| new edges for known concepts ("update the mapping") | curated edges erased from the seed ontology; edges added in later releases (MeSH 2025 → 2026, SNOMED versions) | precision and recall against the gold; null-world false rate; loss gain after acceptance |
| new concepts (frequent unlinked spans) | records added in later releases | placement Hits@k / MRR |
| new relations (empty slots) | the hidden relation's pairs; adopted properties (symmetric, inverse) | adjusted Rand index against the gold relation; property precision |

**Benchmarks:**

| Benchmark | Fit | Pitfall |
|---|---|---|
| **TaxoExpan / TMN sets** (Shen et al., WWW 2020; Zhang et al., AAAI 2021) | "attach a new term to its parent": MRR, Hit@k | random node splits leak siblings into training |
| **SNOMED time-split ontology enrichment** (Dong et al., CIKM 2023; data CC-BY on Zenodo) | discover and place concepts added between SNOMED 2014 and 2017; the closest public match | needs the SNOMED licence, which we hold (2022-05-31 release) |
| **SemEval-2018 Task 9, subtask 2A (medical)** | hypernym discovery by ranking | — |
| **OntoLAMA** (He et al., Findings ACL 2023) | subsumption inference (Disease Ontology, Gene Ontology and others) | — |
| Inductive WN18RR / FB15k-237 splits (Teru et al., ICML 2020) | link prediction for unseen entities | use the inductive splits only; the transductive ones do not test new terms |
| FewRel clustering protocol (Han et al., EMNLP 2018) | discovering new relation types | — |
| **New: MeSH 2025 → 2026 time split** | place 2026 records (mapped heading, pharmacological action) from the PubMed text that mentions them, against NLM's curation | **none exists publicly** (benchmark check); building it is itself a contribution. Needs the MeSH 2025 release, which is public; only 2026 is on disk |

**Recommended set:** the MeSH time split (new), SNOMED enrichment, and TaxoExpan/TMN.

### 3.3 Write: new words from reading

| Benchmark | What it tests | Fit | Pitfall |
|---|---|---|---|
| **Entity Inferences (+ ECBD subset)** (Onoe et al., ACL 2023) | read an entity's definition, then make inferences | **The primary write benchmark.** It is exactly "learn from a description, then infer", with established baselines (fine-tuning, MEND, ROME, definition prepended), and is scored by ranking | ECBD's 2017–21 entities are probably in pretraining; lead with Entity Inferences |
| **COMPS-WUGS** (Misra et al., EACL 2023, best paper) | property inheritance to nonce words ("a wug is a dog, so…") | **Natural fit:** "is-a dog" is a one-edge frame, and composition should inherit the parent's properties; ranking; tested on 22 models | — |
| **ALCUNA** (Yin et al., EMNLP 2023; MIT) | artificial entities derived from the Encyclopedia of Life taxonomy | built from an ontology, so it maps onto frames directly | ranking only |
| **WNLaMPro** (Schick & Schütze, AAAI 2020) | relational probes for rare words | ranking; small-model friendly | — |
| Fictional Knowledge (Chang et al., NeurIPS 2024; CC-BY) | learning fictional facts and generalizing compositionally | contamination-free | small (130 paragraphs) |
| CoLLEGe (Teehan et al., COLM 2024) | new-token embeddings from a few sentences | **the closest method:** cite it, reuse its GRE and slang tasks | definition generation needs generation |
| Real-world: T7-ROOD (H1) and ICD-10-CM FY2027 (H3) | read the record note or code title, then compose | truly unseen | — |

**Recommended set:**
- Entity Inferences;
- COMPS-WUGS;
- ALCUNA;
- T7-ROOD and FY2027 codes as the real-world anchors.

### 3.4 Meta: tool use and self-knowledge

| Benchmark | Use | Fit |
|---|---|---|
| **PopQA** (Mallen et al., ACL 2023) | "when to call the memory tool": tail entities should trigger a call, head entities should not | short answers; Self-RAG and adaptive retrieval are the baselines |
| Toolformer LAMA protocol (Schick et al., NeurIPS 2023) | precedent for a model that learns to call a knowledge tool | protocol |
| P(IK)-style probes (Kadavath et al., 2022, arXiv) | does the model know when its own store is wrong (AUROC, ECE) | works at 135M as a probe on hidden states |
| Faithfulness by intervention (Atanasova et al., ACL 2023; our E12 F1) | corrupt or swap the stored vector, and the answer must change | already built (F1) |

No standard benchmark exists for a model querying its own knowledge store (benchmark check). The E12 twins and
corrupted-store null world can be released as one.

---

## 4. E13: the learning cycle (proposed)

**Question.** Can a model trained passively with an ontology learn new structure, read in new words, use them for
reasoning, and then learn the following text faster? And can it do so when its weights are frozen at 4 bits?

**Track.**
- **T7-ROOD (H1)** is the main track: real 2025–26 PubMed text and MeSH 2026 records. The records are split by entry
  date into *round 1* (seen) and *round 2* (new); the round-2 records' documents are excluded from round-1 training.
- **T5** runs alongside as the controlled upper bound.
- **Hosts:** SmolLM2-360M × 3 seeds; Qwen3-1.7B × 1 seed.

**Stages:**

| Stage | What happens | Tool | Measured |
|---|---|---|---|
| 0. Passive round 1 | joint fine-tuning with the seed ontology (round-1 records, ≈ 20% of curated edges erased) | — | the usual E9 strata |
| 1. Learn | propose missing edges, new relations, new concepts from the trained model (§3.2); accept with the M3 test | learn | precision and recall against the erased edges; null-world false rate |
| 2. Read to learn | round-2 records: read their scope note → frame (new atomics plus learned relations) → insert rows; no gradient | write | loss after the new terms on round-2 evaluation text (zero-shot); property and relation items |
| 3. Reason | answer relation, reverse and two-hop questions about the new terms through the recall tool | read | accuracy with and without recall; against the frame as text |
| 4. Passive round 2 | continue training on round-2 text | — | **tokens to reach a target loss** after the new terms; area under the learning curve; forgetting on round-1 text; general-text locality |
| 5. Frozen 4-bit host | stages 2–4 with the host quantized (GPTQ and plain rounding) and frozen; only the channel (FP16) and its rows change | write, read | loss after the new terms; against QLoRA on the same text at matched tokens and bytes |

**Arms in stages 2 and 4:**
- **read-to-learn** (E11 reader);
- **no reading:** new terms left unlinked, or given subtoken-mean rows, which is standard vocabulary expansion;
- **gold frames:** the oracle reader;
- **definition prepended to the training text:** compute-matched;
- **random equal-degree frames:** a content control;
- **operator controls** (untyped, translation): only if stage 3 is run on them.

**Pre-registered endpoints and kill criteria:**
- **L1 (write):** read − no-read loss after round-2 terms at step 0 < 0, CI excluding 0.
- **L2 (efficiency):** tokens to criterion, read / no-read < 1, CI excluding 1. *If not met: no training-efficiency
  claim.*
- **L3 (4-bit):** INT4 host + read − INT4 host without reading < 0; non-inferior to QLoRA at matched tokens. *If
  QLoRA wins clearly: the claim is only "zero-gradient".*
- **L4 (learn):** precision of accepted edges ≥ 0.8 with a null-world false rate ≤ 5%. *If not met: stage 1 is
  dropped from the cycle and reported as negative.*
- **L5 (reason):** recall-tool accuracy on new terms − no tool > 0. This is E12's Q1, moved to real text.

**Expectation, from what we already have.**
- E9 held-out terms (gold frames, T5) predict a large L1 on T5.
- On T4 the zero-shot gain was small, so L1 and L2 on real text depend on T7 showing a gain.
- E10.9a predicts that stage 1 adds little speed but can add correct edges for unobserved concepts.

**Cost** (360M × 3 seeds, plus Qwen3-1.7B × 1): ≈ 40–55 GPU-h.
- Round 1 reuses T7 training where possible.
- Stage 4 is 5 arms × 3 seeds of short continued training.
- Stage 5 adds a 4-bit pass and QLoRA.

---

## 5. Changes to existing experiments

| Experiment | Change |
|---|---|
| E9 / T7 | add the T7-ROOD document-exclusion variant (H1) next to the current linker holdout; report both |
| T1c / T1c-F | add T1c-ROOD (H2: ICD-9-era training, ICD-10-only test codes) as the HRRBERT replication on text; T1c-F keeps its label holdout |
| E11 (read-to-learn) | add Entity Inferences, COMPS-WUGS, ALCUNA and FY2027 codes; becomes stage 2 of E13 |
| E10 (self-learning) | drop the speed endpoint; the method becomes decompose-then-verify with the M3 test; the endpoints become structure precision and recall against real gold (MeSH time split, SNOMED enrichment) plus utility; E10.2 (joint LM) is built as E13 stage 1 |
| E7 (self-authoring) | becomes one proposal source of *learn*, compared with decomposition and rule closure under the same test |
| E12 (self-query) | add the fictitious reversal set, PopQA (when to call) and the T7-ROOD recall items; 3b and 3c continue as designed |
| Binding program | unchanged; the readout arm (≈ 13–14 Oct) decides whether training-time unbinding helps |
| New data builds (CPU) | T7-ROOD; MeSH 2025 → 2026 time split; ICD-10-CM FY2027 frames; T8 (Wikidata frames for the general benchmarks); benchmark adapters (log-prob ranking) |

---

## 6. Order, gates and cost

**Two results decide this week how much to build:**
1. **E12 recall at 3 seeds (tonight).**
   - Positive: *read* is the lead tool and the meta layer is worth building.
   - Null: *read* reduces to "decodable algebraically", and 3b/3c lose their motivation.
2. **T7 at 3 seeds (≈ 10–11 Oct).**
   - Positive: E13 runs on T7, and the paper's headline is real-world.
   - Null: E13 runs on T5 only, as a mechanism study, and the real-world claim is withdrawn.

**Build order (decisive first):**

| Step | What | GPU-h (rough) | Depends on |
|---|---|---:|---|
| 1 | T7-ROOD, MeSH time split, FY2027 codes, benchmark adapters (CPU) | 0 | — |
| 2 | *write* benchmarks on existing T5/T7 checkpoints (Entity Inferences, COMPS-WUGS, ALCUNA, FY2027) | ≈ 5 | E11 code |
| 3 | E13 on T7-ROOD (or T5 if T7 is null) | ≈ 40–55 | T7 result, step 1 |
| 4 | T8 Wikidata track + *read* benchmarks (reversal, LRE, BEAR, TwoHopFact); Qwen3-1.7B | ≈ 45–60 | E12 Q1 result |
| 5 | *learn* redesign: decompose-then-verify with the M3 test, MeSH and SNOMED time splits, TaxoExpan | ≈ 15–20 | step 1; E13 stage 1 |
| 6 | T1c-ROOD (H2) | ≈ 15–20 | peer session, T1c results |
| 7 | Meta: 3b, 3c, PopQA | ≈ 20–30 | E12 Q1, step 4 |

**Total:** ≈ 150–200 GPU-h. These are rough estimates from measured per-run times; the new harnesses have no timing
history.

---

## 7. Decisions needed

1. **Direction.** Adopt the read–learn–write framing and E13 as the integrating experiment. *Recommended: yes.* Build
   it only after the two gates in §6, so we do not build on a null.
2. **Real-world holdouts.** Approve H1 (T7-ROOD) and H3 (FY2027 codes) now, since they are cheap and CPU-only. Approve
   H2 (T1c-ROOD), which is owned by the peer session, under the MIMIC rules.
3. **Benchmarks.** Approve the recommended sets:
   - **Read:** reversal (fictitious), LRE, BEAR.
   - **Learn:** MeSH time split, SNOMED enrichment, TaxoExpan.
   - **Write:** Entity Inferences, COMPS-WUGS, ALCUNA.
   - **Meta:** PopQA.
4. **Modern host on real text.** Add Qwen3-1.7B to T7 and T8.
5. **Ownership** (proposed):
   - this session: E9/T7, E10 (*learn*) and E13;
   - the peer session: binding, E11, E12 (*read*, *meta*) and T1c;
   - both: the shared store API (M2).
