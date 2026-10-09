# E12 — Explainability by decomposition and self-query (decision 62)

**Status:** design and pre-registration, written on 2026-10-08 and committed **before any evaluation of these measures on a
trained checkpoint** (GPU or CPU). The pilots of §15 run after this commit, are labelled PILOT, use T5 SmolLM2-360M seed 1
only (Qwen3-0.6B / 1.7B for the agentic feasibility pilot), and **do not count toward the endpoints**, which use seeds 1–3 in
the queued runs. Any later change is listed in §16 with its date and reason.

**Origin:** author decision 62 (`resources/plan-improvement/execution.md`, 2026-10-08), after binding step 1
(`experiments/e9-retrofit/preregistration-binding.md`; formal report `experiments/e9-retrofit/report/t5-binding`). On T5
(SmolLM2-360M, seeds 1–3):

- **The roles are stored.** Algebraic filler recovery from the static bundle, held-out concepts, all-atom cleanup:
  C5 0.96 (0.91 on role-ambiguous edges), C5rf 0.98, C5tr 0.14, C5ut 0.14 (0.03 on role-ambiguous edges); exact division
  for C5 0.51.
- **The roles are not read.** Role-swap twin contrast accuracy (`choice` items; chance 0.5): C5 0.494, C5ut 0.498, C5tr
  0.492, C5rf 0.494, C2 0.487, C0′ 0.484, P0 0.492. C5 `own − swap` +0.001 [−0.015, +0.017].

The channel adds the composed vector at the term's last subtoken, *before* the question words; to answer a role question
the host would have to learn unbinding internally, and the LM loss almost never requires it (decision 60: a T5 term's
filler set identifies it).

**Thesis (the author's).** Decomposing the VSA store is **inspection** of the vector the model computes with, not
**narration** (a reasoning trace, which can be unfaithful). It can therefore give deeper explainability than reasoning
traces — *provided the decoding is validated by intervention*. And a **self-query tool** that decodes the model's own store
into text lets the model use its stored roles through reading, without learning unbinding in its weights.

This document tests the two halves separately: **Q1** — does the recall tool make the stored roles usable (phase A, a fixed
pipeline)? **F1** — are the decoded edges what the model's channel route actually uses (faithfulness by store
intervention)? It also pre-registers designs for **3a** (agentic self-query), **3b** (learning from tool-using traces) and
**3c** (a calibrated self-critique loop).

**Code** (tests: `tests/test_self_query.py`, `tests/test_e12_self_query.py`, CPU):
- `src/vsa_embed/self_query.py` — the recall tool, model-independent: `RecallStore` (static stores; unbinding with the
  operator's primary method; typed or all-atom cleanup; slot-aware and slot-free read-back; chained recall; reverse lookup;
  the role-blind bundle readout), `RecallWriter` (text with confidences), `symbolic_lines`, `roleless`, `slot_accuracy`.
- `src/vsa_embed/experiments/e12_self_query.py` — phase A (`evaluate`, `recall`, `queue`).
- `src/vsa_embed/experiments/e12_faithfulness.py` — F1 (`evaluate`, `queue`).
- `src/vsa_embed/experiments/e12_report.py` — Q1, F1 and the secondaries across a stage's runs.
- 3a's harness (`e12_agent`) is built with its feasibility pilot (§11).

**Items:** the E9 item sets, unchanged: `experiments/e9-retrofit/items/role-twins-t5-smollm2-v1` (sha256 of
`items.jsonl.gz` `52cea580…33da`), `role-twins-t5-qwen3-v1`, `role-natural-t4-smollm2-v1` (`4462062a…9398`),
`understanding-t5-smollm2-v1`, `understanding-t4-smollm2-v1`, `new-words-t5-smollm2-v2`, `new-words-t5-qwen3-v2`.

## 0. Related work and the gap (checked 2026-10-08; every page below was opened)

**Tool use and self-query.** Toolformer (Schick et al., NeurIPS 2023) teaches an LM to decide which external APIs to call
from a handful of demonstrations; ReAct (Yao et al., ICLR 2023) interleaves reasoning traces with actions; Self-Ask (Press
et al., Findings of EMNLP 2023) decomposes multi-hop questions into follow-ups, optionally answered by search, and shows the
compositionality gap does not shrink with scale; Self-RAG (Asai et al., ICLR 2024) retrieves passages on demand and
critiques them with reflection tokens. Their tools are external — none reads the model's own state.

**Knowledge-augmented LMs with an editable store.** KnowBert (Peters et al., EMNLP 2019) and Entities as Experts (Févry et
al., EMNLP 2020) inject entity memories at mention spans (opaque learned embeddings); Facts as Experts (Verga et al., NAACL
2021) gives an LM a symbolic fact memory that can be edited without retraining; KBLaM (Wang et al., ICLR 2025) injects
knowledge-base triples as key–value vectors through rectangular attention and reads interpretability off the attention;
Larimar (Das et al., ICML 2024) and MemLLM (Modarressi et al., TMLR 2025) give LMs explicit memories with one-shot updates
or read/write calls; limited-memory LMs (Zhao et al., ICLR 2026) learn targeted lookups instead of memorizing. Their
entries are key–value pairs selected by attention, or external tables — not a superposed role–filler code at the mention
that can be decoded algebraically.

**Introspection.** Language models are partly calibrated about their answers (Kadavath et al. 2022, arXiv) and can learn
some privileged self-prediction (Binder et al., ICLR 2025, "Looking Inward"), which fails on harder and out-of-distribution
tasks; injected concepts are noticed and named only about 20% of the time at the best layer and strength (Lindsey,
Transformer Circuits 2025 / arXiv 2601.01828), and binary injection detection in Llama-3.1-8B is explained by global logit
shifts (Hahami et al. 2025, arXiv). Self-reports are narration; our recall is a decode.

**Faithfulness of explanations.** Chain-of-thought explanations can omit the features that drive the answer (Turpin et al.,
NeurIPS 2023: up to 36% accuracy drop from unmentioned biases), become less faithful with scale under CoT interventions
(Lanham et al. 2023, arXiv), and reveal hint use rarely (Chen et al. 2025, arXiv: often below 20%). Comprehensiveness and
sufficiency are ERASER's (DeYoung et al., ACL 2020); faithfulness is graded, distinct from plausibility (Jacovi & Goldberg,
ACL 2020); decodable information need not be used (Ravichander et al., EACL 2021; Elazar et al., TACL 2021, amnesic
probing). Step 1's twins are an instance: decodable, not used.

**Post-hoc dictionaries.** Sparse autoencoders recover interpretable features (Bricken et al., Transformer Circuits 2023;
Cunningham / Huben et al., ICLR 2024; Templeton et al. 2024; Gao et al., ICLR 2025), but their dictionaries are learned and
incomplete (splicing a 16M-latent SAE into GPT-4 costs the loss of 10% of its pretraining compute; "an incomplete
description"), and strong probe baselines often match them (Kantamneni et al., ICML 2025). Our atom dictionary and relation
operators are the store's own, by construction.

**Decoding a model's own representations.** Patchscopes (Ghandeharioun et al., ICML 2024) uses the model to verbalize its
hidden states and repairs some multi-hop errors; SelfIE (Chen et al., ICML 2024), LatentQA (Pan et al., ICLR 2026),
Activation Oracles (Karvonen et al. 2025, arXiv) and Natural Language Autoencoders (Fraser-Taliente et al., Transformer
Circuits 2026) learn verbalizers, with documented blind spots (Bersia & Gaintseva 2026, arXiv). The logit and tuned lens
(nostalgebraist 2020; Belrose et al. 2023) decode next-token beliefs. Relation decoding is often a linear map of the subject
(Hernandez et al., ICLR 2024); in-context binding uses binding-ID vectors (Feng & Steinhardt, ICLR 2024). Concept bottleneck
models (Koh et al., ICML 2020) are intervenable but the bottleneck is the only path to the output; our store is an
additive side channel the model may ignore.

**VSA / HRR / TPR stores.** Tensor-product structure can be fitted to RNN and Transformer states and validated by
intervention (McCoy et al., ICLR 2019; Soulos et al., BlackboxNLP 2020; McCoy et al. 2026, arXiv 2608.29530; theory in
Zhang & McCoy 2026, arXiv); TP-Attention binds roles inside a Transformer (Schlag et al. 2019, arXiv); HRRs serve as output
layers (Ganesan et al., NeurIPS 2021) or attention (Alam et al., ICML 2023); Dhanraj & Eliasmith (EMNLP 2025) encode hidden
states into VSA vectors as an arithmetic co-processor; Kumar (2026, arXiv 2606.24948) finds that an HRR knowledge-graph
memory recovers one hop but composes two hops at chance; Bronzini et al. (2025, arXiv 2509.25045) decode LLM residual
states by unbinding and codebook cleanup after a learned mapping into a VSA space, and name the lack of causal validation as
their main limitation. Multi-hop and reversal: latent two-hop composition fails on synthetic facts (Balesni et al. 2024–25,
arXiv); models use the first hop more than the second (Yang et al., ACL 2024); the reversal curse holds for parametric facts
but not for facts in context (Berglund et al., ICLR 2024), and may be a binding problem (Wang & Sun, ICLR 2026).
Internalizing explicit steps: stepwise CoT removal (Deng et al. 2024, arXiv), distilling step-by-step (Hsieh et al.,
Findings of ACL 2023). Self-correction without external feedback fails (Huang et al., ICLR 2024); tool-interactive critique
helps (Gou et al., ICLR 2024, CRITIC).

**The gap, honestly.** None of the components is new: LMs call tools and learn when to; they read explicit, editable stores
injected at mentions; they verbalize their own activations; interventional faithfulness metrics are standard; and VSA/TPR
decoders of neural states exist, as post-hoc probes or fitted structure checked by intervention. What may be new is the
conjunction: **(i)** a closed-form decoder (unbind with the store's own relation operator, clean up against its own atom
dictionary) applied to the very vector the LM computes with, so the decode can be checked against a store known by
construction rather than learned; **(ii)** the decode validated by edge-level store interventions scored for
comprehensiveness, sufficiency and specificity; **(iii)** the decoded frame returned to the same model as a self-query tool
and tested on role-swap items the model fails without it. The literature check found no work doing all three (very recent
preprints may have been missed). Two cautions in wording: learned-HRR decoding is *approximate* (crosstalk grows with the
bundle; Kumar 2026 shows interference breaking chained retrieval), so "algebraic, closed-form" is right and "exact" is not;
and a decode shows what the store makes available, not what the model uses — hence F1.

## 1. Questions and primary endpoints

| | Question | Unit | Primary endpoint |
|---|---|---|---|
| **Q1** (recall tool) | Does putting the recalled frame of a twin's own store in context let the model tell role-swap twins apart? | twin pair × seed | twin **contrast accuracy** (`choice`), C5 host: **Q1a** `recall:own − none`, **Q1b** `recall:own − recall:C5ut`; Holm over the two |
| **F1** (faithfulness of decoding) | Through the channel alone (no recall in context), does removing a decoded edge from the store move the model's preference for that edge's filler more than removing another edge? | new word × seed | per-word **comprehensiveness net share** (τ = 0.05 nats): **F1a** C5 against 0, **F1b** C5 − C5ut; Holm over the two |

**Primary setting:** T5 synthetic glossary, SmolLM2-360M, full fine-tuning, seeds 1–3: the runs of R9 / WP-PQ1 and binding
step 1, evaluated in bf16 autocast as trained. Q1 and F1 answer different questions, so no correction is made between them
(stated in advance); each is tested at two-sided α = 0.05 after its own Holm step.

## 2. The recall tool

**Store.** A term's **static frame bundle** `c = Σ_e T_{r_e}(a_e)` (every edge weight 1: binding step 1's `static` condition,
BU-3 / BU-5), read from the run's trained composer (`final.pt`; the host is not loaded). For an existing entry, the composer's
own schedule (for C5sh, the shuffled frame the model was trained with); for a new word or a twin, its frame composed from the
trained atomics and relations, exactly as the channel composes it at insertion.

**Unbinding** (`relations.readout_method`, the operator's primary method): learned HRR (C5) → circular correlation; fixed
unitary (C5rf) → conjugate; translation (C5tr) → subtraction; untyped (C5ut) has no unbinding → the bundle readout, which
ignores the role; the readout arms' operators (learned unitary, bounded spectral, block-diagonal unitary, slotted) → their
primary methods (slotted: within the relation's slot).

**Cleanup.** *Typed* (primary): cosine nearest neighbours among the atomics observed under the relation in the ontology frames;
*all-atom* (secondary): every atomic. A slot with m fillers returns its top m.

**Read-back (slot-aware, primary).** The tool reads the term's slots — the relation multiset of the term's frame, the keys of
the store — and decodes every slot's fillers from the vector. Knowing the slots does not reveal the answer on any test here
(twins share their relation multiset; only the fillers differ). *Slot-free* (secondary, `recall-free`): every relation is
unbound and kept when its best typed cosine reaches a per-relation presence threshold that maximizes presence F1 on ≤ 3,000
seen entries of the store's ontology.

**Chained recall** (two-hop items): unbind r1 from the anchor's store, clean up, follow the recovered filler to the entry it
names (`e9_binding_chain.atom_entries`), unbind r2 from *that entry's* store, clean up. **Reverse lookup** (reverse items):
score every store of the track — and the item set's new words — by `cos(c, T_r(a_F))` (role-blind store: `cos(c, a_F)`);
return the top 5.

**Role-blind stores.** C5ut's whole-frame recall is one line "associated with" its top-K atomics by cosine, K = the number of
slots (no role is stored, so none is claimed). A role query on it (chain hop) is the type-restricted bundle readout.

**Text** (`RecallWriter`; one line per decoded filler, at most 32 per call; the confidence is the cleanup cosine, two
decimals; the filler worded as the item candidates word it):

    recall(dalkkloushfoltquark):
    - dalkkloushfoltquark is a system. (0.91)
    - dalkkloushfoltquark belongs to the finance area. (0.83)
    - dalkkloushfoltquark depends on Tindbreish Migration. (0.58)
    - dalkkloushfoltquark is part of Drun Rebuild. (0.49)

Each line fills the track's own statement template of the relation (`RelationTemplates.statement`, the training wording);
an untemplated relation reads "X <relation words> Y." (a one-word relation: "X has <relation> Y."). A role-blind line reads
`- X is associated with: system (0.40), finance area (0.38), …`. Chained recall writes two calls (`recall(X, owned by):` and
`recall(<bridge>, reports to):`), reverse lookup one (`lookup(owned by, the Zash Team):` and up to five holder statements).

**Prompt.** The context, a newline, then the item's prompt; the candidate continuation is scored after it. PMI uses the
item's **context-free** null prompt (cached once per item set); the twin contrast does not use the null (it cancels). Texts
longer than the scorer's maximum length are an error, never truncated (truncation would cut the candidate).

## 3. Phase A: conditions, item sets and hosts

**Conditions** (the context differs, the question does not):

| condition | context | role |
|---|---|---|
| `none` | nothing (binding step 1's `own` source) | baseline |
| `recall:own` | the recall of the term from the host run's own store | the self-query |
| `recall:<arm>` | from another run of the same stage, host and seed: `C5ut` (untyped), `C5tr` (translation); `C5` for hosts without a store; `C5@s` for P0 (seed s's store) | role-blind controls; the tool without the host's channel |
| `roleless:<store>` | the same decoded fillers in one "associated with" line, roles removed, atom order, **no confidences** (a filler's confidence comes from unbinding its role) | format-matched role-blind control |
| `wrong:<store>` | twins: the partner twin's recall written with this twin's name | does the host follow the recalled text? (predicted to flip) |
| `symbolic` | the gold frame in the recall format, confidence 1.00 | upper bound of a perfect store |
| `definition` | E11's prose definition from the gold frame (`read_to_learn.t5_definition` / `t4_definition`, style `prose`: held-out wordings) | in-context reading (E11) |
| `recall-all`, `recall-free`, `fields`, `noconf` | all-atom cleanup; slot-free decoding; "relation: filler" lines; no confidences | secondaries (format, cleanup) |

**Item sets and the call each condition makes:**
- **role-swap twins** (T5; 300 pairs, 600 terms; `choice` primary, `cloze` secondary): whole-frame recall of each twin's store.
- **natural role-ambiguous items** (T4; 715 anchors): whole-frame recall of the entry's store (frames up to 24 edges).
- **WP-UB understanding** (T5, T4): `two_hop` items with chained recall along the item's path; `reverse` items with reverse
  lookup; at most **150 anchors per subset** (seen, rare, heldout, new; the first ones in item order) to bound cost.
- **new words** (T5 v2): `property` items; the **first 300** words (`e9n-0000`–`e9n-0299`, the v1 base set).

**Decode accuracy is reported with every behavioural number:** per slot, the share of its gold fillers the context states
(role-blind lines: among all the line's fillers); twins: the four critical slots of a pair and whether all four are right;
two-hop: hop 1, bridge, hop 2; reverse: anchor above partner and anchor in the top 5.

**Hosts and stores (queued):**

| track / host | runs | store read |
|---|---|---|
| T5 SmolLM2-360M (primary) | C5, C5ut, C5tr, C5rf, C5sh, C2, C0′ × s1–3; P0 | own (composing arms); C5 (C2, C0′); C5 seeds 1–3 (P0); C5 also reads C5ut and C5tr, C5ut also reads C5 |
| T5 SmolLM2-135M | C5, C2, C0′ × s1–3; P0 | as above (no C5ut: the role-blind reference is `roleless`) |
| T5 Qwen3-0.6B / 1.7B-Base | C5, C2, C0′ × s1–2; P0 (seed 3 when trained) | as above (`roleless` reference) |
| T4 SmolLM2-360M, 135M | seed 1 now (C5, C5ut, C5rf, C2, C0′, P0; 135M: C5, C2, C0′, P0); the other T4 configs once trained | natural items, understanding |
| readout arms U5, U5u, U5sb, U5bu, U5sl, U5tr, U5ut (T5 360M, once trained) | s1–3 | own: twins and F1 |

## 4. Q1 — statistics, predictions, decision rule, refutation readings

**Statistics.** The pairs × seeds table of C5 `recall:own` − reference contrast accuracy over the pairs both of whose twins
link in every run of the contrast; the crossed random-effects model (`statistics.crossed_components`, Satterthwaite t,
two-sided; one seed: a one-sample t); Holm over Q1a and Q1b; the two-way cluster bootstrap (2,000 resamples) as the check.
Power: the step-1 P1 design (300 pairs, 3 seeds, per-pair SD ≈ 0.35) detects 0.04 at 80% power; the predicted effects are
≥ 0.2.

**Predictions** (written before any run):
- `recall:own` far above 0.5 (≈ 0.75–0.9): bounded by the decode (twins are role-ambiguous by construction; step 1's held-out
  all-atom recovery on role-ambiguous edges 0.91) and by the host's reading of a frame in context (`symbolic`).
- `none` ≈ 0.5 (step 1). `recall:C5ut` ≈ 0.5 **by construction**: a twin and its partner have the *same* untyped store, so
  their texts differ only in the name. `recall:C5tr` ≈ 0.5 likewise (the same bag of fillers plus offsets).
- `symbolic` ≈ the host's in-context reading accuracy (≈ 0.85–1.0); `symbolic − recall:own` ≈ the decode's cost.
- `definition` < `symbolic` (prose in held-out wording).
- `roleless:own` ≈ 0.5; `wrong:own` ≈ 1 − `recall:own` (the host follows the text).
- Hosts without a store: C0′ / C2 / P0 + `recall:C5` ≈ C5 + `recall:own` (the tool is host-agnostic text).

**Decision rule Q1** (checked in this order):

| Reading | Conditions |
|---|---|
| **(e) Uninformative** | `symbolic` ≤ 0.6 on the C5 host: the host cannot read roles from a gold frame in context, so neither contrast says anything about the store |
| **(a) "The self-query reads the stored roles"** | Q1a > 0 **and** Q1b > 0 (Holm p < 0.05); labelled "text-driven" when also `wrong:own` < 0.5 (CI excludes 0.5) |
| **(b) Partial** | exactly one of the two significant and positive |
| **(c) No gain from recall** | neither significant (with `symbolic` > 0.6: the decoded text is not usable although a gold one is — compare decode accuracy) |
| **(d) Recall hurts** | a significant negative contrast |

**Refutation readings.**

| # | Observation | Reading |
|---|---|---|
| Q-R1 | `recall:C5ut` or `recall:C5tr` contrast ≠ 0.5 (CI excludes 0.5) | impossible from the stores (the twins' stores are identical): the names carry signal (step 1's B4) or a bug |
| Q-R2 | Q1 (a) but `wrong:own` ≥ 0.5 | the host does not follow the recalled text; the gain is not the stored roles |
| Q-R3 | `recall:own` > `symbolic` (CI excludes 0) | decoded text beats gold text: inspect the confidences and the gold wording before reading Q1 |
| Q-R4 | `roleless:own` > 0.5 (CI excludes 0.5) | role information leaks through the role-stripped line (order, decode errors) |
| Q-R5 | Q1 (a) with the contrast on pairs with a decode error as high as on fully decoded pairs | the gain does not track the decoded content |
| Q-R6 | C0′ / P0 + `recall:C5` ≈ C5 + `recall:own` | not a refutation: the tool works without the channel; the channel adds nothing to the self-query |

**Secondaries (family SQ; Holm within each numbered item):**
- **SQ1** `symbolic − recall:own`, `definition − symbolic`, `recall:own − roleless:own`, `recall:own − wrong:own`,
  `recall:own − recall:C5tr`; the template split (the statement-overlapping first template against the paraphrase).
- **SQ2** the tool on hosts without a store: C0′, C2, P0 + `recall:C5` against their `none`; C5 `recall:own` − C0′
  `recall:C5` (does the channel add to the tool?); the C5ut host with `recall:own` and `recall:C5`.
- **SQ3** `cloze` items; by relation pair; on fully decoded pairs only.
- **SQ4** formats: `recall-all`, `recall-free`, `fields`, `noconf` (C5 host, seed 1 in the primary block; seeds 2–3 only if
  they differ from `recall:own` by more than 0.05).
- **SQ5** replications: SmolLM2-135M, Qwen3-0.6B / 1.7B (seeds 1–2; reference `roleless:own`), every arm against 0.5.
- **SQ6** the other item sets: natural T4 (role contrast), WP-UB two-hop (chained recall against `none` and `symbolic`) and
  reverse (lookup against `none`), new words (property accuracy against `none`, `symbolic`, `definition`; E9 dimension 3 and
  E11 numbers as context), each with its decode accuracy. Predictions: chained recall ≫ `none` on two-hop (the latent two-hop
  failure of Balesni et al.), bounded by hop 1 × hop 2; reverse lookup > `none`; new words `recall:own` ≈ `symbolic` ≫ `none`
  (E9's channel-only new-word property accuracy is 0.25 against chance 0.2).

## 5. F1 — faithfulness of decoding (channel route)

**Terms and items.** The first 300 T5 v2 new words (`e9n-0000`–`e9n-0299`), inserted with their gold frames (the host knows
them only through the channel), and their `property` items (one per templated relation; the gold is the relation's first
filler; 2 templates × 5 candidates). Role specificity uses the 300 twin pairs (`choice` items).

**Tested and decoded edges.** An edge e = (r, f) of a term is *tested* when the term has a property item of r whose gold
candidate words f. It is *decoded* when the run's own recall (static store of the inserted frame, typed cleanup, primary
unbinding) puts f among the top-m fillers of slot r (m = r's multiplicity); for the role-blind C5ut, when f is among the
bundle readout's top K (K = the frame's degree). Each arm is scored on its own decoded edges; all tested edges are a
secondary.

**Interventions** (all terms at once per frame position: a term's row depends on its own frame only):
- **remove e**: e's weight in the composer's weighted sum is set to 0, the other weights kept (decision 24: the bundle is
  normalized, so this renormalizes the rest; attention is not recomputed);
- **keep only e**: every other edge's weight set to 0.

**Measures.** The margin of filler f on its item, `m(f) = mean over templates of [s(f) − mean_{c ≠ f} s(c)]` (summed
log-probabilities; the null prompt is constant across interventions and cancels).
- **Comprehensiveness gap** `g_e = mean_{j ≠ e} m(f | remove j) − m(f | remove e)`: the matched random-edge control is the
  mean over every other single-edge removal of the same term (the expectation of removing a uniformly drawn other edge).
  `g_e > 0`: removing the decoded edge lowers its filler's preference more than removing another edge.
- **"Moves"**: `g_e > τ`, **τ = 0.05 nats** (bf16 replays of identical rows differ by ≤ 0.01 nats per score in E9 dimension 3,
  so a difference of two scores by ≤ 0.02; τ is 2.5 times that). Sensitivity: τ ∈ {0, 0.02, 0.1}.
- **Per-term comprehensiveness net share** `CF = mean over the term's decoded edges of [1(g_e > τ) − 1(g_e < −τ)]` ∈ [−1, 1]:
  0 when removing a decoded edge is exchangeable with removing another edge. The share `mean 1(g_e > τ)` ("the share of
  decoded edges whose ablation moves the model's preference for that filler, in the predicted direction, beyond the matched
  random-edge ablation") is reported with its mirror `mean 1(g_e < −τ)`.
- **Sufficiency** `m(f | keep e) − mean_{j ≠ e} m(f | keep j)`; per-term net share at τ.
- **Specificity**: e is specific when `Δ_e = m(f | remove e) − m(f | full) < −τ` and every other relation's item of the term
  moves less: `max_{r' ≠ r} |Δ_{r'}| < |Δ_e|` (Δ_{r'}: the change of the item's gold margin under the removal of e); the
  share of decoded edges that are specific.
- **Role specificity** (twins): remove twin A's (r1, X); X is A's gold under r1 and the distractor under r2. `RS =
  Δ_{r2}(X) − Δ_{r1}(X)` > 0 when X drops more under its own role. Per pair: the net share over its four removals (A r1,
  A r2, B r1, B r2) at τ, and the mean RS. A role-blind reader scores 0 in expectation: the twin design balances the two
  prompts' sensitivities across A and B. The swap of the twins' frames (step 1's `swap`) is recomputed with the same scorer.

**Statistics.** Per-term CF, terms × seeds crossed model (terms with ≥ 1 decoded tested edge in every run of the contrast);
Holm over F1a and F1b; the two-way bootstrap as the check. Planning: per-term SD of CF ≈ 0.5 gives a standard error ≈ 0.03 with
300 terms and 3 seeds (MDE ≈ 0.08 before the seed component).

**Predictions:**
- **F1a:** CF > 0, modest (≈ 0.1–0.4): the channel moves new-word property items (E9 dimension 3: C5 property 0.249 against
  `none` 0.195; edits of seen terms move `log p(new) − log p(old)` by +0.37 nats), so filler content is used.
- **F1b:** ≈ 0: the channel route reads filler content role-blindly (step 1), so binding should not change filler-level
  faithfulness.
- Sufficiency > 0; specificity moderate; **role specificity ≈ 0 for C5** (no role readout through the channel; step 1's
  `own − swap` ≈ 0) and for C5ut.

**Decision rule F1.**
- **F1a (a) "decoded edges are causally used"**: C5 CF > 0 (Holm p < 0.05); **(b) "not shown"**: CI includes 0; **(c)
  "anti-faithful"**: significant and negative.
- **F1b**: "binding changes edge-level faithfulness" by the sign of a significant C5 − C5ut; else "no difference at the
  filler level".
- **Joint reading with Q1** (stated in advance): F1a (a) with C5 role specificity ≈ 0 and Q1 (a) means *the decoded fillers
  explain the channel route, the decoded roles do not; the roles become behaviourally effective only through the recall
  route*. Then the explanation offered by decoding must be scoped: its role labels are inspection of the store, validated
  as causes only when read back through the tool.

**Refutation readings.**

| # | Observation | Reading |
|---|---|---|
| F-R1 | F1a (b) or (c) although the channel helps new-word property items (E9 dimension 3) | the store's effect is not carried by its edges as decoded (distributed or nonlinear use): decoded edges are not faithful explanations |
| F-R2 | undecoded tested edges as comprehensive as decoded ones | decoding and causal use dissociate: the probe does not track use |
| F-R3 | specificity near 0 with F1a (a) | the use is not edge-specific (a holistic salience effect) |
| F-R4 | C5ut role specificity ≠ 0 (CI excludes 0) | impossible by the twin design for a role-blind store: a bug |
| F-R5 | C5 role specificity > 0 (CI excludes 0) while step 1's twin contrast ≈ 0.5 | the channel reads roles weakly at the margin level but not enough to flip choices; report the effect size |

**Secondaries (family SF):** sufficiency, specificity, τ sensitivity, mean gap in nats, by decode confidence (tercile of the
cleanup cosine), by relation, undecoded edges, all tested edges; role specificity C5, C5ut, C5 − C5ut; `own − swap`; the
same on C5rf, C5tr, C5sh; SmolLM2-135M and Qwen3 C5 (seeds 1–2); the readout arms once trained (the readout reads the role
explicitly — its role specificity is predicted > 0).

## 6. Statistics common to every family

Contrasts are paired by unit (twin pair, term, item) over the units present in every run of the contrast; P0 has one run, which
stands for every seed of its comparison (its `recall:C5@s` is the seed-s value). Two-sided tests; Holm within each primary
family and within each numbered secondary; all intervals 95%. Secondary families are reported as such and never promoted.

## 7. Runs and order

Evaluation only, on finished `final.pt` checkpoints of the main checkout (`experiments/e9-retrofit/runs/…`). Per run: one
phase-A job per item set (GPU lane; `RUN/self-query-<items>/`) and, for composing runs, one F1 job (GPU lane;
`RUN/self-query-faithfulness/`); one report per stage and block (CPU lane; `experiments/e12-self-query/report/<stage>-<tag>/`,
never an E9 `report/<stage>`, amendment 12.4). Exact commands and GPU-h are in `execution.md` (E12) after the pilot.

## 8. Exclusions

- A twin pair counts only if both twins link to their new entries in every run of the contrast; natural anchors, WP-UB
  anchors and new words whose surface does not link are excluded (counts reported).
- F1 counts a term only if it has ≥ 1 decoded tested edge.
- A condition that cannot be built for an item (no definition writer for the track; a reverse item has no definition; a
  two-hop item whose gold path cannot be resolved) is not scored for that item (counts reported).

## 9. What this does not test

Generation (every phase-A item is forced choice or likelihood); whether the model would *choose* to call the tool (3a);
training on tool traces (3b); self-critique (3c); T1c (licensed: no item files; recall texts would hold licensed names);
WordNet and T1 (no role items).

## 10. Designs pre-registered now, run later

### 10.1 — 3a: agentic self-query

**Question.** Can a base LM decide when to call the recall tool, which call to make (whole frame, one role, a chain, a
reverse lookup), and use the result, with only few-shot ReAct-style prompting?

**Design.** Hosts: the Qwen3-0.6B-Base and Qwen3-1.7B-Base C5 runs (T5, LoRA, seeds 1–2), the tools reading the run's own
store. Prompt: an instruction line and three fixed worked demonstrations on *training* terms (one role query, one two-hop
chain of two calls, one reverse lookup), in the format `Question: … / Thought: … / Action: recall[<term>] |
recall[<term>, <relation words>] | lookup[<relation words>, <filler>] / Observation: <the phase-A recall text> / … /
Answer: …`, then the test question. Greedy decoding; the harness stops at `Observation:` and inserts the tool's result; at
most 3 calls and 48 new tokens per step. The answer is read as **forced choice** — the item's candidates scored after the
agent's trace and `Answer:` — and also by exact match of the generated answer.

**Items.** Twins (questions per relation, both twins), WP-UB two-hop and reverse (T5), new words (property).

**Measures.** Call-format success (the first action parses as a tool call naming a known term or filler), call relevance
(the term, and the relation of a role call, are the item's), share of episodes that answer, answer accuracy (twins:
contrast accuracy), against the same questions with no tool (few-shot direct answers) and against the fixed pipeline
(phase A's recall in the same question format).

**Endpoint (when run).** A1: twin contrast accuracy, agentic − no tool (Qwen3-1.7B, seeds 1–2, pairs × seeds); A2: agentic −
fixed pipeline (the cost of choosing the calls); Holm over A1 and A2. Predictions: A1 > 0 only if call-format success ≥ 0.8;
A2 < 0. **Feasibility pilot now** (§15): about 50 questions on Qwen3-0.6B (1.7B if it fits ≤ 4 GB in bf16).

### 10.2 — 3b: learning from tool-using traces (outline; concrete design and cost in §12)

LoRA arms trained on traces generated by the fixed pipeline on *training* terms: one arm learns to call the tool and use it,
one learns the answers without the tool (internalization), with a matched-token control; the twins are rerun **without the
tool**: does practice with recalled roles teach the model to read roles from the channel?

### 10.3 — 3c: calibrated self-critique (outline; concrete design and cost in §13)

The model compares its decoded beliefs (recall with calibrated confidences) with its behaviour (its answer without the tool)
and with held-out data, flags or revises disagreements, and is scored for calibration and selective accuracy, with a
**null-world control**: a store of false structure (shuffled or corrupted frames) that the loop should flag rather than
accept (E10 accepted false structure without retracting it).

## 11. Smoke tests and pilots

The labelled pilots of §15 run after this document's commit, under the GPU limits of the shared machine (≤ 4 GB, ≤ 5 min per
job; CPU preferred), on T5 SmolLM2-360M seed 1 only (Q1: C5, C5ut, C0′, P0 + the C5 store; F1: C5, C5ut), and the 3a
feasibility pilot on Qwen3. They report timing, pipeline checks and **PILOT** numbers; nothing in §§1–10 changes because of
them, and they never enter an endpoint.

## 12. 3b — learning from tool-using traces: concrete design and cost (pre-registered 2026-10-08, before any 3b run)

*Amended 2026-10-09 by 16.5 (decision 64), before any 3b run: the hypothesis is restated as "LoRA learns to use the decoded
store", and a C0′-host arm and a C5rf-store arm are added as equivalence secondaries. B1 and B2 are unchanged.*

**Question.** If the model practises answering role questions with its own recalled frame in context, does it learn to read
roles from the channel — telling role-swap twins apart **without the tool** (internalization)? And does a model trained on
tool-using traces learn to call the tool well?

**Host and what trains.** The T5 SmolLM2-360M C5 runs, seeds 1–3 (the Q1 hosts). Rank-16 LoRA adapters on the host's
attention and MLP projections (`integrations.transformers.add_lora`, the default targets); the host's weights and the whole
channel (composer, projector, gate, P1 context) frozen — the arms may learn to *read* the store, not change it. One
control arm on the C5ut runs (below).

**Training data** (built by the fixed pipeline; no test term, no test name, no test frame):
- *training twins*: 2,000 new pairs built as the test twins (`e9_binding_items.build_twins`, the same two relation pairs),
  with item seed 1 and name seed 29, names disjoint from every E9 item set, frames disjoint from the test twins' frames, and
  no (X, Y) filler pair of a test twin;
- *seen terms*: every seen T5 entry (training frequency ≥ 10, not held out, not an anchor of any E9/E12 item set), one
  question per templated relation of its frame whose options are the gold filler and the term's own filler of another
  relation of the same answer type when it has one (else a frequency-weighted filler of the relation's pool) — so a
  role-blind answer fails;
- questions use the property templates' first two paraphrases (the twins' `choice` wordings); the held-out wording (`cloze`)
  is never trained.

**Arms** (the same questions; loss on the model's own turns only — never on questions or observations):
- **T (tool traces):** the 3a format — `Question … Thought … Action: recall[<term>, <relation>] … Observation: <the store's
  recall> … Thought … Answer: <gold>`;
- **I (internalize):** `Question … Answer: <gold>` with no tool;
- **S (stepwise internalization, Deng et al. 2024):** T's traces for the first epoch, then with the observation removed, then
  with the action removed (I's format in the last epoch);
- **L (control):** LoRA on the same number of tokens of the T5 training text (any update of the same size);
- **I-ut:** arm I on the C5ut runs (whose twins have identical stores): any twin gain there is not role reading.

**Training budget.** 3 epochs; AdamW, lr 2·10⁻⁴, 32 sequences per step, warmup 3%; tokens per epoch ≈ 2.4 M (T, S) and
0.45 M (I, I-ut); L matches T's tokens. SmolLM2-360M LoRA trains at ≈ 12k tokens/s (`e9_plan` throughput model): T and S
≈ 0.17 GPU-h, L ≈ 0.17, I and I-ut ≈ 0.04 per seed.

**Tests** (every arm, seeds 1–3): the twins **without the tool** (`none`: the channel only), `choice` and `cloze`; the twins
with the fixed pipeline (`recall:own`); for arm T, the agentic episodes of 3a; new words (property, no tool); WP-UB two-hop
(no tool and chained recall).

**Endpoints.**
- **B1 (internalization):** twin contrast accuracy without the tool, **I − L** and **S − L** (pairs × seeds; Holm over
  the two).
- **B2 (tool use learned):** arm T's agentic twin contrast and call-format success against 3a's few-shot episodes on the
  same host size (descriptive; T − I on the agentic twins as the test).

**Predictions.** B1 > 0 but well below the tool (≤ 0.65 against `recall:own` ≈ 0.8): unbinding a learned-HRR role is a
linear map of the injected vector, so LoRA can learn it, but the T5 LM loss never asked the host to, and 2,000 twins pairs
are few. S ≥ I. I-ut ≈ 0.5 (its stores cannot separate twins). B2: T calls well-formed tools ≥ 0.9 of the time.

**Refutation readings.** I-ut > 0.5 (CI excludes 0.5): the training leaked the answers through names or templates — read
B1 only after finding the leak; I > L but the cloze (never trained) wording at 0.5: the arm learned the trained wording, not
the role; T's gain without the tool equal to I's: the traces add nothing beyond the answers.

**Cost.** Training (the queue's GPU, alone: `e9_plan`'s throughput model, ≈ 12k tokens/s for SmolLM2-360M LoRA) ≈ 3 seeds ×
(T 0.17 + S 0.17 + I 0.04 + L 0.17) + I-ut 3 × 0.04 ≈ **1.8 GPU-h**. Evaluation (phase A's pilot throughput, an upper bound):
per arm and seed the twins without and with the tool (`none`, `recall:own`; ≈ 3 min), new words without the tool (≈ 1 min),
two-hop without the tool (≈ 1 min) — 5 arms × 3 seeds ≈ 1.3 GPU-h — and arm T's agentic episodes on 100 twin pairs (the r1
question of both twins; ≈ 4 s per question measured on Qwen3-0.6B in the 3a pilot) ≈ 0.7 GPU-h. **Total ≈ 3.8 GPU-h.** Code:
a trace builder and a LoRA trainer on a finished run (`e12_traces`, ≈ one day with tests on the toy world).

## 13. 3c — calibrated self-critique: concrete design and cost (pre-registered 2026-10-08, before any 3c run)

*Amended 2026-10-09 by 16.5 (decision 64), before any 3c run: the same loop with the gold relations as text and with the
prose definition in the prompt, token accounting, and K1b (loop(store) − loop(definition), non-inferiority at −0.05).
K1 and K2 are unchanged.*

**Question.** Can the model check its decoded beliefs against its behaviour and against held-out evidence, keep the right
ones, flag or revise the rest, with calibrated confidence — and **reject false structure** in a world where the store holds
it (E10 accepted false structure without retracting it)?

**Beliefs and their confidence.** A belief is a decoded edge of phase A's recall (term, relation, filler, cleanup cosine).
Its confidence is calibrated per relation: P(decode correct | cosine, margin to the second candidate, frame size), isotonic
regression fitted on the store's seen entries (decode correctness is known there) and applied to test terms. Calibration is
reported on held-out terms and new words: ECE (15 bins), Brier score, AUROC of correct against wrong decodes.

**Behaviour.** The model's answer to the item without the tool (phase A `none`), with its own confidence (the softmax of the
candidates' scores).

**Held-out evidence.** For T5 held-out terms (never linked in training, present in the evaluation text): one evaluation-corpus
sentence about the item's relation, put in context; for new words, none (the honest limitation).

**The loop.**
- *Rule loop (the information available):* answer with the belief when its calibrated confidence p ≥ θ, else with the
  behaviour; flag the item when belief, behaviour and evidence disagree; θ is chosen on a seeded dev half of the new words
  and applied to the other half and to the held-out terms (the T5 WP-C7 zero-shot property items of the held-out terms).
- *Model loop (Qwen3-1.7B-Base C5, seeds 1–2):* a few-shot critique prompt shows the question, the model's own answer, the
  recall with p and the evidence sentence, and asks `Keep`, `Revise to <option>` or `Unsure`; scored by forced choice over
  the three decisions and the options.

**Null world (the control).** The same items with a **corrupted store**: each test term's frame replaced by a same-type
false frame (fillers of each relation redrawn frequency-weighted, as E9's `random_frame`; roles swapped within the twins'
pairs), composed and decoded exactly as the real store. The decode stays faithful to the store, so p stays high: calibration
against the store is not calibration against the world. Measures: the false-belief adoption rate (final answer = the false
recalled filler), the flag rate, and p's calibration against the *world* in both worlds.

**Endpoints (when run).**
- **K1:** accuracy at 80% coverage — each method abstaining on its own lowest-confidence 20% (the loop by the calibrated p
  of its chosen answer, the no-tool answer by its softmax) — of the rule loop − the no-tool answers (the test half of the
  new words and the held-out terms; T5 360M C5, seeds 1–3; items × seeds).
- **K2:** false-belief adoption in the null world, rule loop − naive recall (always adopt the belief); Holm over K1 and K2.
- Secondaries: ECE / Brier / AUROC of calibrated p; the model loop's K1 and K2 (Qwen3-1.7B, seeds 1–2).

**Predictions.** K1 > 0 (beliefs decoded at ≈ 0.96 beat behaviour at ≈ 0.25 on new words). K2 < 0 on seen and held-out
terms (behaviour and evidence disagree with false beliefs) but ≈ 0 on new words (no behaviour or evidence signal): the loop
can only reject false structure where it has an outside signal, as Huang et al. (ICLR 2024) found for self-correction without
external feedback.

**Refutation readings.** K2 ≈ 0 on held-out terms with evidence in context: the loop does not use evidence against its own
store (the E10 failure mode, now measured); calibrated p's ECE on test terms much worse than on seen terms: calibration does
not transfer to composed (zero-shot) stores.

**Cost** (phase A's pilot throughput, upper bounds). Calibration and the rule loop reuse phase A's `none` and `recall:own`
scores (CPU). New GPU work: the null-world recall condition on the twins and the 300 new words (≈ 8.5 min per run; C5 and
C5ut × 3 ≈ 0.85 GPU-h); the evidence condition on the held-out terms' property items (≈ 15 min per run; C5 × 3 ≈ 0.75
GPU-h); the model loop (Qwen3-1.7B-Base C5 in bf16, ≈ 600 items × 2 worlds × 2 seeds as forced choice over the decisions,
≈ 1.5 s each, ≈ 1.0 GPU-h). **Total ≈ 2.6 GPU-h.** Code: the corrupted-store condition, the evidence sentences, the
calibration fit, the loop and its report (`e12_critique`, ≈ one day with tests).

## 14. Open decisions for the author

| # | Decision | Default |
|---|---|---|
| SQ-1 | Q1's primary context is the whole-frame, slot-aware recall in statement wording with confidences | as registered |
| SQ-2 | The role-blind primary reference is the C5ut store's bundle readout (decision 62's spec); `roleless:own` (format-matched) is a secondary, and the reference on hosts without a C5ut run | as registered |
| SQ-3 | PMI uses the context-free null; the twin contrast needs none | as registered |
| SQ-4 | WP-UB items capped at 150 anchors per subset and new words at the first 300 (cost) | as registered |
| SQ-5 | F1's τ = 0.05 nats; the matched control is the mean over every other single-edge removal | as registered |

## 15. Pilots (PILOT; not endpoints)

Run after `6638c39` (this document's first commit) on real checkpoints of the main checkout (read only), outputs in
`experiments/e12-self-query/pilot/` (never in a run folder): T5 SmolLM2-360M **seed 1 only**; the twins' `choice` items only;
the GPU shared with a training job at 100% (15.6 GB in use), every job ≤ 4 GB and ≤ 5 minutes (peak 3.5 GB). **These numbers do
not count toward Q1 or F1**, which use seeds 1–3 in the queued runs. Report: `pilot/report/t5-q1-f1/report.md`.

**Phase A — the twins** (300 pairs; twin contrast accuracy, chance 0.5; decode = the share of the four critical slots the
context states; "all four" = pairs whose four critical slots are all right):

| host | condition | contrast | decode | all four |
|---|---|---:|---:|---:|
| C5 | `none` | 0.507 | — | — |
| C5 | `recall:own` | **0.790** | 0.959 | 0.857 |
| C5 | `recall:C5ut` (role-blind store) | 0.495 | 0.897 (fillers) | — |
| C5 | `recall:C5tr` (translation store) | 0.497 | 0.443 | 0.000 |
| C5 | `roleless:own` | 0.497 | 0.959 (fillers) | — |
| C5 | `symbolic` (gold frame) | 0.863 | 1.000 | 1.000 |
| C5 | `definition` (E11 prose) | 0.748 | — | — |
| C5 | `wrong:own` (partner's recall) | 0.228 | 0.000 | 0.000 |
| C5ut | `none` / `recall:own` / `recall:C5` / `symbolic` | 0.500 / 0.485 / **0.798** / 0.864 | | |
| C0′ | `none` / `recall:C5` / `roleless:C5` / `symbolic` | 0.488 / **0.794** / 0.498 / 0.868 | | |
| P0 | `none` / `recall:C5@1` / `wrong:C5@1` / `symbolic` | 0.492 / **0.690** / 0.311 / 0.730 | | |

Q1 as it would read on this seed (one-sample t over 300 pairs): `recall:own − none` +0.283 [+0.252, +0.313], `recall:own −
recall:C5ut` +0.295 [+0.266, +0.324] — reading (a), "text-driven" (`wrong:own` 0.228). Pairs with all four slots decoded: 0.838
(257 pairs); with a decode error: 0.506 (43 pairs) — the gain tracks the decoded content (Q-R5 not met). `symbolic − recall:own`
+0.073 (the decode's cost); `definition − symbolic` −0.116 (prose in held-out wording reads worse than statements).
**The store, not the host, carries the roles:** the bound store's recall works on every fine-tuned host (C5 0.790, C0′ 0.794,
C5ut 0.798; C5 − C0′ −0.004 [−0.019, +0.011]), the role-blind store's on none (0.495, 0.485). The base host reads the
statements less well (P0: `symbolic` 0.730, `recall:C5` 0.690).

**Cost** (per condition, 4,800 texts of ≈ 230 tokens, effective batch 12, the GPU shared): recall-type conditions 63–91 s
(`recall:own` 86 s, `symbolic` 67–85 s, `wrong` 63–92 s, `definition` 74 s), shorter contexts less (`recall:C5ut` 63–70 s,
`roleless` 34–54 s), `none` 8–24 s (with the null cache); job overhead ≈ 20 s; peak 3.5 GB.

**F1 — faithfulness of decoding** (300 new words, `e9n-0000`–`e9n-0299`, three 100-word parts merged; τ = 0.05 nats; per-word
means; one-sample or paired t over the 300 words):

| measure | C5 | C5ut |
|---|---:|---:|
| decoded tested edges (of 1,529 tested edges) | 1,497 | 1,470 |
| comprehensiveness net share | **+0.342** [+0.299, +0.385] | +0.392 |
| moved beyond the matched control / the other way (share) | 0.599 / ≈ 0.26 | 0.640 / ≈ 0.25 |
| mean gap (nats) | +0.375 | +0.538 |
| sufficiency net share | +0.266 | +0.299 |
| specificity (share of decoded edges) | 0.222 | 0.239 |
| τ = 0 / 0.02 / 0.1 | +0.364 / +0.354 / +0.322 | +0.405 / +0.402 / +0.383 |
| twins: role specificity (net share; mean RS) | −0.036 [−0.091, +0.020]; −0.021 nats | +0.008; +0.001 nats |
| twins: contrast own / swapped frames (recomputed step-1 `swap`) | 0.504 / 0.501 | — |

F1 as it would read on this seed: **F1a (a)** — removing a decoded edge moves its filler's preference more than removing another
edge (+0.342); **F1b** C5 − C5ut −0.050 [−0.098, −0.002] (the untyped store's decoded fillers are, if anything, a little more
comprehensive). Role specificity is ≈ 0 on both stores: through the channel, the model uses the decoded **fillers** but not
the decoded **roles** — the joint reading with Q1 stated in §5 (the roles act only when read back through the tool). C5's 29
words with an undecoded tested edge: +0.07 [−0.23, +0.37] (too few to read). **Cost:** 300 words in 520 s plus the twins'
three passes in ≈ 30 s (2.3 GB peak; batch 64 with the text cache).

**3a — agentic self-query, feasibility** (Qwen3 T5 seed 1; 50 questions per host: both twins of 10 pairs on their first
relation, 15 two-hop, 15 reverse; three verified demonstrations; greedy decoding; hosts in bf16 (≤ 3.8 GB); answers read as
forced choice; outputs `pilot/agent/<host>/merged`). Twin contrast over the 10 pairs; accuracy over all 50 questions.

| host (store) | call well-formed | swapped tool | relevant call | answers itself | accuracy agent / no tool / fixed | twin contrast agent / no tool / fixed |
|---|---:|---:|---:|---:|---|---|
| Qwen3-0.6B C5 (own) | 0.52 | 0.24 | 0.44 | 0.52 | 0.60 / 0.52 / 0.62 | 0.40 / 0.50 / 0.80 |
| Qwen3-1.7B C5 (own) | 0.02 | 0.30 | 0.02 | 0.64 | 0.46 / 0.48 / 0.64 | 0.30 / 0.30 / 0.80 |
| Qwen3-0.6B-Base P0 (C5's store) | 0.82 | 0.00 | 0.76 | 0.88 | 0.72 / 0.58 / 0.92 | **1.00** / 0.60 / 0.90 |
| Qwen3-1.7B-Base P0 (C5's store) | **0.84** | 0.00 | 0.84 | 0.92 | 0.76 / 0.58 / 0.88 | **1.00** / 0.50 / 1.00 |

**Reading (PILOT, 10 pairs and 30 understanding items per host — feasibility, not effect sizes).** A *base* LM emits
well-formed recall calls from three demonstrations (≥ 0.82; twins and two-hop 0.80–1.00, reverse lookups 0.47–0.67) and uses
the results: twin contrast 1.00 against 0.50–0.60 without the tool, two-hop 0.53–0.60 against 0.33–0.40 (five options). The
E9 **C5 runs do not**: LoRA fine-tuning on the T5 corpus broke few-shot format following — the 0.6B swaps the two tools' names
(`lookup[term, relation]`), the 1.7B answers "Action:" with the corpus's meeting-note register ("action item: … to follow up
on …", "The previous notes are archived.") — so their agentic episodes fall below their own fixed pipeline (0.80). The store
is host-agnostic text (phase A: C0′ and the C5ut host read the C5 store as well as C5 does), so the self-query does not need
the host that wrote the store. **Cost:** ≈ 4 s per question on the 0.6B and 8 s on the 1.7B, the two baselines included.

## 16. Amendments

### 16.1 Cost profiles, scorer speed-ups and pilot options (2026-10-08, after `6638c39`; no endpoint, unit, rule or threshold changed)

Decided from the pilot's timings (§15), before any endpoint run:
- **Cost profiles.** The primary block — T5 SmolLM2-360M twins on C5, C5ut and C0′ × seeds 1–3 and P0 — runs every condition of
  §3. The secondary arms (C5tr, C5rf, C5sh, C2), the replications (SmolLM2-135M, Qwen3), the readout arms, the T5 new-word set
  and the WP-UB sets on T4 run the **core** conditions: `none`, the recall, `symbolic`, `roleless` (and on C5 the role-blind
  stores) — not `definition` or `wrong` (`e12_self_query queue --core`). Reason: on the shared GPU a full condition set costs
  ≈ 15 minutes per 360M run.
- **Speed-ups that do not change a score.** F1 tokenizes and links each text once and reuses it across interventions
  (`e12_faithfulness.TextCache`; equal to the standard scorer within 10⁻⁴ on the CPU, `tests/test_e12_self_query.py`); the
  phase-A context batch of SmolLM2-360M is 24 (12 per forward with a context; 3.5 GB peak).
- **Pilot-only options.** `--item-kinds` (phase A; the pilot scored the twins' `choice` items) and `--offset` (F1; the pilot
  split 300 words into three ranges) keep pilot jobs under 5 minutes; the report merges a run's split folders
  (`<folder>-<tag>`). Endpoint jobs score every item kind, and F1's first 300 words in one job.
- **Wording (§2).** An untemplated one-word relation that is a verb in -s keeps its words ("X replaces Y."); another one-word
  relation reads "X has <relation> Y." (T4's `charge`, `branch`). Every relation the T5 items test is templated.
- **3a demonstrations** are verified against the tool's real output: their recalls decode the gold fillers and their lookup
  lists the answer and not the other option (a first draft showed a lookup that did not list its answer).
- **Reporting.** A role-blind context (no role stated) reports its filler-level decode only; the report adds a per-model F1
  table (the readout arms, the secondary arms).

### 16.2 3a's hosts after the feasibility pilot (2026-10-08, before any 3a endpoint run; Q1 and F1 unchanged)

The 3a pilot (§15) shows that the E9 C5 runs (Qwen3 LoRA on the T5 corpus) no longer follow a few-shot tool protocol (well-formed
first calls 0.52 and 0.02), while the untouched base hosts do (0.82 and 0.84). The 3a endpoints (§10.1) therefore read:
- **Primary agent host:** Qwen3-1.7B-Base (P0, untouched) whose tools read the Qwen3-1.7B C5 store of seeds 1–2 (as phase A's
  P0 + `recall:C5@s`); **A1** agentic − no-tool twin contrast, **A2** agentic − fixed pipeline, on that host (pairs × seeds;
  Holm over A1 and A2). Qwen3-0.6B-Base the same, as a secondary.
- **Secondary (self-hosted):** the C5 runs with their own store, reported as the cost of fine-tuning on the store's domain (the
  pilot's format collapse), together with 3b's arm T, which trains the protocol back in.
- Questions as in the pilot, at full size: both twins of all 300 pairs on their first relation, 150 two-hop and 150 reverse
  items (one per anchor, seeded), 3 demonstrations, greedy decoding, forced-choice answers; hosts in bf16 (the 1.7B fits 4 GB
  with the output head computed in vocabulary slices). Cost at the pilot's rate (8 s per question on the 1.7B with both
  baselines): 900 questions × 2 seeds ≈ 4.0 GPU-h for the 1.7B, ≈ 2.0 for the 0.6B, ≈ 3.2 for the two self-hosted C5 runs at
  seed 1 — ≈ 9 GPU-h in all (an upper bound from the shared GPU).

### 16.3 3b and 3c: the harnesses' fixed choices (2026-10-08, before any 3b or 3c run; no endpoint, unit, contrast or threshold changed)

Written with the harnesses (`vsa_embed.experiments.e12_traces`, `e12_critique`; tests `tests/test_e12_traces_critique.py`, CPU)
and before any 3b or 3c job; only a labelled CPU SMOKE ran (SmolLM2-135M C5 seed 1 and Qwen3-0.6B-Base C5 seed 1, a handful of
items, outputs `pilot/smoke/`; not an endpoint). §12 and §13 leave the choices below open, or could not be implemented as
written (items 9b and 10, said so there). Every open value is fixed **by rule**; none was tuned on any data, so 3b needs no
dev split. B1, B2, K1 and K2 are unchanged.

**3b (§12).**
1. *Training questions* (built once, committed: `items/traces-t5-smollm2-v1`; 14,245 questions, every disjointness check 0):
   one question per (term, relation), worded with one of the property templates' first two paraphrases drawn with seed 0
   (one draw per question is what §12's ≈ 0.45 M-token I budget implies; measured 0.53 M); twin options [X, Y] as the test
   twins, seen-term options in a seeded order. Training twins: 2,000 pairs (seed 1, name seed 29, the test set's relation
   pairs depends_on | part_of and owned_by | approved_by); names reserved against every item directory under
   `experiments/e9-retrofit/items` (10,701 names), frames against every T5 test frame (twins and new words of every family,
   1,300), {X, Y} against the test twins (288 pairs). "Not an anchor of any E9/E12 item set" is read literally — no entry
   that a T5 item set of any tokenizer family names (WP-UB anchors, bridges and reverse partners; the edit sets' edited
   entries): 3,005 entries are named, leaving **955 of the 2,845 seen entries** (6,245 questions). "Same answer type" =
   `TrackLexicon.atom_type` (a term filler by its term type); 78 questions find such an own filler, the rest take §12's
   frequency-weighted pool filler.
2. *Trace wording* (T): `Thought: I should recall <term>, <relation words>.` / `Action: recall[<term>, <relation words>]` /
   `Observation: <3a's tool output for that call>` / `Thought: The recall says <option>.` (`The recall names neither option.`
   when the decoded filler is no option: 3.5% of the questions on the 360M C5 seed-1 store; the other option: 0.01%) /
   `Answer: <gold>` — the gold always, as §12 writes. The loss covers exactly what 3a's harness lets the model generate (the
   first thought and the action up to `]`; the second thought, `Answer:` and the answer), never the question, the
   observation or the labels the harness inserts. S's second epoch drops the observation and keeps both thoughts and the
   action; its third is I's format. Tokens per epoch (SmolLM2 tokenizer): T 1.77 M (mean 124, max 174 per sequence),
   T without observation 1.20 M, I 0.53 M.
3. *L*: per epoch as many sequences as T, each a window of T's mean length (124 tokens) of the T5 training corpus — T's
   tokens per epoch matched; fresh seeded windows each epoch, held-out entries unlinked as in training, LM loss on every token.
4. *Optimizer values §12 leaves open*: LoRA α = 32 (`add_lora`'s default; scale 2); AdamW betas (0.9, 0.95), weight decay 0.1,
   gradient clip 1.0, and after the 3% warmup the E9 trainer's cosine decay to 10% (`training.lm._lr`); one sequence per
   question (no packing); the loss the mean over a step's loss tokens; data order seeded by (run seed, epoch); bf16 autocast
   as the E9 runs; the host in eval mode (no dropout). Training twins are inserted with their gold frames and linked while
   training, as at evaluation. (A host that already carries LoRA would have it merged into its frozen weights first; the
   SmolLM2-360M runs carry none.)
5. *Tests added* (secondaries): the twins in the trained question format without the tool (3a's `no_tool` prompt, both twins
   of all 300 pairs, first relation), to tell "learned the role" from "learned the format"; a `base` arm (the untrained run,
   the same tests): B2's "3a's few-shot episodes on the same host size" — 3a ran only on Qwen3, so 3a's harness is run on the
   SmolLM2-360M C5 runs; the agentic episodes also for arm I, since B2's test is T − I on the agentic twins (§12 costed arm T
   only; I-ut runs none, its tool reading a role-blind store). Agentic: both twins of the first 100 pairs, first relation,
   3a's prompt and demonstrations.
6. *Report* (`e12_traces report`): B1 = I − L and S − L on the twins' `choice` contrast without the tool (pairs × seeds, Holm
   over the two); secondaries (family SB, never promoted): T − L, T − I, S − I, I-ut − 0.5 (the leak check), the cloze
   wording, the question format, `recall:own`, new words, two-hop (no tool and chained recall) against L, L − base; B2
   descriptive by arm and its T − I test; §12's predictions checked as written.

**3c (§13).**
7. *Calibration model*: isotonic regression is one-dimensional, so "P(decode correct | cosine, margin, frame size), isotonic
   regression" is a per-relation logistic score of the three features (the cleanup cosine; the margin to the next candidate,
   the (m+1)-th of the slot; log frame size; standardized, ridge 1) followed by isotonic regression of correctness on that
   score; a relation with < 50 decoded seen edges or < 5 of a class uses the pooled model. Seen entries: ≤ 3,000 (phase A's
   `seen_entries`, seed 0). Calibration is reported against the store and against the world, in both worlds.
8. *An item's belief*: of slot r's top-m decoded fillers (m = r's multiplicity in the term's frame; a role-blind store: the
   type-restricted bundle readout), the best one that is an option of the item; none when no decoded filler is an option.
9. *Null world*: (a) the store the tool reads is corrupted; the channel's injection and the behaviour stay the real world's
   (§13's cost reuses phase A's `none`). (b) As written (fillers redrawn frequency-weighted, as E9's `random_frame`) the false
   filler of the tested relation would almost never be one of the item's options (term pools hold hundreds of fillers), so
   false-belief adoption could not be measured; the tested edge (the item's gold) is therefore redrawn among the item's
   distractors (frequency-weighted) and the other edges follow E9's random frame (new words: the frame stored with the item
   set; held-out entries: `e9_tracks.random_frames`, seed 0), no edge of a tested relation keeping the gold. Twins: roles
   swapped within the pair, as written. Null frames are seeded per term and the same in every run.
10. *Rule loop*: the answer is the belief when p ≥ θ, else the behaviour (as written). **Flag** = the available signals
    (belief, behaviour and, on held-out terms with a sentence, the evidence answer: the host's answer with the sentence in
    context) do not all agree. §13 does not say what a flag does; at a fixed coverage it acts by abstention: a flagged item
    is abstained before any unflagged one. Each method then abstains on its own lowest-confidence 20%: the loop by its
    answer's calibrated confidence (the belief's p; for a behaviour answer the behaviour's softmax calibrated by isotonic
    regression on the dev half), no-tool by its softmax, naive recall by p (it always answers with the belief when it has
    one, else with the behaviour). θ ∈ {0, 0.05, …, 1} maximizes the loop's accuracy at 80% coverage on the dev half (ties:
    the smallest), per run. Dev / test halves: the first 300 v2 new words split by word, seed 0.
11. *K1 / K2 units*: per item `1(answered ∧ correct) / 0.8` and, in the null world, `1(answered ∧ answer = the false option) /
    0.8` (each method abstaining 20%); their means are the accuracy and the false-belief adoption at 80% coverage, so the
    items × seeds crossed model of their paired differences tests K1 and K2 as defined. Pool: the new words' test half and
    the held-out terms' WP-C7 property items (1,006 items, three paraphrases, four options). Per-set results (new words,
    held-out, twins) are secondaries: §13's predictions (K2 < 0 on held-out terms, ≈ 0 on new words) and refutation readings
    are read on them.
12. *Evidence*: the first evaluation-corpus sentence (document order) that names the term and exactly one of the item's
    options (`items/critique-evidence-t5-v1`, built once: 750 of the 1,006 items have one; 740 of the 750 state the gold).
13. *Model loop* (secondary, Qwen3-1.7B-Base C5 seeds 1–2 as registered): 300 items per set (seeded; the new words' test half,
    the held-out items) in both worlds; four demonstrations on seen entries with real decodes and their p (Keep; Revise; Keep
    against a real decode error, the evidence being the relation's training statement; Unsure on a low-p decode); forced
    choice over " Keep" / " Revise" / " Unsure", then after "Revise to" over the other options; Unsure abstains; confidence =
    the decision's probability (× the option's for Revise).
14. *Added secondaries*: the loop with evidence revision (a disagreeing evidence answer replaces the loop's); the host reading
    the false recall in context against naive recall (K2); `recall:own` on new words and held-out terms (the fixed pipeline);
    C5ut (role-blind store) on the twins and the new words.

**Cost** (re-estimated from the measured token counts, e9_plan's 12k tokens/s for SmolLM2-360M LoRA and phase A's / 3a's
measured rates on the shared GPU; upper bounds). 3b: training 52 M tokens (per seed T 5.3 M, S 3.5 M, I 1.6 M, L 5.3 M, I-ut
1.6 M) ≈ 1.2 GPU-h, ≈ 1.6 with padding; tests ≈ 7 min per arm and seed (18 jobs) ≈ 2.1; agentic episodes for T, I and base
(200 questions × ≈ 3 s, 9 jobs) ≈ 1.5 — **≈ 5.2 GPU-h** (§12: 3.8; the difference is item 5's added episodes and tests).
3c: C5 × 3 (twins, new words, held-out; every condition) ≈ 25 min each, C5ut × 3 (twins, new words) ≈ 14 min each, the
model loop ≈ 0.5 GPU-h per seed — **≈ 3.0 GPU-h** (§13: 2.6). Commands: `queue-commands-3bc.sh` (not queued).

### 16.4 3b and 3c: the author's decisions on the harness questions (2026-10-08, before any 3b or 3c run)

Decided by the owner session under decision 63's delegation. No endpoint, unit, contrast or threshold changes.
- **K2 keeps its registered rule.** Under that rule, adoption is likely ≈ 0 to be separable: in the CPU smoke ≈ 97% of
  null-world items were flagged. The evidence-override variant (smoke: adoption 1.0 → 0.15 on held-out terms) stays a
  reported secondary. It is not promoted, because it was seen on smoke data.
- **3b's seen terms keep the literal rule:** an entry that anchors no item set. That leaves 955 entries.
- **The model loop also runs on the untouched Qwen3-1.7B-Base host reading the C5 store,** seeds 1–2, as a secondary
  (as 16.2 did for 3a; ≈ +1.0 GPU-h).
- **Accepted:** the GPU total above the registered estimate (3b ≈ 5.2, 3c ≈ 3.0 + 1.0 GPU-h).

### 16.5 3b and 3c after Q1: what is learned, and text competitors (decision 64; 2026-10-09, before any 3b or 3c run)

Decision 64 (author, 2026-10-09), after Q1 at three seeds and T4's natural items. On 2026-10-09 every 3b / 3c job was still
pending (`.jobs`: 3b's 18 GPU jobs at 54.4497, 3c's 10 at 54.4498, both reports at 54.44985); none had started. Q1 found:
the store's decode in context 0.78 against 0.50 without it; a C0′ host reading the HRR store 0.79 (C5 − C0′ −0.01, n.s.) —
the roles live in the store, not the host; a fixed random binding decodes as well as a learned one (P2, learned − fixed
−0.006 [−0.016, +0.003]); on T4's natural items (seed 1) the store 0.88, the gold relations as text 0.89 and the prose
definition 0.91 — on real text the store's advantage is its token cost (no curated definition), not accuracy. An audit of
§12 / §13 against these results found: 3b's prediction rests on the host learning "unbinding a learned-HRR role", with no
arm that varies the host or the binding; 3c's K1 beats only the no-tool answer — no text competitor and no token
accounting. This amendment adds both. **B1, B2, K1 and K2, their units, pools, rules and Holm families are unchanged.**

**3b (§12).**
1. *The hypothesis, restated.* §12's rationale — "unbinding a learned-HRR role is a linear map of the injected vector, so LoRA
   can learn it" — is withdrawn as a claim about a *learned* binding. The hypothesis is now: **LoRA learns to use the decoded
   store** — read in context (arm T: the fixed pipeline and the agentic tests) and, for internalization (B1), from the
   channel's injection of the same store without the decode. Reading a role out of an injected bundle is a linear map
   whatever the binding, learned or fixed random, so §12's prediction values stand (B1 > 0 but ≤ 0.65; S ≥ I; I-ut ≈ 0.5;
   T's calls well-formed ≥ 0.9).
2. *Two hosts added*, with §12's training items, hyperparameters and tests and 16.3's harness (`e12_traces run`):
   (a) **C0′ host** — `runs/t5/SmolLM2-360M-full-C0p-s1..3` (no channel). Its tool reads **C5's store** of the same seed
   (`--store`), as in Q1. Arms **T** and **base**. I, S and L are not run: without a channel nothing can be internalized
   (twins differ only in the injected frame), so they would repeat I-ut's leak check.
   (b) **C5rf** — `runs/t5/SmolLM2-360M-full-C5rf-s1..3`, reading its own fixed-random store. Arms **T**, **I** and **L**;
   I runs without agentic episodes, since B2 is read on C5. S is not run: its curriculum concerns the trace format, not the
   binding, and B1's I − L carries the binding question.
3. *Predictions and contrasts* (secondaries, family S64; reported, never promoted). Each is a pairs × seeds crossed-model
   contrast, read by **equivalence**: two one-sided t tests at α = 0.05 (the 90% CI inside ±δ), no multiplicity adjustment.
   - **C0′ host ≈ C5 host** with the decoded store: T@C0′ − T@C5 on (i) the twins with the fixed pipeline (`recall:own`, C5's
     store; 300 pairs) and (ii) the agentic twin contrast (100 pairs); δ = 0.05.
   - **C5rf ≈ C5**: (iii) B1's contrast, (I − L)@C5rf − (I − L)@C5 (twins, no tool), δ = 0.075; (iv, v) T@C5rf − T@C5 on the
     same two decoded-store tests, δ = 0.05.
   - Read with them (controls): T@C0′ and base@C0′ without the tool ≈ 0.5 (CI includes 0.5; no channel); T@C0′ − base@C0′
     on the agentic twins (do the traces teach the C0′ host the protocol?); (T − base)@C0′ − (T − base)@C5; B1 on C5rf
     (I − L) and T − L on C5rf, without the tool.
   - *Margins.* δ = 0.05 with the decoded store in context. This is five times the host and binding differences Q1 measured
     with the decode in context (−0.01, −0.006) and one sixth of the tool's effect (+0.28): a smaller difference changes no
     reading of what the tool gives. It is attainable: the pilot's C5 − C0′ interval (±0.015 at one seed) implies a 90%
     half-width of ≈ 0.02 on 300 pairs × 3 seeds. The agentic contrast has only 100 pairs and is read as inconclusive when
     its interval is wider than the margin. δ = 0.075 for B1's I − L is half of §12's predicted B1 band (0.5 → ≤ 0.65). A
     difference in differences of four arms near chance has a 90% half-width of ≈ 0.05–0.06, so a tighter margin could
     never be shown. The C5rf comparison of B1 is read only if B1 (I − L) is shown on C5; otherwise it is reported as an
     estimate.
   - *Why secondaries, not co-primary.* (1) B1 stays as registered; a co-primary would change its Holm adjustment. (2) The
     predictions are of equivalence, which a superiority family cannot confirm; their margins are set here, not in §12.
     (3) They qualify *what* B1 and B2 learn, not *whether* it is learned.
   - *Refutation readings.* T@C0′ below T@C5 beyond the margin: using the decoded store depends on the host's channel
     training (unlike Q1's fixed pipeline). B1 on C5rf below C5's beyond the margin: internalization needs a learned binding
     (§12's withdrawn rationale). T@C0′ above 0.5 without the tool: a leak (read as I-ut).
4. *Cost* (16.3's rates, upper bounds): per seed C0′ T 0.45 and base 0.28; C5rf T 0.45, I 0.17 and L 0.28 GPU-h. That is
   **≈ 4.9 GPU-h** for 15 jobs, queued at 54.44971 (C0′) and 54.44972 (C5rf), ahead of 3c (`queue-commands-decision64.sh`).
   The queued report reads them unchanged; arm labels are `T@C0p`, `base@C0p`, `T-rf`, `I-rf` and `L-rf`.

**3c (§13).**
5. *Text competitors to K1.* The same rule loop runs with a text in the prompt instead of the store decode: (i) **the gold
   relations as text** (phase A's `symbolic`) and (ii) **the prose definition** (phase A's `definition`, E11's writer). Both
   reuse `e12_self_query.ContextBuilder`, on K1's pool (the new words and the held-out terms).
   - The belief is the host's answer with the text in context. Its confidence is that answer's softmax, calibrated by
     isotonic regression on the new words' dev half, as the behaviour's in 16.3 item 10.
   - θ is chosen on the dev half by K1's rule. The loop answers with the belief when its confidence is ≥ θ, else with the
     behaviour. It flags an item when its signals disagree (this belief, the behaviour, the evidence answer).
   - A third loop puts **the store's own decode** in the prompt (`recall:own`, read by the host). It separates the store's
     content from the rule loop's reading it without the host.
   - Real world only: the texts come from the gold frame and have no null-world counterpart. K2 is unchanged.
6. *Token accounting.* Every scored context records the prompt tokens it adds per item: the host tokenizer's count for the
   item's first prompt with the context minus without it. The context is added once to each of the item's prompts. This is
   recorded for `recall:own`, `null`, `evidence`, `symbolic` and `definition`; `none` adds 0.
   - Per loop: the store's rule loop adds none for its belief (the decode is read without the host). The text loops add
     their text's tokens.
   - Every loop's flag adds the evidence sentence on held-out items with one. This cost is the same for every loop and is
     reported separately.
7. **K1b** = loop(store) − loop(definition). It uses K1's units and pool (accuracy at 80% coverage; items × seeds crossed
   model). It is a **non-inferiority test at δ = 0.05**: shown when the 95% CI's lower bound is above −0.05 (one-sided
   α = 0.025).
   - K1b is its own test, outside K1 / K2's Holm family. Its null (the store is worse by ≥ δ) differs from theirs, and their
     family stays as registered.
   - It is read together with **the tokens saved per item** (the definition loop's minus the store loop's). "Non-inferior,
     saving N tokens per item" means the store's advantage over a curated definition is its token cost. "Not shown" means
     the definition is worth its tokens on this pool.
   - *Margin.* (a) 0.05 is a sixth or less of K1's expected effect (beliefs decoded at ≈ 0.96 against behaviour at ≈ 0.25
     on new words). A non-inferior store loop thus keeps most of the definition loop's gain over the no-tool answer (the
     usual rule: a margin well under the active control's effect over no treatment). (b) The measured gaps between the
     store's decode and the definition, both in context, are smaller: Q1 twins 0.78 against 0.74, the store ahead; T4
     natural 0.88 against 0.91, the definition ahead by 0.03. 0.05 admits both without admitting a larger loss. (c) The
     pool holds ≈ 780 new-word and 1,006 held-out test items × 3 seeds, so the 95% half-width is ≈ 0.02. Non-inferiority can
     be shown when the true difference is above ≈ −0.03.
   - *Secondaries* (never promoted): K1b per set (new words, held-out terms); loop(store) − loop(gold relations as text), the
     decode's cost against a perfect text; loop(store decode as text) − loop(definition), content with both read by the
     host; each text loop − no tool (their own K1); the tokens per loop.
   - *Prediction:* K1b non-inferior on the pooled items. The beliefs are decoded at ≈ 0.96 and need no reading; the host
     read the definition at 0.74–0.91 in phase A and on T4.
8. *Jobs.* The queued 3c critique jobs score the competitors by default (`--competitors auto`: on for a role-carrying store;
   the C5ut jobs, the null world's role-blind control, skip them). So no queued command line changes. The added cost is two
   contexts on ≈ 1,550 new-word and 1,006 held-out items at phase A's rate: ≈ +15 min per C5 run, ≈ +0.8 GPU-h in all
   (an upper bound). A track without a definition writer skips that competitor and records it; T5 has one. A C5 critique job
   that finished before this code was merged lacks the competitors and is re-run (`jobqueue retry`).
