# E11 — Read-to-learn: one-shot vocabulary from reading a definition

**Status:** design and pre-registration. Written on 2026-10-07 and committed **before any run on a trained checkpoint**.
The labelled smoke tests (§12) ran after this commit; they check that the pipeline works and estimate cost. They change
nothing in §§1–10. Any later change is listed in §13 with its date and reason.

**Origin:** author decision 59 (`resources/plan-improvement/execution.md`, 2026-10-07): a model with the
ontology-composed span channel learns a new term **one-shot by reading its definition**:

1. the definition becomes a frame of relation–filler edges over the existing atomics and relations;
2. the frame is written into the ontology;
3. the channel composes the term's vector;
4. the model uses the term at once, with no gradient step.

This is the self-reflective-learning use case: the model reads, builds its own vocabulary-to-ontology mapping, and
learns from one reading.

**Code:**
- `src/vsa_embed/read_to_learn.py`: definitions, the concept finder, relation typing, readers and frame metrics;
- `src/vsa_embed/experiments/e11_read_to_learn.py`: item sets, the definition scorer, evaluation, report and plan;
- tests: `tests/test_read_to_learn.py`, `tests/test_e11_read_to_learn.py`.

**Items:** `experiments/e11-read-to-learn/items/` (committed with this document).

## 1. Question

**Main question.** After E9 training, does reading a term's definition once — with no gradient step — change the
model's behaviour on that term **after the definition has left the context**, and by how much compared with a gold
frame, a type table, in-context reading and a compute-matched gradient update?

| # | Sub-question | Measured by |
|---|---|---|
| Q1 | **Writing.** How good are the frames that each reader writes from a definition? | Edge precision and recall against the ontology's gold frame (§6.1) |
| Q2 | **Use and persistence.** Does the written frame change the model's answers about the term when the definition is not in context, and its loss on natural text where the term occurs? | The E9 dimension-3 tests; loss after the term (§6.2–6.3) |
| Q3 | **Self-reflection.** Does letting the trained model choose each edge's relation by its own likelihood beat the ontology's type statistics, surface patterns and prompted extraction by the same host? | linker vs typeprior, pattern, host |
| Q4 | **Route.** Is writing a frame better, worse or only more persistent than the text routes? | The definition in context (IKE-style); a one-shot gradient update on the definition at matched compute |

How it connects to what exists:

- **Claim C (E9.C1).** A hand-written frame for an invented word improves property selection by +4–7 points on
  SmolLM2 and +9–16 on Qwen3 (T5, 3 seeds; R9). It is null on T4 natural text: +0.003 at seed 1. The `oracle` reader
  below is that E9 condition. E11 replaces the hand-written frame with one the model writes from text.
- **E7 (self-authoring, H-G).** E7 authors frames from many corpus contexts, verifies them by gradient utility and
  trains. E11 is its one-shot, definition-only, evaluation-only counterpart: no training, and one text per term.
- **Claim D (E10.D1, R10).**
  - E10.0(e) new-word frame inference: F1 +0.07 over nearest-neighbour frames at k = 1 (synthetic).
  - E10.9a: passive + active self-reflection was **not faster** at matched data.
  - The slot acceptance test is uncalibrated against null worlds.
  - E11 offers a real-model setting in which a self-test (does the bound edge make the definition more likely than no
    row?) and a self-chosen relation can both be scored against gold (§11).

## 2. Mechanism (what is written, what is not)

- **New words.** The invented name enters the alias table (`e9_ontology_edit.extended_adapter`); its frame enters the
  composer (`SpanChannel.add_entries`, `FrameComposer.add_concepts`).
- **Held-out real terms** (in the alias table, never linked in training). Their trained gold frame is replaced by the
  read frame (`authoring.replace_frames`).
- **No row** (`none`, an empty frame) means a zero injection. This equals E9's `none` source, where the model sees
  only the name's subtokens.
- **What never changes:**
  - No weight changes.
  - No other entry's row changes. Composed rows are independent; §6.4 checks this, and it is exact on the CPU.
  - The definition is **not** in the prompt at test time, except in the in-context baselines.

## 3. Item sets and definitions

All sets were built before any run with
`python -m vsa_embed.experiments.e11_read_to_learn items …` (seed 0); the manifests record counts and sources.

| Set | Terms | Definition read | Gold frame | Tests | Licence |
|---|---|---|---|---|---|
| **T5-N** `t5-new-smollm2-v1` | 300 invented words: the E9 `new-words-t5-smollm2-v1` items, new combinations of existing atomics | Written from the gold frame in 3 styles (below) | The E9 item frame | E9 dimension 3: property (1,529 items), entailment (600), paraphrase, statement (1,529) | Project-generated |
| **T5-H** `t5-heldout-smollm2-v1` | 560 T5 terms: 360 held-out real terms (in evaluation documents only) + 200 zero-shot terms (in no document) | Written from the gold frame in the same 3 styles | Ontology frame | WP-C7 zero-shot items of those terms (1,566 property groups × 3 paraphrases; 1,680 entailment pairs); loss after the term in the run's evaluation text: 17,513 occurrences of the 360 held-out terms | Project-generated |
| **T4-H** `t4-heldout-smollm2-v1` (natural) | 341 held-out real ChEBI entities (amended §13.1; was 348): of 551 held out, 423 are single-concept entries with a frame, 348 of those have a ChEBI definition, and 341 have a headword the linker links | The entity's own ChEBI 255 `def:` (HTML stripped), read as `<headword>: <definition>` | Ontology frame (ChEBI relations) | WP-C7 zero-shot items (407 property groups, 407 entailment pairs); loss after the term in the T4 evaluation text: 242 terms, 3,595 occurrences | CC BY 4.0 |
| **T4-N** `t4-new-smollm2-v1` | 300 invented compounds (E9 `new-words-t4-smollm2-v1`) | Written from the gold frame: `chebi` (the PubChem/ChEBI phrasing of the T4 training entry texts) and `prose` | E9 item frame | E9 dimension 3 (711 property, 600 entailment, 711 statement) | Project-generated; ChEBI names CC BY 4.0 |
| **T1-H** `t1-heldout-smollm2-v1` (negative control) | 1,972 held-out MeSH descriptors with a scope note and a linkable headword (amended §13.1; was 1,974) | MeSH 2026 ScopeNote, read as `<headword>: <note>` | Ontology frame (parent, pharmacological action, see also, tree branches) | Loss after the term in `eval-pubmed`: 1,295 terms, 46,278 occurrences. No WP-C7 items exist for T1 | Public domain (NLM) |
| **Textbook** `~/data/vsa-llm/e11/items/t4-swap-openstax-chem2e-v1` (exploratory, not committed) | OpenStax *Chemistry 2e* glossary mapped onto ChEBI entries by alias (exact, then singular/plural): 85 of 752 unique terms map (11%), 77 usable, 5 of them held out | The glossary line `<term>: <meaning>` | The entry's trained frame | Frame precision/recall; **frame swap**: the read frame replaces the trained one; loss after the term | **CC BY-NC-SA 4.0** (not CC BY, see §10) |
| **Dev** `t5-dev-smollm2-v1` | 40 extra invented T5 words: E9 builder, seed 101, names reserved against T5-N | 3 styles | Item frame | Used **only** to choose the gradient baseline's learning rate (§5) | Project-generated |

**T5 definition styles** (`read_to_learn.t5_definition`). Each style states every fact of the frame.

- **`glossary`** is in the training distribution. It uses the generator's own first line (`<name>: a <area> <type>
  that <purpose>, owned by <team>.`) and one training fact template per remaining fact. 164 of 300 T5-N definitions
  contain an item template verbatim (e.g. `Status of <x>:`).
- **`dictionary`** uses labelled fields: `Owned by: …`, `Depends on: …`.
- **`prose`** uses held-out wordings (`T5_PROSE`), such as "Responsibility for X rests with the Y Squad." It contains
  no property or statement template and no training fact template: 0 of 300 do, and a test checks this.
- **Primary style: `prose`**, the realistic case of reading a definition worded unlike the training text.

**Readable ceiling.**
- The concept finder (§4) finds 100% of gold fillers in T5 definitions.
- In natural definitions it finds far fewer:
  - T4-H: 14.9% of gold fillers;
  - T1-H: 11.7%.
- Natural definitions therefore state only about 11–12% of the ontology's edges. The rest (elements, charge, branch,
  most roles) are not in the text. The `stated` reader (§4) reports this ceiling.

## 4. Readers (definition → frame)

### Common parts (model-free)

**Concept finder** (`FillerLexicon`).
- *What it matches:* every atom's readable text (the track lexicon), plus the aliases of the entry the atom names.
- *How it matches:* longest match first, at word boundaries, using the track linker's reversed trie over word-ish
  tokens. Nested matches and matches on the headword are dropped.
- *What it skips:* element / charge / branch atoms (T4) and branch codes (T1), whose surfaces are single letters or
  codes.
- It does no parsing. This is the "frozen track linker" of the design.

**Relation typing** (`RelationTyping`).
- Relation r *admits* atom a iff r's fillers in the ontology include an atom of a's type. This is type level: a T5
  "system" term may be the filler of depends_on, uses, replaces, applies_to, describes, produced_by or subtype_of; a
  ChEBI entity may be the filler of any of 10 ChEBI relations.
- A relation is *functional* if one filler appears in ≥ 95% of the frames that use it. A functional relation keeps one
  edge.
- **Ambiguous filler:** a gold filler whose type admits ≥ 2 relations.
  - T5-H: 1,208 of the 4,286 gold edges (28%).
  - On these, the type prior picks the right relation only 68% of the time.
  - So choosing the relation is a real decision there.

### The readers

| Reader | What it does | Needs the model |
|---|---|---|
| `oracle` | The gold frame (upper bound; = E9 `own`) | no |
| `stated` | Gold edges whose filler the finder sees in the text: the readable ceiling | no |
| `typeprior` | Finder + for each found atom the relation it is most often the filler of in the ontology (atom-level frequency; type level if the atom was never a filler). **The ablation of self-reflection** | no |
| `pattern` | Hearst patterns (E7 `hearst_extract`, mapped to the track's is-a / part relations) + a relation-name cue phrase within 60 characters before a found filler, in the same sentence (`owned_by` → "owned by", `has_functional_parent` → "functional parent"). A filler without a cue writes nothing | no |
| **`linker`** (primary reader; "self-reflective") | Finder + the trained model chooses each filler's relation, then self-tests the edge (algorithm below) | yes |
| `linker-all` | `linker` without the self-test (every found filler keeps its best relation) | yes (same scores) |
| `linker-joint` (**declared after the smoke test, §13.3; secondary**) | Starts from the `typeprior` frame. The model re-chooses each mention's relation inside the full frame (one Jacobi sweep), then a leave-one-out self-test | yes |
| `host` | Prompted extraction by the run's own host weights (channel not used). E7's constrained few-shot template (`authoring_prompt_fewshot`) with 2 style-matched demonstrations from seen training terms (never a read term), greedy decoding, ≤ 96 new tokens, E7 parser (`relation: filler` lines); fillers resolved through the finder; ill-typed edges dropped | yes |
| `teacher` (optional) | Claude via the B13 runner (`claude -p`, model pinned `claude-opus-5-5`; E7 `TeacherAuthor`, cached), 8 definitions per call; **open-licence text only** (T5, ChEBI, MeSH; the code refuses NC and credentialed sets) | no (external) |
| `random` | Random frame of equal degree to gold: the E9 rule, same relations, fillers frequency-weighted from each relation's pool | no |
| `none` | No frame (zero row) | no |

**`linker` algorithm (fixed):**

1. **Text read:** the definition with its headword. T5 definitions name the term; T4 and T1 definitions are read as
   `<name>: <definition>`, the dictionary-entry form, which is also the T4 training entry-text form.
2. **Score** of a frame F:
   - the summed log-probability of every token that starts after the headword's first occurrence;
   - computed under the run, with the headword linked and its row composed from F;
   - F = none drops the headword's spans.
3. For each found mention m, each atom a of m, and each relation r that admits a, compute
   `gain(r, a) = score({(r, a)}) − score(none)`. The frame has a single edge, bound and composed by the trained
   channel as any frame is.
4. **Choose:** per mention, keep the (r, a) with the largest gain.
5. **Self-test:** keep the edge iff its gain > 0, i.e. the edge makes the definition more likely than knowing nothing.
   The margin is fixed at 0 and is never tuned.
6. Functional relations keep their best-gain edge. Duplicates are removed.
7. **Cost:** 1 + Σ candidates forward passes over the definition.
   - Mean candidates per definition: T5 13.7–14.6; T4-H 31.2; T4-N 42–50; T1-H 14.1.

**Self-reflective in which sense.** No label and no gold edge enters the reader. The trained model's own likelihood
decides which relation a mentioned concept has and whether the edge is kept. The concept finder supplies only *which
atoms are mentioned*. The relation vocabulary and atom inventory are the existing ontology's, as in E9. A held-out-use
variant (choosing relations by the term's other uses) is not run:
- invented words have no other uses;
- for held-out real terms, the other uses are the test (§6.3).

## 5. Baselines on the same items (the text routes)

**In-context, IKE-style** (`context`).
- The definition plus a newline precedes every item prompt.
- Scored as in `e9_dim3_baselines`: PMI against the same context with the null surface.
- Models:
  - C5, with the term's row zeroed (`context`) and with the linker's frame (`context+linker`), plus `context+oracle`
    on the primary style;
  - C0′, C2 and P0 with the definition.
- The definition's token cost per query is recorded.
- Run on every style for C5, and on the primary style for the other models.

**One-shot gradient update** (`gradient`).
- **Update.** Per term, k Adam steps (β = 0.9 / 0.95, gradient clip 1.0) on the language-model loss of its
  definition (primary style).
  - All host parameters are updated; the channel and context modules are frozen.
  - On channel models the term's own row is dropped, so the host learns from the text alone.
- **Scoring.** The term's items are then scored without the definition in context.
- **Locality.** The mean loss of 8 fixed general-text windows of 512 tokens (`eval-general`) is measured before and
  after each update.
- **Restore.** The weights are restored exactly after every term (tested).
- **Compute matching.** k_matched = round((1 + candidates) / 3), at least 1. The linker scores 1 + candidates forward
  passes over the same text, and one training step costs ≈ 3 forward passes.
  - k_matched mean 4.9–5.2 (T5), 10.8 (T4-H).
  - An ×4 arm (4 · k_matched) is also run.
- **Learning rate**, chosen once per host on the 40 dev words:
  - model: C0′ seed 1;
  - lr ∈ {1e-5, 1e-4, 1e-3}, prose, ×1;
  - rule: the highest mean property accuracy wins, ties go to the smaller lr;
  - written to `experiments/e11-read-to-learn/dev/<host>/gradient_lr.json` and used for every run, track and model of
    that host.
- **Models:** C0′ (the main comparator, the text-only route) and C5 (the same model as the channel route).
- On T4-H the gradient arm also measures loss after the term: per term, update, then score only its own occurrences.

## 6. Metrics

### 6.1 Frames
- Edge precision, recall and F1 against gold (micro, pooled edges).
- `stated_recall`: recall over the gold edges whose filler was found.
- `filler_recall`: the finder's reach.
- Relation accuracy on found gold fillers, and on **ambiguous fillers** only.
- Empty frames and mean edges.

### 6.2 Persistence (definition not in context)
- **New words**, the E9 dimension-3 tests:
  - `property` (PMI argmax among the gold filler and 4 distractors; chance 0.20);
  - `property_new` (resampled edges);
  - `entailment` (true vs corrupted statement; chance 0.50);
  - `paraphrase` (argmax agreement across templates);
  - `statement_accuracy` and `statement_loss` (held-out wordings).
- **Held-out terms**, the WP-C7 zero-shot items: property (3 paraphrases), paraphrase consistency, entailment.

### 6.3 Natural use: loss after the term
- **Windows.** Consecutive 1,024-token windows of the run's evaluation corpus (`data.eval`) that contain ≥ 1 linked
  occurrence of a read term. Spans lie fully inside the window, with ≥ ℓ_min subtokens.
- **Targets.** The 8 targets after each occurrence: the trainer's `after_heldout` window.
- **Readers.** Every reader's frames are written for all read terms at once.
- **Exclusion.** Occurrences inside the term's **own definition document** are reported separately and excluded from
  the primary. Such a document contains `<headword>:`, e.g. its ChEBI entry text or T5 glossary page, which repeat the
  definition just read.
- **Reported:**
  - relative change versus `none`, token-weighted, with a cluster bootstrap over terms (`paired_ratio_bootstrap`);
  - share of the oracle gain recovered: (linker − none) / (oracle − none).

### 6.4 Locality
- **Channel route.**
  - The context-free rows of 2,000 sampled other entries are compared before and after the frames are written
    (max |Δ|; exactly 0 on the CPU).
  - The unlinked-token loss is measured in the read terms' windows. It may change through attention, which is a real
    within-window effect.
- **Gradient route.** Δ general-text loss (mean, max) per update.
- **In-context route.** Nothing persists after the definition leaves the context.

### 6.5 Cost per learned word
Forward passes and tokens (linker), prompt and generated tokens (host), USD (teacher), training steps and tokens
(gradient), context tokens per query (in-context), and wall seconds.

## 7. Primary endpoints and decision rule

**Primary endpoints.** Two endpoints, Holm-adjusted over both, α = 0.05:

- **P1 (synthetic pipeline).**
  - Set and model: T5-N, SmolLM2-360M C5, `prose` style.
  - Measure: **property** selection, `linker − none`, no definition in context.
  - Statistics: paired over items; item differences of seeds 1–3 pooled (the R9 convention); percentile bootstrap,
    2,000 resamples.
- **P2 (natural text).**
  - Set and model: T4-H, SmolLM2-360M C5.
  - Measure: **relative loss after the read terms** in other documents, `linker − none`.
  - Statistics: cluster bootstrap over terms; each seed's sums are added per term; 2,000 resamples.
  - Timing: P2 is confirmatory once T4 seeds 2–3 are trained. Until then the seed-1 value is reported as
    preliminary.

**Key secondaries.** Each is reported with a 95% CI and an unadjusted p.

| # | Contrast |
|---|---|
| S1 | Recovery share R = (linker − none) / (oracle − none) on P1's test (bootstrap CI over the same resamples) and on P2's |
| S2 | `linker − typeprior`: property (P1's test), and relation accuracy on ambiguous gold fillers (paired over edges) |
| S3 | `linker − random` on P1's test (frame content) |
| S4 | `linker − host`, `linker − pattern`, `linker − teacher` (if run); `linker − linker-all` (does the self-test help?); precision of accepted vs rejected edges |
| S5 | In-context: `context − linker`, `context+linker − context` (C5); C0′ + definition vs C5 + linker |
| S6 | Gradient: gains over each model's own `none`. (C0′ + gradient×1 − C0′ none) vs (C5 + linker − C5 none), paired over items. Same model: C5 + gradient vs C5 + linker. Locality of each |
| S7 | Replications: 135M; the `glossary` and `dictionary` styles; T5-H (items and loss); T4-N; T1-H; textbook (exploratory) |

**Power.**
- **P1.** About 4,600 paired item differences (1,529 × 3 seeds), with per-item SD of the difference ≈ 0.4. The
  minimal detectable effect is ≈ 0.015 (80%, two-sided). The E9 gold-frame gain is +0.04 to +0.07.
- **P2 is probably underpowered** (≈ 29k targets over 242 terms):
  - the T4 channel gain after held-out terms is small (C5 − C0′ −0.40% at 360M, seed 1);
  - only ≈ 12% of gold edges are stated in ChEBI definitions.
  - This is stated in advance. The reading of a null P2 depends on `oracle − none` (below).

**Decision rule.** Every condition is fixed now.

| Reading | Conditions |
|---|---|
| **(a) "Learns new vocabulary from one reading, and keeps it" (synthetic glossary)** | P1 > 0 with Holm p < 0.05, **and** S3 > 0 (CI excludes 0), **and** R ≥ 0.5 (point estimate; CI reported). Replication at 135M is reported, not required |
| **(b) "…on natural text"** | (a), **and** P2 shows a loss reduction with Holm p < 0.05. If T4-H `oracle − none` is itself not significant, P2 cannot test reading: the channel cannot use even gold frames there (the R9 T4 null), and (b) is **untestable on T4** rather than refuted |
| **(c) "Self-reflective reading"** (the model's own relation choice adds information) | S2 > 0 on P1's property test **or** on ambiguous-filler relation accuracy (CI excludes 0). Otherwise the claim is "reads with a type table": the model's own choice adds nothing measurable |
| **(d) "In-context does it as well; the channel only adds persistence"** | `context − linker` ≥ 0 or its CI includes 0, **and** `context+linker − context` has a CI including 0 or ≤ 0. The defensible claim is then a zero-context-token *persistence* mechanism: the definition is gone, the frame remains for every later context. If `context+linker − context` > 0 (CI excludes 0), the frame adds beyond reading the text in context |
| **(e) "A gradient step does it as well"** | S6 (gains) has a CI including 0 or favouring gradient at ×1 compute. The channel route then has no accuracy advantage at matched compute. What remains is locality: no weight change, no general-text cost (report the gradient's Δ general loss), and no context tokens |

*Amended 2026-10-09, before any E11 run (§16.1):* reading (b) now also requires the linker's loss gain on T4-H to beat
frames of wrong content (`random`, `linker-random`); P2's p is then an intersection–union p. Readings (f) and R8 are
added in §16.2 and §16.1.

**Refutation readings** (pre-written):

| # | Observation | Reading |
|---|---|---|
| R1 | `linker ≈ none` (P1 CI includes 0) | Reading writes nothing usable |
| R2 | `linker ≈ random` | The composed row signals "a known term is here", not its content (E9.C1's refutation) |
| R3 | `oracle ≈ none` on a set | The channel cannot use frames there (channel bottleneck). Reading is untestable on that set: expected on T1-H, possible on T4 |
| R4 | `linker ≈ typeprior` on property and on ambiguous relation accuracy | Self-reflection adds nothing over the ontology's type statistics |
| R5 | (d) holds | The channel adds persistence only |
| R6 | (e) holds and the gradient's general-text Δ loss ≤ 0.5% (E4's locality margin) | The only advantage of the channel route is zero weight change |
| R7 | On natural definitions `stated_recall` is high but `recall` < 0.2 | The definitions, not the reader, limit the frame (readable ceiling) |

## 8. Runs, seeds and compute

- **Checkpoints** (E9, evaluation only; `experiments/e9-retrofit/runs/{t5,t4}/<host>-{full,frozen}-<model>-s<seed>/final.pt`):

  | Track | Hosts | Seeds | Models |
  |---|---|---|---|
  | T5 | SmolLM2-360M and -135M | 1–3 | C5, C0′, C2 (P0: seed 1) |
  | T4 | same | 1 now; 2–3 once trained (≈ Oct 8–9) | same |
- **Per run:**
  - C5 runs every method and reader;
  - C0′, C2 and P0 run the text routes (`none`, `context`, `gradient`; and loss after the term on T4-H / T5-H).
- **Order** (amended in §13.5):

  | Priority | Jobs |
  |---|---|
  | 62 | the dev learning-rate jobs |
  | 63 | tier 1: the primary endpoints' runs |
  | 64 | tier 2 |
  | 65 | tier 3 |
  | 66 | the pooled report |

  All of these run after the pending 51–61 queue.
- The exact commands and GPU-hour estimates (measured in the smoke test, §12) are in §12 and come from
  `python -m vsa_embed.experiments.e11_read_to_learn plan`.
- **Qwen3 hosts are not part of this pre-registration.** The code supports them once Qwen3 E11 item sets are built.

**Statistics.**
- Percentile bootstrap with 2,000 resamples and seed 0: item level for items, term clusters for loss.
- Holm over P1 and P2.
- Secondaries are reported with CIs and unadjusted p values, and labelled as such.
- Exploratory sets (textbook, T1-H) are labelled.

## 9. What would change the plan

- If T5-N v2 items (decision 56, ≈ 700 per arm) exist before the runs are queued, T5-N v2 is run as a **replication**
  with the same rules. P1 stays on v1 (fixed here).
- If the dev lr choice is at a grid edge (1e-5 or 1e-3), the grid is **not** extended. The edge is reported as a
  limitation of the gradient baseline.

## 10. Data, licences, and the clinical interface

**Sources.**
- **T5 and T4-N:** project-generated definitions.
- **ChEBI 255:** CC BY 4.0. Definitions are committed with attribution in the manifests.
- **MeSH 2026 scope notes:** public domain (NLM).
- **OpenStax *Chemistry 2e*:** **CC BY-NC-SA 4.0**. The design brief assumed CC BY 4.0; the source repository's
  collection metadata, `LICENSE` and README all say CC BY-NC-SA 4.0, and so do *Anatomy and Physiology 2e* and
  *Biology 2e*. The text is kept under `~/data/vsa-llm/e11/` and is not committed or sent to the teacher.
  `DATA_SOURCES.md` §14 records the source, commit, hashes and licence.

**Clinical track interface** (decision 58: credentialed text is read only by local readers). What a clinical
read-to-learn set needs:

1. `heldout_definitions` gets a branch that reads definitions from a **local derived table** under `~/data/vsa-llm/`,
   e.g. SNOMED CT text definitions or UMLS `MRDEF` for held-out concepts. Nothing derived is committed. The item set
   is written under `~/data/vsa-llm/e11/items/`, its manifest licence names the source (e.g. "SNOMED CT …
   credentialed"), and the evaluation output folder is outside the repository.
2. The track needs a `TrackSpec`, a lexicon (atom texts, atom types) and an alias table, as for T1.
3. **Readers allowed:**
   - `oracle`, `stated`, `typeprior`, `pattern`, `random`, `none`;
   - `linker` and `host`, which run on the local GPU with the local model.
4. **Readers forbidden:** `teacher`. `run_teacher` refuses any set whose licence mentions NC, MIMIC, SNOMED, UMLS,
   DUA or "credentialed" (tested).
5. **MIMIC note text** is never a definition source here. MIMIC notes could serve only as local evaluation text for
   the loss after the term, which needs a MIMIC-tokenized evaluation corpus built by the clinical-track agent.

## 11. How this bears on claim D and on the paper

**Claim D.** E10.D1 is now a negative / methods result:
- in the planted world, reflection was not faster at matched data (E10.9a);
- the slot acceptance test accepts false relations in null worlds.

E11 puts both ingredients of "self-reflective learning" into a real trained model, where each can be checked against
gold:

1. **The self-chosen relation** (S2, reading (c)). The model picks the relation whose bound edge best explains the
   text. The test is whether that beats the ontology's type statistics on the ambiguous fillers, where the type prior
   is 68% accurate on T5. This is the one place in the project where a model's own choice of a relation's meaning is
   scored against gold outside a synthetic world.
2. **The self-test** (S4, `linker` vs `linker-all`). An edge is kept only if it makes the definition more likely than
   no row. E11 measures its precision against gold on natural definitions, where most found concepts are *not* gold
   edges (T4-H type-prior precision 0.31). This is a calibration check of a self-acceptance test in a real model; E10
   lacked one outside planted worlds.
3. **The explicit write vs the gradient** (S6, reading (e)). This is E10.9's question in its cleanest form: one
   reading, one explicit write, against a compute-matched gradient step on the same text.
   - If the write wins and persists, that is evidence for "explicit commitment beats passive gradient at one shot"
     (narrowly: relations and atoms given).
   - If the gradient matches it, claim D gains nothing from E11.
   - Either way, claim D stays out of the abstract until E10.2 (decision recorded in the novelty check).

**What would support "learns new vocabulary from one reading, and keeps it"** in the paper:
- reading (a) at 360M (P1 significant, content control passed, ≥ 50% of the gold-frame gain recovered), replicated in
  direction at 135M;
- at least an oracle-supported natural-text result, i.e. reading (b), or T5-H loss after the term in the same
  direction;
- the claim scoped to vocabulary new to the host (R9: the gain follows novelty).

**What would mean "in-context does it as well, so the channel only adds persistence":** reading (d).

- **Paper wording:** "a definition read once is kept as a composed row and used later at zero context tokens, with
  accuracy no better than reading the definition in context".
- That is still a useful property: context windows are finite, and in-context knowledge is gone once the definition
  scrolls out.
- It is not a learning-efficiency claim.
- If reading (e) also holds, the paper says the write matches a gradient step at no weight change and no locality cost.
  It does not say the write is better.

## 12. Smoke tests and commands

**CPU smoke test.** `tests/test_e11_read_to_learn.py` (tiny E9 world; every method). It checks:
- E11's `oracle` and `none` conditions reproduce E9's `own` and `none` scores exactly;
- the definition scorer equals a direct recomputation;
- weights, rows and schedules are restored exactly after every method.

**GPU smoke test and commands:** added below after this document was committed (§12 addendum).

### §12 addendum — GPU smoke test (2026-10-07; SMOKE, not a result)

**Setup.**
- **Run:** SmolLM2-360M C5 seed 1. T5 checkpoint for T5-N, T4 checkpoint for T4-H.
- **Memory:** hard cap of 4 GB (`torch.cuda.set_per_process_memory_fraction`).
- **Shared GPU:** an E9 training job held the GPU at 100% throughout.
- **Optimizer:** the gradient arm used SGD (lr 1e-3, CPU weight backup) to stay under 4 GB. The full runs use Adam with
  the dev-chosen lr.
- **Outputs:**
  - `experiments/e11-read-to-learn/smoke/` (final code);
  - `smoke/v0-before-fixes/` (the first run, which motivated §13.1–13.3).

| Smoke | Size | Wall time | Peak memory | Breakdown (s) |
|---|---|---|---|---|
| T5-N prose | 20 words, 228 items, 10 readers, in context, gradient ×1, locality | 324 s | 3.0 GB | readers 26, persistence 111, in context 138, gradient 47 |
| T4-H chebi | 40 terms, 108 items, 9 readers, in context, 48 windows, locality | 256 s | 3.1 GB | readers 42, persistence 92, in context 78, windows 26 |

Both exceed the 5-minute target, because the GPU was shared. On an idle GPU the same runs take an estimated ≈ 2–3 min
each (assumption: 2× contention).

**Checks passed:**
- every definition headword links (T5-N 766 / 766, T4-H 2,796 / 2,796 scored texts);
- rows of 2,000 other entries are unchanged (max |Δ| 1.5e-7, bf16 noise);
- weights are restored after the gradient arm.

**Smoke numbers.** Too few items for any inference; not used for any decision except §13.

| Measure | Smoke value (n) |
|---|---|
| T5-N frame F1 | oracle 1.00, typeprior 0.90, linker-all 0.82, linker-joint 0.59, linker 0.55, host 0.76, pattern 0.25 |
| T5-N property | none 0.19, oracle 0.21, linker 0.19, linker-joint 0.21, typeprior 0.22, host 0.23, random 0.17 |
| T5-N in context (definition prepended) | 0.61 (`context − linker` +0.42 [+0.34, +0.50], n = 94 items) |
| T5-N `context+linker − context` | −0.01 [−0.04, +0.02] |
| T4-H loss after the term in other documents, vs none (470 targets, 64 occurrences) | oracle −0.91% [−1.10, −0.51], typeprior −0.72% [−0.89, −0.33], linker −0.49% [−0.92, +0.08], random −0.47% [−1.12, +0.67] |
| T4-H frame precision | typeprior 0.35, linker 0.04 |

**Exact queue commands.** `experiments/e11-read-to-learn/queue-commands.sh`, printed by
`python -m vsa_embed.experiments.e11_read_to_learn plan --include-pending`.
- **Jobs:** 61 runnable now; 24 more once T4 seeds 2–3 are trained.
- **Estimated GPU hours** (idle GPU, from the per-concept smoke timings ÷ 2):

  | Tier | Contents | Now | After T4 seeds 2–3 train |
  |---|---|---|---|
  | dev lr | | 0.1 h | — |
  | 1 | T5-N 360M C5/C0′ s1–3, T4-H 360M C5/C0′ s1 | 7.6 h | +3.0 h |
  | 2 | | 12.9 h | +4.8 h |
  | 3 | | 4.0 h | +1.7 h |
  | **Total** | | **≈ 24.6 GPU-h** | **+ 9.5 GPU-h** |

## 13. Deviations and changes after commit

All of these were made on 2026-10-07, after the labelled GPU smoke test (§12) and **before any full run**. None changes
P1, P2, the decision rule or the refutation readings.

### 13.1 Held-out headwords are link-checked (a construction bug fix)

**Bug.** The smoke test found that a read headword sometimes did not link to its own entry. The definition scorer then
cannot inject the term's row; 238 of 1,330 T4 scored texts were affected. A build-time check found the causes:

| Track | Cause | Count |
|---|---|---|
| T5 | The 200 zero-shot terms were read under their internal concept name `synthetic:<Name>` | 200 |
| T4 | The ChEBI display name is an alias of *another* entry | 44 |
| T4 | Charged names such as `quinine(1+)` do not link | 13 |
| T1 | Inverted MeSH headings (`RNA, Ribosomal, 16S`) are not aliases | 265 |
| T1 | Single-token headings cannot link at ℓ_min = 2 | — |

**Fix** (`linkable_headwords`). The headword is the display name if the track linker links it to the term's own entry
in `<headword>: …`. Otherwise it is the entry's canonical linkable alias (`canonical_surfaces`). Otherwise the term is
left out.

| Set | Headword source | Left out | Size |
|---|---|---|---|
| T5-H | name 560 (prefix removed) | 0 | 560 |
| T4-H | name 291, alias 50 | 7 | 348 → **341** (WP-C7 items 416 → 407) |
| T1-H | name 1,699, alias 273 | 2 | 1,974 → **1,972** |

The loss-after-the-term occurrences are unchanged for T4 (3,595) and almost unchanged for T1. No item, frame or
analysis rule changed.

### 13.2 A faster continuation scorer (numerically equal)

In-context prompts are 3–5× longer. The shared adapter computes full-vocabulary log-probabilities at every position,
so the in-context step took 223 of 406 s in the T5 smoke test.

`continuation_scores` computes output logits only at the positions that predict continuation tokens, in batches
sorted by length. It runs the same forward pass, spans, autocast and output head. It is used inside `ItemScorer` for
every condition. A test checks it equals `channel_probes.continuation_logprob` / `e9_ontology_edit.continuation_stats`
to 1e-5 on the CPU. The E9-consistency test (E11 `oracle` / `none` = E9 `own` / `none`) still passes at 1e-6.

### 13.3 A declared secondary reader: `linker-joint`

**Motivation** (the frame metrics of the smoke test). Single-edge scoring, the pre-registered `linker`, is a poor
relation chooser:

| Smoke set | Measure | `linker` | `typeprior` |
|---|---|---|---|
| T5-N, prose, 20 words | Edge precision | 0.74 | 0.90 |
| T5-N, prose, 20 words | Relation accuracy on ambiguous fillers | 0.30 | 0.64 |
| T4-H, 40 terms | Edge precision | 0.07 | 0.36 |
| T4-H, 40 terms | Relation accuracy on ambiguous fillers | 0.18 | 0.84 |

The T5 self-test also rejected many correct edges: recall 0.43, against 0.82 without it.

The likely mechanism: a row composed from a single edge is unlike any trained row (trained frames have ≈ 7 edges), so
its likelihood says more about the row being off-distribution than about the relation.

**`linker-joint`** (`read_to_learn.read_linker_joint`):

1. Start from the `typeprior` frame F0 (all found fillers with their prior relation).
2. **One Jacobi sweep.** For each mention, every type-admitted (relation, atom) candidate replaces that mention's
   edge in F0, or is added to F0 if the mention has no edge there. All other edges stay fixed. The best-scoring
   candidate is kept, and functional relations are de-duplicated.
3. **Leave-one-out self-test** on the swept frame F1: an edge stays iff removing it lowers the definition's
   log-likelihood (margin 0).
4. **Cost:** 1 + Σ candidates + 1 + |F1| forward passes. The gradient baseline's compute match stays tied to `linker`'s
   cost; `linker-joint` costs about the same.

**Status: declared secondary.**
- The primary reader, P1 and P2 stay `linker`, as pre-registered.
- `linker-joint` gets the S1–S6 contrasts (`linker-joint − none`, `− typeprior`, `− oracle`, `− linker`) as
  secondaries, labelled "declared after the smoke test".
- **Open decision for the author:** promote `linker-joint` to primary before the full runs. That would be recorded
  here as a pre-run amendment, with this disclosure.

**What the smoke test saw.** SmolLM2-360M C5 seed 1:
- the first 20 T5-N concepts (`e9n-0000` – `e9n-0019`), prose style;
- the first 40 T4-H concepts (pre-§13.1 ordering).

Only the frame metrics above motivated the change. The item-test numbers of those concepts are in the smoke reports
(`experiments/e11-read-to-learn/smoke/v0-before-fixes/`).

### 13.4 Implementation details settled by the smoke test

- **In-context batches** are halved (they were not reduced before §13.2).
- **Memory.** Peak allocated memory at 360M was 3.0–3.1 GB with SGD and a CPU weight backup (smoke). The full runs use
  Adam with a GPU backup, ≈ 7 GB, which is why their queue jobs ask for an otherwise idle GPU.

### 13.5 Plan scope (cost; secondary sets only)

The smoke timings put the full design at ≈ 34 GPU-h. To bound the cost, the plan (`plan_jobs`) does the following.
None of it touches P1, P2 or their comparators.

1. **Gradient arm only on C0′ and C5**, as §5 already says; C2 and P0 get no frame and in context only.
2. **Secondary sets read their primary style only:** T5-H prose, T4-N chebi. T5-N keeps all three styles (S7).
3. **The per-term gradient on windows runs on T4-H only** (`--window-gradient`), as §5 says.
4. **T1-H** (negative control) runs SmolLM2-360M seed 1, C5 and C0′. It measures frames and loss after the term only
   (no items exist), with windows capped at 2,048 (the T1 evaluation size of decision 12).
5. **Tiers.** The primary endpoints and their key comparators are tier 1 (priority 63); replications and controls are
   tier 2 (64); 135M runs on T5-H and T4-N are tier 3 (65).

### 13.6 The author's decisions of 2026-10-07 (recorded, not deviations)

- **Primary reader:** keep the pre-registered `linker`; `linker-joint` stays secondary (§13.3).
- **Queued:**
  - tier 1 (gradient-dev 360M at priority 51; T5-N 360M C5/C0′ seeds 1–3 and T4-H 360M C5/C0′ seed 1 at 52);
  - `e11-report` at 54.
  - Tiers 2–3 wait until tier 1 has been read.
- **OpenStax textbook** (CC BY-NC-SA 4.0): approved for non-commercial research, local only, never sent to Claude.
- **Claude teacher reader:** not approved; `teacher` is not run.

## 14. Amendment (2026-10-07, before any run): E11-M, many terms read once each

**Origin.** The author approved a follow-up after the smoke test showed that one definition in context beats every
frame (T5 property 0.61 vs ≤ 0.23). The question is whether the picture flips when a model must learn many new terms
from one reading each and then use them together. Code `vsa_embed.experiments.e11_many`; tests `tests/test_e11_many.py`.

### 14.1 Question

When N new terms (N = 25 … 400) have each been read once, how well does each route answer questions about them and
read text that uses several of them? How does that change as N grows, and at what cost per use?

- **In context:** all definitions in the prompt, cut at the window.
- **Retrieval:** the relevant definitions retrieved into the prompt.
- **Frames:** every definition written once into the ontology, used at no context cost.
- **Gradient:** the definitions learned one after another.

### 14.2 Sets (built 2026-10-07, committed)

**T5-M** `items/t5-many-smollm2-v1`:
- **Terms:** the 700 invented words of `t5-new-smollm2-v2`. That set is new: the E9 v2 items of decision 56, with
  definitions written in the same 3 styles as T5-N; it is also T5-N's v2 replication set of §9.
- **Reading:** prose definitions, mean 110 tokens.
- **Episodes:** one seeded order (seed 0). The episode of size N ∈ {25, 50, 100, 200, 400} is its first N terms, so the
  episodes are nested.
- **Items:** the E9 dimension-3 items of those terms (5.1 property items per term).
- **Passages:** 50 per size. Each uses 4 of the first N terms, with 2 facts per term in the training fact templates and
  boilerplate between term blocks. Every mention is recorded.

**T4-M** `items/t4-many-smollm2-v1` (natural):
- **Terms:** the 242 T4-H terms that occur in the evaluation text, in one seeded order.
- **Sizes:** 25, 50, 100, 200, 242.
- **Tests:** the WP-C7 items, and the T4 evaluation windows that hold ≥ 2 distinct read terms among the first N. The
  full set has 721 such 1,024-token tiles. The loss is the after-mention window, other documents as in §6.3.

**How many definitions fit the budget** (SmolLM2 tokenizer; measured):

| Set | Budget | N = 25 | 50 | 100 | 200 | 400 / 242 |
|---|---|---|---|---|---|---|
| T5-M | 2,048 | 18 | 18 | 19 | 17 | 22 |
| T5-M | 7,168 | 25 | 50 | 65 | 66 | 70 |
| T4-M | 2,048 | 25 | 50 | 49 | 49 | 48 |
| T4-M | 7,168 | 25 | 50 | 100 | 173 | 167 |

### 14.3 Routes (all on the same items, passages and windows)

| Route | Condition | What it does |
|---|---|---|
| Frames | `frame:<reader>`, readers `oracle`, `linker`, `linker-joint`, `typeprior`; `none` = no frame | Every term's frame is written at once. A use costs no context token |
| In context | `context:B<budget>`, B ∈ {2,048, 7,168} | The first N definitions in a seeded order under the header "Glossary of new terms:", whole definitions up to B tokens. 7,168 is the 8,192-token window minus 1,024 for the task. The prefix is read once and cached (`PrefixCache`); rows of the new terms are zero |
| Retrieval | `rag:k<k>`, k ∈ {1, 3} | BM25 over the N definitions. Query for items: the term's first prompt; for passages: the passage, retrieving k × (terms in the passage); for windows: the window's text. The top-k definitions form the prefix. Recall@k (the term's own definition retrieved) is recorded |
| Gradient | `gradient` | One pass over the definitions in order. For each: compute-matched Adam steps (as §5), the dev learning rate, the term's own row dropped. Checkpoints after 25, 50, … terms on the same sequence. Weights restored at the end. Runs on C0′ and C5 |

**Models:** SmolLM2-360M C5 (all routes) and C0′ (in context, retrieval, gradient, no frame). Seeds 1–3 on T5-M; T4-M
seed 1 first, then seeds 2–3 once trained.

### 14.4 Metrics

1. Per-term dimension-3 accuracy (property, entailment, paraphrase, statement) on the first N terms, against N.
2. Loss after the new-term mentions in passages (T5-M) and natural windows (T4-M) against N, relative to `none`, with a
   cluster bootstrap over passages / terms.
3. **Interference.**
   - Frames: the first 25 terms scored with only them written vs with all N written. Rows are independent, so the
     maximum margin change must be ≈ 0.
   - Gradient: the property accuracy of the first 25 terms at each later checkpoint (forgetting).
   - In context: accuracy against the share of definitions that fit; each item records whether its term's definition
     was in the prompt.
4. **Cost.**
   - Context tokens and prefill FLOPs (2 · parameters · tokens) per use.
   - Write cost: reader forward tokens, or gradient steps, tokens and FLOPs.

### 14.5 Primary endpoint and decision rule

**Primary M1.**
- Set and model: T5-M, SmolLM2-360M C5, N = 200.
- Measure: **property** accuracy, `frame:linker − context:B2048`, over the items of all 200 terms.
- Statistics: paired over items, item differences of seeds 1–3 pooled, percentile bootstrap (2,000), two-sided
  α = 0.05.
- M1 is its own family. It answers a different question from P1 and P2 and is not adjusted with them.
- **Power:** ≈ 1,035 property items × 3 seeds ≈ 3,100 paired differences; minimal detectable effect ≈ 0.012.

**Decision rule** (fixed now):

| Reading | Conditions |
|---|---|
| **(M-a) "Writing frames beats reading definitions in context once there are many terms"** | M1 > 0 and p < 0.05 |
| **(M-b) "…only with better reading"** | M1 is not significantly positive, but `frame:oracle − context:B2048` at N = 200 is (CI excludes 0). The reader is the bottleneck |
| **(M-c) "In context still wins at N = 200 within 2,048 tokens"** | M1 < 0 (CI excludes 0). The channel's only advantage is the cost per use: 0 context tokens against 2,048, and no prefill |
| **(M-d) Crossover N\*** (descriptive) | Per budget, the smallest N at which `frame:linker` ≥ `context:B` on the seed-pooled point estimates. The oracle and the 7,168 budget are reported alongside |
| **(M-e) "Retrieval removes the scaling problem"** | `rag:k1 − frame:linker` ≥ 0 at every N (CI includes 0 or positive). The channel then keeps only zero per-use context and no retrieval step |
| **(M-f) "A gradient pass scales as well"** | `gradient − frame:linker` ≥ 0 at N = 200, **and** the first 25 terms lose ≤ 0.02 property accuracy between N = 25 and N = 400 |

**Refutation readings:**
- **Rows depend on N** (interference check > 1e-3 in margin). This would contradict row independence, so the run is
  invalid.
- **Passages and windows disagree with the items** (the frame route helps the item tests but not the loss after
  mentions in multi-term text, or the reverse). Report both. The paper claim needs both.

**Prediction** (written before any run, from the E11 smoke test; not an input to any decision):
- At B = 2,048 only ≈ 17 of 200 definitions fit. One definition in context added ≈ +0.42 property, so
  `context:B2048` should sit ≈ +0.04 above `none` at N = 200 and ≈ +0.02 at N = 400.
- The linker frame added ≈ 0 and the oracle frame ≈ +0.02 in the smoke test.
- We therefore expect M1 ≈ 0 or slightly negative (M-c or no difference) at N = 200, and a crossover only for the
  oracle route near N ≈ 400 at B = 2,048.
- We expect retrieval at k = 1 to stay near the single-definition in-context level at every N (M-e).

### 14.6 What it means for the paper

**If M-a holds,** the read-to-learn story gains its strongest form: "with many new terms, writing each definition once
into the ontology beats reading definitions in context at a fixed context budget, at zero context tokens per use".

**If M-c and M-e hold,** the honest claim is narrower:
- the channel is a zero-token persistence mechanism;
- in-context reading and retrieval are more accurate at these sizes;
- the channel's niche is when neither the context budget nor a retrieval step is available.

### 14.7 Compute and placement

The estimate is in the §14 addendum below (CPU smoke test and commands). The recommended placement is priority 53:
after the tier-1 evaluations (52) and before `e11-report` (54). The T4 seeds 2–3 jobs follow their training.

### §14 addendum — CPU smoke test (2026-10-07; SMOKE, not a result) and commands

**Setup.**
- **Run:** SmolLM2-135M T5 seed 1 on the CPU (6 threads); the GPU queue was busy. N = 25, the first episode.
- **Model and routes:**
  - C5: frames (oracle, linker, typeprior, none), in context at 2,048 tokens, retrieval k = 1;
  - C0′: the gradient route (Adam, lr 1e-4, the dev lr not yet chosen).
- **Outputs:** `smoke/many-t5-SmolLM2-135M-C5-s1-cpu/`, `smoke/many-t5-SmolLM2-135M-C0p-s1-cpu-gradient/`.

**Timing.**
- The C5 smoke took 546 s, over the 5-minute target. The cause was the CPU: the 2,048-token prefix route took 231 s.
- A first attempt scored all 400 terms because the read set was not restricted to the requested sizes. It was stopped
  and fixed (`evaluate_many` now reads and scores only the first max(sizes) terms).
- The C0′ gradient smoke took 163 s.

**Smoke numbers.** One seed, 25 terms, 135M, so not evidence for anything:

| Condition | Property accuracy | Loss after mentions in passages |
|---|---|---|
| none | 0.15 | 2.54 |
| frame:oracle | 0.18 | 2.43 |
| frame:linker | 0.18 | 2.50 |
| frame:typeprior | 0.20 | 2.43 |
| context:B2048 (18/25 definitions fit) | 0.39 | 1.85 |
| rag:k1 (recall 1.00) | 0.63 | 1.38 |
| C0′ none | 0.16 | 2.40 |
| C0′ gradient | 0.23 | 3.36 |

- The sequential updates help the items but damage text prediction.
- The frame interference check gives exactly 0.0 (306 items, CPU).

**Commands.** `experiments/e11-read-to-learn/queue-commands-many.sh`, printed by
`python -m vsa_embed.experiments.e11_many plan`.

| Jobs | Priority | Idle-GPU estimate |
|---|---|---|
| 6 × T5-M (360M C5 / C0′ s1–3) | 53 | ≈ 0.89 / 0.67 GPU-h each |
| 2 × T4-M s1 | 53 | ≈ 0.56 / 0.40 GPU-h each |
| e11-many-report | 54 | — |
| **Total now** | | **≈ 5.6 GPU-h** |
| 4 × T4-M seeds 2–3 (after their training) | 53 | ≈ 1.9 GPU-h |

- **T7-H** (§15): 2 jobs after T7's E9 runs train, ≈ 1.0 GPU-h, commented in the same file.

## 15. Amendment (2026-10-07): dictionary sources for "read a dictionary", and the T7 dictionary set

**Survey** (probed 2026-10-07; one or two requests per service; `DATA_SOURCES.md` §15):

| Source | Licence and access | New words? | Gold frames? | Occurrences in local text | Verdict |
|---|---|---|---|---|---|
| Wiktionary (dumps; MediaWiki API; Wikimedia REST `page/definition`) | CC BY-SA 4.0 (+ GFDL); no key; serial requests with a user agent | `Category:English neologisms`: 1,267 pages, 1,175 with English definitions. 855 are dated by quotations; 342 first quoted ≥ 2020, 157 ≥ 2023. There are no "coined in <year>" categories | None: 50 of the titles are WordNet lemmas, and frames would have to come from the definition itself (circular) | Of 146 titles first quoted ≥ 2023, **8** occur in 300,000 FineWeb-Edu documents (40 occurrences). There is no local 2025–26 general text | **Not feasible** for the E11 tests now (no gold, no occurrences, and the host track for general words is WordNet, where the channel is null). Feasible only with a new 2025–26 web-text download (author decision) |
| Free Dictionary API (dictionaryapi.dev) | Wiktionary-derived (CC BY-SA); no key | as Wiktionary | — | — | HTTP 522 (origin down) on 2026-10-07; adds nothing over Wiktionary |
| Wikidata lexemes | CC0; SPARQL endpoint (60 s queries); no key | 100,837 English lexemes. Recent words exist with short glosses (rizz, delulu, enshittification); no attestation dates | Sense → item links are sparse; mappings to WordNet synsets are rare for new words | — | CC0 glosses only; not a test set |
| Wordnik | Account and API key; content from several dictionaries under mixed terms | — | — | — | Excluded: not open-licence |
| Merriam-Webster | Free for non-commercial use, ≤ 1,000 queries / day / key, ≤ 2 reference APIs | — | — | — | Excluded: not open-licence |
| Oxford | Free trial; enterprise licences from £5,000 / year / language | — | — | — | Excluded |
| WordNet 3.0 glosses (local) | WordNet licence (permissive) | No new words (2006) | Yes (WordNet relations) | C3 text | Negative control only: the channel is null on the WordNet track (R9) |
| **MeSH 2026 supplementary-record notes** (local; T7's ontology) | Public domain (NLM) | New *to the host*: 0 mentions in 300k general documents, T7's selection; but only 5 of the usable terms were introduced ≥ 2023 | Yes (T7 frames: mapped heading, pharmacological action, class) | T7 `eval-pubmed`, 2026 PubMed | **Feasible once T7 is trained**: built as an E11 set, below |

**T7-H** `items/t7-heldout-smollm2-v1` (built; runs need T7's E9 checkpoints, not yet queued):
- **Terms:** 908 T7 held-out records. 487 have an informative note and a linkable headword (369 under their name, 118
  under an alias). 372 occur in the 2026 PubMed evaluation text (4,834 occurrences).
- **Reading:** the note with its bibliographic parts removed, read as `<headword>: <note>` (e.g. "1,10-phenanthroline:
  inhibits Zn-dependent metalloproteinases").
- **Caveat (measured):** the notes rarely name the gold fillers.
  - The concept finder sees 1.9% of gold fillers, and `stated` covers 2.1% of the gold edges.
  - The notes describe activities; the frames name MeSH headings, e.g. pharmacological action "Matrix Metalloproteinase
    Inhibitors".
  - So the frame readers will write almost nothing (R7). On T7-H, E11 tests the text routes and the gold frame; the
    reading itself is untestable without a semantic filler matcher, and the Claude teacher is not approved.
- **Proposed runs** (once T7 trains): the E11 `heldout` evaluation, as T4-H in tier 2.

## 16. Amendment (2026-10-09, decision 64, before any E11 run): P2 specificity, a definition-encoder competitor, in-context reading on natural text

**Status when written.** At 2026-10-09 19:36 UTC every E11 job in the queue was `pending` (the dev lr job at 51, tier 1
at 52, E11-M and T7-H at 53, the reports at 54, the T7-ROOD block at 54.4979; the running job was an E9 T7 training run).
No E11 output exists. Nothing below is post hoc, and no part is out of scope because it already ran. The queue script
(§16.5) refuses to cancel a job that is no longer pending; if one has started by then, its set keeps the pre-amendment
readers and the report marks the affected test incomplete (`missing`). E11-M (§14) is unchanged.

**Origin.** Three E9 findings of the T4 arm battery (author, 2026-10-09):

1. **Shuffled frames on real text.** The shuffled-frames arm C5sh gives every entry another entry's frame, held-out
   terms included. On T4 (3 seeds) it keeps C5's after-held-out loss gain: C5 − C5sh +0.01% [−0.08, +0.10]. The own
   frames matter only on seen, rare and unseen terms (after_unseen −0.82%). On T5 (synthetic) C5sh is 4.6% worse after
   held-out terms. On real text, a held-out loss gain from a written frame may therefore be non-specific.
2. **A definition encoder beats C5 on T4 held-out terms.** C6d (the frozen host's mean-pooled last hidden state of the
   verbalized frame) gives C5 − C6d +0.17% [+0.05, +0.29].
3. **In context on T4.** A prose definition in context does as well as the decoded store (0.91 vs 0.88).

**Audit of this pre-registration** (each claim checked against §§5–8, the code and the queue JSON):

| Claim | Check |
|---|---|
| Rule (b) reads P2 (`linker − none` on T4-H) without requiring linker > random | **Confirmed.** S3 (`linker − random`) enters reading (a) on P1 only. `run_report` did compute `linker − random` on the T4-H windows, but only as a secondary |
| No definition-encoder (C6d) row | **Confirmed.** E11 had no encoder route |
| The T7-ROOD runs omit `context` | **Confirmed.** C5 runs `frames,gradient,windows,locality` and C0′ runs `gradient,windows`. Wider than claimed: `context` scored WP-C7 items only, T7-ROOD has none, and no set had an in-context route on the windows |
| Token cost is recorded for the text routes | **Confirmed** for items: context tokens per query, and the gradient's training tokens per update. There was no window route to record |
| (found here) The `random` reader is "the same shape" | **Only for the gold frame.** `random` keeps the gold frame's relations and degree (E9 rule). In the T4-H smoke test (§12) random frames had 12.2 edges and linker frames 2.45, so it is not shape-matched to the linker |

### 16.1 P2 specificity

**Reading (b), amended.** P2 counts as supported only if all three hold on T4-H. The setting is unchanged:
SmolLM2-360M C5, seeds 1–3 pooled, loss after the read terms in other documents, cluster bootstrap over terms, 2,000
resamples.

1. `linker − none` is a loss reduction (as pre-registered).
2. `linker − random` is a loss reduction. This is the author's condition: a random frame of the gold's shape.
3. `linker − linker-random` is a loss reduction. **New reader**, the shape-matched content control:
   - each linker frame keeps its relations and its edge count;
   - every filler is redrawn frequency-weighted from the relation's pool, as `random` draws, but never a filler of the
     linker's or the gold's frame for that relation;
   - no frame stays no frame, and an edge whose pool holds no other filler is dropped (`read_to_learn.read_shape_random`).

Condition 3 is added in this amendment. The author's wording ("frames of the same shape from wrong content") holds for
`random` against the oracle, not against the linker, whose frames are about one fifth the size on T4-H (§12). With
`random` alone, a failure could be a shape effect: more edges may carry more of the non-specific "a known term is here"
gain that C5sh shows. Condition 3 separates content from shape. Requiring both is stricter than the author's rule, never
weaker. **Open for the author:** keep condition 3 in the rule, or report it as a secondary only.

**Test and family.**
- P2's p is the intersection–union p: the largest of the three component p values (each a two-sided cluster-bootstrap p
  with the effect in the stated direction). A conjunction tested this way needs no further multiplicity adjustment
  (Berger 1982).
- **Holm family: {P1, P2}**, α = 0.05, with P2's p = that intersection–union p. The family is unchanged in size, and P1
  is unchanged.
- Everything else in this amendment is a secondary, with an unadjusted p and a 95% CI.

**The same test on every held-out set** (reported, not confirmatory): T7-ROOD-H (the T7-ROOD analogue, C5 seeds 1–3),
T7-H, T5-H and T1-H.
- A set is labelled "specific" if all three components are reductions and the intersection–union p is < 0.05.
- On T7-ROOD-H and T7-H the notes name about 2% of the gold fillers (§15), so most linker frames are empty. For those terms
  `linker` = `linker-random` = `none`: a null there is expected, and it is not evidence of non-specificity.

**Readers.** `random` already ran on every set where the loss after the read terms is read:
- T4-H, T7-ROOD-H, T7-H, T5-H and T1-H, in their C5 jobs (checked in the queue JSON and the plans);
- the text-route models (C0′, C2, P0) read no frames.

`linker-random` is added to every C5 job, T5-N included. On T5-N, `linker − linker-random` on P1's property test is
reported next to S3 as a secondary; reading (a) keeps S3 as written.

**R8** (new refutation reading): `linker` < `none` on the T4-H loss, but `linker` ≈ `random` or `linker` ≈ `linker-random`.
Reading: the read frame's natural-text gain is non-specific, as C5sh's is on T4. Reading (b) then fails, whatever P2's
`linker − none` shows.

### 16.2 The definition-encoder competitor (`encoder`, on the trained C6d arm)

**Route.** Each read term's source vector in a trained C6d run is replaced by C6d's encoding of the text it read, and the
term is then tested with no definition in context and no weight change. The C6d arm (E9 WP-PQ1) works as follows:
- **Source vector:** the frozen pretrained host's last hidden state, mean-pooled over a text, whitened per dimension over
  the non-held-out entries, then unit-L2 (`e9_rowsource.definition_rows`, `row_sources.apply_standardization`).
- **Projector:** a trained MLP reads the vector into the stream at C5's site and gate; its parameters match C5's channel.

**Which text: the definition the linker read, not the verbalized written frame.** Reasons:
- **Same evidence.** The two routes then read exactly the same input once. They differ only in what they store: an
  explicit frame over the ontology, composed by C5's trained composer, or a dense encoding, projected by C6d's trained
  projector. This is Q4's question; the in-context and gradient routes also read the definition.
- **Not a second decoder.** Verbalizing the linker's frame would make the encoder a second decoder of the linker's output.
  It would inherit the linker's reading errors (T4-H smoke: precision 0.04, 2.45 edges), and it would compare composer
  and projector on a fixed frame. E9 already asks that question on gold frames (C6d vs C5).
- **Matching oracle.** C6d's table rows for held-out terms are encodings of their verbalized gold frames. These are the
  route's `oracle`, so `encoder − oracle` is the cost of reading on this route, as `linker − oracle` is for frames.

**Text read.** The headword is removed: a leading `<headword>:` is dropped and every other occurrence becomes `It`
(`encoder_text`). This matches C6d's own construction: its texts use the subject "It" and never contain the name, and the
name's subtokens reach the model at every use anyway.

**Stated asymmetry.** The C6d projector was trained on encodings of verbalized frames ("It is a …. It has role …"), not on
natural definitions. The linker's single-edge frames are likewise unlike trained frames (§13.3). Both routes face a
read-time distribution shift, and that shift is part of the comparison.

**Fidelity check (every job).** The table stores only standardized rows. The job therefore re-encodes the table's stored
verbalizations of every non-held-out entry with the same frozen host, in bf16 on CUDA as the table was built, to recover
the whitening statistics. It reports the cosine between re-encoded and stored rows; ≈ 1 is expected (tested > 0.999 on
the CPU world). `--encoder-fit N` subsamples the reference entries; it is used for smoke tests only.

**Conditions on a C6d run**, in the primary style (T5-N `prose`, T4-H `chebi`, T7-ROOD-H `scr`):
- `none`: a zero source vector. The projector has no bias, so this is no row.
- `oracle`: the term's own table row; for new words, the encoded verbalized gold frame.
- `encoder`: the encoded definition.

They are scored on the same items as C5 (T5-N: E9 dimension 3; T4-H: WP-C7) and on the same windows (T4-H, T7-ROOD-H).

**Contrasts against `frames:linker`.** Key secondaries S8, unadjusted, reported next to P1 and P2. Pairing: C6d seed s
with C5 seed s, on the same items and the same windows (equal target counts are checked).
- **Absolute:** `C6d encoder − C5 linker`, on P1's property test (and the other item tests) and on P2's loss after the
  term.
- **Gains:** `(C6d encoder − C6d none) − (C5 linker − C5 none)`.
- **Within C6d:** `encoder − none`, `oracle − none`, `encoder − oracle`.

**Cost per learned word.** Both routes use 0 context tokens per use. Write costs:

| Route | Write cost per word | Also recorded |
|---|---|---|
| Linker | 1 + candidates forward passes over the definition (T4-H ≈ 32 passes) | tokens |
| Encoder | One forward pass of the frozen host over the definition | tokens; FLOPs = 2 · parameters · tokens |

The encoder also needs a one-time statistics pass per run, which is not a per-word cost, and a second (frozen) host in
memory while it writes.

**Reading (f)** (pre-written): **"A dense encoding of the same definition does as well."**
- **Condition:** `C6d encoder − C5 linker` is not worse on P1's property test (CI includes 0 or favours the encoder) and
  not worse on P2's loss.
- **Then:** the frame route keeps interpretability and editability (explicit edges over a known vocabulary, §6.4
  locality) but no accuracy advantage.
- **If the encoder is better with a CI excluding 0** (as C6d on T4 gold frames), the paper says so.

### 16.3 In-context reading on natural text (`context` with `windows`)

**Route.** When `context` and `windows` are both requested on a held-out or swap set, each evaluation window is read
after the definitions of the read terms it holds:
- the definitions are in the primary style, in order of first occurrence, one per line, whole;
- the prefix is capped at 1,024 tokens and at the host's position limit;
- the prefix is computed once per window (`PrefixCache`, the E11-M machinery).

There are two conditions:
- `context`: the read terms have no row;
- `context+linker` (C5 only): the read terms have the linker's frames.

The loss after the term is computed as in §6.3. The other conditions are unchanged.

**Token accounting**, reported with the per-query item cost:
- definition tokens per window and per occurrence;
- definitions per window;
- the share of occurrences whose own definition was in context.

**Where it runs.** Wherever both methods are requested:

| Set and model | Status |
|---|---|
| T4-H C5 and C0′ | The queued C0′ command lines are unchanged; C5 is replaced anyway (§16.1) |
| T5-H | Not queued |
| T7-ROOD-H C5 and C0′ | Now request `context` (replacement jobs) |
| T1-H, T7-H | Do not request `context`; unchanged |

**Known difference from joint reading.** C5's causal context query (4 tokens) does not reach into the prefix for an
injection in a window's first tokens (`PrefixCache`, §14). A test checks that an empty prefix reproduces the plain
windows (1e-4).

**Contrasts** (secondaries; reading (d) on natural text): `context − none` (C5, C0′), `context − linker`, and
`context+linker − context`.

### 16.4 Report

`report` computes:
- P2's three components and its intersection–union p;
- Holm over {P1, P2};
- the reading (b) decision;
- the specificity table of every held-out set;
- the C6d competitor (paired by seed: absolute and gains);
- the in-context window contrasts;
- a cost table per set and model (write: linker and encoder forward tokens, gradient training tokens; use: context
  tokens per query and per window occurrence).

**Code:**
- `e11_read_to_learn`: `evaluate`, `DefinitionEncoder`, `source_conditions`, `window_context_losses`, `run_report`,
  `decision64_jobs`;
- `read_to_learn.read_shape_random`;
- `e9_rowsource.frozen_host` / `read_verbalizations`;
- `row_sources.apply_standardization`;
- `e9_tracks.relation_pools`.

**Tests:** `tests/test_e11_decision64.py`; `tests/test_t7_rood_followups.py` is updated.

### 16.5 Jobs (`queue-commands-decision64.sh`, printed by `e11_read_to_learn plan-decision64`; not executed here)

Jobs are cancelled by moving their JSON to `.jobs/cancelled/`; the script stops if any of them is no longer pending.
Replacements keep their names and priorities. New jobs take a free fractional slot after the jobs they extend.

| Cancel → re-add (same name) | Priority | Change | GPU-h (was → now) |
|---|---|---|---|
| e11-t5-new-SmolLM2-360M-C5-s1…s3 | 52 | + `linker-random` | 1.63 → 1.68 each |
| e11-t4-heldout-SmolLM2-360M-C5-s1…s3 | 52 | + `linker-random`; windows in context (implied) | 0.92 → 0.97 each |
| e11-t7-heldout-SmolLM2-360M-C5-s1 | 53 | + `linker-random` | 0.60 → 0.62 |
| e11-t7rood-heldout-SmolLM2-360M-C5-s1…s3 | 54.4979 | + `context`, + `linker-random` | 0.60 → 0.66 each |
| e11-t7rood-heldout-SmolLM2-360M-C0p-s1…s3 | 54.4979 | + `context` | 0.40 → 0.43 each |
| e11-t7rood-report | 54.4979 → **54.49792** | + C6d outputs; runs after the C6d jobs | — |
| e11-report | 54 | + C6d outputs (T5-N, T4-H) | — |

| New | Priority | GPU-h |
|---|---|---|
| e11-t5-new-SmolLM2-360M-C6d-s1…s3 (`encoder`; runs exist) | 52.5 | 0.07 each |
| e11-t4-heldout-SmolLM2-360M-C6d-s1…s3 (`encoder,windows`; runs exist) | 52.5 | 0.07 each |
| e11-t7rood-heldout-SmolLM2-360M-C6d-s1…s3 (`encoder,windows`; after T7-ROOD C6d trains at 54.497) | 54.49791 | 0.06 each |

**Totals.** The script queues ≈ 12.4 GPU-h. The replacements are ≈ 11.8 GPU-h, against ≈ 11.2 GPU-h for the jobs they
replace, and the 9 C6d jobs add ≈ 0.6 GPU-h. The net cost of decision 64 is ≈ +1.2 GPU-h (idle-GPU estimates). The
decision-64 parts are estimated, not measured on the GPU: a window read after its prefix is taken as 2 × a batched window,
and the encoder statistics pass as ≈ 90 s on T4 and ≈ 10 s on T5 and T7.

**Unchanged and still valid** (no command line changes):
- e11-gradient-dev-SmolLM2-360M (51) and -t7rood (54.4979);
- the C0′ jobs of T5-N and T4-H (52); T4-H C0′ now also reads its windows in context, ≈ +0.01 GPU-h each;
- e11-t7-heldout-SmolLM2-360M-C0p-s1 (53);
- every e11-many-* job and e11-many-report.

**Not queued, and printed with the new readers by `plan`:** the tier-2/3 jobs of `queue-commands.sh` (T5-H, T4-N, T1-H,
135M). Their lines in that file predate this amendment; re-print them with `plan` before queuing them.
