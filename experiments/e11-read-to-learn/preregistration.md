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
| **T4-H** `t4-heldout-smollm2-v1` (natural) | 348 held-out real ChEBI entities: of 551 held out, 423 are single-concept entries with a frame, and 348 of those have a ChEBI definition | The entity's own ChEBI 255 `def:` (HTML stripped), read as `<name>: <definition>` | Ontology frame (ChEBI relations) | WP-C7 zero-shot items (416 property groups, 416 entailment pairs); loss after the term in the T4 evaluation text: 242 terms, 3,595 occurrences | CC BY 4.0 |
| **T4-N** `t4-new-smollm2-v1` | 300 invented compounds (E9 `new-words-t4-smollm2-v1`) | Written from the gold frame: `chebi` (the PubChem/ChEBI phrasing of the T4 training entry texts) and `prose` | E9 item frame | E9 dimension 3 (711 property, 600 entailment, 711 statement) | Project-generated; ChEBI names CC BY 4.0 |
| **T1-H** `t1-heldout-smollm2-v1` (negative control) | 1,974 held-out MeSH descriptors with a scope note | MeSH 2026 ScopeNote, read as `<heading>: <note>` | Ontology frame (parent, pharmacological action, see also, tree branches) | Loss after the term in `eval-pubmed`: 1,296 terms, 46,310 occurrences. No WP-C7 items exist for T1 | Public domain (NLM) |
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
- **Order:**
  1. the dev learning-rate jobs;
  2. the evaluations, priority 62 (after the pending 51–61 queue);
  3. the report (63).
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

## 13. Deviations and changes after commit

(None yet.)
