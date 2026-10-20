# Logic in the store and reading from real text: the complete experimental plan

**Written:** 2026-10-17 by workplace-fd, at the author's request:
> "create a complete experimental plan - with every run - that explores every section and every ablation - aim for
> prelimary experiments for now - but later on also create a full scale experimental plan - we want to cover all
> possible angles and possibilities"

and, for reading:
> "do we need the model to match the best relations and connections via REACT-style framing?"

**Companion to:** [`2026-10-17-vsa-predicate-logic.md`](2026-10-17-vsa-predicate-logic.md) (the "proposal"; section
numbers below refer to it).

**Status:**
- **Tier P (preliminary)** is pre-registered in `experiments/e14-logic/preregistration.md` (committed before any
  non-smoke run) and is being queued at the author's instruction of 17 Oct: "queue up all of these preliminary
  experiments".
- **Tier F (full scale)** is a plan only. Each of its blocks needs a gate from tier P, its own pre-registration
  addendum, and the author's go-ahead.
- **2026-10-20: tier P's GPU jobs have run** (RD-4/7, RD-3, LG-B/C/D, LG-P). Results are in
  `manuscript/findings/2026-10-20-e14-tier-p-gpu.md`.
  - **G1 fails** on the SmolLM2 hosts: they misread negated recall lines. The Qwen3 hosts gain +0.42 to +0.47 at every
    depth.
  - **G2 fails** on C1–C4.
  - **RD-4's rule** gives no E13 amendment: explicit recall is −0.10 on T4 property and +0.31 on T5. RD-4b comes next.
  - **RD-12** is retried after a fix: three of its four arms are pending.
  - **Follow-ups queued** (pre-registration §§13–15; the author's go-ahead, 20 Oct):
    - RD-4b, selective recall with a context-free null (8 jobs, 54.4959975);
    - LG-B verdict format, gate G1′ (4 jobs, 54.4959976);
    - LG-P hard problems (4 jobs, 54.4959977);
    - reports at 54.4959979.
    - Script: `experiments/e14-logic/queue-commands-followup.sh`; about 4 GPU-h.
  - **Every finished experiment explained:** `manuscript/findings/2026-10-20-e14-experiments-explained.md`.

---

## Status of tier P (2026-10-17)

**Queued GPU jobs** (`experiments/e14-logic/queue-commands-rd.sh`, `queue-commands-lg.sh`). All are before the toolkit
band, so with the current queue they start in about 3.5 days:

| Priority | Jobs | GPU-h (estimate) |
|---|---|---:|
| 54.49595 | RD-4/7: T4 held-out and T5 new × (360M C5 s1–3, C0′ s1) = 8 | ≈ 8 |
| 54.49596 | RD-3: T4 and T5 × (SmolLM2-360M, Qwen3-1.7B) = 4 | ≈ 1.4 |
| 54.49597 | LG-B/C: 7 item sets × 4 hosts = 4 jobs | ≈ 3.3 |
| 54.49598 | LG-D: trees, lists and edits × 4 hosts = 4 jobs | ≈ 0.9 |
| 54.49599 | LG-P: 4 hosts (all policies; the generative ones on Qwen3) | ≈ 4.7 |
| 54.495995 | reports: RD-4, RD-3, LG, LG-P (CPU lane) | — |
| 54.495996 | **RD-12 pilot** (author's go-ahead, 17 Oct): interleaved passive ↔ active reading; T4 360M C5 s1; arms cycle, single, gold, passive × 3 rounds | ≈ 4 |
| 54.4959969 | RD-12 report (CPU lane) | — |
| | **total** | **≈ 22** |

**Already run on the CPU (descriptive):**
- RD-1a: the coverage audit;
- RD-1b: the teacher audit, $2.19;
- RD-2: dictionary breadth;
- RD-5: the bootstrap simulation;
- RD-6: structure readers;
- LG-P3: retrieval;
- the RD-3 teacher frames ($1.96).

**LG-A** (all algebra grids) and the **LG-C CPU parts** (3 seeds) are also done. **Gate G0 passes.** The soft settings
only tie fair symbolic baselines on the CPU.

The results are in `experiments/e14-logic/runs/` and summarised in proposal §9.4b and in the finding record
`manuscript/findings/2026-10-17-e14-tier-p-cpu.md`.

**Teacher spend:** $4.15 of the $25 cap.

**Deviations from the plan's tier-P column**, recorded in pre-registration §9:
- `mc-rel` uses a relation prior;
- edits are two-way choices;
- decoys were added to the first-order problems;
- `exec-vsa-stressed` was added;
- the RD-4 cost estimate is about 8 GPU-h rather than 3;
- LG-A3 runs forward chaining only; the backward mode and rule records stored as vectors move to tier F.

---

## 0. How this plan is organised

**Two tiers:**
- **P (preliminary):** "does it work at all, and where does it break?"
  - CPU work uses the existing code's algebra.
  - GPU work scores existing models only (no training).
  - Small item sets, a few hosts.
  - It covers **every section and ablation of the proposal at two or three levels**.
- **F (full scale):** "how far does it go, and is it robust?"
  - Every level of every ablation.
  - More seeds, worlds, tracks and hosts.
  - The training arms.
  - Sized for the manuscript or a follow-up paper.

**Work packages:**

| Package | Proposal section | What it tests |
|---|---|---|
| **LG-A** | §2, §6, §7 L0 | the algebra: rules, chains, joins, nesting, connectives, numbers, trees; clean-up, operators, fillers, capacity, baselines (CPU) |
| **LG-B** | §7 L1 | a language model reads the executor's conclusions, against rules as text |
| **LG-C** | §7 L2 | the "soft" settings where the store should beat symbols: zero-shot entities, aliases, noisy facts, few-example rules, edit propagation |
| **LG-D** | §2.6, §7 L3 | decision trees and decision lists, read by a model |
| **LG-P** | §8 | full first-order logic with the language model as the prover's search policy |
| **LG-T** | §7 L4b, §8.6 | training arms: the model learns to use or run logic (F only) |
| **RD** | §9 | reading facts from real text: the four failure stages, ReAct-style reading, the bootstrap question (critical for self-learning) |

Each experiment is listed with its question, design, levels (tier P → tier F), runs, metrics, primary contrast,
decision rule, cost and dependencies. §9 maps every proposal section to its experiments. §10 is the run inventory.

---

## 1. Conventions shared by every experiment

**Statistics:**
- Paired bootstrap, 4,000 resamples, clustered by theory (LG), entity (RD) or tree (LG-D).
- Seeds pooled, with the per-seed spread reported.
- Holm within each experiment's primary family.
- 95% intervals.
- **Rough detectable effects** (paired accuracy differences): about ±0.05 at 300 items per cell (tier P), about ±0.03 at
  1,000 or more (tier F).

**Seeds:**
- **Algebra (CPU):** P 3 seeds; F 10 seeds.
- **Item worlds:** P one generated world per item set; F three independent worlds (generator seeds 1–3), so that the
  variance across worlds is measured.
- **Scoring trained runs:** P uses the seeds that exist (s1–3 where available); F uses all.

**Hosts** (all frozen; nothing in tiers P or F is trained except LG-T and LG-P6):

| ID | Host | Tier |
|---|---|---|
| H-135 | SmolLM2-135M (P0, untouched) | P, F |
| H-360 | SmolLM2-360M (P0) | P, F |
| H-q06 | Qwen3-0.6B-Base | P, F |
| H-q17 | Qwen3-1.7B-Base | P, F |
| H-C5 | SmolLM2-360M C5, T4 and T5 runs s1–3 (the method's models) | P (RD only), F |
| H-C0′ | SmolLM2-360M C0′ s1–3 (plain fine-tuning) | P (RD only), F |
| H-135-C5 | SmolLM2-135M C5 s1–3 | F |
| H-q-C5 | Qwen3 C5 runs (where they exist) | F |
| H-q35 | Qwen3.5 hosts, after the Qwen3.5 block reports | F |

**Item vocabulary:**
- Invented names (the T5 generator's name machinery), checked against WordNet and a general-text sample.
- Held-out entities, rules and depths are fixed at generation.
- Some rules are deliberately **counterfactual**, so following them goes against general knowledge.

**Fairness, applied to every model-reading comparison:**
- **Information parity:** every condition gets the same underlying facts and rules.
- **Tokens:** prompt tokens are reported for each condition.
- **Prompts:** templates and scoring are the same across conditions.
- **Controls:** a shuffled or wrong store, a no-binding store, and an exact symbolic reference.

**Data rules:**
- Decision 58 applies: no MIMIC, SNOMED CT or UMLS anywhere, and no T1c.
- Teacher (Claude) calls run on public data only (ChEBI CC BY 4.0, MeSH, synthetic text), each with a hard `--max-usd`
  cap.
- Licences are recorded per item set. CLUTRR is CC BY-NC, so we write our own CLUTRR-*style* generator instead of using
  its data.

**Outputs:**
- Code: `src/vsa_embed/experiments/e14_*.py` and `src/vsa_embed/logic/`.
- Configs, items, the pre-registration and reports: `experiments/e14-logic/`.
- Large data: `~/data/vsa-llm/e14/`.
- Job names start `e14-`; report jobs contain `-report` (CPU lane).

**Queue placement (tier P):**
- **GPU scoring jobs:** priority 54.49595–54.49599. That is after addendum 2's reports (54.4959) and **before the
  toolkit band** (54.496), so that the reading results (RD-4) arrive before E13's self-learning jobs. Tier P adds about
  14 GPU-hours ahead of the toolkit band.
- **Moving a job:** `jobqueue cancel <name>`, then re-add the same command at another priority.
- **CPU-only work** (LG-A; RD-1/2/5/6): run directly with 4 threads, outside the queue. A CPU-lane job waits for every
  job ahead of it, so queued CPU work would not start for days.

---

## 2. Infrastructure (built once, used by every package)

| ID | Module | What | Tier |
|---|---|---|---|
| X1 | `logic/world.py` | the world generator: entities with typed categorical and numeric attributes; binary relations with fixed, Poisson or heavy-tailed degree; rules of each kind (inverse, symmetric, sub-relation, chain/composition, inheritance down `is_a`, 2–3-premise conjunction, exception by negation as failure, recursion); trees and decision lists; FOL problems with disjunction and negation; held-out splits; renderers (facts, rules and trees as sentences; proofs) | P |
| X2 | `logic/datalog.py` | exact semi-naive Datalog with stratified negation; exact tree and list executors; a small sound first-order prover (backward chaining, proof by cases, proof by contradiction, function-free); the reference for every experiment | P |
| X3 | `logic/store.py` | the VSA store: fact vectors and per-entity frames on the repo's `HRRAlgebra` (also unitary HRR, MAP, FHRR); clean-up (nearest, typed, soft Hopfield, resonator); rule mapping vectors (by hand, mean, least squares); a rule-record executor (forward and backward chaining over the store, with symbolic join checks after clean-up); tree and list executors; fractional power encoding; margins and abstention; decode to text | P |
| X4 | `experiments/e14_algebra.py` | the LG-A grid runner and report (CSV, JSON, Markdown) | P |
| X5 | `experiments/e14_items.py` | item writers for LG-B, LG-C, LG-D and LG-P (each condition's prompt text precomputed from X2 and X3) | P |
| X6 | `experiments/e14_score.py` | forced-choice and cloze scoring of item sets on any host (option log-probabilities, length-normalised, with a null-prompt calibration as in E12), token counts, per-item outputs | P |
| X7 | `experiments/e14_prover.py` | the LG-P harness: the tools of §8.3, policies (model scored choice, model generative ReAct, random, breadth-first, set of support, exhaustive), budgets, proof logs | P |
| X8 | `experiments/e14_reading.py` | the RD experiments: audit, dictionary expansion, relation-choice readers (including the ReAct-style reader), the explicit-recall scorer (on E11's `ItemScorer` and E12's `RecallStore`), the bootstrap simulation, structure readers | P |
| X9 | `experiments/e14_report.py` | pooled reports across seeds and hosts; the decision of each gate | P |

Reused, unchanged:
- `algebra.HRRAlgebra` / `UnitaryHRRAlgebra`, `cleanup`;
- `self_query.RecallStore` / `RecallWriter`;
- `e11_read_to_learn.ItemScorer` / `load_read_set` / `run_context`, `e5_common.open_run`;
- `kg_baselines` (AMIE-style mining, TransE, RotatE, ComplEx), `ontology_hypotheses` (E10's structural hypotheses);
- `jobqueue`.

---

## 3. LG-A: the algebra (CPU; the proposal's §2, §6 and L0)

All LG-A experiments run on random or trained vectors with no language model. One run is one (configuration, seed),
and each writes a row of `metrics.csv`. **The gate G0** (proposal §7) is read from A2 and A1.

### LG-A1. One rule as one vector (mapping vectors; Z1, Z2 from examples)

- **Question:** for which rule kinds, widths and numbers of examples does one vector apply a rule to unseen entities?
  Does it survive correlated (trained) fillers and being applied to a whole frame?
- **Levels:**

| Factor | Tier P | Tier F |
|---|---|---|
| rule kind | inverse; symmetric; sub-relation; inverse + rename (3 arguments) | + argument permutation of 4–5 arguments; predicate-only rename; composite of 2 rules |
| estimator | by hand; mean correlation; least squares | + ridge sweep; mapping memory (Emruli-style, a store of example mappings) |
| examples k | 1, 3, 10, 30, 100 | 1, 2, 3, 5, 10, 30, 100, 300, 1,000 |
| width | 256, 1,024, 4,096 | 128, 256, 512, 1,024, 2,048, 4,096, 8,192 |
| fillers | random; correlated (cosine 0.3); **trained T5 C5 rows (s1)** | + correlated 0.1 and 0.5; trained T4, T7 and WordNet rows; 135M and 360M |
| applied to | one fact; a frame of 8 facts | + frames of 1, 4, 16 and 24 facts |
| clean-up pool | 2,000 entities | 2,000; 20,000; 100,000 |
| test entities | disjoint from the examples | + nearest-neighbour split (test entities similar to the examples) |

- **Runs:** P 4 × 3 × 5 × 3 × 3 × 2 × 3 seeds = **3,240** cells (about 20 CPU-minutes). F about 150,000 cells (about
  10 CPU-hours).
- **Metrics:** all slots correct; per-slot accuracy; clean-up margin; calibration (ECE) of the margin.
- **Primary (P):** all-slots accuracy at k = 10, width 1,024, by filler kind.
- **Decision:** if trained fillers fall below 0.9 where random fillers reach 0.99, the survey's warning (§3.2) holds,
  and every downstream use needs whitening, or rules defined over role-keyed atoms rather than composed vectors.

### LG-A2. Chains: transitive and composed rules executed by unbinding (Z3)

- **Question:** how does accuracy fall with depth, load and width? Which clean-up and storage layout keeps it high?
- **Levels:**

| Factor | Tier P | Tier F |
|---|---|---|
| depth | 1–8 | 1–16 |
| load (edges per frame) | 4, 8, 16, 32 | 2, 4, 8, 12, 16, 24, 32, 48 |
| width | 256, 512, 1,024, 2,048 | + 4,096, 8,192 |
| degree profile | fixed; heavy-tailed | + Poisson; the empirical T4 and T7 profiles |
| clean-up | nearest (all entities); typed (candidates of the relation) | + soft Hopfield (β sweep, 1–3 steps); none until the final hop |
| storage | one store per entity; one global bundle | + sharded by relation; precomputed closure stored in the frame |
| entities | 5,000 | 5,000; 50,000; 117,659 (WordNet scale, as in Crawford et al.) |
| algebra | HRR (unitary roles); MAP; FHRR | + block-unitary; GHRR (non-commutative); the trained C5 operator |

- **Runs:** P 4 loads × 4 widths × 2 profiles × 2 clean-ups × 2 storages × 3 algebras × 3 seeds = **1,152** chains
  sets (about 40 CPU-minutes). F about 40,000 (about 15 CPU-hours).
- **Metrics:** accuracy per hop and up to each hop; the fit of the "(hop-1 accuracy)^depth" model.
- **Primary (P):** accuracy at depth 3 against load at width 1,024 (gate G0's depth-3 criterion).

### LG-A3. Rules with shared variables (joins): the Datalog fragment

- **Question:** does forward chaining over the store reproduce the exact closure, and where do false conclusions
  appear and spread?
- **Rule shapes:**
  - chain join `r1(x,y) ∧ r2(y,z) → r3(x,z)`;
  - star join `r1(x,y) ∧ r2(x,z) → r3(y,z)`;
  - inheritance `is_a(x,c) ∧ p(c,v) → p(x,v)`;
  - 2-premise conjunction;
  - exception by negation as failure;
  - recursion (ancestor to depth 4).
- **Levels:**
  - number of rules: P {1, 5, 20}; F {1, 5, 20, 50, 200};
  - width: P {256, 1,024}; F up to 4,096;
  - mode: forward chaining to saturation; P adds backward chaining from goals; F adds rule records stored as vectors
    (`IF⊛… + THEN⊛…`) against rules held by the executor;
  - load: P {8}; F {4, 8, 16}.
- **Runs:** P 6 shapes × 3 × 2 × 2 × 3 seeds = **216**. F about 5,000.
- **Metrics:**
  - closure precision and recall against X2's exact closure;
  - false conclusions per true one;
  - **error propagation** (the share of wrong conclusions derived from earlier wrong ones);
  - steps.
- **Primary (P):** closure F1 at width 1,024 with 20 rules.

### LG-A4. Function terms, nesting and argument order

- **Question:** how deep can terms nest, and which encoding keeps argument order?
- **Levels:**
  - nesting depth: P 1–4; F 1–8;
  - encoding: P {plain convolution chain, permutation per argument position, clean-up per level with stored
    intermediate terms}; F adds GHRR and matrix binding;
  - arity: P 2–5;
  - width: P {256, 1,024}.
- **Tests:**
  - telling `f(g(a))` from `g(f(a))` (order);
  - swapped arguments in n-ary facts;
  - decoding nested terms.
- **Runs:** P 4 × 3 × 4 × 2 × 3 seeds = **288**.
- **Primary:** order accuracy for the plain chain against permutations. The prediction is about 0.5 for the plain chain,
  as in the binding report.

### LG-A5. Connectives and quantifiers

- **What is tested:**
  - **∧:** k-of-n soft matching (bundled premises) against exact matching;
  - **∨:** a bundle of the alternatives;
  - **¬:** negation as failure by a similarity threshold, calibrated on held-out seeds as in experiment 00;
  - **∃:** a threshold query over memory;
  - **∀∃ nesting:** F only, to document the limit.
- **Levels:** P widths {256, 1,024}, loads {8, 16}.
- **Runs:** P **96**.
- **Metrics:**
  - ROC AUC;
  - calibration error (ECE);
  - the share of queries answered when the system may abstain and must keep accuracy ≥ 0.95.

### LG-A6. Numbers and thresholds (fractional power encoding)

- **What is tested:** equality and nearness, and `x ≤ t`, three ways:
  - (i) a bundle over the allowed range;
  - (ii) decode the number from its vector, then compare;
  - (iii) F only: a thermometer code.
- **Levels:**
  - width: P {256, 1,024};
  - resolution: P {10, 100 levels};
  - range: P one.
- **Runs:** P **72**.
- **Metrics:** accuracy of the comparison; error against distance from the threshold.

### LG-A7. Decision trees and decision lists (the algebra side)

- **Levels:**

| Factor | Tier P | Tier F |
|---|---|---|
| encoding | node memory; one nested vector; flat bundle of paths; decision list (ordered rules) | + transition table (Osipov-style); a mapping memory of conditions to actions |
| depth / list length | 1, 2, 3, 4, 6, 8 / 3, 6, 12 | 1–12 / 3–50 |
| features × values | 20 × 8 | 10–100 × 2–16 |
| numeric tests | none; half the tests | + all numeric |
| input noise | 0; 10% of attributes wrong | 0–30% |
| width | 256, 1,024 | 256–4,096 |
| edits | one node changed after building | + subtree replacement |

- **Runs:** P (6 + 3) × 4 encodings × 2 × 2 × 2 × 3 seeds = **864**.
- **Metrics:**
  - execution accuracy;
  - how accuracy degrades with input noise (soft bundles against exact execution);
  - whether answers follow edits.

### LG-A8. Clean-up ablation and resonator factorisation

- **Clean-up variants** (the kinds are in A2 and A7):
  - nearest neighbour over all vectors;
  - type-restricted;
  - soft Hopfield;
  - none until the final step;
  - F adds a learned clean-up (`SoftCleanup`, trained).
- **Resonator networks:** factor a bound pattern of 2–3 unknown variables into their values ("unify several
  variables at once").
  - P: width {1,024}; codebooks of 50, 200 and 1,000; 2 or 3 factors.
  - F: up to 5 factors and codebooks of 10,000.
- **Runs:** P **36** resonator cells; the clean-up cells are counted in A2 and A7.

### LG-A9. Operators

HRR with unitary roles, MAP (bipolar), FHRR (complex phases) and the fixed random operator are in A1, A2 and A7 (P). F
adds block-unitary and GHRR, and the **trained relation operators** of the T4 and T5 C5 runs, loaded from their
checkpoints.

### LG-A10. Fillers (the survey's warning)

- Random, synthetic correlated and trained composed rows are in A1 and A2 (P).
- F adds held-out composed vectors, rows from 135M against 360M runs, and **whitened** trained rows (the remedy if A1's
  decision rule fires).

### LG-A11. The executor on the real ontologies' own stores

- **Question:** can the trained stores carry out the rules that hold in their own ontologies?
  - **T4 store:** the trained C5 s1 store via `RecallStore`.
    - Inverse: `is_conjugate_acid_of` ↔ `is_conjugate_base_of`.
    - Symmetric: `is_enantiomer_of`, `is_tautomer_of`.
    - Transitive: `is_a` at 2–3 hops.
    - Inheritance of `has_role` down `is_a`, marked "defeasible" because ChEBI does not guarantee it.
  - **T7 store:** `is_a` along tree codes.
  - **WordNet store:** hypernym closure.
- **Gold:** the exact closure of the same ontology (X2).
- **Subsets:** seen entries against held-out entries (composed zero-shot).
- **Runs:**
  - P: 3 tracks × rule kinds × {s1} = **about 12** cells (CPU, store only);
  - F: s1–3, 135M and 360M, and T1, T8 (Wikidata inverse and sub-property constraints).
- **Primary:** derived-fact accuracy on held-out entries, compared with seen entries.

### LG-A12. Baselines

- **Exact Datalog (X2):** the ceiling, in every experiment.
- **Knowledge-graph embeddings** (`kg_baselines`: TransE, RotatE, ComplEx), trained on the facts of the A1/A3 worlds.
  - They predict rule conclusions statistically.
  - Held-out entities have no embedding (transductive); reported as such.
  - P: RotatE and ComplEx on one world.
- **Rule miners:**
  - P: `kg_baselines.mine_rules` (AMIE-style);
  - F: AnyBURL and AMIE 3 (external Java tools, installed outside the pinned environment), and E10's
    `StructuralHypothesis` miner.
- **Runs:** P **6**.

**Gate G0** (from A1 and A2; proposal §7):
- explicit execution ≥ 0.95 at depth 3 at some width ≤ 2,048 under the T4 load profile;
- and mapping vectors from ≤ 10 examples ≥ 0.95 on unseen entities.

**Tier P cost of LG-A:** about 6,000 cells, **about 2–3 CPU-hours** with 4 threads.

---

## 4. LG-B: a language model reads the executor's conclusions (GPU scoring; proposal L1)

### LG-B1. Deduction questions by depth: the main experiment

- **Items (X1, X5):** PrOntoQA-style theories in invented vocabulary.
  - **Content:** rules ("every wumpus is a dalk"; counterfactual rules included) and facts ("Brindle is a wumpus").
  - **Questions:** "Brindle is a florp: true or false?" at proof depths 0–5.
  - **Balance:** half the answers true, half false (the false ones come from a negated rule along the chain).
- **Size:** P 1 world × 6 depths × 200 questions = **1,200**. F 3 worlds × 7 depths (0–6) × 500 = 10,500, with
  "unknown" as a third label (open-world variant).
- **Prompt conditions (what precedes the question):**

| Condition | Content | Tier |
|---|---|---|
| `none` | nothing | P |
| `facts` | the entity's stated facts only | P |
| `text` | **the rules and facts used by the executor, as sentences (main baseline)** | P |
| `text+demo` | the same plus one solved example | P |
| `text+cot` | the same; the model writes reasoning before answering (generation) | F |
| `exec-vsa` | **the conclusions and a short proof trace, decoded from the VSA store (ours)** | P |
| `exec-vsa-conclusion` | the decoded conclusion only, no trace (format ablation) | P |
| `exec-vsa-noconf` | `exec-vsa` without similarity scores (format ablation) | P |
| `exec-sym` | the same content from exact Datalog (the cost of VSA noise) | P |
| `exec-shuffled` | conclusions from a store with permuted rules (does the model follow the store?) | P |
| `exec-nobind` | conclusions from a store without binding (C5ut-style) | P |
| `text-retrieved` | the top 5 rules by keyword overlap (retrieval baseline, used in B6) | P |

- **Hosts:** P H-135, H-360, H-q06, H-q17. F adds H-C5, H-C0′, H-135-C5, H-q-C5 and H-q35.
- **Answer formats:** P forced choice (true/false) and cloze (the derived class among 4 candidates). F adds open-world
  three-way answers.
- **Primary (P):** `exec-vsa − text` by depth (1–5), Holm over depths, per host; and `exec-vsa − exec-sym`.
- **Gate G1** (proposal §7):
  - `exec-vsa − text > 0` (Holm) at depth ≥ 3 for H-135 and H-360;
  - and `|exec-vsa − exec-sym| ≤ 0.03`.

### LG-B2. Prompt-format ablation

Part of B1's conditions (P):
- trace against conclusion only;
- similarity scores against none.

F adds:
- proof-ordered against shuffled lines;
- terse "field" lines against sentences;
- 1, 2 or 4 recall lines per fact.

### LG-B3. Splits

- **P:**
  - counterfactual against neutral rules;
  - depth.
  - "Held-out rules" (predicates never seen in any demonstration) are part of the P world.
- **F:**
  - **held-out entities whose store vectors are composed zero-shot** (LG-C1);
  - real-word entities with counterfactual rules (prior conflict).

### LG-B4. Shortcut control (SimpleLogic-style samplers)

- **Items:** propositional Horn theories over invented adjectives, depth 0–5, drawn by two samplers:
  - rule-priority (RP): rules first, then labels;
  - label-priority (LP): labels first, then rules consistent with them.
- **Size:** P 600 per sampler. F 2,000 per sampler, plus a third sampler that balances rule counts.
- **Conditions:** `text`, `exec-vsa`, `exec-sym`.
- **Primary:** `(exec-vsa − text)` under LP against RP. If the store's gain is real deduction, it does not shrink under
  LP, while the text baseline may.

### LG-B5. Rules with shared variables (CLUTRR-style composition)

- **Items:** kinship-like composition tables over 6 invented relations (`r1 ∘ r2 → r3`), chains of length k = 2–6
  (P; F up to 10), with held-out compositions.
- **Size:** P 500.
- **Conditions:** as in B1.
- **Primary:** `exec-vsa − text` against k.

### LG-B6. Size of the rule base

- **Levels:**
  - number of rules in the theory (distractors added): P {10, 50}; F {10, 50, 200, 1,000};
  - the `text` condition contains every rule (truncated at the host's context window, with truncation reported);
  - `text-retrieved` contains the top 5.
- **Primary:** the token cost and accuracy of `text` against `exec-vsa` as the rule base grows.

### LG-B7. Answer formats and abstention

- **P:** forced choice and cloze.
- **F:** three-way answers with "unknown". When the store abstains (margin below the calibrated threshold), the
  condition writes "no conclusion" and the model's "unknown" rate is measured.

**Tier P runs:** 4 hosts × 1 job each (B1–B7 item sets together, about 4,000 items × up to 11 conditions). **About 1.0
GPU-h** per SmolLM2 job and **1.5** per Qwen3 job; **about 5 GPU-h** in all.

---

## 5. LG-C: where the store should beat symbols (proposal L2)

### LG-C1. Zero-shot entities and incomplete frames

- **CPU part (P):**
  - Entities whose own frame lacks a fact that holds for similar entities (shared type regularities).
  - The VSA store answers by similarity plus rule (nearest typed neighbour), against exact Datalog (which cannot) and
    RotatE/ComplEx (statistical, transductive).
  - Edges dropped: 0, 25, 50%.
  - **Runs:** P 3 × 2 widths × 3 seeds = **18**.
- **Language-model part:**
  - P: the B1 world's held-out entities, with store vectors composed zero-shot. Folded into B1 as a subset.
  - F: T4 and T7-ROOD held-out terms.

### LG-C2. Aliases and paraphrased mentions

- **CPU part (P):**
  - Mentions with 1–2 character edits, abbreviations and synonyms.
  - Resolution by character n-gram hypervectors (the classic HDC text encoding), against exact match and against a
    fuzzy string matcher (Levenshtein distance plus the alias table).
  - **Runs:** P 3 noise kinds × 2 widths × 3 seeds = **18**.
- **Language-model part (F):** questions mention entities by alias.

### LG-C3. Noisy facts

- **Noise:** wrong edges injected at P {0, 10, 30%} (F {0, 5, 10, 20, 30, 40%}), plus real reader output in F (E11 and
  RD readers).
- **CPU part (P):** closure precision and recall, store against Datalog, with and without abstention. **Runs:** **18**.
- **Language-model part (P):** `exec-vsa@10%`, `exec-vsa@30%`, `exec-sym@10%` and `exec-sym@30%` added to B1's
  conditions on depths 2–4.

### LG-C4. Rules from a few examples

- **Examples:** k = 1, 3, 5, 10 (P); F adds 30 and 100.
- **Methods:**
  - a mapping vector (A1);
  - AMIE-style mining from the same k pairs (`kg_baselines.mine_rules`);
  - E10's structural hypotheses;
  - the model given the k examples in context, then asked about a new entity (in-context learning);
  - F adds a model fine-tuned on the k examples.
- **Size:** P 4 rule kinds × 4 k × 100 queries.
- **CPU runs:** **48**. **GPU:** folded into the B1 jobs.

### LG-C5. Edit propagation (the MQuAKE question)

- **Edit:** in the B1 world, change one base fact (an entity's class).
- **Questions:** entailed facts 1–3 hops downstream, plus unrelated facts (locality).
- **Conditions:**
  - store re-derived after the edit (`exec-vsa`);
  - **stale store** (not re-derived; the control);
  - text with the edited fact;
  - no edit;
  - F adds a gradient edit of a fine-tuned model.
- **Size:** P 300 edits × 3 hops.
- **GPU:** folded into the B1 jobs.

### LG-C6. Cost and compactness

Descriptive, computed from the logs:
- bytes per fact and per rule;
- milliseconds per derivation;
- prompt tokens of each condition.

**Gate G2** (proposal §7):
- the store beats Datalog with a fuzzy front end in at least one of C1–C4 and is not worse in the others;
- C5's store re-derivation propagates edits where the stale store does not.

---

## 6. LG-D: decision trees and decision lists, read by a model (proposal L3)

- **D1. Trees.**
  - **Items:** trees over invented attributes and values of T5-style kinds (tier, status, cadence, area), depth
    {2, 3, 4, 6}. Each item gives one entity's attributes and asks for the leaf.
  - **Size:** P 150 inputs per depth (600). F depths 2–10, 3 worlds, 500 per depth.
- **D2. Decision lists:** length {3, 6, 12} with first-match and a default (P 300).
- **D3. Numeric tests:** half the trees use numeric thresholds (P).
- **D4. Edits:** one node's threshold or leaf changed; does the answer follow? (P 200).
- **D5. Real rule sets (F):**
  - MedCalc-Bench's rule-based calculators (public PMC patients; the licence is checked before use);
  - a RuleArena subset;
  - both hand-encoded as trees and lists.
- **D6. Noisy inputs:** P on the CPU side (A7); F with a model.
- **Conditions:**
  - `none`;
  - the tree as if–else text, plus the entity's attributes;
  - the tree as nested bullets;
  - **the decoded path and leaf from node memory (ours)**;
  - the leaf only;
  - the path from the exact executor;
  - the path from a shuffled tree.
- **Hosts:** as in LG-B.
- **Primary:** `path-vsa − tree-text` by depth.
- **Cost:** P 4 jobs, **about 1.5 GPU-h**.

---

## 7. LG-P: full first-order logic with the model as the search policy (proposal §8)

- **Items (X1, X7):** PrOntoQA-OOD-style first-order problems in invented vocabulary that need:
  - **proof by cases:** "X is an A or a B; every A is a C; every B is a C";
  - **proof by contradiction:** "every A is not a B; …";
  - plus Horn chains.
  - Depth: P 1–4. F 1–8, plus FLDx2-style items and existential witnesses.
- **Size:** P 4 depths × 100 problems = 400, in two variants: few distractor clauses, and many distractor clauses
  (premise selection).
- **Policies:**

| ID | Policy | Tools sound? | Tier |
|---|---|---|---|
| P-cot | the model writes a proof and an answer (generation) | no tools | P (Qwen3 only; small SmolLM2 generations are degenerate) |
| P-unsound | ReAct generation, with an `assert[...]` tool that accepts anything | no | P (Qwen3) |
| **P-score** | **sound tools; at each choice point the model ranks the legal next steps (scored choice)** | yes | P (all hosts) |
| P-react | sound tools; the model writes its calls in ReAct style (generative, grammar-checked) | yes | P (Qwen3) |
| P-random | sound tools; random choice | yes | P |
| P-bfs | sound tools; breadth-first | yes | P |
| P-sos | sound tools; set of support (a fixed heuristic) | yes | P |
| P-ceiling | sound tools; exhaustive with a large budget | yes | P |

- **Budgets:** P {10, 30} steps. F {5, 10, 20, 50, 100}.
- **LG-P3. Premise selection** (retrieving candidate clauses for `rules_for`):
  - methods: VSA retrieval, keyword overlap, exhaustive, oracle;
  - knowledge-base size: P {50, 500} clauses; F up to 10⁵.
- **LG-P4. Soft unification:** aliases and composed entities (F).
- **LG-P5. Budget curve:** P two budgets; F five.
- **LG-P6. Policy learning (F; training):**
  - fine-tune the 360M (and 135M) model as the policy on proofs the tools accept (expert iteration, rounds 0–3, 3
    seeds);
  - test on held-out rules, entities and depths;
  - compare with frozen Qwen3.
- **LG-P7. Host sweep (F):** adds Qwen3.5.
- **Metrics:**
  - accuracy;
  - **soundness** (the share of "proved" answers that are entailed);
  - proofs found within budget;
  - steps and model calls per proof;
  - tokens.
- **Primary (P):** steps to proof of `P-score` against `P-random` and `P-bfs` at equal budget, on held-out rules.
  Soundness of `P-score` should be 1.00 and that of `P-unsound` below 1.
- **Cost:** P 4 hosts, **about 4 GPU-h** (Qwen3 generative policies dominate).

---

## 8. RD: reading facts from real text (proposal §9; critical for self-learning)

**The four stages under test** (proposal §9.2):
1. stated;
2. found;
3. labelled;
4. used.

Each RD experiment isolates one stage. **The ReAct-style reader** (the author's question) is RD-3's main new condition.

### RD-1. Coverage audit: what does the text state, and what can the ontology express?

- **(a) Ontology → text, computed (CPU):** the share of gold edges stated, by relation, for:
  - T4 held-out (done in the proposal: 11.7%);
  - T5 (a sanity check: 100%);
  - T1 MeSH scope notes and T7/T7-ROOD notes (P);
  - F adds WordNet glosses and T8.
- **(b) Text → ontology, teacher-annotated:**
  - **Task:** Claude lists every statement in a definition and labels it:
    - expressible by one of the track's relations (with the partner's words);
    - expressible but the partner is not in the dictionary;
    - not expressible (structural, provenance, quantitative, other).
  - **Sample:** P 100 T4 and 100 T1 definitions (cap **$15**). F 500 per track for T4, T1, T7 and WordNet, with a second
    annotation pass on 100 to check agreement (cap $60).
- **Primary:** the share of statements that are expressible, and the share that are expressible with the partner in the
  dictionary.

### RD-2. Dictionary breadth

- **Question:** does a broader dictionary find more of the stated partners?
- **Expansions:**
  - P: ChEBI synonyms (EXACT and RELATED, release 255) as aliases of the T4 atoms;
  - P: chemical shorthand normalisations ("anti-oxidant" ↔ "antioxidant", hyphen and case variants);
  - F: functional-group names, formula shorthand ("C8" → carbon, "thia" → sulfur), MeSH entry terms for T1/T7.
- **Plus a shrinking curve:** the finder's recall with 25, 50 and 75% of the atoms removed. Its slope estimates what
  *growing* the dictionary buys.
- **Runs:** P 3 expansions + 3 ablation levels × 3 seeds = **12** (CPU).
- **Primary:** recall of stated partners (`filler_recall`) before and after expansion.

### RD-3. Choosing the relation, and the ReAct-style reader

- **Items:** the stated edges of T4 held-out (322 edges, 341 terms) and of T5 `prose` (P). F adds T1 and T7.
- **Readers:**

| Reader | How it chooses | Tier |
|---|---|---|
| `linker` (existing) | which frame makes the definition most likely (needs role sensitivity) | P (from E11 frames) |
| `typeprior` (existing) | the most frequent relation for the partner's type | P (from E11 frames) |
| `host` (existing) | few-shot generative extraction | P (from E11 frames) |
| **`mc-rel`** | explicit multiple choice: "In this definition, how is X related to Y? (a) is a … (b) has role … (c) …" (options: the relations allowed for Y's type), scored by forced choice on the host | P: H-360, H-q17 |
| **`react`** | **ReAct-style matching** in scored-choice form: (1) for each span the finder or fuzzy matcher returns, the model chooses among the top-5 candidate concepts or "none"; (2) for the chosen concept, `mc-rel` chooses the relation or "not stated"; (3) a consistency tool checks the edge against the ontology's typing and inverse rules, and an edge that fails is dropped | P: H-360, H-q17 |
| `react-gen` | the same loop with generated tool calls (`find[...]`, `relations_for[...]`, `propose[...]`, `check[...]`, `commit[...]`) | F: H-q17, H-q35 |
| `teacher` (existing E11 reader) | Claude extracts the edges | P: T4 held-out, cap **$10** |

- **Primary:** precision and recall on stated edges, and accuracy on *ambiguous* edges (where the partner could fill
  more than one relation), for `react` against `linker` and `typeprior`.
- **Cost:** P 2 hosts × 2 tracks, **about 0.5 GPU-h**, plus $10 of teacher spend.

### RD-4. The route by which read facts are used: implicit injection against explicit recall

**The most important experiment for self-learning.**
- **Items:** E11's own read sets and item tests:
  - `t4-heldout-smollm2-v1`: 341 terms; property, paraphrase and entailment tests;
  - `t5-new-smollm2-v1`: 300 invented words; style `prose`.
- **Frames:** the readers' frames from E11's runs (`frames.jsonl`): `oracle`, `stated`, `typeprior`, `linker`; the
  teacher's frames in F.
- **Conditions:**

| Condition | The term's vector | The prompt before each item |
|---|---|---|
| `none` | no frame | — |
| `implicit:<reader>` | the reader's frame composed (E11's route) | — |
| `explicit-sym:<reader>` | no frame | the reader's frame written as recall lines (E12 format, confidence 1.00) |
| `explicit-vsa:<reader>` | no frame | the reader's frame composed in the run's store, then **decoded** (E12's recall) |
| `both:oracle` | gold frame | gold recall lines |
| `definition` | no frame | the definition itself (E11's `context`) |
| `noisy:<p>` (RD-7) | — | gold frame with p% of edges replaced by wrong partners of the same type, written explicitly; and the same injected implicitly |

- **Hosts:**
  - P: H-C5 T4 s1–3 and T5 s1–3 (both routes);
  - P: H-C0′ T4 s1 and T5 s1 (explicit routes only; it has no composing channel);
  - F: + 135M C5, Qwen3 C5, P0.
- **Primary (P), per track:**
  - `explicit-sym:oracle − none`;
  - `explicit-sym:oracle − implicit:oracle`;
  - `explicit-sym:stated − implicit:stated`;
  - Holm over the three tests (property, paraphrase, entailment).
- **Decision:**
  - **If explicit recall of gold or stated frames helps on T4 where implicit injection did not,** the self-learning
    route changes to read → store → explicit recall. The E13 amendment (proposal §9.5, item 6) is then recommended.
  - If explicit recall of gold frames does **not** help on T4, the E11 test items do not reward frame facts. The next
    step is items graded on stated facts only (F: RD-4b).
- **Cost:** P 8 jobs (6 C5 + 2 C0′), **about 0.4 GPU-h each, about 3 GPU-h**.

### RD-5. The bootstrap ("does it pick up steam?")

- **Simulation (CPU, P):**
  - **Corpus:** every T4 entry's ChEBI definition.
  - **Start:** with 25, 50 or 75% of the atoms in the dictionary.
  - **Each round:**
    - read every definition, using the finder over the current dictionary and the `typeprior` relation choice;
    - accept edges above a confidence threshold;
    - add each newly read entry's name and synonyms to the dictionary as a new atom.
  - **Tracked over rounds 1–5:** found stated partners, precision and recall.
  - **Readers:** in P, by their measured precision (the RD-3 readers' precision, applied as error rates).
- **Runs:** P 3 starts × 3 precisions × 3 seeds = **27**.
- **F:** real reading rounds with the teacher and `react` readers on T7-ROOD notes, feeding E13's round structure.
- **Primary:** the slope of recall over rounds, and whether precision holds.
- **Decision:**
  - a positive slope with stable precision supports "build up first, then it picks up steam";
  - a falling precision means errors compound, and reading needs an acceptance test (E13's held-out utility test)
    before commit.

### RD-6. Readers for the structured part of the frame

- **Readers (CPU, P):**
  - T4: elements from the ChEBI formula; charge from the ChEBI charge field; `branch` from the ontology root;
  - T7/T1: parents from MeSH tree numbers.
- **Measured:** recall of those relations (about 42% of the T4 frame).
- **The combined reader** (structure plus stated text) feeds RD-4 in F.
- **Runs:** P **2** (T4, T1).

### RD-7. How much reader noise can the use route tolerate?

Folded into RD-4 (P): the `noisy:<p>` conditions with p ∈ {10, 30, 50%}, explicit and implicit. It gives the reader
precision that self-learning needs.

### RD-8. E13 consequences

No runs. The proposal's §9.5 item 6 amendment is drafted only if RD-4's decision rule fires, and needs the author's
approval before E13's T7-ROOD jobs (54.4996–54.49979) run.

### RD-11. Multi-round active reading with relation discovery (P; pre-registration §11)

*Author, 17 Oct: read multiple times, reason over the available relations, reflect, condense.*

- **Rounds:** 3 rounds of read twice → consolidate (agreement of the two readings) → reflect (cluster `NEW`
  proposals into relation types; accept those with ≥ 8 supporting proposals) → next round with the new relations.
- **Tracks:**
  - T4: 160 ChEBI definitions, with **4 gold relations hidden at the start** (`has_functional_parent`,
    `is_conjugate_acid_of`, `is_conjugate_base_of`, `has_part`);
  - T1: 120 MeSH scope notes.
- **Reader:** Claude, capped at $8 + $7.
- **Measures per round:**
  - facts per definition;
  - visible-relation precision and recall (single pass against consolidated);
  - **gold recall over all relations** (new relations mapped to hidden gold relations by edge Jaccard);
  - each hidden relation's best Jaccard;
  - the reachable ceiling;
  - the `NEW` and unresolved shares.
- **Predictions:**
  - (a) gold recall rises over rounds;
  - (b) at least 2 of the 4 hidden relations are rediscovered (Jaccard ≥ 0.3);
  - (c) consolidation adds ≥ 0.05 precision;
  - (d) the T1 `NEW` share falls.

### RD-12. The interleaved passive ↔ active cycle (F)

- **Design:** passive (gradient) 1 → active reading 1 (RD-11's step) → commit (facts and new relation slots) →
  passive 2 → … → active 3.
- **Machinery:** E13 extended to N rounds; new slots initialised as in E10's learnable ontology.
- **Models:** 135M and 360M C5 on T7-ROOD and T4.
- **Readers:** Claude, and Qwen3-1.7B. With the host as reader, the cycle tests whether passive rounds make the model a
  better reader.
- **Measures per round:**
  - gold recall;
  - loss after held-out terms;
  - E11 items through explicit recall;
  - **tokens to criterion against passive-only training at matched compute** (E10.9's AULC design, with the content
    control of random commits).
- **Cost:** to be estimated from E13's measured cost once its T7-ROOD cycle reports. E13 budgets about 46 GPU-h for one
  T7-ROOD cycle at SmolLM2-360M plus Qwen3-1.7B. A 135M pilot with shortened passive rounds would be a fraction of
  that. Teacher ≤ $40.

### RD-9. Is a larger model a better reader? (F)

- **Readers:** generative extraction and `react-gen` across H-135, H-360, H-q06, H-q17, H-q35 and the teacher, on T4,
  T1 and T7.
- **Measured:** the size curve of stated-edge F1. This answers "is the model not smart enough?" directly.

### RD-10. Ontology density × dictionary breadth (F)

- **Design:** 2 × 3.
  - Dictionary: base, or expanded (RD-2).
  - Frames: thinned T4 frames from the richness stage-1 arms (`C5isa`, `C5comp`, full).
- **Measured:** reading recall and use, with the richness runs as hosts. This answers "do we need a larger ontology?".

---

## 9. Coverage: every proposal section and the experiments that test it

| Proposal section | Claim or open question | Tier P | Tier F |
|---|---|---|---|
| §2.1 facts, two layouts | fact vectors and frames both work | A1 (facts), A2 (frames) | A2 storage variants |
| §2.2 queries and variables | unbinding plus clean-up = variable binding | A1, A2, A3 | A3 rule records |
| §2.3 functions, nesting, order | depth limit; commutativity | A4 | A4 GHRR/matrix |
| §2.4 (1) mapping vectors | one vector per rule, from few examples | A1, C4 | A1 full, mapping memory |
| §2.4 (2) rule records + executor | Datalog closure via store | A3, A11 | A3 rule records as vectors |
| §2.4 (3) relation algebra | KGE patterns, implicit rules | A12 | A12 + operator axioms (IterE) |
| §2.5 who runs the loop | symbolic controller / model tools / model internally | A3; LG-P | LG-P6, LG-T |
| §2.6 trees and lists | node memory vs nested vs bundle vs list | A7, D1–D4 | A7/D full, D5 real rules |
| §2.6 numbers | FPE nearness vs order | A6, D3 | A6 thermometer |
| §2.7 the "direct" fragment table | ∧ ∨ ¬ ∃ and limits | A5 | A5 nested ∀∃ |
| §3.2 survey warning | correlated fillers break transfer | A1, A10 | A10 whitening |
| §4 our results | capacity, chains, order, reverse | A2, A4 | A2 at WordNet scale |
| §5 verdict: soft settings | where the store beats symbols | C1–C5 | C1–C5 full + LM |
| §6 illustration | random-vector mechanics | A1, A2, A7 (supersede) | — |
| §7 L1 model reads conclusions | exec-vsa vs rules as text | B1–B7 | B full hosts, worlds, CoT |
| §7 L2 G2 | soft-setting advantage | C1–C6 | C with LM, real readers |
| §7 L3 trees read by model | path-vsa vs tree text | D1–D4 | D5, D6 |
| §7 L4b / §8.6 internal execution | model learns to execute | — | LG-T2, LG-P6 |
| §8 the prover as search policy | sound tools + model policy | P (all policies) | P3–P7 |
| §9.2 stage 1 stated | text states the ontology | RD-1a, RD-6 | RD-1 all tracks |
| §9.2 stage 1b expressible | ontology expresses the text | RD-1b | RD-1b full, agreement |
| §9.2 stage 2 found | dictionary coverage | RD-2 | RD-2 full expansions |
| §9.2 stage 3 labelled | relation choice; ReAct reader | RD-3 | RD-3 react-gen; RD-9 |
| §9.2 stage 4 used | implicit vs explicit route | RD-4 | RD-4 + 135M, Qwen3, teacher frames |
| §9.4 "larger ontology?" | breadth vs density | RD-2 | RD-10 |
| §9.4 "model smart enough?" | size curve of readers | RD-3 (2 hosts) | RD-9 |
| §9.4 "picks up steam?" | bootstrap | RD-5 (simulation) | RD-5 real rounds |
| §9.5 (6) E13 | amendment | RD-8 (decision only) | — |

### The training arms (LG-T, tier F only)

| ID | Arm | Question | Runs |
|---|---|---|---|
| LG-T1 | C5 trained on **closure-augmented** frames (forward-chained derived facts added to T4/T7 frames), against C5 and a shuffled-closure control | does making implicit facts explicit help the model (the richness question with free richness)? | T4, T7 × 135M, 360M × 3 seeds × 2 arms = 24 runs, about 1.5 GPU-h each |
| LG-T2 | readout arms trained to chain two unbinding steps | can the model execute chains internally? | after the readout arms report; 360M × 3 seeds × 2 variants |
| LG-T3 | rule-following fine-tune on `exec-vsa` prompts against `text` prompts | does training with store conclusions generalise to held-out rules (the RuleTaker comparison at small scale)? | 360M × 3 seeds × 2 = 6 runs |
| LG-T4 | = LG-P6 policy learning | — | — |

---

## 10. Run inventory and cost

### Tier P (preliminary; pre-registered as E14 and queued or run now)

| Experiment | Runs | Lane | Cost |
|---|---:|---|---|
| LG-A1 mapping vectors | 3,240 cells | CPU (direct) | ≈ 20 CPU-min |
| LG-A2 chains | 1,152 sets | CPU | ≈ 40 CPU-min |
| LG-A3 joins | 216 | CPU | ≈ 15 CPU-min |
| LG-A4 nesting/order | 288 | CPU | ≈ 5 CPU-min |
| LG-A5 connectives | 96 | CPU | ≈ 5 CPU-min |
| LG-A6 numbers | 72 | CPU | ≈ 2 CPU-min |
| LG-A7 trees/lists | 864 | CPU | ≈ 40 CPU-min |
| LG-A8 resonator | 36 | CPU | ≈ 10 CPU-min |
| LG-A11 real stores | ≈ 12 | CPU | ≈ 15 CPU-min |
| LG-A12 baselines | 6 | CPU | ≈ 20 CPU-min |
| LG-C1–C4 CPU parts | 102 | CPU | ≈ 20 CPU-min |
| LG-B1–B7, C1/C3/C4/C5 model parts | 4 jobs (one per host) | GPU | ≈ 5 GPU-h |
| LG-D1–D4 | 4 jobs | GPU | ≈ 1.5 GPU-h |
| LG-P (all policies, P3, P5) | 4 jobs | GPU | ≈ 4 GPU-h |
| RD-1a / RD-1b | 1 + 1 | CPU + teacher | minutes; ≤ $15 |
| RD-2 | 12 | CPU | minutes |
| RD-3 | 4 jobs + teacher | GPU + teacher | ≈ 0.5 GPU-h; ≤ $10 |
| RD-4 + RD-7 | 8 jobs | GPU | ≈ 3 GPU-h |
| RD-5 | 27 | CPU | minutes |
| RD-6 | 2 | CPU | minutes |
| Reports (LG, RD) | 3 | CPU lane | — |
| **Total** | | | **≈ 14 GPU-h, ≈ 4 CPU-h, ≤ $25 teacher** |

### Tier F (full scale; after the gates, each block pre-registered separately)

| Block | Scale | Cost (estimate) |
|---|---|---|
| LG-A full grids (A1–A12, 10 seeds, WordNet scale, GHRR, trained operators, whitening) | ≈ 250,000 cells | ≈ 40 CPU-h |
| LG-B full (3 worlds, 7 depths, 3-way answers, CoT, 10 hosts) | ≈ 30 jobs | ≈ 30 GPU-h |
| LG-C full with the model (T4/T7-ROOD zero-shot terms, aliases, real readers, fine-tune edits) | ≈ 20 jobs + 6 fine-tunes | ≈ 20 GPU-h |
| LG-D full + real rule sets (MedCalc-Bench, RuleArena) | ≈ 15 jobs | ≈ 10 GPU-h |
| LG-P full (P3 at 10⁵ clauses, P4, budgets, Qwen3.5) | ≈ 15 jobs | ≈ 15 GPU-h |
| LG-P6 policy learning (expert iteration, 2 sizes, 3 seeds, 3 rounds) | 18 trainings + evals | ≈ 30–40 GPU-h |
| LG-T1–T3 training arms | 36 runs | ≈ 55 GPU-h |
| RD full (RD-1b 2,000 annotations, RD-3 react-gen, RD-4 + hosts, RD-5 real rounds, RD-9, RD-10) | ≈ 40 jobs | ≈ 20 GPU-h; ≤ $100 teacher |
| **Total** | | **≈ 180–190 GPU-h, ≈ 40 CPU-h, ≤ $160 teacher** |

---

## 11. Order of work and decision points

1. **Now (CPU, no queue impact):**
   - X1–X4;
   - the LG-A grids;
   - RD-1a, RD-2, RD-5, RD-6;
   - RD-1b and RD-3's teacher part, which are capped.
2. **Queued now at 54.49595–54.49599 (GPU):**
   - RD-4/RD-7 first (54.49595);
   - then RD-3 (54.49596);
   - LG-B/C (54.49597);
   - LG-D (54.49598);
   - LG-P (54.49599).
   - Reports at 54.495995 (CPU lane).
   - Each job is queued as soon as its code passes its tests and a labelled smoke.
3. **Gate G0** (from LG-A) and the **RD-4 decision**. These are reported to the author, who decides on the E13
   amendment.
4. **Gate G1** (from LG-B), then **gate G2** (from LG-C).
5. **Tier F,** block by block, each with its own pre-registration addendum and the author's go-ahead. The GPU-heavy
   blocks (LG-T, LG-P6) come after the program's current queue.
