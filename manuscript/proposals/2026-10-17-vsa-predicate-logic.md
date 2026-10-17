# Logic in the store: encoding predicate logic and decision procedures with VSA binding

**Written:** 2026-10-17 by workplace-fd, at the author's request:
> "Create a document outlining and exploring this idea - see if it is worth exploring and how can we explore it via what
> methodology - also do research into how predicate logic can be encoded and if there are already vsa analogs that can do
> this - maybe we can encode predicate logic directly? Via relationships and functions and premises and deductions?"

**The idea (author, 17 Oct):** VSAs already encode an ontology through binding, unbinding and composition. Could the same
algebra encode *logic* (predicates, rules, decision trees), so that a model can reason with discrete rules and apply them
zero-shot, which statistical learning does poorly?

**Status:** exploration and proposal.
- Nothing here is pre-registered or queued, and nothing goes on the queue until the author decides (§11).
- The only new numbers are a CPU illustration of the algebra on random vectors, with no language model (§6,
  `vsa_logic_demo.py`, `vsa_logic_demo.json`). It checks the mechanics. It is **not a result for the paper**.
- All other numbers are our existing results, with their sources.
- The literature in §3 was checked by three research passes on 17 Oct against arXiv, publisher pages, ACL Anthology and
  DBLP. Items they could not confirm are marked *(unverified)*.

---

## 0. Short answer

- **Can predicate logic be encoded in a VSA directly?**
  - **Yes, for a useful fragment.** Facts with several arguments, relations, nested function terms (to a bounded depth)
    and if–then rules (Horn clauses, the Datalog fragment) all have standard encodings more than 30 years old (Smolensky
    1990; Plate 1995).
  - **One step of deduction is algebra.** Queries with variables are answered by unbinding. Some rules are a single
    "mapping vector" that applies to any arguments, including ones never seen. Rule chains are repeated
    unbind-then-clean-up steps.
  - **Full first-order logic is not direct.** Nested quantifiers, disjunctive conclusions, negation and proof search need a
    control loop around the algebra. The VSA then acts as the prover's memory, not as the prover.
- **Can the model then reason zero-shot?** That depends on who does the reasoning.
  - **The algebra applies a rule exactly to new entities.** Its operations do not depend on what the arguments are, so it
    is systematic by construction. Our illustration confirms this (§6). At 1,024 dimensions, a rule vector estimated from
    3 examples was applied to 500 facts about 1,000 unseen entities: 0.998 correct, and 1.000 from 10 examples.
  - **Our own data say the model will not do this implicitly.** The binding test P1 was null: the model ignores bound
    roles inside its input vectors.
  - **But the model reads decoded results very well.** E12 lifted role assignment from 0.55 to 0.88 on T4 (§4).
  - **So the realistic design is "the store reasons, the model reads".** Rules live in the store; an executor applies
    them by unbinding and clean-up; the derived facts are decoded into the prompt, as in E12. A later, riskier stage
    trains the model to call the executor or to run the steps itself.
- **Is it worth exploring?** Yes, scoped narrowly and gated (§5).
  - **For:**
    - It extends the program's strongest result (E12).
    - The first two stages are cheap: CPU, then scoring only.
    - The research passes found **no prior work that pairs a VSA rule store with a language model for deduction**, or
      that tests rules *added after training* on unseen entities.
    - The algebraic pieces are old and sound: Kanerva's mapping vectors (2010), Kvasnička & Pospíchal's modus ponens in
      HRR (2006), and Crawford et al.'s 13-level WordNet traversal with clean-up (2016).
  - **Against:**
    - An exact symbolic engine (Datalog or Prolog) is the obvious baseline, and on clean symbolic data it wins.
    - The VSA must earn its place where symbols fail: entities known only by a composed vector, noisy or paraphrased
      mentions, rules induced from a few examples, and one compact store shared with the model.
- **Full first-order logic (§8):** the language model can be the prover's *search policy*, through ReAct-style tool
  calls, if sound tools perform and check every step. A weak policy then costs search time, not correctness.
- **Reading facts from real text (§9) fails for four separate reasons,** and only one is "the model is too small":
  - the definitions state only 12% of the ontology's facts;
  - the reader's relation choice relies on role sensitivity the model does not have;
  - even gold facts do not help through the implicit route;
  - the 360M model cannot extract facts generatively.
- **Recommended first steps:**
  - the reading diagnostics of §9 (most urgent for self-learning);
  - stage L0 (§7), a CPU-only fidelity study of rule execution at the widths and loads of our real ontologies, with its
    own short pre-registration. The deeper stages are decided by its outcome.
- **The run-by-run plan** (a preliminary tier and a full-scale tier) is in
  [`2026-10-17-vsa-logic-experiment-plan.md`](2026-10-17-vsa-logic-experiment-plan.md).

---

## 1. The idea in plain words

**The problem.**
- Neural networks learn rules statistically. They pick up the surface features that tend to come with a rule's
  conclusion, rather than the rule itself.
- The cleanest demonstration is SimpleLogic (Zhang et al., IJCAI 2023):
  - A BERT model scored near 100% on propositional deduction problems drawn like its training data.
  - It fell to about 57% at depth 6 when the same kind of problem was drawn by a different sampler.
  - Hand-set weights solve the task exactly, so the failure is not capacity. It is learning shortcuts (for example, that
    the number of rules correlates with the answer).
- This is the old systematicity argument in modern form. Fodor & Pylyshyn (1988) and Marcus (2001) argued that networks
  lack operations over variables. SCAN and COGS showed sequence models failing on new combinations.
- Large models narrow the gap but still fail on longer proofs (PrOntoQA; "Faith and Fate") and on counterfactual rules.

**What a VSA offers.**
- Binding is variable binding.
  - `AGENT ⊛ john` puts the value *john* into the slot AGENT.
  - Unbinding AGENT gets *john* back, whatever *john* is.
- Because the operation ignores what the filler is, a rule written in terms of slots applies the same way to any
  filler. That includes fillers the system has never seen.
- This is the property the critics said networks lack, and it holds by construction rather than by learning.

**Where our program stands.**
- The store already holds *facts*. Each term's composed vector is a frame `Σ_r R_r ⊛ y_r` of its relations, and E12
  shows a model can use those facts once they are decoded into text.
- The step from facts to *logic* adds three things:
  - **(a) rules** stored next to the facts;
  - **(b) a deduction procedure** that applies the rules with the algebra;
  - **(c) a way to hand the conclusions to the model**, which E12 already provides.

**What "zero-shot" means here.** Three different claims, to be kept apart:

| Kind | What is new at test time | Example |
|---|---|---|
| **Z1: new entities** | the arguments | the rule "if X is the conjugate acid of Y, Y is the conjugate base of X", applied to a chemical never seen in training, whose vector is composed from the ontology |
| **Z2: new rules** | the rule itself, added after training with no weight change | an analyst writes "the escalation contact of a system is its owner's manager"; the system answers for every system at once |
| **Z3: new depth or combinations** | the length of the chain, or a new pairing of known rules | trained or demonstrated on 2-step chains, asked a 5-step one |

The literature has many systems that do Z1 (rule learners, below) and one strong behavioural system for Z2 and Z3 with
rules given as text (RuleTaker). None does Z2 with a persistent, editable vector store read by a language model.

---

## 2. How predicate logic maps onto VSA operations

**Notation:**
- `⊛` bind (circular convolution in HRR);
- `⊘` unbind (circular correlation, or the exact inverse for unitary vectors);
- `+` bundle (superposition);
- `clean(v)` snap `v` to the nearest stored vector (clean-up memory);
- `B^x` fractional power encoding of a number `x`.

**The mapping at a glance:**

| Logic | VSA encoding | Used by | Main limit |
|---|---|---|---|
| constant (an entity) | a random vector, or for our terms the ontology-composed vector | every operation | none, if the vectors are nearly orthogonal |
| predicate with arguments `p(a, b)` | `PRED⊛p + ARG1⊛a + ARG2⊛b` (a *fact vector*), or the per-entity frame `Σ_r R_r⊛y_r` | lookup, rules | about 8 bound pairs per vector at 256 dims; 16 (exact) to 32 (0.98) at 1,024 (§4, §6) |
| a set of facts (knowledge base) | a clean-up memory of fact or frame vectors; **not** one bundle | queries | one bundle of everything fails (our "global memory": 0.07 against 0.88) |
| query with a variable `p(a, ?x)` | unbind the variable's role: `clean(fact ⊘ ARG2)` | deduction steps | needs the right fact or frame first |
| function term `f(g(a))` | nested binding `F⊛(ARG⊛(G⊛(ARG⊛a)))`, or positional permutations | nested data | noise grows with depth unless cleaned up at each level; circular convolution is commutative, so order must come from roles or permutations |
| rule `p(x,y) → q(y,x)` (renames or swaps roles) | **one mapping vector** `M` with `M ⊛ fact ≈ conclusion`, written by hand or estimated from examples | forward chaining in one step | covers only rules that rearrange roles and swap predicate symbols |
| rule with a shared variable `p(x,y) ∧ q(y,z) → r(x,z)` | an explicit **rule record** (premise and conclusion patterns with variable roles) run by an executor; for chains, repeated unbind + clean-up | multi-step deduction | each step can fail; with clean-up, error per step stays constant |
| relation algebra (inverse, composition, symmetry) | operations on relation keys: inverse = conjugate, composition = product (as in RotatE) | link prediction | composition is commutative under HRR/FHRR, so `r1∘r2 = r2∘r1` |
| ∧ (and) | all premises matched; or one bound conjunction key `a⊛b` | matching | bundled premises give soft matching (k of n) |
| ∨ (or) | bundle the alternatives | matching | cannot tell which branch held |
| ¬ (not) | no native operation: closed-world absence (similarity below a threshold), a NOT role, or a learned negator (as in BetaE) | negation as failure | the threshold needs calibration |
| ∀ (for all) | implicit: a rule over variables applies to every filler | rules | nested quantifiers have no direct form |
| ∃ (there exists) | "some stored vector is similar enough" | queries | a threshold, so a soft answer |
| numbers and thresholds | `B^x` (fractional power encoding); similarity is a kernel in `x` | decision trees | similarity is about nearness, not order; `x ≤ t` needs a set encoding or an explicit compare step |
| decision tree | a node memory: `TEST⊛feature + VAL⊛value + YES⊛child + NO⊛child` | classification | exact to depth 8 at 1,024 dims with clean-up per node; one nested vector degrades from depth 2 and is at chance by depth 6 (§6) |

### 2.1 Facts: two equivalent layouts

- **Fact vectors.** One vector per fact: `PRED⊛conjugate_acid_of + ARG1⊛X + ARG2⊛Y`. This is Plate's and Kanerva's
  layout, and the natural one for arbitrary predicate arity.
- **Entity frames (ours).** One vector per entity, holding all its outgoing binary facts: `v_X = R_conj⊛Y + R_isa⊛F + …`.
  This is what CG-VSA builds, and what E12 decodes.
- **Converting between them:** the frame is a bundle of the entity's facts with the subject slot left implicit.
- **Arity:** predicates with three or more arguments need fact vectors, or a reified event entity that has a frame of
  its own.

### 2.2 Queries and variables

- A query `conjugate_acid_of(X, ?y)` is one unbinding: `clean(v_X ⊘ R_conj)`. E12's "recall" is exactly this.
- **Variable binding and unification are unbinding plus clean-up.**
  - The variable is a role; finding its value is unbinding the role from a matching fact.
  - Snapping the result to a stored constant turns a noisy vector back into a symbol.
  - Two premises that share a variable must produce the same cleaned symbol, which is a symbolic equality test after
    clean-up.

### 2.3 Functions and nested terms

- **The encoding.** A term like `mother(father(a))` nests: `F_mother ⊛ (ARG ⊛ (F_father ⊛ (ARG ⊛ a)))`.
- **Two problems:**
  - **Noise grows with nesting**, because each level multiplies the error. Clean-up at every level fixes this, but only
    when the intermediate terms are stored, so that something exists to clean up against.
  - **Circular convolution is commutative**, so `F_mother ⊛ F_father = F_father ⊛ F_mother`. Our binding report
    measured this: path order is not recoverable (0.49–0.50, chance 0.5). Order has to come from distinct argument
    roles, from permutations (`ρ(x)` for position), or from a non-commutative binding (matrix binding, GHRR).
- **In practice,** function terms are best turned into named entities with frames (a reified `father(a)` gets its own
  vector), which is how ontologies already work.

### 2.4 Rules (premises → conclusion)

Three encodings, from the most "algebraic" to the most "symbolic":

1. **A mapping vector (one vector per rule).**
   - **Idea:** for rules that only rearrange roles, a single vector `M` turns any premise into its conclusion:
     `M ⊛ fact(p, a, b) ≈ fact(q, b, a)`.
   - **Example:** the inverse rule `parent(a, b) → child(b, a)`. By hand,
     `M = ARG2⊛ARG1⁻¹ + ARG1⊛ARG2⁻¹ + child⊛parent⁻¹`. Applied to a fact, the matching terms give the conclusion; the
     other six cross terms are noise.
   - **From examples:** `M` can be estimated from a few pairs of premise and conclusion by averaging their correlations
     (Plate; Kanerva's "dollar of Mexico"; Emruli, Gayler & Sandin) or by least squares per frequency (Neumann).
   - **Implications as transformations:** Kvasnička & Pospíchal (2006) encode implications themselves this way, so that
     modus ponens is a convolution (§3.3).
   - **The key property:** `M` contains no entity, so it applies to entities never seen. That is Z1, plus Z2 from
     examples. Our illustration: 3 examples suffice at 1,024 dimensions (§6).
   - **Limit:** this covers inverse, symmetric and role-renaming rules, sub-relations, and "same arguments, different
     predicate" rules. A rule with a variable shared between two premises (a join) is not a single vector.
2. **An explicit rule record run by an executor.**
   - **The record:** `RULE = IF ⊛ (premise patterns, with variable roles X, Y, Z) + THEN ⊛ (conclusion pattern)`.
   - **Running it:** an executor loop matches the premises against the store, unbinds and cleans up the variables,
     checks that shared variables agree, builds the conclusion by binding, and writes it back to the store.
   - **This is forward chaining,** the way the Semantic Pointer Architecture runs rules, with an action-selection loop.
     Backward chaining runs the same steps from the goal.
   - **For chain rules** (`located_in` is transitive; "owner's manager"), execution is repeated unbind + clean-up: one
     frame per hop. Our binding report shows this works with one store per entity (two-hop 0.88) and fails with one
     shared memory (0.07).
3. **Rules as relation algebra.**
   - Knowledge-graph embeddings that are VSAs in all but name (HolE uses HRR's correlation; RotatE uses FHRR-style
     rotations) model inverse (a conjugate), symmetry and composition (a product) as properties of relation vectors.
   - Rules there are *implicit* in learned geometry: learned, not written, and not editable.

### 2.5 Deduction: who runs the loop

The algebra gives single steps. Deduction needs a loop that:
- chooses which rule to apply;
- applies it;
- cleans up;
- stores the result;
- stops.

The options:

| Who runs the loop | What it is | Precedent |
|---|---|---|
| a small symbolic controller | a Python loop over unbind/clean-up calls | SPA's basal-ganglia action selection; NVSA's rule executor |
| the language model, as a tool user | ReAct-style calls such as `apply[rule, entity]` | our E12 agent pilot: Qwen3-1.7B makes well-formed recall calls 81% of the time (§4) |
| the language model, internally | trained to unbind and chain in its own layers | our readout arms (trained to unbind), queued; no precedent for rules |

Forward chaining to saturation (derive everything, then store it) turns deduction into lookup. The model then only
needs E12-style recall. That is the simplest first design, and it costs store capacity.

### 2.6 Decision trees and decision lists

- **(i) Node memory.**
  - Each internal node is `TEST⊛feature + VAL⊛value + YES⊛child + NO⊛child`, stored under its own address.
  - An input is a record `Σ_f FEAT_f ⊛ value_f`.
  - **Execution:**
    - unbind TEST and clean up, giving the feature;
    - unbind that feature's role from the input and clean up, giving the input's value;
    - unbind VAL and clean up, giving the node's value;
    - compare the two values;
    - unbind YES or NO and clean up, giving the next node's address;
    - repeat until a leaf (a class vector).
  - **Clean-up after every step keeps the error per step constant,** so depth costs little (§6: 1.00 at depth 8 with
    1,024 dimensions).
- **(ii) One nested vector.** The whole tree is a single vector, with each child's vector bound inside its parent.
  It is compact, but there is nothing to clean up against between levels, so noise compounds. In §6 it is at 0.49 by
  depth 4 and at chance (0.10) by depth 6, at 1,024 dimensions.
- **(iii) A flat bundle of paths.**
  - Each root-to-leaf path is a conjunction of tests, a Horn rule `f₁=v₁ ∧ f₂=v₂ → class`.
  - The tree is the bundle of its path rules, matched by similarity (how many tests hold).
  - This is the soft, "k of n" form typical of hyperdimensional classifiers. It suits fuzzy inputs, not exact
    execution.
- **(iv) A decision list.** An ordered list of rule records ("if A then X, else if B then Y, …"), executed in order.
- **Numeric tests.**
  - Fractional power encoding (`B^x`) makes similarity a smooth kernel of distance, so "x is near 5" is native.
  - "x ≤ t" is not native. It can be encoded as a bundle over the allowed range, or done as a compare step after
    clean-up.
- **What zero-shot means for trees:** a new tree is written into node memory at test time. The executor is the same for
  every tree, so nothing is trained. The test is whether execution stays exact as depth, width and load grow, and whether
  a model can use the decoded path.

### 2.7 So: can predicate logic be encoded "directly"?

| Fragment | Direct in VSA? | Notes |
|---|---|---|
| Facts, relations of any arity, queries with variables | **yes** | standard; capacity is the constraint |
| Role-rearranging rules (inverse, symmetric, sub-relation, renaming) | **yes, one vector per rule** | a mapping vector; can be estimated from a few examples |
| Horn rules with joins (Datalog), bounded depth | **yes, with an executor loop and clean-up** | the executor is a few lines of control; the work is algebra |
| Function terms, bounded nesting | **yes, with care** | clean-up per level; permutations for order |
| Decision trees and decision lists | **yes** (node memory) | exact at depth 8 in §6 |
| Negation as failure, existence | **approximately** | thresholds; calibration needed (experiment 00 has calibrators) |
| Full first-order logic (nested quantifiers, disjunctive heads, equality reasoning) | **no**, not directly | needs a prover; the VSA can be its memory and similarity-based matcher |

The Datalog fragment plus decision trees covers most of what ontologies and business rules need: inheritance down
`is_a`, inverse and symmetric relations, chains such as "owner's manager", typed constraints and triage trees.

---

## 3. What already exists

This summarises three research passes run on 17 Oct.
- **Verification:** citations were checked against arXiv, DOI and publisher records, ACL Anthology, OpenAlex or DBLP.
- **Survey-only items** (seen only in the bibliography of the Kleyko et al. Part II survey) are marked *(survey)*.
- **Unconfirmed items** are marked *(unverified)*.
- **Negative findings** ("none found") are search results, not proofs of absence.

### 3.1 Structure in vectors: the foundations

| Work | What it does | Limit |
|---|---|---|
| Smolensky 1990, "Tensor product variable binding…", *Artificial Intelligence* 46:159–216 | Structure = Σ filler ⊗ role; exact unbinding with dual roles | dimension multiplies with nesting depth |
| Plate 1995, "Holographic reduced representations", *IEEE TNN* 6(3):623–641; book, CSLI 2003 | circular convolution compresses the tensor product back to *n* dims; predicates as frames (`eat + agent⊛john + object⊛fish`); decode by correlation plus clean-up | approximate; noise compounds with depth |
| Kanerva 2009, "Hyperdimensional computing…", *Cognitive Computation* 1(2):139–159 | binary spatter codes: XOR binding, majority bundling, permutation for order | — |
| Gayler 2003, "VSAs answer Jackendoff's challenges…", ICCS/ASCS; arXiv:cs/0412059 | coins "VSA"; argues binding plus superposition plus clean-up meets the variable-binding challenges | position paper |
| Kleyko, Rachkovskij, Osipov, Rahimi 2023, survey Parts I and II, *ACM Computing Surveys* 55(6):130 and 55(9):175 | the standard reference; Part II §3.1.3.6 indexes VSA logic inference | — |

### 3.2 Rules as transformations that carry over to new fillers (closest to our §2.4 option 1)

- **Kanerva 2010.** "What we mean when we say 'What's the dollar of Mexico?'", AAAI Fall Symposium, pp. 2–6.
  - Countries are frames (`NAM⊛usa + CAP⊛wdc + MON⊛dol`). The single mapping vector `USA ⊛ MEX` moves *every* filler
    with the same role: dollar ↦ peso, Washington ↦ Mexico City.
  - This is the canonical "one vector, any filler" result.
- **Plate.** "Structure matching and transformation…" (1997) *(survey)*, with hand-built transformation vectors; and
  "Analogy retrieval and processing…", *Expert Systems* 17(1), 2000.
- **Neumann 2002.** "Learning the systematic transformation of holographic reduced representations", *Cognitive
  Systems Research* 3(2):227–235.
  - It learns `T` from example pairs so that `T ⊛ A ≈ B`.
  - `T` generalised to new elements and to structures more complex than the training ones. These were toy structures.
  - Our least-squares estimate in §6 is the closed-form version.
- **Emruli, Gayler & Sandin 2013,** IJCNN, pp. 872–879; **Emruli & Sandin 2014,** *Cognitive Computation* 6(1):74–88.
  - Example mappings are stored in a sparse distributed memory and recalled to map new structures.
  - It works best at about 10⁴ dims.
- **Rasmussen & Eliasmith 2011.** "A neural model of rule generation in inductive reasoning", *Topics in Cognitive
  Science* 3(1):140–153.
  - A Raven's-matrix rule is the average of transformations between neighbouring cells, applied to the last cell.
  - Panels arrive already symbolic, and mostly progression rules are handled.
- **A warning that matters for us.** The survey (Part II §3.1.1.2) notes that learned transformations assume
  *dissimilar* atoms and "might not work as expected" when fillers are similar to each other.
  - Our composed term vectors are correlated (they share parts of the ontology).
  - Rule transfer with them is therefore untested. L0 must test it.

### 3.3 Deduction with VSA operations (closest to "premises and deductions directly")

- **Kvasnička & Pospíchal 2006,** "Deductive rules in holographic reduced representation", *Neurocomputing*
  69(16–18):2127–2139.
  - **The closest prior work on direct deduction.**
  - An implication `p → q` is encoded in HRR, and a transformation vector turns it into a premise-to-conclusion pair, so
    that modus ponens and modus tollens are a convolution.
  - Universal instantiation (`P(x) → P(a)`) is a convolution with `x* ⊛ a`.
  - A three-rule chain (give → own → can sell) is shown.
  - **Limits:**
    - The substitution vector is built from the already-instantiated target, so unification effectively happens outside
      the algebra.
    - The encoding of disjunction produces fallacies through cross-talk.
    - Every step needs clean-up, and scale is not analysed.
- **Summers-Stay 2017–2019** (arXiv:1707.03232; *BICA* 25:113–118, 2018; IntelliSys 2019 *(survey)*).
  - Propositional deduction with summed word embeddings, where `a → b` is encoded as `−a + b`, so chains cancel to the
    conclusion.
  - It uses real embeddings, but stays propositional.
- **Production systems:**
  - Stewart, Choo & Eliasmith 2010, CogSci *(survey)*: the basal ganglia select the rule whose condition best matches
    the state.
  - Dolan & Smolensky 1989, "Tensor Product Production System", *Connection Science* 1(1): an earlier tensor version.
  - Spaun (Eliasmith et al. 2012, *Science* 338:1202–1205; *How to Build a Brain*, OUP 2013): eight fixed tasks with
    hand-designed control.
  - Choo & Eliasmith 2013, CogSci *(survey)*: instructions as rule sequences.
- **Crawford, Gingerich & Eliasmith 2016,** "Biologically plausible, human-scale knowledge representation", *Cognitive
  Science* 40(4):782–821.
  - **The best precedent for multi-hop lookup with clean-up.**
  - It covers all 117,659 WordNet synsets, 512-dim pointers `Σ relation ⊛ target`, with unbinding and clean-up at every
    hop.
  - Results: 99% on single relations, and 98% on hierarchy traversal up to 13 levels.
  - It has no rules and no variables.
- **Widdows & Cohen 2015,** "Reasoning with vectors…", *Logic Journal of the IGPL* 23(2):141–173. Approximate deduction
  over subject–predicate–object bindings (predication-based semantic indexing), used for drug repurposing.
- **Tensor product representations:**
  - Schlag & Schmidhuber 2018 (NeurIPS);
  - the TP-Transformer (arXiv:1910.06611; venue *(unverified)*);
  - Soulos et al. 2023, "Differentiable tree operations…" (ICML): tree operations as linear tensor-product
    operations, 100% out of distribution on synthetic tasks;
  - Smolensky et al. 2022, "Neurocompositional computing" (*AI Magazine* 43).
  - None of these does first-order unification.
- **Domingos 2025,** "Tensor Logic", arXiv:2510.12269. Datalog rules written as tensor contractions (einsum). Not a VSA,
  and no unification of function terms.
- **Not found:** a VSA paper that implements general first-order unification or resolution.

### 3.4 Clean-up, factorisation and capacity

- **Resonator networks:**
  - Frady, Kent, Olshausen & Sommer 2020, *Neural Computation* 32(12):2311–2331, arXiv:2007.03748;
  - Kent, Frady, Sommer & Olshausen 2020, ibid. 2332–2388, arXiv:1906.11684.
  - They factor a bound product of unknown factors by iterated "search in superposition" over known codebooks.
  - This is the tool for unifying several variables at once. It has no convergence guarantee.
- **Capacity theory:**
  - Frady, Kleyko & Sommer 2018, *Neural Computation* 30(6):1449–1513, which covers crosstalk noise across VSA models;
  - Thomas, Dasgupta & Rosing 2021, *JAIR* 72:215–249;
  - Clarkson, Ubaru & Yang 2023, arXiv:2301.10352.
  - **Not found:** a capacity theory for *multi-step deduction* with clean-up at each hop. §6B's "(hop-1 accuracy)^depth"
    is the simple empirical version.

### 3.5 Automata, controllers, classifiers and trees

- **Finite-state machines:**
  - Osipov, Kleyko & Legalov 2017, IECON, pp. 3276–3281. The transition table is `Σ input ⊛ state ⊛ ρ(next)`, and the
    next state is read by unbinding plus clean-up.
  - Cotteret et al. 2024, *Neural Computation*, arXiv:2212.01196: in attractor networks.
- **Other machines:** a hyperdimensional stack machine (Yerxa et al. 2018 *(survey)*) and context-free grammars (beim
  Graben et al. 2022 *(survey)*).
- **Flat rule sets as controllers:** Levy, Bajracharya & Gayler 2013 *(survey)*; Neubert, Schubert & Protzel
  2016/2019 (*KI* 33(4)).
  - A robot policy is `Σ condition ⊛ action`, and unbinding with the current state gives the action.
  - This is the "flat path bundle" of §2.6 (iii).
- **Hyperdimensional classifiers** (e.g. Kleyko et al. 2018, *IEEE TNNLS* 29(12)): class prototypes, not rules.
- **Not found:** decision trees or decision lists stored and executed in hypervectors, or any handling of rule priority
  and defaults.

### 3.6 VSA-like methods under other names: knowledge-graph and query embeddings

- **Knowledge-graph embeddings:**
  - TransE (Bordes et al., NIPS 2013).
  - **HolE** (Nickel, Rosasco & Poggio, AAAI 2016, arXiv:1510.04935) scores triples with HRR's correlation.
  - ComplEx (Trouillon et al., ICML 2016).
  - Hayashi & Shimbo (ACL 2017) proved HolE and ComplEx equivalent, so HRR and complex-diagonal (FHRR-style) relation
    operators are interchangeable.
  - **RotatE** (Sun et al., ICLR 2019) is FHRR binding with phase vectors. It models symmetry, inversion and
    composition (a product of phases, so commutative).
  - **What they share:** rules are implicit in learned geometry, and entities are transductive (an unseen entity has no
    vector).
- **Logical queries in vector space:**
  - GQE (Hamilton et al., NeurIPS 2018): ∃ and ∧.
  - Query2Box (Ren, Hu & Leskovec, ICLR 2020): adds ∨ via disjunctive normal form.
  - BetaE (Ren & Leskovec, NeurIPS 2020): adds ¬ with one free variable.
  - CQD (Arakelyan et al., ICLR 2021): t-norms over a pretrained link predictor. CQDᴬ (NeurIPS 2023) adds negation.
  - A 2025 "logic-constrained VSA" preprint (Lu et al., arXiv:2509.10837; construction *(unverified)*).
  - **What they share:** queries over a fixed graph; no stored rules; no ∀.

### 3.7 Differentiable logic and rule learning

- **The systems:**
  - Neural Theorem Provers (Rocktäschel & Riedel, NIPS 2017): soft unification of embeddings; proof paths explode.
  - ∂ILP (Evans & Grefenstette, *JAIR* 61, 2018): rules from templates; generalise to new constants; small domains.
  - Neural LP (Yang, Yang & Cohen, NIPS 2017) and DRUM (Sadeghian et al., NeurIPS 2019): chain rules applied
    inductively.
  - TensorLog (Cohen 2016; *JAIR* 67, 2020): Datalog as sparse matrix operations over one-hot entities.
  - Logic Tensor Networks (Badreddine et al., *AI* 303, 2022): fuzzy first-order constraints during training.
  - Logical Neural Networks (Riegel et al., arXiv:2006.13155): explicit, editable weighted logic.
- **Rule miners as baselines:**
  - AMIE (WWW 2013), AMIE+ (*VLDB J.* 2015) and AMIE 3 (ESWC 2020);
  - AnyBURL (Meilicke et al., IJCAI 2019).
- **Evaluation on unseen entities:** the inductive splits of GraIL (Teru, Denis & Hamilton, ICML 2020) and ILPC 2022
  (arXiv:2203.01520). Every test relation is seen in training.
- **What they share:** no language-model interface; no distributed or superposed store; never evaluated on **rules
  inserted after training**.

### 3.8 Neurosymbolic VSA systems that reason

- **NVSA** (Hersche et al., *Nature Machine Intelligence* 5:363–375, 2023): a neural front end, then VSA abduction and
  rule execution. 88.1% on I-RAVEN, with rules fixed in advance.
- **Learn-VRF** (NeurIPS 2023 MATH-AI workshop, arXiv:2401.16024) and **ARLC** (NeSy 2024, arXiv:2406.19121): VSA rules
  that are learned, or programmed and then fine-tuned.
  - They generalise out of distribution to unseen attribute–rule pairs and use fractional power encoding for continuous
    values.
  - ARLC beats large language models on I-RAVEN.
  - Only Raven's matrices; no language model.
- **HRR as a deep-learning layer:** Ganesan et al., NeurIPS 2021; Hrrformer (Alam et al., ICML 2023); GHRR (Yeung et
  al., arXiv:2405.09689), which adds **non-commutative** binding, the fix for our order problem.
- **Dhanraj & Eliasmith 2025** ("Improving rule-based reasoning in LLMs using neurosymbolic representations", EMNLP
  2025, arXiv:2502.01657). **The closest LLM + HRR work.**
  - A linear encoder maps Llama-3.1-8B hidden states at layer 17 to 2,048-dim HRR vectors.
  - The operation (3-digit arithmetic, bitwise operations) is solved in VSA space, then decoded and mixed back in.
  - **Differences from ours:** no relational rules, no variables over entities, no editable store, and held-out
    operations are gated off rather than solved.
- **Other recent work:**
  - PathHD (Liu et al., arXiv:2512.09369) encodes knowledge-graph paths as GHRR vectors so that an LLM can retrieve
    them: retrieval, not rules.
  - Hyperdimensional Probe (Bronzini et al., arXiv:2509.25045) decodes LLM states with a VSA probe.
  - "Attention as Binding" (arXiv:2512.14709) is a position paper.

### 3.9 Language models and rules: behaviour, solvers, benchmarks

- **Behavioural rule following with rules as text:**
  - **RuleTaker** (Clark, Tafjord & Richardson, IJCAI 2020, arXiv:2002.05867). RoBERTa-large trained to depth ≤ 3 scores
    97.6% at depth 5 on new templated theories, but it is fine-tuned for the task, keeps rules in the context window, and
    gives no guarantee.
  - ProofWriter (Findings of ACL 2021) adds proofs and an "unknown" label.
  - RuleBERT (EMNLP 2021) uses soft rules.
- **Failures:**
  - **SimpleLogic** (Zhang et al., IJCAI 2023): BERT about 57% at depth 6 under a shifted sampler; the statistical
    shortcut result.
  - **PrOntoQA / PrOntoQA-OOD** (Saparov & He, ICLR 2023; Saparov et al., NeurIPS 2023): models compose rules but fail on
    long proofs. The smallest model tested is 11B.
  - "Faith and Fate" (Dziri et al., NeurIPS 2023): errors compound with depth.
  - "Reasoning or Reciting?" (Wu et al., NAACL 2024): counterfactual rule variants.
  - RuLES (arXiv:2311.04235), RuleBench (arXiv:2407.08440, CCL 2025) and "Inductive or Deductive?"
    (arXiv:2408.00114): weak deduction, worst with counterfactual premises.
- **Systematicity:** Fodor & Pylyshyn 1988 (*Cognition* 28); Marcus 2001 (*The Algebraic Mind*); SCAN (Lake & Baroni,
  ICML 2018); COGS (Kim & Linzen, EMNLP 2020). Counterpoint: meta-learning reaches human-like systematicity (Lake &
  Baroni, *Nature* 623, 2023).
- **Language model plus solver:** these are the closest *architectures* to "the store reasons, the model reads".
  - Logic-LM (Pan et al., Findings of EMNLP 2023);
  - LINC (Olausson et al., EMNLP 2023);
  - SatLM (Ye et al., NeurIPS 2023);
  - Faithful CoT (Lyu et al., IJCNLP-AACL 2023).
  - All of them translate the problem into logic and call an exact solver.
- **Language model as reasoner over selected facts:**
  - Selection-Inference (Creswell, Shanahan & Higgins, ICLR 2023): a 7B model beats a 280B model.
  - LAMBADA backward chaining (Kazemi et al., ACL 2023).
- **Knowledge editing:**
  - ROME (arXiv:2202.05262) and MEMIT (ICLR 2023) edit facts in the weights.
  - **MQuAKE** (Zhong et al., EMNLP 2023) shows that edits do not propagate to the facts they entail. A rule store
    propagates edits by construction, which is a natural test (§7, L2e).
- **Benchmarks for our setting:**
  - PrOntoQA-OOD's generator (invented concepts, hop and width control, unseen-rule splits; Apache-2.0);
  - SimpleLogic's samplers (MIT);
  - CLUTRR (kinship chains k = 2–10 with held-out patterns; CC BY-NC; rename everything to invented words);
  - FLD/FLDx2 (depth 1–8, Apache-2.0).
  - ProofWriter and FOLIO are widely mirrored online: transfer checks only.
- **Real-rule sets:**
  - RuleArena (ACL 2025);
  - MedCalc-Bench (NeurIPS 2024 Datasets & Benchmarks; public PMC patients);
  - BoardgameQA (NeurIPS 2023 Datasets & Benchmarks).
  - None tests faithful execution of a *given* decision tree.

### 3.10 The gap

The searches found no work that:
1. **pairs a VSA rule store with a language model for relational deduction.** The closest are Dhanraj & Eliasmith
   (arithmetic in hidden states) and PathHD (path retrieval);
2. **tests rules inserted after training, on unseen entities, against the same rules given as text.** Inductive
   knowledge-graph work tests unseen entities with rules learned in training; RuleTaker tests rules as text with no store;
3. **tests rule transfer with correlated, learned fillers.** The survey flags this as a risk; all VSA rule-transfer
   results use random atoms or Raven's attributes;
4. **maps the depth × width × load frontier of multi-step VSA deduction on a real ontology.** Crawford et al.'s WordNet
   traversal is the nearest case, and it has no rules;
5. **stores and executes decision trees or decision lists in hypervectors;**
6. **shows edit propagation to entailed facts through a vector store**, where LM weight-editing is known to fail
   (MQuAKE).

**Closest competitors and what each lacks:**

| Competitor | Has | Lacks |
|---|---|---|
| Dhanraj & Eliasmith 2025 | HRR inside an LLM | relational rules, entity variables, editable store, zero-shot rules |
| ARLC / NVSA | programmable, editable VSA rules; out-of-distribution rule–attribute pairs | a language model; quantification over entities; anything but Raven's |
| Kvasnička & Pospíchal 2006 | modus ponens and tollens in HRR | unification inside the algebra; scale; a language model |
| TensorLog, Neural LP, DRUM, LNN, ∂ILP | explicit rules applied to new entities | a language model; distributed store; post-training rules |
| RuleTaker / ProofWriter | new rules as text, deep chaining | a persistent store; guarantees; small frozen models (it is fine-tuned) |
| Logic-LM / LINC | exact solver behind an LLM | a compact store; soft matching; entities without symbols |

---

## 4. What our own results already say

| Question | Result | Source |
|---|---|---|
| Can facts be read back from the store? | T4: 0.908 of decoded edges correct (C5); **0.960** with a fixed random operator (C5rf) | `manuscript/findings/2026-10-17-e12-t4-explicit-recall.md` |
| Does the model use bound roles *implicitly*? | **No.** Role-swapped twins at chance (C5 0.494; P1 null) | `experiments/e9-retrofit/report/t5-binding/report.md` |
| Does the model use decoded facts *explicitly*? | **Yes.** T4 role assignment: none 0.554 → decoded **0.882** (gold text 0.897; definition 0.915); +0.328 [+0.302, +0.355] | E12 finding record |
| Is binding necessary for that? | **Yes.** No-binding store 0.500; partners without role labels 0.479 | E12 finding record |
| Do zero-shot (composed) vectors work? | Held-out terms: decoded 0.925 against gold 0.929 | E12 finding record |
| Can any model read the store? | Yes: C0′ host 0.877; untouched P0 0.857 | E12 finding record |
| Multi-step lookup in the algebra | One store per entity: hop 1 0.942, chained two-hop 0.884; one shared memory 0.074 (local − global +0.81 [+0.76, +0.87]) | binding report, step 3 |
| Multi-step use by the model | Decoded two-hop chain: T5 0.19 → 0.80 (gold 0.84); T4 0.23 → 0.62 (gold 0.84) | `manuscript/update-2026-10-12.md` E12-4 |
| Reverse lookup | Algebra fine (reverse MRR 0.959), but decoded reverse text does not help the model (T4 0.55 → 0.48; gold 0.77) | binding report; E12-4 |
| Order along a path | Not encoded (0.49–0.50) | binding report, step 3 |
| Capacity of one vector | 256 dims: 0.98 of relations recovered at 8 per vector, 0.66 at 16, 0.24 at 32. On real T4, decoded lags gold at 13–24 edges (−0.037) | `update-2026-10-12.md` B-3; E12 finding record |
| Does the algebra need to be learned? | No: the fixed random operator decodes best (0.960) | E12 finding record |
| Can a model *call* the store as a tool? | Qwen3-1.7B (frozen, seed 1): well-formed calls 0.81; twins 0.85 (fixed pipeline 0.85; no tool 0.50); two-hop 0.61 (fixed 0.81) | `experiments/e12-self-query/agent/qwen3-1.7B-base-C5store-s1/report.md` |
| Can rules be *discovered* from the vectors? | Weakly. E10 tests structural hypotheses (symmetric, transitive, inverse-of, composition-of, sub-relation-of); few accepted slots adopt a rule (1 of 19 in the "absent" condition), and enforcing them adds nothing (0.000) | `experiments/e10-self-semantics/runs/e10.1-v1/report.md` |
| Can facts be read from real text? | Not yet: the E11 readers fail on real definitions; the 360M host reader produces 0 edges | `update-2026-10-12.md` E11; addendum 2 §7 |

**What this says about the idea:**
1. **Executing rules is the tractable part; discovering them is not.** Every ingredient for execution is measured and
   works: decode, clean-up, chaining with one store per entity, fixed operators, zero-shot vectors. Discovering rules from
   vectors (E10) and reading facts from text (E11) are weak. So the proposal should take rules as *given* (written by
   people, mined symbolically, or induced from a few labelled examples) and test how they are executed and used.
2. **The model reads; it does not unbind.** Any design that relies on the model doing algebra inside its layers starts
   from a null result (P1). Designs that hand it decoded conclusions start from a strong one (E12).
3. **Capacity and storage layout decide depth.** One store per entity plus clean-up per step chains well; one shared
   memory does not. Rules and derived facts must go into their own memories, not into the term vectors, which are
   already near capacity on T4.
4. **Reverse and order are weak spots.** Rules that need reverse lookup ("who has X as owner?") or ordered composition
   need either inverse relations stored explicitly (which a mapping vector can derive) or a non-commutative binding.
5. **The prompt format matters.** Decoded reverse lookups did not help even though the algebra got them right, so how
   conclusions are written into the prompt has to be piloted.

---

## 5. Is it worth exploring?

**Verdict: yes, as a gated side line that extends E12. It should not delay the current manuscript.** In order of weight:

**Reasons for:**
1. **The gap is real.** The research passes found no work that combines a VSA/HRR rule store with a language model for
   first-order deduction, and none that tests rules *inserted after training* on unseen entities.
   - The closest, Dhanraj & Eliasmith (EMNLP 2025), puts HRR vectors inside Llama's hidden states for arithmetic and
     bitwise operations. It has no relational rules or editable store, and held-out operations are gated off.
   - ARLC and NVSA have programmable VSA rules, but only for Raven's matrices and with no language model.
   - Inductive rule learners (Neural LP, DRUM, GraIL) handle unseen entities but have no language model and no
     post-training rule insertion.
2. **It builds on our strongest result.** E12 shows that a model uses decoded store contents, at 3 seeds, on real
   chemistry text, for any host. Adding rules changes *what is decoded* (derived facts instead of stored facts), not the
   pathway.
3. **It gives the VSA framing a job that a lookup table cannot do.** At present "VSA" earns its place for explicit,
   role-specific readout (E12). Rules applied algebraically to entities that exist only as composed vectors are a
   further, distinctly algebraic use.
4. **Clean zero-shot tests are possible.** Invented vocabulary (as in T5 and PrOntoQA) removes world knowledge; held-out
   entities and held-out rules can be separated exactly; and the ground truth is computable.
5. **It is cheap to start.**
   - L0 is CPU-only and takes minutes of compute.
   - L1 is scoring only, on existing models, in GPU-hours.
   - Neither needs training.

**Reasons for caution:**
1. **An exact symbolic engine wins on clean symbols.** A 50-line Datalog evaluator over the same ontology gives exact
   answers. If the benchmark has clean symbolic entities and stated rules, the VSA is a noisy Prolog. The VSA must show
   an advantage where symbols are missing or unreliable:
   - entities that exist only as composed vectors (held-out terms);
   - aliases and paraphrased mentions resolved by similarity;
   - facts from a noisy reader;
   - rules induced from 3–10 examples as mapping vectors;
   - a single fixed-size store shared with the model.
2. **"The model reasons" would be an overclaim.** In the realistic design the executor reasons and the model reads. That
   is a legitimate and useful architecture (it is what Logic-LM and LINC do with external solvers), but the paper must
   say so. The ambitious version (L4: the model learns to run the steps) has no evidence yet.
3. **Small models may already do shallow deduction from text.** The fair main baseline is "the same rules and facts as
   text in the prompt". If small hosts solve depth 1–2 from text, the gain lives only at depth 3 and above, which is
   exactly where we want to look.
4. **Capacity is binding at our widths.** At 256 dims, frames hold about 8 facts cleanly. Derived facts cannot simply be
   bundled into term vectors. They need their own memory (an engineering cost, not a blocker).
5. **The ingredients are known; only the combination is new.** Rule mapping vectors, VSA rule execution and LM-plus-solver
   systems all exist. A reviewer will ask what is new beyond the combination. The answer has to be the Z2 evidence (rules
   added after training, applied to zero-shot composed entities, read by an unchanged model) and the soft-symbol
   advantage (L2).

**What would make it a contribution:**
> "Rules written into a fixed-size VSA store after training apply exactly to entities that exist only as ontology-composed
> vectors. A small language model that cannot follow the same rules given as text at depth ≥ 3 answers correctly when the
> executor's conclusions are decoded into its prompt. Against an exact symbolic engine, the store is equal on clean
> symbols and better when entities or facts are known only approximately."

**What would end it:**
- L0 cannot execute depth-3 chains at ≥ 0.95 at a practical width (1,024–2,048) under our real loads;
- or L1 shows that rules as text are as good as executor output for the small hosts;
- or L2 shows that the symbolic engine with a fuzzy-matching front end matches the store in every soft setting.

---

## 6. A first check: does the algebra execute rules? (illustration, not a result)

**What was run:** `vsa_logic_demo.py`, which uses the repository's own `HRRAlgebra`.
- Random vectors, 3 seeds, about 95 s on 4 CPU threads.
- No language model and no data.
- **Fillers** are Plate-style Gaussian vectors; **roles** are unitary.
- **Clean-up** is a nearest-neighbour search over all stored vectors of the right kind.

This checks the mechanics the proposal relies on. It is not an experiment of the program.

**A. One rule as one vector, applied to unseen entities (Z1, and Z2 from examples).**
- **Rule:** `parent(a, b) → child(b, a)`, applied to 500 test facts whose 1,000 entities were never used when forming the
  rule.
- **Scoring:** a test is correct only if all three slots decode correctly (predicate, both arguments). There are 2,000
  candidate entities and 20 predicates.
- **The mapping vector comes three ways:**
  - written by hand (the role swap);
  - estimated from k example pairs by averaging their correlations;
  - estimated by least squares per frequency.

| Width | By hand | From 1 example | From 3 | From 10 | From 30 | From 100 |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 0.960 | 0.03 / 0.00 | 0.47 / 0.50 | 0.85 / 0.90 | 0.91 / 0.94 | 0.95 / 0.95 |
| 1,024 | 1.000 | 0.18 / 0.04 | **1.00** / 1.00 | 1.00 / 1.00 | 1.00 | 1.00 |
| 4,096 | 1.000 | 0.32 / 0.65 | 1.00 / 1.00 | 1.00 | 1.00 | 1.00 |

(Cells with two values give averaged correlations / least squares.)

- **One example fails.** The estimated rule then still contains that example's entities, so it does not generalise.
- **From 3 examples (1,024 dims), the rule applies to unseen entities without error.**
- **The rule contains no entity.** That is the systematicity argument made concrete: the same vector works for any
  filler.
- At 256 dims, the cross-term noise caps accuracy at about 0.96 even with the rule written by hand.

**B. A transitive rule executed by chaining (Z3).**
- **The world:** 5,000 entities, each with a frame of `load` bundled relations out of 24 types; chains along
  `located_in`.
- **Each hop:** unbind `located_in` from the current frame, clean up to the nearest entity, fetch that entity's frame.
- **Scoring:** a chain counts only while every hop so far is right.

| Width, load | hop 1 | hop 2 | hop 3 | hop 4 | hop 6 |
|---|---:|---:|---:|---:|---:|
| 256, 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 256, 8 | 0.989 | 0.970 | 0.951 | 0.932 | 0.886 |
| 256, 16 | 0.612 | 0.394 | 0.266 | 0.171 | 0.066 |
| 256, 32 | 0.231 | 0.047 | 0.007 | 0.000 | 0.000 |
| 1,024, 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1,024, 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1,024, 32 | 0.976 | 0.941 | 0.912 | 0.889 | 0.843 |

- **Hop 1 reproduces the program's capacity sweep** at 256 dims: 0.99 / 0.61 / 0.23 here against 0.98 / 0.66 / 0.24.
- **Later hops multiply.** Accuracy at depth d is close to (hop-1 accuracy)^d, because clean-up resets the noise at
  every hop. Depth costs only the per-hop error, never a growing one.
- **Load is the binding constraint, not depth.**
  - At 256 dims and T4's average load (8 edges), six-hop chains are 0.89.
  - At T4's largest frames (13–24 edges), 256 dims fails.
  - At 1,024 dims, 16 edges per frame chain perfectly.

**C. A decision tree executed by unbinding.**
- **The trees:** random complete binary trees, each test "feature f equals value v", over 20 features with 8 values
  each, with 10 classes at the leaves.
- **Scale:** 10 trees per depth (4 at depths 6 and 8) × 200 inputs × 3 seeds.
- **Chance level:** 0.10.

| Width | Encoding | depth 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---:|---:|---:|---:|---:|---:|
| 256 | node memory (clean-up per node) | 0.991 | 0.981 | 0.969 | 0.963 | 0.950 | 0.935 |
| 256 | one nested vector | 0.991 | 0.870 | 0.411 | 0.279 | 0.117 | 0.087 |
| 1,024 | node memory | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| 1,024 | one nested vector | 1.000 | 0.898 | 0.803 | 0.486 | 0.092 | 0.130 |

- **The two storage layouts behave very differently.**
  - With node memory, the tree executes exactly to depth 8 at 1,024 dims.
  - A tree held in one nested vector degrades from depth 2 and reaches chance by depth 6.
- **This mirrors the binding report's chain result** (one store per entity 0.88, one shared memory 0.07). Rules and
  trees need their own addressable memory with clean-up. They cannot be squeezed into a single vector.
- **Nothing is trained.** Each tree is written at test time and run by the same executor.

**What this does and does not show.**
- **It shows** that the algebra executes rules, chains and trees exactly enough at 1,024 dims and realistic loads. It
  also shows that rules can be estimated from a handful of examples and that they transfer to unseen entities.
- **It does not show** any of the following:
  - that this works with *trained* ontology-composed vectors, which are correlated, unlike random ones. Experiment 00
    found that correlation 0.3 costs about 7 points of top-1. The VSA survey warns that learned transformations may fail
    when fillers are similar (§3.2);
  - that it works with heavy-tailed degrees;
  - that a language model benefits.

  These are L0's and L1's questions.

---

## 7. Methodology: a staged plan with gates

Every stage gets its own short pre-registration (a new E14 folder, or an E12 addendum) before any non-smoke run, with the
program's usual conventions (seeds, paired bootstraps, Holm within a family, fairness section). Each later stage runs only
if the earlier stage's gate passes.

### L0. Rule execution in the algebra (CPU; no language model)

**Question:** at what width, load and depth does the executor stay exact, and which rule encodings work?

**World generator** (invented names, T5-style, contamination-free):
- entities with typed attributes;
- binary relations with realistic degree distributions (T4-like: 8 edges on average, up to 24; heavy tails as in
  experiment 00);
- rule sets of these kinds:
  - inverse and symmetric;
  - sub-relation;
  - chains (`r1 ∘ r2 → r3`);
  - inheritance down `is_a`;
  - conjunctions of 2–3 premises;
  - one exception rule (negation as failure).
- decision trees and decision lists over the attributes, with categorical and numeric tests.

**Encodings compared:**
- rule mapping vectors, by hand and from k = 1–100 examples;
- explicit rule records with an executor;
- node-memory trees, nested-vector trees, path bundles.

**Grid:**
- width 256, 512, 1,024, 2,048;
- frame load 4–32;
- depth 1–8;
- number of rules 1–50;
- clean-up per step on or off;
- operators: HRR, unitary HRR, the fixed random operator (C5rf-like), permutation-based for order;
- **fillers:** random atoms, and the **trained, correlated** composed vectors of existing T4/T5 C5 runs. This is the
  untested case the survey warns about (§3.2).

**Splits:**
- entities disjoint between rule formation and test (Z1);
- rules never used in formation (Z2);
- depth beyond any example (Z3).

**Metrics:**
- derived-fact accuracy (exact, after clean-up) by depth;
- precision and recall of the saturated closure against the symbolic closure;
- calibration of the similarity margin, used to abstain when unsure (experiment 00's calibrators).

**Baselines:**
- an exact Datalog evaluator (the ceiling, and the reference for correctness);
- a knowledge-graph embedding (RotatE or ComplEx) trained on the same facts. It learns rule patterns statistically and is
  transductive, so it cannot handle held-out entities. That is the point of the comparison.

**Real-ontology check:**
- the same executor on T4 (ChEBI: `is_conjugate_acid_of` ↔ `is_conjugate_base_of` as an inverse pair;
  `is_enantiomer_of` and `is_tautomer_of` as symmetric relations; `is_a` as transitive);
- and on T7 MeSH (tree codes as transitive `is_a`);
- with the ontology's own closure as gold, including the held-out terms' composed vectors.

**Gate G0:**
- explicit execution ≥ 0.95 at depth 3 at some width ≤ 2,048 under the T4 load profile;
- and mapping vectors from ≤ 10 examples ≥ 0.95 on unseen entities.

**Cost:** about one day of code (generator, executor, report); minutes to hours of CPU.

### L1. A language model reads the executor's conclusions (scoring only; existing models)

**Items:** PrOntoQA-OOD-style deduction questions, regenerated with our own invented vocabulary (the generator is
Apache-2.0). Formats are true / false / unknown and a cloze for the derived filler.
- **Depth:** 0–6.
- **Subsets:** seen entities; held-out entities (zero-shot vectors); held-out rules (rules that appear only in the store,
  never in any text).
- **Controls:**
  - a CLUTRR-style kinship subset with invented relation names (rules with shared variables);
  - a SimpleLogic-style sampler control (the "rule-priority" and "label-priority" samplers), to check that a gain is not a
    statistical shortcut.

**Prompt conditions (what goes before the question):**

| Condition | Content | What it isolates |
|---|---|---|
| nothing | — | prior |
| facts as text | the relevant facts, no rules | does the answer need the rules? |
| **rules + facts as text** | the same rules and facts the executor uses, in plain sentences | the **main baseline**: the model must deduce |
| rules + facts + worked example | as above, plus one solved example (few-shot) | best fair prompting for small models |
| **executor conclusions, VSA** | the derived facts and a short proof trace, decoded from the store (E12 format) | ours |
| executor conclusions, symbolic | the same, from the exact Datalog engine | the cost of VSA noise |
| shuffled rules | conclusions from a store with permuted rules | shows the model follows the store, not its priors |
| no-binding store | conclusions from a C5ut-style store | binding is needed (as in E12) |

**Hosts:** P0 at 135M and 360M; C0′ and C5 (existing T5 runs); Qwen3-0.6B and 1.7B (frozen), as in the E12 agent pilot.

**Primary:**
- `executor-VSA − rules-as-text` by depth, Holm over depths 1–6;
- and `executor-VSA − executor-symbolic` (expected near 0 on clean items).

**Fairness:**
- every condition gets the same underlying information;
- token counts are reported per condition;
- the executor gets no fact the text condition lacks;
- invented names remove world knowledge;
- some rules are deliberately counterfactual, which tests rule-following rather than recall.

**Gate G1:**
- `executor-VSA − rules-as-text > 0` (Holm) at depth ≥ 3 for at least the 135M and 360M hosts;
- and `executor-VSA` within 0.03 of `executor-symbolic`.

**Cost:** 1–2 days of code (reusing `e12_self_query`'s decode and prompt format); GPU-hours of scoring. It fits after the
toolkit band, or between jobs at a priority the author chooses.

### L2. Where the store should beat symbols ("soft" settings)

**The test:** the same questions, in settings where an exact engine has trouble:
- **(a) zero-shot entities:**
  - T4 and T7-ROOD held-out terms known only through their composed vectors;
  - the exact engine gets the term's ontology frame, so this is fair;
  - the difference shows only where the frame is incomplete or noisy.
- **(b) aliases and paraphrased mentions,** resolved by similarity. The symbolic engine gets a fair fuzzy front end:
  string similarity plus the alias table.
- **(c) noisy facts:**
  - edges from the E11 readers or the teacher-read T7 ontology (addendum 2 §7b);
  - the executor's similarity margins against the engine's all-or-nothing matches.
- **(d) rules induced from 3–10 examples:**
  - as mapping vectors;
  - against symbolic induction (an AMIE-style miner, or E10's structural hypotheses) from the same examples;
  - against the same examples given to the model in context.
- **(e) edit propagation:**
  - edit one stored fact (as in the program's editing tests);
  - check whether the *entailed* conclusions change with it, the MQuAKE failure for weight editing (§3.9);
  - compared with editing the text, and with an edited fine-tuned model.
  - The store re-derives by construction; the question is whether the model's answers follow.

**Primary:** `executor-VSA − executor-symbolic(fuzzy)` in settings (a)–(d), Holm over the four. Setting (e) is a
secondary, compared against the text and fine-tuned conditions.

**Gate G2:** the store is better in at least one setting and not worse elsewhere. Only then is a manuscript claim of
"logic in the store" justified.

### L3. Decision trees and decision lists

**Items:**
- synthetic triage and policy trees over T5-style attributes, depth 2–6;
- categorical and numeric tests (fractional power encoding);
- each tree written at test time.

**Conditions:**
- nothing;
- the tree as if–else text;
- the decoded path and leaf from node memory;
- the leaf only;
- a shuffled tree.

**Edits:** change one node's threshold or leaf and check whether answers follow, as in the program's editing tests.

**Metrics:** execution accuracy by depth; answer accuracy; how faithfully answers follow edits.

**Note:** the research passes found no benchmark for executing a *given* decision tree. RuleArena and MedCalc-Bench are
the nearest real-rule sets; MedCalc-Bench uses public PMC patients, not MIMIC.

### L4 (later, only after G1). The model runs the steps itself

- **(a) Tool use, and full first-order logic with the model as the prover's search policy:** see §8 for the design. It
  replaces the earlier tool-use sketch.
- **(b) Internal execution:**
  - after the readout arms report (models trained to unbind), train a model to chain unbind and clean-up steps;
  - test whether it extrapolates to unseen rules and depths.
  - This is the only stage where "the model reasons" would be the claim, and the highest-risk one.

**The complete run-by-run plan** for L0–L4, §8 and §9 is the companion document
[`2026-10-17-vsa-logic-experiment-plan.md`](2026-10-17-vsa-logic-experiment-plan.md). It has a preliminary tier now and a
full-scale tier later.

### Summary of stages

| Stage | Compute | Training? | Gate | Manuscript use if it passes |
|---|---|---|---|---|
| L0 algebra | CPU, minutes–hours | no | G0 | methods / appendix: capacity and depth frontier for rules |
| L1 model reads conclusions | GPU-hours | no | G1 | a short section extending E12 ("from facts to rules") |
| L2 soft settings | GPU-hours | no | G2 | the "logic in the store" claim |
| L3 decision trees | CPU + GPU-hours | no | descriptive | an example application |
| L4 model executes | GPU-days | yes | — | follow-up paper |

---

## 8. Full first-order logic: the language model as the prover's search policy

*Added 2026-10-17 at the author's question: "For full first-order logic — can the prover be the LLM itself using some
kind of ReAct-style call or otherwise?"*

**Short answer:** yes, provided the model *chooses* the proof steps and something exact *performs and checks* them. The
model on its own should not be the prover.

### 8.1 Why the model alone cannot be the prover

- **Chain-of-thought proofs carry no validity guarantee.** One bad step makes the conclusion wrong, and the chance of a
  bad step grows with proof length. PrOntoQA and "Faith and Fate" (§3.9) both show accuracy falling with depth, and
  SimpleLogic shows models learning shortcuts instead of rules.
- **Our hosts are small.** In the E12 agent pilot, frozen Qwen3-1.7B:
  - made well-formed tool calls 0.81 of the time;
  - scored 0.61 on two-hop questions, against 0.81 when the same lookups were made for it by a fixed pipeline;
  - managed well-formed reverse-lookup calls only 0.43 of the time.

  A 10-step first-order proof run by such a model, unaided, would usually contain a bad step.

### 8.2 The division of labour

A first-order prover does three different kinds of work. They suit different components:

| Part | What it is | Who does it | Why |
|---|---|---|---|
| **Choosing** | which goal to expand; which rule or premise to try; which entity to use for an existential ("there is an x such that…"); when to split into cases; when to argue by contradiction | **the language model** | mistakes cost only search time, never correctness; guessing what is worth trying is where statistical learning is good |
| **Executing** | matching symbols and filling in variables (unification, including nested function terms); applying a rule; forming the conclusion; putting formulas into clause form (Skolemization) | **exact code and the algebra** | must be exact and must work the same for any argument |
| **Certifying** | checking that every step follows | **the tools themselves** | if the tools implement only sound inference rules, every state they reach is sound; nothing the model writes in free text is ever treated as a proven fact |

**Consequences:**
- **Answers stay correct however weak the model is.** A poor model fails to find a proof within budget, and the answer
  is reported as "unknown". It never reports a wrong "proved".
- **Search is unavoidable.** First-order logic has no procedure that always terminates: proof search may run forever, so
  some heuristic must decide what to try. That is the part a learned model can improve. Learned guidance has improved
  classical first-order provers and proof assistants. Examples *(from memory, not checked in this session's research
  passes; verify before citing)*:
  - ENIGMA and rlCoP for first-order provers;
  - GPT-f, HTPS and LeanDojo/ReProver, where a model proposes each step and Lean checks it;
  - DeepMath, for premise selection.
- **Closest language-model precedents among the verified works (§3.9):**
  - LAMBADA (backward chaining with modules built from language models);
  - Selection-Inference (alternating between selecting facts and inferring a step);
  - LINC and Logic-LM (translation into logic, then an exact solver).

### 8.3 ReAct-style design: an extension of the E12 agent harness

**Tools** (each implements one sound step; the harness keeps the proof tree):

| Tool | Does | Implemented by |
|---|---|---|
| `goal[formula]` | parses the question into clause form (negated goal for refutation) | symbolic (Skolemization, normal form) |
| `rules_for[goal]` | candidate clauses whose conclusion could match the goal | **VSA retrieval** (unbind the predicate role; similarity ranking), every candidate re-checked exactly |
| `facts[pattern]` | stored facts matching a pattern with variables | VSA store plus clean-up (E12 recall), exact check |
| `apply[clause, goal]` | unifies and returns the new subgoals with their variable bindings, or rejects the step | symbolic unifier (with the occurs check) |
| `cases[disjunction]` | opens one branch per disjunct; every branch must close | symbolic |
| `assume_not[goal]` | starts a proof by contradiction | symbolic |
| `witness[variable, entity]` | proposes a value for an existential variable | the model proposes; `apply` checks |
| `done[]` | stops when every branch is closed, or answers "unknown" when the budget runs out | symbolic |

**Design choices that matter for small hosts:**
- **The harness runs routine steps automatically.** Forward chaining over the Horn (simple if–then) part runs to
  completion before the model is asked anything. The model is consulted only at real choice points. That keeps calls
  per proof small: the pilot was capped at 3 calls, and small models slip on long call chains.
- **Grammar-constrained decoding** makes every call well-formed, which removes the pilot's 19% of format failures. The
  policy then only has to choose *which* well-formed call to make.
- **Choices can be scored instead of generated.** At each choice point, the harness lists up to k legal next steps and
  the model ranks them by likelihood. This is the forced-choice reading E12 already uses, and it suits 135M–360M models
  that cannot generate calls reliably.
- **Budgets.** A fixed maximum number of steps per question, reported. Running out answers "unknown", which is scored
  separately from "wrong".

### 8.4 What the VSA adds here, and what it does not

**Adds:**
- **The clause memory:** facts and rules stored compactly, readable by any host.
- **Fast candidate retrieval** (premise selection). With large ontologies this is the bottleneck of classical provers:
  thousands of clauses, of which a handful are relevant.
- **Soft matching:** for entities with no exact symbol (aliases, composed held-out terms), similar to the "soft
  unification" of Neural Theorem Provers. Every soft match is marked as such in the proof.

**Does not add:**
- **Exact unification of nested function terms** is cheap and exact in symbolic code. It should not be done in the
  algebra: §2.3's noise and commutativity problems would make it unreliable.

**So in the full first-order setting, the VSA's role narrows to memory, retrieval and soft matching.** The paper should
say so.

### 8.5 What "zero-shot" means here

- **A new rule is a new clause in the store.** The tools apply it soundly the moment it is written, with no training.
- **The model only has to generalise "what's worth trying next"** to a rule it has not seen. This is a much more
  forgiving skill than doing the logic. A bad choice wastes a step; it does not produce a wrong answer.

### 8.6 The honest limit, and when this is worth it

**On small, clean problems a classical prover after translation will beat any language-model loop.** LINC with Prover9
already does this on ProofWriter. The model-guided version is worth it only when one of the following holds:
1. the search space is large (a big ontology, many premises), so blind search is too slow;
2. the inputs are not cleanly symbolic, so soft matching and translation errors matter;
3. the scientific question is whether a small model can *learn* the search policy, with the tools acting as the teacher.
   - **Expert iteration:** proofs the tools accept become training data for the policy, with no human labels.
   - This links to the E13 learning cycle.

### 8.7 How to test it

The full run list is in the plan's work package LG-P. In brief:
- **Items:**
  - problems that need cases and contradiction, from FLD/FLDx2 (Apache-2.0, depth 1–8) or PrOntoQA-OOD's
    proof-by-cases and proof-by-contradiction splits;
  - regenerated with invented words;
  - plus large-ontology items where premise selection matters (T4/T7 closure questions with distractor clauses).
- **Conditions:**
  - chain-of-thought only;
  - the model with tools but no soundness checks (it may assert facts);
  - **the model with sound tools (ours)**;
  - a random policy with sound tools;
  - a fixed classical strategy with sound tools (breadth-first; set of support);
  - Prover9 after gold translation (the clean-symbol ceiling);
  - and the same with VSA retrieval replaced by exhaustive or keyword retrieval.
- **Measures, by depth:**
  - accuracy;
  - soundness (the share of "proved" answers that are entailed; 1.0 is expected with sound tools);
  - proofs found within budget;
  - steps per proof;
  - tokens per proof.
- **Hosts:**
  - frozen Qwen3-0.6B and 1.7B with worked examples;
  - SmolLM2-360M with scored choices;
  - then a 360M policy fine-tuned on generated proofs, tested on held-out rules, entities and depths.
- **Key result:** does the model's policy find proofs in fewer steps than random or fixed search, at equal budget, on
  rules it has never seen?

---

## 9. Reading facts from real text: why it fails, and what it means for self-learning

*Added 2026-10-17 at the author's request: "explore the failure mode here — what is the intuition on why it fails — do
we need a larger ontology? is the model not smart enough? Is the coverage of the ontology too weak and not broad enough?
Do we need to build up the ontology more before it picks up steam? This one will be critical for the self-learning
piece."*

**Sources:**
- the E11 report (`experiments/e11-read-to-learn/report/report.md`, 3 seeds);
- the E11 many-words report (`report-many/report.md`);
- the per-term frames of `runs/t4/SmolLM2-360M-full-C5-s1/e11-t4-heldout/frames.jsonl` and
  `runs/t5/SmolLM2-360M-full-C5-s1/e11-t5-new/summary.json` (seed 1; frames are deterministic given the run);
- addendum 2 §7 (the T7 readers).

**The per-relation breakdown in §9.2 was computed for this document** from those frames. It is not in a report, and
should be recomputed by the report code before it is cited.

### 9.1 What was tried

**E11 setup:**
- **Task:** a new word is learned by reading its definition once, with no weight change.
- **Reading:** a reader turns the definition into a frame (relation, partner pairs), and the frame is composed into the
  word's vector, which the model then sees in its input.
- **Readers compared:**
  - `linker`: a concept finder locates known concepts in the text; the model picks each one's relation by which choice
    makes the definition most likely;
  - `host`: few-shot extraction by the host's own weights;
  - `typeprior`: the most likely relation for the found concept's type;
  - `pattern`: regular expressions;
  - `stated`: gold edges whose partner is mentioned in the text;
  - `oracle`: the full gold frame;
  - `random`: controls.

**What happened:**
- **Synthetic text (T5): a small gain.** The linker improves property questions by +0.031 [+0.025, +0.038].
- **Real ChEBI definitions (T4): nothing.** The linker gives +0.006 (n.s.), and the pre-registered specificity test P2
  failed.
- **MeSH notes (T7):**
  - the 360M host reader produced 0 edges on 4 notes (degenerate completions);
  - a Claude teacher read 1 edge from 2 notes in the smoke;
  - only 49% of T7-ROOD terms have a usable note at all.

### 9.2 The failure, stage by stage

To be used, a fact must pass four stages. Each has its own failure:

| Stage | Question | T5 (synthetic definitions) | T4 (real ChEBI definitions) |
|---|---|---|---|
| **1. Stated** | does the text state the fact? | **all** gold partners are mentioned (by construction) | **15%** of gold partners are mentioned; **12%** of gold edges are stated (322 of 2,744 over 341 terms) |
| **2. Found** | does the finder link the mention to a known concept? | 1.00 of mentioned partners | (same as 1: the text is searched only for known names) |
| **3. Labelled** | is the right relation chosen for a found partner? | linker 0.80 (**0.37** where the partner could fill more than one relation); host 0.99; type prior 0.92 | linker **0.17**; host 0.83; type prior **0.86** |
| **4. Used** | does the model's answer improve when the frame is in its input? | gold frame +0.070; **type-prior frame +0.069** (the same); linker +0.031 | **gold frame +0.002 [−0.004, +0.009], n.s.**: even perfect facts do not help |

**For comparison, the definition itself in the prompt:**
- T5: +0.42 over the linker frame;
- T4: +0.11 over the linker frame.

The model clearly *can* use the information when it is in its context.

**Which facts real definitions state** (T4, 341 held-out terms):

| Relation | Gold edges | Stated in the definition | Linker recovers | Host recovers | Type prior recovers |
|---|---:|---:|---:|---:|---:|
| `has_role` | 719 | 5.0% | 0.8% | 0.0% | 4.9% |
| `is_a` | 650 | 35.1% | 5.7% | 33.2% | 32.2% |
| `contains_element` | 598 | **0%** | 0% | 0% | 0% |
| `branch` | 341 | **0%** | 0% | 0% | 0% |
| `charge` | 216 | **0%** | 0% | 0% | 0% |
| `has_functional_parent` | 52 | 51.9% | 1.9% | 0% | 34.6% |
| `is_tautomer_of` | 18 | 44.4% | 0% | 11.1% | 5.6% |
| conjugate acid / base | 98 | 10.2% | 2.0% | 1.0% | 1.0% |
| all 13 types | 2,744 | **11.7%** | **1.7%** | **8.0%** | **9.7%** |

**Two examples** (ChEBI release 255; CC BY 4.0):
- **(4R)-limonene.** Definition: "*An optically active form of limonene having (4R)-configuration.*"
  - The gold frame has 6 edges: is a limonene; enantiomer of (4S)-limonene; a role; contains carbon; neutral; a root
    branch.
  - The text names one partner (limonene).
  - The enantiomer is implied by "optically active form… (4R)" but not named. The role, the element, the charge and the
    branch are absent.
- **(R)-lipoic acid.** Definition: "*The (R)-enantiomer of lipoic acid. A vitamin-like, C8 thia fatty acid with
  anti-oxidant properties.*"
  - Several gold facts are present, but **paraphrased**: "anti-oxidant properties" is the role *antioxidant*; "C8"
    means it contains carbon; "thia" means it contains sulfur.
  - A string-matching finder misses all three. A reader that understands chemistry prose would not.

### 9.3 The intuition: four separate reasons, not one

1. **The text and the ontology describe different things.**
   - A ChEBI definition says what makes a compound *distinct* (its configuration, its parent, its class).
   - The ontology records *everything*: elements, charge, every role, every class.
   - 42% of the T4 gold frame (`contains_element`, `charge`, `branch`) comes from the chemical **structure**, and no
     definition states it. Roles are rarely stated (5%).
   - **So a perfect reader of the definition could recover only a minority of the gold frame,** and E11's items test
     gold-frame facts. Much of the "failure" is a mismatch between what is read and what is graded.
   - **Implication:** facts of this kind need a structured reader (formula → elements and charge; the MeSH tree → its
     parents), not a language reader.
2. **The linker uses a signal the model does not provide.**
   - It chooses a relation by asking which frame makes the definition most likely under the model. That only works if
     the model is sensitive to *which role* a partner is bound to. P1 showed it is not: role-swapped twins score at
     chance (0.494).
   - So the linker's choices follow the partner's *type*, not the text:
     - on T5, 0.80 overall but 0.37 where the type does not decide the relation;
     - on T4, 0.17, far below the trivial type prior (0.86).
   - **This is a method failure that a bigger model would not fix**, because it rests on implicit role use, which is
     null at every size tested.
3. **Even perfect facts are not used through the implicit route.**
   - On T4 the gold frame helps no more than no frame (+0.002, n.s., at every vocabulary size from 25 to 242 words in
     the many-words report).
   - On T5 the gold frame helps exactly as much as a frame holding only type information (+0.070 against +0.069).
   - **The injected vector acts as a type signal, not as a carrier of specific facts.** That is again the P1 result.
   - On T4 the word's own name often already carries its type (chemical names are compositional: "(R)-lipoic acid"
     contains "lipoic acid"), so the type signal adds little there. This last point is a hypothesis to test (RD-4 in
     the plan).
   - **Explicit recall works** (E12: 0.55 → 0.88 on T4). But nobody has yet tried *reading* a definition into the store
     and then using it by explicit recall.
4. **The small model cannot extract facts by generating them.**
   - The 360M host reader works on templated synthetic text (T5 precision 0.99).
   - It recovers stated `is_a` facts on T4 almost perfectly (33.2% of `is_a` edges, against 35.1% stated).
   - It produces nothing on open MeSH prose (0 edges from 4 notes).
   - For open-ended generative extraction, **the 360M model is not capable enough.** A Claude teacher is. Qwen3-1.7B is
     untested.

### 9.4 The author's questions, answered

- **Do we need a larger ontology?**
  - **Not a denser one** (more edges per entry). That widens the gap between what the text says and what the gold frame
    contains. T4 already has 8.1 edges per entry, and only 12% are in the text.
  - **Probably a broader dictionary.** That means more concept names, synonyms and the vocabulary definitions actually
    use: functional groups ("chloro group"), properties ("anti-oxidant"), formula shorthand ("C8", "thia").
  - This raises how much of the text can be linked, and it is measurable (RD-2).
- **Is the model not smart enough?**
  - **For generative extraction at 360M: yes.**
  - **For the linker: no.** Its method depends on implicit role use, which fails at every size tested.
  - **For using the facts: no.** The implicit pathway fails even with gold facts, while explicit recall works at 135M
    (E12: 0.894).
  - So "smarter model" fixes only one of the four stages.
- **Is the ontology's coverage too weak or too narrow?**
  - **In one direction it is the reverse:** the ontology is far richer than the text, and the text covers about 12% of
    it.
  - **The other direction is unmeasured:** how much of what the text says can the ontology's 13 relation types and its
    dictionary express? Structural descriptions ("in which one hydrogen is replaced by…") have no relation type. This
    is the first thing to measure (RD-1).
- **Do we need to build the ontology up before it picks up steam?**
  - **Plausible, but unproven, and it has a precondition.**
  - **The mechanism:** the finder can link only partners it already knows. Each word read adds a new concept, which
    makes later definitions more linkable, so recall could grow over rounds.
  - **But wrong edges also compound.** A snowball needs reader precision high enough that errors do not poison later
    reading. At the linker's T4 precision (0.06), it would snowball errors.
  - **Testable directly:** a bootstrap curve of recall and precision over rounds of reading, as a function of starting
    dictionary size and reader precision (RD-5).

### 9.5 What to do, in order of expected value

1. **Change the route by which read facts are used: from implicit injection to explicit recall.**
   - Read → store → decode into the prompt (E12), instead of composing the read frame into the input vector.
   - **Test it first with the gold and stated frames** on E11's held-out T4 and T5 items (scoring only, existing runs).
     If explicit recall of *gold* frames helps where implicit injection did not, the route is fixed independently of
     any reader.
   - This is the most important single experiment for self-learning (RD-4).
2. **Grade readers on what the text states,** and split the frame into text facts and structure facts.
   - Language readers are scored on stated edges.
   - Structure facts (elements, charge, tree position) come from structured parsers. For ChEBI, the formula and
     structure are in the release.
3. **Replace the linker's relation choice, and read ReAct-style.** Candidates:
   - the type prior (0.86);
   - explicit multiple-choice questions to the model ("in this definition, how is X related to Y?"), scored E12-style;
   - a larger host (Qwen3-1.7B);
   - the Claude teacher as an upper bound.

   All are scored on stated edges, including ambiguous ones (RD-3).

   **The ReAct-style reader (author's question, 17 Oct).** It answers yes to "do we need the model to match the best
   relations and connections via ReAct-style framing?":
   - Both fixable reading failures (2: finding paraphrased partners; 3: choosing the relation) are *matching*
     problems.
   - A tool loop makes each match an explicit choice:
     - `find[span]`: candidate concepts from the dictionary, aliases and fuzzy matching;
     - `relations_for[x, y]`: the relations the ontology allows for the pair's types;
     - `propose[x, r, y, evidence]`;
     - `check[edge]`: typing, inverse and cardinality rules from §2;
     - `commit[]`.
   - The model only chooses among listed candidates. That is explicit reading, which works (E12), rather than implicit
     role sensitivity, which does not (P1).
   - For 135M–360M models the choices are scored, not generated.
   - The `check` step is the reading analogue of §8's sound tools: an edge that breaks the ontology's rules is never
     committed. This is what keeps errors from compounding in the bootstrap (RD-5).
4. **Broaden the dictionary** with ChEBI synonyms, functional groups and formula shorthand, and measure the gain in
   linked mentions (RD-2).
5. **Measure the bootstrap curve, with a precision gate** (RD-5).
6. **Flag for E13 (the self-learning cycle).**
   - E13's write stage uses `store:linker` and the implicit route, so on T7-ROOD's real notes it will inherit failures 2
     and 3.
   - E13 is pre-registered and queued (T5 at 54.4985–54.4989, T7-ROOD at 54.4996–54.49979), so any change needs an
     amendment and the author's decision.
   - **Proposed amendment** (not made): add `store:typeprior`, `store:teacher` and an explicit-recall use condition to
     the T7-ROOD write and reason stages, before those jobs run.

---

## 10. Risks and how to frame this honestly

- **Overclaiming reasoning.** Say "the store executes rules; the model uses the conclusions" unless L4 shows otherwise.
  The result is a neurosymbolic system, not a network that learned logic.
- **The symbolic baseline.** Always report the exact engine next to the store. If the store only matches it, the claim
  is a compact, model-readable store with zero-shot entities, not better reasoning.
- **Capacity and depth.** Report the depth × width × load frontier (L0) and abstain below calibrated margins, rather
  than choosing settings where the store happens to work.
- **Prompt length.** Executor conclusions are short, but a proof trace grows with depth. Report tokens per condition.
  The "rules as text" baseline grows with the size of the rule base, which is one of the store's arguments.
- **Benchmark contamination.** Use regenerated PrOntoQA-OOD and CLUTRR-style items with invented vocabulary. Use
  ProofWriter and FOLIO, which are widely mirrored online, only as transfer checks.
- **Credentialed data.** No MIMIC text is used. A clinical decision-tree demonstration would use public rule sources
  (MedCalc-Bench rules and PMC cases) or synthetic cases. Decision 58 applies.

---

## 11. Recommended next steps

0. **Reading (§9), independent of the logic line, and the most urgent for self-learning:**
   - run RD-4 (explicit recall of gold, stated and read frames on E11's items; scoring only);
   - run RD-1 and RD-3 (coverage audit; relation-choice methods on stated edges);
   - decide on the E13 amendment before E13's T7-ROOD jobs reach the front of the queue.
1. **Author decision:** pursue L0 now (CPU, no queue impact), or park the whole line until the manuscript's current
   experiments are written up.
2. **If L0:**
   - write a short pre-registration (E14 or an E12 addendum: grid, gate G0, baselines);
   - then build the generator, the executor and the Datalog reference;
   - run on CPU.
   - The illustration script in this folder is the starting point.
3. **If G0 passes:** pre-register L1 and queue its scoring after the toolkit band. The PrOntoQA-OOD generator and a CLUTRR-style
   generator with invented names supply the items.
4. **Manuscript:** nothing changes in the outline or abstract now. If G1 passes, a short "from facts to rules"
   subsection goes after E12, and the discussion's future-work paragraph mentions decision procedures. A full treatment
   (L2–L4) is better as a follow-up paper.

---

## Sources

- **Our results:**
  - `manuscript/findings/2026-10-17-e12-t4-explicit-recall.md`;
  - `experiments/e9-retrofit/report/t5-binding/report.md` (step 3);
  - `manuscript/update-2026-10-12.md` (B-3, E12-4, E11);
  - `experiments/e12-self-query/agent/qwen3-1.7B-base-C5store-s1/report.md`;
  - `experiments/e10-self-semantics/runs/e10.1-v1/report.md`;
  - `experiments/00-capacity-and-algebra/README.md`;
  - `experiments/e9-retrofit/preregistration-addendum-2026-10-15.md`.
- **Illustration:** `manuscript/proposals/vsa_logic_demo.py` → `vsa_logic_demo.json` (3 seeds, random vectors, CPU).
- **Literature:** §3 (citations inline; verification markers explained at the top of §3).
