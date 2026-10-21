# 15 — E14, part 2: logic in the vector store

**Overview.** This file asks whether predicate logic can be encoded directly in vector-symbolic form, and whether that
helps a language model. Predicate logic here means relations, rules, deductions and decision trees.
- **15.1–15.3:** what the algebra can execute, and how to store it.
- **15.4:** whether the store beats ordinary symbolic logic anywhere.
- **15.5–15.6:** whether a frozen language model can use an executor's conclusions.
- **15.7:** whether a language model can steer a sound prover.

**Terms used throughout.**
- **VSA:** vector-symbolic architecture.
- **HRR:** holographic reduced representations. Facts are stored by *binding* (circular convolution) a role vector to a
  filler vector and *superposing* (adding) the results.
- **Unbinding** with **clean-up** (nearest match among known vectors) reads a filler back.
- **Executor:** code that runs logic over the store (chains, joins, trees) and writes its conclusion as text.
- **exec-vsa / exec-sym:** the executor over the VSA store, or over an exact symbolic store.
- **Hosts:** frozen base models SmolLM2-135M and -360M, and Qwen3-0.6B and -1.7B.
- **Statistics:** accuracy differences (positive = better), paired bootstrap, \* = Holm-significant within the
  pre-registered family.

Pre-registration: `experiments/e14-logic/preregistration.md` §§3–6, 10, 14–16.

---

# 15.1 — What the algebra executes (LG-A) and gate G0

**Status:** done (CPU) · **When:** 2026-10-17 · **Sources:** `manuscript/findings/2026-10-17-e14-tier-p-cpu.md` Part 2; `experiments/e14-logic/runs/lga/`

## Research gap
Proposals to "do logic in vectors" rarely state where the algebra breaks. We needed capacity limits, rule types that
work, and failure modes, before claiming anything about language models.

## Intuition
Binding preserves structure, so a rule should be a vector that maps facts to conclusions. A chain should be a sequence
of unbindings with clean-up at each hop.

## Methodology
Synthetic worlds of random vectors at widths 256–2,048, with 3 seeds. The grids:
- **A1:** rules as single vectors;
- **A2:** chains;
- **A3:** joins (the Datalog fragment);
- **A4:** nesting and order;
- **A5:** connectives;
- **A6:** numbers (fractional power encoding);
- **A7:** decision trees and lists;
- **A8:** a resonator for factorisation;
- **A11:** the trained T4 store's own rules;
- **A12:** rule mining and knowledge-graph baselines.

**Gate G0:** depth-3 chains ≥ 0.95 at load 8, and rules from ≤ 10 examples ≥ 0.95 on unseen entities, both at width
≤ 2,048.

## Experiment
CPU only, about a day of compute.

## Results
- **G0 passes.**
  - Depth-3 chains score 1.00 from width 512.
  - An inverse rule learned from three examples scores 1.00 on unseen entities at width 1,024.
- **Exact at width ≥ 1,024:** joins (closure F1 1.00), negation as failure, numbers and decision lists. HRR, MAP and
  FHRR perform the same.
- **Hard limits:**
  - one global bundle of all facts scores 0.00, so storage must be per entity;
  - a single convolution cannot express rules that keep part of a fact fixed (symmetry, sub-relation): ≤ 0.16 at any
    width;
  - a rule applied to a whole frame fails (≤ 0.02).
- **Trained, correlated fillers cost 0.21** at width 256.
- **The real trained T4 store chains held-out terms** through two-hop `is_a` at precision and recall 0.94.

## Limitations
- Synthetic worlds; trained fillers only at width 256.
- Our resonator is weaker than published ones (A8 is not citable yet).
- The baselines in A12 are told the rules, so A12 is not a learning contest.

## Next steps
Check the resonator against the literature. Give derived facts their own memory: only 6–16% of converse facts exist in
T4 frames.

---

# 15.2 — Concatenation instead of superposition (X1–X7), and storing trees by their shape (X8)

**Status:** done (CPU) · **When:** 2026-10-17 · **Sources:** `manuscript/findings/2026-10-17-e14-concat-trees-reading-cycle.md` §§1–2

## Research gap
Superposition mixes every fact in a vector, so noise grows with load and some rules are impossible (15.1). The author
proposed the HRRCat idea from the HRRBERT work: concatenate blocks instead of adding them. The author also proposed
laying a tree out by its structure.

## Intuition
If each role, fact or tree node has its own block, rules become block permutations. Nothing is superposed except a
node's own content.

## Methodology
- **Layouts compared at equal total width:**
  - one wide HRR;
  - redundant blocks;
  - one block per role;
  - one block per fact;
  - fact × role blocks;
  - per-relation slots.
- **X7:** the three trained T4 stores combined by averaging cosines.
- **X8, decision trees:**
  - one nested vector;
  - node memory;
  - heap concatenation (node i in block i);
  - level concatenation.

## Experiment
Widths 1,024–16,384, 3 seeds. X8 used depths 2–12 with the input record fixed at 4,096 dimensions.

## Results
- **Role blocks make the impossible rules exact:** symmetry and sub-relation score 1.00.
- **Fact × role blocks make whole-frame rules exact** (1.00 at every width).
- **Redundant blocks equal one wide vector:** width is width.
- **X7, three trained stores combined:** decoding reaches 0.979, against 0.89–0.92 for one store.
- **X8, heap concatenation is exact (1.00) to depth 12,** even with 64-dimensional blocks. A single nested vector fails
  at any width (0.11 at depth 8).

## Limitations
- X7's ensemble has three times the parameters of one store; an equal-size store was not trained.
- Heap concatenation trades memory for exactness (0.5–2M dimensions at depth 12).
- Irregular trees are untested.

## Next steps
Proposed: an equal-size trained store for X7, and irregular trees with a pre-order layout.

---

# 15.3 — Does the store beat symbolic logic anywhere? (LG-C, CPU) — gate G2, part 1

**Status:** done · **When:** 2026-10-17 · **Sources:** `manuscript/findings/2026-10-17-e14-tier-p-cpu.md` (LG-C)

## Research gap
The strongest claim for vector logic is that it handles "soft" cases better than symbols. Those cases are similarity,
aliases, noisy facts, and rules from a few examples.

## Intuition
Graded similarity should let the store complete patterns, match misspelled names and tolerate wrong facts where exact
symbols break.

## Methodology
Four settings, each against a fair symbolic baseline:

| Setting | Symbolic baseline |
|---|---|
| C1, completing a mostly-true type property | rule induction |
| C2, aliases with 1–2 edits or abbreviations | prefix plus edit distance |
| C3, 10% / 30% wrong facts | exact Datalog |
| C4, rules from k = 1 or k ≥ 3 examples | AMIE-style mining |

**Gate G2:** the store wins somewhere and loses nowhere, and edits propagate.

## Experiment
Synthetic worlds, 3 seeds.

## Results
**No advantage anywhere:**

| Setting | Store | Baseline |
|---|---|---|
| C1 | 0.74–0.79 | 0.75–0.81 |
| C2 | 0.97–1.00 | 0.996–1.00 |
| C3 | identical | identical |
| C4 | 1.00 | 1.00 |

**G2 fails on its first condition.**

## Limitations
- Clean synthetic data favour symbols; real noisy inputs are untested.
- The mapping-vector chain rule does not check the shared middle entity, which is a latent unsoundness.

## Next steps
The manuscript should not claim "soft reasoning better than symbols". The supported claim is an exact executor with a
model-readable interface (15.5–15.6).

---

# 15.4 — Does a frozen model read the executor's conclusions? (LG-B, LG-C) — gate G1

**Status:** done · **When:** 2026-10-19 · **Sources:** `experiments/e14-logic/report/lg/`; `manuscript/findings/2026-10-20-e14-tier-p-gpu.md` §3; diagnostics `experiments/e14-logic/report/diagnostics-2026-10-20.md`

## Research gap
If an executor does the logic, can a small model relay its answer where it cannot reason from the rules itself?

## Intuition
Copying a stated conclusion is easier than multi-hop reasoning. Even small models should profit.

## Methodology
- **Worlds of invented words** (no prior knowledge):
  - 1,200 true/false taxonomy questions at depth 0–5;
  - propositional items;
  - composition chains;
  - rule bases of 10 and 50 rules;
  - few-example rules;
  - edits.
- **Conditions:** rules as text, keyword retrieval, `exec-vsa`, `exec-sym`, and controls (no binding, shuffled,
  stale).
- **Primary:** `exec-vsa − text` by depth, Holm over depths.
- **Gate G1:** both SmolLM2 hosts significantly positive at depth ≥ 3, with |exec-vsa − exec-sym| ≤ 0.03.

## Experiment
Four frozen hosts; forced choice by length-normalised log-probability.

## Results
- **Qwen3 hosts relay the executor decisively.**
  - +0.42 to +0.47\* at every depth.
  - Rules as text decay with depth (Qwen3-1.7B: 1.00 → 0.57); the executor holds 1.00.
  - Edits propagate: +0.62 and +0.92 against a stale store.
  - The controls show the content is read: no binding −0.50, shuffled −0.35.
- **G1 fails on the SmolLM2 hosts.** They answer "True" to "X is not a Y": 0.15 on false items. They score 1.00 when
  asked to fill in the blank.

## Limitations
- Base models only; synthetic worlds.
- `exec-vsa − exec-sym` for SmolLM2 (+0.03 to +0.05) reflects the confidence numbers in the text, not the store.

## Next steps
The negation hypothesis was tested next (15.5).

---

# 15.5 — Is the small-model failure only negation wording? (LG-B verdict format) — gate G1′

**Status:** done · **When:** 2026-10-20 · **Sources:** `experiments/e14-logic/report/lg-verdict/`; `manuscript/findings/2026-10-20-e14-followups.md` §2

## Research gap
15.4 suggested small models misread negated conclusions. Rewording the conclusion as a verdict should fix that, if
wording is the cause.

## Intuition
"The claim 'X is a Y' is False" states the answer token directly.

## Methodology
- The same 1,200 + 600 items, with byte-identical contexts except the conclusion line, rewritten as a verdict.
- **Gate G1′:** as G1, with the verdict format.
- The pre-registered alternative: small hosts cannot relay a true/false verdict at all.

## Experiment
Four frozen hosts, about 0.3 GPU-h in total.

## Results
- **G1′ fails; the alternative holds.** The verdict format only moves the SmolLM2 bias. Given the verdict line alone,
  SmolLM2-360M answers "False" 99.7% of the time, even when the verdict reads "True".
- **The Qwen3 hosts are unchanged** (≥ 0.94 on both labels).

## Limitations
Only one alternative format was tried. Instruction-tuned small models might behave differently.

## Next steps
Closed. For hosts around 360M, the executor interface must be fill-in-the-blank or answer-copy, not true/false.

---

# 15.6 — Decision trees read by a model (LG-D)

**Status:** done · **When:** 2026-10-19 · **Sources:** `experiments/e14-logic/report/lg/` (LG-D)

## Research gap
Can models use a tree executor's result (15.2), where they cannot execute a written-out tree?

## Intuition
Following a written tree requires several conditional steps. Reading the executor's path does not.

## Methodology
- **Items:** 600 trees (depth 2–6), 300 decision lists (length 3–12) and 200 edits, over invented attributes.
- **Conditions:** tree as text, tree as bullets, `path-vsa` (the executor's path and leaf), plus symbolic, leaf-only,
  shuffled and stale controls.
- **Primary:** `path-vsa − tree-text`, Holm over depths.

## Experiment
Four frozen hosts.

## Results
- `path-vsa` scores 0.997 on every host, against 0.37 for the tree as text. The gain grows with depth (+0.51 → +0.75\*).
- Edits are followed perfectly (+1.00 against stale).
- The leaf alone suffices.

## Limitations
The executor's output essentially *is* the answer, so this is an interface test, not a test of reasoning. Inputs were
clean.

## Next steps
Proposed: noisy or partial inputs.

---

# 15.7 — The language model as the prover's search policy (LG-P)

**Status:** done (tier P, hard problems, hybrid policy) · **When:** 2026-10-19 to 2026-10-20 · **Sources:** `experiments/e14-logic/report/{lgp, lgp-hard, lgp-sos-score}/`; `manuscript/findings/2026-10-20-e14-followups.md` §3; `manuscript/findings/2026-10-20-e14-evening.md` §4

## Research gap
The author asked whether the prover for full first-order logic could be the language model itself, in a ReAct-style
loop. A model alone produces unsound proofs. Sound tools with a model choosing the steps might combine soundness and
guidance.

## Intuition
The model sees all the clauses and can guess which step leads to a known fact. The tools guarantee that any proof found
is valid.

## Methodology
- **Problems:** one-entity first-order problems (Horn chains, proof by cases, proof by contradiction).
- **Tools:** sound backward, case-split and contradiction steps.
- **Policies:**
  - the model scores the candidate steps (`score`);
  - or writes the step number (`react`);
  - non-model searchers: random, breadth-first, set of support;
  - model without tools: `unsound` and `cot`.
- **Primary:** `score − random`, proofs found within the budget.
- **The hard-problem version** adds dead-end decoy chains (depth 4 × 3 per class). The difficulty was chosen on the
  non-model policies only.

## Experiment
- **Tier P:** 800 problems.
- **Hard problems:** 156 provable problems at budgets 10, 30 and 100. Budgets were derived exactly from one depth-first
  run.

## Results
- **Soundness comes from the tools:** 1.00 with them, about 0.51–0.57 without.
- **Tier P problems were too easy:** random search found 92% within 30 steps.
- **On hard problems, Qwen3-1.7B guides the search:** +0.11\*, +0.15\* and +0.13\* over random at budgets 10, 30 and
  100.

  | Proofs found within budget | 10 | 30 | 100 |
  |---|---:|---:|---:|
  | random | 0.115 | 0.513 | 0.750 |
  | set of support | 0.199 | 0.641 | 0.994 |
  | Qwen3-1.7B, `score` | 0.224 | 0.667 | 0.878 |

  It ties set of support at budgets 10 and 30 and trails it at 100. The smaller hosts gain only at budget 30.
- **Writing the step number (`react`) is worse than scoring the steps** (−0.12 to −0.20 on Qwen3-1.7B).
- **The hybrid is best:** set of support orders the steps and the model breaks ties (`sos-score`). It beats set of
  support at budgets 10 and 30 on every host.
  - Qwen3-1.7B: +0.141\* and +0.096\*. At budget 10 set of support finds 0.199 of proofs; the hybrid finds 0.340.
  - Nothing changes at budget 100.
  - The prediction for Qwen3-1.7B was met. The prediction that the SmolLM2 hosts would show no gain was not: the
    135M host gains +0.096\* and +0.064\*.

## Limitations
- Base models only.
- Single-entity problems.
- A weak chain-of-thought baseline (160 new tokens).

## Next steps
The hybrid is done. Premise retrieval at 10⁴–10⁵ clauses is approved (tier F, 2026-10-20) and being built.
