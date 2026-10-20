# Every finished experiment, explained (2026-10-20)

**Recorded:** 2026-10-20 by workplace-d2, at the author's request: "for experiments that have finished, explain in
detail: the setup, what they are testing, the key finding, any gaps and limitations, if the experiment was fair, and what
is next".

**Scope:**
- **E14 tier P:** every package that has finished, on the CPU and the GPU.
- **The three preliminary results of 2026-10-18,** which also finished.

**Sources:** these findings records hold the full numbers:
- `2026-10-17-e14-tier-p-cpu.md`;
- `2026-10-17-e14-concat-trees-reading-cycle.md`;
- `2026-10-18-prelim-readout-incontext-scale.md`;
- `2026-10-20-e14-tier-p-gpu.md`.

**Conventions:**
- paired bootstrap with 4,000 resamples;
- seeds pooled;
- 95% intervals;
- Holm within each pre-registered primary family;
- `*` marks Holm-significant results;
- "pp" means percentage points of accuracy.

**What each entry covers:**
- **Setup:** what was run.
- **Tests:** the question it answers.
- **Finding.**
- **Gaps:** limitations, and what it cannot tell us.
- **Fair?:** whether the comparison is a fair one.
- **Next.**

## At a glance

| # | Experiment | One-line result | Fair? | Next |
|---|---|---|---|---|
| A1 | RD-1a coverage audit | definitions state only 12% of a term's gold facts | yes, conservative | done; feeds A5/A8 |
| A2 | RD-1b teacher audit | 54% (ChEBI) and 30% (MeSH) of statements fit a relation | yes, unvalidated labels | relation discovery (A6) |
| A3 | RD-2 dictionary breadth | synonyms add nothing (0.116 → 0.117) | yes | closed |
| A4 | RD-5 bootstrap | no snowball: dictionary 0.25 → 0.28, then stalls | yes, narrow mechanism | an LLM-reader version (tier F) |
| A5 | RD-6 structure readers | the structured 42% of a frame comes from data, not text | by construction | combine with text (tier F) |
| A6 | RD-11 multi-round reading | finds relations (T1 +6), but gold recall only 0.157 → 0.162 | mostly | RD-12 (running) |
| A7 | RD-3 relation choice | ReAct beats the linker (+0.22 to +0.42 F1), loses to the type prior on T4 | mostly; our ReAct over-proposes | precision filter (tier F) |
| A8 | RD-4/7 explicit recall | +0.31 on new terms; −0.10 on real chemicals | **partly** (PMI null, long lists) | **RD-4b (queued)** |
| B1 | LG-A algebra, gate G0 | chains, joins, numbers and trees are exact at ≥ 1,024 dims; G0 passes | yes | resonator check |
| B2 | LG-C soft settings (CPU) | the store ties fair symbolic baselines; G2 fails | yes | real noisy inputs (tier F) |
| B3 | Concatenation (X1–X7) | role blocks make the impossible rules exact; a 3-store ensemble reaches 0.98 | yes, except X7 width | equal-width X7 |
| B4 | Tree layouts (X8) | heap concatenation is exact to depth 12 | unequal memory, by design | irregular trees |
| B5 | LG-P3 premise retrieval | recall@5 = 1.00 | too easy | stress test (tier F) |
| B6 | LG-B/C models read the executor | Qwen3 +0.42 to +0.47 at every depth; SmolLM2 fails (negation); G1 fails | yes, with a format confound | **verdict format (queued)** |
| B7 | LG-D trees with models | executor path 0.997 against tree text 0.37 | an interface test, not reasoning | noisy inputs (tier F) |
| B8 | LG-P model as search policy | soundness 1.00 with tools; guidance ≈ random | problems too easy | **hard problems (queued)** |
| C1 | Readout arms (binding program) | R1 looks null: twin contrast 0.49–0.50 in all arms | yes | official report (running) |
| C2 | Uncapped in-context glossary | method gain unchanged; the plain model gains less | yes | closed |
| C3 | 125M scaling, seeds 1–3 | C5 − C0 = −0.33% (all), −1.21% (long spans), as predicted | yes | the larger sizes |

---

# Part A — Reading facts from real text (RD)

**The question behind all of Part A is the author's:** why does reading facts from real text fail, and what would make
self-learning work? Proposal §9 splits a fact's path into four stages:
1. **stated:** the text says it;
2. **found:** the reader finds the partner;
3. **labelled:** the reader picks the right relation;
4. **used:** the model benefits from the stored fact.

## A1. RD-1a: what real definitions state (CPU)

**Setup:**
- T4's 341 held-out ChEBI definitions, against their 2,744 gold frame edges.
- An edge counts as *stated* when its partner's name or a dictionary alias appears in the definition.
- Three readers were scored against the same edges: E11's linker, the host reader and the type prior.

**Tests:** stage 1. How much of what we want the model to learn is in the text at all?

**Finding:**
- **11.7% of gold edges are stated.**
- By relation:
  - `is_a` 35%;
  - `has_role` 5%;
  - `contains_element`, `charge` and `branch` 0%. These three are 42% of all edges.
- The readers get 1.7% (linker), 8.0% (host) and 9.7% (type prior).

**Gaps:**
- String matching under-counts paraphrases, for example "an octanoyl derivative" for a parent of octanoic acid. The 12%
  is therefore a lower bound; RD-1b checks the other direction.
- One track only. T5 states 100% by construction.

**Fair?** Yes, as a descriptive audit. The bias is conservative: it under-counts what the text states.

**Next:** done. It explains why every reader has a low ceiling (A6, A7), and why RD-4b adds the stated-only items.

## A2. RD-1b: what the text says that the ontology could store (CPU + Claude, $2.19)

**Setup:**
- Claude labelled each statement of 100 definitions per track:
  - T4: ChEBI;
  - T1: MeSH scope notes, open licence.
- The six categories were ours: expressible by a track relation, expressible but the partner is not a concept,
  structural, a property no relation covers, provenance/quantitative, and other.

**Tests:** whether the ontology is too small. Is the failure on the ontology's side (no relation for what the text says)
or the reader's?

**Finding:**
- Statements a track relation can express: **54% on T4, 30% on T1.**
- About 77% of those also have the partner in the dictionary. So about 41% (T4) and 23% (T1) of what definitions say
  could be stored today.
- **On MeSH, 42% of statements are properties no relation covers.** The ontology really is too narrow there.

**Gaps:**
- The labels come from one LLM, with no human agreement check.
- 100 definitions per track; the category scheme is ours.

**Fair?** Yes, for its purpose. The caveat is label validity.

**Next:** the relation-discovery step in RD-11 (A6) is the response. A human spot check of 50 labels would make the
number citable.

## A3. RD-2: does a larger dictionary help? (CPU)

**Setup:**
- The partner finder ran with the base surfaces (33,941), then with ChEBI 255 synonyms and hyphen/case variants
  (65,514), then with the text de-hyphenated.
- A shrinking curve: 25, 50, 75 and 100% of the atoms' surfaces, 3 seeds.

**Tests:** stage 2, whether finding partners is limited by names.

**Finding:**
- **Synonyms add nothing:** gold partners found 0.116 → 0.116 → 0.117.
- The shrinking curve is linear (0.028 / 0.057 / 0.088 / 0.116). Recall equals the share of partners the text names.

**Gaps:**
- It tests surface lookup only. Partners an LLM reader could infer from a paraphrase are outside this test, and are
  covered by A6 and A7.

**Fair?** Yes.

**Next:** closed. "Bigger dictionary" is ruled out as the fix.

## A4. RD-5: does reading snowball? (CPU simulation)

**Setup:**
- 5,000 T4 definitions, 5 rounds, 3 seeds.
- Start with 25, 50 or 75% of the atoms linkable.
- Each round reads every definition with the finder and the type prior's relation choice, then adds each read entry's
  own atom to the dictionary.

**Tests:** the author's "do we need to build up the ontology before it picks up steam?" The specific mechanism tested:
read terms becoming linkable partners for later reading.

**Finding:**
- **No snowball.** The dictionary grows 0.25 → 0.28 (0.50 → 0.53, 0.75 → 0.76), then stalls after round 2.
- Edge precision is 0.22–0.28, and recall of all gold edges is 0.03–0.08.
- Read entries are rarely other entries' partners. Most mentioned concepts are not in the frame: mentions are not
  facts.

**Gaps:**
- One mechanism, one reader (the finder plus the type prior), one track.
- It does not test frame *enrichment* of existing atoms, or an LLM reader.

**Fair?** Yes for the mechanism tested. It does not settle the broader "build up first" question.

**Next:** tier F could repeat this with an LLM reader and with frames enriched over rounds. RD-12 (running) tests the
broader cycle with a model in the loop.

## A5. RD-6: the structured part of a frame (CPU)

**Setup:** read `contains_element` from the formula, `charge` from the record, and `branch` from the root of the `is_a`
ancestry, for the T4 held-out terms.

**Tests:** how much of a frame no text reader can ever supply, and whether structure readers supply it.

**Finding:** these edges are **42% of gold**, recovered at precision and recall 1.00.

**Gaps:** this is by construction. The ontology was built from the same ChEBI fields, so 1.00 measures plumbing, not
reading.

**Fair?** It is an accounting, not a contest, and is reported as such.

**Next:**
- A combined reader (structure plus text) for tier-F RD-4.
- The design point for self-learning: read structure from structured data, and use text only for what text says.

## A6. RD-11: multi-round active reading with relation discovery (CPU + Claude, $11.22)

**Setup:**
- **T4:** 160 ChEBI definitions, with 4 of the gold relations hidden at the start.
- **T1:** 120 MeSH scope notes.
- **Each round, Claude reads every definition twice** (the relation order shuffled). The facts both readings agree on
  are kept (consolidation).
- **A reflect step** clusters the reader's `NEW` proposals into relation types and accepts those with ≥ 8 supporting
  proposals.
- 3 rounds.

**Tests:**
- whether iterated reading with reflection grows the knowledge, even when gold is not reachable at first;
- whether the hidden relations are rediscovered.

**Finding:**
- **T4:**
  - 2 of 4 hidden relations rediscovered by name: `has_functional_parent` (recall 0.33, precision 0.23–0.26) and
    `has_part` (recall 0.13);
  - gold recall 0.157 → 0.162, against a text ceiling of 0.232;
  - the conjugate relations were not found, since definitions rarely mention them.
- **T1:**
  - 6 new relation types accepted;
  - the `NEW` share fell 0.72 → 0.32, and facts per note rose 2.04 → 2.40;
  - gold recall cannot rise, because the new knowledge is outside the gold relations.
- **Consolidation added little:** +0.002 precision on T4, +0.037 on T1.

**Gaps:**
- Gold recall measures only knowledge the ontology already has. The value of new relations must be measured downstream.
- Name matching to the hidden relations is conservative: a correct relation under another name counts as missed.
- **Disclosure:** the first-pass analysis wrongly counted rediscovered names as visible. It was corrected by
  `reanalyze` before the results were written up.

**Fair?** Mostly. Two readings by the same strong reader are not independent, which is why consolidation barely
matters.

**Next:** **RD-12, running.** It interleaves passive training with active reading and retracts relations that don't
help. Three arms are pending after a fix; the `gold` arm is done.

## A7. RD-3: choosing the relation, including the ReAct-style reader (GPU)

**Setup:**
- **Items:** every *stated* gold edge of the T4 held-out and T5 new sets, each with its definition. The bootstrap
  unit is the term: 248 T4 terms with at least one stated edge, and 300 T5 words.
- **Readers:**
  - E11's linker, type prior and host frames;
  - **`mc-rel`:** for each mention, a scored choice over relations: evidence PMI plus log P(relation | partner);
  - **`react`:** exact mentions → fuzzy trigram spans with a "none of these" option → relation → typing check;
  - Claude as teacher, T4 only.
- **Hosts** for the new readers: SmolLM2-360M and Qwen3-1.7B-Base.
- **Primary:** `react − linker` and `react − typeprior` on stated-edge F1, Holm per track.

**Tests:** stage 3. Is relation labelling the bottleneck, and does a smarter, ReAct-style matching fix it (the author's
question)?

**Finding:**

| Reader, stated-edge F1 (precision) | T4, 360M / 1.7B | T5, 360M / 1.7B |
|---|---|---|
| type prior | 0.573 (0.37) | 0.916 |
| linker | 0.101 (0.06) | 0.518 |
| `mc-rel` | 0.446 / 0.415 | 0.915 / 0.937 |
| `react` | 0.337 / 0.320 (0.19) | 0.917 / 0.936 |
| teacher (Claude) | **0.603 (0.67)** | — |

- `react` beats the linker everywhere (+0.22 to +0.42*).
- `react` loses to the type prior on T4 (−0.24*) and ties it on T5.
- **A 5× larger host does not help.** A frontier teacher gains only +0.03 F1 over the type prior.
- **So labelling is not the T4 bottleneck; coverage (stage 1) is.**

**Gaps:**
- Our `react` loses on precision because its fuzzy-span step over-proposes partners. That is a weakness of this
  implementation, not necessarily of ReAct-style reading.
- Base models, not instruction-tuned.
- The type prior and linker rows are identical across hosts (they come from E11 and use no host).

**Fair?**
- Mostly. The type prior is a strong baseline: it uses training-split relation statistics, which is legitimate.
- **Disclosure:** `mc-rel` received its relation prior after the smoke run showed it always picking `has_part`. This
  was pre-registered (§9) before any non-smoke run, and the prior-free version is reported as `mc-rel-lm`.

**Next:** tier F, which needs a go-ahead:
- a precision filter for `react` (a typing or functional check before committing a partner);
- an instruction-tuned host.

## A8. RD-4 / RD-7: can stored facts be *used*? Explicit recall against implicit injection (GPU)

**Setup:**
- **Items:** E11's own held-out items.
  - T4: 341 real ChEBI terms, 407 property items per seed.
  - T5: 300 invented words, 1,529 property items per seed.
  - Each track has property, paraphrase-consistency and entailment tests.
- **Hosts:** SmolLM2-360M C5, seeds 1–3, plus C0′ (plain fine-tuning) s1.
- **Conditions:**
  - `none`;
  - `implicit:<reader>`: the frame composed into the term's vector, E11's route;
  - `explicit-sym:<reader>`: the frame written as recall lines before the question;
  - `explicit-vsa`: the frame stored in the model's own VSA store and decoded back into lines;
  - `both`;
  - `definition`;
  - `noisy-*`: 10/30/50% of edges given wrong partners.
- **Scoring:** PMI against a "this" null prompt.
- **Primary:** (a) `explicit-sym:oracle − none`, (b) `− implicit:oracle`, and (c) the same for `stated` frames. Holm
  over 9 per track.

**Tests:** stage 4, the decisive one for self-learning. Even when the facts are right, does the model benefit, and by
which route?

**Finding:**
- **New terms (T5): explicit recall is a large win.**
  - Property +0.31* over `none` and +0.24* over implicit injection; paraphrase and entailment are also positive.
  - Implicit injection alone gives +0.07.
  - The store's decoded recall is as good as the symbolic record (±0.01).
  - Recall from the type prior's frames is as good as the oracle's. The noise curve falls monotonically.
- **Real chemicals (T4): explicit recall *lowers* property accuracy (−0.10*)** but raises paraphrase (+0.12*) and
  entailment (+0.11*).
- **Decision rule:** no E13 amendment.

**Gaps** (found in exploratory diagnostics after the primaries):
- **The recall did contain the answer:** for 399 of 407 T4 property items, and never a distractor.
- **The PMI null prompt also read the recall,** so part of the recall's evidence cancelled itself.
- **Long recall lists dilute.** A 15-line list of chemical names lowered even *raw* accuracy (0.733 → 0.696), while
  2–4-line lists raised it to 0.831.
- **T4 terms are partly known from their names alone** (`none` = 0.61), so there is little headroom and much to
  disturb.
- C0′ has one seed only.

**Fair?** **Partly.**
- The design was fair as pre-registered.
- The scoring disadvantages every context condition, both explicit recall and the definition. And the whole-frame
  recall confounds "the facts" with "a long list".
- The T5 result is robust to both issues: it is large, and the diagnostics show the effects work against it.

**Next:** **RD-4b, queued at 54.4959975** (pre-registration §13). It tests:
- recall of only the item's relation lines;
- a context-free null;
- a wrong-partner control;
- stated-only items.

If selective recall wins on T4, an E13 amendment (read → store → selective recall) is drafted for you.

---

# Part B — Logic in the store (LG)

**The question behind Part B:** can predicate logic be encoded directly in VSA (relations, rules, deductions), and does
that help a language model?

## B1. LG-A: the algebra and gate G0 (CPU)

**Setup:**
- Synthetic worlds of random vectors, 3 seeds, widths 256–2,048 (to 16,384 in the addendum).
- **Grids:**
  - A1: one rule as one mapping vector;
  - A2: chains by unbinding with clean-up per hop;
  - A3: joins / Datalog with calibrated thresholds;
  - A4: nesting and order;
  - A5: connectives;
  - A6: numbers (fractional power encoding);
  - A7: trees and lists;
  - A8: resonator factorisation;
  - A11: the trained T4 store's own rules;
  - A12: rule mining and KG-embedding baselines.

**Tests:** whether the algebra can *execute* logic reliably, and at what width and load.

**Finding:**
- **Gate G0 passes:** depth-3 chains at load 8 score 1.00 from 512 dims; an inverse rule from k = 3 examples scores 1.00
  on unseen entities.
- **Chains, joins (closure F1 1.00 at 1,024), negation as failure, numbers and decision lists are exact at ≥ 1,024 dims.**
  HRR, MAP and FHRR are indistinguishable.
- **Hard limits:**
  - one global bundle of all facts scores 0.00, so storage must be per entity;
  - fixed-point rules (symmetry, sub-relation in place) are impossible for one convolution, scoring ≤ 0.16 at any
    width;
  - a rule applied to a whole frame fails (≤ 0.02).
- **Trained, correlated fillers cost 0.21** at 256 dims.
- **The real trained T4 store chains held-out terms** through 2-hop `is_a` at precision and recall 0.94.

**Gaps:**
- Synthetic worlds; trained fillers only at 256 dims.
- Our resonator is weaker than published ones, so A8 is not yet citable.
- The A12 baselines are not a learning contest: the executor is given the rules.

**Fair?** Yes, as capability tests. Each limit was predicted before the run.

**Next:**
- check the resonator against Kent et al.;
- the store's derived facts need their own memory, since only 6–16% of converses exist in T4 frames.

## B2. LG-C, CPU part: the soft settings where the store "should" beat symbols, and gate G2

**Setup:** each setting paired with a fair symbolic baseline:
- C1, completing a mostly-true type property: rule induction;
- C2, aliases with 1–2 edits or abbreviations: prefix plus edit distance;
- C3, 10% / 30% wrong facts: Datalog;
- C4, rules from k = 1 or k ≥ 3 examples: AMIE-style mining.

**Tests:** gate G2. Is there any setting where vector logic beats symbolic logic?

**Finding:**
- **No advantage anywhere:** C1 0.74–0.79 against 0.75–0.81; C2 0.97–1.00 against 0.996–1.00; C3 identical; C4 1.00
  against 1.00.
- Margins detect decoding noise, not wrong facts.
- **G2 fails on its first condition.** Its second condition (edit propagation) passes on the Qwen3 hosts (B6).

**Gaps:**
- Clean synthetic data, where symbols are at their best. Real fuzzy inputs (OCR'd names, real alias noise) are untested.
- The mapping vector "solves" a chain rule without checking the shared middle entity, which is a latent unsoundness.

**Fair?** Yes; if anything generous to the symbolic side, whose baselines are strong.

**Next:**
- tier F with real noisy inputs, and a join test with mismatched middles;
- **the manuscript claim changes** to "exact executor with a model-readable interface", not "logic in the store".

## B3. Concatenation instead of superposition, X1–X7 (CPU; the author's HRRCat idea)

**Setup:**
- The same rules and stores under several layouts at equal total width:
  - one wide HRR;
  - redundant blocks;
  - one block per role;
  - one block per fact;
  - fact × role blocks;
  - per-relation slots.
- Widths 1,024–16,384.
- **X7:** combine the three trained T4 stores (seeds 1–3) by averaging cosines.

**Tests:** whether concatenation removes the superposition noise and the algebraic limits found in B1.

**Finding:**
- **Role blocks make the "impossible" rules exact:** symmetry and sub-relation score 1.00. The rule becomes a block
  permutation.
- **Fact × role blocks make whole-frame rules exact** (1.00 at every width).
- **Trained fillers with role blocks score 0.99–1.00.**
- **Redundant blocks equal the wide vector,** so width is width.
- **Per-relation blocks** keep light relations exact at any load, but cost heavy relations at equal width.
- **X7:** the 3-store ensemble decodes at **0.979** (held-out) against 0.89–0.92 for single stores. Conjugate acid
  0.81–0.89 → 1.00.

**Gaps:**
- **X7's ensemble has 3× the parameters** of one store. A single 768-wide trained store is the fair comparison, and we
  don't have one.
- The layout rule (light relations in blocks, heavy ones wide) is from synthetic loads.

**Fair?** X1–X6: yes, at equal width. X7: no, by width. It is reported as "independent stores combined", not as "better
at equal size".

**Next:** an equal-width single store for X7, which needs a training run (tier F).

## B4. Tree layouts by the tree's structure, X8 (CPU; the author's idea)

**Setup:**
- Complete binary trees: 20 features × 8 values, 10 classes, depth 2–12, 3 seeds; the input record fixed at 4,096 dims.
- **Layouts:**
  - one nested vector (16,384);
  - node memory (associative);
  - **heap concatenation** (node i in block i);
  - level concatenation.

**Tests:** the author's question, whether concatenating by the tree's structure beats storing the tree in a single
vector.

**Finding:**
- **Heap concatenation is exact (1.00) at every depth to 12, even with 64-dim blocks.**
- The nested vector collapses: 0.11 at depth 8, failing at any width (B1, X5).
- Node memory needs blocks ≥ 256.
- Level concatenation holds until a level has about L/8 nodes.

**Gaps:**
- Heap width grows as 2^d: 0.5M–2M dims at depth 12.
- Complete trees only; irregular trees (a preorder layout with subtree sizes) are untested.

**Fair?** It is a layout study at unequal memory, by design: it trades memory for exactness, and the widths are reported.

**Next:** irregular and sparse trees; heap for the top levels plus node memory for deep sparse parts.

## B5. LG-P3: premise selection by VSA retrieval (CPU)

**Setup:** rule memory as `HEAD⊛head + BODY⊛body`; retrieve the rules for a goal by unbinding HEAD; 50 and 500 clauses;
width 1,024.

**Tests:** whether the store can fetch the right premises for a prover.

**Finding:** recall@5 is 1.00 at both sizes.

**Gaps:** too easy. Each vector holds only two bindings, and the classes are random.

**Fair?** Fair but uninformative.

**Next:** a stress test at 10⁴–10⁵ clauses with correlated class names (tier F).

## B6. LG-B / LG-C with models: does a frozen model read the executor's conclusions? Gate G1 (GPU)

**Setup:**
- **Synthetic worlds of invented words** (seed 1), so the host has no prior knowledge:
  - B1: 1,200 true/false taxonomy questions at depth 0–5, plus 600 cloze;
  - B4: 2 × 600 propositional items;
  - B5: 500 composition chains, k = 2–6;
  - B6: rule bases of 10 and 50;
  - C4: 1,600 few-example rules;
  - C5: 300 edits × 3 hops.
- **Conditions:**
  - `none`;
  - `text` (rules as text; the host must reason);
  - `text-retrieved` (keyword RAG);
  - `exec-vsa` (the VSA executor's trace and conclusion as recall lines);
  - `exec-sym` (the same from an exact symbolic executor);
  - controls: no binding, shuffled, noisy, stale.
- **Hosts:** four frozen base models (SmolLM2-135M / 360M, Qwen3-0.6B / 1.7B).
- **Scoring:** forced choice by length-normalised log-probability.

**Tests:** level L1 of the proposal. If an executor does the logic, can a small model relay its conclusion, where it
cannot do the reasoning from text?

**Finding:**
- **Qwen3 hosts: yes, decisively.**
  - `exec-vsa − text` is +0.42 to +0.47* at every depth.
  - Rules as text decay with depth (Qwen3-1.7B: 1.00 → 0.57); the executor holds 1.00.
  - Composition chains score 1.00 at every k.
  - **Edits propagate** (+0.62 / +0.92 against a stale store).
  - **The controls confirm content is read:** no binding −0.50, shuffled −0.35.
- **SmolLM2 hosts: G1 fails.**
  - The diagnosis: they answer "True" to "X is not a Y" (false items 0.15), while scoring 1.00 in cloze format.
  - On propositional items ("no rule shows that …") they reach 0.94.

**Gaps:**
- **A format confound,** negation in true/false, decided G1.
- `exec-vsa − exec-sym` for SmolLM2 (+0.03 to +0.05) comes from the confidence numbers, not from the store.
- Synthetic worlds; frozen base models; no chain-of-thought baseline in B.

**Fair?** Yes, for what it claims.
- The executor does the work, and `text` asks the host to. The gain is "executor plus relay" against "host reasoning".
  That is the intended comparison, not a claim that the model got better at reasoning.

**Next:** **verdict format, queued at 54.4959976** (pre-registration §14). It is the same recall with the conclusion
written as "the claim … is True/False", and gives gate G1′.

## B7. LG-D: decision trees and lists read by a model (GPU)

**Setup:**
- **Items:**
  - 600 trees (depth 2/3/4/6) over invented attributes;
  - 300 decision lists (length 3/6/12);
  - 200 edits.
- **Conditions:**
  - `tree-text` (the tree written out; the host must execute it);
  - `tree-bullets`;
  - `path-vsa` (the executor's path and leaf);
  - `path-sym`, `leaf-vsa`, `path-shuffled`, `path-stale`.
- **Hosts:** the four hosts of B6.

**Tests:** level L3, whether models use tree execution results, and whether the store's tree executor (B4) is
model-readable.

**Finding:**
- `path-vsa` scores 0.997 on every host, against 0.37 for `tree-text`. The gain grows with depth (+0.51 → +0.75*).
- Edits are followed perfectly (+1.00 against stale).
- `leaf-vsa = path-vsa`: the leaf alone suffices.

**Gaps:**
- The executor's output essentially *is* the answer, so this is an interface test.
- Inputs are clean; the noisy-input robustness question is not addressed here.

**Fair?** Fair as an interface test, and reported as such. It does not measure reasoning.

**Next:** tier F with noisy or partial inputs, where the soft path might matter.

## B8. LG-P: the model as the prover's search policy (GPU)

**Setup:**
- **Problems:** 800 one-entity first-order problems: Horn chains, proof by cases, proof by contradiction; depth 1–4;
  3 or 30 distractor clauses; 408 provable.
- **Sound tools:** backward step, case split, contradiction.
- **The model's role:** at each choice point it ranks the candidate steps (`score`), or names one (`react`).
- **Baselines:**
  - non-model searchers: `random`, `bfs`, `sos`;
  - model policies without sound tools: `unsound` (an "accept without proof" option) and `cot` (prose proof, no tools).
- **Budgets:** 10 and 30 steps.

**Tests:** the author's question, whether the prover can be the LLM itself in a ReAct-style loop.

**Finding:**
- **Soundness: every policy with the sound tools scores 1.00, on every host.** Without them, `unsound` and `cot` score
  0.51–0.57, about chance.
- **Guidance: no benefit at this difficulty.**
  - `score − random` ≈ 0. Only Qwen3-1.7B gains, +0.029* at budget 10.
  - Random search already finds 92% within 30 steps; set of support finds 100%.
  - The generative `react` is worse than `score` on Qwen3-1.7B (0.60 against 0.79 at budget 10).

**Gaps:**
- **The problems were too shallow** to need guidance.
- The CoT baseline is weak: base models, 160 new tokens, no examples.

**Fair?** The soundness comparison is fair and clean. The guidance comparison was uninformative, not unfair.

**Next:** **hard problems, queued at 54.4959977** (pre-registration §15). Decoy chains of depth 4 × 3 per class were
chosen on non-model policies only (random 0.51 at budget 30, set of support 0.64), with budgets 10/30/100.

---

# Part C — Finished outside E14 (the 2026-10-18 preliminaries)

## C1. Readout arms (binding program; workplace-9a's area, numbers from per-run outputs)

**Setup:**
- SmolLM2-360M T5 runs trained with a readout that unbinds the store, under 8 operator arms, 3 seeds each:
  - learned HRR;
  - unitary;
  - bounded spectrum;
  - block unitary;
  - slotted;
  - fixed random;
  - controls: translation and untyped.

**Tests:** R1, "with a trained readout, binding matters".

**Finding:**
- **R1 looks null.** Every arm, including the controls, is at chance on role-swapped twins (0.49–0.50).
- The store is readable with the gold role (oracle MRR 0.94–0.99), but the model's own role query is mostly wrong
  (0.07–0.20).

**Gaps:** the numbers are from per-run files. The official report is running on the CPU lane now.

**Fair?** Yes; the controls are included.

**Next:** the official report, then 9a's write-up. For E14, this is the same message as A8: models don't learn to query
by role, so give them explicit recall.

## C2. In-context glossary without the cap (E9 addendum 2 §1)

**Setup:**
- A 4,096-token glossary covering 98.8% of each window's entries, against the 1,024-token cap (45% coverage on T4, 79%
  on T5).

**Tests:** whether the method's gain over an in-context glossary survives full coverage.

**Finding:**
- **The method's gain is unchanged:** T4 −0.36%*, T5 −10.93%*.
- **The plain model gains *less* from the longer glossary** (−0.66%* → −0.25%, n.s.). This was against the prediction.

**Gaps:** one host size.

**Fair?** Yes.

**Next:** closed. The dilution effect matches A8's long-list finding.

## C3. 125M scaling screen, seeds 1–3 (E4 addendum 2 §3)

**Setup:** `scale-v1` 125M models, C5 against C0, seeds 1–3.

**Tests:** whether the predicted size trend holds across seeds.

**Finding:**
- C5 − C0 = −0.33% (SD 0.04) overall and −1.21% (SD 0.07) after long spans.
- Both match the predictions (about −0.3% and −1.1%).

**Gaps:** the report tool analyses one seed per size; it needs a pooled-seed mode.

**Fair?** Yes.

**Next:** the larger sizes, per the scaling plan.

---

# Part D — The picture so far

1. **Executing logic is solved at modest width; deciding *what* to store is not.**
   - Chains, joins, numbers and trees are exact (B1, B3, B4).
   - Real text states only 12% of the facts we want (A1), and labelling is not the bottleneck (A7).
2. **The interface matters more than the algebra.**
   - Models gain hugely when the executor's conclusion is handed to them (B6, B7).
   - They gain little when facts are hidden inside the term's vector (A8 implicit, C1).
   - Small models stumble on the *wording* of negative conclusions (B6), and long recall lists dilute (A8, C2).
3. **The defensible manuscript claim:** an exact executor with a model-readable interface, plus selective explicit
   recall for new knowledge. Not "soft reasoning better than symbols": G2 fails.
4. **Running and queued:**
   - RD-12, the reading cycle, 3 arms;
   - RD-4b, selective recall;
   - LG-B verdict format;
   - LG-P hard problems.
   - About 4 GPU-h after RD-12. Reports on the CPU lane at 54.4959979.
