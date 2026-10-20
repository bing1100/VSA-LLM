# 10 — E10: a self-learning ontology

**Overview.** E10 asks whether a model can learn, test and repair its own ontology instead of relying only on the
curated one. Every result so far is negative or narrow.
- **10.1, the planted world:** a learnable ontology recovers erased relations, discovers hidden ones and repairs errors.
  But a no-learning readout and standard rule mining explain most of it, and its acceptance test accepts false relations
  in worlds that have none.
- **10.2, real WordNet with a frozen GPT-2:** nothing transfers. The learner does no better than graph-only baselines,
  and its self-tests are at chance.
- **10.3, E10.9a:** pairing passive learning with active self-reflection is **not faster** at matched data
  (pre-registered primary null). It does write correct edges for concepts no observation covers.
- **10.4, the follow-up "learn" tool (E10.L):** its decision rule is calibrated on synthetic null worlds. Its 100
  real-track jobs are still pending.

The paper makes no self-learning claim (claims ledger row E10.D1: "no protocol claim (negative / methods result)").

**Terms used throughout.**
- **VSA (vector-symbolic architecture):** an algebra that stores a structured record in one vector.
- **HRR (holographic reduced representations):** the VSA we use. **Binding** combines a relation (role) vector with a
  related concept (filler) vector by circular convolution. A concept's vector is the sum of its bound pairs.
  **Unbinding** applies the relation's inverse and returns a noisy copy of the filler.
- **Frame:** a concept's list of edges (relation, filler). An **ontology** is all frames.
- **Learnable ontology:** each edge has a learnable mass. Curated edges start at mass 1, candidate edges at 0.
  **L2-to-prior** pulls each learnable part back towards its curated or pretrained starting value.
- **Blank slot:** a new relation operator with no prior. It opens empty and may capture candidate edges.
- **Null world:** a world with nothing to discover, so every accepted relation or edge is false.
- **Metrics:**
  - **AUC:** ranking quality, 0.5 = chance;
  - **ARI** and **Jaccard:** agreement of a discovered relation with the hidden one;
  - **F1:** the balance of precision and recall;
  - **MRR (mean reciprocal rank):** 1 / rank of the right answer, averaged.
- **Baselines:**
  - **AMIE:** standard Horn-rule mining (e.g. "part_of reversed implies has_part");
  - **KGE (knowledge-graph embeddings):** TransE, RotatE and ComplEx, which complete a graph from the graph alone.
- **Codes from the main study (used in 10.4):**
  - **C5:** the method, which adds a vector composed from a term's frame to a pretrained language model;
  - **C2:** a free per-term lookup table;
  - **C0′:** the same host fine-tuned without the method;
  - **T4, T5, T7:** the tracks on real chemistry, a synthetic glossary, and new 2026 MeSH names.
- **Statistics.** Unless stated: mean [95% t-interval] over seeds. A positive difference means the first arm is better.

Report: `reports/R10-self-learned-semantics.md`. Pre-registrations: R10 "E10.9" (E10.9a);
`experiments/e10-self-semantics/e10-learn/preregistration.md` (E10.L).

---

# 10.1 — A learnable ontology on planted data (E10.0), and the baselines that narrow it

**Status:** done · **When:** E10.0 2026-10-02; baselines (WP-PQ2) 2026-10-03 · **Sources:**
`reports/R10-self-learned-semantics.md` §§1–4 and "E10 baselines for claim D";
`experiments/e10-self-semantics/runs/e10.0-v3/`, `runs/e10.0-baselines-v2/`; `manuscript/claims.md` row E10.D1;
`manuscript/update-2026-10-12.md` E10-1

## Research gap
Ontology learning has strong standard tools: rule mining, link prediction, relation clustering. We asked whether a
composer whose ontology is learnable can do four things the curated store cannot. It should recover erased edges and
discover relations it was never given. It should accept or reject its own hypotheses by held-out tests. And it should
map a new word zero-shot (hypothesis H-H).

## Intuition
Treat the ontology as parameters pulled towards the curated version. Gradient descent can then raise the mass of a
missing edge that the data support. A blank slot can collect pairs that behave alike, and a held-out test can decide
what to keep.

## Methodology
- **World:** 304 concepts in a 4-level is-a tree; 7 relations with planted logic (e.g. *similar to* is symmetric,
  *has part* is the inverse of *part of*, *located in* is transitive); ≈ 1,330 edges.
- **Teacher and observations:** an HRR teacher (d = 64). Each concept has 12 noisy observations: training 0–3,
  validation 4–7, and an audit split 8–11 that no learning step reads.
- **Priors:** concept and relation vectors start at cosine 0.8 to the truth, a stand-in for pretrained vectors.
- **Tests:**
  - learnability grid (3⁴ settings);
  - erased-edge recovery at 10–70% erasure;
  - blank-slot discovery, with three relations hidden ("absent") or two merged ("collapsed");
  - held-out self-tests;
  - new-word frame inference;
  - offline revision ("dreaming").
- **Refutation clauses (pre-written):** (i) no recovery above matched random candidates; (ii) no slot alignment above
  chance; (iii) self-acceptance no better than random acceptance; (iv) new-word frames no better than nearest-neighbour
  frames.
- **Seeds:** evaluation seeds 101/202/303. Dev seeds 7–9 chose the hyper-parameters.
- **Baselines (WP-PQ2):** AMIE; TransE, RotatE, ComplEx; an IterE-style axiom reading; a no-learning readout that
  correlates the training observations with prior-bound candidates; two null worlds; fresh validation splits.

## Experiment
81 CPU jobs (0.71 CPU-h) for E10.0, then 51 baseline jobs on the same worlds, pools and seeds.

## Results
**E10.0 alone meets no refutation clause; R10 rates clause (ii) "supported, weakly":**
- **Recovery:** AUC 0.984–0.989 at 10–70% erasure, precision ≈ 1.0, recall 0.83 → 0.38.
- **Discovery:** mean best Jaccard 0.441 [0.302, 0.580] against 0.839 for oracle labels; ARI 0.386 [−0.005, 0.778].
- **Property tests:** 14 rules adopted, 13 true of the matched hidden relation.
- **Self-test:** edges +0.103 [+0.082, +0.124] accuracy over random acceptance at the same rate. The in-sample test
  accepts 98–99%.
- **New words:** frame F1 +0.07 [+0.01, +0.14] over nearest-neighbour frames at 1 observation, +0.32 [+0.24, +0.41] at 8.
- **Repair:** revision passes fix 0.46–0.53 of injected wrong edges and damage ≤ 0.3% of correct ones.
- **L2-to-prior matters:** freeing the concept vectors costs −0.319 [−0.370, −0.268] in zero-shot fit.
- **Negative:** building the ontology from an empty or core seed fails (edge F1 0.126 and 0.254); one-relation-at-a-time
  growth is no better than plastic or all-at-once growth.

**The baselines take most of it away:**

| Sub-claim | Learnable ontology | Baseline | Reading |
|---|---|---|---|
| Erased-edge recovery (AUC) | 0.984–0.989 | no-learning readout 0.909–0.917; KGE and AMIE 0.54–0.77 | learning adds ≈ +0.07; the edges are present in the training signal |
| Property identification | correct on accepted slots | AMIE on the same pairs: the same correct rules (5/5 absent, 3/3 collapsed), plus more false ones | matches rule mining |
| Logic in the learned operators | HRR role vectors: AUC 0.45 | — | none |
| False slots accepted per null-world run | — | 2.67 [1.23, 4.10] and 4.33 [2.90, 5.77], in every run | no false-discovery control |

- **Fresh validation splits change nothing.** The edge test still accepts 40–45% of proposed distractor edges.

**Reading:** H-H survives E10.0. By the pre-written framing rule, the baselines turn it into a negative / methods
result, not a protocol claim.

## Limitations
- The world was built for the mechanism: an HRR teacher from the learner's own family, and priors at cosine 0.8. E10.0
  is therefore an upper bound.
- The ARI varies widely between seeds (0.21–0.51).
- The no-learning readout was added after the first baseline run (disclosed). The fresh splits are simulated.
- The sources disagree on wording. The draft's contribution list (item 7) still uses the superseded claim ("discovers
  hidden relations at about half of oracle alignment …"). Its §5.11 and the claims ledger use the narrowed negative /
  methods wording, which is more recent, and we follow it.

## Next steps
- **Closed:** the protocol claim.
- **Done in another form:** the null-calibrated acceptance test R10 asked for. It is E10.L's `holm+decoy` rule (10.4).
- **Not run (proposed in R10):** the clustering, revision, resonator and WN18RR baselines (D-B3, D-B6, D-B7, D-B9).

---

# 10.2 — WordNet with a frozen GPT-2: nothing transfers (E10.1)

**Status:** done (negative) · **When:** 2026-10-03 · **Sources:** `reports/R10-self-learned-semantics.md` "E10.1"
and "B.5"; `experiments/e10-self-semantics/runs/e10.1-v1/`, `runs/e10.1-baselines-v2/`;
`manuscript/draft.md` §5.11

## Research gap
E10.0's world was made for the learner. We needed a real ontology and real model signals.

## Intuition
If the frozen model's hidden states carry WordNet's relations, the learner should recover and discover them as it did
in the planted world.

## Methodology
- **Data:** 3,000 of E2's 6,000 WordNet concepts, d = 128, 3 seeds.
- **Targets:** frozen GPT-2 hidden states from several templates per concept (768-d "anchors").
- **Design:** the four E10.0 clauses, with 15 learnability settings instead of the full grid. The source-variety test
  is dropped, because WordNet has no observation sources.
- **Baselines:** the same KGE models and AMIE, on graph-only recovery at 30% and 50% erasure (seeds 11/22/33).

## Experiment
A GPU job extracted the anchors; the learning ran on the CPU (planned at ≈ 9–10 CPU-h).

## Results
| Readout | Synthetic (E10.0) | WordNet, frozen GPT-2 (E10.1) |
|---|---|---|
| Erased-edge recovery AUC, 30% erased | 0.987 | 0.589 |
| Self-acceptance vs random | +0.10 | ≈ 0 (edges 0.46 vs 0.45; slots 0.35 vs 0.37) |
| New-word frame F1, 4 observations, vs nearest neighbour | 0.64 vs 0.36 | 0.007 vs 0.267 |
| Rules adopted / true | 13 of 14 true | 7 adopted of 37 slots, 4 true |
| Revision repairs wrong edges | ≈ 50%, ≤ 0.3% damage | 7–12%, with 8–12% damage |
| Best zero-shot fit | L2-to-prior | everything fixed (0.149 vs 0.094) |

- **Most clauses fail.**
  - (i) holds only at 50% erasure (AUC 0.53–0.61).
  - (ii) holds for collapsed relations (ARI 0.33). For absent relations only the per-seed permutation test passes
    (ARI 0.04).
  - (iii) and (iv) are refuted.
- **Graph-only baselines match or beat the learner.** ComplEx reaches 0.628 [0.618, 0.638] against 0.589 [0.527,
  0.651] (ΔAUC +0.04 [−0.03, +0.11]). AMIE closure is at chance (0.508).
- **Two structural positives survive.** One-relation-at-a-time discovery beats all-at-once (collapsed ARI +0.31 [0.26,
  0.36]). In continual discovery of *similar to*, one-at-a-time growth with revision between stages reaches Jaccard
  0.36, against 0.15 for all-at-once.

## Limitations
- Frozen anchors were already known to be a poor target (E2 test MRR 0.03–0.08; studies 01a–01c). The test may fail on
  the signal, not on the learner.
- One model and a 3,000-concept slice, on a reduced design.

## Next steps
- **Proposed, not scheduled:** E10.2, the same test inside joint language-model training. R10 calls it decisive. The
  plan places it after the recipe sweep is analysed, pending an author decision. The draft keeps an `[[E10.2]]`
  placeholder.

---

# 10.3 — E10.9a: active self-reflection plus passive learning is not faster

**Status:** done; pre-registered primary null · **When:** pre-registered 2026-10-04 (`decefae`); results 2026-10-04
(`8ee946f`) · **Sources:** `reports/R10-self-learned-semantics.md` "E10.9";
`experiments/e10-self-semantics/runs/e10.9-v1/`; `experiments/e10-self-semantics/e10.9-active-passive.yaml`;
`resources/plan-improvement/execution.md` decisions 52–53; `resources/plan-improvement/experiments.md` row E10.9

## Research gap
After the null-world result, the author reframed the question (decision 52). Learn passively by gradient first. Then
reflect: test what a learned relation means, and commit the best explanation as an explicit write. Does this hybrid
learn "much faster" than passive learning alone? The null world becomes a test of what a wrong commitment costs.

## Intuition
A gradient learner can only adjust edges it already has. A committed rule, such as "this relation is the inverse of
*part of*", writes edges for concepts that no observation covers. That could reach a given fit with less data.

## Methodology
- **World:** E10.0's planted world with three hidden relations.
- **Structure fraction ρ ∈ {1, 0.5, 0.25, 0}:** the share of planted pairs kept. ρ = 0 is the null world.
- **Budgets:** f ∈ {0.05, 0.1, 0.2, 0.4, 0.7, 1.0} of 1,632 observations, given as nested prefixes; 4,000 steps; a
  new slot every 800 steps.
- **Arms:**
  - **P:** passive, training on every observation of the budget (matched data);
  - **P-split:** passive, on the hybrid's gradient data only;
  - **H:** hybrid. Every 800 steps it tests property hypotheses on held-out pairs and commits the passing ones (adds
    rule-implied edges, prunes violations);
  - **H+R:** H plus a revision pass that may retract a commitment;
  - controls **H-rand** (a random write of the same size), **H-AMIE** (AMIE's explanation), **P+compute** (matched
    compute) and **A-first** (reflection before any passive phase).
- **Metric:** zero-shot fit of test concepts that no arm ever observes.
- **Primary:** AULC (area under the fit-versus-budget curve, linear f), H − P at ρ = 1, over 5 seeds. "Faster" iff its
  95% t-interval lies above 0. "Much faster" also needs half of P's data or less, in ≥ 4 of 5 seeds.
- **E10.9b gate:** both H − P and H − P+compute must be above 0.
- **Disclosure:** a dev-seed pilot ran before the pre-registration.

## Experiment
715 arm-runs in 160 CPU jobs; 65 min wall, 3.2 CPU-h. The first launch failed on untestable predictions; the fix
(`f671ef2`) changes nothing at f = 1.

## Results
| Quantity (ρ = 1 unless stated) | Estimate [95% t] | Criterion |
|---|---|---|
| **Primary:** AULC H − P, linear f | −0.003 [−0.019, +0.013] | "faster": **failed** |
| AULC H − P, ln f | −0.013 [−0.026, −0.001] | secondary 1: H slower on this axis |
| Share of P's data for H to reach P's full-data fit | 0.60 (≤ 0.5 in 3 of 5 seeds) | "much faster": **failed** |
| AULC H − P+compute | −0.003 [−0.019, +0.013] | gate: **failed** |
| AULC P − P-split (cost of the held-out split) | +0.018 [+0.013, +0.022] | — |
| Test fit H − P-split at f = 1 (equal gradient data) | +0.021 [+0.006, +0.035] | — |
| AULC H − H-rand (content) | +0.030 [+0.006, +0.054] | refutation "content irrelevant": not met |
| AULC H − H-AMIE (generator) | +0.001 [−0.004, +0.007] | — |

- **The held-out split costs as much as reflection gains.** The hybrid tests on half of the data it reads. A passive
  learner trains on those observations instead.
- **The whole gain is rule-implied edges.** They recover 0.38 [0.18, 0.59] of the hidden edges of unobserved test
  concepts, at precision 0.81. Without them, H's fit (0.681) equals P-split's (0.679).
- **Reflection pays only when the structure is intact.** Against the null world, where H − P is just the cost of the
  held-out split, reflection is worth +0.015 AULC at ρ = 1 and nothing at ρ ≤ 0.5 (H − P −0.018 at ρ = 0.5, 0.25 and 0).
- **Wrong commitments are inert and never retracted.** In the null world H commits 4.0 false slots per run.
  - At equal gradient data they cost nothing measurable: H − P-split +0.0006 [−0.0002, +0.0014].
  - H+R retracted 0 of 19 false commitments at f = 1.
  - The non-inferiority of H+R against P fails at ρ = 0 (−0.0087 [−0.0149, −0.0024]) and ρ = 0.25 (−0.0107 [−0.0168,
    −0.0046]). The decomposition attributes the deficit to the held-out split.
- **Passive learning must come first.** The gate refused to commit untrained slots in 40 of 40 runs. Forcing the commit
  cost −0.010 [−0.014, −0.006] against P.

**Reading:** no "learns faster" claim. The E10.9b gate is not met, so E10.9b is not queued. A narrow, synthetic-only
mechanism remains: committed closure rules write correct edges for never-observed concepts, and the gain depends on the
rule's content.

## Limitations
- A planted world. A wrong commitment is cheap here, because false slots almost never adopt a closure rule.
- P+compute equals P because the passive learner plateaus by step 100. The compute control is uninformative.
- The dev-seed pilot preceded the pre-registration (disclosed). Nothing else changed after it.

## Next steps
- **Closed:** E10.9b. The design stays ready (execution.md).
- **Proposed in R10:** a self-test that needs no separate held-out split, or a regime where held-out evidence is free
  (held-out text windows in joint LM training).

---

# 10.4 — E10.L: the "learn" tool, calibrated decompose-then-verify

**Status:** partly done (synthetic calibration done; real-track runs queued) · **When:** pre-registered 2026-10-08
(`0bdcbff`); calibration 2026-10-08 (`88c4351`); amendment 1 2026-10-08 (`ded66ec`); amendment 2 2026-10-09 (`e9a83b0`);
100 jobs pending at queue priority 54.4995–54.49959 · **Sources:** `experiments/e10-self-semantics/e10-learn/preregistration.md`;
`experiments/e10-self-semantics/e10-learn/README.md`; `experiments/e10-self-semantics/e10-learn/runs/synthetic-v1/{prior,truth}/report.md`;
`manuscript/update-2026-10-15.md` ("Already queued")

## Research gap
E10.9a refuted the speed claim, and the baselines showed that the acceptance test had no false-discovery control. The
remaining question is whether what a model absorbs by passive training can be made explicit, correct and calibrated on
real data (decision 63). The pre-registered claim at stake: "the learn tool writes correct edges that the curated
ontology lacks, at a measured false-discovery rate, better than graph-only completion".

## Intuition
Take a concept's passive vector: its C2 row, or the C0′ host's hidden states at its mentions. Map it into the store's
space and decompose it over the trained C5 dictionary. New atoms are proposed edges. Accept an edge only if it passes a
held-out test whose false-acceptance rate we measure on null worlds built from the same data.

## Methodology
- **Proposals:**
  - `decompose`: a cross-fitted ridge decoder, then non-negative orthogonal matching pursuit (OMP) that adds up to 2 new
    atoms;
  - `closure`: AMIE rules at PCA confidence ≥ 0.8.
- **Test:** the utility of the edge on held-out mentions, against the same edge added to 8 same-type control concepts
  (one-sided Welch t, ≥ 4 observations).
- **Rule `holm+decoy`:** an edge needs both components:
  - Holm over all proposals at α = 0.05;
  - a target–decoy false-discovery threshold at q = 0.05, cross-fitted over two halves of the concepts.
- **Null worlds:** complete (nothing erased), relabel, swap, permute, and planted (synthetic only).
- **Primary endpoints (T7, pooled over seeds 1–3):**
  - **L4a:** ≥ 30 accepted edges, precision credibly above 0.80, and every null ≤ 5% (inclusive);
  - **L4b:** recall at precision 0.8 above AMIE, TransE, RotatE and ComplEx.

  The `c2` and `features` arms form one Holm family (amendment 1).
- **Secondaries:**
  - S1: placement of new MeSH 2025 → 2026 records, by MRR and Hits@k;
  - S2: T4 and T5;
  - S4: the `c5full` positive control;
  - S9: a fixed random binding store (amendment 2).
- **Synthetic calibration:** the E10.0 world, seeds 101/202/303, with a noisy-prior dictionary (misfit) and an oracle
  one.

## Experiment
Only the synthetic calibration has run. The 100 real-track jobs (78 plus 22 for S9) are all pending in the queue. The
15 October update estimated about 24 October for them.

## Results
**Synthetic calibration, `holm+decoy`, decompose source:**

| | Noisy-prior dictionary | Oracle dictionary |
|---|---|---|
| Precision | 0.742 (only seed 101 accepts any) | 0.864 [0.745, 0.982] |
| Recall | 0.038 [−0.125, 0.200] | 0.610 [0.497, 0.723] |
| Accepted per seed | 10.3 | 136.0 |
| Highest null rate | 0.0051 (planted) | 0.0500 (relabel) |
| Recall at precision 0.8, shared pool | 0.610 | 0.975 |

- **Holm alone is not calibrated** under misfit: it accepts 30.56% (complete null) and 28.70% (planted null).
- **The graph-only baselines fall short.** At precision 0.8 they recall AMIE 0.272, TransE 0.325 / 0.311, RotatE
  0.285 / 0.289 and ComplEx 0.249.
- **Against the pre-registered expectation:**
  - every null ≤ 5% in both dictionaries: **met**, with the oracle's relabel null exactly at the inclusive ceiling;
  - oracle precision ≈ 0.86 and recall ≈ 0.59: **met** (0.864 / 0.610);
  - misfit ≈ 0.68 / 0.07: precision higher, recall lower (0.742 / 0.038).
- The calibration does not decide L4a or L4b.

## Limitations
- No real-data result exists yet.
- The CPU smokes on T4 and T5 were near chance: the C2-row decoder's out-of-fold R² ≈ 0, and 0 edges were accepted in
  the S9 smoke. The pre-registration says its "no learn claim" readings are likely.
- A relabel null at the ceiling makes relation identity the weakest point. Claims about *which* relation an edge
  carries are secondary.
- The nulls count true but uncurated edges as false, so they are upper bounds.

## Next steps
- **Queued:** 100 jobs (GPU extraction, CPU erased-edge runs, MeSH placement, TaxoExpan, reports) in the
  54.4995–54.49959 band. The estimates are ≈ 1.85 GPU-h and ≈ 8.6 CPU-h for the main 78 jobs, plus ≈ 3.7 CPU-h for S9.
- E13's stage-1 gate L4 reads the L4a / L4b outcome as written.

---

# Sources

- Report: `reports/R10-self-learned-semantics.md` (E10.0, E10.1, "E10 baselines for claim D", "E10.9").
- Runs: `experiments/e10-self-semantics/runs/{e10.0-v3,e10.0-baselines-v2,e10.1-v1,e10.1-baselines-v2,e10.9-v1}/`.
- E10.L: `experiments/e10-self-semantics/e10-learn/preregistration.md` (§§1–12, amendments 1–2), `README.md`,
  `runs/synthetic-v1/{prior,truth}/report.md`, `smoke/README.md`; job files `.jobs/e10l-*.json` (read only, 100 pending).
- Ledgers and plans: `manuscript/claims.md` row E10.D1; `manuscript/draft.md` §5.11 and contribution 7;
  `resources/plan-improvement/experiments.md` rows E10.2, E10.9 and decisions 52–53;
  `resources/plan-improvement/execution.md` (E10.9b design, E10.2 placement).
- Summaries: `manuscript/update-2026-10-12.md` (E10-1 to E10-3); `manuscript/status-2026-10-08.md` §3.7;
  `manuscript/update-2026-10-15.md` (queue estimate).
- Commits: `c446682` (E10.0), `e72258e` (E10.1), `0a38784` / `c0f9929` (baselines), `decefae` (E10.9a
  pre-registration), `8ee946f` (E10.9a results), `0bdcbff`, `88c4351`, `ded66ec`, `e9a83b0` (E10.L).
