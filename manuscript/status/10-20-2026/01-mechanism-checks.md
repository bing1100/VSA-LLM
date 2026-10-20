# 01 — Mechanism checks: VSA algebra, frozen-row factorisation and planted structure

**Overview.** This file collects the project's earliest mechanism experiments, in the order they first ran.
Exploration 01.1 (experiment 00, July 2026) checks that the vector algebra stores and retrieves relations, and maps
how many it can hold. Explorations 01.2–01.4 (experiments 01, 01b and 01c, July 2026, re-checked after an
implementation audit on 30 September) test the project's original idea: that a frozen GPT-2's vectors can be rebuilt
by binding WordNet relations with HRR, so that new words could get vectors without training. That idea failed at
every step. Relation labels carry some signal, but HRR never beats simpler operators once the evaluation is fair.
Exploration 01.5 (E0, 30 September) opens the joint-training programme. On synthetic data with planted structure,
the context query, the ontology-induced edge weights and the split test each recover what they were built to
recover (gate G1 passed). File 02 takes these three mechanisms to real data.

**Terms used throughout.**
- **VSA (vector-symbolic architecture):** a way to store symbolic structure in fixed-width vectors by *binding* (combining
  two vectors into one that resembles neither) and *bundling* (adding vectors).
- **HRR (holographic reduced representation):** the VSA we use; it binds by circular convolution. *Unitary HRR*
  restricts the relation vector to a unit-magnitude spectrum, so unbinding is exact. *MAP* binds by elementwise
  multiplication.
- **Frame:** a concept's list of (relation, filler) edges from the ontology, e.g. (*is a*, *bird*). **Atomic:** a learned
  dictionary vector for a filler. **Static bundle (M0):** the normalised sum of a concept's bound edges.
- **Operators** (how a relation acts on a vector): *additive* or *untyped* (no relation, a plain sum); *offset* (adds a
  relation vector); *diagonal* (elementwise product with a relation vector); *low-rank*; *HRR*; and *fixed random* (an
  operator frozen at its random start, used as a control).
- **Frozen host:** a pretrained model whose weights we never change (here GPT-2). **Anchor:** the host vector we try to
  predict for a concept: an input-embedding row (*static*) or a hidden state after a prompt (*contextual*). *Centred*
  anchors have the training mean removed.
- **Mechanisms of the joint-training programme:**
  - **M1:** attention over a concept's edges, steered by a context *query*;
  - **M2:** edge weights computed from the concept's own frame through a rank-k form, so a concept never seen in
    training needs no parameters of its own (*induced* weights; *free* weights are one learned row per concept);
  - **M3, the split test:** splits a dictionary vector whose usages pull it in conflicting directions.
- **Gate G1:** the pre-registered go/no-go check that E0 had to pass before the real-data experiments E1–E3.
- **Protocol 2:** the corrected evaluation adopted after the 30 September audit. The candidate is named in advance or
  chosen on validation data; ties are ranked at mid-rank; all methods share one candidate set; shuffled controls change
  every label (a *derangement*); and controls are matched in capacity.
- **Metrics:** MRR (mean reciprocal rank); kNN overlap (share of a concept's 10 nearest neighbours recovered); cosine;
  ARI (adjusted Rand index; 0 = chance, 1 = perfect agreement); ECE (expected calibration error).
- **Statistics and signs:** unless stated, values are means over 3 seeds with 95% Student-t intervals in brackets. A
  difference "A − B" is positive when A scores higher. Higher is better for every metric here except ECE, MSE and the
  false-split rate.

---

# 01.1 — VSA algebra and capacity frontier (experiment 00)

**Status:** done (deferred milestones never run) · **When:** runs 2026-07-17; committed 2026-08-05; re-executed
bit-for-bit by the audit on 2026-09-30 · **Sources:** `experiments/00-capacity-and-algebra/README.md`;
`experiments/00-capacity-and-algebra/runs/smoke/report.md`; `experiments/00-capacity-and-algebra/runs/realistic/report.md`;
`experiments/00-capacity-and-algebra/realistic.yaml`; `resources/vsa-understanding/experiments/00-capacity-and-algebra/README.md`;
`resources/plan-improvement/audit.md` (F4, F6, F10, F12); `resources/plan-improvement/proposal.md` §2

## Research gap
Every later experiment stores a concept's relations as a bundle of bound (relation, filler) pairs in one vector. We did
not know how many pairs fit at a given width before readout fails, whether the choice of algebra matters, or whether a
readout can flag its own errors. Without these numbers, a later failure could be a capacity failure rather than a
failure of the idea.

## Intuition
Binding a relation with a filler gives a vector nearly orthogonal to both. Several bound pairs can share one vector,
and each filler comes back by unbinding with its relation and picking the nearest candidate ("cleanup"). Interference
grows with the number of pairs and shrinks with width, so a capacity frontier should exist. The research dossier
expected unitary HRR, which inverts exactly, to beat ordinary real HRR. We also expected the gap between the best and
second-best cleanup scores to predict errors, which would allow calibrated abstention.

## Methodology
- **Algebras:** real HRR (circular convolution by FFT), unitary HRR and MAP. Unit tests check that FFT and naive
  circular convolution agree in value and gradient.
- **Task:** bundle role–filler pairs, unbind each queried role, and rank the true filler among 512 candidate atomics.
  Metrics: top-1 accuracy, MRR, cosine margin.
- **Smoke sweep:** width 128 or 512 × bundle size 4, 16 or 64 × noise σ 0 or 0.1 × 3 algebras × 2 seeds, with 32
  queries per condition.
- **Realistic sweep (milestone 2):** 3 algebras × width 256 or 512 × mean candidate cosine 0.0 or 0.3 × three degree
  profiles = 36 conditions. Degree is the number of pairs per bundle: fixed at 16; Poisson with mean 16; or
  heavy-tailed (power law, 4–64). A Platt calibrator maps the observable top-1/top-2 similarity gap to a confidence. We
  fit it on seeds 101–103 and evaluate on seeds 201–203.
- **Gates.** The dossier promotes one default algebra only if it reaches ≥ 95% one-hop recovery at the target load,
  abstention error ≤ 5%, and a memory or latency advantage over exact retrieval. The run checks ECE ≤ 0.05 and held-out
  selective accuracy ≥ 0.95 per condition.

## Experiment
CPU only. The smoke sweep ran 72 trials. The realistic sweep recorded 39,360 query observations (12 memories per seed,
512 candidates per query).

## Results
**Smoke.** At width 512 and bundle size 16, all three algebras reach 100% top-1, except unitary HRR (96.9%). At width
128 and bundle size 16, top-1 falls to 45–56%. At bundle size 64 no algebra reaches 50%, even at width 512 (best
0.484). Noise σ 0.1 matters little; interference between bundled pairs causes the failures.

**Realistic sweep** (each row averages over the other factors):

| Factor | Level | Top-1 | ECE |
|---|---|---:|---:|
| Width | 256 | 0.632 | 0.046 |
| | 512 | 0.894 | 0.019 |
| Degree | fixed 16 | 0.847 | 0.020 |
| | Poisson 16 | 0.819 | 0.026 |
| | heavy-tailed | 0.622 | 0.051 |
| Algebra | real HRR | 0.755 | — |
| | unitary HRR | 0.766 | — |
| | MAP | 0.767 | — |

- Candidate correlation 0.3 lowers top-1 from 0.799 to 0.726 and accepted coverage from 0.748 to 0.598.
- At width 512, with fixed or Poisson degree and uncorrelated candidates, every algebra reaches 0.985–0.995 top-1.
  With heavy-tailed degree at width 512 it falls to 0.713–0.830.

**Gates.** ECE ≤ 0.05 holds in 29/36 conditions; held-out selective accuracy ≥ 0.95 in 17/36; both in 15/36. No
algebra meets the promotion gate, so none is promoted. The prediction that unitary HRR beats real HRR on bundled
cleanup is not supported.

## Limitations
- Atomics are synthetic and at most mildly correlated (mean cosine 0.3); learned embeddings may be more correlated.
- The smoke sweep is diagnostic only: 2 seeds, 32 queries, and, at noise 0, only as many distinct queries as the bundle
  size (audit F6).
- The selective-accuracy gate is decided on point estimates; 19/36 Wilson intervals contain 0.95. The threshold rule
  is also confounded with the claim that heavy-tailed degree is *the* failure regime (audit F6).
- The run's decision text is hard-coded, and memory and latency against exact retrieval were never measured (audit F10).
- The realistic run wrote no manifest, and all runs predate the first git commit (audit F4).
- In this library MAP is the same operator as *diagonal* (audit F12), so "MAP" here and "diagonal" later are one thing.

## Next steps
- **Adopted as a design rule:** cap or shard high-degree frames, and build no memory infrastructure (proposal §2).
- **Closed:** generalised HRR, path depth, sharding, learned cleanup and degree-aware calibration were deferred and never
  run.
- **Done later, elsewhere:** in jointly trained models, unbinding a word's vector recovers its related words at 0.99 MRR
  on held-out words of track T5, a synthetic company glossary (`manuscript/update-2026-10-12.md`, B-1).

---

# 01.2 — Rebuilding frozen GPT-2 rows from WordNet (experiment 01 and its pilot 01a)

**Status:** done (pilot gate failed; closed) · **When:** runs 2026-07-17; committed 2026-08-05; pilot re-executed
bit-for-bit by the audit on 2026-09-30 · **Sources:** `experiments/01-ontology-factorization/wordnet-gpt2-pilot.yaml`;
`experiments/01-ontology-factorization/runs/smoke/report.md`;
`experiments/01-ontology-factorization/runs/wordnet-gpt2-pilot/report.md`;
`resources/vsa-understanding/experiments/01-ontology-factorization/README.md`; `resources/plan-improvement/audit.md`
(F4, F5c); `manuscript/claims.md` (F.1)

## Research gap
The project began with a hope. If a pretrained model's embedding rows are roughly HRR compositions of ontology
relations, a new word could get a row from its relations with no gradient update. The research dossier called this
*zero-gradient insertion* (its experiment 02). It needs frozen rows to carry HRR-composable structure, and we had not
checked whether they do.

## Intuition
Give each ontology filler a shared vector, bind it with its role, and sum the results into a predicted row. If GPT-2's
rows reflect WordNet, a held-out word's row should follow from its recipe. Typed binding should beat an untyped sum,
because it records which role each filler plays.

## Methodology
- **Synthetic sandbox:** a hidden HRR teacher generates the target rows. Held-out recipes contain only (role, value)
  atomics seen in training. Controls: bag of factors, shuffled structure, nearest recipe, train-mean row.
- **Pilot 01a:** frozen GPT-2 input rows (revision `607a30d`) and WordNet 3.0. We used 1,200 lowercase single-token
  nouns that map to exactly one noun synset. A word's recipe is its lexicographer class plus four hypernym ancestors.
- **Methods:** typed HRR, typed MAP, untyped additive, shuffled structure, nearest recipe, train-mean row.
- **Splits:** three 960/240 splits. Identical recipes stay in one partition, and every held-out atomic appears in
  training. No held-out row enters the loss.
- **Metrics:** held-out row cosine and kNN overlap.
- **Gate (set in the config):** typed HRR must beat additive by ≥ 0.02 kNN overlap and the train-mean row by ≥ 0.01
  cosine.

## Experiment
VSA width 128, 300 steps, learning rate 0.02, seeds 11, 22 and 33. The sandbox also used three seeds.

## Results
**Sandbox:** typed HRR reaches 0.897 kNN overlap against 0.221 for the bag of factors (+0.676); shuffled structure
0.070, nearest recipe 0.214, train mean 0.077. Gate **passed**. An earlier version gave each role its own value IDs;
the bag could then infer the role and nearly tied (0.893 vs 0.897). One shared value vocabulary is now a required
control.

**Pilot 01a** (held-out words, mean of 3 splits):

| Method | Row cosine | kNN overlap |
|---|---:|---:|
| typed HRR | 0.331 | 0.107 |
| typed MAP | 0.417 | 0.114 |
| untyped additive | 0.442 | **0.139** |
| shuffled structure | 0.260 | 0.042 |
| nearest recipe | 0.325 | 0.099 |
| train-mean row | **0.529** | 0.043 |

Gate **failed**: HRR − additive = −0.032 kNN overlap; HRR − train mean = −0.197 cosine. Structure helps (additive 0.139
vs shuffled 0.042), but typed binding does not.

## Limitations
- Chance kNN overlap is 0.042 (audit F5c), so the shuffled and train-mean rows sit at chance. Additive's lift over
  chance is 0.097. The dossier's high-degree figure (additive 0.202) partly reflects a higher chance floor in that band.
- A shared mean direction dominates raw cosine: the train-mean row scores best on cosine and at chance on neighbours.
  The cosine half of the gate therefore says little.
- One host, one ontology, one hypernym path per word, and static rows that mix all senses of a word.
- Earlier diagnostic runs let identical recipes cross partitions and overstated every structured method.
- The pilot was not re-run under protocol 2 (only 01b and 01c were). The audit reproduced it bit-for-bit; the sandbox
  did not reproduce exactly (nondeterminism, conclusion unchanged). Reports were hand-edited after generation (audit F4).

## Next steps
- **Done:** experiment 01b replaced the single path with seven shared relations and contextual anchors (01.3).
- **Closed:** zero-gradient insertion stayed blocked. Claim F.1 was later refuted (file 02, exploration 02.2).

---

# 01.3 — Shared relation operators on a real WordNet graph (experiment 01b)

**Status:** done (protocol-2 re-run fails both criteria; closed) · **When:** Stages A and B 2026-07-17; protocol-2
re-run 2026-09-30 · **Sources:** `experiments/01b-global-local-relational-factorization/reduced.yaml`;
`experiments/01b-global-local-relational-factorization/stage-b-reduced-v2.yaml`;
`experiments/01b-global-local-relational-factorization/runs/reduced/report.md`;
`experiments/01b-global-local-relational-factorization/runs/stage-b-reduced/report.md`;
`experiments/01b-global-local-relational-factorization/runs/stage-b-reduced-v2/report.md`, `PROVENANCE-NOTE.md`,
`split_stats.csv`; `experiments/01b-global-local-relational-factorization/runs/stage-b-smoke/STALE.md`;
`resources/vsa-understanding/experiments/01b-global-local-relational-factorization/README.md` (with Errata);
`resources/plan-improvement/audit.md` (F2, F3, F5a, F6); `resources/plan-improvement/gates.md`;
`reports/R0-reproduction.md`; `manuscript/claims.md` (F.1, F.2)

## Research gap
01a gave every edge equal weight, used one nested hypernym path, and asked a fixed algebra to explain a static row. Its
failure might reflect that narrow design rather than the idea. We asked whether globally shared relation operators can
explain how a frozen host relates WordNet concepts. We also asked which operator family gives the best trade-off
between built-in bias, capacity and data.

## Intuition
Predict a target concept's anchor from a source concept's anchor through one operator per relation, shared by every
pair. If WordNet relations act as fixed transformations in host space, a compact operator should transfer to new
concepts from little data. HRR needs one vector per relation, so it should beat an identity map and a label-shuffled
copy of itself. Contextual anchors (the hidden state after "Definition: ⟨gloss⟩ Term: ⟨lemma⟩") avoid mixing senses.

## Methodology
- **Stage A (synthetic):** three teacher families (HRR, diagonal, rank-8 low-rank) × four learners (additive, HRR,
  diagonal, low-rank) × three data budgets (32, 128, 512 edges per relation) × residual width 0 or 8 × 3 seeds = 216
  conditions. Metric: held-out MSE. Gate: each teacher is best recovered by its own family in ≥ 2 of 3 cases.
- **Stage B (real):** frozen GPT-2 (`607a30d`), WordNet 3.0, seven relation types × 80 edges = 560 edges. Anchors are
  static or contextual, raw or centred. Splits are node-, token- or edge-disjoint, plus edge-disjoint at matched size.
- **Methods:** additive (identity), relation mean, offset, HRR, HRR started at the identity, diagonal, rank-8 low-rank,
  a parameter-matched low-rank, and a label-shuffled copy of each.
- **Primary endpoint (protocol 2):** target MRR on node-disjoint, contextual, centred anchors. The candidate (HRR) and
  control (additive) are named in advance. Criteria: MRR gain ≥ 0.01 with 3/3 paired wins (retrieval), and cosine gain
  ≥ 0 (reconstruction).

## Experiment
CPU, single-threaded, 250 steps, seeds 11, 22 and 33; the seeds re-split one graph sample. The protocol-2 graph has
974 nodes (the recorded run had 966). Node-disjoint splits train on 163–172 of the 560 edges and discard 276–284
crossing edges.

## Results
**Stage A:** each teacher is best fit by its own family (3/3). At 512 edges per relation, the HRR teacher gives HRR an
MSE of 0.00014 against 0.01741 for additive. Gate **passed**. Matched HRR and diagonal learners were near the noise
floor at 32 edges per relation; the low-rank learner needed 128.

**Stage B as recorded (July):** HRR 0.2713 vs additive 0.2520 (+0.0194, 3/3 seeds), reported as "retrieval pass,
reconstruction fail". The audit found the runner had picked HRR by test MRR. The seed gains give a t-interval of
[0.007, 0.032], whose lower end lies below the 0.01 margin.

**Stage B under protocol 2** (node-disjoint, contextual, centred):

| Method | Target MRR |
|---|---:|
| additive | 0.2254 |
| HRR | 0.2277 |
| HRR started at identity | 0.2279 |
| **diagonal** | **0.2540** |
| offset | 0.2096 |
| low-rank (rank 8) | 0.1926 |
| low-rank, parameter-matched | 0.1772 |
| shuffled HRR | 0.1818 |

- HRR − additive: +0.0023 [−0.0643, +0.0689], 1 of 3 wins. Retrieval criterion **failed**.
- HRR − shuffled HRR: positive in 3/3 seeds, mean +0.046 [−0.012, +0.104]. Relation labels carry some signal.
- Cosine, HRR − additive: −0.0394. Reconstruction criterion **failed**.
- On static centred anchors, additive is best (0.4689).

## Limitations
- The protocol-2 run reuses the recorded graph and seeds, so it is development evidence, not a fresh test. Its three
  seeds re-split one graph (audit F6).
- Node-disjoint training uses about 30% of the edges (audit F3).
- The criteria are proxies. The dossier's seven gate items (behavioural transfer, a second ontology, locality and
  others) were never evaluated.
- The dossier's original relation-accuracy claim was empty by construction (audit F5a).
- The manifest records a dirty tree at `f9d49d8`, because this runner recorded git state at the end of the run; the
  provenance note says the run launched from clean commit `84d1ff4`.
- The run was made on 30 September, the day the original machine's faulty RAM was found. `gates.md` asked for such runs
  to be re-checked; `reports/R0-reproduction.md` does not list this one.

## Next steps
- **Done:** experiment 01c tried to rescue reconstruction (01.4).
- **Closed:** Stages C–E (a relation-scoring sidecar, adaptation curves, behavioural confirmation) were never run; the
  programme moved to joint training.

---

# 01.4 — Reconstruction rescue on a fixed graph (experiment 01c)

**Status:** done (all three protocol-2 studies fail; closed) · **When:** studies 01c.0–01c.5 on 2026-07-17; protocol-2
re-runs of 01c.3–01c.5 on 2026-09-30 · **Sources:**
`experiments/01c-developmental-relation-discovery/runs/basis-confirmation-development-v2/report.md`;
`experiments/01c-developmental-relation-discovery/runs/taxonomy-distribution-development-v2/report.md`;
`experiments/01c-developmental-relation-discovery/runs/source-neighborhood-development-v2/report.md`;
the matching `*-v2.yaml` configs and `PROVENANCE-NOTE.md` files;
`experiments/01c-developmental-relation-discovery/runs/{smoke,development}/STALE.md`;
`resources/vsa-understanding/experiments/01c-developmental-relation-discovery/README.md` (implementation log, Errata);
`resources/plan-improvement/audit.md` (F1, F5b, F8, F9); `manuscript/claims.md` (F.1); `manuscript/status-2026-10-08.md` §3.4

## Research gap
In 01b as recorded, HRR ranked targets best but rebuilt them worst; the offset operator rebuilt them best. Row insertion
needs both. We asked whether an HRR operator that starts at the identity, adapts its basis and bounds its correction can
win on both. The planned discovery of new relations (study 01d) also required a validated operator first.

## Intuition
Most of a target anchor may be the source's own geometry. A relation offset explains a broad shift, and a small HRR
correction could add the relation-specific detail. Starting at the identity means the HRR term survives only if it helps
on held-out data. When this failed, we narrowed the task twice: to taxonomy relations only (01c.4), then to complete
parent sets (01c.5), where one source has several correct parents.

## Methodology
- **Candidate:** basis + offset + bounded residual HRR, with the correction capped after decoding into host space.
- **Controls:** offset; pure HRR; an equal-parameter diagonal control; a *rotated-diagonal* control (diagonal binding
  inside a fixed random orthogonal mix), added in protocol 2 because the basis cancels out of the plain diagonal control
  (audit F8); low-rank (rank 4, and parameter-matched); and a derangement-shuffled copy of each.
- **01c.3:** seven relations, node-disjoint, contextual centred anchors. Endpoint: cosine gain over offset, with MRR no
  more than 0.02 below pure HRR.
- **01c.4:** hypernym and instance hypernym only, trained with several positives per query. Endpoint: distribution MRR,
  ≥ +0.01 over the rotated control.
- **01c.5:** 149 (source, relation) queries kept whole with all their parents. Endpoints: query MRR and set recall@10,
  each ≥ +0.01 over the rotated control.
- **Dossier gate (nine items):** among them, a validation-selected candidate must beat offset by ≥ +0.02 cosine, beat
  its shuffled and equal-parameter controls, keep MRR, and replicate on a second ontology.

## Experiment
CPU, single-threaded, 250–300 steps, three fresh seeds per study (13/31/53, 17/37/59, 19/41/61). Graphs: 974 nodes and
560 edges (01c.3); 412 and 240 (01c.4); 385 and 301 (01c.5). The two confirmation seeds were never opened. Studies
01c.0–01c.2 are marked stale and not cited.

## Results

| Study (endpoint) | HRR candidate | Rotated control | Diagonal control | HRR − rotated control | HRR − its shuffled copy | Criterion |
|---|---:|---:|---:|---|---|---|
| 01c.3 (cosine) | 0.3831 | 0.3840 | 0.3832 | — | — | **failed** |
| 01c.4 (distribution MRR) | 0.2325 | 0.2339 | 0.2423 | −0.0014 [−0.0065, +0.0037] | +0.1652 [+0.1100, +0.2204] | **failed** |
| 01c.5 (query MRR) | 0.3244 | 0.3414 | 0.3597 | −0.0169 [−0.0521, +0.0182] | +0.0511 [+0.0191, +0.0831] | **failed** |

- 01c.3: HRR − offset cosine +0.0042 [−0.0081, +0.0166]; MRR 0.0233 below pure HRR.
- **Leakage found.** The recorded 01c.5 result, "rank-4 low-rank is best" (0.4386), came from test parents seen in
  training: 0.5704 on queries with a seen parent against 0.1705 without one. With parents held out, low-rank scores
  0.3378 and ties its own shuffled copy (0.3426).
- **Reading:** true relation labels matter (HRR beats its shuffled copy), but circular convolution adds nothing over
  elementwise relation conditioning. The 01c gate **failed**.

## Limitations
- Small graphs, with three seeds on one graph sample per study.
- In 01c.4 a query has 1.02 valid targets on average, so "distribution" MRR is close to ordinary MRR (audit F5b).
- Two of the three protocol-2 manifests record dirty trees (state recorded at the end of the run; the provenance notes
  say both launched from clean commit `5762dca`). The machine caveat of 01.3 applies here too.
- Frozen GPT-2, one ontology, one-hop transfer only. Multi-hop composition, which VSA algebra predicts it should handle
  well, was not tested here.

## Next steps
- **Closed:** study 01d and row insertion were never started; the dossier's stopping rule forbids further tuning here.
- **Done, elsewhere:** the dossier asked for a multi-hop test. In joint training, chained two-hop recall through per-word
  vectors reaches 0.88 correct, against 0.07 through one global superposed memory (`manuscript/status-2026-10-08.md` §3.4).
- **Done:** the operator became an ablation in every later experiment (file 02).

---

# 01.5 — E0: do the three mechanisms recover planted structure? (gate G1)

**Status:** done (gate G1 passed; final) · **When:** 2026-09-30 to 2026-10-01; reproduced on the replacement machine
2026-10-02 · **Sources:** `experiments/e0-synthetic-identifiability/e0.yaml`;
`experiments/e0-synthetic-identifiability/e0-parent-routing.yaml`;
`experiments/e0-synthetic-identifiability/runs/{e0-development,d02-parent-routing,d02-parent-routing-v2,d02-parent-sync}/report.md`
and `summary.json`; `manuscript/tables/e0.md`; `manuscript/draft.md` §5.1; `resources/plan-improvement/experiments.md`
§E0; `resources/plan-improvement/gates.md` (G1); `manuscript/claims.md` (E0.1–E0.6); `reports/R0-reproduction.md`;
`manuscript/update-2026-10-12.md` (E0-1)

## Research gap
After 01a–01c, the programme switched to training the composed vectors jointly with the language model. That programme
rests on three new mechanisms: the context query (M1), induced edge weights (M2) and the split test (M3). Before
spending GPU time on real data, we needed to know whether each can recover its target structure when that structure
exists, and whether it does no harm when it does not. Otherwise a later failure on real data could not be read.

## Intuition
Build small synthetic worlds that contain exactly the targeted structure: edge weights that depend on context; edge
weights from a low-rank rule; vectors secretly used with two meanings. A mechanism that fails in such a world is broken.
One that succeeds is at least capable.

## Methodology
All three tests use width 64, HRR binding, seeds 101, 202 and 303, and 95% t intervals paired by seed. The gates were
written in `experiments.md` before the run.
- **D0.1 (M1).** A teacher with 64 atomics, 6 relations, 400 concepts (80 held out) and 8 context prototypes (2 held
  out). Edge weights depend on context with strength 0.5–4; a second teacher has static weights. Learners: M0; the 01b
  salience model; M1 without a query; M1 with a query; free weights per (concept, context). Metric: cosine on held-out
  concepts in held-out contexts. *Gate:* M1 − M0 interval above 0 from strength 1 up; on the static teacher, M1 at most
  0.005 below M0.
- **D0.3 (M2).** A rank-4 bilinear teacher. Free, induced or hybrid (induced plus a penalised per-concept residual)
  weights at k ∈ {4, 8, 16, 32}; 80 of 400 concepts held out with disjoint compositions. *Gate:* induced or hybrid
  clearly above free.
- **D0.2 (M3).** Students start collapsed: 48 atomics of which 8 secretly carry two senses, or 6 relations of which 2
  hide two sub-operators; 600 concepts, 100 held out. Schedule: 1,500 warm-up, 2,000 growth and 1,000 final steps; 199
  permutations, p < 0.01; merge above cosine 0.98. Policies: none; M3 with a permutation null; M3 with an
  Anderson–Darling test; a coherence-only screen; random splits at matched count; oracle. *Gate:* precision and recall
  ≥ 0.9, ARI ≥ 0.8, false-split rate ≤ 5%.

## Experiment
CPU, single-threaded. The development run (commit `2cc5d0b`) covered all three tests. Three later D0.2 runs changed
how usages without gradient evidence are routed; the last (parent fallback with synced parents, `9986d72`) became the
M3 default. On the replacement machine, every run reproduced at 1e-12 (8,100 + 154 cells for the development run;
576 + 115 per variant run).

## Results
**D0.1: the context query recovers context-dependent composition.** Held-out cosine:

| Context strength | M0 | M1 with query | M1 − M0 |
|---|---:|---:|---|
| 0.5 | 0.596 | 0.954 | +0.3585 [+0.3355, +0.3816] |
| 1 | 0.570 | 0.964 | +0.3943 [+0.3084, +0.4802] |
| 2 | 0.536 | 0.968 | +0.4325 [+0.2275, +0.6375] |
| 4 | 0.484 | 0.945 | +0.4604 [+0.1238, +0.7969] |

On the static teacher, M1 − M0 = −0.000014 [−0.000023, −0.000004], inside the margin. M1 without a query falls from
0.884 to 0.560 as strength rises; free per-context weights reach 0.664–0.802. **Passed.**

**D0.3: induced weights transfer.** Induced k = 4: 0.976 [0.939, 1.013]; hybrid k = 4: 0.978; free: 0.598–0.696;
M0: 0.623. Larger ranks lose a little (induced 0.893 at k = 32). **Passed.**

**D0.2: the split test finds planted senses, but needs merging and a routing fix.**
- Development run: M3 reaches precision = recall = ARI = 1.0 for atomics and relations, with no false split after
  merging. **Passed.**
- Before merging, M3 splits 37.5% of the single-sense atomics (40.0% in the variants), so merging is required.
- The alternatives do worse. Anderson–Darling keeps 0.083–0.142 false splits; coherence-only recall is 0.875–0.917
  (atomics) and 0.667 (relations); random splits give ARI 0.018.
- Usages without gradient evidence were routed near chance (held-out ARI 0.046). Splitting relations therefore changed
  held-out cosine by −0.0629 [−0.0988, −0.0269] (harm).
- With the M3 default (such usages keep the parent, held at the mean of its children): atomics precision 0.963 (seed
  range 0.889–1.000), recall 1.0, ARI 1.0, false-split rate 0.008; relations 1.0 / 1.0 / 1.0 / 0. Held-out cosine change
  against no growth: atomics +0.0041 [+0.0002, +0.0081]; relations +0.0071 [−0.0021, +0.0163].

**Gate G1: passed.**

## Limitations
- The worlds are built to match the mechanisms, so these results are upper bounds.
- We chose the M3 default after seeing the development run and re-ran it on the same three seeds. It is not a fresh
  confirmation.
- Small scale: width 64, at most 600 concepts, 8 two-sense atomics, 2 two-part relations. At context strength 4 the
  interval spans +0.12 to +0.80.
- "Splitting improves held-out composition" holds only marginally, for atomics. The supported claim is "no harm".

## Next steps
- **Done:** the real-data versions ran on frozen GPT-2: E1 for M1, E2 for M2, E3 for M3. E2 passed in part; E1 and E3
  failed (file 02).
- **Done:** M3 was later tried inside from-scratch joint training (file 02, exploration 02.3).

---

# Sources

- `experiments/00-capacity-and-algebra/` — `README.md`, `smoke.yaml`, `realistic.yaml`, `runs/smoke/`, `runs/smoke-v2/`,
  `runs/realistic/`
- `experiments/01-ontology-factorization/` — `smoke.yaml`, `wordnet-gpt2-pilot.yaml`, `runs/smoke/`,
  `runs/wordnet-gpt2-pilot/`
- `experiments/01b-global-local-relational-factorization/` — `reduced.yaml`, `stage-b-reduced-v2.yaml`, `runs/reduced/`,
  `runs/stage-b-reduced/`, `runs/stage-b-reduced-v2/`, `runs/stage-b-smoke/STALE.md`
- `experiments/01c-developmental-relation-discovery/` — `*-v2.yaml`, `runs/basis-confirmation-development-v2/`,
  `runs/taxonomy-distribution-development-v2/`, `runs/source-neighborhood-development-v2/`, `runs/smoke/STALE.md`,
  `runs/development/STALE.md`
- `experiments/e0-synthetic-identifiability/` — `e0.yaml`, `e0-parent-routing.yaml`, `runs/e0-development/`,
  `runs/d02-parent-routing/`, `runs/d02-parent-routing-v2/`, `runs/d02-parent-sync/`
- `resources/vsa-understanding/experiments/{00-capacity-and-algebra,01-ontology-factorization,01b-global-local-relational-factorization,01c-developmental-relation-discovery}/README.md`
- `resources/plan-improvement/audit.md`, `gates.md`, `experiments.md`, `proposal.md`, `formulation.md`
- `reports/R0-reproduction.md`
- `manuscript/claims.md`, `manuscript/draft.md` (§5.1, Appendix A), `manuscript/tables/e0.md`,
  `manuscript/status-2026-10-08.md`, `manuscript/update-2026-10-12.md`
- Dates: manifests' `created_at` fields and `git log --date=short` on each experiment folder
