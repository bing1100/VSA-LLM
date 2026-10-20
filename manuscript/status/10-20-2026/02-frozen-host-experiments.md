# 02 — Frozen-host experiments: E1, E2 (with claim F.1) and E3 on frozen GPT-2

**Overview.** These three experiments (2–4 October 2026) moved the mechanisms that passed on planted data (file 01,
exploration 01.5) onto real data, with a frozen GPT-2 as the target. E1 (02.1) asks whether a context query makes
composed vectors follow GPT-2's states for polysemous words in real sentences, and whether it picks the right sense.
E2 (02.2) asks which edge weighting and which binding operator best compose WordNet and MeSH concepts into GPT-2's
concept vectors. It also carries claim F.1, the verdict on the project's first idea of fitting HRR compositions into a
frozen model's rows. E3 (02.3) asks whether the split test recovers WordNet senses and relation subtypes that we merged
on purpose. E2's gate passed only in part, and E1's and E3's gates failed. These results moved the programme to joint
training, and the paper uses them as motivation, not as claims.

**Terms used throughout.**
- **VSA (vector-symbolic architecture):** stores symbolic structure in fixed-width vectors by *binding* (combining two
  vectors into one that resembles neither) and *bundling* (adding vectors). **HRR (holographic reduced
  representation):** the VSA we use; it binds by circular convolution.
- **Frame:** a concept's list of (relation, filler) edges from the ontology. **Atomic:** a learned dictionary vector for
  a filler. **Static bundle (M0):** the normalised sum of a concept's bound edges, all weighted equally.
- **Frozen GPT-2:** GPT-2 with its weights never changed. **Anchor:** the GPT-2 hidden state we try to predict for a
  concept or a word occurrence; *centred* anchors have the training mean removed.
- **Mechanisms:** **M1**, attention over a concept's edges, steered by a context *query*; **M2**, edge weights computed
  from the concept's own frame through a rank-k form, so a new concept needs no parameters of its own; **M3, the split
  test**, which splits a dictionary vector whose usages pull it in conflicting directions.
- **Edge weightings** (the "mapping" column in the E2 run tables): *binary* (all weights 1, i.e. M0); *salience* (the
  feature-salience model of experiment 01b); *induced* (M2 at rank k); *hybrid* (induced plus a penalised per-concept
  residual, which a new concept lacks).
- **Operators** (how a relation acts on a filler): *HRR*; *HRR started at the identity*; *diagonal* (elementwise
  product); *tied low-rank* (rank 1, matched in parameters to HRR); *orthogonal*; *fixed random HRR* (frozen at its
  random start, a control); *untyped* (no binding, a plain sum). The F.1 table also uses *additive* (= untyped),
  *offset* (adds a relation vector) and *rotated diagonal* (diagonal binding inside a fixed random rotation).
- **MeSH:** Medical Subject Headings, the 2026 release. **Protocol 2:** the corrected evaluation adopted after the
  30 September audit (candidate named in advance, mid-rank ties, shared candidate sets, every shuffled label changed,
  capacity-matched controls; file 01 gives details).
- **M1 queries:** the *local-context query* (code P1) is the mean of the sentence's token embeddings up to the word; the
  *host-state query* (code P2) is GPT-2's own layer-3 state at the word.
- **Arm codes of the later joint-training experiments** (used only in "Next steps"): **C5**, the full method; **C2**, a
  free per-term lookup table with the same parameter budget; **C0**, the same model trained from scratch without the
  method; **C5dev**, C5 with the split test switched on.
- **Metrics:** MRR (mean reciprocal rank); VE (variance explained, 1 − squared error / target variance; 0 means as good
  as predicting the mean, negative means worse); ARI (adjusted Rand index; 0 = chance, 1 = perfect agreement); MFS (the
  most-frequent-sense baseline: always guess WordNet's first sense); kNN overlap (share of a concept's 10 nearest
  neighbours recovered).
- **Statistics and signs:** 3 seeds (11, 22, 33); 95% Student-t intervals, paired by seed, in brackets. A difference
  "A − B" is positive when A scores higher, and higher is better, except for the false-split rate. Loss changes in "Next
  steps" are relative to a baseline, and **negative means better**.

---

# 02.1 — E1: contextual composition against frozen GPT-2 states (gate failed)

**Status:** done (gate failed; final) · **When:** run v1 2026-10-03; clean re-run v2 2026-10-04 · **Sources:**
`experiments/e1-contextual-composition/e1-gpt2.yaml`; `experiments/e1-contextual-composition/runs/v2/report.md` and
`summary.json`; `experiments/e1-contextual-composition/runs/v1/SUPERSEDED.md`;
`src/vsa_embed/experiments/e1_contextual.py` (definitions); `resources/plan-improvement/experiments.md` §E1;
`resources/plan-improvement/gates.md` (D1 and its clean re-run); `manuscript/update-2026-10-12.md` (E2-3);
`manuscript/claims.md` (H-C.1, H-C.2); `manuscript/paper-readiness.md`

## Research gap
On planted data, the context query let M1 recover context-dependent meanings that a static bundle could not (01.5).
Polysemous words in real sentences are the natural test. Does a query make the composed vector follow a word's meaning
in each sentence, and does the attention land on the right sense? This is hypothesis H-C, and E1 was its cheapest
real-data test before the costly joint-training runs.

## Intuition
The union frame of "bank" holds the edges of the river sense and of the money sense. A query built from the sentence
can weight the matching sense's edges, so the composed vector should move toward GPT-2's state for that occurrence. The
attention mass per sense then doubles as a sense prediction.

## Methodology
- **Data:** SemCor sense-tagged sentences. A lemma qualifies if ≥ 2 of its WordNet senses have ≥ 10 sentences each (at
  most 40 sentences per sense and 8 senses per lemma).
- **Target:** GPT-2's centred hidden state at the word's last subtoken, at layer 12. A target-only audit chose the layer
  by between-sense variance share (0.086, 0.086 and 0.092 at layers 6, 9 and 12).
- **Concept:** the lemma's union frame over its senses.
- **Learners:**
  - M0 per lemma; M0 given the gold sense's frame (an upper bound);
  - M1 without a query; M1 with the local-context query; M1 with the host-state query (the pre-named primary);
  - the primary with the diagonal, fixed random HRR or untyped operator;
  - attention over free per-sense vectors with the host-state query (seen lemmas only; it has no vectors for new ones).
- **Splits:** 20% of lemmas held out entirely; 20% of the other lemmas' sentences held out.
- **Metrics** (on held-out lemmas): VE within each lemma (the denominator is the spread of that lemma's targets); context
  MRR within each lemma (the rank of each sentence's own target among the lemma's targets); sense accuracy of the
  attention mass against MFS.
- **Gate (pre-registered):** M1 beats M0 on VE and on context MRR with intervals above 0, *and* its sense accuracy beats
  MFS.

## Experiment
GPT-2 only; 690 lemmas and 45,470 sentences; composer width 256; 4,000 steps, batch 256, learning rate 0.003; one
RTX 3090. Run v1 (3 October) started with an unrelated uncommitted edit in the tree, so we re-ran it unchanged from
clean commit `ccb9204` as v2 (4 October). We cite v2.

## Results
Held-out lemmas, run v2:

| Learner | VE | Context MRR | Sense accuracy |
|---|---:|---:|---:|
| M0 per lemma | −3.5213 | 0.0720 | — |
| M0, gold sense (upper bound) | −3.5124 | 0.0749 | — |
| M1, no query | −3.4878 | 0.0720 | 0.2382 |
| M1, local-context query | −3.5964 | 0.1034 | 0.2339 |
| **M1, host-state query (primary)** | **−2.8613** | **0.1230** | 0.2477 |
| — with the diagonal operator | −2.8019 | 0.1231 | 0.2510 |
| — with fixed random HRR | −3.1223 | 0.1224 | 0.2396 |
| — untyped | −3.1110 | 0.1210 | 0.2521 |
| MFS baseline | — | — | **0.4920** |

Primary contrasts (M1 with the host-state query − M0 per lemma):
- VE +0.66 [+0.36, +0.96]: passes.
- Context MRR +0.051 [+0.050, +0.052]: passes.
- Sense accuracy − MFS −0.24 [−0.29, −0.20]: **fails**.

**E1 gate: failed.** Run v1 gave the same picture (VE +0.69 [+0.42, +0.96]; sense accuracy − MFS −0.25 [−0.29, −0.21]).

Further readings:
- **Operators tie.** The three alternatives lie within 0.002 context MRR of HRR (means only; the run reports no paired
  intervals for them).
- **Knowing the gold sense barely helps M0** (VE −3.5124 vs −3.5213), so GPT-2's layer-12 states follow a sense's
  WordNet frame only weakly.
- **Free per-sense attention** (seen lemmas only) explains far more variance than M1 with the host-state query
  (VE −0.2391 vs −2.3247) but ranks contexts worse (context MRR 0.2995 vs 0.3888).
- **Hypothesis H-C:** M1 explains more variance than M0, but it does not beat free per-sense attention on variance,
  and its attention does not align with senses. `manuscript/paper-readiness.md` labels both parts of H-C refuted on
  frozen anchors.

## Limitations
- VE is negative for every learner: each prediction is worse than the lemma's mean target. The gains are relative, and
  frozen GPT-2 states are a poor target (as in E2).
- The host-state query is GPT-2's own layer-3 state at the same word. Part of its gain may come from that state carrying
  information about the layer-12 target, not from choosing edges. No control in the run separates the two; this is our
  reading, not a tested result.
- One host. The plan also named Qwen2.5-0.5B, which did not run.
- The operator ablation reports means only.
- `manuscript/draft.md` §5.2 still marks E1 "pending", and `manuscript/claims.md` (dated 3 October) lists H-C.1 as
  partial and H-C.2 as pending. Both predate the gate decision.

## Next steps
- **Done:** M1, in its local-context form, stayed in the full method (C5) for joint training. `gates.md` carries E1
  forward as evidence that context helps, not as a sense-alignment claim.
- **Not started:** sense alignment inside jointly trained models (E5.2). E5 has a pre-registration but no runs; it
  needs checkpoints from the committed 125M from-scratch run (D4.1).
- **Closed:** no sense-alignment claim on frozen hosts.

---

# 02.2 — E2: edge weighting × operator frontier on frozen GPT-2, and the F.1 verdict

**Status:** done (gate partly passed; final). Claim F.1: refuted · **When:** E2 run v2 2026-10-02; F.1 decided from the
protocol-2 re-runs of 2026-09-30 · **Sources:** `experiments/e2-mapping-operator-frontier/runs/v2/report.md`,
`manifest.json`; `experiments/e2-mapping-operator-frontier/e2.yaml`; `src/vsa_embed/experiments/e2_frontier.py`;
`resources/plan-improvement/gates.md` (D2); `manuscript/claims.md` (F.1–F.3, E2.1–E2.4); `manuscript/draft.md` §5.3,
§5.13, Appendix A; `manuscript/update-2026-10-12.md` (E2-1, E2-2, E4-2)

## Research gap
On planted data, induced edge weights (M2) transferred to new concepts where free weights could not (01.5). On real
ontologies, does M2 beat simpler weightings, and does any binding operator beat a fixed random one or no binding? The
answers chose the operator for E3 and for joint training. The project also needed a verdict on its first idea, fitting
HRR compositions into a frozen model's rows (F.1).

## Intuition
If edge importance follows a low-rank pattern that relation and filler vectors can predict, induced weights should beat
equal weights on new concepts. If binding carries role information that the host uses, a learned operator should beat
a fixed random one and an untyped sum.

## Methodology
- **Concepts:** 6,000 WordNet synsets aligned to single GPT-2 tokens, and 6,000 MeSH 2026 descriptors with scope notes.
- **Anchor:** GPT-2's final-layer state at the term's last subtoken after "Definition: {gloss} Term: {term}", centred
  on the training mean and normalised.
- **Composer:** width 256, projected to GPT-2's 768; 600 steps, batch 1,024, learning rate 0.01, MSE plus cosine loss;
  training budgets of 25%, 50% and 100%.
- **Grid:** 7 weightings (binary; salience; induced at k = 4, 8, 16, 32; hybrid at k = 8) × 7 operators, plus a
  label-shuffled control per operator for the primary weighting (induced, k = 8). Shuffles change every label.
- **Splits:** node-disjoint; 20% test, 10% validation.
- **Metric:** test MRR of each concept's prediction among all test concepts (one shared candidate set, ties at
  mid-rank). We select cells on validation MRR only.
- **Gate (pre-registered):** induced ≥ salience and induced > binary on held-out concepts; the hybrid residual helps seen
  but not held-out concepts; the best operator goes forward, and a tie with fixed random is left to the joint-training
  operator ablation.
- **Claim F.1:** "fitting HRR (circular-convolution) composition to a frozen host's embedding rows transfers better than
  capacity-fair operators". We judge it on the protocol-2 results of 01a–01c (file 01, 01.2–01.4).

## Experiment
One RTX 3090 on the replacement machine (`bhux-tiny`), clean commit `07a7315`, 2 October 2026; 49 cells per ontology,
2.29M–3.39M parameters each. Run v1 came from the faulty-RAM machine and is marked stale.

## Results
**Absolute fit is poor:** test MRR is 0.033–0.040 on WordNet and 0.067–0.076 on MeSH across all 49 cells, and VE is
negative in every cell.

**Gate contrasts** (difference in test MRR):

| Contrast | MeSH | WordNet | Gate item |
|---|---|---|---|
| induced − salience | +0.0065 [−0.0021, +0.0151] | +0.0023 [−0.0057, +0.0102] | ≥ 0: passed (point estimate) |
| induced − binary | +0.0062 [−0.0032, +0.0156] | +0.0028 [−0.0084, +0.0140] | > 0: **failed** |
| hybrid − induced, held-out | +0.0003 [−0.0017, +0.0023] | +0.0011 [−0.0068, +0.0090] | no held-out gain: as expected |
| selected − fixed random | −0.0001 [−0.0054, +0.0052] | −0.0015 [−0.0094, +0.0064] | no separation |
| selected − untyped | +0.0011 [−0.0036, +0.0059] | −0.0005 [−0.0037, +0.0027] | no separation |

- **Validation picks:** MeSH, induced k = 4 with diagonal; WordNet, binary with tied low-rank.
- **Relation labels carry a small signal.** True − shuffled labels is positive for all 7 operators on both ontologies.
  The interval excludes 0 for WordNet HRR (+0.0109 [+0.0042, +0.0176]) and, on MeSH, for HRR at identity
  (+0.0057 [+0.0035, +0.0080]), tied low-rank (+0.0061 [+0.0043, +0.0078]), fixed random HRR
  (+0.0143 [+0.0079, +0.0208]) and untyped (+0.0045 [+0.0030, +0.0059]).
- **E2 gate: partly passed** (non-inferior to salience; does not beat binary).
- **Decisions:** E3 uses HRR at identity, the best one-vector-per-relation family on WordNet validation (0.0595 vs
  0.0596 for the overall pick). Joint training keeps HRR for now.

**Claim F.1: refuted.** Protocol-2 outcomes behind the verdict:

| Study | Outcome |
|---|---|
| 01a | additive kNN overlap 0.139 > HRR 0.107 |
| 01b | HRR 0.228 vs additive 0.225 (+0.0023 [−0.064, +0.069], 1/3 wins); diagonal 0.254 best |
| 01c.3 | HRR − offset cosine +0.004 [−0.008, +0.017]; the control ≥ HRR |
| 01c.4 | HRR − control −0.001 [−0.007, +0.004] |
| 01c.5 | diagonal 0.360 > rotated diagonal 0.341 > HRR 0.324; the recorded low-rank lead (0.439) was leakage and drops to 0.338 |

E2 gives the same answer for frozen GPT-2 hidden states: no operator separates from fixed random or untyped.

## Limitations
- Low absolute scores limit power; operator contrasts have interval half-widths of 0.003–0.008 MRR (`draft.md` §5.13).
- One host and one prompt template. The prompt contains the gloss, so targets partly encode definition text.
- The planned grid also named unitary HRR and a bounded-residual operator; neither ran.
- Contrasts are reported one by one, without multiple-comparison correction.
- `update-2026-10-12.md` (E2-1) gives the weighting-versus-bundle gain as "+0.006 / +0.002"; the run reports +0.0062
  (MeSH) and +0.0028 (WordNet).
- For F.1, 01a was not re-run under protocol 2, and the 01b and 01c re-runs carry the caveats given in 01.3 and 01.4.

## Next steps
- **Done:** the operator question in joint training. In the from-scratch operator screen (50M parameters, 3 seeds), all
  operators tie. Loss change after held-out terms against the free table C2 (negative = better): the full method C5
  with HRR −0.18%, fixed random −0.25%, translation (the relation vector is added) −0.13%, untyped −0.21%, all
  significant. The project therefore says "compositional parameter sharing", not "VSA binding", in every regime
  (`update-2026-10-12.md`, E4-2).
- **Closed:** frozen-host row insertion (F.1) is off the table; the programme moved to joint training.

---

# 02.3 — E3: the split test on real WordNet (gate failed)

**Status:** done (gate failed; final) · **When:** 2026-10-04 · **Sources:**
`experiments/e3-developmental-wordnet/e3.yaml`; `experiments/e3-developmental-wordnet/runs/v2/report.md`,
`summary.json`, `metrics.jsonl`, `cards.json`; `experiments/e3-developmental-wordnet/runs/hrr-v1/STALE.md`;
`src/vsa_embed/experiments/e3_developmental.py`; `resources/plan-improvement/experiments.md` §E3;
`resources/plan-improvement/gates.md` (D3); `manuscript/update-2026-10-12.md` (E2-4);
`experiments/e4-small-lm/analysis/scale-v1-toolkit/report.md`; `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md`;
`manuscript/update-2026-10-15.md` §7; `resources/plan-improvement/execution.md`

## Research gap
The split test (M3) found planted double meanings on synthetic data (01.5). Real dictionaries merge senses and relation
subtypes under one label. Can M3 discover them from conflicting gradients when the targets are a real model's vectors?
Hypothesis H-D says yes, with ARI above random splits and above uniform enlargement at matched parameters, and ≤ 5%
false splits. It is refuted if uniform enlargement does as well per added parameter.

## Intuition
If one vector serves two meanings, its usages pull it in opposite directions: strong per-usage gradients that cancel.
The split test detects this and splits the vector along the main direction of disagreement. Merging WordNet senses on
purpose tells us the right answer.

## Methodology
- **Collapse:** every synset filler becomes an atomic for its first lemma name, so `bank.n.01` and `bank.n.09` become
  one "bank" atomic. Relations: {part, member, substance} meronym → meronym; the three holonyms → holonym; {hypernym,
  instance hypernym} → hypernym.
- **Targets and splits:** as in E2 (6,000 WordNet concepts, frozen GPT-2 anchors, node-disjoint); operator HRR at
  identity, chosen by E2.
- **Policies:** none; uniform enlargement (wider atomics, parameters matched to M3's final count); random splits at
  matched count; a coherence-only screen without the test; the M3 default (permutation null, parent fallback with synced
  parents, merging); for relations also a small pool of "stem-cell" relation experts with sparse routing (the dossier's
  01d design).
- **Metrics:** split precision and recall against the merged atomics; false-split rate on single-sense atomics; ARI of
  the training-usage partition against the original synsets; held-out MRR and its change per added parameter.
- **Gate (pre-registered):** ARI above random splits and uniform enlargement (intervals above 0), false-split rate
  ≤ 5%, and better held-out MRR per added parameter than uniform enlargement.

## Experiment
GPT-2; composer width 256; schedule of 300 warm-up, 900 growth and 300 final steps; growth budget 5%; 199 permutations,
p < 0.01; one RTX 3090, clean commit `797ceaf`. Each seed has 15–17 merged atomics and 226–242 single-sense ones, and 3
merged relation labels against 8 others. Run `hrr-v1` came from the faulty-RAM machine and is marked stale.

## Results
**Atomics** (mean of 3 seeds):

| Policy | Splits | Precision | Recall | ARI | False-split rate | Test MRR |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.0 | — | 0.000 | — | 0.000 | 0.0336 |
| uniform enlargement | 0.0 | — | 0.000 | — | 0.000 | 0.0349 |
| random splits | 29.3 | 0.057 | 0.102 | 0.029 | 0.084 | 0.0269 |
| coherence only | 233.0 | 0.031 | 0.460 | 0.009 | 0.228 | 0.0359 |
| **M3** | 70.7 | 0.066 | 0.291 | 0.014 | **0.185** | 0.0347 |

- M3 − uniform enlargement, test MRR: −0.0002 [−0.0076, +0.0072]. M3 − none: +0.0011 [−0.0048, +0.0069].
- M3 − random ARI could not be computed (ARI is undefined for random splits in one seed). By point estimate, M3's ARI
  (0.014) is below random's (0.029).
- False-split rate 0.185 against the 0.05 limit.

**Relations:** no policy splits anything. Test MRR is 0.0373–0.0380 for every policy; the stem-cell pool reaches ARI
0.001.

**E3 gate: failed** for atomics and for relations. Uniform enlargement matches M3 per added parameter, which is H-D's
own refutation condition, so H-D is refuted on frozen GPT-2.

## Limitations
- The frozen targets give a weak signal (test MRR ≈ 0.035, VE ≈ −4 per seed), so gradient disagreement may reflect
  fitting noise more than sense structure.
- The planned sense-induction baselines (NP-MSSG, AdaGram) did not run, nor did the fixed-threshold split rule the
  literature re-check proposed.
- Split settings differ from E0's (growth budget 5% vs 50%; at least 4 usages per candidate vs 6), and we found no
  record of a sweep over them on WordNet.
- `manuscript/draft.md` §5.4 still marks E3 "pending", and `manuscript/claims.md` (dated 3 October) lists H-D.1 as
  partial; both predate the gate decision.

## Next steps
- **Done (closed):** M3 was dropped as a primary condition from the from-scratch screen (`gates.md`, D3).
- **Done (exploratory, seed 1):** M3 inside from-scratch joint training, as C5dev (50M parameters, 500M tokens, atomics
  only). Loss change against C0 (negative = better): all text C5dev −0.36% [−0.41, −0.32] vs C5 −0.38% [−0.43, −0.34];
  after held-out terms −0.18% [−0.38, +0.02] vs −0.05% [−0.25, +0.14]. The intervals cover evaluation windows only, not
  seed variance. The 15 October update reads this as "about the same as the method".
- **Proposed, not run:** revisiting the split test with a learnable ontology during joint training (E10.2); it awaits an
  author decision (`execution.md`).

---

# Sources

- `experiments/e1-contextual-composition/` — `e1-gpt2.yaml`, `runs/v2/` (`report.md`, `summary.json`, `manifest.json`),
  `runs/v1/SUPERSEDED.md`
- `experiments/e2-mapping-operator-frontier/` — `e2.yaml`, `runs/v2/` (`report.md`, `manifest.json`), `runs/v1/STALE.md`
- `experiments/e3-developmental-wordnet/` — `e3.yaml`, `runs/v2/` (`report.md`, `summary.json`, `metrics.jsonl`,
  `cards.json`), `runs/hrr-v1/STALE.md`
- `experiments/e4-small-lm/analysis/scale-v1-toolkit/report.md`; `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md`
- `src/vsa_embed/experiments/e1_contextual.py`, `e2_frontier.py`, `e3_developmental.py` (definitions only)
- `resources/plan-improvement/experiments.md` (§E1–E3), `gates.md` (D1, D2, D3), `formulation.md` (M1–M3),
  `proposal.md` (§2, §3), `execution.md`
- `manuscript/claims.md` (F.1–F.3, E2.1–E2.4, H-C, H-D), `manuscript/draft.md` (§5.2–5.4, §5.13, Appendix A),
  `manuscript/paper-readiness.md`, `manuscript/status-2026-10-08.md` (§3.8), `manuscript/update-2026-10-12.md`
  (E2-1–E2-4, E4-2), `manuscript/update-2026-10-15.md` (§7)
- File 01 of this digest (`manuscript/status/10-20-2026/01-mechanism-checks.md`) for experiments 01a–01c behind F.1
- Dates: manifests' `created_at` fields and `git log --date=short` on each experiment folder
