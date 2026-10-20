# 09 — The binding programme: does the store inside a term's vector hold usable structure?

**Overview.** The method stores each term's ontology entry inside the vector it adds to the language model. We asked
whether that store holds structure the model can use. The binding programme (author decision 60, pre-registered
2026-10-08) answers in three steps on the synthetic glossary track T5, with replications on other tracks. The answer is
the same at every step.
- **The store is readable (09.1).** Unbinding recovers a term's related concepts almost perfectly, and a fixed random
  operator does as well as a learned one.
- **The model does not read roles from the store (09.2), even with a trained readout (09.4).** Both pre-registered
  decisive tests are null.
- **Binding beats simpler operators only where a concept could fill two roles (09.3).** On T5 that means reverse lookup;
  on real chemistry (T4) it also means chaining.
- **The operators behave as designed after training (09.5).** The non-commutative operator keeps the order of a path.
- **Editing frames changes the readout arms' behaviour (09.6),** but binding arms do no better than the no-binding
  control.

The official reports of 2026-10-20 supersede the preliminary readout-arm numbers of 2026-10-18. Every number they share
is the same; the official reports add intervals and the loss endpoint R2.

**Terms used throughout.**
- **VSA (vector-symbolic architecture):** an algebra that stores a structured record in one vector.
- **Binding** combines a role vector (a relation, such as *part of*) with a filler vector (a related concept) into one
  vector that resembles neither. A record is the sum of its bound pairs. **Unbinding** applies the role's inverse to
  the sum and returns a noisy copy of the filler. **Cleanup** snaps that copy to the nearest known concept. Cleanup is
  *typed* when it considers only fillers seen under that relation.
- **HRR (holographic reduced representations):** the VSA we use. It binds by circular convolution and unbinds by
  circular correlation, an approximate inverse. A **unitary** role has an exact inverse.
- **Store:** a term's static frame bundle, the sum of its bound role–filler pairs.
- **C5:** the method. It composes each multi-token term's vector from its store and adds it at the term's last token
  during fine-tuning.
- **Operator controls:** **C5rf** uses a fixed random unitary binding that is never trained; **C5ut** is untyped (the
  sum of fillers, no roles); **C5tr** uses translation (filler + relation vector).
- **Baselines:** **P0**, the original pretrained model; **C0′**, the same host fine-tuned on the same text without the
  method (the fair baseline); **C2**, a free per-term lookup table with C5's parameter budget.
- **U5\* (readout arms):** C5 plus a trained head that unbinds the store (09.4). The suffix names the operator.
- **Tracks:** **T5**, a synthetic company glossary of invented words in templated text; **T4**, real chemistry (ChEBI
  entries, PubMed abstracts); **T7**, new 2026 MeSH names in PubMed; **T1**, MeSH on PubMed; **WordNet**, general text.
- **MRR (mean reciprocal rank):** for each query, 1 / the rank of the correct concept; averaged. 1 means always first.
- **Twin contrast:** the decisive role test of 09.2, defined there. Chance is 0.5.
- **Statistics.** Unless stated, contrasts use a crossed random-effects model over units × seeds (Satterthwaite t,
  two-sided 95% CI), with Holm correction within each pre-registered family. A positive accuracy or MRR difference
  means the first arm is better. For loss, a positive relative change means *higher* loss, i.e. worse.

Pre-registration: `experiments/e9-retrofit/preregistration-binding.md` (§§1–7 step 1; §13 step 2; §13.1 U5rf; §14 step 3).

---

# 09.1 — B-1: The relations stored in a term's vector can be read back

**Status:** done · **When:** pre-registered and run 2026-10-08 (T5); T4, T7, T1, WordNet reports 2026-10-19; T5 report
regenerated 2026-10-20 · **Sources:** `experiments/e9-retrofit/preregistration-binding.md` §§1–7;
`experiments/e9-retrofit/report/t5-binding/report.md`; `experiments/e9-retrofit/report/t5-qwen3-binding/report.md`;
`experiments/e9-retrofit/report/{t4,t7,t1,wordnet}-binding/report.md`; `manuscript/update-2026-10-12.md` (B-1)

## Research gap
Bound vectors decode well when every vector is random. It was untested whether the algebra survives joint training
with a language model, whose loss never asks for decoding. The standard way to read relations out of a model is a
learned linear decoder (LRE; Hernandez et al., ICLR 2024). We compared the two.

## Intuition
If training leaves the algebra intact, a term's vector stays a readable record of its entry. Unbinding a relation and
cleaning up should then return the right concept, even for terms the model never trained on.

## Methodology
- **Probe A (algebraic):** for each edge of a concept's store, unbind the relation with the arm's own method and run
  typed cleanup against the trained concept dictionary. The score is MRR per concept.
- **Probe B (LRE-style):** a ridge decoder per relation, trained on seen concepts, applied to the composed vector or to
  hidden states.
- **Held-out concepts:** 560 entries kept out of training; their vectors are composed zero-shot from the ontology.
- **Primary endpoints:** P2a, C5 minus a frequency baseline (always guess the most frequent filler); P2b, C5 − C5rf.
- **Predictions (written before running):** P2a, C5 far above the frequency baseline; P2b, C5 − C5rf < 0, because a
  learned non-unitary role adds crosstalk.

## Experiment
We probed trained checkpoints; nothing was retrained. Primary setting: SmolLM2-360M, fully fine-tuned on T5 for 50M
tokens, seeds 1–3. Replications: SmolLM2-135M (3 seeds) and Qwen3-0.6B / 1.7B (LoRA, 2 seeds). Probe-only runs: T4 and
T7 (360M, 3 seeds), T1 and WordNet (seed 1).

## Results
**T5, SmolLM2-360M, held-out concepts (560 × 3 seeds):**

| Contrast | Estimate [95% CI] | Pre-registered prediction |
|---|---|---|
| P2a: C5 − frequency | +0.7747 [+0.7644, +0.7850] | **passed** |
| P2b: C5 − C5rf | −0.0062 [−0.0157, +0.0032], p 0.123 | **failed** (no difference) |
| LRE − algebra (C5) | −0.2022 [−0.2119, −0.1926] | — |

- C5 reaches MRR 0.987 with typed cleanup; the frequency guess reaches 0.212. The fixed random operator reaches 0.993.
- Without type information, the untyped store recovers roles at only 0.083, as designed (every relation ties).
- The learned linear decoder reads less than the algebra in every arm: LRE − algebra runs from −0.1285 to −0.2129, and
  reaches −0.7760 for the control whose frames are shuffled between terms.

**Replications (C5 − frequency, held-out MRR):**

| Host / track | Seeds | Estimate [95% CI] |
|---|---:|---|
| SmolLM2-135M, T5 | 3 | +0.7791 [+0.7709, +0.7873] |
| Qwen3-0.6B, T5 | 2 | +0.7585 [+0.7201, +0.7969] |
| Qwen3-1.7B, T5 | 2 | +0.7464 [+0.7255, +0.7672] |
| SmolLM2-360M, T4 | 3 | +0.4334 [+0.3843, +0.4825] |
| SmolLM2-360M, T7 | 3 | +0.4673 [+0.4567, +0.4778] |
| SmolLM2-360M, T1 | 1 | +0.7690 [+0.7607, +0.7773] |
| SmolLM2-360M, WordNet | 1 | +0.6455 [+0.6400, +0.6510] |

- **On T4, learned binding decodes worse than fixed random binding:** C5 − C5rf −0.0623 [−0.1130, −0.0115], p 0.0333.
  P2b's prediction holds on real chemistry, though not on T5.

## Limitations
- Typed cleanup restricts candidates to fillers seen under the relation, so it supplies type information.
- The frequency baseline is weak on T5. On real tracks it is stronger, so the margins are smaller.
- Qwen3 has no untyped or translation arms. T1 and WordNet have one seed.
- Readable is not used. 09.2 tests use.
- The T5 report was regenerated from a working tree with uncommitted changes (`git_dirty: true` in its manifest). The
  binding reports are untracked in git.

## Next steps
- **Done:** step 1 on every open track.
- **Closed:** whether learning the operator helps decoding. It does not on T5 and hurts on T4.
- T1c aggregates exist under the clinical licence; this digest does not read them.

---

# 09.2 — B-2: The model does not use the stored roles by itself (pre-registered P1: null)

**Status:** done · **When:** pre-registered and run 2026-10-08; readout arms added to the report 2026-10-20 ·
**Sources:** `experiments/e9-retrofit/preregistration-binding.md` §§1, 5–7;
`experiments/e9-retrofit/report/t5-binding/report.md`; `experiments/e9-retrofit/report/t5-qwen3-binding/report.md`;
`experiments/e9-retrofit/report/t4-binding/report.md`; `manuscript/status-2026-10-08.md` §3.4

## Research gap
A readable store is not the same as a used one. On T5 the set of fillers almost identifies each term (filler sets
collide in 0% of terms), so next-token training never needs a role. We built items where only the roles differ.

## Intuition
Take two new words with the same two fillers in swapped roles. An untyped or translation store gives the twins almost
the same vector; an HRR store does not. If the model reads roles, it should answer the twins differently.

## Methodology
- **Items:** 300 pairs of invented names (A, B). A = rest + (r1, X) + (r2, Y); B = rest + (r1, Y) + (r2, X). The
  relation pairs are *depends on / part of* and *owned by / approved by*, 150 pairs each. The names are inserted at
  evaluation, so their vectors are composed zero-shot.
- **Twin contrast:** for each relation and prompt template, z = (s_A(X) − s_A(Y)) − (s_B(X) − s_B(Y)), signed so that
  z > 0 when the twins' preferences follow their frames. A pair scores the share of its 4 comparisons with z > 0. Chance
  is 0.5. The scores s are PMI against a null prompt in which the term is replaced by "this" (`choice` items).
- **Primary P1:** C5 − C5ut and C5 − C5tr, Holm over the two. Power: 80% to detect ≈ 0.04.
- **Sources:** `own` (the model's own store), `none` (rows zeroed), `swap` (each twin reads its partner's frame).
- **Prediction:** C5 0.55–0.65, C5ut and C5tr ≈ 0.50, and C5 `own − swap` > 0.

## Experiment
SmolLM2-360M on T5 with every arm (3 seeds); SmolLM2-135M (P0, C0′, C2, C5; 3 seeds); Qwen3-0.6B and 1.7B (same four;
2 seeds). On T4 we used 715 natural role-ambiguous items instead of twins (360M, 3 seeds).

## Results
**P1, T5, SmolLM2-360M (300 pairs × 3 seeds):**

| Contrast | Estimate [95% CI] | Holm p | Prediction |
|---|---|---:|---|
| C5 − C5ut | −0.0036 [−0.0155, +0.0083] | 1 | **failed** |
| C5 − C5tr | +0.0017 [−0.0138, +0.0172] | 1 | **failed** |

- **Every model sits near chance (0.484–0.498):** C5 0.494, C5ut 0.498, C5tr 0.492, C5rf 0.494, C2 0.487, C0′ 0.484,
  P0 0.492.
- **The swap test fails too:** C5 scores 0.494 with its own frames and 0.506 with its partner's.
- **Replications:** SmolLM2-135M C5 0.496 (C5 − C0′ −0.0092 [−0.0264, +0.0081]); Qwen3-0.6B C5 0.497; Qwen3-1.7B C5
  0.500.
- **The model's answers ignore the algebra.** On the two-hop items, the model is right equally often whether the
  algebra is right or wrong (φ −0.006 for C5). On reverse-lookup items the algebra is right on every item, but the
  model answers 0.591 correctly.
- **T4 natural items:** C5 0.554 against C0′ 0.546; C5 − C5ut +0.0091 [+0.0018, +0.0164], Holm p 0.145 (n.s.).

**Reading:** pre-registered decision (c), "no behavioural readout of binding". With 09.1 this is refutation reading B1:
the store is readable but the host does not read roles from it. Binding is unused, not useless.

## Limitations
- The twins exist only on synthetic T5, with two relation pairs.
- The test measures implicit use through next-token scoring. A model might use roles in ways these prompts miss.
- Qwen3 and 135M lack the untyped and translation arms, so P1 itself runs at 360M only.

## Next steps
- **Done:** a trained readout (09.4). It does not change the result.
- **Done:** explicit recall, which decodes the store into the prompt (E12, pre-registered Q1). It passes: on the same
  twins C5 reaches 0.776 against 0.496 with no recall (+0.2794 [+0.2550, +0.3039]). The E12 file covers it.

---

# 09.3 — B-3: Chains, reverse lookup and capacity: binding wins only where roles are ambiguous

**Status:** done (T5, T4) · **When:** pre-registered and run 2026-10-08 (T5); T4 report 2026-10-19; T5 report regenerated
2026-10-20 · **Sources:** `experiments/e9-retrofit/preregistration-binding.md` §14;
`experiments/e9-retrofit/report/t5-binding/report.md`; `experiments/e9-retrofit/report/t4-binding/report.md`;
`experiments/e9-retrofit/report/binding-sweep/report.md`; `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s{1,2,3}/binding-chain/report.md`

## Research gap
Kumar (arXiv 2606.24948) found that an HRR knowledge-graph memory answers one hop but fails at zero-shot two hops,
because one global sum of all facts overflows. Wang & Sun (ICLR 2026) cast the reversal curse as a binding problem. We
asked whether per-term stores chain, and whether binding matters for reverse lookup.

## Intuition
A T5 store holds about 3–13 bindings; one global memory of 4,200 concepts in 256 dimensions holds far too many. Per-term
stores should chain where the global memory fails. Binding should matter only where a filler could sit under two
roles. Elsewhere the filler alone names its role, and a plain sum works as well.

## Methodology
- **Algebra only:** each run's trained composer, read from its checkpoint; the host model is not loaded.
- **Endpoints (T5, SmolLM2-360M, seeds 1–3):**
  - **C1:** C5's chained two-hop top-1 through per-term stores minus through one global memory, on 300 ontology paths.
  - **C2:** chained two-hop on 3,791 two-hop items, C5 − C5ut and C5 − C5tr (Holm).
  - **C3:** reverse-lookup reciprocal rank on 174 queries whose filler occurs under ≥ 2 relations, C5 − C5ut and
    C5 − C5tr (Holm).
- **Descriptive:** fidelity by store size; the "soft chain" (bridges mixed by the first hop's softmax); path order;
  a random-vector capacity sweep.
- **Predictions:** C1 large and positive; C2 positive but modest, with C5tr ≈ C5ut; C3 positive; soft chain ≥ hard
  chain; path order ≈ 0.5 for commutative operators.

## Experiment
CPU jobs, about 10 s per T5 run and 1 min per T4 run, on every composing run of T5 and T4. T1 and WordNet ran C1 at
seed 1. T7's fillers name no concept, so it has no two-hop paths.

## Results
| Endpoint | T5 (SmolLM2-360M, 3 seeds) | T4 (SmolLM2-360M, 3 seeds) |
|---|---|---|
| C1: local − global | +0.8144 [+0.7618, +0.8671] | +0.5400 [+0.3921, +0.6879] |
| C2: C5 − C5ut | +0.0109 [−0.0153, +0.0371], Holm 0.292 | +0.3976 [+0.3571, +0.4381]\* |
| C2: C5 − C5tr | +0.1057 [+0.0855, +0.1258]\* | +0.3630 [+0.3225, +0.4035]\* |
| C3: C5 − C5ut | +0.3803 [+0.3261, +0.4345]\* | +0.4956 [+0.4523, +0.5388]\* |
| C3: C5 − C5tr | +0.1905 [+0.1460, +0.2351]\* | +0.7234 [+0.6821, +0.7647]\* |

\* Holm-significant.

- **T5 means:** C5 chains at 0.884 through per-term stores and 0.074 through the global memory.
- **Path order is not kept** by any commutative operator (C5 0.492).
- **Fixed random binding chains as well or better.** On T4, C5rf chains at 0.732 against C5's 0.692, with reverse MRR
  0.933 against 0.760.
- **Capacity.** Fidelity falls as a store grows: C5 seed 1 decodes at 1.000 with 3 bindings and 0.930 with 13. On
  random vectors at 256 dimensions, unitary HRR recovers 0.98 of fillers at 8 bindings, 0.66 at 16 and 0.24 at 32.
- **Other tracks (C1, seed 1):** T1 +0.7500; WordNet +0.4933.

**Predictions:**
- C1: **passed** on T5 and T4.
- C2: on T5, **partial** (pre-registered reading: binding beats translation, not untyped; and C5tr ≠ C5ut). On T4,
  **passed** ("binding enables algebraic chaining").
- C3: **passed** on both tracks.
- Soft chain ≥ hard chain: **failed** (C5 soft 0.446–0.526 against hard 0.875–0.891 per seed).

## Limitations
- These are properties of the algebra. 09.2 shows the model's own answers do not follow them.
- On T5 the first hop's filler is often the only candidate of its type in the frame, so C2 is a weak test there, as
  pre-registered. No T5 frame holds one filler under two roles; on T4, 12% do.
- On T4, 28 of 365 two-hop items were dropped because their options do not map to concepts.
- The 12 October summary said binding wins "only for reverse lookup". That holds on T5. The later T4 report shows it
  also wins for chaining on real chemistry.

## Next steps
- **Done:** step 3 on T5 and T4, and the readout arms (09.5).
- **Done elsewhere:** chained explicit recall into the prompt (E12) and VSA executors (E14) build on these results.
- T1c aggregates were produced under the licence and are not read here.

---

# 09.4 — Readout arms: models trained with a readout that unbinds (decision 60 step 2; R1 null)

**Status:** done · **When:** pre-registered 2026-10-08 (§13); U5rf added 2026-10-09 (§13.1); training 2026-10-16 to
10-17; evaluations 2026-10-17 to 10-19; preliminary note 2026-10-18; official report 2026-10-20 · **Sources:**
`experiments/e9-retrofit/preregistration-binding.md` §13, §13.1; `experiments/e9-retrofit/report/t5-binding/report.md`
("Step 2"), `analysis.json`; `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-U5*-s{1,2,3}/readout/report.md`;
`experiments/e12-self-query/report/t5-readout/report.md`; `manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §1

## Research gap
Attention has been proposed as soft unbinding, with dedicated unbinding heads (Dhayalkar, AAAI-26 workshop), but the
proposal was untested. 09.2 showed the host does not read roles by itself. Perhaps it needs a component built to unbind.

## Intuition
Give the model a head that picks a relation, unbinds it from the latest term's store and adds the filler to the
residual stream. If binding matters once something reads it, binding arms should now beat the controls on the twins.

## Methodology
- **Readout:** at layer 11 of 32, a linear map of the hidden state gives a softmax over the 23 relations plus "no
  query". The head unbinds them from the store of the latest linked term (within 32 tokens), runs one typed soft
  cleanup step, and adds the result through a gate that starts closed.
- **Training:** C5's recipe and the language-model loss only, with no auxiliary role loss.
- **Arms:**

  | Arm | Operator | Unbinding |
  |---|---|---|
  | U5 | learned HRR (C5's) | correlation |
  | U5u | learned unitary HRR | conjugate (exact) |
  | U5sb | bounded spectrum (magnitudes in [0.5, 2]) | adjoint |
  | U5bu | block-diagonal unitary, non-commutative | transpose (exact) |
  | U5sl | slotted: 3 relation groups in separate slots | conjugate within slot |
  | U5rf | fixed random unitary (as C5rf) | conjugate (exact) |
  | U5tr | translation (control) | subtraction |
  | U5ut | untyped (control) | the store itself |
- **Primary endpoints:**
  - **R1:** twin contrast U5 − U5ut and U5 − U5tr, Holm.
  - **R2:** U5 − C5, relative loss after held-out terms (paired window bootstrap, 10,000 resamples).
- **Secondaries:** U5 − U5rf (a family of one); S2.1, the new families − U5ut.
- **Diagnostics:**
  - role accuracy at positions before a filler (chance 1/23 = 0.043);
  - read MRR (filler recovery with the head's own query) and oracle MRR (with the gold relation);
  - loss with the gate switched off.
- **Predictions:**
  - R1 (a) with U5 − U5ut ≈ +0.05 to +0.15;
  - R2 (a), −1% to −3%;
  - role accuracy above chance for every binding arm;
  - for the controls, oracle recovery ≈ 1/k (k = store size).

## Experiment
We trained 24 runs (8 arms × seeds 1–3; SmolLM2-360M, T5, 50M tokens each) and evaluated each on the twins, the
readout diagnostics, the probes, the chains (09.5), the edits (09.6) and explicit recall (E12).

## Results
**Primary and secondary contrasts:**

| Endpoint | Estimate [95% CI] | p | Prediction |
|---|---|---|---|
| R1: U5 − U5ut | +0.0061 [−0.0069, +0.0191] | Holm 0.355 | **failed** |
| R1: U5 − U5tr | +0.0133 [−0.0037, +0.0303] | Holm 0.247 | **failed** |
| R1 secondary: U5 − U5rf | +0.0033 [−0.0095, +0.0161] | 0.608 (family of one) | — |
| S2.1: U5u / U5sb / U5bu / U5sl − U5ut | −0.0053 / +0.0017 / −0.0011 / −0.0006 | Holm 1 each | — |
| R2: U5 − C5, after held-out (relative) | +0.0239 [+0.0214, +0.0266] | 0.0002 (bootstrap) | **failed** (worse, not better) |

**Per arm (3-seed means; loss columns relative, after held-out terms; positive = higher loss):**

| Arm | Twin contrast | Role accuracy | Read MRR | Oracle MRR | Gate off − on | On − C5 |
|---|---:|---:|---:|---:|---:|---:|
| U5 | 0.501 | 0.123 | 0.125 | 0.980 | +0.0201 | +0.0240 |
| U5u | 0.490 | 0.103 | 0.107 | 0.990 | +0.0192 | +0.0645 |
| U5sb | 0.497 | 0.103 | 0.108 | 0.990 | +0.0202 | +0.0514 |
| U5bu | 0.494 | 0.068 | 0.078 | 0.988 | +0.0150 | +0.0396 |
| U5sl | 0.494 | 0.196 | 0.196 | 0.971 | +0.0068 | +0.0721 |
| U5rf | 0.498 | 0.107 | 0.111 | 0.990 | +0.0182 | +0.0572 |
| U5tr | 0.488 | 0.120 | 0.115 | 0.938 | +0.0101 | −0.0254 |
| U5ut | 0.495 | 0.114 | 0.124 | 0.978 | +0.0182 | +0.0296 |

Every "gate off − on" and "on − C5" interval excludes 0.

- **R1 reading (c):** "even an explicit readout trained by the LM loss does not use binding". A fixed random operator
  serves the readout as well as a learned one.
- **R2 reading (c):** "the readout hurts held-out loss" against C5. Within each arm the readout lowers loss (switching it
  off costs 0.68–2.02%), but the arms' channels end up worse than C5's. Only translation beats C5, as plain translation
  (C5tr) already did on T5 held-out loss.
- **The head mostly queries the wrong relation.** Read MRR is 0.078–0.196, against 0.938–0.990 with the gold relation.
  Role accuracy exceeds chance in every binding arm (0.068–0.196), as predicted. The controls score as high (0.114–0.120),
  so the head's accuracy owes nothing to binding.
- **The controls' prediction failed.** Their oracle MRR (0.978 untyped, 0.938 translation, all edges) is far above 1/k.
  Pre-written reading RB2: the typed cleanup supplies the role.
- **The roles are in these stores.** Decoded into the prompt (E12 explicit recall), the binding arms' stores give twin
  contrast 0.785–0.830. The controls give 0.503 (U5tr) and 0.505 (U5ut), and no recall gives 0.491–0.504.

## Limitations
- Training used the LM loss only. An auxiliary role loss was recorded as an alternative, not run (it risks training a
  copying device).
- One host, one synthetic track, two relation pairs in the twins.
- The diagnostics do not split oracle recovery by role ambiguity, so RB2 is read on all edges.
- The 2026-10-18 note's "the readout still lowers loss a little" holds within each arm only. Against C5 the readout
  arms are worse (R2). The note's other numbers match the official report.

## Next steps
- **Closed:** R1 is null. The programme has moved to explicit recall (E12, E13) and external executors (E14).
- **Not queued (author's call, BU-8):** a Qwen3-1.7B U5 / U5ut pair and a 50M from-scratch readout screen.

---

# 09.5 — Binding-chain fidelity of the readout arms' operators

**Status:** done · **When:** chain jobs 2026-10-19; U5sb re-run 2026-10-20 after fix `6f35eeb`; report 2026-10-20 ·
**Sources:** `experiments/e9-retrofit/preregistration-binding.md` §13 (decision 61 predictions), §14;
`experiments/e9-retrofit/report/t5-binding/report.md` ("Step 3"); `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-U5*-s{1,2,3}/binding-chain/report.md`;
`experiments/e9-retrofit/report/binding-sweep/report.md`; commit `6f35eeb`

## Research gap
Decision 61 added three operator families to the readout arms: a bounded spectrum, a block-diagonal unitary and a
slotted layout. Circular binding commutes, so it cannot keep the order of a path stored in one vector; block-diagonal
rotations do not commute. We checked whether each family's algebra behaves as designed after LM training.

## Intuition
Operators with exact inverses should decode best. Learned HRR read by exact division should decode worst, because
division amplifies noise at weak frequencies. A non-commutative operator should tell "r1 then r2" from "r2 then r1".

## Methodology
- The step-3 analysis (09.3) ran on each readout arm's trained composer.
- **Fidelity:** typed MRR, by method.
- **Two-hop:** 2,000 ontology paths, scored at hop 1, chained, from the true bridge ("oracle"), soft chain, and through
  one global memory.
- **Reverse lookup:** 1,000 queries.
- **Path order:** 1,000 opposite-order pairs at m = 0, 2 and 6 distractor paths (chance 0.5).
- **Predictions (decision 61):**
  - fidelity: unitary ≈ bounded ≥ learned HRR (adjoint) ≫ learned HRR with exact division;
  - block-unitary ≈ circular binding at one hop, and near 1 against 0.5 on path order at m = 0;
  - slotted ≈ a single vector at T5's even load (33.0 / 33.5 / 33.5%);
  - soft chain ≥ hard chain; oracle ≥ chained.
- **Fix:** the first U5sb jobs failed. Commit `6f35eeb` (2026-10-20) uses the family's own exact division, and the jobs
  were re-run.

## Experiment
CPU analysis of the 24 readout-arm composers, with no host model.

## Results
**3-seed means (official report):**

| Arm | Fidelity MRR (method) | Hop 1 | Chained | Oracle | Global | Reverse MRR | Path order, m = 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| U5 | correlation 0.975; inverse 0.838; exact 0.673 | 0.941 | 0.865 | 0.964 | 0.076 | 0.932 | 0.497 |
| U5u | conjugate 0.992 | 0.973 | 0.907 | 0.989 | 0.053 | 0.941 | 0.500 |
| U5sb | adjoint 0.991; exact 0.991 | 0.973 | 0.905 | 0.986 | 0.054 | 0.942 | 0.509 |
| U5bu | transpose 0.989 | 0.970 | 0.894 | 0.981 | 0.061 | 0.928 | **1.000** |
| U5sl | conjugate 0.982 | 0.938 | 0.864 | 0.968 | 0.062 | 0.915 | 0.503 |
| U5rf | conjugate 0.992 | 0.974 | 0.908 | 0.989 | 0.052 | 0.942 | 0.500 |
| U5tr | subtract 0.919 | 0.775 | 0.676 | 0.866 | 0.087 | 0.642 | 0.489 |
| U5ut | bundle 0.976 | 0.894 | 0.815 | 0.950 | 0.068 | 0.886 | 0.500 |
| C5 (no readout) | correlation 0.989; inverse 0.844; exact 0.702 | 0.942 | 0.884 | 0.985 | 0.074 | 0.959 | 0.492 |

- **Fidelity ordering: passed.** Unitary 0.992 ≈ bounded 0.991 ≥ learned HRR 0.975 ≫ learned HRR with exact division 0.673.
- **Block-unitary: passed.** It matches circular binding at one hop (0.970 against U5u's 0.973). It keeps path order at
  1.000 for m = 0, 2 and 6 in every seed; every commutative operator stays at 0.440–0.547 per run and m.
- **Slotted:** slightly below a single vector (fidelity 0.982 against 0.992; chained 0.864 against 0.907). No test was
  pre-registered for this.
- **Soft chain ≥ hard chain: failed in all 24 runs** (soft 0.29–0.80 against hard 0.67–0.92 per run).
- **Oracle ≥ chained: passed** in every arm.
- **Global memory fails** for every arm (0.052–0.087), as in 09.3.

## Limitations
- These are algebraic properties. On the two-hop items the readout arms' own answers are right 0.283–0.287 of the time
  whatever the algebra does (φ −0.013 to −0.001).
- Path order is measured on vectors built from the trained atomics, not in the model's behaviour.
- The table's means carry no intervals. Commit `6f35eeb` quotes seed 1 (U5sb 0.996); the report gives the 3-seed mean
  (0.991).

## Next steps
- **Done:** the step-3 analysis on every readout arm.
- No follow-up is queued. Order-sensitive structures (trees, lists) are tested with explicit executors in E14.

---

# 09.6 — Editing the readout arms' frames without retraining (edit-v2)

**Status:** partly done (per-run reports; no pooled cross-arm report) · **When:** first attempt failed 2026-10-17;
re-run 2026-10-18 to 10-19 · **Sources:** `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-U5*-s{1,2,3}/edit-v2/report.md`;
`experiments/e9-retrofit/report/t5-readout/report.md` (dimension 3, v1 items);
`manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §4; `manuscript/claims.md` row E9.C1;
`resources/plan-improvement/execution.md` decision 56

## Research gap
Claim C says we can change a model's behaviour on a concept by editing its frame, with no weight update. On T5 it holds
for C5 (claims row E9.C1). We asked whether it holds for the readout arms, and whether binding matters for edits.

## Intuition
If the model uses a frame's content, a frame written for a new word should beat a random frame of the same size.
Replacing one filler should shift the model's preference towards the new filler.

## Methodology
- **Items "v2"** (decision 56, a superset of v1):
  - **New words:** 700 invented names, each linked to a new frame. Sources: `own`, `none`, `mean_row`, `random_frame`
    (equal size). Tests: property choice (chance 0.200), entailment (chance 0.500), paraphrase, statements.
  - **Edits:** 700 frame edits (367 seen, 333 held-out concepts) and 1,081 neighbour concepts.
- **Edit metrics (ROME / MEMIT style):**
  - ES: the share of prompts that prefer the new filler.
  - EM: the change in log p(new) − log p(old), in nats.
  - Control: the same edit to a different filler of the same type.
- **Statistics:** paired bootstrap over items (2,000 resamples), Holm over baselines within each test.
- **Sign:** a positive own − baseline or EM − control means the edit works.
- **No prediction** was pre-registered for the readout arms on these items. S2.5 named "v2 items against C5", but C5
  has no v2 run.

## Experiment
The first attempt (2026-10-17) failed because the alias table was missing. Without it, the jobs fell back to WordNet.
The jobs were re-run with `--alias-table …/t5.json`: 24 runs, bf16, no weight update.

## Results
**Ranges over seeds 1–3 (each cell: lowest and highest per-run value):**

| Arm | Property, own frame | Own − random frame (property) | EM − control, all 700 edits |
|---|---|---|---|
| U5 | 0.2502–0.2742 | +0.0354 to +0.0608 | +0.2122 to +0.2404 |
| U5u | 0.2364–0.2767 | +0.0240 to +0.0616 | +0.2251 to +0.2457 |
| U5sb | 0.2372–0.2792 | +0.0244 to +0.0637 | +0.2078 to +0.2430 |
| U5bu | 0.2481–0.2651 | +0.0398 to +0.0553 | +0.2018 to +0.2535 |
| U5sl | 0.2395–0.2527 | +0.0329 to +0.0429 | +0.1578 to +0.1868 |
| U5rf | 0.2353–0.2776 | +0.0267 to +0.0607 | +0.2099 to +0.2451 |
| U5tr | 0.3204–0.3264 | +0.0933 to +0.1050 | +0.1955 to +0.2568 |
| U5ut | 0.2466–0.2843 | +0.0393 to +0.0711 | +0.1789 to +0.2600 |

- **The frame's content matters in every run.** The own frame beats the random frame on property and on entailment in
  all 24 runs (every 95% CI excludes 0). With no row, property sits near chance (0.195–0.202).
- **Translation scores highest.** U5tr reaches 0.3204–0.3264 on property, against 0.2353–0.2843 for every other arm.
  The untyped control falls within the binding arms' range. No cross-arm test exists (see Limitations).
- **Edits move the model, mostly on seen concepts.** Over all 700 edits, EM − control is positive in all 24 runs (CIs
  exclude 0), and ES rises from 0.495–0.507 to 0.533–0.549. On the 367 seen concepts, EM − control runs from +0.3016 to
  +0.4531.
- **Edits barely move held-out concepts:** EM − control −0.0126 to +0.0825, with the CI including 0 in 9 of 24 runs.
- **The same pattern appears on v1 items** in the official batch report: property C5 0.249–0.265, U5 0.252–0.275,
  C5tr 0.312–0.316, U5tr 0.302–0.315.

## Limitations
- There is no pooled cross-arm report, and no C5 run on v2 items. The pre-registered comparison against C5 (S2.5)
  cannot be made yet. The v1 comparison exists but uses fewer items.
- The effects are small in absolute terms (own-frame property 0.2353–0.3264 against chance 0.200) and come from a
  synthetic track. On real chemistry (T4), claim C is null at 3 seeds (claims row E9.C1).

## Next steps
- **Not queued:** a C5 v2 run and a pooled report. Decision 56 called for the T5 evaluations to be re-run on v2; we
  found no C5 v2 output in the run folders.
- **Closed for this programme:** binding brings no editing advantage over untyped or translation stores.

---

# Sources

- Pre-registration: `experiments/e9-retrofit/preregistration-binding.md` (§§1–7, §12.1–12.5, §13, §13.1, §14).
- Official reports (generated 2026-10-08 to 2026-10-20; untracked in git at the time of writing):
  - `experiments/e9-retrofit/report/t5-binding/report.md` and `analysis.json` (steps 1–3, readout arms included);
  - `experiments/e9-retrofit/report/t5-qwen3-binding/report.md`;
  - `experiments/e9-retrofit/report/t4-binding/report.md`, `t7-binding/`, `t1-binding/`, `wordnet-binding/`;
  - `experiments/e9-retrofit/report/binding-sweep/report.md` (random-vector sweep);
  - `experiments/e9-retrofit/report/t5-readout/report.md` (R9 batch with the readout arms; dimension 3 on v1 items);
  - `experiments/e12-self-query/report/t5-readout/report.md` (explicit recall with the readout arms' stores).
- Per-run outputs: `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-{C5,C5rf,C5ut,C5tr,U5*}-s{1,2,3}/`
  `{binding-chain,readout,edit-v2}/report.md`.
- Summaries: `manuscript/update-2026-10-12.md` (B-1 to B-3); `manuscript/status-2026-10-08.md` §3.4;
  `manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §1, §4 (preliminary, superseded);
  `manuscript/findings/2026-10-20-e14-experiments-explained.md` C1.
- Plans and ledgers: `resources/plan-improvement/execution.md` (decisions 56, 60, 61, 62, 64; step 2 and 3 blocks);
  `manuscript/claims.md` row E9.C1.
- Commits: `4a45616` (step 1 pre-registration), `e1bdf4c` (§13), `4954e79` (§14), `b277bb7` (U5rf), `62ea22d`
  (2026-10-18 preliminary), `6f35eeb` (2026-10-20 chain fix).
