# E9 — the binding and unbinding program (decision 60)

**Status:** design and pre-registration of step 1, written on 2026-10-08 and committed **before any evaluation of these
measures on a trained checkpoint** (GPU or CPU). The labelled smoke tests (§11) run after this commit; they check that the
pipeline works and estimate cost, and change nothing in §§1–10. Steps 2 (the unbinding readout arm) and 3 (chained two-hop,
reverse lookup, capacity) are pre-registered in §13 and §14, each committed before its own runs. Any later change is
listed in §12 with its date and reason.

**Origin:** author decision 60 (`resources/plan-improvement/execution.md`, 2026-10-08). Decision 54's 50M operator screen
(`experiments/e4-small-lm/analysis/opscreen-v1`) and the T5 controls (R9, WP-PQ1) agree: **composition matters, the
binding operator does not**, for next-token loss — C5 beats C2, the same-site row sources and C0′, while C5ut (untyped:
a bundle of fillers) ≈ C5 and C5tr (translation `x + t_r`) is slightly better. The author's measurement of role ambiguity
suggests why: a term's filler set almost identifies it.

| Track | terms whose filler multiset collides once roles are dropped (decision 60) | same filler under two roles within one frame (decision 60) | fillers that occur under ≥ 2 relations (this document) | edges whose filler does (this document) |
|---|---:|---:|---:|---:|
| T5 | 0% | 0% | 46.2% | 19.5% |
| T4 | 0.5% | 12% | 63.1% | 34.7% |
| T1 | — | — | 44.3% | 25.6% |
| WordNet | 2% | — | 79.9% | 40.1% |
| T7 | 0% | — | 0% | 0% |
| T1c | 0.1% | 3.7% | 64.2% | 41.6% |

(This document's own count of the first column — entries whose filler multiset equals another entry's while their
frames differ — gives T5 0.0%, T4 1.05%, T1 0.37%, WordNet 8.2%, T7 0.0%, T1c 0.24%; same filler under two roles: T5 0%,
T4 12.1%, T1 1.4%, WordNet 4.1%, T7 0%, T1c 3.7%.)

So the next-token objective rarely needs a role read out: knowing *which fillers* a term has is almost always enough.
**Hypothesis H-B:** binding matters where a role has to be read out — where two terms differ only in which filler holds
which role, or where one filler must be retrieved for one relation from a bundle that holds several plausible fillers.

**Related work** (checked 2026-10-08):
- Hernandez et al., ICLR 2024, "Linearity of Relation Decoding in Transformer Language Models": relation decoding from a
  subject representation is often well approximated by a linear relational embedding (LRE) — implicit, learned
  unbinding; a NeurIPS 2025 follow-up studies the structure of these operators. Probe B is LRE-style.
- Dhayalkar, arXiv 2512.14709 (AAAI-26 LSRLM workshop): attention as soft unbinding; proposes binding / unbinding heads,
  untested. Step 2's readout is such a head, tested.
- Dhanraj & Eliasmith, EMNLP 2025: HRR over a frozen host's hidden states for rule-based reasoning.
- Kumar, arXiv 2606.24948: an HRR + modern-Hopfield knowledge-graph memory works at one hop and fails at zero-shot two hops,
  from the capacity of one global superposition. Step 3 compares local (per-concept) against global superposition on the
  same trained atomics.
- Wang & Sun, ICLR 2026 (arXiv 2504.01928): the reversal curse as a binding problem (a JEPA-style fix). Step 3's reverse
  lookup.
- arXiv 2512.09369: generalized-HRR knowledge-graph path retrieval for LLMs (step 3's chains).
- Feng & Steinhardt, ICLR 2024, "How do Language Models Bind Entities in Context?": binding IDs — in-context binding is
  read out by the host's own mechanism.
- Hayashi & Shimbo, ACL 2017: HolE (circular-correlation scoring) ≡ ComplEx; HRR binding is a KG-embedding operator.

**Code** (tests: `tests/test_unbinding.py`, `tests/test_e9_binding.py`, CPU):
- `src/vsa_embed/relations.py`: `unbind()` on every composition operator family (§2); `src/vsa_embed/cleanup.py`: hard,
  type-constrained and soft cleanup; `FrameComposer.raw_bundle` / `unbind`.
- `src/vsa_embed/experiments/e9_binding_probe.py`: probes A and B (`probe`, `queue`).
- `src/vsa_embed/experiments/e9_binding_items.py`: role-swap twins and natural role-ambiguous items (`items`, `evaluate`,
  `queue`).
- `src/vsa_embed/experiments/e9_binding_report.py`: P1, P2 and the secondaries across a stage's runs.

**Items** (committed with this document; `items.jsonl.gz` is gzip with mtime 0, so the digests are reproducible):

| Item set | concepts | items (choice / cloze) | sha256 `concepts.jsonl` | sha256 `items.jsonl.gz` |
|---|---:|---:|---|---|
| `items/role-twins-t5-smollm2-v1` | 600 (300 pairs) | 1,200 / 1,200 | `4a34899c…48e7` | `52cea580…33da` |
| `items/role-twins-t5-qwen3-v1` | 600 (300 pairs; the same frames and names) | 1,200 / 1,200 | `4a34899c…48e7` | `52cea580…33da` |
| `items/role-natural-t4-smollm2-v1` | 715 (seen 300 / rare 300 / held-out 115) | 1,430 / 1,430 | `1ab2d96e…d902` | `4462062a…9398` |

Full digests are in each `manifest.json`.

## 1. Questions and primary endpoints

| | Question | Unit | Primary endpoint |
|---|---|---|---|
| **P1** (decisive for "binding") | Do the models tell apart two new words whose frames hold the same fillers with two roles swapped? | twin pair × seed | role-swap **twin contrast accuracy** (`choice` items, source `own`): **C5 − C5ut** and **C5 − C5tr**, Holm over the two |
| **P2** | Is the trained learned-HRR store algebraically readable — unbind a role, clean up — on concepts composed zero-shot, and does learning the operator help or hurt that against fixed random binding? | held-out concept × seed | **P2a:** held-out MRR of C5 (static bundle, typed cleanup, correlation unbinding) − the frequency baseline on the same edges; **P2b:** the same MRR, C5 − C5rf; Holm over P2a and P2b |

**Primary setting:** T5 synthetic glossary, SmolLM2-360M, full fine-tuning, seeds 1–3, the runs of R9 / WP-PQ1 (bf16
autocast as trained). P1 and P2 answer different questions, so no correction is made between them (stated in advance);
each is tested at two-sided α = 0.05 after its own Holm step.

## 2. Primitives

**Unbinding** (`RelationTransform.unbind`; the first method is the primary one):

| Family (arm) | primary | other |
|---|---|---|
| `hrr` (C5: learned roles, not unitary) | `correlation`: circular correlation, the involution (the adjoint) | `inverse`: regularized FFT division `F⁻¹[V·conj(R)/(|R|² + λ)]` |
| `unitary_hrr` (C5rf: fixed; the learned unitary family of step 2 trains the phases) | `conjugate` (exact) | — |
| `orthogonal` | `transpose` (exact) | — |
| `diagonal` | `inverse` (regularized division) | — |
| `map` | `auto`: self-inverse where the role is bipolar, else regularized division | `self`, `inverse` |
| `low_rank*` | `woodbury`: `(I + L R)⁻¹` exactly | — |
| `translation` (C5tr) | `subtract`: `v − t_r` | — |
| `additive` / `untyped` (C5ut) | none — `unbind()` raises | `bundle`: the vector itself, ignoring the role |

`λ = 0.01 · mean power` of the relation's spectrum or diagonal. The learned unitary HRR family already exists
(`UnitaryHRRRelation`: trainable phases of unit-magnitude spectra, exactly invertible; `random_fixed:` freezes it).

**Cleanup** (`cleanup.py`): cosine nearest neighbour over the run's trained atomic dictionary (`all`), or over the atomics
observed under the relation in the frames (`typed`: the type-constrained variant); `SoftCleanup` (modern-Hopfield
steps with a learned inverse temperature) for step 2.

**Ranks.** Filtered (the frame's other fillers of the same relation are removed from the candidates), ties at half weight
(the expected rank under random tie-breaking); top-1 = the expected top-1 under random tie-breaking. MRR per concept =
the mean reciprocal rank over its edges; a subset's MRR = the mean over its concepts.

## 3. Probe A — algebraic decodability (every composing arm)

For every probed concept and every edge `(r, f)` of the frame the composer stores (C5sh: its shuffled schedule), unbind
`r` from the concept's bundle with the arm's own unbinding and clean up (§2). The bundle is the **un-normalized** sum
`Σ_e w_e T_{r_e}(a_e)` (`FrameComposer.raw_bundle`): for linear operators normalization does not change a cosine readout;
for the affine translation the un-normalized sum is the only fair input.

**Conditions:**
- `static` — the static frame bundle, every edge weight 1 (the store; **primary**);
- `nocontext` — the trained attentive weights without a context term;
- `neutral` — the attentive weights with the P1 context of the neutral mention "We discussed X" (the term's own surface:
  the concept's name when it is an alias with ≥ ℓ_min subtokens, else the canonical alias);
- `context` — the attentive weights in the term's real evaluation contexts: the first 4 occurrences in the run's own
  evaluation windows (the trainer's `eval_windows`); per edge the mean over occurrences. T5's 200 generator zero-shot terms
  occur in no document and are absent from this condition only.

**Methods:** the arm's primary unbinding; for learned HRR also `inverse`.

**Role recovery** (the reverse question — which relation holds filler f?): every relation `r'` is scored by
`cos(c, T_{r'}(a_f))` (for HRR: the filler unbound from the bundle, matched against the role vectors), filtered (other
relations holding f in the same frame removed), over all relations (`all`) or those observed with f (`typed`). For the
untyped bundle every relation ties, so it scores exactly chance.

**References on the same edges:** `frequency` — candidates ranked by the relation-conditional frequency of the filler in the
training-visible frames (not held out), the matched prior; `chance` — the expected reciprocal rank `H_n / n` of a random
ranking of the same filtered candidates.

**Subsets:** `seen` (training frequency ≥ 10), `rare` (1–9), `unseen` (0, not held out), `heldout` (held-out and generator
zero-shot terms: composed zero-shot); at most 3,000 entries per subset (a seeded sample on large tracks, identical for every
run of a track; every T5 entry is probed).

**Splits:** frame size (superposition load: degree 1–4, 5–7, 8–10, ≥ 11, and exact degree in `edges.npz`); role
ambiguity — `ambiguous` (the filler occurs under ≥ 2 relations in the frames), `unambiguous`, `reuse` (the same filler under
two roles within this frame), `multi` (≥ 2 fillers of this relation in this frame).

## 4. Probe B — learned (LRE-style) decodability (every model)

Per relation, a ridge map (standardized features, strength by GCV over 10⁻²…10⁴) from a concept representation to the
multi-hot set of its fillers, **trained on the seen concepts** (not held out, training frequency ≥ 1; their scores
out-of-fold over 5 folds) and **applied to the held-out and unseen ones**. Ranks over the fillers the training concepts carry
(filtered as in §2); a filler never seen under the relation in training is *uncovered* and scores 0 (coverage reported).
Relations with < 10 training concepts or < 2 classes are not fitted (listed). Targets are the ontology frames.

**Representations:**
- `composed_static` (the static bundle, normalized) and `composed` (the composition without context) — every composing arm;
- `free_row` — C2's free row as the model reads it (held-out terms: the mean-row fallback);
- `source_row` — C6m / C6d / C6g's frozen source row;
- `injected_row` — the row injected after the projector (every channel);
- `hidden_middle` (layer ⌊L/2⌋) and `hidden_final` — the host's hidden state at the term's last subtoken in the neutral
  mention, for P0, C0′, C2 and C5 (entries whose surface links to them; the same entries for every model).

Probe A asks whether the store is algebraically readable; probe B asks whether the information is there at all.

## 5. Behavioural role items

**Role-swap twins** (T5; `e9_binding_items --kind twins`). 300 pairs of new words A, B (invented names: pronounceable, not a
WordNet lemma, not an alias word, not one token, ≥ 2 subtokens, absent from 20M tokens of the training corpus; inserted at
evaluation with the dimension-3 machinery, so their rows are composed zero-shot):
A = rest + (r1, X) + (r2, Y), B = rest + (r1, Y) + (r2, X).
- **Relation pairs:** the templated relations that share fillers *and* co-occur in real frames. On T5: `depends_on` /
  `part_of` (128 shared fillers; systems) and `owned_by` / `approved_by` (43; policies), 150 pairs each. `depends_on` /
  `uses` share the most fillers (282) but never co-occur in one frame (`uses` is a process relation, `depends_on` a system /
  project / tool one), so twins of that pair would have no real type and are not built.
- **X ≠ Y:** drawn uniformly from the fillers observed under both r1 and r2 (distinct texts), so either role assignment is
  plausible.
- **rest:** a real donor's frame with exactly one r1 and one r2 filler (not held out, not synthetic): the category edge
  (`is_a`) kept, every other filler resampled frequency-weighted, never X or Y. Every frame is new.

An untyped bundle or a translation gives A and B the same vector up to the attention weights, so they are structurally
limited on these items; HRR and unitary binding are not.

**Natural role-ambiguous items** (T4; `--kind natural`). Existing entries whose frame holds F under r1 and G under r2 (F not
under r2 and G not under r1 in that frame), with F observed under r2 and G under r1 elsewhere in the ontology (each a
plausible filler of the other's role); one pair per anchor (seeded); anchors: seen 300, rare 300, held-out 115 (every
eligible held-out anchor). Mostly `is_a` against `has_functional_parent` / `is_conjugate_base_of` / `is_conjugate_acid_of`.

**Items** (per concept and relation): `choice` — the relation's two property prompts, candidates in a fixed order [X, Y]
(twins) or [F, G] (natural), PMI against the null prompt (the term replaced by "this"); `cloze` — the relation's held-out
wording (the third paraphrase or the statement prefix), per-token log-probability minus the null prompt's. Scored with
`e9_understanding`'s continuation scorer (output head in float32; its §10.3).

**Scores:**
- **twin contrast** (P1): for relation r and template t, `z = ±[(s_A(X) − s_A(Y)) − (s_B(X) − s_B(Y))]`, signed so that
  `z > 0` when the twins' preferences differ in the direction of their frames. The null prompt is identical for A and B and
  cancels. A pair's **contrast accuracy** = the share of its 2 relations × 2 templates with `z > 0` (ties 0.5). A model
  that cannot tell the twins apart scores 0.5 in expectation (the twins' names are random).
- **role contrast** (natural): `z = [PMI_r1(F) − PMI_r1(G)] − [PMI_r2(F) − PMI_r2(G)]` per template index; > 0 when the
  model prefers F more under r1 than under r2 (a filler's general salience cancels); accuracy as above, chance 0.5.
- **item accuracy:** PMI argmax = gold, per item and template (chance 0.5).

**Sources:** `own` (every model); `none` (the concepts' rows zeroed: what the host alone does; every channel model);
`swap` (twins, composing arms: each twin reads its partner's frame — a model that follows its frames flips).

## 6. Statistics

- **P1:** the pairs × seeds table of C5 − reference contrast accuracy over the pairs both of whose twins link in every run
  of the contrast; the crossed random-effects model (`statistics.crossed_components`, Satterthwaite t, two-sided); Holm over
  the two references; the two-way cluster bootstrap (2,000 resamples) reported as the check.
- **P2:** the held-out concepts × seeds table of per-concept MRR differences; the same model. P2a's reference (the frequency
  baseline) is seed-independent.
- **Secondaries:** the same model; Holm within each numbered family of §7.
- **Power (planning):** with 300 pairs, 3 seeds and a per-pair SD of the difference ≈ 0.35 (four ±1 comparisons per pair),
  the 80%-power minimum detectable P1 difference at α = 0.025 is ≈ 0.04 in contrast accuracy.

## 7. Predictions, decision rules, refutation readings

**Predictions** (written before running; T5, 360M):
- **P1:** C5 − C5ut > 0 and C5 − C5tr > 0, but modest (C5 contrast accuracy ≈ 0.55–0.65; C5ut, C5tr ≈ 0.50). The C5 twins
  differ in their composed rows, but the host was trained on text whose roles are almost always recoverable from the
  filler set, so its readout of roles may be weak. C5 `own − swap` > 0.
- **P2a:** C5 ≫ the frequency baseline on held-out concepts (MRR well above the prior; `typed`).
- **P2b:** C5 − C5rf < 0: fixed random unitary roles decode better than learned, non-unitary roles (correlation of a
  non-unitary role adds crosstalk; the LM loss does not reward decodability).
- C5ut and C5tr: `typed` filler recovery near C5 on unambiguous edges and clearly lower on ambiguous ones; role recovery at
  chance (C5ut exactly).
- Probe B: the composed vector of every composing arm carries its fillers (high held-out MRR); the hidden state of C5 at a
  held-out term carries them better than C0′'s (whose held-out terms are new strings).

**Decision rule P1.**

| Reading | Conditions |
|---|---|
| **(a) "Binding is read out where a role must be read out"** | C5 − C5ut > 0 **and** C5 − C5tr > 0 (Holm p < 0.05); labelled "frame-driven" when also C5 `own − swap` > 0 (CI excludes 0) |
| **(b) "Partial"** | exactly one of the two significant and positive |
| **(c) "No behavioural readout of binding"** | neither significant. With P2a (a), the store is readable but the host does not read roles out of it — the case step 2's explicit readout tests |
| **(d) "Role-blind beats binding"** | a significant negative contrast |

**Decision rule P2.**
- **P2a:** (a) "readable beyond the prior" — C5 − frequency > 0 (Holm p < 0.05); (b) "not readable beyond the prior" — the CI
  includes 0; (c) "below the prior" — significant and negative (training made the store unreadable algebraically).
- **P2b:** "learned more / less decodable than fixed binding" by the sign of a significant C5 − C5rf; else "no difference".

**Refutation readings.**

| # | Observation | Reading |
|---|---|---|
| B1 | P2a (a) and P1 (c) | the store is algebraically readable, but the host does not read roles from it: binding is unused, not useless |
| B2 | P1 (a) and P2a (b) | the host reads roles through a route other than unbinding; check probe B on the composed vector against probe A |
| B3 | C5ut role recovery above chance | impossible by construction (every relation ties); a bug |
| B4 | P0, C0′, C2 or C6* twin contrast ≠ 0.5 (CI excludes 0.5) | the twins' names carry signal; the twin assignment is random, so investigate before reading P1 |
| B5 | C5 `own − swap` ≤ 0 while P1 (a) | the twin effect is not frame-driven |
| B6 | natural T4 role contrast of C0′ > 0.5 | the host knows these roles from text; the channel's share is C5 `own − none` |

**Secondaries** (Holm within each):
- **S1** twins: every arm against C5 and every model against 0.5 (C5rf, C5sh, C6m/d/g, C2, C0′, P0; C5ut, C5tr); `cloze`
  items; C5's `own − none` and `own − swap`.
- **S2** natural T4 items: C5 − C5ut, C5 − C0′, C5 `own − none`, by subset (held-out first).
- **S3** probe A by condition (static, nocontext, neutral, context), method (correlation, inverse) and cleanup (all, typed).
- **S4** the capacity curve: MRR by frame size (superposition load), per arm.
- **S5** role-ambiguous edges: C5 − C5ut and C5 − C5tr on `ambiguous` edges; `reuse` and `multi` edges (T4, T1c).
- **S6** role recovery per arm.
- **S7** LRE against algebra: probe B (`composed_static`) − probe A (static, typed) on held-out edges, per arm.
- **S8** hidden states: probe B on P0, C0′, C2, C5 (middle, final), held-out: C5 − C0′.
- **S9** replications: SmolLM2-135M (P0, C0′, C2, C5), Qwen3-0.6B / 1.7B seeds 1–2 (twins `-qwen3-v1`); T4 (seed 1 now,
  every T4 config once trained); T1 and WordNet seed 1 (controls: probe only); T7 (no role ambiguity: decodability only);
  T1c (`--licensed`: aggregates only, no items).

## 8. Runs and order

Evaluation only, on finished `final.pt` checkpoints (main checkout, `experiments/e9-retrofit/runs/…`). Per run: one probe
job (GPU lane; `RUN/binding-probe/`) and, where items exist, one item job (GPU lane; `RUN/role-<items>/`); one report per
stage (CPU lane; `report/<stage>-binding/`).

| Block | Runs | Probe | Items |
|---|---|---|---|
| primary | T5 SmolLM2-360M: P0, C0′, C2, C5, C5rf/ut/tr/sh, C6m/d/g × s1–3 (31) | yes | twins (C5 arms: own, none, swap; C2, C6*: own, none; C0′, P0: own) |
| size | T5 SmolLM2-135M: P0, C0′, C2, C5 × s1–3 (10) | yes | twins |
| hosts | T5 Qwen3-0.6B / 1.7B: P0, C0′, C2, C5 × s1–2 (14; seed 3 when trained) | yes | twins (`-qwen3-v1`) |
| natural | T4 seed 1, SmolLM2-360M / 135M (8); every T4 config once trained | yes | natural T4 |
| controls | T1, WordNet seed 1 (8 + 8) | yes | — |
| new vocabulary | T7 (14, once trained) | yes | — |
| licensed | T1c (20, once trained), `--licensed` | aggregates only | — (item files would hold SNOMED names) |

## 9. Exclusions

- A twin pair counts in a contrast only if both twins link to their new entries in every run of that contrast (counts
  reported). Natural anchors whose surface does not link are excluded.
- Probe A's `context` condition omits entries with no occurrence in the evaluation windows. Probe B omits relations that
  cannot be fitted (§4) and concepts whose representation is missing (hidden states of unlinked surfaces).
- On large tracks probe A and B read at most 3,000 entries per subset (§3).

## 10. What this does not test

- Generation: every item is forced choice or likelihood.
- Trained role readout (step 2), chained unbinding, reverse lookup and capacity (step 3).
- Role items on T1c: the item files would hold licensed names; T1c gets probe aggregates only.

## 11. Smoke tests

Smoke tests run after the commit of this document and are labelled SMOKE: short subsets on the CPU, and at most 4 GB and
5 min of GPU. They report timing and pipeline checks only, never a result.

## 12. Amendments

(none yet)
