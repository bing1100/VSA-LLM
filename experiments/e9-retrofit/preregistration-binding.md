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

### §11 addendum — step 1 smoke tests (2026-10-08; SMOKE, not a result)

Run after `4a45616` (this document's first commit) on real checkpoints of the main checkout, outputs in a scratch directory
(never in run folders); no endpoint or contrast was computed or read. The GPU was shared with a training job (100%
utilization, 15.6 GB in use), so the timings are upper bounds.

- **CPU, T5 SmolLM2-135M C5 s1**, probe with 100 entries per subset (400 entries, 3,074 edges): every condition (static,
  nocontext, neutral, context: 606 occurrences of 310 entries in the 1,024 evaluation windows) × both methods
  (correlation, inverse) in 18 s; probe B on five representations; hidden states link 400 of 400 entries (30 layers,
  middle layer 15); 250 s in all on a loaded CPU (load average 16 on 12 cores).
- **CPU, twins** (10 pairs, sources own / none / swap): 20 of 20 twins link to their inserted entries, every source × item
  kind populates, 13 s. **Natural T4** (20 anchors, T4 SmolLM2-135M C5 s1): 20 of 20 link, 10 s.
- **GPU, T5 SmolLM2-360M C5 s1**, the full probe (4,200 entries, 31,193 edges; 13,320 real-context occurrences; 4,200 of
  4,200 surfaces link): **65 s, peak 2.17 GB**. The first attempt exceeded 5 min: probe B's float64 ridge SVDs run ≈ 1 s
  each on the shared consumer GPU (FP64) against 77 ms on the CPU, so the ridge fits now always run on the CPU in float64
  (`learned_probe(..., cpu)`; the estimator is unchanged).
- **GPU, T4 SmolLM2-360M C5 s1**, the full probe (9,850 entries — 3,000 per subset, all 850 held out — and 80,054 edges;
  9,816 surfaces link): **144 s, peak 2.88 GB**.
- **GPU, twins on T5 360M C5 s1** (300 pairs × own / none / swap; 600 of 600 link): **61 s**. **Natural T4 on T4 360M C5 s1**
  (715 anchors × own / none; 715 of 715 link): **96 s**.
- The T1c inputs (alias table, ontology, evaluation corpus) resolve through the track's loaders; T1c runs are not trained
  yet, so its licensed path is covered by the unit test (aggregates only, relations by index).

**Cost** (GPU-h from these timings, i.e. at shared-GPU speed: upper bounds; Qwen3 scaled by WP-UB's measured throughput
ratios):

| Block | Jobs | GPU-h |
|---|---|---:|
| T5 SmolLM2-360M (31 runs) + 135M (10): probe + twins | 41 + 41 | ≈ 1.05 |
| T5 Qwen3-0.6B / 1.7B seeds 1–2 | 14 + 14 | ≈ 1.35 |
| T4 seed 1 (360M, 135M: P0, C0′, C2, C5): probe + natural | 8 + 8 | ≈ 0.45 |
| T4, every other config once trained | 27 + 27 | ≈ 1.6 |
| T1, WordNet seed 1 (probe) | 16 | ≈ 0.7 |
| T7 (probe, once trained) | 14 | ≈ 0.35 |
| T1c (probe, `--licensed`, once trained) | 20 | ≈ 1.0 |
| **Total** | | **≈ 6.5** |

## 12. Amendments

### 12.1 Origin wording (2026-10-08; no endpoint changed)

Decision 60's text was revised on `main` after this document's first commit (`de0d9cd`): on T5 (SmolLM2-360M, 3 seeds)
untyped bundling ties learned HRR (C5ut − C5 = +0.2%, n.s.), **translation beats it (+4.8%, significant)** and a fixed random
operator loses 5.2% (significant) on held-out terms, while in decision 54's 50M screen every operator ties. The origin
paragraph's "C5tr slightly better" should read so. Nothing in §§1–10 depends on it.

### 12.2 Decision 61 (2026-10-08; step 1 unchanged)

Decision 61 adds two operator families (`spectral_bounded`: learned phases with magnitudes in [0.5, 2]; `block_unitary`:
block-diagonal orthogonal, non-commutative) and a slotted layout (three load-balanced relation groups in slots of d/3,
unitary within each) as readout arms of step 2. Step 1's design, endpoints and runs are unchanged. Steps 1 and 3 read these
arms' checkpoints once they are trained (their probe and item jobs are queued with the step-2 chain, after their
training); their pre-registration, with decision 61's stated predictions, is §13.

### 12.3 Steps 2 and 3 (2026-10-08)

§13 (step 2) is committed with the step-2 code before any step-2 run or GPU smoke; §14 (step 3) likewise before step 3's
runs. Neither changes §§1–10.

### 12.4 Arm-batch report folders (2026-10-08; no endpoint changed)

`e9_plan` planned the readout arms' R9 batch report (`t5-report-s1-2-3-readout-SmolLM2-360M`) with `--output
experiments/e9-retrofit/report/t5 --overwrite`, the folder R9, the claims ledger and the draft cite (the WP-PQ1 report, `dd612dc`),
which `--overwrite` would have deleted before recomputing (found by the coordinator, who changed the queued job by hand to
`report/t5-readout`). Now a batch with arms carries a tag in its job names (`-readout-`, `-pq-`; `e9_plan.batch_tag`) and writes
its report to `report/<stage>-<tag>`; only a base batch (P0 / C0′ / C2 / C5) writes `report/<stage>`
(`tests/test_e9_binding.py::test_no_arm_batch_writes_the_stage_report_folder`). Job names are unchanged. Everything this
program queues follows the same rule: the stage analyses write `report/<stage>-binding`, never `report/<stage>`.

## 13. Step 2 — the unbinding readout arm (pre-registered 2026-10-08, before any step-2 run)

This section is committed with the step-2 code, before any step-2 training run and before its GPU smoke. It folds in
decision 61 (two operator families and a slotted layout) and states decision 61's predictions.

**Architecture** (`src/vsa_embed/readout.py`, `channel.readout`; alternatives recorded, not run):

| Piece | Choice | Recorded alternative |
|---|---|---|
| concept | the most recent linked span injected at p with p ≤ t < p + 32 (prefix-causal) | attention over the preceding linked concepts in the window |
| query | a linear map of the host's hidden state at the input of decoder layer L = round(depth / 3) (SmolLM2-360M: 11 of 32; 135M: 10 of 30) → softmax over the R relations plus "no query" | the final layer (needs a second forward) |
| store | the concept's static frame bundle `Σ_e T_{r_e}(a_e)` (every edge weight 1) | the composer's attention weights (`source: attentive`); the occurrence's context-weighted vector would let attention hide the queried edge |
| unbinding | the operator's primary method (§2; spectral_bounded: adjoint; block_unitary: transpose; slotted: conjugate within the slot) | exact division where defined |
| cleanup | one soft modern-Hopfield step over the shared atomic dictionary, inverse temperature learned (initial 16), restricted to the atomics observed under the relation (slotted: within its slot) | several steps; no type constraint |
| injection | `Σ_r p_r f_r` ("no query" adds 0) → projector (256 → width) → gate `σ(w·[h; P f] + b)`, w = 0 and b = 0 at initialization (C5's gate bias) → added to the residual stream at layer L | — |
| training | the LM loss only, with C5's recipe (full fine-tuning; host lr 3e-5; channel and readout lr 1e-3; 50M tokens; seeds 1–3). The readout's initialization draws from its own generator, so every other initialization equals C5's of the same seed | an auxiliary supervised role / filler loss: at most a secondary variant, not run (it risks training a copying device) |

**Arms** (T5, SmolLM2-360M × seeds 1–3; `e9_plan.READOUT_ARMS`):

| Arm | Operator of the channel and the readout | Unbinding |
|---|---|---|
| U5 | learned HRR (C5's) | correlation |
| U5u | learned unitary HRR (trainable phases) | conjugate (exact) |
| U5sb | bounded spectral circulant: learned phases, magnitudes `0.5 · 4^σ(s)` ∈ [0.5, 2] (condition number ≤ 4; decision 61a) | adjoint; `exact` read by the probes |
| U5bu | block-diagonal unitary: 16 blocks of 16 × 16 rotations, `exp` of learned skew-symmetric matrices; non-commutative (decision 61b) | transpose (exact) |
| U5sl | slotted unitary: the relations split into 3 groups, greedily load-balanced by their T5 edge counts (loads 10,293 / 10,438 / 10,462 edges = 33.0 / 33.5 / 33.5%), slots of 86 / 85 / 85 dimensions (total 256), learned unitary roles within each slot; each filler's coordinates of the slot are its slot vector (decision 61c; `RUN/slots.json`) | conjugate within the slot |
| U5tr | translation (control) | subtraction |
| U5ut | untyped (control) | the bundle readout |

On T5 both twin relation pairs straddle slots (`depends_on` in slot 2 and `part_of` in slot 1; `owned_by` in slot 0 and
`approved_by` in slot 1), so on the twins the slotted layout delineates the swapped roles by construction.

Each readout arm is also **evaluated with the readout gate switched off** (`e9_binding_readout`; decision 61): the
operator's loss without the readout, in place of separate non-readout arms of the new families. **Baselines:** C5 (the same
channel without a readout), C0′ plus an LRE decoder (step 1's probe B on C0′'s hidden states), C2.

**Evaluation chain** (per arm; `e9_plan.readout_jobs`): the WP-PQ1 `pq` chain (track zero-shot, dimension-3 v1) and
rescoring (filler strata); the v2 dimension-3 items; the WP-UB understanding items (`own`: two-hop, reverse, …); the twins
(`own`, `none`, `swap`; a row override also switches the readout off for that entry); the strict non-copy and filler strata
(`e9_freqbias score`); the readout evaluation (losses with the gate on and off on the run's own evaluation windows; role
prediction where the next tokens verbalize a filler of the readout's concept; filler recovery by the readout's vector and by
the store read at the correct role, `oracle`); step 1's binding probe.

**Primary endpoints (step 2):**
- **R1** (decisive: "with a trained readout, binding matters") — twin contrast accuracy (`choice`, `own`), **U5 − U5ut**
  and **U5 − U5tr**, Holm over the two; the units × seeds crossed model as P1.
- **R2** ("the readout helps the model") — **U5 − C5** relative loss on `after_heldout` at the final evaluation (the
  trainer's windows; seeds pooled per window; paired window bootstrap, 10,000 resamples, two-sided).

R1 and R2 answer different questions; no correction between them.

**Decision rules.** R1: (a) both contrasts > 0 (Holm p < 0.05) — "binding matters once a role is read out"; (b) one; (c)
neither — "even an explicit readout trained by the LM loss does not use binding"; (d) a significant negative contrast. R2:
(a) < 0 with the CI excluding 0 — "the readout lowers held-out loss"; (b) CI including 0; (c) > 0 — "the readout hurts".

**Predictions** (written before running). Decision 61's, as the author stated them:
- unbinding fidelity: unitary ≈ bounded ≥ learned HRR (adjoint) ≫ HRR with exact division (probe A on the trained stores;
  exact division of learned HRR is read in step 3's fidelity analysis);
- role-specific readout for untyped and translation ≈ 1/k (the readout evaluation's `oracle` filler recovery at frame degree
  k, against the frame's other same-type fillers);
- block-unitary ≈ circulant at one hop, better at path-in-one-vector order (step 3);
- slotted ≈ single vector when the load is even and worse when uneven, any benefit coming from delineation (T5's load is
  even; the twins straddle slots).

And this document's: R1 (a), with U5 − U5ut ≈ +0.05 to +0.15 contrast accuracy; R2 (a), modest (−1% to −3%); the
readout's role prediction above chance (1/23) at filler positions for every binding arm, at chance-level oracle recovery on
role-ambiguous edges for U5ut and U5tr.

**Refutation readings.**

| # | Observation | Reading |
|---|---|---|
| RB1 | R1 (a) while the readout's role accuracy is at chance | the twin effect comes from the channel, not from the readout's query |
| RB2 | U5ut / U5tr oracle filler recovery well above 1/k on role-ambiguous edges | the typed cleanup supplies the role (type information), not binding |
| RB3 | gate-off loss = gate-on loss for an arm | the readout is unused (check the gate's mean) |
| RB4 | U5bu ≠ U5u at one hop by a wide margin | non-commutativity is not the only difference (parametrization, block size) |

**Secondaries** (Holm within each):
- S2.1 twins: U5u, U5sb, U5bu, U5sl − U5ut; U5 − C5; every arm against 0.5.
- S2.2 readout diagnostics per arm (role accuracy and mass, read and oracle filler recovery; by subset, degree, relation).
- S2.3 each arm's gate-off loss and its on − off difference; the new families' gate-off losses against C5.
- S2.4 loss strata against C5: `after`, `after_rare`, `after_unseen`, `after_heldout_strict`, the filler strata.
- S2.5 WP-UB items (composite, two-hop, reverse) and the v2 dimension-3 items against C5.
- S2.6 step 1's probe on every arm (fidelity: U5u, U5sb, U5bu, U5sl against U5).
- S2.7 C0′ + LRE decoder (probe B, hidden states, held-out) against U5's oracle filler recovery.
- Optional, not queued by default: a 50M from-scratch screen and a Qwen3-1.7B T5 readout pair (U5 against U5ut, seed 1).

### §13 addendum — step 2 smoke tests (2026-10-08; SMOKE, not a result)

Run after `e1bdf4c` (§13's commit), outputs in a scratch directory; no endpoint was computed or read. The GPU was shared
with a training job (100% utilization), so throughputs are noisy upper bounds on cost.

- **CPU (tests):** toy readout arms (U5, U5ut, U5sl, U5bu) train through the real trainer, save and reload with the
  readout (`channel.readout.*` in `final.pt`); the readout is prefix-causal, its initialization leaves every other
  initialization of the run unchanged, a row override switches it off for that entry, and `e9_binding_readout`
  replays the final evaluation with the readout on and differs with it off.
- **GPU, T5 SmolLM2-135M, 30 optimizer steps of 1 × 1,024 tokens** (C5's config with each arm's channel): peak 3.30–3.35 GB;
  tokens/s C5 15.7k, U5 16.9k, U5bu 14.6k, U5sb 13.6k, U5sl 11.0k (the slotted operator loops over its slots). A first
  attempt at 2 × 1,024 tokens peaked at 4.8 GB (C5), above the 4 GB smoke budget, and was rerun at 1 × 1,024. The slotted
  composer recorded the T5 load 10,293 / 10,438 / 10,462 edges.
- **GPU, `e9_binding_readout` on the smoke U5 run** (4 evaluation windows): replay ok, 13 strata with the gate on and off,
  196 diagnosed positions (seen 164, rare 7, held-out 25), 1.8 s.
- **Cost** (`e9_plan --dry-run`): 21 trainings × 67 min × 1.15 (the readout overhead assumed for every arm; the slotted arm may
  be ≈ 1.3) ≈ 27.0 GPU-h, 189 chained evaluations ≈ 7.1 GPU-h, the R9 batch report 0.4: **≈ 34.5 GPU-h**. The optional
  Qwen3-1.7B pair (U5, U5ut, seed 1) ≈ 3.2 GPU-h per run by the memory probe (+ evaluations).

## 14. Step 3 — chained two-hop, reverse lookup and capacity (pre-registered 2026-10-08, before any step-3 run)

Committed with the step-3 code before any chain job reads a trained checkpoint. Algebra only: on each composing run's
trained composer (`e9_binding_chain chain`; the composer is read from `final.pt`, the host is not loaded, so the jobs run
in the CPU lane) and on random vectors (`sweep`). Stores are the static frame bundles; unbinding is the operator's own;
cleanup is typed (within the relation's slot for the slotted layout).

**Runs.** Every composing run of T5 (SmolLM2-360M: C5, C5rf, C5ut, C5tr, C5sh × s1–3; 135M: C5 × s1–3; Qwen3 C5 × s1–2;
the readout arms U5* once trained), T4 (seed 1 now, every other composing config once trained), T1 and WordNet seed 1
(controls), T7 (its fillers name no concept: no two-hop) and T1c (`--licensed`: aggregates only, once trained). The WP-UB
items (`understanding-t5-smollm2-v1`, `-t5-qwen3-v1`, `-t4-smollm2-v1`) add the item-level outcomes on T5 and T4.

**Measures.**
- *Fidelity:* typed filler recovery by frame size k for every unbinding method, plus exact (unregularized) spectral division
  for learned HRR.
- *Two-hop:* up to 2,000 ontology paths per run (an anchor from the probe's entries; its r1 filler names a concept; that
  concept's frame holds (r2, f2)) and the WP-UB two-hop items (the item's path, bridge and five options): the first hop, the
  chained second hop (the argmax filler's own store), the second hop from the true bridge (`oracle`) and a soft chain (the
  candidate bridges' stores mixed by the first cleanup's softmax at β = 16).
- *Reverse lookup:* up to 1,000 (entry, r, F) queries; every concept scored by `cos(c, T_r a_F)`; the entry's filtered rank
  (other concepts holding (r, F) removed); split by whether F occurs under ≥ 2 relations; and the WP-UB reverse pairs (the
  anchor against its partner).
- *Capacity:* one global memory `G_N = Σ_j k_j ⊛ ĉ_j` of N concept stores (N = 1, 4, 16, …, 16,384 and all, in a seeded
  order; keys: the unitary projection of the trained atomic that names the concept, else of a seeded random vector, so key
  unbinding is exact and the loss comes from superposition); typed recovery of up to 300 members' fillers against N; two-hop
  through the global memory of every concept on the first 300 ontology paths.
- *Path order* (decision 61b): 1,000 pairs of opposite-order paths in one vector per m ∈ {0, 2, 6} distractor paths
  (slotted: relation pairs within one slot, since chained binding across slots is empty).
- *Synthetic sweep* (random vectors; descriptive): d ∈ {64, …, 2,048} × k ∈ {1, …, 32} for unitary HRR, Gaussian HRR
  (correlation; exact division), the bounded spectral family (adjoint; exact), block unitary, translation and untyped; one
  global memory against N (k = 8, d = 256); path order (d = 256). Output: `report/binding-sweep`.

**Primary endpoints (step 3)** (T5, SmolLM2-360M, seeds 1–3):
- **C1** (Kumar's contrast on trained atomics) — C5's chained two-hop top-1 through the local stores − through one global
  memory of every T5 concept, on the same 300 ontology paths; paths × seeds crossed model.
- **C2** (binding for chaining) — chained two-hop top-1 among the WP-UB T5 two-hop items' options, **C5 − C5ut** and
  **C5 − C5tr**; items × seeds; Holm over the two.
- **C3** (binding for reverse lookup) — reverse-lookup reciprocal rank on queries whose filler occurs under ≥ 2 relations,
  **C5 − C5ut** and **C5 − C5tr**; queries × seeds; Holm over the two.

**Decision rules.** C1: > 0 with p < 0.05 — "per-concept superposition keeps chaining possible where one global memory fails"
(Kumar's failure, replicated on trained atomics); CI including 0 — no difference; < 0 — the global memory is better. C2:
both > 0 (Holm) — "binding enables algebraic chaining"; one — partial; neither — "role-blind stores chain as well (the typed
cleanup suffices)"; a significant negative — role-blind better. C3: likewise, "binding enables reverse lookup where fillers
are role-ambiguous".

**Predictions** (written before running).
- C1: large and positive — at 4,200 stores in 256 dimensions the global memory is near chance, the local chain near the
  oracle.
- C2: positive but modest — on T5 the first hop's filler is often the only candidate of its type in the frame, where typed
  cleanup solves untyped stores too; C5tr ≈ C5ut.
- C3: positive — an untyped store cannot tell which relation holds F, and ambiguous fillers expose it.
- Decision 61: block-unitary ≈ circulant at one hop (fidelity, two-hop) and better at path order (near 1 against 0.5 at
  m = 0); unitary ≈ bounded ≥ learned HRR with the adjoint ≫ learned HRR with exact division; slotted ≈ single vector at T5's
  even load.
- The soft chain ≥ the hard chain; the oracle second hop ≥ the chained one.

**Secondaries** (Holm within each):
- S3.1 item-level agreement of the algebra with each model's behaviour on the same WP-UB items (P(behaviour | algebra right /
  wrong), φ), including the readout arms' behaviour (their WP-UB evaluation in step 2's chain).
- S3.2 fidelity by k and method per arm (decision 61's ordering).
- S3.3 path order per arm (U5bu against U5u, U5sb, U5 and C5).
- S3.4 capacity curves per arm; local recovery by k.
- S3.5 the readout arms, C5rf and C5sh on C1–C3's measures.
- S3.6 T4 (two-hop items, reverse), T1c aggregates, the 135M and Qwen3 replications; the synthetic sweep (descriptive).

**Exclusions.** WP-UB items whose bridge, answer or options do not map to atomics are dropped (counted in the summary);
ontology paths are sampled only where the first-hop filler names a concept.

**Queueing.** Every step-3 job writes into its run folder (`RUN/binding-chain`) or `report/<stage>-binding` /
`report/binding-sweep`, never a stage's base report folder (§12.4).
