# E10.L — the *learn* tool, decompose-then-verify: pre-registration

**Date:** 8 October 2026. **Decision:** 63 (WP TK-L). **Status:** fixed before the synthetic evaluation seeds and before
any queued real run. The only runs made before this file are listed in §10 (dev seed 7 and labelled CPU smokes).
**Code:** `vsa_embed.learn` (the method), `vsa_embed.experiments.e10_learn` (runner), `tests/test_learn.py`,
`tests/test_e10_learn.py`. **Design:** `manuscript/toolkit-methodology-2026-10.md` §1.2, §2 M3, §3.2, §4 stage 1.

## 1. Question

Can a passively trained model's knowledge be made **explicit, correct and calibrated**? The tool turns what passive
training absorbed into proposed edges `(concept, relation, filler)` and accepts only those that pass a held-out test
whose false-acceptance rate is measured on null worlds built from the same data. The goal is not speed: E10.9a refuted
that. The claim at stake is: "the learn tool writes correct edges that the curated ontology lacks, at a measured
false-discovery rate, better than graph-only completion".

## 2. Method (fixed)

**Proposal sources** (`learn.propose`; one proposal per edge, never an edge already in the frame):
1. `decompose` — a concept's **passive vector** is mapped into the store's space by a cross-fitted ridge decoder (5
   folds, GCV strength; targets = the static stores `Σ T_r(a)` of the *erased* frames of the other seen concepts, so no
   erased edge and no concept's own target enters its decoder). It is then decomposed over the trained C5 dictionary
   (atomics and operator, typed candidates from the erased frames) by **warm-start non-negative OMP**. The concept's
   erased frame is forced into the support, and up to **2** new atoms are added (stopping correlation 0.1). Lasso and the
   resonator-style unbind pursuit are implemented as alternatives, not run.
2. `closure` — AMIE-style Horn rules mined on the erased graph of the seen concepts (concept-valued relations only), adopted
   at PCA confidence ≥ 0.8 (support ≥ 2, head coverage ≥ 0.01); their closure edges for probe concepts.
3. `author` (E7 cards) — implemented as an adapter; not run here (E7's jobs are pending).

**Acceptance (M3).** For each proposal `(c, r, a)`, the utility on each held-out observation `y_o` of `c` is
`Δ_o = cos(z_c + T_r(a), y_o) − cos(z_c, y_o)`, where `z_c` is the store of the erased frame. Held-out observations are
host hidden states at `c`'s occurrences in documents the proposer did not read, decoded into the store space by the fold
decoder that did not fit `c`. The control is **head**: the same edge added to 8 same-type concepts whose frames lack
it. Type = the frame's relation signature; types with fewer than 5 members are merged into "other". The statistic is
the one-sided Welch t of `Δ_o` against the control heads' mean utilities, and a proposal needs at least 4 observations.

**Decision rule (primary): `holm+decoy`.** Both components must accept:
- **Holm** over all proposals of a run, α = 0.05 (FWER, valid under any dependence when the p-values are valid).
- **Target–decoy FDR** at q = 0.05 on the held-out utility. The decoys are the proposals that the same pipeline makes in
  the **complete null world**: the same data with nothing erased, so every proposal beyond the curated frame counts as
  false. Threshold τ = the smallest target utility with `(1 + #decoys ≥ τ) / #targets ≥ τ ≤ q`, per source. It is
  **cross-fitted** over two halves of the concepts: a concept's threshold comes from the other half's targets and decoys.

Why both: on dev seed 7 Holm alone was calibrated when the store's dictionary matched the evidence (planted-null false
acceptance 0.6%), but **not** under dictionary misfit (17.9%). The decoys reproduce the misfit (complete-null rate 19.0%
under Holm) and the combined rule held every null at ≤ 1.8% in both regimes (§8). Decoy-only fails whenever the complete
world yields few decoys (T5 positive-control smoke: relabel / swap / permute nulls at 97–100%).
Variants reported: `holm`, `bh`, `knockoff` (knockoff+ with one swap decoy per proposal), `decoy`. MDL is not
implemented.

**Null worlds** (all judged by the same rule at the real run's thresholds; rate = accepted / tested):
- `complete`: nothing erased. Rate measured with the cross-fitted thresholds, so no half's own decoys set its threshold.
- `relabel`: each proposal's relation is replaced by another relation that admits the filler (labels permuted within type).
- `swap`: each proposal's filler is replaced by the same relation's filler of another same-type term.
- `permute`: the proposer reads a same-type term's passive vector (a pipeline-level null).
- `planted` (synthetic only): observations generated without the erased edges.

Every decoy excludes every curated edge, so a null rate is an upper bound: true but uncurated edges count as false.

## 3. Data, checkpoints and arms

| Track | Role | Store / passive / evidence host (SmolLM2-360M, seeds 1–3) |
|---|---|---|
| **T7** (MeSH 2025–26 SCRs) | **primary** | C5 / C2 / C0′ of `experiments/e9-retrofit/runs/t7/` (training at 51–54; due ≈ 10–11 Oct) |
| T4 (ChEBI) | secondary | `runs/t4/` |
| T5 (synthetic glossary) | secondary | `runs/t5/` |

- **Erasure.** Seen concepts are those with training frequency ≥ 10, not held out, and ≥ 2 edges. Each of their
  *content*-relation edges is erased with p = 0.2, keeping ≥ 1 edge per concept. Content relations have ≥ 10 distinct
  fillers (T4: not charge, branch; T7: not record_class, branch_top; T5: not status, cadence, tier). The erasure seed is
  0, the same for every checkpoint seed, so the concepts × seeds table is crossed.
- **Probes.** Seen concepts with ≥ 1 erased edge, a passive vector and ≥ 4 held-out observations; a seeded sample of at
  most 2,000 (the same for every seed). Recall is over the probes' erased edges, including those the dictionary cannot
  type (typed coverage reported).
- **Evidence** (`extract`, GPU). The C0′ host is trained on the same text with no channel and has never seen a frame.
  Its hidden states are taken at layer ⌊L/2⌋, at the last subtoken of each linked occurrence, without injection:
  first the whole evaluation split, then 20,000 training-split windows spread evenly, up to 32 occurrences per entry.
  Occurrences are split by document (seeded hash): half A is the `features` arm's passive vector, half B the test's
  held-out observations (at most 16).
- **Arms.** `c2` (**primary**): the C2 free-table row of the same term, the passive vector named in the brief.
  `features` (secondary): the C0′ half-A mean. `c5full` (**positive control**, not a learn claim): the trained C5 store of
  the *full* frame.

## 4. Primary endpoints (T7, arm `c2`, rule `holm+decoy`, decompose source; pooled over seeds 1–3)

- **L4a.** Met iff all three hold:
  - pooled precision of accepted decompose edges against the erased gold is ≥ 0.80;
  - ≥ 30 edges are accepted in total (so a handful of lucky acceptances cannot meet it);
  - **every** null world's pooled false-acceptance rate is ≤ 5%: complete, relabel, swap, permute; each null must have
    tested at least one proposal.

  Reported with the pigeonhole bootstrap over concepts × seeds (precision and recall CIs) and the Wilson interval of
  each null rate.
- **L4b.** On the shared pool (each erased edge of a probe plus 2 distractors: a random typed filler of the same
  relation, and the same filler under another relation that admits it, never a curated edge), compare the recall at
  precision 0.8 of `decompose` with each of AMIE closure, TransE, RotatE and ComplEx. Each difference must be > 0, with
  its 95% two-way bootstrap CI (concepts × seeds, 2,000 resamples) excluding 0 and Holm p ≤ 0.05 over the 4.
  - The baselines are the WP-PQ2 code (`kg_baselines`): KGE with dimension 64, 100 epochs, 32 negatives, mini-batch 4,096,
    trained on the seen concepts' erased graph; AMIE with its defaults over the concept-valued relations.
  - The decompose pool score is `1 + c/(1+c)` for a selected atom with coefficient c, else its correlation with the final
    residual.

## 5. Secondary endpoints (reported separately; no claim from any one of them)

- **S1 — placement, MeSH 2025 → 2026** (TK-H3L primary set).
  - Store T7: C5, seeds 1–3, test split, `t7_group == "eval"`. SCRs are placed through `mapped_to`; descriptors through
    `mapped_to` as a proxy, since T7 has no parent frames. Both are reported per kind; "seen" is a sanity check; dev is
    reported apart.
  - Store T1: seed 1, every item, through `parent` (SCRs by proxy).
  - Candidates: `candidates: null` means the store's typed candidates of the relation. Gold outside them counts as not
    placed; coverage is reported.
  - New-term vector: the C0′ host's mention of the name and up to 7 aliases (`We discussed {x}`), decoded. The decoder is
    fitted on the store entries' own mentions (`extract-items --store-entries`).
  - Store score: unbind + typed cleanup; `correlate` is reported as an alternative decoding.
  - Baselines:
    - **text-only**: the cosine of the term's host vector with the centroid of the candidate parent's known children;
    - **TransE / RotatE / ComplEx**: trained on the store's frames, with the term reached through a ridge map from host
      vectors into the entity space.
  - Metrics: MRR, Hits@1/5/10. The store − baseline difference in reciprocal rank uses the two-way bootstrap (items ×
    seeds), with Holm over the 4 baselines.
  - Secondary sets: `placement-min1` (≥ 1 abstract) and MeSH 2024 → 2025.
- **S2** — L4a / L4b on T4 and T5 (same settings).
- **S3** — the `features` arm (hidden-state passive vector) on T7, T4 and T5.
- **S4** — `c5full` positive control. If it fails, i.e. accepted precision < 0.8 or proposal coverage < 0.5 on T7, the
  primary result is read as **inconclusive (machinery)**, not as "no learn".
- **S5** — the variant rules (`holm`, `bh`, `knockoff`, `decoy`), each with its null rates.
- **S6** — the `closure` source: accepted precision / recall.
- **S7** — TaxoExpan SemEval noun and verb (official split, primary) on the WordNet-track store, seed 1. TMN noun and verb
  are secondary: a re-drawn split, not comparable to published TMN numbers, and the WordNet track trained on 99% of
  their test synsets. Items and outputs stay local.
- **S8** — synthetic calibration (§6).

## 6. Synthetic calibration (CPU, in the repository: `runs/synthetic-v1`)

E10.0's planted world (`make_ontology_world`, d = 64, 304 concepts, 12 observations per concept) with evaluation seeds
101 / 202 / 303:
- proposer: the mean of observations 0–3;
- held-out observations: 4–11;
- every source, null and baseline as above, plus the planted null;
- two dictionaries: `prior` (noisy, cos 0.8 to the teacher: misfit) and `truth` (oracle).

Reported per rule:
- mean [95% t] over seeds of precision and recall;
- pooled null rates;
- pool AUC and recall at 0.8.

Expectation from dev seed 7: `holm+decoy` keeps every null ≤ 5% in both arms, with precision ≈ 0.86 / recall ≈ 0.59
(`truth`) and ≈ 0.68 / 0.07 (`prior`). The synthetic result does not decide L4a / L4b.

## 7. Statistics

As the program:
- the pigeonhole (two-way) bootstrap over concepts × seeds (`statistics.two_way_cluster_bootstrap`'s design; ratio
  statistics recomputed on the weighted pool), 2,000 resamples;
- Holm within each endpoint family;
- Wilson intervals for null rates;
- mean [95% t] over seeds for synthetic summaries.

Seeds are the three checkpoint seeds of each track; the erasure, probes and nulls use seed 0.

## 8. Readings (fixed now)

| Outcome | Reading |
|---|---|
| L4a met, L4b met | the learn tool makes passive learning explicit on real data at a measured false-discovery rate, and ranks erased edges better than graph-only completion |
| L4a met, L4b not | calibrated and correct, but no better than graph completion at ranking: "explicit, not superior" |
| L4a fails on a null rate > 5% | the M3 test is **not calibrated** on real data: **no learn claim**; a methods result (which null failed) |
| L4a fails on precision < 0.8 or < 30 accepted (nulls ≤ 5%) | calibrated but cannot certify edges from C2 rows: **no learn claim** on the primary; S3 (hidden states) and S4 (positive control) say whether the passive vector or the evidence limits it |
| L4b met, L4a not | decomposition ranks erased edges better than graph completion but cannot certify them: "ranking only" |
| S4 fails | **inconclusive (machinery)** |
| S1: store − text-only and − KGE > 0 | the store places new MeSH records better than text similarity and graph embeddings |
| S1: store ≤ text-only | placement by decoding is not supported; the store adds nothing over the host's own text similarity |

Smoke evidence (§10) makes the "no learn claim" rows likely. They are pre-registered as legitimate outcomes, and the
E13 stage-1 gate L4 reads them as written.

## 9. Queue (not queued by the author of this file)

The commands are in `queue-commands.sh` (`e10_learn queue`). There are 81 jobs, all evaluation-only on existing or
pending checkpoints, inside the 54.4995 slot:

| Priority | Lane | Jobs |
|---|---|---|
| 54.4995 | GPU | T7 evidence extraction (seeds 1–3); T7 store-entry and MeSH item vectors |
| 54.49951 | CPU | T7 erased runs (3 arms × 3 seeds) |
| 54.49952 | CPU | T7 MeSH placement |
| 54.49953 | GPU / CPU | T1 MeSH placement |
| 54.49954 | GPU | T4 / T5 evidence extraction |
| 54.49955 | CPU | T4 / T5 erased runs |
| 54.49956 | GPU / CPU | WordNet TaxoExpan / TMN |
| 54.49959 | CPU | reports (`-report`) |

Estimates are ≈ 1.85 GPU-h and ≈ 7.6 CPU-h:
- **GPU:** a 360M forward on 4 CPU threads ran at ≈ 500 tokens/s in the smoke, taken as ≈ 30k tokens/s on the RTX 3090.
  T7 extraction is ≈ 20M tokens, ≈ 0.2 GPU-h.
- **CPU:** erased runs are 0.05–0.3 CPU-h each, scaled from the smokes. T7 jobs sit behind T7's training in priority
  order.

## 10. Disclosures, deviations and what was run before this file

**Dev seed 7** (synthetic, both dictionaries) chose:
- `max_new` = 2 (grid {2, 4} × threshold {0.1, 0.2} × relative coefficient {0, 0.5} × control {head, filler});
- the head control (filler control fails the nulls under Holm: up to 80%);
- the rule `holm+decoy`;
- the decoy statistic `utility`: on dev it separated gold from complete-null decoys at AUC 0.92, against 0.80 for t;
- the decoy normalization: the decoy count directly. The per-proposal N_target / N_decoy scaling collapsed whenever
  decoys were few.

No real-data number entered these choices.

**CPU smokes (label SMOKE, 360M seed 1, not results).** All four smokes below used the 360M seed-1 checkpoints:
- T4: 64 evaluation windows, 300 probes. No proposal was testable. Pool AUC: decompose 0.68, prior 0.76, LRE 0.82,
  TransE / RotatE / ComplEx 0.82 / 0.79 / 0.81, AMIE 0.51. The C2-row decoder's out-of-fold R² was ≈ 0.
- T5: 128 windows, 85 testable probes. Holm accepted 1 edge (gold); every null was ≤ 3%. Pool AUC was 0.49–0.55 for
  every method.
- T5 `c5full`: proposal coverage 0.79; 7 edges accepted at precision 1.0.
- T5 `features`: 0 accepted under the primary rule.

These smokes led to two changes made before this file:
1. evidence extraction adds the training split, because only 17–18% of T4 / T7 seen concepts have ≥ 4 held-out mentions
   in the evaluation split;
2. probes require held-out evidence.

A WordNet / TaxoExpan-noun dev placement smoke (3,000 fit entries, 10 KGE epochs) gave MRR 0.001 (store), 0.039 (text-
only) and ≤ 0.002 (KGE).

**Deviations from the brief:**
1. The primary rule is `holm+decoy`, not Holm alone; Holm alone is a reported variant. Reason: §2, dev seed 7.
2. The **loss utility** (`LossUtilityTest`: LM loss on held-out windows) is implemented but not queued. The existing C5
   hosts were trained with the full frames, so a loss test would read the erased edges back through training. It
   becomes clean in E13 stage 1, where training erases them.
3. The held-out evidence includes training-split occurrences. The C0′ host never saw a frame, and the proposal / test
   halves are document-disjoint.
4. KGE / AMIE see the seen concepts' erased graph, not the full track graph.
5. Placement candidates are the store's vocabulary (typed candidates), not every node of the before snapshot. T7
   descriptors use `mapped_to` as a proxy relation.
6. ICD-10-CM FY2027 (H3) and OET need the licensed T1c store (decision 58; peer session): the evaluator supports them,
   but no job is queued. MedConceptsQA is a ranking set for the TK-B harness, not placement.
7. MDL is not implemented. The knockoff and decoy rules cover FDR control.
