# E9 and E4 — addendum 2 of 2026-10-15: follow-ups and the ontology-richness dose-response

**Status:** pre-registration. Written on 2026-10-15 and committed **before any non-smoke run** of these items. The labelled
smokes listed in §7 ran before this commit. They checked the pipelines and change nothing below. Any later change goes in §11,
with its date and reason.

**Origin:** author, 2026-10-15.
- *"add all proposed ideas and new experiments to the queue"*: the proposed experiments of `manuscript/update-2026-10-15.md`.
- *"the richer ontology task can be queued after the tool kit task"*: the richness dose-response, the author's idea of
  15 Oct, runs after the decision-63 toolkit band.
- The quantization side-module control (B-B1) stays deferred (author, 2026-10-11).
- The teacher spend of about $50 for §7b is approved by "add all proposed ideas" (capped at $60, below).

**Code:**
- `e9_plan` arms:
  - `H5add` (hybrid) and `C1h` (hashed memory);
  - `C5isa`, `C5comp`, `C5half`, `C5w512` (thinned or widened ontology);
  - `C5mesh`, `C5meshsh`, `C5tch`, `C5tchsh`, `C5host`, `C5hostsh` (enriched ontology).
- `training.lm.frame_variant`: `relations:<names>`, `edge_fraction:<p>`.
- New module `experiments/t7_enrich.py` (enriched T7-ROOD ontologies and the gated queue step).
- `e9_report` labels.
- Tests: `tests/test_e9_pq.py`, `tests/test_t7_enrich.py`.

Unless a row says otherwise, everything uses the base batches' recipe, windows, strata and Holm conventions.

## 1. In-context glossary without the cap (T4, T5)

`e9_incontext score --max-prefix 4096` on P0 (s1), C0′ and C5 (s1–3), written to `incontext-full/`. Priority 54.4952,
report 54.49529.

**Why:** at 1,024 tokens the glossary covered only 45% (T4) and 79% (T5) of the window's entries (addendum 1, §3).

**Primary:** as addendum 1 §3, "channel given the glossary" (`C5 frame − C0′ frame`) on the same strata.

**Prediction:** the glossary's own gain (`C0′ frame − C0′`) grows with coverage. `C5 frame − C0′ frame` stays negative
(Holm) on T5's held-out and T4's unseen and long strata, but shrinks compared with the capped run.

**Reading:**
- If `C5 frame − C0′ frame` loses Holm significance on a primary stratum that the capped run had, the "adds on top of
  the definition" claim is limited to the strata where it survives.

## 2. T7 at SmolLM2-135M, seeds 2–3

C0′, C2 and C5, with the full chain and claim-B rescoring. Priority 54.4953.

**Prediction:** seed 1 replicates. 135M's gain after rare and long terms is larger than 360M's (intervals of
135M − 360M exclude 0), and there is no held-out cost at 135M (`C5 − C0′` on `after_heldout` includes 0).

## 3. Scaling screen: seeds 2–3 at 125M (C0, C5)

Priority 54.4952; `e4_scale` report `analysis/scale-v1-seeds` at 54.49529.

**Primary:** C5 − C0 at 125M on `all` and `after_len3plus`, pooled over seeds 1–3, with the seed spread.

**Prediction:** about −0.3% (`all`) and about −1.1% (`after_len3plus`), both significant.

**Reading:** this does not reopen the phase-2 rule, which was computed on seed 1 as pre-registered. It firms up the 125M gap.

## 4. Hybrid in pretrained hosts: `H5add` (T4, T7; 360M; 3 seeds)

The arm is HRRBERT's `compose_add`: C5's attentive composer at half width (128) plus a free per-entry row, matched to C5's
channel parameters (`e4_plan.matched_hybrid_widths`; T4 2.33M against C5's 2.35M, T7 1.10M against 1.10M). It gets the `pq`
chain. Priority 54.4955; reports `report/{t4,t7}-addendum2` at 54.4959.

**Primary:** `H5add − C5` on `after_heldout` and `after_len3plus`.

**Prediction:** `H5add − C5 < 0` on `after_heldout`, because the free row's mean fallback is neutral there and the
composed part is smaller. On `after_len3plus` its CI includes 0.

**Reading:**
- If H5add is not worse than C5 on any stratum and better after held-out terms, the hybrid replaces C5 as the recommended
  design for pretrained hosts.

## 5. Structureless sharing at C5's parameters: `C1h` (T4, T7; 360M; 3 seeds)

A hashed span memory: one bucket per term's token sequence, with the number of buckets matched to C5's budget (T4 2,444
buckets of 960 = 2.35M parameters; T7 1,137 buckets). Terms share rows at random. It has no frames, so no edit or
zero-shot items; rescoring only. Priority 54.4955.

**Primary:** `C5sh − C1h` on `after_len3plus` (T4).

**Prediction:** `C5sh − C1h < 0` (Holm): sharing shaped like an ontology, even with the wrong frames, beats random sharing
at equal parameters.

**Reading:**
- `C5sh ≈ C1h` means any parameter sharing explains the natural-text gain. The ontology's shape does not matter.
- `C5sh < C1h` means the ontology's sharing pattern matters even when the frames are wrong.

## 6. Ontology richness, stage 1: thinner T4 ontologies (135M and 360M; 3 seeds)

One fixed text and test set; only the frames change (`training.lm.frame_variant`, fixed thinning seed). The levels:

| Arm | Frames | Edges / entry |
|---|---|---|
| `C5isa` | `is_a` only | 1.95 |
| `C5comp` | `is_a`, `contains_element`, `charge`, `branch` | 6.44 |
| `C5half` | half of every entry's edges, all 13 types | 4.08 |
| C5 (existing runs) | the full ontology, 13 types | 8.08 |
| `C5w512` | the full ontology at composed width 512 | 8.08 (4.70M parameters, against 2.35M) |

C5sh at 135M (3 seeds) is added as the specificity control at that size. Priority 54.49992, after the toolkit band.
Report `report/t4-richness` at 54.49993.

**Primary:** the dose-response of `arm − C0′` across `C5isa`, `C5comp` and C5 on `after_len3plus`, `after_unseen` and
`after_rare_seen`. Point estimates with 95% window bootstraps pooled over seeds, and `C5 − C5isa` with Holm over the three
strata.

**Secondaries:**
- `C5half − C5comp` (relation types against edge count);
- `C5w512 − C5` (capacity);
- the size interaction, `(C5 − C5isa)_360M − (C5 − C5isa)_135M`.

**Predictions:**
- (a) Monotone: `C5isa` gains least, then `C5comp`, then C5. `C5 − C5isa < 0` (Holm) on at least two of the three strata.
- (b) `C5w512 ≈ C5`: capacity is not binding at about 8 edges per entry.
- (c) No directional prediction for the interaction. The author's hypothesis says larger hosts exploit richer ontologies
  more; at a fixed ontology the smaller host gains more (addendum 1 §5).

**Reading:**
- If (a) fails (the gain is flat in richness), on natural text the channel is a structured-capacity device, not a
  knowledge channel.
- A positive interaction (360M's slope larger) supports the author's hypothesis.

## 7. Ontology richness, stages 2–3: richer T7-ROOD ontologies (135M and 360M; 3 seeds)

T7-ROOD is the clean held-out track: its training text excludes every abstract that names a held-out entry. Its ontology is
thin, mostly MeSH tree codes (5.2 edges per entry, 7 relation types). Three enriched copies are built (`t7_enrich`). In each,
the entries, alias table, held-out set and text are unchanged; only the frames and the dictionary differ.

- **(a) `rood-v1-meshrich`: MeSH heading facts, built now** (deterministic).
  - **New relation types:** `heading_parent`, `heading_action`, `heading_see_also`, `branch_third`.
  - **Effect:** 8,160 of 8,184 entries gain edges; frames grow from 5.15 to 9.47 edges per entry; 1,819 new atomics.
  - **Parameters:** C5 has 1.56M, against 1.10M at base; the shuffled control has the same 1.56M.
  - **Arms:** `C5mesh` and `C5meshsh`, priority 54.49994.
- **(b) `rood-v1-teacher`: facts Claude reads from each entry's MeSH note.**
  - **Source:** the record `<Note>`, bibliographic parts removed. Only **4,011 of 8,184 entries have a usable note.**
  - **Relations:** the base relation set, with fillers resolved through the concept finder.
  - **Budget:** `--max-usd 60`, read in a seeded random order, so a stop at the cap reads a random subset.
  - **Jobs:** the build job at 54.49995; then the gated queue job at 54.499951 queues `C5tch` and `C5tchsh` at 54.499955,
    if ≥ 5% of the entries gained an edge.
- **(c) `rood-v1-hostread` (stage 3): the trained T7-ROOD C5 (360M, seed 1) reads the same notes.**
  - **Jobs:** the build job at 54.49996; the gated queue job at 54.499961 queues `C5host` and `C5hostsh` at 54.499965,
    only if ≥ 10% of the entries gained an edge.
  - **Smoke:** reading 4 notes with a T7 360M C5 run added 0 edges (degenerate completions; see below). The gate is
    therefore expected to skip it.
- **Base comparisons at 135M:** P0, C0′, C5 and C5sh on T7-ROOD at 135M (3 seeds), priority 54.499935. The 360M base runs
  are the toolkit band's.
- **Report:** `report/t7rood-richness` (`e9_report` on `runs/t7rood`) at 54.49999.

**Primary (per enriched variant):** `C5<variant> − C5` on `after_heldout` (T7-ROOD's clean zero-shot stratum) and
`after_len3plus`.

**Secondaries:**
- content against amount: `C5<variant> − C5<variant>sh`;
- the size interaction, as §6.

**Predictions:**
- (a) `C5mesh − C5 < 0` on `after_len3plus`.
- No directional prediction for `after_heldout`.
- (b) `C5tch − C5` is small: about half the entries are read, and the relation set is the base one.

**Reading:**
- An enriched variant better than base **and** better than its own shuffled copy means richer ontology *content* helps.
- Better than base but equal to its shuffled copy means the gain is from more shared parameters, not more knowledge.

**Smokes before this commit (labelled, not results):**
- `t7_enrich mesh`, as above.
- **`read --reader host`:** T7 360M C5 s1 on CPU, 4 notes, 0 edges. The completions repeat a relation label or echo the
  prompt.
- **`read --reader teacher`:** 2 notes, 1 call, $0.06, 1 edge added.

## 8. Reports, priorities and multiplicity

- **Priorities.**
  - 54.4952–54.49533: §§1–3.
  - 54.4955–54.4959: §§4–5.
  - After the toolkit band (54.496–54.4999): §6 at 54.49992–54.49993, and §7 at 54.499935–54.49999.
- **Multiplicity:** each section's primary is read on its own, with Holm within the section's family. No reading depends on
  another section's.

## 11. Changes after this commit

(none)
