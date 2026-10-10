# E9 — frame swap: does a trained model use a term's own frame?

**Status:** design and pre-registration. Written on 2026-10-09 and committed **before any non-smoke frame-swap evaluation**
(GPU or CPU). The labelled CPU smokes (§7) ran after this commit on a window subset; they check the pipeline and the cost and
change nothing in §§1–6. Any later change is listed in §8 with its date and reason.

**Origin:** decision 64 (author 2026-10-09). On T4 (real chemistry, SmolLM2-360M, 3 seeds) C5 − C0′ on `after_heldout` is
−0.40% [−0.54, −0.24], but the shuffled-frames arm C5sh — trained with every entry, held-out included, reading another
entry's frame (`training.lm.frame_variant`, a derangement) — keeps that gain (C5 − C5sh +0.01% [−0.08, +0.10]) while losing
it on seen, rare and unseen terms (C5 − C5sh `after_unseen` −0.82%, significant). On T5 (synthetic) C5sh is 4.6% worse after
held-out terms. Retraining asks whether a model trained on wrong frames does as well; this test asks, **within one trained
model and without retraining**, whether the gain after a term depends on that term's own frame.

**Code:** `src/vsa_embed/experiments/e9_frameswap.py` (`score`, `report`, `commands`); tests `tests/test_e9_frameswap.py`;
queue commands `experiments/e9-retrofit/frameswap/queue-commands.sh` (not executed by the author of this document).

## 1. Design

A finished run is re-scored on its own evaluation windows and strata (bf16, `training.lm.evaluate`; `own` must replay the
run's final evaluation in `eval_windows.npz`) under variants that change the rows of a **target entry set only**; every other
span reads its own row and the channel stays on.

| Variant | Target entries read | Question |
|---|---|---|
| `own` | their own frame (reference) | — |
| `other` | another target entry's frame: one seeded cycle through the target set (no fixed point; same training status) | is the gain specific to the term's own frame? |
| `other-any` | a random non-target entry's frame | what C5sh gave held-out entries |
| `empty` | nothing (zero row: no injection) | does the injected row help at all, within this model? |
| `mean` | the mean context-free row over non-target entries | does a constant row recover the gain? |

`other` / `other-any` remap the target spans' entry id at the model input: the channel composes the other frame with the
span's own local context (exactly a frame swap for the induced concept factor of every E9 compose arm; a row-source arm gets
the other entry's frozen source vector). Derangements and draws are seeded by (swap seed 0, the run's seed, the target set),
so the three seeds of a model use three derangements.

**Target sets:** `heldout` (never linked in training; primary), `unseen` (training frequency 0, not held out), `rare_seen`
(frequency 1–9). Each is scored in the same job, sharing `own`; its matched stratum is `after_heldout` / `after_unseen` /
`after_rare_seen`. Entries with an empty frame are dropped from the sets and pools.

**Runs** (`queue-commands.sh`): SmolLM2-360M C5, C5sh and C6d × seeds 1–3 on T4, T5, T7, T7-ROOD, T8 and T1c-ROOD;
SmolLM2-135M C5 on T5 (seeds 1–3) and T4 (seed 1); C5 seed 1 on T1 and WordNet (both hosts). T1c-ROOD outputs are
aggregates under `experiments/e9-retrofit/frameswap/t1c-rood/` (no filler strata, no entry ids or names; its `.npz` files
stay out of git).

## 2. Statistics

Per model group (track × host × model), target set, stratum and variant: `variant − own`, token-weighted, pooled over the
seeds per evaluation window (the windows are shared across seeds), with a 95% cluster bootstrap over windows (10,000
resamples, seed 0; `statistics.paired_ratio_bootstrap`), absolute (nats per target token) and relative to `own`'s loss.
p-values are the two-sided percentile-bootstrap p; **Holm over the four variants within each target set × stratum**. Tracks
answer separate questions (different corpora and ontologies) and are not corrected across. Per-seed estimates are reported
as well.

## 3. Primary endpoint

**`other − own` on `after_heldout` (target set `heldout`), SmolLM2-360M C5, per track, pooled over seeds 1–3**, tested
two-sided at α = 0.05 after Holm (§2). The direction is pre-specified: positive = swapping in another held-out term's frame
hurts, i.e. the model uses the term's own frame.

**Reading rule:** a held-out gain of C5 counts as **ontology-specific** only if `other − own` > 0 on `after_heldout` with Holm
p < 0.05. If not, the gain after held-out terms is not attributed to the content of their frames (even when C5 − C0′ is
significant): `empty − own` and `mean − own` then say whether the injected row helps in a non-specific way (a term marker, a
type or frequency prior).

## 4. Secondary endpoints (reported; no confirmatory claim)

1. `other-any − own`, `empty − own`, `mean − own` on `after_heldout` (C5, 360M), per track.
2. All four variants for the `unseen` and `rare_seen` target sets on their matched strata.
3. **C6d specificity:** the same table for C6d (definition-encoder rows): `other − own` on `after_heldout` says whether
   C6d's held-out gain is specific to the term's own definition.
4. **C5sh:** the same table; its `own` frame is the shuffled frame it was trained with.
5. Filler / non-filler split of the matched strata where the track has a filler table (T4, T5), and the strata `after`,
   `unlinked` (locality: should move by much less than the matched stratum) and `all`.
6. SmolLM2-135M, T1 and WordNet (single seed or a null C5 − C0′): descriptive.

## 5. Predictions (from the arm battery, made before any frame-swap result)

| Track | `other − own`, `after_heldout` (C5) | `other − own`, `after_unseen` / `after_rare_seen` (C5) | `empty − own`, `after_heldout` (C5) |
|---|---|---|---|
| T5 | **> 0** (own frame matters; C5sh +4.6% worse after held-out) | > 0 | > 0, at least as large as `other − own` |
| T4 | **≈ 0** (C5 − C5sh +0.01% on held-out: the held-out gain is not frame-specific) | > 0 (C5 − C5sh `after_unseen` −0.82%) | > 0 (C5 − C0′ −0.40%) |
| T1, WordNet | ≈ 0 (C5 − C0′ null) | ≈ 0 | ≈ 0 |
| T7, T7-ROOD, T8, T1c-ROOD | no directional prediction | no directional prediction | no directional prediction |

C5sh: `other − own` ≈ 0 on `after_heldout` (its held-out entries read arbitrary frames) and > 0 on `rare_seen` on T5 (it can
bind seen terms to their shuffled frames). C6d: no directional prediction. `mean − own` ≥ 0 wherever `other − own` > 0.

## 6. What would change the conclusions

- `own_check`: `own` must reproduce the run's stored final evaluation (equal target counts; window-sum differences at the
  level of bf16 GPU nondeterminism, as `e9_rescore`'s `ref_check`). A run that fails is excluded and reported.
- A significant `other − own` < 0 (another term's frame is better than the own) is reported as a failure of the reading
  rule's premise, not as specificity.

## 7. Smoke (CPU, labelled SMOKE)

`experiments/e9-retrofit/frameswap/smoke/`: T4 SmolLM2-360M C5 s1 and C5sh s1 on the first few evaluation windows, all
target sets and variants, fp32 on the CPU (so `own` differs from the bf16 GPU evaluation by float precision only). Aggregate
numbers only; they test the pipeline and the cost, not the hypotheses.

## 8. Changes after registration

**Amendment 1 (author, 2026-10-10, before any `linked` run): the `linked` target set.** On natural text the most
consistent gain is after and inside linked multi-token terms (C5 − C0′ after 3+-subtoken terms: T4 −1.09%*, T7 −0.57%*,
T1c seed 1 −0.93% / −1.82%; inside −1.54% / −0.28% / −0.52%). The arms say much of it is not frame-specific (T4: C5sh
keeps about 80% of it, C5 − C5sh −0.20%*; TransE ties; T7: C5sh ≈ C5). This set asks the within-model question for the
whole channel.

- **Target set `linked`:** every entry the evaluation windows link, held-out entries included. Matched stratum
  `after_len3plus`. Also reported: `inside`, `after_len2`, `after_heldout`, `after_unseen`, `after_rare_seen`.
  Variants `own`, `other` (derangement within the set), `empty`, `mean`. `other-any` is dropped: its pool is the entries
  the windows never link.
- **Separate outputs:** `frameswap/<stage>-linked/<run>/`, so the held-out / unseen / rare outputs and their reports are
  untouched.
- **Primary of this amendment:** `other − own` on `after_len3plus`, pooled over seeds, Holm over the three variants.
  Positive = each term's own vector carries part of the multi-token gain.
- **Secondaries:** `empty − own` (how much of the gain needs the injected vector at all) and `mean − own` (whether an
  average vector does as well).
- **Predictions (from the arms, before any `linked` result):**
  - T4 C5: `empty − own` ≈ +1% (most of C5 − C0′); `other − own` small and positive (≈ +0.2%).
  - T7 C5: `empty − own` > 0; `other − own` ≈ 0.
  - T1c: no prediction (seed 1, no arms yet).
  - T5: large on both.
- **Runs:**
  - finished: T4 (360M C5, C5sh, C6d × 3 seeds; 135M C5 s1), T5 (360M C5 × 3), T1c (360M and 135M C5 s1;
    licensed: aggregates only, `--fillers none`), T7 (360M C5, C5sh × 3);
  - once trained: T7 C6d; T7-ROOD, T8 and T1c-ROOD C5, C5sh, C6d × 3.
