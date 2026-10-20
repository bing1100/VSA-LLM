# E13 — proposed amendment 3 (draft, 2026-10-20): selective recall in stage 3

**Status:**
- **A draft for the author's approval.** Not part of the pre-registration until approved, and nothing is queued.
- It must be approved, implemented and committed **before any E13 stage-3 (`reason`) job runs**.
- No E13 job has run yet. The T5 block starts at 54.4985, with about 235 jobs ahead of it on 2026-10-20.

**Why:** E14's pre-registered decision rule (E14 pre-registration §13) fired. In RD-4b
(`manuscript/findings/2026-10-20-e14-followups.md` §1), SmolLM2-360M C5, 3 seeds:
- **Recall of only the asked relation's lines** before an item raised property accuracy:
  - T4 real chemicals: 0.614 → 0.862 (+0.248*);
  - T5 new words: 0.190 → 0.845 (+0.655*).
- **Recall of the whole frame** did worse:
  - T4: −0.036 against no recall, and −0.284* against selective recall;
  - T5: −0.158* against selective recall.
- **A wrong-partner control** fell below no recall. The model uses the content.
- **The frames' source matters** (property gain over no recall):

  | Frames | T4 | T5 |
  |---|---:|---:|
  | oracle | +0.25 | +0.66 |
  | type prior | +0.10 | +0.62 |
  | linker | +0.05 | +0.28 |

**What E13 does now:**
- Stage 3 (L5) scores `recall:own`: the recall tool's text of the **whole** read store, from the `linker` reader's
  frames.
- E12's scorer already uses a null prompt without the context (checked in `e12_self_query.score_prompts`), so the PMI
  artefact found in E14 does not apply here.

## Proposed changes (stage 3 only)

Stages 0–2 and 4–5, endpoints L1–L4, and every arm's frames are unchanged.

1. **A new condition, `recall-rel:own`.** The same recall tool and store, restricted to the lines of the relation the
   item asks about:
   - relation (`negation/affirm`) and reverse items: that relation;
   - two-hop items: the anchor's lines of the first relation, plus the bridge's lines of the second.

   Scored with E12's scorer, like `recall:own`.
2. **L5 gets a co-primary, L5b:** `recall-rel:own − none` per anchor, with the same statistic as L5 (anchors × seeds,
   two-way cluster bootstrap).
   - The Holm family grows from 6 to 7 tests.
   - L5 itself is kept, so the registered result stays comparable.
3. **Secondary, descriptive:** `recall-rel:typeprior − none`. Stage 3 is scored once more, with the type prior's frames
   written instead of the linker's.
   - This needs no new stage-4 arm.
   - It measures how much a better reader would give (in E14, the type prior recovered most of the oracle's gain on T5).

**Predictions:**
- L5b > 0 on T5;
- L5b ≥ L5;
- `recall-rel:typeprior` ≥ `recall-rel:own` (linker), because the linker's frame F1 on T5 new words is 0.55.

**Cost:**
- Stage-3 evaluations grow by about 50% (1–2 extra conditions): about +2 GPU-h on T5, and the same on T7-ROOD.
- No training.

**Implementation:** about a day's work, in E12's condition parser and E13's `reason`, with a smoke run. It is not
started.

## Decisions for the author

- **Approve as written,** or with L5b as a secondary instead of a co-primary.
- **Whether to add the type-prior reader** (change 3).
- **Alternative:** leave E13 unchanged and report E14's result separately.
