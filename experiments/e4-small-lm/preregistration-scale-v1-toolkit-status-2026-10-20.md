# Phase 1b toolkit pre-registration: status note, 2026-10-20 (C5teach teacher spend)

**Not an amendment.** Nothing in §§1–4 of `preregistration-scale-v1-toolkit.md` changes: the arms, the configs, the
comparisons, strata, statistics, predictions and decision rule stay as registered on 2026-10-10. This note records the
author's approval of the teacher spend, the cap, the spend so far and the preparation of the C5teach jobs.

## Approval

- **Author, 2026-10-20:** "C5teach (E4 with Claude-read ontologies): approve about USD 73 in teacher calls", plus "run
  them and queue".
- **What the USD 73 covers:** §6's projection for the **hybrid** variant (C5teach), from
  `scale-v1-toolkit/teacher-pilot/projection.json`: 25,282 synsets × USD 0.002906 per synset = USD 73.47 (≈ 1.1 h at 4
  workers). The digest `manuscript/status/10-20-2026/03-e4-small-models-from-scratch.md` §03.4 quotes the same figure.
- **Not approved:** the full variant C5teachF (117,659 synsets, ≈ USD 342). It is not built and not queued. Under rule
  4.2 it needs its own approval, and only if the hybrid advances.

## Cap and spend

- **Cap:** USD 80 of new teacher spend, hard. `c3_teacher teach --max-usd` is cumulative over the store's `ledger.jsonl`.
  That ledger stood at USD 2.4090 before 2026-10-20 (the pilot's 34 calls), so the cap passed is `--max-usd 82.40`. The
  pilot's calibration call (USD 0.071) is in `teacher-pilot/ledger-calibration.jsonl`, outside the store.
- **Spend on 2026-10-20 so far: USD 0.0682** (1 call, 25 synsets, 0 failed, 15.5 s; model `claude-opus-5-5`, effort
  `medium`, prompt `c3-teacher-v1`). The store ledger now totals USD 2.4772.
- **The call was the build smoke** (`teach --scope hybrid --limit 25`, new spend capped at USD 0.50). Its batch is the full
  build's first batch, so the build reuses it from the cache at no cost.
- **Smoke answers:** 25 of 25 senses answered, 78 proposed edges, 96.2% mapped (pilot 97.3%), 3 unmapped (`no_ancestor`).
- **Full build: not run on 2026-10-20.** The session's permission system refused the spending command. The build waits
  for the author to run it, or to permit it:

      PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher teach --scope hybrid --max-usd 82.40
      PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher ontology --scope hybrid

- **Remaining work:** 24,670 synsets in 987 calls, ≈ USD 71.7 and ≈ 1.1 h at the pilot's rate.
- **Hybrid scope:** 11,800 entries (5,900 held-out + 5,900 matched trained, no overlap) and 25,282 synsets. 612 of the
  synsets are answered: 587 by the pilot and 25 by the smoke.
- **Licensing:** only open-licence text goes to the teacher: WordNet 3.0 lemmas, parts of speech and glosses (NLTK). No
  corpus text, and no clinical or licensed data.

| after the build (to fill in) | value |
|---|---|
| new spend on 2026-10-20 (USD, ≤ 80) | |
| synsets read / failed calls | |
| teacher entries in `ontology-hybrid.pt` | 11,800 (expected) |
| parse rate, edges per entry | |
| `answers.jsonl` sha256 (in the ontology's `teacher` record) | |

## Built without spend (2026-10-20)

- `~/data/vsa-llm/c3-teacher/wordnet-gpt2-v1/reference-hybrid-1024x1024.npz` (`c3_teacher reference`, sha256
  `9399d45354ec986386a1a4a4f9491d37fc841a5d5c1def044bd60effd8047a72`). It was checked against phase 1a's 50M C5
  evaluation windows:
  - `ref_after_teacher_heldout` holds 53,768 targets, equal to `after_heldout`, as §1 requires;
  - `ref_after_teacher_trained` holds 51,951 targets.

## Jobs prepared, not queued

- **Queue script:** `experiments/e4-small-lm/queue-commands-c5teach.sh` (not executed). Its jobs:
  - C5teach;
  - C5@teachref and C0@teachref;
  - two CPU-lane reports.
- **Setup:** the configs are the existing `configs/scale-v1-toolkit/` ones (identical to phase 1a's 50M C5 / C0 configs
  except the arm keys, checked on 2026-10-20). The priorities are placeholders (`PRIO_TRAIN`, `PRIO_REPORT`). A guard
  refuses to queue until the hybrid ontology exists, has 11,800 teacher entries and no unanswered synset, and matches the
  current `answers.jsonl`.
- **Estimate:** ≈ 2.10 GPU-h, from the measured 2.06 h of phase 1a's `scale-v1-50M-C5-s1` plus 2 × ≤ 0.02 h of rescoring.
- **Reporting notes** (no design change):
  - **Main report.** C0, C5, C5dev and C5teach in one report with `--pool-ontologies --references C5`. `e4_report`'s Holm
    family per stratum is every comparison in the grid. Here that is 5: the registered four plus C5 − C0. Its adjusted p
    values are therefore at least as conservative as Holm over the four.
  - **Teacher reference strata.** These are read in a second report against C5@teachref / C0@teachref (evaluation only,
    so k(L) is undefined there). That report is written to `analysis/scale-v1-toolkit-c5teach-teachref`.
  - **Folders.** Both reports go to new folders. The 2026-10-13 toolkit report is left as it is.

## CPU SMOKE (plumbing only, not a result)

- **Setup:** a partial hybrid teacher ontology built with `c3_teacher.teacher_ontology` (303 teacher entries whose
  synsets were all answered, the rest WordNet). It is checked against C3: same keys plus `teacher`, same entry count,
  held-out entries, training frequencies and dtypes, in-range relations and fillers. Reference strata on 32 × 256 windows.
- **Runs:** tiny C0, C5 and C5teach, then C5@teachref and C0@teachref (evaluation only, from the tiny `final.pt`). They
  were built from the real `scale-v1-toolkit` configs with `scale_toolkit.SMOKE_OVERRIDES` (120 steps, CPU).
- **Training:** everything trained and evaluated. C5teach logged `ref_after_teacher_heldout` and
  `ref_after_teacher_trained`; the held-out one has the same target count as `after_heldout`.
- **Rescoring:** the evaluation-only rescoring reproduced the trained run's final losses exactly (max |Δ| 0).
- **Reports:** both report commands ran. With `--pool-ontologies` C5teach shares C5's cohort; without it, it falls into a
  separate cohort. Peak RSS was 2.9 GB.
