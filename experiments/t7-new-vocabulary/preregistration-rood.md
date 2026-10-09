# E9 on T7-ROOD — the HRRBERT-style holdout on real text (decision 63, holdout H1): pre-registration

**Status:** written on 2026-10-08 and committed **before any T7-ROOD run** (nothing is queued; the queue commands are in
`queue-commands-rood.sh`). No T7 v1 result existed when it was written (T7 seeds 1–3 are in the queue, due ≈ 10–11
October), so the predictions below are made blind to both tracks. Any later change goes to §8 with its date and reason.

**Origin:** decision 63 (author, 2026-10-08) adopts the toolkit methodology
(`manuscript/toolkit-methodology-2026-10.md`, §2 M1 "ROOD-text"). On T4, T1-open and T7 a held-out term's channel row is
untrained, but the term still occurs, unlinked, in the training text (`experiments/t4-chemistry/README.md`, holdout): the
host learns the word from text and only the channel is zero-shot. HRRBERT held its 32 codes out of every training
document. T7-ROOD does the same on T7.

## 1. The track

`t7rood` (`e9_tracks.TRACKS`; build `t7-rood.yaml` → `runs/rood-v1`, data `~/data/vsa-llm/tracks/t7-newvocab/rood-v1`;
details and numbers in `ROOD.md`). It is T7 v1 with one change:

- **Held fixed:** the ontology (8,184 MeSH 2026 SCRs, frames over 3,295 descriptor atomics), the alias table (sha256
  `5df73251…`), the 908 held-out entries (holdout sha256 `5e133b5f…`, the same presample and choice), every evaluation corpus
  (`eval-pubmed`, `eval`, `eval-general`: identical tokens and spans to T7 v1, checked by the build), the evaluation windows
  (4,096 on `eval-pubmed`), the training-token budget (T7 v1's 57,268,320 SmolLM2 tokens) and the 50/50 domain/general mix.
- **Changed:** every training document (PubMed or general) that mentions any name of a held-out record is dropped
  (case-insensitive, whole-token matching over all candidate names of the record, linked or not); the dropped domain
  tokens are refilled from training-side abstracts of the same PubMed files that mention no selected name. The build
  decodes every written training document again: 0 of them contain a held-out name (`ROOD.md` §3).

A side effect, stated in advance: the dropped abstracts also held names of seen records, so seen entries get fewer
training spans on T7-ROOD, and the refill text has no linked span. Both work against C5 on T7-ROOD (fewer channel
updates), so they make prediction P2 harder to meet, not easier.

## 2. Runs

| Block | Host | Models | Seeds | Stage | Priority |
|---|---|---|---|---|---|
| primary | SmolLM2-360M, full fine-tuning (the E9 recipe: 50M tokens, host lr 3e-5, channel lr 1e-3, gate bias 0) | P0, C0′, C2, C5 | 1–3 | `t7rood` | 54.497 |
| secondary | Qwen3-1.7B-Base, LoRA r 64 (host lr 2e-4, `scale_to_host`), the T4/T5 Qwen3 recipe | P0, C0′, C2, C5 | 1 | `t7-qwen3`, `t7rood-qwen3` | 54.4975, 54.4977 |

The comparison track is T7 v1 (`t7`, SmolLM2-360M seeds 1–3, already queued) and, for Qwen3, `t7-qwen3` (queued with
this block). Configs differ from T7's only in their data paths and stage names (`experiments/e9-retrofit/configs/`).

## 3. Primary endpoint

**C5 − C0′ relative loss on `after_heldout`** on T7-ROOD: the final-evaluation loss on the 8 tokens after a held-out
term's span (`eval-pubmed`, 4,096 windows, ℓ_min 2), token-weighted and pooled over seeds 1–3 (P0 pairs with every seed),
as a relative difference with its 95% cluster bootstrap CI over evaluation windows, Holm-adjusted over the three
references (C0′, C2, P0) within the stratum. This is the `e9_report` dimension-1 table (`report/t7rood`), unchanged.
Test: two-sided, α = 0.05 after Holm. Negative = C5 better.

The stratum has power: on the shared `eval-pubmed` corpus the 4,096 trainer windows hold 5,515 held-out occurrences of
302 entries with ≥ 5 occurrences (the whole split: 8,786 and 411; T7 v1 feasibility at ℓ_min 2, `runs/v1/feasibility.json`),
above the E9 criterion of 2,000 occurrences and 300 entries.

## 4. Predictions (written before any T7 or T7-ROOD result)

- **P1 (direction).** C5 − C0′ < 0 on `after_heldout` on T7-ROOD, with the Holm-adjusted CI excluding 0.
- **P2 (ordering against T7 v1).** The relative gain on T7-ROOD is at least T7 v1's:
  (C5 − C0′)/C0′ on T7-ROOD ≤ (C5 − C0′)/C0′ on T7 v1 for `after_heldout`.
  *Why:* on T7 v1, C0′ reads every held-out name in 15,530 of the 68,264 training abstracts (22.7%) during its 50M
  tokens of fine-tuning, so its loss after those names improves without any channel; C5 then adds only what the frame
  carries beyond the text. On T7-ROOD neither model ever reads the names (the hosts' pretraining is unlikely to hold them:
  0 mentions in 300,000 FineWeb-Edu documents by construction), so C0′ cannot learn the word from text and the composed
  row is the only source of what the name means.
  *Test:* the difference of the two relative gains, paired by evaluation window (the windows are identical on both
  tracks), pooled over seeds 1–3, with a 95% cluster bootstrap over windows (10,000 resamples). P2 holds if the point
  estimate is ≤ 0; it is *confirmed* if the CI's upper end is < 0, and *contradicted* if its lower end is > 0.
- **P3 (where the difference comes from).** C0′'s own `after_heldout` loss relative to P0 improves less on T7-ROOD than
  on T7 v1 (C0′ − P0 closer to 0), because C0′ no longer reads the names. This is the mechanism check for P2.

## 5. Decision readings

| T7 v1 held-out gain | T7-ROOD held-out gain (P1) | Reading |
|---|---|---|
| significant | significant, ≥ v1 (P2 holds) | The channel transfers knowledge to words the host never read. T7-ROOD becomes the headline real-world zero-shot number (T7 v1 is its lower bound); E13 runs on T7-ROOD rounds. |
| significant | significant, < v1 (P2 contradicted) | Part of v1's gain needs the host to have read the word (the channel sharpens a word learned from text). Both numbers are reported; the zero-shot claim is limited to the T7-ROOD size. |
| significant | null | The v1 gain depended on text exposure. The real-world zero-shot claim is withdrawn; E13 runs on T5 only (toolkit plan §6). |
| null | significant | Text exposure masked the channel (C0′ learned the word). T7-ROOD carries the real-world claim. |
| null | null | T7 is null (toolkit plan §6: E13 on T5 only; the novelty ordering of R9 is broken). |

## 6. Secondary analyses (no correction across them; Holm within each family as `e9_report` does)

- **Qwen3-1.7B** (seed 1): C5 − C0′ on `after_heldout` on `t7-qwen3` and `t7rood-qwen3`, with the same predicted direction
  and ordering (P1, P2). One seed: the CIs cover evaluation windows only, and this is flagged. It is a replication on a
  modern host, not a test of its own.
- The other E9 strata on T7-ROOD (`after_unseen`, `after_rare_seen`, long-token, `unlinked`, `all`) and general-text
  locality (`eval-general`): reported as for every track. The `unseen` and `rare` strata are not compared entry by entry
  with T7 v1, because their training frequencies differ between the tracks.
- Dimension 2 (quantization) and dimension 3 (new words and edits, `items/{new-words,edits}-t7rood-<family>-v1`): reported as
  for every track.

## 7. What is fixed in advance

- The held-out set, the evaluation windows and the training budget are those of T7 v1; nothing is re-chosen after a run.
- A failed run is rerun from its checkpoint (the queue's retry); no run is dropped for its result.
- The T7 v1 numbers used in P2 and P3 are those of stage `t7` (SmolLM2-360M, seeds 1–3, `report/t7`) as finished.
- The P2 paired-window comparison is a small script over the two stages' `eval_windows.npz` files; it is written after
  this document and changes nothing above.

## 8. Changes after this document

- **2026-10-08 (before any run): the P2 / P3 script.** `src/vsa_embed/experiments/t7_rood_compare.py` implements §4 as
  written: windows pooled over seeds 1–3, P0 standing for every seed, 10,000 resamples. Its queue job is
  `t7rood-vs-t7-report` at 54.4974 (`queue-commands-rood-followups.sh`). A Qwen3 seed-1 run of the same script,
  `t7rood-vs-t7-qwen3-report`, gives the §6 secondary. Nothing in §§1–7 changes.
