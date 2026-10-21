# E5 amendment 1 (2026-10-20): E5.1–E5.5 retargeted to the scale-v1 125M checkpoints

**Status:** written on 2026-10-20, before any non-smoke E5 run. Approved by the author: "E5 explainability: run it on
the existing 125M scaling checkpoints (needs an amendment)" and "run them and queue". It amends
[`preregistration.md`](preregistration.md) (2026-10-02). Everything this file does not change stays as pre-registered
there: definitions, item sets, sources, statistics, rubrics and reportability rules. Queue commands:
[`queue-commands-scale-v1.sh`](queue-commands-scale-v1.sh). Its author did not run that script.

**Blinding note.** Before the predictions (§A.6) were written, the labelled CPU smokes (§A.9) ran on the real
checkpoints. Only the E5.1 smoke had run (4 evaluation windows, C5 seed 1) and its tables were seen. The E5.1
prediction is therefore **not blind**. It is stated qualitatively, from the prior evidence cited, and the smoke numbers
are not used. The E5.2–E5.5 smokes ran after §A.6 was written.

## A.1 Why the amendment

- The pre-registration names the D4.1 125M checkpoints: C0, C1, C2, the best of C3–C7, and C8 `random_fixed` and
  `untyped`, each with 3 seeds. They were never trained. Decision 54 (2026-10-07) replaced the E4 core with reduced
  screens, and no E5 number has been computed.
- Decision 65's from-scratch scaling screen (`experiments/e4-small-lm/preregistration-scale-v1.md`) left nine 125M
  checkpoints with `final.pt` in `experiments/e4-small-lm/runs/scale-v1/`. Phase 1a trained seed 1 (2026-10-12/13); seeds
  2–3 of C0 and C5 followed on 2026-10-18:

  | Run | Channel | Seeds |
  |---|---|---|
  | `125M-C0-s{1,2,3}` | none | 3 |
  | `125M-C5-s{1,2,3}` | `compose`: HRR, width 256, attentive (key 8), P1 context window 8, gate bias −2 | 3 |
  | `125M-C2-s1` | `free`: width 22 + projector, held-out entries read the mean trained row | 1 |
  | `125M-HRRAdd-s1` | `compose_add`: P(c + U f), composer width 128 + free width 11 | 1 |
  | `125M-HRRCat-s1` | `compose_cat`: P[f ; c], composer width 128 + free width 11 | 1 |

## A.2 How the scale-v1 runs differ from D4.1, and whether the item sets still fit

**Same as the pre-registration assumed:**
- Data: the C3 corpus `~/data/vsa-llm/c3/wordnet-gpt2-v1` (WordNet-linked FineWeb-Edu, GPT-2 BPE, ℓ_min 2).
- Ontology: 101,500 entries over 8,192 atomics and 16 relations, with 5,900 held-out entries.
- Architecture and length: the GPT-2 125M host (12 × 768 × 12, context 1,024) and 500M training tokens.
- Evaluation windows: the trainer's 1,024 fixed windows, which E5.1 reuses ("1,024 at D4.1").

**Different:**
- **Recipe.** The scale-v1 recipe uses 32,768 tokens/step (micro-batch 16 × 2), peak lr 1.333e-3 (∝ 1/width; the
  125M lr check planned before D4.1 never ran), 10M warmup tokens, and cosine decay to 0.1×.
- **Conditions.**
  - C1, C3, C4, C6, C7 and C8 do not exist at 125M.
  - C5 is the opscreen's fixed configuration, not "the best of C3–C7" chosen by a screen.
  - C2 has one seed.
  - The hybrids HRRAdd and HRRCat (one seed each) were not part of D4.1.
- **No semantic head.** `semantic_weight` is 0, so E5.4's `semantic_rank` test does not apply. The pre-registration
  already limits that test to runs with `semantic_weight > 0`.

**Tokenizer and ontology match the E5 item sets, so no item set is rebuilt.** Checked on 2026-10-20:
- The runs' tokenizer is GPT-2 BPE (the corpus manifest's `tokenizer: gpt2`, read by `channel_probes.run_tokenizer`).
- The runs' `data.ontology` is the file named in both item manifests, `~/data/vsa-llm/c3/wordnet-gpt2-v1/ontology.pt`.
- Its alias-table digest, as `resolve_alias_table` gives it today (checks `entry_concepts`, `digest` and
  `heldout_entries` all true), is `c802b817351dc79228e92d6414f916e3c030f3f98d0ffa15b463dd730c8c3d15`. That equals:
  - `ontology_alias_sha256` in `items/c3-heldout-gpt2-v1/manifest.json` and `items/c3-synthetic-gpt2-v1/manifest.json`;
  - `alias_table_sha256` of the evaluation corpus.

`c3-heldout-gpt2-v1` and `c3-synthetic-gpt2-v1` are therefore used unchanged, with 400 concepts and 1,735 prompts
each. The ontology file has no `concept_frames`, so E5.2 rebuilds the per-concept frames from WordNet
(`build_wordnet_ontology`, `--max-degree 16`). The code requires them to reproduce the run's entry schedule exactly and
stops otherwise; the smoke confirmed they do (§A.9). The E5.3 builders select items from the same ontology and alias
table. `build neighbours` refuses runs whose alias tables differ, and all nine runs share one.

## A.3 Retargeted checkpoints and roles

- **"Best composition condition" = C5, seeds 1–3, fixed in advance.** C5 is the only composition condition with three
  seeds, and it is the method as the E4 screens defined it. It is not chosen by any E5 result. On loss, HRRAdd beats C5 at
  125M in scale-v1, but with one seed it cannot carry a three-seed endpoint.
- **References:**
  - C0 seeds 1–3: E5.4 `own` means no channel; E5.5 token rows and `concept_surface`; the E5.3 neighbour system (seed 1).
  - C2 seed 1: E5.4 `own` means the mean-row fallback; the E5.5 free rows; the E5.3 neighbour system.
- **Exploratory: HRRAdd seed 1 and HRRCat seed 1.**
  - They run in E5.1, E5.2, E5.4 and E5.5 and are reported descriptively, with one seed each.
  - They take part in no endpoint and no Holm family of a primary.
  - They contain an attentive composer of the C5 type at half width, plus a free per-entry row. Removing every edge
    leaves the free part, so their `remove_all = channel off` check is expected to be **False**. That is not a failure.
- **Not used:**
  - every 50M or 20M scale-v1 run, the opscreen and the toolkit runs (outside "the 125M scaling checkpoints");
  - the continued-pretraining (D4.4) and T2 runs;
  - the S0 pilot.

## A.4 What runs, part by part (changes in bold)

| Part | Runs | Primary endpoint | Change |
|---|---|---|---|
| E5.1 faithfulness | C5 s1–3 (primary); HRRAdd, HRRCat s1 (exploratory) | comprehensiveness top − random at k = 2 (stratum `k2`) on C5 > 0 in all 3 seeds, Holm p < 0.05 in each | **Controls dropped:** static C3 composer, C8 operators |
| E5.1 probe ablation | C5 s1–3 (`--probes wic,wsd --probe-k 2`) | secondary, as pre-registered | none |
| E5.2 senses | C5 s1–3 (primary); HRRAdd, HRRCat s1 | attention − WN1 accuracy (paired bootstrap over items), CI excluding 0 in all 3 seeds; gold-mass lift > 0 secondary | **"C5/C6" → C5** (no C6) |
| E5.3 neighbours (rare) | reference C5 s1, others C0 s1 and C2 s1 | "strongly related" pairs out of 40 per system, one-tailed Fisher, reference > other | none (pre-registered systems exist); **budget cap** |
| E5.3 neighbours (held-out) | as above | as above | **budget cap** |
| E5.3 edge explanations | C5 s1 | score-2 counts out of 40, Fisher top > random, top > neighbours | **budget cap** |
| E5.3 M3 cards | — | — | **dropped** |
| E5.3 T1-open rare MeSH | — | — | **dropped** |
| E5.4 zero-shot | `c3-heldout` and `c3-synthetic` on C5 s1–3, C0 s1–3, C2 s1, HRRAdd s1, HRRCat s1 (18 evaluations) | `own` − strongest structure-only baseline on `c3-heldout` property accuracy, C5, Holm p < 0.05 in all 3 seeds; `own` (C5) − `own` (C2), item level | **C5 − C2 on seed 1 only**; `c6-devtools` dropped |
| E5.5 frequency | all nine runs | R² of channel rows, C5 vs C2 free rows (and vs C0 `concept_surface`) | **C2 single seed** |

**E5.1.**
- What runs is unchanged: 1,024 windows; k ∈ {1, 2, 4}; 2 random draws; KL and top-1 flips; a cluster bootstrap over
  windows (2,000 resamples); Holm over each run's contrasts.
- **Dropped: the controls** (the static C3 composer and the C8 operators). No 125M run has them. The endpoint is
  within-model (top-k vs random-k draws on the same C5 model), so it can be computed without them. What is lost is the
  125M sanity anchor "top ≈ random when weights are uniform". The unit test
  `test_static_composer_comprehensiveness_equals_random` still covers that property of the code. The fraction of
  multi-edge occurrences with uniform weights is reported for every run.
- Secondary endpoints unchanged: sufficiency, held-out vs seen strata, and probe-level ablation on C5 only.

**E5.2.** The analysis, SemCor items and baselines (WN1; SemCor MFS cross-fitted over document halves) are unchanged.
The pre-registration's best attentive condition "(C5/C6)" becomes C5, because no C6 run exists.

**E5.3.**
- **Judge.** Unchanged: `claude-opus-5-5` through headless `claude -p` (`judging.claude_cli_runner`, the J.1-calibrated
  harness); 3 calls per item over 2 paraphrases; blinded calibration items mixed in; the same reportability rule
  (calibration ≥ 0.9 and κ ≥ 0.6, else inconclusive).
- **Neighbour studies.** The pre-registered systems "best channel (reference), C0, C2 (125M seed 1)" are C5 s1, C0 s1 and
  C2 s1. The primary space is `state`. Selection rules and n = 40 pairs per system are unchanged.
- **Edge explanations.** Built from C5 s1.
- **Dropped: M3 cards.** They need the `cards.json` of a developmental (C6) run, and there is no 125M one. The only
  developmental run with cards is the 50M toolkit run `scale-v1-toolkit/50M-C5dev-s1`, which is outside this amendment.
- **Dropped: T1-open rare MeSH.** It is not a scale-v1 checkpoint. Dropping it also keeps every judge call on
  non-clinical WordNet/FineWeb-Edu text, as the author required.
- **New: judge budget**, see §A.5.

**E5.4.**
- Items, sources, fitting (4,000 seen entries with frequency ≥ 10, seed 0), tests and statistics are unchanged.
- Corpus tests (after-span loss, even-numbered documents only) run on `c3-heldout`. `c3-synthetic` has no corpus tests,
  as built.
- C0 runs `own` only: it has no channel, and the code allows no other source for it.
- **Adapted: the second primary contrast.** `own` (C5) − `own` (C2), item level across seeds, can only use the seed the
  two conditions share, seed 1 (the R5 report pools shared seeds). It has less power than planned. C5's three-seed mean
  against C2 seed 1 is reported descriptively.
- **Dropped:**
  - `c6-devtools-v1` (no C6-devtools run);
  - the host item sets (no pretrained hosts in scope).
- Unchanged: H-E is refuted if `definition_mean` ties `own`, and text-evidence sources are never part of the
  structure-only claim.

**E5.5.**
- Unchanged: the probe (ridge, 5-fold out-of-fold, GCV strength) and the representations.
- **Adapted:**
  - C2 has one seed. The C5-vs-C2 endpoint compares C5's seed-level interval with C2's single value, alongside each run's
    item-bootstrap R² interval.
  - The token counts of the training stream are computed once and saved inside the E5 folder
    (`runs/scale-v1/gpt2-train-counts.npy`), not in `~/data/vsa-llm/cache/`.

**Outputs.** Runs go to `experiments/e5-explainability/runs/scale-v1/<part>/<run>/`. The report goes to
`experiments/e5-explainability/report/scale-v1/`. The name `reports/R5-explainability` stays reserved for the
pre-registered D4.1 E5, should it ever run.

## A.5 E5.3 judge budget (new)

The author's limit is **USD 25 for the whole E5.3 under this amendment** (and USD 0.50 for the smoke). The code enforces
it.

**How the cap works.** `e5_rating grade --max-usd 25 --cache runs/scale-v1/rating/judge-cache` keeps a hard cumulative
cap over `judge-cache/spend-ledger.jsonl`:
- The ledger has one row per call attempt with its USD. Every grade job shares it.
- Before each call, a bound on its cost is reserved: $0.10, then 2 × the largest call seen. The call is made only if
  spent + bound ≤ cap.
- A failed call without a cost envelope is charged the mean call cost.
- The cap is required with `--runner cli`.
- Tests: `test_spend_cap_is_hard_cumulative_and_shared_through_the_ledger` and
  `test_spend_cap_charges_failed_calls_and_is_required_for_the_cli`.

**Priority and stopping rule.**
1. Studies are graded in this order: **(1) neighbours (rare)**, the pre-registered replication of HRRBERT's study;
   **(2) edge explanations** (H-F.2); **(3) neighbours (held-out)**.
2. A study starts only if its projected cost fits the remaining budget. The projection is its uncached calls × the mean
   cost per call in the ledger ($0.04 before any call). A study that does not fit is not started and is reported as
   **"not run (budget)"**. It writes `budget-refused.json` and exits 0, so the queue job does not fail and hold back
   later CPU-lane jobs. The R5 report lists the study as not run.
3. If the cap stops a study partway, because the projection was wrong, the study is labelled **INCOMPLETE** and is
   not reported as a result.
4. Verdicts stay cached, so a raised cap resumes without paying again. Raising the cap needs the author.

**Projection.**

| Study | Items | Calls (3 per item) |
|---|---|---:|
| Neighbours (rare) | ≤ 120 pairs (3 systems × 40, fewer where neighbour lists overlap) + 20 preferences + 10 + 5 calibration | ≤ 465 |
| Edge explanations | 40 × 3 explanations + 10 calibration | 390 |
| Neighbours (held-out) | as rare | ≤ 465 |

Measured costs per call:
- the C5 calibration (J.1): $2.60 / 90 calls = $0.029;
- the pre-registration's rubric validation: mean $0.039;
- this amendment's smoke (§A.9): 9 neighbour calls, $0.2505, mean **$0.0278** (range $0.020–0.053), about 5 s per call.

At $0.028–0.029 per call:
- (1) costs ≈ $13.0–13.5. (1) + (2) ≈ $23.7–24.8, just under the cap. Study (2) runs only if (1)'s measured mean leaves
  room (mean ≤ $0.0292 when (1) needs all 465 calls).
- (3) (≈ $13) does not fit and is expected to be "not run (budget)".
- Edge prompts are longer than neighbour prompts. If their calls cost more than the ledger mean, (2) can pass the
  projection and then be cut by the cap (INCOMPLETE). The cap still holds.

Summary: (1) fits, (2) is marginal and (3) is expected to wait for a raised cap. All three item sets are built on the
GPU regardless, so a later grade costs only judge calls.

## A.6 Predictions (written before any non-smoke run; see the blinding note)

**Original predictions kept.**
- E5.1: top-k ≫ random (experiments.md E5.1).
- The hypotheses and refutation rules: H-F (top ≈ random → refuted), H-C (attention ≤ WN1 → alignment part refuted)
  and H-E (`definition_mean` ties `own` → refuted).

The predictions below are this amendment's.

**E5.1 (not blind; qualitative).**
- Primary: comprehensiveness top − random at k = 2 is > 0 and Holm-significant in all 3 C5 seeds, so **H-F is
  supported**. The weights are learned jointly with the LM and are not uniform. On pretrained hosts, E12's decoding
  faithfulness was +0.36*.
- Sufficiency is in the expected direction (keeping the top-k loses less than keeping a random k).
- Effects are concentrated on seen entries. On held-out strata (`k2_heldout`) the channel effect is about 0 and top −
  random is not significant: at 125M, scale-v1's `after_heldout` C5 − C0 is about 0.
- The hybrids go in the same direction with smaller contrasts, because the free row carries part of the information.

**E5.2.**
- **H-C's alignment part is refuted.** Attention − WN1 is < 0 in all 3 seeds, roughly −0.2 to −0.4. The gold-mass lift
  over uniform weights is about 0 (within ±0.02).
- Reasons: on frozen GPT-2 anchors, E1 had attention − MFS = −0.24 [−0.29, −0.20]. Next-token loss does not reward sense
  selection within a union frame. WN1 is a strong baseline on SemCor.

**E5.3.**
- Neighbours (rare): C5 > C0 on "strongly related" counts with one-tailed Fisher p < 0.05 (C5 about 12–25 of 40, C0
  about 2–8 of 40). C5 > C2 in the same direction, not necessarily significant at n = 40.
- Neighbours (held-out), if run: C5 > C0 and C5 > C2 (C2 gives every held-out entry the same mean row).
- Edge explanations, if run: top edges ≈ random edges (not significant, consistent with the E5.2 prediction).
  H-F.2 (explanations improve over nearest-neighbour explanations) is **not supported**.
- Calibration ≥ 0.9 and κ ≥ 0.6, as in J.1.

**E5.4.**
- `c3-heldout`, C5: `own` beats `none`, `random` and `mean_row` on property accuracy by about +0.01 to +0.04, but the
  primary endpoint (beating the *strongest* structure-only baseline, Holm, in all 3 seeds) is **not met**.
  `graph_projection` or `surface_mean` comes within noise of `own`.
- **`definition_mean` ties `own`** (difference within ±0.02 and not significant), so **H-E is refuted** under its own
  criterion.
- Reason: from scratch at 125M the composed zero-shot rows do not help the LM after held-out terms (scale-v1 `after_heldout`
  C5 − C0 is about 0; at 125M × 3 seeds, +0.18% [+0.04, +0.32]). In E9 on pretrained hosts the own frame beat a random
  frame and the mean row by only about +0.04 in property accuracy.
- `own` (C5) − `own` (C2) on seed 1: small and positive, not significant.
- `c3-synthetic`: `own` − `none` is larger than on `c3-heldout`, because the invented name carries no surface
  information. The comparisons against the strongest structure-only baseline still tie.

**E5.5.**
- R² of the C5 channel rows is lower than that of the C2 free rows by ≥ 0.1. C2's rows for rare entries stay near their
  initialization and those for frequent entries move, so frequency is readable from C2's rows.
- C5's trained rows have R² ≥ their freshly initialized composer's (`concept_rows_init`).
- C0's `concept_surface` R² lies between the two. `token_rows` R² is the highest of all.
- The unit-normalized C2 rows are less predictive than the raw ones.

## A.7 Compute

No E5 job has run on the GPU, so these figures combine measured analogues with the pre-registration's planning figures.

**Measured analogues (`.jobs/*.json`, started → finished):**
- scale-v1 125M training: 500M tokens in 241–247 min, ≈ 34k tokens/s with evaluations. Forward-only is ≈ 3× that.
- the full channel-probe suite at 135M: 2.0–2.2 min per run (`probes-cpt-pilot-SmolLM2-135M-*`);
- E5.4 structure-only zero-shot at 135M / 360M (E9): 0.3–1.4 min per run (`wordnet-SmolLM2-*-zeroshot`).

**CPU smoke timings (§A.9):**
- E5.2: 3,000 SemCor instances in 29 s, about 4 min projected for all 226,036 instances.
- E5.5: under 25 s per run at 3,000 items, including the token counts.

| Part | Jobs | Per job | GPU-h |
|---|---:|---|---:|
| E5.1 faithfulness (1,024 windows × 27 forwards ≈ 28M tokens + KL) | 5 | ≈ 7 min (planning figure; ≈ 5 min forward at ≈ 100k tokens/s) | 0.6 |
| E5.1 probe ablation (wic, wsd, fast settings × 4 policies), inside the C5 jobs | (3) | ≈ 8 min (4 × the 135M full-suite time) | 0.4 |
| E5.2 senses (linking is CPU-bound inside the job) | 5 | ≈ 4 min | 0.35 |
| E5.4 zero-shot, all sources (12 channel evaluations) and `own` only (6 C0 evaluations) | 18 | ≈ 8 min (planning figure) / ≈ 1 min | 1.7 |
| E5.3 item builders (state space over the shared pool × 3 runs; edges) | 3 | ≈ 3–5 min | 0.2 |
| **Total GPU lane** | **31** | | **≈ 3.3** (≈ 4.3 with a 30% margin) |

**CPU lane, 13 jobs, 0 GPU-h:**
- E5.5 × 9: ≈ 3 min each. The first computes the token counts in seconds.
- 3 grade jobs: ≈ 40 min for ≈ 465 calls at ≈ 5 s per call; a refused study takes seconds.
- The report: ≈ 2 min.

In all, ≈ 1.7 h of CPU lane.

## A.8 Limitations fixed in advance

- **Seeds.** Single seeds for C2 and the hybrids. The C5-vs-C2 contrasts (E5.4, E5.5) and every hybrid result are
  single-seed.
- **Recipe.** The recipe is not tuned at 125M (lr ∝ 1/width, never checked there). 500M tokens is below compute-optimal
  for 125M.
- **Conditions.** No controls (C1, C3, C8) and no developmental arm, so E5.1 cannot show a uniform-weight or wrong-
  operator contrast at 125M, and E5.3 has no M3 cards.
- **Judge budget.** At USD 25 the held-out neighbour study, and possibly the edge study, will not be graded. E5.3 results
  remain **LLM-graded estimates**.
- **Real aliases.** `c3-heldout` uses real aliases. A from-scratch model saw these surface forms in training, so it may
  know them. That is why the items exist for from-scratch models only; `c3-synthetic` is the contamination-free check.

## A.9 Smoke checks (labelled SMOKE; not results)

**Setup.** On 2026-10-20, CPU only (`CUDA_VISIBLE_DEVICES=`; the GPU was busy), with the real scale-v1 125M checkpoints
and the real data. Every run was a separate process with a timeout, and peak RSS was ≤ 5.3 GB. Outputs stayed in the
session scratchpad. They check the pipeline only: no number from them is a result, and none is used anywhere.

| Part | Run(s) | Settings | Wall, peak RSS | Checks |
|---|---|---|---|---|
| E5.1 | C5 s1 | 4 windows, k 1 2 4, 2 draws, KL | 365 s, 3.6 GB | `remove_all` = channel off **True** (max abs difference 0.0); uniform-weight fraction 0.000; every stratum and contrast written |
| E5.1 | HRRAdd s1 | 2 windows, k 2, 1 draw | 76 s, 3.7 GB | composer hook works on `compose_add`; `remove_all` = channel off **False**, as expected (§A.3) |
| E5.1 probes | C5 s1 | `probe_ablation`, wic + wsd, k 2, tiny settings (WiC train 60, WSD semeval2007, 2 train occurrences per lemma) | 1,306 s, 4.5 GB | 4 policies + 4 comparisons written |
| E5.2 | C5 s1 | `--limit 3000` | 29 s, 2.5 GB | WordNet frames rebuilt and **reproduce the entry schedule** (no `ValueError`); 116 of 3,000 instances linked to a polysemous entry containing the gold sense (≈ 8,700 projected for SemCor; pre-registration ≈ 7,800) |
| E5.4 | C5 s1 | `c3-heldout` and `c3-synthetic`, first 8 concepts; fit 200, 2 contexts, 20 generator steps, 16 windows | 150 s / 77 s, 5.3 GB | all 10 sources produced; 7 of 8 (`c3-heldout`) and 8 of 8 (`c3-synthetic`) concepts linked |
| E5.4 | C0 s1, C2 s1 | `c3-heldout` subset | 22 s / 70 s | C0 runs `own` only; C2 runs all 10 sources |
| E5.5 | C5 s1, C0 s1, C2 s1 | `--max-items 3000` | 24 s / 16 s / 18 s, ≤ 4.9 GB | training-stream token counts computed and saved (402 KB) and reused; representations per condition as specified (`concept_rows_init` for C5 only) |
| E5.3 build | C5 s1, C0 s1, C2 s1 | neighbours rare and held-out: 2 concepts × top 2, pool 300, 2 calibration items; edges: n 3 | 31 s / 33 s / 22 s, ≤ 4.5 GB | item directories written; all runs share one alias table; `build cards` stops with `cards.json not found` (cards correctly unavailable) |
| E5.3 grade (fake runner) | the 3 smoke item sets | shared cache and ledger, `--max-usd 25` | seconds | analysis, Fisher tests, κ, budget record written |
| E5.3 grade (**real judge**) | 3 items (1 pair, 1 preference, 1 calibration pair; WordNet terms only) | `claude -p`, `claude-opus-5-5`, 3 calls each, `--max-usd 0.50` | 46 s | projection $0.36 fits; **9 calls, $0.2505 spent** (mean $0.0278), not exhausted; κ 1.0; calibration 1/1 |
| R5 report | all smoke folders | `--resamples 50` | 4 s | every section and figure produced |

**Tests.** `tests/test_e5_experiments.py` passes (29 tests, including 2 new tests for the spend cap). The tests of the
E7 code that reuses `SpendCap` (`tests/test_e7_g3.py`) also pass.
