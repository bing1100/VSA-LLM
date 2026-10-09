# T1c-ROOD — HRRBERT's "really out of distribution" protocol on clinical text

**Status:** design and pre-registration, written on 2026-10-08 and committed **before any GPU run**. The CPU data
preparation ran before this commit and produced no model output and no outcome: `select-v1` (the frozen ROOD code set
and its digests), `prepare-v1` (the coding label space and its leakage audit) and `build-v1` (the language-model
corpora, the feasibility of the held-out stratum and their leakage audit). Their aggregates are in `runs/` and in §§3, 5,
9 and 10. Any later change is listed in §15 with its date and reason.

**Origin.** Decision 63 (author, 2026-10-08) approved holdout H2 (T1c-ROOD) of
`manuscript/toolkit-methodology-2026-10.md` §2 M1, work package TK-H2. M1 notes that T1c-F holds its 382 codes out as
*labels* only, with the admissions still in training. T1c-ROOD removes the admissions as well.

**What HRRBERT did** (`resources/vsa-paper.md`, "Really-Out-Of-Distribution", appendix "List of 32 ROOD Codes"):
- 32 ICD codes were withheld entirely from pre-training and fine-tuning;
- every MIMIC-IV patient with any of them (about 30k) formed the test set ("ROOD Overall"), and six patients whose
  records held only those codes formed "ROOD Unseen";
- HRRBase (code embeddings composed from SNOMED CT) was compared with unstructured embeddings on disease prediction
  for those patients. On ROOD Unseen it reached precision 83.5 against 46.2.

**Code.** `src/vsa_embed/experiments/t1c_rood.py` has the stages `select`, `prepare`, `build`, `alias-table`,
`analyze`, `report` and `plan`. T1c-F's `encode` and `train` stages run with this folder's config through four opt-in keys:
`paths.base_root`, `head.free_fallbacks`, `analysis.midranks` and a reordered label space (`source_index`). `icd_coding`
gains `FallbackSource` and tie-aware ranks, `data.mimic.iter_notes` gains opt-in note filters, and `e9_tracks` gains
track `t1c-rood`. Tests: `tests/test_t1c_rood.py` (synthetic fixtures only).

**Configs.** `rood.yaml` (selection and coding: T1c-F's keys and values plus the opt-in keys) and `t1c-rood.yaml` (the
language-model corpus; extends `../t1c.yaml`). **Committed aggregates:** `runs/`. **Licensed outputs:**
`~/data/vsa-llm/t1c/rood-v1/` (mode 700).

## 1. Data availability and the choice of design

- **MIMIC-IV-Note is not on this machine.** The licensed folder `data/` holds:
  - `mimic-iv-3.1/hosp` and `mimic-iv-3.1/icu`: structured tables only (`diagnoses_icd.csv.gz`, `admissions.csv.gz`, …),
    with no note module;
  - MIMIC-III 1.4 (`physionet.org/files/mimiciii/1.4/`, `NOTEEVENTS.csv.gz` included);
  - the group's structured HRRBERT inputs (`mimic-iv_data/{ood,rood}`).
  
  No file name matches `discharge`, `radiology` or `mimic-iv-note`.
- **Design (B) is therefore not implementable.** Design (B) is M1's code-system time split: train on ICD-9-era text and
  test on ICD-10-only codes in MIMIC-IV notes. It needs ICD-10-era note text, which is absent here.
- **Design (A) is built:** HRRBERT's ROOD protocol within MIMIC-III / T1c-F, with ICD-9 codes. The ROOD codes are
  withheld from every training input of every model compared:
  - the coding head's labels and admissions;
  - the language model's corpus, by patient and by mention;
  - the channel's linker.

## 2. Question

When a set of ICD-9 diagnosis codes, their patients and every training document that names their SNOMED CT concepts
are removed from all training, do code vectors **composed from SNOMED CT frames** still assign those codes to the
withheld patients' discharge summaries?

The comparison is with:
- a free code table, which has no trained row for them;
- the T1c-F baselines (TransE, the code title read by the host, GRAM over the ICD-9 tree, random vectors).

A second question: does the language model with the ontology channel (E9 C5), trained without these patients and
mentions, model the text after a ROOD concept better than the same training without the channel (C0′)?

## 3. The ROOD code set (frozen; `select-v1`)

**Universe.** T1c-F v1's framed ICD-9 diagnosis codes (5,589; T1c-F pin `5450b37b…`). A code is eligible if all of the
following hold:
1. It is a T1c-F trained code: not among T1c-F's 382 label-held-out codes, and with at least one training admission.
2. It is carried by at least **10** discharge-summary admissions over all splits, so it has at least 10 test positives.
3. It is **concept-disjoint**: none of its SNOMED member concepts belongs to any other framed code. Its composed vector
   and its concepts are then neither copied from nor trained through another label. T1c-F had 25 held-out codes that
   shared a member set with a trained code; this rule leaves none.
4. It is **hub-free**: every linker entry of every member concept is contained in at most 5 other entries' aliases
   (T1c's `holdout_max_containing`). This keeps the alias-disjoint closure local.
5. It lies in T1c-F's natural ln-frequency bins [−12,−10), [−10,−8) or [−8,−6) (HRRBERT's bins).

**Selection.** Within each bin, the eligible codes are ranked by `sha256("t1c-rood-v1:" + code)`. The first **24 / 16 /
8** are taken, **48 codes** in all, 1.5 × HRRBERT's 32. The quotas follow the natural distribution (rare codes are the
most numerous) and keep the excluded share of patients near 10%, close to HRRBERT's (about 30k MIMIC-IV patients).
**Pin:** `cb7eb61c85ed40c880724439777c960dfb19edaf35e1271dafb25aea345708f6` (sha256 of the sorted list;
`rood.expected_sha256`). The list itself stays under the data root.

**Counts** (`runs/select-v1/summary.json`):

| step | codes |
|---|---:|
| framed codes | 5,589 |
| T1c-F trained, not label-held-out | 5,033 |
| ≥ 10 admissions | 2,084 |
| and concept-disjoint | 1,596 |
| and hub-free (eligible) | 1,257 (538 / 519 / 173 in the three bins) |
| chosen | 48 (24 / 16 / 8) |

- The chosen codes have 10–1,078 admissions each (median 28; 5,734 in total).
- 10 come from one-to-many maps.
- They have 80 member concepts, 2 of which have no alias.

**Concept holdout for the language model.** The 80 member concepts and the closure of their 78 linker entries under
alias containment and shared concepts (T1c's `holdout_closure`) give 123 entries and **125 concepts**. Pin
`34fc579eec1e1658654953dc6130cbb63e955be64a84b7ca0d383f2187fdcfc2` (`rood.expected_concept_holdout_sha256`). This
replaces T1c's 2,069-concept holdout on the ROOD track (§9).

## 4. Exclusion

- **ROOD patients** are the patients with any MIMIC-III admission (all of `DIAGNOSES_ICD`, not only discharge-summary
  admissions) carrying a ROOD code.
  - 4,888 of 46,517 patients (10.5%) and their 8,661 admissions.
  - 5,483 admissions carry a ROOD code; for 499 of them a ROOD code is the principal diagnosis.
  - The exclusion is by **patient**, because T1c splits by patient and a patient's other admissions describe the same
    conditions.
- **Coding head (T1c-F's task).** Every admission of a ROOD patient leaves train and dev. They become split `rood`
  (8,409 discharge-summary admissions).
  - Head training keeps 37,694 of 44,909 admissions and dev 2,248 of 2,699, which removes 16.1% of the train/dev
    admissions.
  - Labels are reordered: trained (4,733), T1c-F's held-out (382, still label-only), ROOD (48) and never trained (426).
    The never-trained group is T1c-F's 174 naturally unseen codes plus 252 codes that lost every training admission.
  - Training frequencies, bins and the denominator are recomputed on the remaining training admissions
    (418,294 label occurrences). The ROOD strata keep each code's natural T1c-F bin (`base_bin`).
- **Language-model corpus** (`t1c-rood.yaml`). The track is T1c (SNOMED CT 2022-05-31 × MIMIC-III notes, 50/50 with
  FineWeb-Edu, 130M SmolLM2 tokens), with three changes:
  1. every note of every ROOD patient is dropped (any category, with or without an admission);
  2. the 125 ROOD concepts are the linker holdout: their aliases never link in training, so the channel never sees them;
  3. every training document, note or general text, that mentions one of the 268 held-out aliases as whole words is
     dropped (H1's document exclusion). Continued pretraining therefore never reads a ROOD concept's name. Only the
     host's original pretraining can have seen the name.
- **What the models still see.**
  - The SNOMED CT frames of the ROOD concepts. These are the composition's input, as in HRRBERT.
  - The ROOD codes' ICD-9 titles, read by the frozen host for the `title` baseline only.
  - The ICD-9 hierarchy (GRAM's ancestors).
  - TransE vectors fitted on SNOMED frames. TransE uses no MIMIC data.

## 5. Test sets

- **Evaluation pool** (12,780 admissions): every discharge-summary admission never used to train or stop a head. These
  are the evaluation-side admissions of non-ROOD patients (4,371) and every ROOD patient's admission (8,409). Each ROOD
  code is scored over the whole pool, so its negatives are never-trained documents too. Training admissions are not
  used as negatives: a head may score documents it has seen differently.
- **ROOD-any** (5,334 admissions: 4,801 train-side, 533 eval-side): admissions with at least one ROOD code. This is
  HRRBERT's "ROOD Overall": every patient with any ROOD code. The ROOD admissions carry 1.07 ROOD codes on average
  (at most 4).
- **ROOD-majority** (13 admissions): admissions in which ROOD codes are at least half of the framed codes. This is the
  closest text analogue of HRRBERT's "ROOD Unseen" (records made only of ROOD codes: 6 patients there). In text an
  admission has about 12 codes, so all-ROOD admissions are almost absent; we checked the size before freezing. With 13
  admissions it is **descriptive only**.
- **ROOD-primary** (482 admissions): ROOD-any admissions whose principal diagnosis (`SEQ_NUM` 1) is a ROOD code, so the
  stay was mainly *about* an unseen condition. **Secondary** (R2 only).
- Why not "all codes unseen": an admission with only unseen codes does not exist in practice (≈ 2–4 in any 30–50-code
  ROOD set we sized). HRRBERT's input was the code sequence itself, so an all-ROOD record meant an unseen input. In
  text the input is the note, so the unseen part is the label, and ROOD-any is the right unit.

## 6. Conditions, encoders and the free head's fallback

- **Head and conditions:** T1c-F's head unchanged, with the conditions `free`, `composed_head` (primary arm), `transe`,
  `title`, `random`, `gram` and `composed_free`. `composed_c5` is added with the C5-ROOD run. Every rule follows T1c-F
  §5: the same hyper-parameters, seeds 1–3, identical batches within a seed, and no tuning.
- **ROOD codes in the composed arms:** in `composed_head` they are composed from their frames by atomics and relations
  trained on other codes only (the induced concept factor has no per-code parameter). In `composed_c5` the frozen
  C5-ROOD composer composes them.
- **The free head has no trained row for a ROOD code.** ROOD ids lie outside the training label set, so they receive
  no gradient (tested). The **pre-registered fallback is the mean of the trained rows** (`free_mean`, the control). It
  is the best label-agnostic guess a free table can make; it gives every ROOD code one score per admission, so its
  per-code AUC measures generic "any diagnosis" signal.
  - Secondary fallbacks: the untouched initial row (`free`, T1c-F's rule and HRRBERT's "unstructured", ≈ chance by
    construction) and the zero row (`free_zero`).
  - The fallbacks need no extra training. The trained `free` head is scored with its never-trained rows replaced
    (`head.free_fallbacks`).
- **Encoders:**
  - **P0-360M (primary):** the frozen SmolLM2-360M. It never saw MIMIC, so its states are clean by construction. These
    are T1c-F's pass-1 states, reused only on an exact manifest match (§15.1).
  - **C0p-ROOD-360M and C5-ROOD-360M:** the seed-1 E9 runs on the ROOD corpus (§9). C5 reads with its channel on, using
    track `t1c-rood`'s alias table.

## 7. Endpoints and decision rules

- **R1 (primary): the ROOD macro-AUC.** The mean over the 48 ROOD codes of the per-code AUC over the evaluation pool,
  averaged over seeds, on P0-360M. Primary contrast: **`composed_head` − `free_mean`**.
  - *Supported:* Δ > 0 with two-sided bootstrap p < 0.05.
  - *Refuted:* the upper 95% bound of Δ is below 0.02.
  - Otherwise *inconclusive*.
- **Specificity** (R1; Holm over competitors): `composed_head` − each of `free`, `free_zero`, `title`, `transe`, `gram`,
  `random`, `composed_free` (and `composed_c5`). The readings are T1c-F §9's:
  - ≈ `gram` (95% CI within ±0.01) or `gram` better: an ICD hierarchy is enough;
  - ≤ `title`: the name read by the host is enough;
  - ≈ `transe`: the structure carries the gain, not the operator;
  - ≈ `random`: no information.
- **R1 per natural bin** ([−12,−10), [−10,−8), [−8,−6)): descriptive, HRRBERT's frequency view.
- **R2 (secondary):** the ROOD positives of ROOD-any, ROOD-primary and ROOD-majority admissions, ranked among the 48
  ROOD codes (zero-shot: MRR, recall@1/5/10) and among all labels (generalized recall@10/100). Ranks count ties as
  half (`analysis.midranks`), so a constant fallback is not ranked first by its ties. Each condition is compared with
  `free_mean` on the MRR, with Holm across conditions.
- **Encoders:** R1 and R2 are replicated on C0p-ROOD and C5-ROOD with the same rules, and Holm within each encoder.
  Whether C5-ROOD helps composed codes more than C0p-ROOD does is descriptive.
- **T1c-F's own endpoints under ROOD** (E1 on its 382 label-held-out codes, E2 slope): computed by T1c-F's `analyze`
  on the ROOD heads; descriptive.
- **L1 (exploratory, the language model):** the loss on `after_heldout` targets of the whole `eval-rood` split, C5 −
  C0′ pooled over seeds 1–3. On track `t1c-rood` the held-out entries are exactly the ROOD closure. The contrast uses a
  cluster bootstrap over evaluation windows, with C5 − C2 and C5 − P0 as context.
  - It is exploratory because the stratum is below the ROOD feasibility bar (§9).
  - A negative Δ with a 95% CI below 0 is reported as "C5 lowers the loss after never-seen clinical concepts". Nothing
    else is claimed from it.

## 8. Statistics

- **R1: a two-way bootstrap, B = 2,000** (T1c-F §8). In each replicate:
  - pool admissions get Poisson(1) weights, and the per-code AUC under those weights is exact (`WeightedAuc`);
  - ROOD codes are resampled with replacement;
  - seeds are resampled within each condition.
  
  One draw is shared by every condition, so the contrasts are paired. CIs are percentile intervals and p-values are
  two-sided bootstrap p-values.
- **R2:** the same admission weights; per-subset weighted MRR.
- **L1:** `statistics.paired_ratio_bootstrap` over evaluation windows (2,000 resamples).
- **Power (planning):** 48 codes with at least 10 positives each among about 12.8k pool admissions. A per-code AUC SD of
  about 0.1 gives a macro-AUC SE of about 0.015 before pairing. The paired contrast against `free_mean` resolves
  Δ ≈ 0.03.

## 9. The language-model corpus and the E9 runs (`build-v1`)

- **Corpora** (SmolLM2; `~/data/vsa-llm/t1c/rood-v1/lm`):

  | corpus | content | size |
  |---|---|---|
  | `train` | 50.0% note tokens (exact) | 130.0M tokens; 120,846 notes + 60,986 general documents |
  | `eval-rood` | discharge summaries of ROOD admissions that mention a held-out alias (whole words) | 4.98M tokens; 1,039 of the 5,980 summaries; 4,865 windows |
  | `eval-mimic` | T1c's first 40,000 evaluation-patient notes | 22.35M tokens |
  | `eval-general` | C3's 5,000 evaluation documents | 5.0M tokens |

  - The training stream drew 30,006 ROOD-patient notes, which were dropped.
  - The mention filter dropped 392 further notes and 51 general documents.
- **The stratum exists only where a ROOD concept is mentioned.** A first build of `eval-rood` over all 5,980 ROOD
  discharge summaries (23.5M tokens) had 1,561 held-out occurrences on 29 entries, 16 of them with at least 5
  occurrences (ℓ_min 2); E9's 2,048 windows held 135. Selecting the summaries that mention a held-out alias keeps
  those occurrences in a smaller split, so the whole-split rescoring is cheap.
- **Held-out (ROOD) stratum of `eval-rood`** (ℓ_min 2): 1,560 occurrences on 29 entries, 16 of them with at least 5
  occurrences, on the whole split; E9's 2,048 windows hold 676 occurrences (12 entries with at least 5). `eval-mimic`
  holds 269 (T1c's evaluation patients rarely mention them).
- **ROOD feasibility bar** (`t1c-rood.yaml`): at least 1,000 occurrences and at least 20 entries with at least 5
  occurrences on the whole split. Decision 17's 300 entries cannot be met by about 50 codes. The bar is **not met**
  (16 entries), so **L1 is exploratory** and is read on the whole split (`e4_quant --windows 4865`, as T1c-2).
- **E9 runs:** track `t1c-rood`, stage `t1c-rood`, SmolLM2-360M, P0 / C0′ / C2 / C5 × seeds 1–3, with T1c's recipe
  unchanged (50M tokens, full fine-tune, channel lr 1e-3, 2,048 evaluation windows on `eval-rood`).
  - There are no per-run probe or item evaluations: T1c's dimension-3 items are built on T1c's holdout, which is
    trained here.
  - The configs are in `experiments/e9-retrofit/configs/t1c-rood/` (`e9_plan --dry-run`).

## 10. Leakage audit (aggregates; `runs/prepare-v1`, `runs/build-v1`)

| check | count |
|---|---:|
| coding: train / dev admissions with a ROOD code | 0 |
| coding: train / dev admissions of ROOD patients | 0 |
| coding: ROOD labels inside the trained range | 0 |
| coding: ROOD labels with a training positive | 0 |
| LM: training spans of held-out (ROOD) entries | 0 |
| LM: decoded training documents mentioning a held-out alias (all 181,832 decoded) | 0 |
| LM: ROOD-patient note texts found among the decoded training documents (texts unique to ROOD patients) | 0 |

- **Positive control** for the note-identity check: 119,815 of the other patients' note texts are found in training.
- **Templated texts:** 1,437 ROOD-patient note texts also belong to other patients (templated reports), and 394 of them
  occur in training through those other patients.
- The decoded training corpus round-trips exactly, because `build_corpus` drops any document that does not.

## 11. What is not claimed

- No ICD-10 and no code-system shift (design (B) needs MIMIC-IV-Note).
- No claim on "all codes unseen" admissions: ROOD-majority has 13 admissions and is descriptive.
- No claim that the host never saw the ROOD concepts' names. SmolLM2's web pretraining can contain them; only E9's
  continued pretraining and the coding head are clean.
- The language-model endpoint is exploratory (§9).
- No state-of-the-art MIMIC-III coding (T1c-F §11 applies).

## 12. Runs and compute (`queue-commands.sh`; not queued)

**Run order** (decision 63's band; decisive first, author 2026-10-08, §15.1):

| priority | jobs |
|---:|---|
| 54.4998 | frozen-host P0-360M encode (reuse of T1c-F's states, §15.1), its coding heads (seeds 1–3) and the R1 analysis |
| 54.49981 | E9 training on the ROOD corpus (P0 / C0′ / C2 / C5 × seeds 1–3) |
| 54.49982 | C0p-ROOD and C5-ROOD encodes and heads, `composed_c5` on P0, whole-split and general-text rescoring |
| 54.49983 | the remaining analyses (P0 with `composed_c5`, C0p-ROOD, C5-ROOD, T1c-F's endpoints under ROOD) |
| 54.49984 | reports (names contain `-report`) |

**Cost** (idle-GPU estimates from T1c / T1c-F's measured costs; total ≈ 25.4 GPU-h):
- E9, 10 runs: ≈ 10.5 GPU-h (`e9_plan --dry-run`: 1.16 per trained run);
- P0 coding and R1: ≈ 2.4 GPU-h when T1c-F's P0 states are reused, else ≈ 3.9 (a separate 1.5 GPU-h encode);
- C0p / C5-ROOD coding: ≈ 8.9 GPU-h;
- whole-split and general-text rescoring: ≈ 1.6 GPU-h;
- analyses: ≈ 0.6 GPU-h.

The P0 block needs no E9 run, so R1 is read after about 2.4–3.9 GPU-h.

## 13. Licence and DUA

- MIMIC-III is credentialed, and SNOMED CT and UMLS are licensed. Decision 58's rules apply unchanged.
- Everything derived lives under `~/data/vsa-llm/t1c/rood-v1/` (mode 700):
  - the code list, patient and admission identifiers, the concept closure and the label tables;
  - the corpora, states, scores and per-code AUCs.
- The committed run folders hold aggregates only. `.gitignore` admits `summary.json`, `metrics.json`, `endpoints.json`,
  `report.md`, `manifest.json` and `resolved_config.yaml` there.
- No stage prints a note, a row, a code, a title or a concept, and nothing is sent to any external service. The
  synthetic end-to-end test checks that the run folders receive no code string and no per-item array.

## 14. Smoke tests

- `tests/test_t1c_rood.py` has 17 tests, on synthetic fixtures only. They cover:
  - the hashed, stratified selection and its pin;
  - patient-level exclusion: no ROOD admission or patient survives into train or dev, ROOD ids lie outside the trained
    range, and a code that loses every training admission becomes never trained;
  - no gradient reaches a ROOD code's free row, and the mean / zero fallbacks;
  - tie-aware ranks;
  - the mention filter, the ROOD document streams and the opt-in note filters;
  - the default-off behaviour of the new keys: the T1c and T1c-F configs are unchanged and so are track `t1c` and
    `base_root`;
  - an end-to-end synthetic train → ROOD analysis run that writes aggregates only;
  - the L1 window pairing.
- T1c-F's and T1c's suites pass unchanged.
- No GPU smoke was run: the GPU belongs to the queue. Every GPU stage is T1c-F's, which has passed its GPU smoke (T1c-F
  §14); ROOD changes only the label space and the scoring passes.

## 15. Deviations and changes after commit

- **15.1 (2026-10-08, author, before any run) — run order and the P0 encode; no design change.**
  - **Order.** Decisive first (table in §12): the frozen-host coding block and its R1 analysis run before E9 training.
    The E9 runs use all three seeds now.
  - **The frozen-host P0 encode** (`encode --reuse-base`) reuses T1c-F's P0 states (job `t1cf-encode-P0-360M`, queued
    earlier at 54.41) only if all of the following hold:
    - their manifest is complete;
    - it matches field by field: host, channel off, token store and its truncation (`tokens_dir`, `max_tokens`,
      tokenizer, token count), chunk and segment sizes, precision, admission count;
    - their segment plan is identical;
    - T1c-F's admission order equals this data root's.
  - **Reuse writes only `REUSED.json`** (the base manifest's sha256) in ROOD's states folder. A base manifest that
    changes later is refused when the states are opened.
  - **When the base states are absent, incomplete, or older than the recorded fields,** the encode runs into ROOD's own
    folder (`rood-v1/coding/states/P0-360M`).
  - **A definite mismatch fails loudly.**
  - ROOD never writes T1c-F's states and T1c-F never reads ROOD's, so neither job can fail or overwrite because of the
    other. The base root is no longer read silently (`resolve_states_dir`).
  - Tests: `tests/test_t1c_rood.py`, encode-reuse tests (synthetic).
- **15.2 (same day, before any run) — the bootstrap's seed draws.** Seeds are drawn per condition from a generator
  keyed by the condition's name, so a contrast does not depend on which other conditions are present. The later P0
  analysis with `composed_c5` (`-pass2`) then reproduces the primary contrast exactly. The estimator is unchanged.
- **15.3 (2026-10-09, decision 64, author; before any T1c-ROOD run) — E9 control arms, two reading rules for L1, and a
  secondary untyped coding head.**
  - **Timing.** Every T1c-ROOD job is pending (54.4998–54.49984): no encode, head, analysis, E9 run or report has run.
  - **Why.** On T4 (real chemistry, SmolLM2-360M, 3 seeds), C5 − C0′ on `after_heldout` is −0.40% [−0.54, −0.24], but
    shuffled frames keep that gain (C5 − C5sh +0.01% [−0.08, +0.10]) and the definition encoder beats C5 there (C5 − C6d
    +0.17% [+0.05, +0.29]). On T5, C5sh is 4.6% worse and C5 beats C6d by more than 10%. In E12, a fixed random binding
    (C5rf) decodes as well as a learned one. Whether binding matters for the composed coding head is therefore open too.
  - **E9 arms added** (track `t1c-rood`, SmolLM2-360M, seeds 1–3, the §9 recipe; configs in
    `experiments/e9-retrofit/configs/t1c-rood/`; queued by the coordinator at 54.49981 with `--no-evals`, with the
    row-source table job `t1c-rood-rowsource-definition-SmolLM2-360M`):
    - **C5sh**: shuffled frames (every entry reads another entry's frame);
    - **C6d**: the definition encoder (the frozen host's mean-pooled hidden state of the entry's verbalized frame,
      through a trained projector at C5's site).
  - **Where they are read.** The stage's E9 report (`t1crood-e9-report`, 54.49984) reads every run of the stage. Its arm
    table gives C5 − arm on `after_heldout` of the runs' final evaluation: 2,048 windows of `eval-rood`, where the held-out
    stratum is exactly the ROOD closure. The whole-split rescoring jobs (`t1crood-quant-rood`, `-general`) list P0, C0′, C2
    and C5 only, so the whole-split L1 has no control. The controls are read on the 2,048 windows.
  - **Reading rules for L1** (exploratory, like L1 itself):
    - **(a)** §7's statement "C5 lowers the loss after never-seen clinical concepts" may be called
      **ontology-specific** only if C5 − C5sh < 0 on `after_heldout`, with the Holm-adjusted p < 0.05 and the CI
      excluding 0.
    - **(b)** It may be called "better than standard new-word vectors" only if C5 − C6d < 0 there, with the same test.
    - Otherwise L1 is reported without either qualifier.
  - **Within-model specificity.** The held-out frame-swap rescore (`e9_frameswap`, pre-registered separately in
    `experiments/e9-retrofit/preregistration-frameswap.md`, being built) swaps held-out entries' frames at evaluation on the
    trained C5 itself. Its rules are its own.
  - **Coding head: `composed_head_untyped` (a secondary specificity condition).**
    - **Definition.** `composed_head`'s composer with no binding (`operator: untyped`, `v_e = a_e`): the attention-weighted
      bundle of the frame's filler atomics. The relation still enters the attention keys, as in E9's C5ut. Dimension,
      attentive composition, induced concept factor, key 8, head, batches and stopping are otherwise identical. It
      separates frame content (which fillers) from binding (which role each filler plays).
    - **Seeding.** It is seeded by its name (`NAME_SEED_BASE` + its index in `CONDITIONS`), not by its position in the
      job's list, so it is the same whether trained with the other conditions or alone (same seed, same batch order:
      tested). Every pre-registered condition keeps its positional seed.
  - **Statistics (outside every pre-registered family).**
    - It is **not** in R1's primary contrast, the specificity family or the vs-control families (its vs-control row is
      unadjusted and flagged `secondary`). The pre-registered contrasts are identical with and without it (tested).
    - **Its own family:** `composed_head − composed_head_untyped` on R1 and on the ROOD-any MRR, with Holm over the two
      (the `binding` block of each `analysis-rood-<encoder>` report and of the final report).
    - The same contrast on E1 and E2 enters T1c-F's descriptive endpoints under ROOD.
  - **Readings, per encoder:**
    - Δ > 0 with Holm p < 0.05 and the CI excluding 0: binding contributes beyond the frame's content.
    - CI including 0: the composed head's result is carried by which fillers a code has; binding is not shown to matter.
    - Δ < 0 (significant): bundling without roles codes better.
    - R1, its decision rules and §7's specificity readings are unchanged.
  - **Jobs.** `rood.yaml` lists the condition after the seven of §6, and the head jobs of every encoder gain it.
    - The nine pending head-training jobs are replaced: P0-360M at 54.4998, and C0p-ROOD and C5-ROOD at 54.49982.
    - The `composed_c5`-only jobs (`-c5dict`) are unchanged. The analyses pick the condition up without a command change.
    - Commands: `experiments/e9-retrofit/queue-commands-decision64-rest.sh`, part 2. They add ≈ 0.12 GPU-h per head job,
      ≈ 1.1 GPU-h in all.
  - **Code and tests.**
    - Code: `t1c_icd_frequency` (`SECONDARY_CONDITIONS`, `condition_offset`, `binding_contrasts`) and `t1c_rood`
      (families, `binding`, report, plan).
    - Tests: `tests/test_t1c_rood.py`, on synthetic fixtures only. No stage ran on MIMIC, SNOMED CT or UMLS data for this
      amendment.
  - the mean-row fallback is the primary control, with the initial (random) and zero rows secondary;
  - training documents mentioning a ROOD name are dropped;
  - `eval-rood` is selected by mention;
  - E9 runs all three seeds.

## 16. Build record

`runs/build-v1/report.md` and `summary.json` hold every number of §§9–10. The CPU build took 534 s: 4 tokenizer workers,
then the decoded-text audit. The first build, whose `eval-rood` held all 5,980 ROOD summaries, is summarised in §9; its
summary is kept under the data root (`build-v1-first-summary.json`). The training corpus was rebuilt identically: same
stream, same counts.
