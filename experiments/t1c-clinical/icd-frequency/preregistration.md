# T1c-F — Frequency bias and never-trained codes in clinical code assignment from notes

**Status:** design and pre-registration, written on 2026-10-07 and committed **before any GPU run**. The CPU data
preparation ran before this commit and produced no model output and no outcome: `prepare-v1` (counts and the frozen
holdout digest), `tokenize-v1` (token counts) and `kge-v1` (the TransE fit). The
labelled smoke tests (§14) run after it; they check that the pipeline works and estimate cost, and change nothing in
§§1–12. Any later change is listed in §15 with its date and reason.

**Origin:** the author, 2026-10-07: "measure the frequency bias improvement like in the original HRR paper with MIMIC
and how ontologies improve frequency bias". It builds T1c's proposal (a) (`resources/plan-improvement/execution.md`,
T1c section): rare and unseen ICD-9 coding from MIMIC-III discharge summaries with a pre-registered code-level holdout.

**The paper it follows.** HRRBERT (`resources/vsa-paper.md`) composed SNOMED CT HRR embeddings for ICD codes in a BERT
over MIMIC-IV code sequences and reported three frequency results:
1. the unstructured code embeddings cluster by frequency in a t-SNE;
2. masked-code top-10 / top-100 accuracy, in 7 bins of ln relative frequency (width 2, −14 to 0), against
   unstructured embeddings with Dunnett's test — HRRBase was weaker on frequent codes and non-zero in the rarest bin;
3. ROOD: on codes never seen in pre-training, precision 83.5 against 46.2.

T1c-F asks the same three questions of a **text** task on the licensed clinical track: assigning ICD-9 diagnosis codes
to discharge summaries with a causal host.

**Code:**
- `src/vsa_embed/icd_coding.py`: bins, the holdout, label sources, the head, training, metrics, statistics, probes;
- `src/vsa_embed/experiments/t1c_icd_frequency.py`: stages `prepare`, `tokenize`, `kge`, `encode`, `train`, `analyze`, `plan`;
- tests: `tests/test_icd_frequency.py` (synthetic fixtures only).

**Config:** `experiments/t1c-clinical/icd-frequency/icd-frequency.yaml`. **Committed aggregates:**
`experiments/t1c-clinical/icd-frequency/runs/`. **Licensed outputs:** `~/data/vsa-llm/t1c/icd-frequency-v1/` (mode 700).

## 1. Question

**Main question.** In ICD-9 coding of discharge summaries, do code vectors **composed from SNOMED CT frames**, compared
with free learned code vectors:
- (a) **reduce frequency bias**, so that performance depends less on how often a code was trained; and
- (b) **give useful scores to codes never seen as labels in training**?

**Sub-questions.**
- **Is it the SNOMED frames?** Or would any hierarchy do (GRAM over the ICD-9-CM tree), any knowledge-graph embedding
  (TransE), or simply the code's name read by the host (the title encoder)?
- **Does the T1c C5 dictionary transfer?** The composer trained inside the language model (E9 C5) is used frozen to
  give code vectors, with no coding-trained atomics.
- **Does the encoder matter?** Encoders are P0 (the original host), C0′ (continued pretraining on T1c) and C5 (the same
  with the ontology channel).

## 2. Task and data

- **Admissions.** MIMIC-III 1.4 `NOTEEVENTS` with category "Discharge summary" (from T1c's extraction), joined to
  `DIAGNOSES_ICD` by `HADM_ID`. Admissions with at least one diagnosis code are kept.
  - Text: the admission's discharge summaries (report and addenda), concatenated in `ROW_ID` order, as in CAML.
  - Patient split: T1c's (`SUBJECT_ID` sha256 bucket < 1,000 of 10,000 → evaluation). A dev split for early stopping
    takes the training-side patients with bucket in [1,000, 1,500).
  - Counts: 44,909 train, 2,699 dev and 5,114 evaluation admissions (`runs/prepare-v1/summary.json`). 22, 2 and 2 of
    them have no framed label and stay as all-negative documents.
- **Codes and the map.** ICD-9-CM → SNOMED CT uses NLM's ICD9CM_SNOMED_MAP (2021-12):
  - the one-to-one concept when there is one;
  - otherwise the **maximal concepts** of the one-to-many set (those with no ancestor in the set), at most 8, ordered by
    how much of the set they subsume.
  
  A code is **framed** if it maps to a T1c concept. That gives 5,589 framed codes (3,881 one-to-one, 1,708
  one-to-many), covering 88.2% of the evaluation label occurrences. The other 1,329 codes are outside the task for
  every condition: every condition scores the same label set.
- **Code frame.** The union of the member concepts' T1c frames: hierarchy, semantic tag, is-a and attribute edges over
  T1c's 8,192 atomics and 62 relations, with 8.7 edges on average. The rebuilt ontology is checked equal to E9's
  `ontology.pt` (concept, atomic and relation names), so the C5 composer reads the same ids.
- **Text processing.**
  - SmolLM2 tokenizer; the first 8,192 tokens of each admission (`runs/tokenize-v1`: median 3,358 tokens, 99th
    percentile 9,814; 2.6% of admissions truncated; 191.7M of 194.3M tokens kept).
  - Independent 1,024-token windows (E9's sequence length).
  - States: the host's last layer after the final norm, mean-pooled over 32-token segments. The head attends over these
    segments.

## 3. Frequency and the bins (HRRBERT's)

- **Frequency.** f(c) = the number of training admissions carrying c, divided by all diagnosis-label occurrences of
  those admissions. The denominator is 527,577 and counts all codes, framed or not. Dev and evaluation admissions are
  not counted.
- **Bins.** ln f in [−14,−12), [−12,−10), …, [−2,0), HRRBERT's 7 bins of width 2. They correspond to training counts:

  | bin | counts | trained codes (codes with ≥ 1 evaluation positive) | held-out codes |
  |---|---|---:|---:|
  | [−14,−12) | 1–3 | 2,135 (323) | — |
  | [−12,−10) | 4–23 | 1,635 (959) | 202 |
  | [−10,−8) | 24–176 | 880 (871) | 131 |
  | [−8,−6) | 177–1,307 | 325 (325) | 49 |
  | [−6,−4) | 1,308–9,662 | 54 (54) | — |
  | [−4,−2) | 9,663–71,399 | 4 (4) | — |
  | [−2,0) | ≥ 71,400 | 0 | — |

  The top bin is empty in MIMIC-III: the most common code holds about 3.4% of the labels. It is reported empty, not
  merged, so the bins stay HRRBERT's.
- **The held-out bin** is the ROOD analogue: 382 codes never trained as labels. It is also reported by each code's
  natural bin.
- **Naturally unseen codes.** 174 framed codes have no training occurrence under the patient split (82 dev and 117
  evaluation positives). They are exploratory only: too few to test.

## 4. The code-level holdout (frozen)

- **Eligible codes.** Framed codes with at least 5 training admissions, in bins [−12,−10), [−10,−8) and [−8,−6).
- **Selection.** Within each bin, codes are ranked by `sha256("t1c-icd-frequency-v1:" + code)` and the first
  round(0.13·n) are held out: 202 of 1,553, 131 of 1,011 and 49 of 374, so **382 codes**. The pin is the sha256 of the
  sorted list, `5450b37b542048e52e6725d8e567bb9f67296a1d35c0f3eeb17944af5a1df298` (config `holdout.expected_sha256`;
  a rebuild that differs fails). The list itself is under the data root.
- **Removal.** Held-out codes are taken out of training: they are neither positives nor negatives. Admissions keep
  their other codes and their text. This removes 34,349 training positives (6.5% of the labels).
- **Evaluation positives.** 40,503 over all admissions, 4,000 of them in the evaluation split.
- **Not eligible.** The head bins (−6, −4), which hold 58 codes and 204k positives, so the label distribution's head
  stays intact; and codes with fewer than 5 occurrences, which have too few positives for a per-code AUC.
- **Recorded properties.**
  - 25 held-out codes have exactly the member-concept set of a trained code, so their composed vectors are identical;
    E1 is also reported without them.
  - 115 held-out codes come from one-to-many maps.
  - 5 held-out codes have no title.
- **Interaction with T1c's SNOMED holdout** (2,069 concepts never linked in E9 training). 137 label codes have a
  T1c-held-out member: 7 held-out, 127 trained and 3 naturally unseen. The T1c holdout is unchanged. For the C5 encoder,
  those concepts' spans are linked and composed zero-shot, as in E9's evaluation. The 7 codes are an exploratory split.
  The coding holdout is independent of T1c's.

## 5. The head and the conditions

**Head** (LAAT / CAML style, the same for every condition):
- Documents: `Z = tanh(W_d · LN(H))`, with H the segment states and attention width 256.
- Every label parameter comes from the code vector u_l: `[q_l; o_l; b_l] = MLP(LN(u_l))` (512 hidden, GELU). This is
  the query, the output vector and the per-label bias.
- Logit: `s_l = Σ_i softmax_i(⟨Z_i, q_l⟩/√256) · ⟨Z_i, o_l⟩ + b_l + b_0`.

There is **no free per-label bias**, so a label's frequency prior can only come from its code vector. A code never
trained is scored by exactly the rule of a trained one. LN removes the vector's norm, so only its direction matters.

**Conditions** (code vectors u_l; 256-dimensional unless noted):

| # | name | code vector | never-trained code gets |
|---|---|---|---|
| (i) | `free` (control) | a free learned vector per code, N(0, 0.02²) as BERT / HRRBERT | its initial vector (no gradient, no weight decay) — HRRBERT's "unstructured" |
| (ii-a) | `composed_head` (**primary arm**) | `FrameComposer` with the C5 channel's settings (HRR binding, attentive bundling, induced concept factor, key 8), atomics and relations fresh and trained only by the coding loss — HRRBase's analogue | the composition of its frame by the same trained atomics and relations |
| (ii-b) | `composed_c5` | the trained T1c C5 composer (SmolLM2-360M, seed 1), **frozen**; code frames added by `add_concepts` (E9's zero-shot insertion) | the same (after T1c seed 1) |
| (iii) | `transe` | TransE (E9's C6g trainer: L1, self-adversarial negatives, 256-d, 20k steps) on the frames of the 16,550 concepts the task needs (label members, and the concepts atomics name, unified with their atomics: 93,849 triples, 17,262 entities; `runs/kge-v1`); code = mean over members; standardized; frozen | its TransE vector |
| (iv) | `title` | the code's long title (`D_ICD_DIAGNOSES`) read by the same frozen host: mean of last-layer states over the title tokens (C5 reads it with its channel); standardized; frozen; the 108 codes without a title get the mean title vector | its title vector |
| (v) | `random` | fixed random unit vectors (per seed) | a random vector |
| (vi) | `gram` | GRAM (Choi et al. 2017): attention over the free embeddings of the code and its ICD-9-CM ancestors (4-character subcategory, 3-character category, chapter; 7,576 nodes); attention MLP 128 | its untrained leaf plus its **trained ancestors** — the standard clinical rare-code baseline |
| (vii) | `composed_free` (secondary) | `composed_head` + a free residual N(0, 0.02²) — HRRBERT's HRRAdd | composed + its initial residual |

**Training.**
- AdamW at lr 1e-3. Weight decay 0.01 applies to the document projection and the label-MLP matrices; none to label
  sources, biases or norms.
- Batches of 32 admissions, length-bucketed. BCE over the 5,033 training labels; gradient clip 1.0; bf16 autocast in
  the attention.
- At most 12 epochs, with early stopping per head on dev BCE (patience 2; the best epoch is restored).
- No tuning per condition.
- **Pairing.** Every condition of a seed is trained in one job on **identical batches**: the batch order is a function
  of the seed only. `composed_c5`, trained later, sees the same order.
- **Seeds** 1, 2 and 3 (head initialization and batch order): 3 per condition and encoder.

## 6. Encoders

- **P0-360M** — the frozen SmolLM2-360M (T1c's P0) in bf16. This is pass 1, available now, and the **primary encoder**.
- **C0p-360M, C5-360M** — the T1c seed-1 E9 runs. Each is a full fine-tune on 50M tokens of the T1c mix, which holds
  training-patient notes and no evaluation-patient text; neither has seen a label.
  - C5 reads with its channel on. The linker is E9's T1c evaluation alias table (`e0ab5fa3…`, ℓ_min 2; spans that cross
    a 1,024-token window are dropped).
  - Coupling: when a held-out code's concept occurs as a span, the C5 encoder injects a row from the composer that also
    gives `composed_c5` its code vectors. That is a legitimate mechanism, but it is read against the optional C5 run with
    its channel off (§13).
- 135M hosts are not planned.

## 7. Metrics

**Admission sets.**
- Seen (trained) codes are scored on the **evaluation admissions** (5,114).
- Held-out codes are scored on **all admissions** (52,722; never trained as labels anywhere). This is the primary set;
  the evaluation admissions alone are secondary.

**Measures.**
- **Per-code AUC** (midrank), across admissions.
- **Top-k accuracy** (k = 10, 100), as HRRBERT: for each positive (admission, code), whether the code ranks in the
  admission's top k among all 5,589 labels.
  - Bin values are **code-balanced** (the mean over codes of each code's hit rate) and **pooled** (recall@k over the
    bin's positives).
  - Held-out codes: generalized top-k (ranked among all labels) and zero-shot recall@k (ranked among the 556
    never-trained labels).
- **Macro-AUC per bin**; overall seen macro-AUC, micro-F1 (logit > 0), P@8 and P@15 (descriptive).
- **Frequency-bias slope:** the OLS slope of per-code AUC on ln f over the 2,536 trained codes with at least one
  evaluation positive. Codes are equally weighted.
- **Frequency gap:** the macro-AUC of frequent trained codes (ln f ≥ −8, at least 177 occurrences) minus that of rare
  trained codes (ln f < −10, at most 23).
- **Trained-vs-held-out gap at matched natural frequency**, per bin [−12,−10), [−10,−8) and [−8,−6), on evaluation
  admissions.
- **Frequency decodability of the code vectors** (seen codes; per seed, then averaged):
  - (a) cross-validated ridge R² of ln f from u_l (5 outer folds; penalty chosen by an inner 4-fold CV);
  - (b) the same from the label parameters [q; o; b];
  - (c) Spearman ρ between b_l and ln f;
  - (d) **neighbour agreement**: Spearman ρ between ln f and the mean ln f of the 10 cosine nearest neighbours;
  - (e) HRRBERT's **t-SNE** (exact, perplexity 30, 750 iterations, 3,000 random labels, seed 1), coloured by ln f with
    held-out codes marked, and (d) computed in t-SNE coordinates.

## 8. Statistics

- **Two-way bootstrap, B = 2,000.** One draw per replicate is shared by every condition, so comparisons are paired:
  - admissions get Poisson(1) weights (one vector over all admissions; the evaluation set uses its restriction);
  - codes are resampled with replacement, seen and held-out sets separately;
  - seeds are resampled with replacement within each condition.
- Per-code AUC under the weights is exact (`WeightedAuc`). CIs are percentile intervals; p-values are two-sided
  bootstrap p-values.
- **Primary family:** Holm over E1 and E2 (α = 0.05).
- **Many-to-one (the role of Dunnett's test):** each other condition against `free` on every endpoint, with Holm across
  conditions within the endpoint. HRRBERT's per-bin Dunnett over seeds (n = 3) is also reported, for comparability only.
- **Specificity:** `composed_head` minus each of `random`, `title`, `transe`, `gram`, `composed_c5` and
  `composed_free`, on E1 and E2, with Holm over the competitors.

## 9. Primary endpoints and decision rules

Primary encoder: **P0-360M**. Primary contrast: **`composed_head` − `free`**.

- **E1 — never-trained codes (the ROOD analogue).** Held-out macro-AUC over all admissions.
  - *Supported:* Δ > 0 with Holm p < 0.05.
  - *Refuted:* the upper 95% bound of Δ is below 0.02. The composed vectors then carry no usable information about
    unseen codes.
  - `free` is near 0.5 by construction (its never-trained rows are random). E1 tests whether composition reaches
    better than chance. The specificity contrasts (§8) test whether that is due to SNOMED composition.
- **E2 — frequency bias.** Δslope, the per-code AUC slope on ln f.
  - *"Reduces frequency bias":* Δslope < 0 with Holm p < 0.05, **and** the rare-code macro-AUC difference is ≥ 0 with
    a lower 95% bound above −0.01.
  - *"Flattening by damage":* Δslope < 0, but the rare codes do not gain and the frequent codes lose more than 0.01.
    The slope then flattens because frequent codes get worse, not because rare codes get better. HRRBERT saw HRRBase
    weaker on frequent codes, so this is a live outcome.
  - *Refuted:* Δslope is not negative, or not significant.

**Readings** (pre-stated; P0 first, then each encoder):
- `composed_head` ≈ `gram` on E1 (95% CI of the difference within ±0.01), or `gram` better: an ICD hierarchy is
  enough, and the SNOMED-specific claim is not supported.
- `composed_head` ≤ `title`: the code's name read by the host is enough; the frames add nothing over the title.
- `composed_head` ≈ `transe`: the structure carries the gain, not the composition operator.
- `composed_head` ≈ `random` on E1: no information, which would point to a bug.
- E1's gain only on the 25 codes that share a concept set with a trained code: alias copying, not composition. E1 on
  the other 357 codes is reported.
- `free` ridge R² and neighbour agreement ≫ `composed_head`'s: HRRBERT's t-SNE claim reproduced as numbers. If
  `composed_free` (HRRAdd) moves back towards `free`, its free part re-learns frequency, as HRRBERT suggested for
  HRRAdd / HRRCat. If `composed_head` ≈ `free`, frequency leaks through atomics trained mostly by frequent codes.
- `composed_c5` vs `composed_head` (after T1c seed 1): does structure trained in the language model transfer to coding
  frozen?
  - `composed_c5` ≥ `composed_head` on E1: the LM-trained dictionary is a ready code embedding.
  - `composed_c5` < `composed_head`: coding needs atomics trained by coding.
- **Encoders** (pass 2): E1 and E2 are replicated on C0′ and C5 with the same rules, Holm within each encoder. Whether
  C5 helps composed codes more than C0′ does (the interaction) is descriptive.

## 10. Power (planning numbers)

- **E1.** About 380 held-out codes with positives over all admissions. A per-code AUC spread of about 0.1 gives a
  macro-AUC SE of about 0.005; across 2–3 seeds the paired differences are smaller. Power against `free` (≈ 0.5) is
  ≈ 1 for any Δ ≥ 0.02. The specificity contrasts resolve |Δ| ≈ 0.01–0.015.
- **E2.** 2,536 codes spread over 11 ln-units of frequency. The slope SE is read off the bootstrap in the smoke-free run.
  There is no power claim here: E2 is decided by the rule above.

## 11. What is not claimed

- No state-of-the-art MIMIC-III coding: the encoder is a frozen causal host, and procedure codes and unframed codes are
  out of the task.
- No ICD-10: MIMIC-IV has ICD-10 only as structured data, and no MIMIC-IV notes are on this machine. HRRBERT's own
  structured-code setting is the place for ICD-10.
- No claim about the 174 naturally unseen codes beyond descriptive numbers.

## 12. Runs and compute (placement in §14)

- **CPU** (the main checkout, before the GPU jobs): `prepare` (16 s), `tokenize` (95 s, 8 workers) and `kge` (31 min, 20k TransE steps).
  All three have already run in this worktree, and their outputs sit in the shared data root.
- **Pass 1, now** (P0-360M): `encode`, 3 × `train` (7 conditions per job) and `analyze`, at priority 55.
- **Pass 2, after T1c seed 1 trains (51–54)** at priority 56–57:
  - `encode` C0p-360M and C5-360M;
  - 3 × `train` per encoder (8 conditions, with `composed_c5`);
  - 3 × `train composed_c5` on P0 (the same batches as pass 1);
  - `analyze` for all three encoders.
- **Optional:** C5 with its channel off (§6).
- GPU-h come from the smoke test (§14).

## 13. Licence and DUA

- MIMIC-III is PhysioNet credentialed data, SNOMED CT is licensed, and ICD code titles belong to the credentialed
  context.
- Everything derived from them lives under `~/data/vsa-llm/t1c/icd-frequency-v1/` (mode 700):
  - code and holdout lists, frames, titles;
  - tokens and spans, host states;
  - scores and ranks, code vectors;
  - t-SNE coordinates and the t-SNE figure.
- The committed run folders hold aggregates only: counts, digests, and metrics per condition and bin. `.gitignore`
  admits only `summary.json`, `metrics.json`, `endpoints.json`, `report.md`, `manifest.json` and `resolved_config.yaml`
  there.
- No stage prints a note, a row, a code, a title or a concept. Nothing is sent to any external service.
- The synthetic end-to-end test checks that the run folders receive no code string and no per-item array.
- The t-SNE figure contains no labels, but stays out of git until the author decides (open decision).

## 14. Smoke tests and commands

### §14 addendum — smoke tests (2026-10-07, after the commit of §§1–13; SMOKE, not results)

The smokes check that every stage runs on the real data and measure cost. Their numbers come from a few hundred
admissions and a dozen gradient steps, so they say nothing about the endpoints. Aggregates are in
`runs/*-cpusmoke*` and `runs/*-gpusmoke*`.

- **Unit tests.** `tests/test_icd_frequency.py` has 21 tests on synthetic fixtures. It includes an end-to-end
  synthetic run checking that the run folders receive no code string and no per-item array.
- **Pass-2 code path.** This used an existing E9 C5 run (T5 glossary, 135M, read-only), on CPU, with synthetic frames
  and spans. Results:
  - the trained composer loads from `final.pt`;
  - frozen insertion (`add_concepts`) leaves existing rows unchanged;
  - with the channel on, host states change from the injection position onward and not before.
  
  It also found that `load_run` needs the explicit alias table: T1c's `ontology.pt` has no sidecar table and would
  fall back to WordNet. Fixed before the preregistration commit.
- **CPU smoke** (SmolLM2-135M on CPU, the first 48 admissions):
  - encode: 106 s at 1.6k tokens/s;
  - training, 7 heads × 1 epoch: 6.6 s;
  - analysis: 820 s, mostly the exact t-SNE (57 s per condition on CPU) and the NumPy ridge CV. Both now run on the
    analysis device.
- **GPU smoke** (P0 = SmolLM2-360M bf16, the first 200 admissions = 757,837 tokens). It ran next to a queue training
  job that used 20.8 GB of memory and 100% of the GPU.

  | stage | wall time | detail | peak GPU memory |
  |---|---:|---|---:|
  | encode | 51 s | admissions at 21.9k tokens/s, then 5,589 titles | 1.17 GB |
  | train (7 heads × 2 epochs, 169 training admissions, batch 32) | 4.6 s | ≈ 0.33 s per batch for all 7 heads, dev included | 1.47 GB |
  | analyze (1 seed, 200 replicates) | 84 s | — | — |

  Total ≈ 2.7 min, within the ≤ 4 GB / ≤ 5 min smoke limit.

  Smoke aggregates (not results):
  - seen macro-AUC 0.49–0.59; held-out macro-AUC 0.47–0.57 over the 94 held-out codes with positives among the 200
    admissions;
  - ridge R² of ln f from the code vectors (init → trained):
    - `free` −0.002 → 0.21 after 12 steps;
    - `composed_head` 0.085 → 0.16;
    - `gram` 0.088 → 0.34;
    - `title` 0.22 and `transe` 0.125 (fixed vectors);
    - `random` ≈ 0.
- **Cost of the full runs** (idle-GPU estimates; the smoke's shared GPU is taken to be ≈ 1.6× slower — an
  assumption):
  - encode, 191.7M tokens: 2.4 h at the smoke's rate, ≈ 1.5 GPU-h idle for P0, ≈ 1.7 GPU-h for C0′ / C5 (fp32
    weights under autocast);
  - training, 1,404 batches per epoch: ≈ 7.7 min per epoch at the smoke's rate, ≤ 12 epochs, 6–8 expected. That is
    ≈ 0.8 GPU-h per 7-condition seed (scoring ≈ 5 min included) and ≈ 0.9 per 8-condition seed;
  - `composed_c5` alone on P0: ≈ 0.15 GPU-h per seed; analysis ≈ 0.15 GPU-h.
  - **Pass 1 ≈ 4.1 GPU-h; pass 2 ≈ 9.8 GPU-h.**
  - Disk: states ≈ 12 GB per encoder; heads ≈ 1.7 GB per train job.
- **Queue commands:** `queue-commands.sh` in this folder, printed by the `plan` stage and never executed here.
  - Pass 1 (P0) is at priority 55, right after T1c seed 1 (51–54). It has no dependency, so a lower number would run it
    first.
  - Pass 2 (C0′ and C5 encoders, 8 conditions; the C5 dictionary on P0) is at 56, its analyses at 57.
  - Within a priority, the queue runs jobs in creation order: encode → train → analyze.

## 15. Deviations and changes after commit

- **15.1 (2026-10-07, before any full run) — an added descriptive probe.** Each condition's code vectors are also
  saved **at initialization**, before any gradient step (`vectors.pt: source_init`). The ridge R² and neighbour
  agreement are reported for them too, which separates frequency implied by the content (a frame, a title) from
  frequency learned in training — HRRBERT's claim. No endpoint, rule or condition changes. The CPU-smoke analysis
  predates it and lacks these columns.
- **15.2 — implementation, same estimators.** The ridge CV runs in torch on the analysis device: one
  eigendecomposition of the standardized Gram matrix per split, covering every penalty, with the same inner 4-fold
  choice. Batches are copied as float16 and converted on the device.
- **15.3 — output naming.** `analyze --label` gives the pass-2 analyses their own folders (`analysis-<encoder>-pass2`),
  so pass 1's are not overwritten.
