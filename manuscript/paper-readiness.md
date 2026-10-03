# Paper-readiness audit

**Date:** 2026-10-03 (WP-PQ2, after the [novelty check](novelty-check-2026-10.md)).

**Scope:** every experiment that may enter the paper, checked against the paper's bar:
- **seeds:** ≥ 3 for any claim;
- **statistics:** CIs, Holm, a pre-registered criterion;
- **controls:** the novelty check's lists A-B1…A-B8, B-B1…B-B7, C-B1…C-B7, D-B1…D-B9 and §6.5;
- **reproducibility:** a clean-commit manifest, committed configs and run folders, data cards, holdout hashes, leakage and contamination audits;
- **compute:** spent so far and still needed.

**Sources:** [claims.md](claims.md), [gates.md](../resources/plan-improvement/gates.md), [execution.md](../resources/plan-improvement/execution.md) (on `main` `dc6585d`, which adds the PQ1/PQ2 and Qwen block notes), [queue.md](../resources/plan-improvement/queue.md), [reproducibility.md](reproducibility.md), the reports R0, R1, R9, R10, and the run folders.

**Machine-checked parts.** Manifests, commit status, the compute ledger, holdout hashes and contamination fields come from `scripts/paper_readiness.py`. It is read-only and never writes into a run folder or `.jobs/`. Regenerate with:

```bash
PYTHONPATH=src python scripts/paper_readiness.py --main /home/bhux/workplace/VSA-LLM --output manuscript/tables/paper_readiness.json
```

It scans two trees:
- `--repo`, this checkout. "Committed" is answered by `git ls-tree` on `HEAD` (`75c8a04`, this branch) and `main` (`dc6585d`).
- `--main`, the main checkout on disk, which also holds runs in progress and runs not yet committed.

The numbers below are from the run of 2026-10-03, with the queue last finishing at 15:55 UTC. The rows for the two items this work package built while the audit ran (E9 dimension-3 baselines, code `3661b81`; E10 baselines, code `0a38784`, runs `e10.0-baselines-v2` / `e10.1-baselines-v2`) were filled in after their code landed.

**Verdict scale:**
- **ready**: can be written up as is, with the wording stated;
- **ready (labelled)**: only as exploratory, synthetic or negative context;
- **needs X**: specific runs or fixes are missing;
- **not started**.

---

## 0. Verdicts at a glance

| Experiment | Supports | Seeds done / needed | Verdict |
|---|---|---|---|
| E0 synthetic identifiability | E0.1–E0.6 (M1, M2, M3 on planted structure) | 3 / 3 | **ready (synthetic)** |
| E2 mapping × operator frontier | F.2, F.3, E2.1–E2.4; OP.1 (frozen) | 3 / 3 | **ready** (negative/relative result); clean up the stale `runs/v1` |
| E1 contextual composition | H-C.1, H-C.2 (refuted on frozen anchors) | 3 / 3 | **needs:** commit the run folder; re-run from a clean tree or disclose the dirty manifest; claims ledger not updated |
| E3 developmental WordNet | H-D.1 | 0 / 3 | **not started** (queued, ≈ 2 GPU-h) |
| S0 sanity pilot (R1) | context for H-A only (never a gate) | 2 (scratch), 1 (CPT) / — | **ready (labelled exploratory)**; under-trained recipe (D4.0) |
| D4.0 recipe sweep | methods detail (frozen recipe) | 1 per cell (9 of 11 cells) | **needs:** the 2 pending 32k cells, commit the run folders, freeze the recipe |
| E4 core (D4.7/8/9, D4.1) | H-A.1–A.9, OP.1, VSA.1, N.M4, N.H-A (**the pre-registered core**) | 0 / 3 | **not started**; G2 pre-registration file missing (≈ 143 GPU-h) |
| E9 T5 SmolLM2-360M/135M | A, B, C (and H-B.1, H-E.1 in part) | 1 / 3 | **needs:** seeds 2–3 (queued); PQ1 controls (A, B); PQ2 dimension-3 baselines (C) |
| E9 T4 / T1-open / WordNet (SmolLM2) | A (natural text), A locality (negative control) | 0 / 3 | **not started** (seed 1 queued) |
| E9 Qwen3-1.7B/0.6B (T5) | A, B, C on a modern host (reported separately) | 0 complete (s1 running) / 3 | **needs:** s1 finishing; s2 queued; s3 not queued |
| E9 Qwen3-4B, Qwen3.5-2B/0.8B (T5) | host-family generality (descriptive) | 0 / 1 planned | **not started** (queued / probe only) |
| E9 dimension-3 baselines (PQ2) | C (C-B1–C-B6 in part) | 0 / 3 (implemented, not run) | **needs:** the evaluation-only jobs (`e9_plan --track t5 --dim3-baselines --queue`: ≈ 4.3 GPU-h for T5 SmolLM2 s1, ≈ 10 with s2–3); code `3661b81` with tests; the known-fact toy check and a CPU smoke on the real T5 C5-135M run pass |
| E10.0 synthetic | D (protocol, synthetic) | 3 / 3 | **ready only as a negative / methods result** (outline framing rule; claims ledger E10.D1); **needs** a null-calibrated acceptance test before any discovery claim. The PQ2 baselines are done and narrow D further: recovery is largely decodable without learning (a no-learning prior readout reaches AUC 0.91–0.92 vs 0.98–0.99; graph-only AMIE/TransE/RotatE/ComplEx/IterE 0.54–0.77); AMIE on the captured pairs makes the same correct property identifications as the riddle step; null worlds with no hidden relation accept 2.7–4.3 false slots per run; a fresh validation split changes nothing, Holm over proposals cuts null-world edge acceptance from 40–45% to 3–4%. Stays synthetic-scoped and out of the abstract |
| E10.1 WordNet (frozen GPT-2) | D (negative transfer) | 3 / 3 | **ready (labelled negative)** |
| E10.2 joint LM | D (decisive test) | 0 / 3 | **not started** (≈ 20–30 GPU-h, after the E4/E9 recipe) |
| E7 self-authoring | H-G.1–G.3 | 0 / 3 | **not started** (D7.0 corpora built; authoring jobs queued) |
| E5 explainability / zero-shot | H-F.1, H-F.2, H-C.2, H-E.1 | 0 / 3 | **not started** (pre-registration present; needs D4.1 checkpoints) |
| E8 application tracks | E8.1, T.1 | corpora done; results 0 | **ready** for T.1 (cardinality); results not started |
| B6, B8, B10, R0 | R0.1, R0.2, B.1 | — | **ready**; dirty manifests disclosed in R0; B10 has no git record |
| C5 judge calibration | J.1 | 2 harnesses | **ready** (v2 manifest clean; v1 has no manifest) |

---

## 1. Per-experiment readiness

Abbreviations:
- **t-CI:** 95% Student-t interval over seeds.
- **boot:** cluster bootstrap over evaluation windows or items.
- **prereg:** a criterion fixed in writing before the run. Gates in experiments.md / execution.md count; formal files exist only for E5. E4's is missing.
- **GPU-h:** queue hours from `.jobs` (last attempt per job).

### 1.1 Mechanism experiments (E0–E3)

| Exp. | Claim | Seeds | Statistics | Controls present | Controls missing | Reproducibility | GPU-h used | Verdict |
|---|---|---|---|---|---|---|---|---|
| **E0** | E0.1–E0.6 | 3 (101, 202, 303) | t-CI, paired by seed; G1 thresholds pre-written in experiments.md; no Holm (one primary endpoint per sub-experiment) | static teacher; free factors; random splits; Anderson–Darling and coherence-only screens; parent-routing variants | none required for the synthetic claim; the "threshold differentiation" split baseline of the re-check (decision 20) is not run | `e0-development` @ `2cc5d0b` clean; `d02-parent-sync` @ `9986d72` clean (M3 default); `d02-parent-routing` @ `bf5f535` clean; **`d02-parent-routing-v2` @ `01e8090` dirty** (a superseded variant, disclosed in reproducibility.md §5); bit-exact on the new machine (R0) | 0 (CPU) | **ready (synthetic)** |
| **E2** | F.2, F.3, E2.1–E2.4 | 3 (11, 22, 33) | t-CI; gate pre-written; no Holm (paired contrasts reported individually) | derangement-shuffled labels; `random_fixed`, `untyped`; capacity-fair operators | none required for the frozen-anchor result (the translation operator of decision 20 belongs to E4 C8) | `runs/v2` @ `07a7315` clean; **`runs/v1` (old machine, untrusted) is committed with `metrics.jsonl` only and no manifest**; reproducibility.md says gates.md cites the wrong commit (`57e4e23`) for v2 | 0.90 | **ready** after marking `runs/v1` stale |
| **E1** | H-C.1 (variance; refuted on frozen anchors), H-C.2 (sense alignment; refuted) | 3 (11, 22, 33) | t-CI over seeds (paired); gate pre-written (experiments.md E1): **FAIL** | static M0, oracle-sense M0, free-sense attention, MFS, operator ablation (`diagonal`, `random_fixed`, `untyped`) | none for a negative result | **`runs/v1` @ `783358b` dirty, and the run folder is not committed** (neither `HEAD` nor `main`); claims.md still lists H-C.1 as "partial" | 0.31 | **needs:** commit, then re-run clean (0.3 GPU-h) or disclose; update the claims ledger |
| **E3** | H-D.1 | 0 | gate pre-written (ARI vs random and uniform enlargement, false-split ≤ 5%) | planned: none, uniform, random, coherence, M3, NP-MSSG/AdaGram, stem-cell pool | threshold-differentiation rule (re-check) | `runs/hrr-v1` (old machine, untrusted) committed with `metrics.jsonl` only and no manifest; `d3-e3-v2` pending at priority 30 | 0 | **not started** (≈ 2 GPU-h, execution.md 3-day block) |

### 1.2 From-scratch core (E4) and its precursors

| Exp. | Claim | Seeds | Statistics | Controls present | Controls missing | Reproducibility | GPU-h used | Verdict |
|---|---|---|---|---|---|---|---|---|
| **S0 pilot (R1)** | none; context for H-A, H-E | 2 (50M scratch), 1 (SmolLM2-135M LoRA CPT) | boot (10k) with Holm per stratum; explicitly not a gate | C0, C1, C2, C3, C5; probes by link status | C1 not parameter-matched (R1); C1h, C1s, C3t, C8 absent; recipe under-trained (D4.0: 1.3 nats worse than the best setting) | 13 runs clean (several SHAs `cfbb50e`…`80e1362`, all after the `9606bc8` speed-up); `analysis/pilot-v2` clean; committed | 5.25 (training 5.03, probes 0.22) | **ready (labelled exploratory)**: never pooled; effect sizes to be re-measured under the frozen recipe |
| **D4.0 recipe** | methods (recipe choice) | 1 per cell | descriptive (final loss) | 3 × 3 grid tokens/step × LR | 32k tokens/step cells (2 pending, priority 30) | 9 run folders clean, but **not committed** on `HEAD` or `main`; queue.md still lists D4.0 as "todo" | 3.66 | **needs:** 2 runs (≈ 0.8 GPU-h), commit, freeze the recipe in G2 |
| **E4 core** (D4.8, D4.7, D4.9, D4.1; then D4.3/D4.4/D4.5) | H-A.1–A.9, H-B.1–B.2, OP.1, VSA.1, N.M4, N.H-A | 0 / 3 | planned: boot + Holm, one-sided locality, `k(L)` with CIs (e4_report) | planned: C0, C1 (matched at G2), C1s, C1h, C2/C2f, C3t, C3–C7, C8 operators | **`experiments/e4-small-lm/preregistration.md` not found (G2 not taken)**; C1h-E, C1m, alias stratum and translation operator await G2 (decision 20); §6.5: injection-site ablation, linker-noise curve | configs for pilot/recipe/cpt-pilot/e9-check committed; no core configs yet | 0 | **not started**: blocks the short "core result" paper (outline §b) |

### 1.3 E9 retrofit × quantization × ontology editing (claims A, B, C)

The current evidence is R9: T5, SmolLM2-360M and -135M, seed 1. The controls named below come from the novelty check §2.5 (A-B), §3.4 (B-B) and §4.4 (C-B).

**Status of the controls across all E9 rows:**
- **Present:** P0, C0′, C2 (free table), C5. Inside C5, for new words: none / mean-row / random equal-degree frame. For edits: a control edit to a random filler and neighbourhood specificity. Quantization: INT8/INT4 RTN, variants A and B. General-text locality; probes.
- **In progress by WP-PQ1**, priority 51 per execution.md on `main`, not yet in `.jobs`:
  - operator and specificity ablation of C5 (`random_fixed`, `untyped`, translation, shuffled frames): A-B6;
  - same-site row sources (subtoken-mean/FVT, definition encoder, KG embedding): A-B1, A-B2, A-B4 in part;
  - filler vs non-filler loss split: A-B5;
  - quantization controls: channel off at INT4 (B-B2), GPTQ/AWQ/HQQ/NF4 (B-B3), quantized embeddings (B-B7), DiD reporting (B-B6), a bit-matched side module (B-B1);
  - ≥ 3 seeds on T5/T4 for SmolLM2-360M.
- **In progress by WP-PQ2 (this WP):** dimension-3 baselines C-B1, C-B2, C-B3 (rows), C-B4 (transplant), C-B5 (channel-off), C-B6 (intra-entity locality). Implemented in `3661b81` (`vsa_embed.knowledge_editing`, `experiments/e9_dim3_baselines.py`, `e9_plan --dim3-baselines`, `e9_report --dim3-baselines`); not yet run on any stage (row below).
- **Unassigned:**
  - A-B3: in-context verbalized frame at first mention for the *LM-loss* strata (dimension 1);
  - A-B7: C1h-E, an Engram-style hashed memory in the retrofit setting;
  - B-B4: revert diagnostics (delta norms vs the INT4 step; CPU);
  - B-B5: a QAT/QLoRA arm;
  - C-B3 (d)/(e): a CoLLEGe-style generator and the gradient upper bound;
  - C-B4: relation-shuffled and filler-only frames;
  - C-B5: dose–response over 1–3 edges;
  - C-B6: portability (2-hop) and alias subjects;
  - C-B7: ALCUNA/COMPS-style items, about 700 items per arm (now 300 new words and 200 edits), and an item × seed mixed-effects model;
  - §6.5: KnowLA-style KGE baseline, KGE-initialized table, parameter-matched LoRA plus an OOD-retention suite.

| Track · host | Claim | Seeds done / queued | Statistics | Reproducibility | GPU-h used | Verdict |
|---|---|---|---|---|---|---|
| **T5 · SmolLM2-360M/135M** | A, B, C | s1 done; s2–3 queued (87 jobs, priorities 31–34) | boot (10k) over windows / items, Holm within stratum; single-seed flag in the report; **no pre-registered criterion** (the questions and readouts in execution.md E9 were written before the runs, with no decision rule) | training manifests clean (`bbe4044`, `eeb1833`, `db75346`; no trainer diff between them); per-run evaluations clean except **4 dirty P0-135M evaluations** @ `1b020b4` (`edit`, `edit-int4`, `zeroshot`, `zeroshot-int4`; no diff in the evaluation modules to the clean runs' commit); **`report/t5` @ `783358b` dirty**; `quant/t5` and `quant-general/t5` clean; T5 holdout `e7313dce…` recomputed and consistent; new-word items `contamination_free` (78.8M characters checked, 0 reserved clashes); T5 `leaked_into_train = 0` | 6.67 GPU + 0.36 CPU-in-queue (report); engagement check 1.42 | **needs:** seeds 2–3 (≈ 13 GPU-h, queued); PQ1 controls for A/B; PQ2 baselines for C (evaluation-only, ≈ 4.3 GPU-h for s1); regenerate the report from a clean tree |
| **T4 · SmolLM2** (natural chemistry text) | A ("domain terms" beyond synthetic) | s1 queued (priorities 35–38) | as T5 | T4 corpus clean (`16176c2`), holdout `b58e504f…` consistent; Qwen3 relink pins it | 0 | **not started** (≈ 6.5 GPU-h for s1; s2–3 ≈ 13 more, not queued) |
| **T1-open · SmolLM2** (PubMed/MeSH) | A (natural text) | s1 queued (39–42) | as T5; T1 rare stratum exploratory (decision 13) | T1 corpus clean (`cfbb50e`), holdout `1c477afd…` consistent; `slice-v1` stale | 0 | **not started** (≈ 7 GPU-h s1) |
| **WordNet · SmolLM2** (negative control) | A locality ("no effect on known vocabulary") | s1 queued (43–46) | as T5 | C3 host corpus clean (`57e4e23`), holdout `7f2462ed…` | 0 | **not started** (≈ 6.5 GPU-h s1) |
| **T5 · Qwen3-1.7B/0.6B** (LoRA 64) | A, B, C on a modern host (not pooled with SmolLM2) | s1: P0, C0′ done, C2 running, C5 and all of 0.6B pending; s2 queued (47–50); **s3 not queued** | as T5 | memory probe `memory/qwen3-v1` @ `783358b` **dirty**; finished run folders **not committed yet** (`Qwen3-1.7B-Base-frozen-P0-s1`, `-lora-C0p-s1`; C2 in progress without a manifest) | 3.51 done + 0.5 running; probe 0.04 | **needs:** s1 to finish (≈ 18 GPU-h block), s2 (≈ 18, queued), s3 (≈ 18, not queued) |
| **T5 · Qwen3-4B** | generality (single seed, descriptive) | queued (31 jobs, priorities 56–59) | as T5 | configs committed (`75c8a04`) | 0 | **not started** (≈ 34 GPU-h training + evaluations, execution.md) |
| **T5 · Qwen3.5-2B/0.8B** (separate env) | generality; separate host family | memory probe queued (priority 52); block queued after it | as T5 | env lock file and kernel smokes committed (`experiments/e9-retrofit/env`); items byte-identical to SmolLM2/Qwen3 | 0 | **not started** (≈ 20–25 GPU-h + probe ≈ 0.5) |
| **Dimension-3 baselines (PQ2)**, every E9 stage | C: C-B1 (ROME / MEMIT / AlphaEdit on P0 and C0′), C-B2 (in-context frame, IKE), C-B3 a–c (FVT subtoken-mean and definition-encoder rows at the channel site), C-B4 (frame transplant), C-B5 (channel-off audit), C-B6 (intra-entity locality) | none run; per stage after its runs (`--seeds` / `--models` filters) | same items and metric code as `e9_ontology_edit`; item bootstrap with Holm; stage level: the C5 ontology edit vs every model × method, paired over items and seed-averaged | code `3661b81` with tests (ROME's rank-one property, AlphaEdit's null-space property, exact weight restore, channel-off revert = 0); known-fact toy check on SmolLM2-135M (`experiments/e9-retrofit/dim3-toy/smollm2-135m-v1`); CPU smoke on T5 C5-135M (6 edits, 8 new words; a pipeline check, not a result) | < 0.05 (timing micro-benchmark) | **needs:** the jobs: T5 SmolLM2 s1 ≈ 4.3 GPU-h, s2–3 ≈ 5.7 more; Qwen3-1.7B/0.6B s1 ≈ 10.2; ≥ 3 seeds before claim C's comparison is written |

**Claim-level reading for E9.**
- **A** can be written with the novelty check's §2.6 wording once seeds 2–3 land. "Beats new-token initialization / approaches in-context frames" needs PQ1 (A-B1, A-B2, A-B4) and the unassigned A-B3. "Domain terms" needs T4/T1.
- **B** stays out of the abstract until B-B1–B-B3 pass with ≥ 3 seeds (PQ1).
- **C** needs C-B1/C-B2 (PQ2) and ≥ 3 seeds. The items are under-powered for Δ ≈ 0.06 (C-B7).

### 1.4 E10 self-learned semantics (claim D)

| Exp. | Claim | Seeds | Statistics | Controls present | Controls missing | Reproducibility | Compute | Verdict |
|---|---|---|---|---|---|---|---|---|
| **E10.0 synthetic** | D (H-H clauses i–iv, synthetic) | 3 (101, 202, 303); dev seeds 7–9 for hyper-parameters | t-CI; per-seed permutation tests with Holm; refutation clauses pre-written in execution.md (H-H); the permutation reading of clause (ii) was added after the first run (labelled in R10) | M3 splitting, stem-cell pool, oracle labels, random candidates, filler-frequency ranking, random acceptance at the matched rate, in-sample acceptance, NN/random frames, oracle dictionary | D-B1 AMIE/IterE, D-B2 TransE/RotatE/ComplEx (+ a no-learning prior readout), D-B4 null worlds, D-B5 fresh split / Holm: **done** (WP-PQ2, `runs/e10.0-baselines-v2`, seeds 101/202/303, dev seeds 7–9 for thresholds); **unassigned:** D-B3 (k-means/GMM, TransG/CTransR, IRM discovery), D-B6 (pruning / random / MDL acceptance for dreaming), D-B7 (resonator, Lasso frame inference), D-B8 (L2-SP/EWC λ sweep, Faruqui retrofitting), D-B9 (WN18RR / FB15k-237 hidden relations; > 3 seeds; initial-mass sweep) | `runs/e10.0-v3` @ `d51bf54` clean; single-threaded workers (bit-replayable) | 0.72 CPU-h | **needs:** a null-calibrated acceptance test (refit-based or MDL) with the null world as its pre-registered check; D-B3, D-B6–D-B9. Then at most one synthetic-scoped sentence, narrower than novelty §5.6: "recovers erased edges present in its training signal; identifies symmetric and inverse relations from captured pairs as well as rule mining" (R10, E10 baselines) |
| **E10.1 WordNet** | D (negative transfer on frozen anchors) | 3 | as E10.0 | as E10.0, plus graph-only D-B1/D-B2 (PQ2, `runs/e10.1-baselines-v2`: ComplEx recovers erased edges at AUC 0.628 vs the learnable ontology's 0.589, CIs include 0) | as E10.0 (D-B4/D-B5 not repeated: the self-tests are at chance here) | `runs/e10.1-v1` @ `eeb1833` clean; anchors job 0.01 GPU-h | ≈ 8.0 CPU-h (4 workers × 2.0 h wall) | **ready (labelled negative)** |
| **E10.2 joint LM** | D (decisive) | 0 | refutation clauses as E10.0 on the LM's strata | planned: fixed ontology (C5), C2 | — | — | — | **not started** (≈ 20–30 GPU-h, after the recipe) |
| **E10.3 self-reflective host** | D (natural-language naming) | 0 | — | planned: teacher naming, random acceptance, compute-matched CPT | — | — | — | **not started** (≈ 10 GPU-h + LLM calls, after E7) |

### 1.5 Other result experiments and supporting items

| Exp. | Claim | Seeds | Statistics | Reproducibility | GPU-h used | Verdict |
|---|---|---|---|---|---|---|
| **E7** self-authoring | H-G.1–G.3 | 0 | planned (E7.2 gate: 3 seeds, CI > 0, locality ≤ 0.5%) | D7.0 corpora `runs/general-v1`, `devtools-v1` @ `903b9cd` clean; 4 smoke folders without manifests (`smoke/*`, labelled smoke); 9 authoring jobs pending at priority 60 | 0 | **not started** (D7.1 ≈ 1 GPU-h; D7 ≈ 40 GPU-h) |
| **E5** explainability, zero-shot | H-F.1, H-F.2, H-C.2, H-E.1 | 0 | `experiments/e5-explainability/preregistration.md` **present** | items committed; `c3-synthetic-*` and `c6-devtools-v1` contamination-free (113–116M characters checked); `c3-heldout-gpt2-v1` is not synthetic (contamination_free = false, as intended) | 0 | **not started** (≈ 30 GPU-h, needs D4.1) |
| **E8** tracks T1–T6 | T.1 (done), E8.1 | corpora: 1 build each | feasibility thresholds harmonized at G2 (decision 17, pending) | every track corpus manifest clean; holdouts recomputed and consistent for C3, T1, T2, T3, T4, T5, T6; T2 and T5 `leaked_into_train = 0`; C6 leakage audit 0 | 0 | **ready** for T.1; results **not started** (D8 ≈ 45 + D4.5 ≈ 25 GPU-h) |
| **B6 / B8 / WP-probe / R0** | R0.1, R0.2 | — | cell-by-cell comparison (`compare_runs.py`) | **B6 `2ad45e5`, B8 `e2078a4`, B8-PTQ `8e83d64` dirty**, disclosed as R0.2 with re-runs at HEAD; WP-probe `34a7909` clean | not in the queue | **ready** (disclosed) |
| **B10** generation benchmark | B.1 | — | descriptive | `experiments/b10-benchmarks/generation.{json,md}`: **no manifest and no git state in the JSON** (reproducibility.md attributes it to `05073c3`) | 0 (CPU; GPU timings qualitative) | **ready** (add provenance if cited) |
| **C5** judge calibration | J.1 | 2 harnesses × 30 items × 3 calls | accuracy, Fleiss κ vs a pre-set bar | v2 @ `de4e512` clean; **v1 has no manifest** | — | **ready** |
| **01a–01c frozen-host (Appendix A)** | F.1 (refuted) | 3 | protocol-2 t-CIs | **`01b…/stage-b-reduced-v2` @ `f9d49d8` and `01c…/{basis-confirmation,source-neighborhood}-development-v2` @ `5762dca` dirty**; 12 pre-provenance manifests (schema 1, no git state) in 00/01/01b/01c | 0 (CPU) | **ready (labelled appendix)**, with the dirty state disclosed (reproducibility.md §5 does) |

---

## 2. Reproducibility audit (machine-checked)

### 2.1 Manifests

Main checkout on disk; folders = run artefacts found by the script.

| Experiment | folders | clean | dirty | no git state | manifest missing | data cards | not committed on `main` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00-capacity-and-algebra | 3 | 0 | 0 | 2 | 1 | 0 | 0 |
| 01-ontology-factorization | 2 | 0 | 0 | 2 | 0 | 0 | 0 |
| 01b-global-local-relational-factorization | 5 | 0 | 1 | 4 | 0 | 0 | 0 |
| 01c-developmental-relation-discovery | 9 | 1 | 2 | 6 | 0 | 0 | 0 |
| b6-host-memory | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| b8-probe-validation | 2 | 0 | 2 | 0 | 0 | 0 | 0 |
| c3-general-corpus | 4 | 3 | 0 | 0 | 1 | 0 | 0 |
| c5-judge-calibration | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| c6-devtools-benchmark | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| e0-synthetic-identifiability | 4 | 3 | 1 | 0 | 0 | 0 | 0 |
| e1-contextual-composition | 1 | 0 | 1 | 0 | 0 | 0 | **1** |
| e10-self-semantics | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| e2-mapping-operator-frontier | 2 | 1 | 0 | 0 | 1 | 0 | 0 |
| e3-developmental-wordnet | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| e4-small-lm | 29 | 29 | 0 | 0 | 0 | 0 | **9** |
| e5-explainability | 5 | 0 | 0 | 0 | 0 | 5 | 0 |
| e7-self-authoring | 6 | 2 | 0 | 0 | 4 | 0 | 0 |
| e9-retrofit | 63 | 42 | 6 | 0 | 1 | 14 | **3** |
| t1-open-clinical | 3 | 2 | 0 | 0 | 0 | 1 | 0 |
| t2…t6 track corpora, wp-probe-validation | 10 | 10 | 0 | 0 | 0 | 0 | 0 |

**Dirty-tree manifests** (`git_dirty: true` at run start):
- `experiments/01b-global-local-relational-factorization/runs/stage-b-reduced-v2` (`f9d49d8`)
- `experiments/01c-developmental-relation-discovery/runs/basis-confirmation-development-v2` (`5762dca`)
- `experiments/01c-developmental-relation-discovery/runs/source-neighborhood-development-v2` (`5762dca`)
- `experiments/b6-host-memory/runs/3090-v1` (`2ad45e5`; disclosed, R0.2)
- `experiments/b8-probe-validation/runs/v1` (`e2078a4`; disclosed)
- `experiments/b8-probe-validation/runs/v1-ptq` (`8e83d64`; disclosed, R0.2)
- `experiments/e0-synthetic-identifiability/runs/d02-parent-routing-v2` (`01e8090`; superseded variant)
- `experiments/e1-contextual-composition/runs/v1` (`783358b`; **also not committed**)
- `experiments/e9-retrofit/memory/qwen3-v1` (`783358b`; engineering probe)
- `experiments/e9-retrofit/report/t5` (`783358b`; **the R9 report itself**)
- `experiments/e9-retrofit/runs/t5/SmolLM2-135M-frozen-P0-s1/{edit,edit-int4,zeroshot,zeroshot-int4}` (`1b020b4`)

**Run folders without a manifest:**
- `experiments/00-capacity-and-algebra/runs/realistic`
- `experiments/c3-general-corpus/runs/v1` (old machine; superseded by v2)
- `experiments/c5-judge-calibration/v1`
- `experiments/c6-devtools-benchmark/v1` (a data card with a leakage audit, but no provenance)
- `experiments/e2-mapping-operator-frontier/runs/v1` (old machine, untrusted, committed)
- `experiments/e3-developmental-wordnet/runs/hrr-v1` (old machine, untrusted, committed)
- `experiments/e7-self-authoring/smoke/{devtools-cpu,general-cpu,general-cpu/report-r6,verify-cpu}` (smoke runs)
- `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-lora-C2-s1` (training in progress)

Also without a manifest: `experiments/b10-benchmarks/generation.json`. It holds no run artefact the script recognizes and records no git state.

**Finished run folders not committed** (on disk in the main checkout, absent from both `HEAD` and `main`):
- `experiments/e1-contextual-composition/runs/v1`. gates.md D1 cites it.
- the 9 D4.0 recipe runs, `experiments/e4-small-lm/runs/recipe/50M-C0@b{64k,128k,256k}_lr{0.001,0.002,0.004}-s1`. gates.md D4.0 cites them.
- `experiments/e9-retrofit/runs/t5-qwen3/Qwen3-1.7B-Base-{frozen-P0,lora-C0p}-s1`. Expected: they will be committed with their stage.

**Data cards without git state.** The 14 E9 dimension-3 item manifests and the 5 E5 item manifests record no commit. They do record the ontology/alias-table sha256 and the contamination checks, and they are committed, so this is acceptable. Recording the builder commit in future item manifests would close it.

**Configs.** Every `*.yaml` under `experiments/` outside `runs/` is committed on `main`, including the E9 configs up to Qwen3-4B.

**Commit drift inside one comparison.** The E9 T5 seed-1 trainings span three commits (`bbe4044`, `eeb1833`, `db75346`). `git diff` shows no change to `training/`, `span_channel.py`, `compose.py`, `integrations/` or `data/` between them. Likewise, no evaluation module changed between `1b020b4` and `db75346`.

### 2.2 Holdouts

Recomputed as sha256 over the sorted names, joined by newlines (`tracks.common.names_sha256`), from each run folder's `holdout_concepts.txt`, and compared with every recorded value: the run summary, `items/holdout.json`, and `expected_holdout_sha256` in the track configs.

| Track | concepts | recomputed | pinned | recorded values agree |
|---|---:|---|---|---|
| C3 general (WordNet × FineWeb-Edu) | 4,105 | `7f2462ed…` | match | yes (2) |
| T1-open clinical | 1,976 | `1c477afd…` | match | yes (4) |
| T2 developer tools | 1,078 | `75ee2217…` | match | yes (5) |
| T3 product | 182 | `df7acfde…` | — | yes (3) |
| T4 chemistry | 429 | `b58e504f…` | match | yes (5) |
| T5 enterprise glossary | 360 | `e7313dce…` | match | yes (5) |
| T6 legal | 186 | `8148fa06…` | — | yes (3) |

Host relinks reuse these holdouts:
- C3 SmolLM2/Qwen2.5 record `7f2462ed…`;
- T4 Qwen3 pins `b58e504f…`;
- T5 Qwen3 and Qwen3.5 pin `e7313dce…`.

reproducibility.md §4 still lists T2 as "no hash, open item". The T2 hash now exists (`75ee2217…`, `t2.yaml`), so that row is stale.

### 2.3 Leakage, contamination and audits

- **Contamination-free invented names.** Checked against WordNet lemmas, alias-table words, the tokenizer vocabulary and the first 20M training tokens:
  - E9 new-word items for T5 (SmolLM2, Qwen3, Qwen3.5), T4 (SmolLM2, Qwen3), T1 and WordNet: 78.8–93.1M characters checked, 0 rejected, 0 reserved clashes;
  - E5.4 synthetic items for GPT-2, SmolLM2 and Qwen2.5: 113.8–115.6M characters.
- **Held-out leakage into training text.** `leaked_into_train = 0` for:
  - T5 (`v1`, `v1-qwen3`, `v1-qwen35`);
  - T2 (`v1`, `v1-gpt2`);
  - the C6 benchmark.
- **Held-out spans in host training corpora.** C3 host corpora: 0 held-out spans in train (reproducibility.md §4).
- **Evaluation audit.** The audit of 01a–01c (resources/plan-improvement/audit.md F1–F12) found split leakage, test-set selection and metrics inflated by construction. The protocol-2 re-runs are the reported numbers.
- **Cross-machine reproduction.** R0: E0 and D0.2 bit-exact; B6/B8 within tolerance.
- **Not found.**
  - No contamination check for the *natural-text* tracks against host pretraining (T4 ChEBI names, T1 PubMed 2025–26 abstracts). R9 and the novelty check flag host overlap as a risk.
  - No lexical-overlap (filler-copy) split yet (A-B5; WP-PQ1).

### 2.4 Pre-registration and documentation state

- `experiments/e5-explainability/preregistration.md`: **present**.
- `experiments/e4-small-lm/preregistration.md`: **not found**. G2 has not been taken, and the core experiment cannot start under its own rules without it.
- E9 has no pre-registered decision rule. Its questions and readouts were written in execution.md before the runs. E10 has pre-written refutation clauses.
- **Stale documents:**
  - queue.md still shows D1 as "queued" (it is done), D4.0 as "todo" (9 of 11 cells done) and E1 runs as not done.
  - reproducibility.md §6 ledger still says "*fill from logs*", and §5 lists the S0 pilot as pending.
  - claims.md has no rows for E9/E10 (claims A–D) or for the D1 outcome.
- **Data card.** The full list of 158 data files with hashes is `~/data/vsa-llm/DATA_SOURCES.md` + `SHA256SUMS`, *outside the repository*. reproducibility.md §3 summarizes it with hash prefixes. Commit a copy (or the `SHA256SUMS` file) for the artefact release.

---

## 3. Compute ledger

**Source:** `.jobs/*.json` of the main checkout, read-only.
- **Period covered:** 2026-10-02 17:36 to 2026-10-03 15:56 UTC.
- **Timing:** the last attempt of each job. Only 3 probe jobs were retried, and they were short.
- **Not covered** (it ran outside the queue):
  - the old machine;
  - the R0 re-runs of B6/B8;
  - C4 throughput;
  - CPU experiments (E0, E10, corpora).

| Block | priorities | done | running | pending | GPU-h | CPU-in-queue h |
|---|---|---:|---:|---:|---:|---:|
| S0 pilot, 50M training | 10 | 10 | 0 | 0 | 4.14 | — |
| S0 pilot, SmolLM2-135M CPT training | 11 | 3 | 0 | 0 | 0.89 | — |
| S0 pilot, probes (both) | 12 | 13 | 0 | 0 | 0.22 | — |
| E2 frontier (`v2`) | 20 | 1 | 0 | 0 | 0.90 | — |
| E9 engagement check | 20 | 5 | 0 | 0 | 1.42 | — |
| E10.1 anchors | 21 | 1 | 0 | 0 | 0.01 | — |
| E9 T5 SmolLM2 seed 1 (training, evaluations, quant, report) | 22–25 | 59 | 0 | 0 | 6.67 | 0.36 |
| D4.0 recipe sweep | 25, 30 | 9 | 0 | 2 | 3.66 | — |
| E9 Qwen3 memory probe | 26 | 1 | 0 | 0 | 0.04 | — |
| E9 T5 Qwen3 seed 1 | 26–29 | 2 | 1 | 56 | 3.51 (+ 0.5 running) | — |
| E1 contextual (`v1`) | 30 | 1 | 0 | 0 | 0.31 | — |
| **Total done** | | **105** | **1** | **392** | **21.76** | **0.36** |

**CPU work outside the queue:**
- E10.0: 0.72 CPU-h.
- E10.1: ≈ 8.0 CPU-h (4 single-threaded workers × 7,207 s wall).
- E0 and corpus builds: not timed in a manifest.

**Planned vs measured.** The plan assumed E1 and E2 would each cost 10–20 GPU-h; they cost 0.3 and 0.9. The SmolLM2 E9 block per track × seed measures 7.0 queue-hours, against ≈ 6.5 planned.

**Measured unit costs used in §4:**

| Run type | GPU-h |
|---|---|
| SmolLM2-360M full fine-tune, 50M tokens | 1.11–1.14 per run |
| SmolLM2-135M full fine-tune, 50M tokens | 0.51–0.52 per run |
| Per-run evaluations, 360M (probes + zero-shot + edit, bf16 and INT4) | ≈ 0.13–0.24 |
| Per-run evaluations, 135M | ≈ 0.09–0.14 |
| `e4_quant`, per stage | ≈ 0.29 per corpus |
| Qwen3-1.7B LoRA-64 C0′, 50M tokens | 3.49 |
| 50M from scratch, 100M tokens | ≈ 0.41 |

**Queued but not run (from `.jobs`):**
- E9 T5 Qwen3 s1, rest: ≈ 14 GPU-h left of ≈ 18;
- E3: ≈ 2;
- D4.0 32k cells: ≈ 0.8;
- E9 SmolLM2 T5 s2–3: ≈ 13;
- T4 s1: ≈ 6.5;
- T1 s1: ≈ 7;
- WordNet s1: ≈ 6.5;
- Qwen3 T5 s2: ≈ 18;
- Qwen3.5 probe: ≈ 0.5, then its block ≈ 20–25;
- Qwen3-4B: ≈ 34 + evaluations;
- E7 D7.1: ≈ 1.

Total ≈ 126 GPU-h, plus the Qwen3-4B evaluations. WP-PQ1 (priority 51) and WP-PQ2's dimension-3 baselines come on top (evaluation-only: ≈ 4.3 GPU-h for T5 SmolLM2 s1, ≈ 10 with s2–3, ≈ 10 for the Qwen3-1.7B/0.6B s1 block).

---

## 4. Prioritized gaps

Estimates use the measured unit costs of §3. They are labelled *(plan)* where they come from execution.md, and *(illustrative)* where the arm count is not fixed yet.

### P0: before the corresponding claim enters the abstract

| # | Gap | Claim | Owner / status | GPU-h |
|---:|---|---|---|---:|
| 1 | **≥ 3 seeds for E9 T5 SmolLM2** (seeds 2–3) | A, B, C | queued (priorities 31–34) | ≈ 13 |
| 2 | **E9 dimension-1/2 controls:** operator/specificity ablation (A-B6), same-site FVT / definition-encoder / KG rows (A-B1/A-B2/A-B4), filler vs non-filler split (A-B5), channel off at INT4, GPTQ/AWQ/HQQ/NF4, quantized embeddings, bit-matched side module, DiD (B-B1–B-B3, B-B6, B-B7) | A, B | **WP-PQ1**, in progress (priority 51) | ≈ 1.35 per SmolLM2-360M arm × seed with evaluations; e.g. 6 arms × 3 seeds × 2 tracks ≈ 49 *(illustrative)*; quantizer sweeps are evaluation-only (≈ 0.3–0.6 per stage per quantizer) |
| 3 | **E9 dimension-3 baselines:** in-context frame / IKE (C-B2), ROME / MEMIT / AlphaEdit on C0′/P0 (C-B1), frame transplant, channel-off audit, intra-entity locality (C-B4–C-B6), FVT / definition-encoder rows (C-B3 a–c) | C | **WP-PQ2**: implemented (`3661b81`), not queued; `e9_plan --track t5 --dim3-baselines --queue` | T5 SmolLM2 s1 ≈ 4.3, s1–3 ≈ 10; Qwen3-1.7B/0.6B s1 ≈ 10.2 (per run: SmolLM2 0.3–0.8, Qwen3-1.7B 1.0–2.3) |
| 4 | **E4 core:** write and commit `preregistration.md` (G2), finish D4.0 (2 cells), then D4.8, D4.7, D4.9, D4.1 at 3 seeds | H-A (core), OP.1, VSA.1, N.M4 | not started | ≈ 0.8 + 15 + 12 + 50 + 65 ≈ **143** *(plan)* |
| 5 | **Natural-text tracks for "domain terms":** T4 and T1-open s1 (queued), then s2–3 | A | s1 queued | s1 ≈ 13.5; s2–3 ≈ 27 more |
| 6 | **In-context verbalized frame for the LM-loss strata** (A-B3; ECBD/Onoe protocol: frame or definition at first mention, continuation tokens scored, token cost reported) | A | **unassigned**: PQ1's list covers row sources, PQ2's covers the dimension-3 items | ≈ 0.1–0.2 per evaluated model *(illustrative; one pass over 1,024 windows)*; ≈ 1–2 for one stage |
| 7 | **E10 baselines:** AMIE-style rules, IterE-style properties, TransE/RotatE/ComplEx recovery, null-world FDR, reusable-holdout control (D-B1, D-B2, D-B4, D-B5) | D | **WP-PQ2: done** (`0a38784`; R10 "E10 baselines"). They narrow D (see §1.4); the new P0 gap is a null-calibrated slot acceptance test | 0 GPU (≈ 0.9 CPU-h for E10.0 + E10.1) |
| 8 | **E10.2 joint LM test** (D stays out of the abstract until then) | D | not started | ≈ 20–30 *(plan)* |
| 9 | **≥ 3 seeds on Qwen3** (s1 running, s2 queued, s3 not queued) | A–C on a modern host | partly queued | s3 ≈ 18 |
| 10 | **Reproducibility hygiene for cited results** (see the list below) | all | — | ≈ 0.4 |

**Gap 10, the hygiene items:**
- commit E1 `runs/v1` and the 9 D4.0 recipe runs;
- re-run E1 from a clean tree (0.31 GPU-h) or disclose its dirty manifest;
- regenerate `report/t5` from a clean tree (CPU, ≈ 0.4 h in the queue);
- re-run the 4 dirty P0-135M evaluations (≈ 0.03 GPU-h);
- mark the old-machine folders E2 `v1`, E3 `hrr-v1` and C3 `v1` as stale;
- update claims.md (D1; claims A–D), queue.md, and reproducibility.md (§5, §6, and the T2 hash).

### P1: for the full paper

| # | Gap | Claim | GPU-h |
|---:|---|---|---:|
| 11 | Method baselines (§6.5): KnowLA-style per-concept KGE at our site and at KnowLA's; a KGE-initialized jointly trained table; a retrofit Engram adapter / C1h-E (A-B7); a parameter-matched LoRA plus an OOD-retention suite; an injection-site ablation; a linker-noise curve | method, A | ≈ 1.35 per SmolLM2-360M arm × seed → ≈ 6 arms × 3 seeds ≈ 24 on T5 *(illustrative)*; the injection-site and linker-noise arms also belong in E4 |
| 12 | Remaining C controls: relation-shuffled and filler-only frames (C-B4), dose–response over 1–3 edges (C-B5), portability / 2-hop and alias subjects (C-B6), ALCUNA/COMPS-style items with about 700 items per arm, and an item × seed mixed-effects model (C-B7); a CoLLEGe-style generator and the gradient upper bound (C-B3 d/e) | C | evaluation-only: ≈ 0.05–0.1 per run per item set; building the items is CPU |
| 13 | Remaining B controls: revert diagnostics (B-B4, CPU); a QAT/QLoRA arm with and without the channel (B-B5); a third host family for B (Qwen3, queued) | B | B-B5 ≈ 1.35 per arm × seed on SmolLM2-360M (≈ 8 for 2 arms × 3 seeds) |
| 14 | Remaining D controls: clustering / TransG / IRM discovery (D-B3), dreaming ablations (D-B6), resonator and Lasso frame inference (D-B7), an L2-SP/EWC/Faruqui sweep (D-B8), and WN18RR / FB15k-237 hidden relations with > 3 seeds and an initial-mass sweep (D-B9) | D | CPU (E10.0 costs 0.7 CPU-h per grid; WN18RR at E10.1 scale ≈ 8 CPU-h) |
| 15 | E3 developmental WordNet (queued) | H-D.1 | ≈ 2 |
| 16 | Qwen3-4B (single seed, descriptive) and the Qwen3.5-2B/0.8B block (separate host family) | generality | ≈ 34 + evaluations; ≈ 20–25 + 0.5 probe |
| 17 | E5: faithfulness, sense alignment, zero-shot rows, LLM-graded estimate (needs D4.1) | H-F, H-E | ≈ 30 *(plan)* |
| 18 | E7 self-authoring (D7.1 queued; D7.2/D7.3 after G3) | H-G | ≈ 1 + ≈ 40 *(plan)* |
| 19 | E8 tracks (D4.5 T1-open from scratch, D8.2–D8.6) and the cross-track table | E8.1 | ≈ 25 + ≈ 45 *(plan)* |
| 20 | E4 extensions: D4.4 CPT on three hosts, D4.3 quantization and table compression (H-B.2) | H-B | ≈ 55 + ≈ 10 *(plan)* |
| 21 | Contamination check of the natural-text tracks against host pretraining (T4, T1; the Onoe et al. lexical-overlap confound), and `SHA256SUMS` / DATA_SOURCES committed as the data card | A, reproducibility | 0 (CPU) |

**Budget view.**

| Item | GPU-h |
|---|---:|
| P0 queued already (items 1, 5 for s1, 9 for s1–s2) | ≈ 58 |
| P0 still to queue: E4 core (item 4) | ≈ 143 |
| P0 still to queue: E10.2 (item 8) | ≈ 25 |
| P0 still to queue: Qwen3 s3, T4/T1 s2–3 | ≈ 45 |
| P0 still to queue: WP-PQ1 (item 2) | ≈ 49 *(illustrative)* |
| WP-PQ2 dimension-3 baselines (item 3) | ≈ 10 (T5 SmolLM2 s1–3) + ≈ 10 (Qwen3 s1) |
| P1 (items 11–20) | ≈ 300 |

The committed program was ≈ 400 GPU-h. Without escalation, P0 alone fits: ≈ 22 spent + ≈ 320 for P0, plus the PQ2 evaluation-only jobs. The full P1 list does not fit on top of the E4 core: the author's G6 decision applies.
