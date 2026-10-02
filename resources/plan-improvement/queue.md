# Work queue

Every committed task from [tasks.md](tasks.md), in the order it will be executed. Claude works down this list continuously and updates the status column as tasks land (`todo` → `running` → `done`, or `blocked` with a reason). Gate decisions G1–G5 are taken by applying the pre-registered rules and recorded in [gates.md](gates.md) for the author's review; work continues after each. Escalation tiers X1–X4 (GPU work beyond the committed ≈ 400 hours) are **not** queued: they wait for the author's approval at G6.

> **2026-10-02:** blocker resolved on the new machine `bhux-tiny` (see [gates.md](gates.md)). The full remaining queue, the author's decisions of 2026-10-02 (open-clinical T1, S0 sanity pilot first) and the report deliverables are in [execution.md](execution.md).

GPU jobs are queued through the local job queue (B11) once it exists; until then they run one at a time from the shell.

| # | Task | Phase | Depends on | Status |
|---:|---|---|---|---|
| 1 | A1 evaluation validity | 0 | — | done (5762dca) |
| 2 | A2 control fairness | 0 | — | done (5762dca) |
| 3 | A3 gates and provenance | 0 | — | done (5762dca) |
| 4 | A4 statistics helpers | 0 | — | done (5762dca) |
| 5 | A5 splits and nits | 0 | — | done (5762dca) |
| 6 | A6 protocol-2 re-runs, stale marks, dossier errata, proposal §2 refresh | 0 | A1–A5 | done |
| 7 | W1 dossier index and roadmap update | W | A6 | done |
| 8 | C1 literature search → `related-work.md` | C | — | done (ca06236; baselines applied to experiments.md) |
| 9 | G0 freeze M1/M3/M4 interfaces | gate | A1–A5, C1 | done ([gates.md](gates.md)) |
| 10 | B1 `compose.py` (FrameComposer, M0/M1/M2) | B | A5 | done |
| 11 | B2 relation adjoints | B | A2 | done |
| 12 | B9 synthetic teachers for E0 | B | B1 | done |
| 13 | B3 `context.py` (P1 encoder, P2 sidecar) | B | B1 | done |
| 14 | B4 `developmental.py` (M3) | B | B1, B2 | done |
| 15 | D0.1 contextual composition identifiability (CPU) | D | B1, B9 | done |
| 16 | D0.3 factored-mapping transfer (CPU) | D | B1, B9 | done |
| 17 | D0.2 split detection (CPU) | D | B4, B9 | done |
| 18 | G1 mechanisms recover planted structure | gate | D0 | done ([gates.md](gates.md)) |
| 19 | B5 `span_channel.py` (linker, cardinality report, injection) | B | B1 | done |
| 20 | B6 `integrations/transformers.py` (+ SmolLM2-360M download) | B | B5 | done |
| 21 | B7 from-scratch LM training harness | B | B5 | done |
| 22 | B14 `convergence.py` | B | B7 | done |
| 23 | B11 local GPU job queue | B | B7 | done |
| 24 | B8 evaluation harness (probes, PTQ, faithfulness) | B | B7 | done (LAMBADA GPT-2 0.310 vs ≈ 0.325 reference; INT8 ≤ 1.5% PPL, INT4 +12–37%) |
| 25 | B10 CI and benchmarks | B | B1–B5 | done (CI workflow, property tests, `experiments/b10-benchmarks/generation.md`: sparse composition 100–280× faster and ~100–220× less memory than materializing 100k rows on CPU) |
| 26 | C3 general corpus, linking, holdout, cardinality tables | C | B5 | done (`runs/v2`: 1.10B train tokens, 0 skipped docs, holdout 4,105 concepts / 5,900 entries, sha256 `7f2462ed…`) |
| 27 | C2 clinical data (SNOMED CT, UMLS, MIMIC, PubMed) | C | B5 | T1-open done (`experiments/t1-open-clinical/runs/v1`: MeSH 2026 + PubMed 2026, 300M GPT-2 + 130M SmolLM2 train tokens, holdout 1,976 entries `1c477afd…`; held-out stratum feasible at ℓ_min 2, rare stratum borderline → exploratory); SNOMED CT / UMLS / MIMIC waiting for the author's files |
| 28 | C4 throughput benchmark on the 3090 | C | B6, B7 | done |
| 29 | B12 `authoring.py` (M5) | B | B4, B5, B6 | done |
| 30 | B13 `judging.py` (Claude Code judge) | B | — | done |
| 31 | C5 LLM-judge protocol and calibration set | C | B13 | done |
| 32 | D1 E1 contextual composition on frozen anchors | D | A6, B1, B3, C3 | queued |
| 33 | D2 E2 mapping × operator frontier | D | A6, B1, B2, C2 | done (`runs/v2`; gate partial: induced ≥ salience, not > binary; operators tie with random_fixed — [gates.md](gates.md)) |
| 34 | D3 E3 developmental recovery on WordNet | D | D2, D0.2 | queued |
| 34a | S0 sanity pilot (author request 2026-10-02; not a gate) | D | C3, WP-E4R, WP-host | running |
| 35 | D4.0 harness shake-out; freeze training recipe | D | B7, B8, B11, B14, C3 | todo |
| 36 | D4.8 early no-channel baselines | D | D4.0 | todo |
| 37 | G2 pre-registration (`preregistration.md`, escalation rule) | gate | D1–D3, C4 | todo |
| 38 | C6 developer-tools benchmark | C | B5 | done (benchmark v1, leakage audit 0) |
| 39 | D4.7 span-cardinality feasibility | D | C3, D4.0 | todo |
| 40 | D4.9 50M screen | D | G2 | todo |
| 41 | D4.1 125M convergence runs | D | D4.9 | todo |
| 42 | G3 core claim at the committed budget; W2 interim report | gate | D4.1 | todo |
| 43 | D4.4 continued-pretraining track | D | B6, G3 | todo |
| 44 | D4.3 quantization and table compression | D | B8, D4.1 | todo |
| 45 | D5.1, D5.2, D5.5 faithfulness, sense alignment, frequency | D | D4.1 | todo |
| 46 | D4.5 clinical track T1 | D | C2, G3 | todo |
| 47 | D5.3, D5.4 LLM-graded rating study, zero-shot insertion | D | C5, C6, D4.5 | todo |
| 48 | G4 explainability and zero-shot claims | gate | D5 | todo |
| 49 | C7 application tracks T3–T6 data | C | B5, B8 | done (T4 chemistry, T5 glossary feasible; T3 product, T6 legal infeasible on the seen-rare stratum — held-out-only gate) |
| 50 | D7 self-authoring round 1 | D | B12, D4.4 | implemented (WP-E7; D7.0/D7.1 can run before G3, D7.2/D7.3 after G3) |
| 51 | G5 self-authoring claim | gate | D7 | todo |
| 52 | D8.2–D8.6 application tracks T2–T6 | D | C6, C7, G3 | todo |
| 53 | D8.8 cross-track report; D4.6 E4 report | D | D8, D4 | todo |
| 54 | W3 evidence report and escalation request | W | all above | todo |
| 55 | G6 escalation decision (author approval required) | gate | W3 | waiting on author |
