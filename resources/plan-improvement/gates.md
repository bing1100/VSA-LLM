# Gate decisions

Decisions taken by applying the pre-registered rules in [tasks.md](tasks.md) and [experiments.md](experiments.md), recorded for the author's review. Work continues after each gate; the author can overturn any decision here. G6 (escalation beyond the committed GPU budget) is the author's.

## G0 — interfaces frozen (2026-09-30)

**Inputs.** A1–A5 on `main` (5762dca, 84d1ff4, 22834c3); B1–B4 implemented and tested (dedeff7, ca06236); C1 literature note ([related-work.md](related-work.md)).

**Decision: freeze the M1/M3/M4 interfaces as implemented** — `FrameComposer(schedule, atomic_count, relation_count, dimension, operator, mode, concept_factor, key_dimension, context_dimension, temperature, weight_mode, output_dimension)` with `compose/forward/explain/parameter_groups`, `DevelopmentalDictionary.begin/observe/grow/consolidate/allocate`, and the P1/P2 modules in `context.py`. M4 is specified (formulation §4) and will be built against this composer interface (B5).

**Changes forced by C1** (applied to the plan, not to the interfaces): claims narrowed to the wording in proposal §7; new baselines C1h (hashed n-gram span memory) and C3t (type/relation-only composition) in E4, free-sense attention in E1, Anderson–Darling split test and NP-MSSG/AdaGram in E0.2/E3, GRAM on T1, equal-bytes compression tables, à la carte and CoLLEGe-style generators, EntiGraph-style and OLLM/LLMs4OL controls in E7. None of these requires an interface change: C1h and C3t are span-channel conditions (B5/B7), the others are experiment-side baselines.

**Legacy reproducibility.** Single-threaded, the protocol-aware runners reproduce the four recorded configs bit-for-bit on all shared columns (old vs new code, 1,944 rows, 0 differing cells). Multithreaded CPU training is not bit-reproducible across processes on this machine, so new runs set `num_threads: 1`.

## G1 — mechanisms recover planted structure (2026-09-30)

**Inputs.** E0 development run, 3 seeds, CPU single-threaded (`experiments/e0-synthetic-identifiability/runs/e0-development/`, commit `2cc5d0b`).

**Decision: PASS — proceed to E1–E3.**

- D0.1: attentive composition with a context query recovers held-out concept × held-out context targets far better than the static bundle (+0.36 to +0.46 cosine across context strengths 0.5–4, CIs exclude zero) and equals it on the static teacher (no harm).
- D0.3: the induced concept factor transfers to composition-disjoint concepts (held-out cosine 0.976 at k = 4 vs 0.60–0.70 for the free factor and 0.62 for M0); rank k = k* = 4 is best, larger ranks lose a little.
- D0.2: M3 with the permutation null recovers all planted polysemous atomics and relation sub-types at convergence (precision = recall = 1.0, ARI on training usages 1.0, no surviving false splits). The screen alone over-splits (raw false-split rate 0.38); consolidation merges spillover splits back, so the consolidation phase is required, not optional. The Anderson–Darling variant keeps more false splits (0.08) and coherence-only misses senses (recall 0.92 atomics, 0.67 relations); random splits do not recover structure (ARI 0.02).

**Caveat found by E0 (not covered by the gate).** Held-out usages of a split vector carry no gradient and were routed near chance (ARI ≈ 0.05), so splitting slightly lowered held-out cosine (0.939 vs 0.946 without growth). Two changes follow: routing now uses the frame context (formulation §3.4), and M3 gains `route_unobserved: parent`, which keeps the unsplit vector for usages without evidence (single-seed check: held-out cosine 0.948 vs 0.949 without growth; precision 0.89, recall 1.0). A three-seed D0.2 run of the parent variant is in `runs/d02-parent-routing/`; E3 and E4-C6 use whichever variant has the better held-out cosine at equal training-usage recovery.

**G1 caveat resolved (2026-09-30).** Three-seed D0.2 with parent fallback and synced parents (`runs/d02-parent-sync/`): held-out cosine with M3 0.9503 vs 0.9462 without growth (atomics) and 0.9094 vs 0.9022 (relations), with precision 0.96 / 1.0, recall 1.0, ARI 1.0 on training usages and surviving false-split rate 0.008 / 0. This variant (`route_unobserved: parent`, `sync_parent: true`) is the M3 default for E3 and E4-C6.
