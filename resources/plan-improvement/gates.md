# Gate decisions

Decisions taken by applying the pre-registered rules in [tasks.md](tasks.md) and [experiments.md](experiments.md), recorded for the author's review. Work continues after each gate; the author can overturn any decision here. G6 (escalation beyond the committed GPU budget) is the author's.

## G0 — interfaces frozen (2026-09-30)

**Inputs.** A1–A5 on `main` (5762dca, 84d1ff4, 22834c3); B1–B4 implemented and tested (dedeff7, ca06236); C1 literature note ([related-work.md](related-work.md)).

**Decision: freeze the M1/M3/M4 interfaces as implemented** — `FrameComposer(schedule, atomic_count, relation_count, dimension, operator, mode, concept_factor, key_dimension, context_dimension, temperature, weight_mode, output_dimension)` with `compose/forward/explain/parameter_groups`, `DevelopmentalDictionary.begin/observe/grow/consolidate/allocate`, and the P1/P2 modules in `context.py`. M4 is specified (formulation §4) and will be built against this composer interface (B5).

**Changes forced by C1** (applied to the plan, not to the interfaces): claims narrowed to the wording in proposal §7; new baselines C1h (hashed n-gram span memory) and C3t (type/relation-only composition) in E4, free-sense attention in E1, Anderson–Darling split test and NP-MSSG/AdaGram in E0.2/E3, GRAM on T1, equal-bytes compression tables, à la carte and CoLLEGe-style generators, EntiGraph-style and OLLM/LLMs4OL controls in E7. None of these requires an interface change: C1h and C3t are span-channel conditions (B5/B7), the others are experiment-side baselines.

**Legacy reproducibility.** Single-threaded, the protocol-aware runners reproduce the four recorded configs bit-for-bit on all shared columns (old vs new code, 1,944 rows, 0 differing cells). Multithreaded CPU training is not bit-reproducible across processes on this machine, so new runs set `num_threads: 1`.
