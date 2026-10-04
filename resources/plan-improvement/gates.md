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

## BLOCKER — faulty RAM on the experiment machine (2026-09-30)

**Evidence.** `scripts/memcheck.py` (6 GiB, 4 address-dependent patterns, 3 re-reads each) found **204 corrupted 64-bit words**, with counts changing between re-reads of unchanged data. The same day: two large Hugging Face downloads failed their sha256 (FineWeb shard 001 twice, LexGLUE eurlex train, multi_eurlex), FineWeb shards whose on-disk sha256 is correct raised "Corrupt snappy compressed data" when decompressed, the Rust tokenizer panicked / returned an out-of-range id, `realloc(): invalid pointer`, a transient "bad marshal data" when importing a `.pyc` that is intact on disk, and background processes died without a traceback. `git fsck --full` is clean.

**Consequence.** Results computed on this machine cannot be trusted until the memory is fixed. All experiments were stopped. Partial E1/E2/E3/C3 runs are left uncommitted.

**Reproducibility evidence so far.** The single-threaded legacy replay (old vs new code, 1,944 metric rows) was bit-identical, and repeated sha256 of the two FineWeb shards is stable, so corruption is intermittent rather than pervasive; committed results (E0, A6 re-runs, B6/B8, C5 calibration) were produced under the same conditions and must be re-run and compared after the fix before they are relied on.

**Needed from the author.** Run memtest86+ from boot; if it reports errors, disable XMP/EXPO (run the RAM at JEDEC speed) or reseat/replace the failing DIMM; then re-run `python scripts/memcheck.py 20 8` until it reports 0. After that, Claude re-runs the committed experiments, checks they reproduce, and resumes the queue.

**memtester confirmation (2026-10-01).** `memtester 4.7.1` (built from source, run without root on 16 GiB of unlocked memory, 1 loop) reported **2,122 failures at ≥ 308 distinct offsets**, in every test category (stuck address, random value, compare XOR/SUB/MUL/DIV/OR/AND, …). The flips are overwhelmingly single bits at positions 1 and 3 of 64-bit words (1,162 and 994 occurrences), with a further cluster of rarely-flipped higher bits. A consistent low-bit pattern points to a hardware fault on a data line or memory running outside its stable settings (XMP/EXPO overclock), not software. memtest86+ (`/boot/mt86+x64`, GRUB menu) is still the authoritative test; it needs a reboot by the author.

### Resolved: new machine `bhux-tiny` (2026-10-02)

The experiment machine was replaced by `bhux-tiny` (Ryzen 5 8600G, 60 GiB RAM, RTX 3090). Every committed result that later work relies on was re-run there and compared with `scripts/compare_runs.py` (details in [runbook.md](runbook.md) and `reports/R0-reproduction.md`):

| Check | Outcome |
|---|---|
| RAM screen `memcheck.py 16 4` and `16 8` | 0 bad words (old machine: 204 at 6 GiB) |
| E0 development (`2cc5d0b`) | MATCH at 1e-12 (8,100 + 154 cells); G1 passes again |
| D0.2 parent-sync (`9986d72`), parent-routing (`bf5f535`), parent-routing-v2 (`01e8090`) | MATCH at 1e-12 (576 + 115 cells each) |
| B6 host memory | identical peak memory in all 19 rows, tokens/s within 1% (run at HEAD: the recorded dirty-tree commit `2ad45e5` crashes with transformers 4.54) |
| B8 probes (`e2078a4`) | LAMBADA, WiC identical; Spearman within 0.0035 |
| B8 PTQ | identical PPL at HEAD (the recorded commit `8e83d64` predates the full-precision output head of `4fae7cb`) |
| C5 judge calibration v2 | accuracy 30/30, Fleiss κ 0.926 (v1 0.812) |

Env `vsa-repro`: Python 3.12.4, torch 2.11.0+cu128, transformers 4.54.0, numpy 2.2.6, nltk 3.8.1, PyYAML 6.0.2, safetensors 0.5.3, scipy 1.18.1, pyarrow 25.0.1, tokenizers 0.21.4, torchao 0.18.0.

**Decision: blocker lifted.** Committed results stand. Partial E1/E2/E3/C3 outputs from the old machine stay untrusted; their re-runs use new run IDs.

**Judge harness note (C5 v2).** v1 used lean `claude -p --tools ""`; v2 used Claude Code subagents with the default system prompt. Agreement numbers are comparable, individual verdicts are not. Both harnesses pass calibration, so later studies may use either; each study records its harness (see [execution.md](execution.md)).

**Author decisions of 2026-10-02.** T1 runs on open MeSH + PubMed (labelled open-clinical) until SNOMED CT / UMLS / MIMIC files are supplied; a small S0 sanity pilot (≈ 6 GPU-h) runs before the committed E4 blocks; G6 escalation still needs the author's approval. See [execution.md](execution.md).

## D2 (E2) — mapping × operator frontier on frozen GPT-2 anchors (2026-10-02)

**Inputs.** `experiments/e2-mapping-operator-frontier/runs/v2/` (bhux-tiny, commit `07a7315` per its manifest), WordNet and MeSH 2026, 6,000 concepts each, node-disjoint test, 3 seeds, 7 mapping × 7 operator cells plus shuffled-label controls.

**Pre-registered gate (experiments.md E2).** M2-induced transfers at least as well as feature salience and beats binary; the best operator is carried to E4, and a tie with `random_fixed` is recorded for C8 to decide.

**Outcome.**
- Induced vs salience (test MRR): MeSH +0.0065 [−0.0021, +0.0151], WordNet +0.0023 [−0.0057, +0.0102] → **non-inferior, not superior**. Induced vs binary: MeSH +0.0062 [−0.0032, +0.0156], WordNet +0.0028 [−0.0084, +0.0140] → **"beats binary" fails**.
- Validation-selected cell: MeSH `induced_k4 / diagonal`, WordNet `binary / low_rank_tied`. **No operator separates from `random_fixed:hrr` or `untyped`** (selected − random_fixed: MeSH −0.0001 [−0.0054, +0.0052], WordNet −0.0015 [−0.0094, +0.0064]).
- Relation labels carry signal: true − shuffled labels is positive for every family, with CIs above zero for several (e.g. WordNet `hrr` +0.0109 [+0.0042, +0.0176]; MeSH `hrr_identity`, `low_rank_tied`, `random_fixed:hrr`, `untyped`).
- Absolute fit to frozen anchors stays poor (test MRR 0.03–0.08; variance explained < 0), as in 01c.

**Decisions.**
1. E2 gate: partial pass (non-inferior to salience; does not beat binary). Recorded, not a stop rule.
2. **E3 operator: `hrr_identity`.** The WordNet validation pick `low_rank_tied` cannot be used, because M3 relation splitting needs a vector-per-relation family; `hrr_identity` is the best such family on WordNet validation (0.0595 vs 0.0596 for the pick, within noise).
3. **E4 (provisional, finalized at G2):** frozen-anchor experiments do not distinguish operators, so the E4 composition operator stays `hrr` (the family the program is about, and what the S0 pilot uses), with induced rank k = 4 for the attentive mapping, and the question is left to the E4 C8 operator ablation (`hrr`, `diagonal`, `low_rank`, `bounded_residual`, `random_fixed`, `untyped`). Per tasks.md rule 7, reports say "compositional parameter sharing" unless C8 separates `hrr` from `random_fixed`.

## T1-open holdout re-frozen before any training (2026-10-02)

The T1-open slice build (`experiments/t1-open-clinical/runs/slice-v1`) pinned holdout `fd4244ea…`, but the full build produced `1c477afd…`: the slice config capped `eval_general_docs`, which feeds the chars-per-token calibration of the 50/50 PubMed/general mix, so the training stream (and its frequency pre-sample) differed by one document. No run had used either holdout, so the track holdout is re-frozen from the full config (`t1.yaml` pins `1c477afd…`; slice-v1 marked stale). Rule 2 of tasks.md (freeze before the first run that could see it) holds.

## D1 (E1) — contextual composition on frozen GPT-2 anchors (2026-10-03)

**Inputs.** `experiments/e1-contextual-composition/runs/v1` (SemCor, 690 polysemous lemmas, 45,470 sentences, layer 12, 3 seeds). **Gate (experiments.md E1): FAIL.** M1 with the host-state query (P2) beats static M0 on held-out lemmas — variance explained +0.69 [+0.42, +0.96] and context MRR +0.051 [+0.049, +0.053] — but absolute variance explained stays negative (−2.8) and the attention's sense accuracy is below the most-frequent-sense baseline (−0.25 [−0.29, −0.21]). Operators tie again (hrr ≈ diagonal ≈ random_fixed ≈ untyped). Reading: on frozen anchors, a context query improves composition relative to static bundling but does not align with WordNet senses; consistent with E2 and E10.1, frozen-host targets are the wrong regime. Carried to G2 as evidence that context helps (M1 kept for E4 C5) but not as a sense-alignment claim.

## D4.0 recipe sweep (2026-10-03)

`experiments/e4-small-lm/runs/recipe` (C0, 50M, 100M tokens, seed 1): final loss by tokens/step × peak lr — 262k: 5.63 / 5.99 / 5.98 (lr 1e-3 / 2e-3 / 4e-3); 131k: 4.88 / 5.16 / 5.19; **65k: 4.42 / 4.29 / 4.79**. The pilot recipe (262k, 1e-3) is 1.3 nats worse than the best setting at the same tokens: the S0 pilot ran in an under-trained regime, and its effect sizes should be re-measured under the frozen recipe. Two 32k-tokens/step runs (lr 1e-3, 2e-3) are queued before the recipe is frozen.

## D3 (E3) — developmental recovery on WordNet, frozen GPT-2 anchors (2026-10-04)

**Inputs.** `experiments/e3-developmental-wordnet/runs/v2` (operator `hrr_identity`, 3 seeds). **Gate: FAIL** for atomics and relations. Atomics: M3 splits 70.7 per run with precision 0.066, recall 0.29, ARI 0.014 and a false-split rate of 0.185 (gate ≤ 0.05); test MRR equals uniform enlargement (−0.0002 [−0.0076, +0.0072]). Relations: no policy (M3, coherence-only, random, stem-cell pool) splits anything. Reading: as E1, E2 and E10.1, frozen GPT-2 anchors do not carry the sense/relation structure the mechanisms are meant to recover; M3's E0 success (G1) does not transfer to this regime. M3 (C6) is therefore not carried into the E4 screen as a primary condition; it can be revisited in joint training (E10.2).

## D1 (E1) clean re-run (2026-10-04)

`runs/v2` (clean commit) reproduces v1: M1-P2 − M0 on held-out lemmas variance explained +0.66 [+0.36, +0.96], context MRR +0.051 [+0.050, +0.052], sense accuracy − MFS −0.24 [−0.29, −0.20]; gate FAIL. Cite v2.

## D4.0 recipe frozen (2026-10-04)

C0 at 50M × 100M tokens, final loss: 32k tokens/step 4.29 (lr 1e-3) / **4.22 (lr 2e-3)**; 65k 4.42 / 4.29 / 4.79 (lr 4e-3); 131k 4.88 / 5.16 / 5.19; 262k 5.63 / 5.99 / 5.98. **Frozen recipe for 50M from scratch: 32,768 tokens/step (micro-batch 32 × 1), peak lr 2e-3, cosine to 0.1×, warmup 5M tokens.** The 125M recipe gets a two-run lr check (1e-3, 2e-3) at the same tokens/step before D4.1. The S0 pilot (262k, 1e-3) is superseded as an effect-size estimate.
