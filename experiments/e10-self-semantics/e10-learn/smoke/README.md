# E10.L CPU smokes (label SMOKE: pipeline checks, not results)

All of these ran on 2026-10-08 on the CPU, with 4 threads and SmolLM2-360M seed-1 checkpoints. The extractions are
deliberately tiny, so almost nothing is testable. The numbers check that each stage runs and what it costs; they are
not evidence for or against L4a / L4b. These are the final-code reruns. The smokes seen before the pre-registration
(T4 with 300 probes and no evidence filter) are listed in `preregistration.md` §10.

| Folder | What | Outcome |
|---|---|---|
| `t4-c2-s1` | T4 erased-edge recovery: C2 rows; evidence = 64 evaluation windows; KGE 5 epochs | 16 probes; 23 proposals (18 testable); 0 accepted |
| `t5-c2-s1` | T5, C2 rows; evidence = 128 evaluation windows | 85 probes; 170 proposals (125 testable); 0 accepted (`holm+decoy`) |
| `t5-c5full-s1` | positive control (trained C5 store of the full frame) | proposals cover 0.79 of erased edges; 7 accepted at precision 1.0 |
| `t5-features-s1` | C0′ hidden-state passive vector | 29 probes; 58 proposals; 0 accepted |
| `place-t5-heldout-fixture-s1` | placement evaluator on a 40-item fixture (T5 held-out terms, `is_a` / `owned_by`) | MRR: store 0.39, text 0.49, KGE 0.17–0.22 |

`extraction-cost.json` holds the cost of each extraction.

**Pool AUC in the smokes** (shared candidate pool; recall at precision 0.8 in brackets):

| Track | decompose | prior | LRE | AMIE | TransE | RotatE | ComplEx |
|---|---|---|---|---|---|---|---|
| T4 | 0.73 (0.34) | 0.65 (0.37) | 0.86 (0.54) | 0.51 (0.03) | 0.88 (0.57) | 0.83 (0.46) | 0.83 (0.49) |
| T5 | 0.53 (0.05) | 0.50 | 0.53 | 0.50 | 0.55 | 0.50 | 0.49 |

- On T5 every method is near chance.
- The C2-row decoder's out-of-fold R² is ≈ 0 on both tracks: 34-d free rows carry little that a linear map can turn into a
  256-d store.
- Every null false-acceptance rate is 0–3%.

**Costs** (these feed the estimates in `queue-commands.sh`):
- Extraction: a 360M forward runs at ≈ 500 tokens/s on 4 CPU threads (T4: 64 × 512 tokens in 79 s including ≈ 10 s
  load). Mentions: 3,000 store-entry mentions in 92 s; 349 items in 14 s.
- Erased run: T4 at 16 probes took 38 s; KGE training dominates, at 25.9 s for 3 models × 5 epochs on 59k triples.

A TaxoExpan-noun dev placement smoke on the WordNet store is kept local, because that data carries no explicit
licence: 349 items, gold coverage 0.53; MRR 0.001 (store), 0.039 (text-only), ≤ 0.002 (KGE).
