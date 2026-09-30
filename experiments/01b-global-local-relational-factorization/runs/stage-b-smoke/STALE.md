# Stale run

`stale: true` — marked 2026-09-30 by the implementation audit ([audit.md](../../../../resources/plan-improvement/audit.md), finding F4).

Per-relation `target_mrr` in `metrics.csv` was computed against 3–11 candidates (41–42 in the current code), inflating it by +0.22 on average. Re-summarizing with current code flips the recorded PASS to FAIL because the matched `shuffled_low_rank` control is missing. The YAML gained `min_paired_wins`/`min_cosine_gain` after the run.

Keep for provenance only; do not cite these numbers.
