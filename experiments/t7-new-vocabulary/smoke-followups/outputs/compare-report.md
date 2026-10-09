# SMOKE (not a result): T7-ROOD against a copy of itself

Host SmolLM2-135M; seeds 1 (P0 stands for every seed); T7-ROOD runs `/tmp/claude-1000/-home-bhux-workplace/71272da5-02b7-47b1-a057-5887c33fb83d/scratchpad/smoke-followups/e9/runs/t7rood`, T7 v1 runs `/tmp/claude-1000/-home-bhux-workplace/71272da5-02b7-47b1-a057-5887c33fb83d/scratchpad/smoke-followups/e9/runs/t7`. Relative differences Σ (candidate − reference) / Σ reference over the final-evaluation windows, pooled over seeds; 95% cluster bootstraps over windows (200 resamples, the same windows for both tracks). Pre-registration: `experiments/t7-new-vocabulary/preregistration-rood.md` §4.

**Flag:** single seed: CIs cover evaluation windows only.

## after_heldout (16 windows, 16 target tokens per track and seed pool)

| prediction | contrast | T7-ROOD | T7 v1 | Δ = ROOD − v1 | p | predicted | reading |
|---|---|---|---|---|---:|---|---|
| P2 | (C5 − C0') / C0' | +0.82% [+0.53, +1.31] | +0.82% [+0.53, +1.31] | +0.00% [+0.00, +0.00] | 1.0000 | Δ ≤ 0 | holds (point estimate ≤ 0) |
| P3 | (C0' − P0) / P0 | -0.67% [-0.88, -0.30] | -0.67% [-0.88, -0.30] | +0.00% [+0.00, +0.00] | 1.0000 | Δ > 0 | does not hold (point estimate ≤ 0) |

P2 is the pre-registered ordering (the gain on T7-ROOD at least T7 v1's); P3 is its mechanism check (C0′ improves less after held-out names it never read). Δ values are differences of relative differences, in percentage points of the reference model's loss.
