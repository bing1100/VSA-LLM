# E12 3c — calibrated self-critique — SMOKE

Pre-registration: `experiments/e12-self-query/preregistration.md` §13 and amendment 16.3. K1 and K2: items × seeds crossed model of the per-item units `1(answered ∧ correct) / 0.8` and `1(answered ∧ adopted) / 0.8` (their means are the accuracy and the false-belief adoption at 80% coverage), Holm over the two.

**SMOKE: these numbers do not count toward the pre-registered endpoints.**

## SmolLM2-135M

Runs: {'C5': [1]}

### C5

θ per seed (dev half of the new words): {1: 0.0}

| endpoint | estimate |
|---|---|
| K1: rule loop − no tool (accuracy at 80% coverage) | +0.6552 [+0.4425, +0.8678] (p 3.57e-07, Holm 7.15e-07; 36 units × 1 seeds) |
| K2: rule loop − naive recall (false-belief adoption at 80% coverage, null world) | +0.0000 [+0.0000, +0.0000] (p 1, Holm 1; 36 units × 1 seeds) |

Reading: K1: the rule loop beats the no-tool answers at 80% coverage; K2: no difference shown: the loop accepts false structure as naive recall does; refutation: K2 ≈ 0 on held-out terms with evidence in context: the loop does not use evidence against its own store

| pool / world / method | accuracy at 80% | adoption at 80% | accuracy (all) | adoption (all) | flag rate |
|---|---:|---:|---:|---:|---:|
| pooled real no_tool | 0.345 | 0.069 | 0.306 | 0.056 | 0.000 |
| pooled real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| pooled real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.722 |
| pooled real loop_revise | 0.966 | 0.000 | 0.972 | 0.000 | 0.722 |
| pooled real recall_context | 0.931 | 0.000 | 0.889 | 0.000 | 0.000 |
| pooled null no_tool | 0.345 | 0.069 | 0.306 | 0.056 | 0.000 |
| pooled null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| pooled null loop | 0.000 | 1.000 | 0.000 | 1.000 | 0.972 |
| pooled null loop_revise | 0.241 | 0.724 | 0.361 | 0.611 | 0.972 |
| pooled null recall_context | 0.069 | 0.828 | 0.056 | 0.833 | 0.000 |
| new real no_tool | 0.375 | 0.062 | 0.350 | 0.050 | 0.000 |
| new real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| new real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.650 |
| new real loop_revise | 1.000 | 0.000 | 1.000 | 0.000 | 0.650 |
| new real recall_context | 0.875 | 0.000 | 0.850 | 0.000 | 0.000 |
| new null no_tool | 0.375 | 0.062 | 0.350 | 0.050 | 0.000 |
| new null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| new null loop | 0.000 | 1.000 | 0.000 | 1.000 | 0.950 |
| new null loop_revise | 0.000 | 1.000 | 0.000 | 1.000 | 0.950 |
| new null recall_context | 0.062 | 0.812 | 0.050 | 0.800 | 0.000 |
| heldout real no_tool | 0.231 | 0.077 | 0.250 | 0.062 | 0.000 |
| heldout real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| heldout real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.812 |
| heldout real loop_revise | 0.923 | 0.000 | 0.938 | 0.000 | 0.812 |
| heldout real recall_context | 1.000 | 0.000 | 0.938 | 0.000 | 0.000 |
| heldout null no_tool | 0.231 | 0.077 | 0.250 | 0.062 | 0.000 |
| heldout null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| heldout null loop | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| heldout null loop_revise | 0.769 | 0.154 | 0.812 | 0.125 | 1.000 |
| heldout null recall_context | 0.077 | 0.846 | 0.062 | 0.875 | 0.000 |
| twins real no_tool | 0.462 | 0.538 | 0.562 | 0.438 | 0.000 |
| twins real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| twins real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.438 |
| twins real loop_revise | 1.000 | 0.000 | 1.000 | 0.000 | 0.438 |
| twins real recall_context | 0.615 | 0.385 | 0.500 | 0.500 | 0.000 |
| twins null no_tool | 0.462 | 0.538 | 0.562 | 0.438 | 0.000 |
| twins null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| twins null loop | 0.000 | 1.000 | 0.000 | 1.000 | 0.562 |
| twins null loop_revise | 0.000 | 1.000 | 0.000 | 1.000 | 0.562 |
| twins null recall_context | 0.385 | 0.615 | 0.500 | 0.500 | 0.000 |

| calibration of p (seed-averaged) | n | accuracy | mean p | ECE | Brier | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| heldout null vs store | 45 | 0.978 | 0.935 | 0.042 | 0.043 | 0.932 |
| heldout null vs world | 45 | 0.022 | 0.935 | 0.913 | 0.895 | 0.557 |
| heldout real vs store | 45 | 0.978 | 0.962 | 0.042 | 0.031 | 0.977 |
| heldout real vs world | 45 | 0.978 | 0.962 | 0.042 | 0.031 | 0.977 |
| new null vs store | 63 | 0.984 | 0.989 | 0.008 | 0.003 | 1.000 |
| new null vs world | 63 | 0.048 | 0.989 | 0.941 | 0.936 | 0.533 |
| new real vs store | 63 | 1.000 | 0.990 | 0.010 | 0.002 | — |
| new real vs world | 63 | 1.000 | 0.990 | 0.010 | 0.002 | — |
| seen entries (the fit set, in-sample) | 20670 | 0.986 | 0.985 | 0.001 | 0.008 | 0.992 |
| twins null vs store | 74 | 0.986 | 0.986 | 0.011 | 0.004 | 1.000 |
| twins null vs world | 74 | 0.770 | 0.986 | 0.226 | 0.220 | 0.526 |
| twins real vs store | 74 | 0.986 | 0.986 | 0.011 | 0.004 | 1.000 |
| twins real vs world | 74 | 0.986 | 0.986 | 0.011 | 0.004 | 1.000 |

| secondary | estimate |
|---|---|
| K1 (new) | +0.6250 [+0.3249, +0.9251] (p 0.000338; 20 units × 1 seeds) |
| K2 (new) | +0.0000 [+0.0000, +0.0000] (p 1; 20 units × 1 seeds) |
| K1 naive − no tool (new) | +0.6250 [+0.2699, +0.9801] (p 0.00158; 20 units × 1 seeds) |
| K1 loop+evidence revision − no tool (new) | +0.6250 [+0.3249, +0.9251] (p 0.000338; 20 units × 1 seeds) |
| K2 loop+evidence revision − naive (new) | +0.0000 [+0.0000, +0.0000] (p 1; 20 units × 1 seeds) |
| K2 recall in context (host reads the false recall) − naive (new) | -0.1875 [-0.6234, +0.2484] (p 0.379; 20 units × 1 seeds) |
| K1 (heldout) | +0.7692 [+0.4413, +1.0971] (p 0.000158; 16 units × 1 seeds) |
| K2 (heldout) | +0.0000 [+0.0000, +0.0000] (p 1; 16 units × 1 seeds) |
| K1 naive − no tool (heldout) | +0.7692 [+0.3632, +1.1753] (p 0.00107; 16 units × 1 seeds) |
| K1 loop+evidence revision − no tool (heldout) | +0.6923 [+0.3563, +1.0283] (p 0.000526; 16 units × 1 seeds) |
| K2 loop+evidence revision − naive (heldout) | -0.8462 [-1.1601, -0.5322] (p 3.88e-05; 16 units × 1 seeds) |
| K2 recall in context (host reads the false recall) − naive (heldout) | -0.1538 [-0.4818, +0.1741] (p 0.333; 16 units × 1 seeds) |
| K1 (twins) | +0.5385 [+0.2024, +0.8745] (p 0.00383; 16 units × 1 seeds) |
| K2 (twins) | +0.0000 [+0.0000, +0.0000] (p 1; 16 units × 1 seeds) |
| K1 naive − no tool (twins) | +0.5385 [+0.0047, +1.0723] (p 0.0483; 16 units × 1 seeds) |
| K1 loop+evidence revision − no tool (twins) | +0.5385 [+0.2024, +0.8745] (p 0.00383; 16 units × 1 seeds) |
| K2 loop+evidence revision − naive (twins) | +0.0000 [+0.0000, +0.0000] (p 1; 16 units × 1 seeds) |
| K2 recall in context (host reads the false recall) − naive (twins) | -0.3846 [-0.7795, +0.0102] (p 0.0555; 16 units × 1 seeds) |

Coverage–accuracy (pooled test items, first seed; real world): coverage → no tool / naive / loop

| coverage | no tool | naive | loop | loop (null world): adoption |
|---:|---:|---:|---:|---:|
| 0.20 | 0.571 | 1.000 | 1.000 | 1.000 |
| 0.25 | 0.444 | 1.000 | 1.000 | 1.000 |
| 0.30 | 0.364 | 1.000 | 1.000 | 1.000 |
| 0.35 | 0.308 | 1.000 | 1.000 | 1.000 |
| 0.40 | 0.286 | 1.000 | 1.000 | 1.000 |
| 0.45 | 0.250 | 1.000 | 1.000 | 1.000 |
| 0.50 | 0.333 | 1.000 | 1.000 | 1.000 |
| 0.55 | 0.300 | 1.000 | 1.000 | 1.000 |
| 0.60 | 0.364 | 1.000 | 1.000 | 1.000 |
| 0.65 | 0.348 | 1.000 | 1.000 | 1.000 |
| 0.70 | 0.360 | 1.000 | 1.000 | 1.000 |
| 0.75 | 0.370 | 1.000 | 1.000 | 1.000 |
| 0.80 | 0.345 | 1.000 | 1.000 | 1.000 |
| 0.85 | 0.323 | 1.000 | 1.000 | 1.000 |
| 0.90 | 0.312 | 1.000 | 1.000 | 1.000 |
| 0.95 | 0.324 | 1.000 | 1.000 | 1.000 |
| 1.00 | 0.306 | 1.000 | 1.000 | 1.000 |

## Model loop (secondary)

- Qwen3-0.6B-Base C5: K1 (model loop − no tool): +0.0000 [-0.7965, +0.7965] (p 1, Holm 1; 6 units × 1 seeds)
- Qwen3-0.6B-Base C5: K2 (model loop − naive recall): -0.8000 [-1.4503, -0.1497] (p 0.025, Holm 0.0501; 6 units × 1 seeds)

