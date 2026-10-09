# E12 3c — calibrated self-critique — SMOKE

Pre-registration: `experiments/e12-self-query/preregistration.md` §13, amendments 16.3 and 16.5. K1 and K2: items × seeds crossed model of the per-item units `1(answered ∧ correct) / 0.8` and `1(answered ∧ adopted) / 0.8` (their means are the accuracy and the false-belief adoption at 80% coverage), Holm over the two.

**SMOKE: these numbers do not count toward the pre-registered endpoints.**

## SmolLM2-135M

Runs: {'C5': [1]}

### C5

θ per seed (dev half of the new words): {1: 0.0}

| endpoint | estimate |
|---|---|
| K1: rule loop − no tool (accuracy at 80% coverage) | +0.7143 [+0.3874, +1.0412] (p 0.00025, Holm 0.0005; 18 units × 1 seeds) |
| K2: rule loop − naive recall (false-belief adoption at 80% coverage, null world) | +0.0000 [+0.0000, +0.0000] (p 1, Holm 1; 18 units × 1 seeds) |

Reading: K1: the rule loop beats the no-tool answers at 80% coverage; K2: no difference shown: the loop accepts false structure as naive recall does; refutation: K2 ≈ 0 on held-out terms with evidence in context: the loop does not use evidence against its own store

**K1b (amendment 16.5)** — rule loop (store) − rule loop (the prose definition in the prompt), accuracy at 80% coverage, non-inferiority at −0.05 (one-sided α = 0.025; not in K1 / K2's Holm family): +0.0714 [-0.2734, +0.4162] (p 0.668; 18 units × 1 seeds). Reading: non-inferiority not shown (95% CI lower bound ≤ −0.05); the store's rule loop adds 99 fewer prompt tokens per item than the definition loop.

| loop (real world, pooled test items; seed means) | accuracy at 80% | prompt tokens added per item |
|---|---:|---:|
| no_tool | 0.286 | 0.0 |
| loop | 1.000 | 0.0 |
| recall_context | 1.000 | 158.3 |
| loop_recall_text | 1.000 | 158.3 |
| loop_symbolic | 1.000 | 159.0 |
| loop_definition | 0.929 | 98.9 |
| (every loop's flag: the evidence sentence, held-out items) | — | 7.8 |

| K1b secondary | estimate | reading |
|---|---|---|
| K1b (new) | +0.0000 [-0.6988, +0.6988] (p 1; 9 units × 1 seeds) | non-inferiority not shown (95% CI lower bound ≤ −0.05) |
| K1b (heldout) | +0.0000 [-0.4941, +0.4941] (p 1; 9 units × 1 seeds) | non-inferiority not shown (95% CI lower bound ≤ −0.05) |
| loop (store) − loop (gold relations as text) (pooled) | +0.0000 [-0.3798, +0.3798] (p 1; 18 units × 1 seeds) | non-inferiority not shown (95% CI lower bound ≤ −0.05) |
| loop (store) − loop (gold relations as text) (new) | +0.0000 [-0.6988, +0.6988] (p 1; 9 units × 1 seeds) | non-inferiority not shown (95% CI lower bound ≤ −0.05) |
| loop (store) − loop (gold relations as text) (heldout) | +0.0000 [-0.4941, +0.4941] (p 1; 9 units × 1 seeds) | non-inferiority not shown (95% CI lower bound ≤ −0.05) |
| loop (store decode as text, host reads) − loop (definition) (pooled) | +0.0714 [-0.1947, +0.3375] (p 0.579; 18 units × 1 seeds) | non-inferiority not shown (95% CI lower bound ≤ −0.05) |
| K1 of loop_definition: loop_definition − no tool (pooled) | +0.6429 [+0.2475, +1.0382] (p 0.00319; 18 units × 1 seeds) | differs (95% CI excludes 0) |
| K1 of loop_symbolic: loop_symbolic − no tool (pooled) | +0.7143 [+0.3206, +1.1079] (p 0.00135; 18 units × 1 seeds) | differs (95% CI excludes 0) |
| K1 of loop_recall_text: loop_recall_text − no tool (pooled) | +0.7143 [+0.3206, +1.1079] (p 0.00135; 18 units × 1 seeds) | differs (95% CI excludes 0) |

| pool / world / method | accuracy at 80% | adoption at 80% | accuracy (all) | adoption (all) | flag rate | prompt tokens added |
|---|---:|---:|---:|---:|---:|---:|
| pooled real no_tool | 0.286 | 0.071 | 0.222 | 0.056 | 0.000 | 0.0 |
| pooled real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0 |
| pooled real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.833 | 0.0 |
| pooled real loop_revise | 0.929 | 0.000 | 0.944 | 0.000 | 0.833 | 0.0 |
| pooled real recall_context | 1.000 | 0.000 | 0.944 | 0.000 | 0.000 | 158.3 |
| pooled real loop_recall_text | 1.000 | 0.000 | 0.944 | 0.000 | 0.833 | 158.3 |
| pooled real loop_symbolic | 1.000 | 0.000 | 0.944 | 0.000 | 0.833 | 159.0 |
| pooled real loop_definition | 0.929 | 0.000 | 0.944 | 0.000 | 0.833 | 98.9 |
| pooled null no_tool | 0.286 | 0.071 | 0.222 | 0.056 | 0.000 | 0.0 |
| pooled null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.0 |
| pooled null loop | 0.000 | 1.000 | 0.000 | 1.000 | 0.944 | 0.0 |
| pooled null loop_revise | 0.214 | 0.714 | 0.389 | 0.556 | 0.944 | 0.0 |
| pooled null recall_context | 0.071 | 0.929 | 0.056 | 0.833 | 0.000 | 161.6 |
| new real no_tool | 0.143 | 0.143 | 0.111 | 0.111 | 0.000 | 0.0 |
| new real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0 |
| new real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.889 | 0.0 |
| new real loop_revise | 1.000 | 0.000 | 1.000 | 0.000 | 0.889 | 0.0 |
| new real recall_context | 1.000 | 0.000 | 0.889 | 0.000 | 0.000 | 160.3 |
| new real loop_recall_text | 1.000 | 0.000 | 0.889 | 0.000 | 0.889 | 160.3 |
| new real loop_symbolic | 1.000 | 0.000 | 0.889 | 0.000 | 0.889 | 160.3 |
| new real loop_definition | 1.000 | 0.000 | 0.889 | 0.000 | 0.889 | 99.2 |
| new null no_tool | 0.143 | 0.143 | 0.111 | 0.111 | 0.000 | 0.0 |
| new null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.0 |
| new null loop | 0.000 | 1.000 | 0.000 | 1.000 | 0.889 | 0.0 |
| new null loop_revise | 0.000 | 1.000 | 0.000 | 1.000 | 0.889 | 0.0 |
| new null recall_context | 0.143 | 0.857 | 0.111 | 0.667 | 0.000 | 166.1 |
| heldout real no_tool | 0.286 | 0.000 | 0.333 | 0.000 | 0.000 | 0.0 |
| heldout real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0 |
| heldout real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.778 | 0.0 |
| heldout real loop_revise | 0.857 | 0.000 | 0.889 | 0.000 | 0.778 | 0.0 |
| heldout real recall_context | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 156.3 |
| heldout real loop_recall_text | 1.000 | 0.000 | 1.000 | 0.000 | 0.778 | 156.3 |
| heldout real loop_symbolic | 1.000 | 0.000 | 1.000 | 0.000 | 0.778 | 157.7 |
| heldout real loop_definition | 1.000 | 0.000 | 1.000 | 0.000 | 0.778 | 98.7 |
| heldout null no_tool | 0.286 | 0.000 | 0.333 | 0.000 | 0.000 | 0.0 |
| heldout null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.0 |
| heldout null loop | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 |
| heldout null loop_revise | 0.714 | 0.143 | 0.778 | 0.111 | 1.000 | 0.0 |
| heldout null recall_context | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 157.0 |
| twins real no_tool | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.0 |
| twins real naive | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0 |
| twins real loop | 1.000 | 0.000 | 1.000 | 0.000 | 0.500 | 0.0 |
| twins real loop_revise | 1.000 | 0.000 | 1.000 | 0.000 | 0.500 | 0.0 |
| twins real recall_context | 0.667 | 0.333 | 0.500 | 0.500 | 0.000 | 228.0 |
| twins real loop_recall_text | 0.667 | 0.333 | 0.500 | 0.500 | 0.750 | 228.0 |
| twins null no_tool | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 0.0 |
| twins null naive | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.0 |
| twins null loop | 0.000 | 1.000 | 0.000 | 1.000 | 0.500 | 0.0 |
| twins null loop_revise | 0.000 | 1.000 | 0.000 | 1.000 | 0.500 | 0.0 |
| twins null recall_context | 0.333 | 0.667 | 0.500 | 0.500 | 0.000 | 228.0 |

| calibration of p (seed-averaged) | n | accuracy | mean p | ECE | Brier | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| heldout null vs store | 23 | 0.957 | 0.901 | 0.056 | 0.067 | 0.909 |
| heldout null vs world | 23 | 0.000 | 0.901 | 0.901 | 0.876 | — |
| heldout real vs store | 23 | 0.957 | 0.935 | 0.073 | 0.059 | 0.955 |
| heldout real vs world | 23 | 0.957 | 0.935 | 0.073 | 0.059 | 0.955 |
| new null vs store | 31 | 0.968 | 0.980 | 0.014 | 0.005 | 1.000 |
| new null vs world | 31 | 0.032 | 0.980 | 0.948 | 0.939 | 0.533 |
| new real vs store | 31 | 1.000 | 0.987 | 0.013 | 0.004 | — |
| new real vs world | 31 | 1.000 | 0.987 | 0.013 | 0.004 | — |
| seen entries (the fit set, in-sample) | 20670 | 0.986 | 0.985 | 0.001 | 0.008 | 0.992 |
| twins null vs store | 38 | 0.974 | 0.974 | 0.021 | 0.008 | 1.000 |
| twins null vs world | 38 | 0.763 | 0.974 | 0.230 | 0.217 | 0.580 |
| twins real vs store | 38 | 0.974 | 0.974 | 0.021 | 0.008 | 1.000 |
| twins real vs world | 38 | 0.974 | 0.974 | 0.021 | 0.008 | 1.000 |

| secondary | estimate |
|---|---|
| K1 (new) | +0.8571 [+0.3630, +1.3513] (p 0.00395; 9 units × 1 seeds) |
| K2 (new) | +0.0000 [-0.4941, +0.4941] (p 1; 9 units × 1 seeds) |
| K1 naive − no tool (new) | +0.8571 [+0.3630, +1.3513] (p 0.00395; 9 units × 1 seeds) |
| K1 loop+evidence revision − no tool (new) | +0.8571 [+0.3630, +1.3513] (p 0.00395; 9 units × 1 seeds) |
| K2 loop+evidence revision − naive (new) | +0.0000 [-0.4941, +0.4941] (p 1; 9 units × 1 seeds) |
| K2 recall in context (host reads the false recall) − naive (new) | -0.1429 [-0.9154, +0.6297] (p 0.681; 9 units × 1 seeds) |
| K1 (heldout) | +0.7143 [+0.1934, +1.2352] (p 0.0133; 9 units × 1 seeds) |
| K2 (heldout) | +0.0000 [+0.0000, +0.0000] (p 1; 9 units × 1 seeds) |
| K1 naive − no tool (heldout) | +0.7143 [+0.1934, +1.2352] (p 0.0133; 9 units × 1 seeds) |
| K1 loop+evidence revision − no tool (heldout) | +0.5714 [+0.0506, +1.0923] (p 0.0353; 9 units × 1 seeds) |
| K2 loop+evidence revision − naive (heldout) | -0.8571 [-1.3513, -0.3630] (p 0.00395; 9 units × 1 seeds) |
| K2 recall in context (host reads the false recall) − naive (heldout) | +0.0000 [-0.6988, +0.6988] (p 1; 9 units × 1 seeds) |
| K1 (twins) | +0.5000 [-0.0769, +1.0769] (p 0.0796; 8 units × 1 seeds) |
| K2 (twins) | +0.0000 [-0.5958, +0.5958] (p 1; 8 units × 1 seeds) |
| K1 naive − no tool (twins) | +0.5000 [-0.5212, +1.5212] (p 0.285; 8 units × 1 seeds) |
| K1 loop+evidence revision − no tool (twins) | +0.5000 [-0.0769, +1.0769] (p 0.0796; 8 units × 1 seeds) |
| K2 loop+evidence revision − naive (twins) | +0.0000 [-0.5958, +0.5958] (p 1; 8 units × 1 seeds) |
| K2 recall in context (host reads the false recall) − naive (twins) | -0.3333 [-1.1215, +0.4549] (p 0.351; 8 units × 1 seeds) |

Coverage–accuracy (pooled test items, first seed; real world): coverage → no tool / naive / loop

| coverage | no tool | naive | loop | loop (null world): adoption |
|---:|---:|---:|---:|---:|
| 0.20 | 0.500 | 1.000 | 1.000 | 1.000 |
| 0.25 | 0.500 | 1.000 | 1.000 | 1.000 |
| 0.30 | 0.400 | 1.000 | 1.000 | 1.000 |
| 0.35 | 0.333 | 1.000 | 1.000 | 1.000 |
| 0.40 | 0.286 | 1.000 | 1.000 | 1.000 |
| 0.45 | 0.375 | 1.000 | 1.000 | 1.000 |
| 0.50 | 0.333 | 1.000 | 1.000 | 1.000 |
| 0.55 | 0.300 | 1.000 | 1.000 | 1.000 |
| 0.60 | 0.273 | 1.000 | 1.000 | 1.000 |
| 0.65 | 0.250 | 1.000 | 1.000 | 1.000 |
| 0.70 | 0.308 | 1.000 | 1.000 | 1.000 |
| 0.75 | 0.286 | 1.000 | 1.000 | 1.000 |
| 0.80 | 0.286 | 1.000 | 1.000 | 1.000 |
| 0.85 | 0.267 | 1.000 | 1.000 | 1.000 |
| 0.90 | 0.250 | 1.000 | 1.000 | 1.000 |
| 0.95 | 0.235 | 1.000 | 1.000 | 1.000 |
| 1.00 | 0.222 | 1.000 | 1.000 | 1.000 |

## Model loop (secondary)

- Qwen3-0.6B-Base C5: K1 (model loop − no tool): +0.0000 [-0.7965, +0.7965] (p 1, Holm 1; 6 units × 1 seeds)
- Qwen3-0.6B-Base C5: K2 (model loop − naive recall): -0.8000 [-1.4503, -0.1497] (p 0.025, Holm 0.0501; 6 units × 1 seeds)

