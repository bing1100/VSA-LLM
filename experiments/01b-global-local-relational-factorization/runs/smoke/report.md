# Experiment 01b Stage A — synthetic teacher × learner matrix

Conditions: **24**. Maximum data budget: **32 edges/relation**.

## Mean held-out MSE at maximum budget

| Teacher | additive | diagonal | hrr | low_rank | Winner |
|---|---:|---:|---:|---:|---|
| diagonal | 0.00424 | 0.00001 | 0.00450 | 0.00252 | diagonal |
| hrr | 0.03410 | 0.02242 | 0.00002 | 0.02208 | hrr |
| low_rank | 0.08849 | 0.08563 | 0.09056 | 0.00186 | low_rank |

## Stage-A code gate

Aligned-family wins: **3/3**; gate: **PASS**.

This is an identifiability and implementation gate, not evidence for pretrained-model benefit. Inspect `metrics.csv` for data-to-threshold, generalization gaps, salience recovery, and residual strata.
