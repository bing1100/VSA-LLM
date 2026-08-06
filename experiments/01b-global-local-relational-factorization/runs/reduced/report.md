# Experiment 01b Stage A — synthetic teacher × learner matrix

Conditions: **216**. Maximum data budget: **512 edges/relation**.

## Mean held-out MSE at maximum budget

| Teacher | additive | diagonal | hrr | low_rank | Winner |
|---|---:|---:|---:|---:|---|
| diagonal | 0.00224 | 0.00016 | 0.00221 | 0.00111 | diagonal |
| hrr | 0.01741 | 0.01085 | 0.00014 | 0.00925 | hrr |
| low_rank | 0.04140 | 0.04029 | 0.04022 | 0.00022 | low_rank |

## Stage-A code gate

Aligned-family wins: **3/3**; gate: **PASS**.

This is an identifiability and implementation gate, not evidence for pretrained-model benefit. Inspect `metrics.csv` for data-to-threshold, generalization gaps, salience recovery, and residual strata.
