# E9 in-context frame baseline, 4,096-token glossary

In-context frame baseline (`e9_incontext`; addendum 2026-10-11 §3): each evaluation window scored with a glossary of the verbalized frames of its linked entries in front (`frame`), the same number of other entries' lines (`other`) or nothing (`none`); only the window's own targets count, with the trainer's strata. Rows are `candidate − reference`, relative to the reference loss, pooled over common seeds per window, 95% cluster bootstrap over windows, Holm over the comparisons within a stratum (* = Holm p < 0.05). Negative = the candidate predicts better.

## t4 · SmolLM2-360M

Models: C0' s1, C0' s2, C0' s3, C5 s1, C5 s2, C5 s3, P0 s1

Glossary cost: `frame` 2025 tokens per window on average (max 4096), entry coverage 98.8%, held-out entry coverage 98.8%; `other` 2258 tokens.

### `after_heldout`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -0.25% [-0.54%, +0.02%] | -0.0037 | 0.145 | 107,241 |
| in-context, P0 (P0 frame − P0) | -1.65% [-2.02%, -1.29%]* | -0.0395 | 0.0016 | 35,747 |
| channel (C5 − C0′) | -0.39% [-0.53%, -0.24%]* | -0.0055 | 0.0016 | 107,241 |
| channel given the glossary (C5 frame − C0′ frame) | -0.36% [-0.50%, -0.22%]* | -0.0051 | 0.0016 | 107,241 |
| glossary given the channel (C5 frame − C5) | -0.23% [-0.51%, +0.05%] | -0.0032 | 0.145 | 107,241 |
| both vs neither (C5 frame − C0′) | -0.61% [-0.93%, -0.30%]* | -0.0088 | 0.0016 | 107,241 |
| glossary specificity, C0′ (frame − other) | -0.95% [-1.20%, -0.70%]* | -0.0137 | 0.0016 | 107,241 |
| glossary specificity, C5 (frame − other) | -0.94% [-1.20%, -0.70%]* | -0.0136 | 0.0016 | 107,241 |

### `after_unseen`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -0.20% [-0.57%, +0.09%] | -0.0031 | 0.417 | 29,376 |
| in-context, P0 (P0 frame − P0) | -0.88% [-1.14%, -0.65%]* | -0.0309 | 0.0016 | 9,792 |
| channel (C5 − C0′) | -2.29% [-2.72%, -1.90%]* | -0.0355 | 0.0016 | 29,376 |
| channel given the glossary (C5 frame − C0′ frame) | -2.26% [-2.69%, -1.88%]* | -0.0349 | 0.0016 | 29,376 |
| glossary given the channel (C5 frame − C5) | -0.17% [-0.54%, +0.13%] | -0.0026 | 0.417 | 29,376 |
| both vs neither (C5 frame − C0′) | -2.46% [-3.01%, -1.96%]* | -0.0381 | 0.0016 | 29,376 |
| glossary specificity, C0′ (frame − other) | -0.50% [-0.85%, -0.28%]* | -0.0078 | 0.0016 | 29,376 |
| glossary specificity, C5 (frame − other) | -0.51% [-0.87%, -0.29%]* | -0.0077 | 0.0016 | 29,376 |

### `after_rare_seen`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | +0.18% [-0.05%, +0.41%] | +0.0029 | 0.115 | 94,374 |
| in-context, P0 (P0 frame − P0) | -1.51% [-1.76%, -1.28%]* | -0.0499 | 0.0016 | 31,458 |
| channel (C5 − C0′) | -0.91% [-1.06%, -0.77%]* | -0.0143 | 0.0016 | 94,374 |
| channel given the glossary (C5 frame − C0′ frame) | -0.83% [-0.98%, -0.69%]* | -0.0131 | 0.0016 | 94,374 |
| glossary given the channel (C5 frame − C5) | +0.26% [+0.03%, +0.48%] | +0.0040 | 0.058 | 94,374 |
| both vs neither (C5 frame − C0′) | -0.65% [-0.92%, -0.39%]* | -0.0103 | 0.0016 | 94,374 |
| glossary specificity, C0′ (frame − other) | -0.55% [-0.76%, -0.36%]* | -0.0087 | 0.0016 | 94,374 |
| glossary specificity, C5 (frame − other) | -0.50% [-0.70%, -0.31%]* | -0.0079 | 0.0016 | 94,374 |

### `after_len3plus`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -0.14% [-0.29%, +0.00%] | -0.0020 | 0.108 | 555,327 |
| in-context, P0 (P0 frame − P0) | -1.53% [-1.74%, -1.34%]* | -0.0410 | 0.0016 | 185,109 |
| channel (C5 − C0′) | -1.08% [-1.16%, -1.01%]* | -0.0161 | 0.0016 | 555,327 |
| channel given the glossary (C5 frame − C0′ frame) | -1.01% [-1.08%, -0.94%]* | -0.0149 | 0.0016 | 555,327 |
| glossary given the channel (C5 frame − C5) | -0.06% [-0.21%, +0.08%] | -0.0009 | 0.415 | 555,327 |
| both vs neither (C5 frame − C0′) | -1.14% [-1.31%, -0.99%]* | -0.0170 | 0.0016 | 555,327 |
| glossary specificity, C0′ (frame − other) | -0.83% [-0.99%, -0.69%]* | -0.0124 | 0.0016 | 555,327 |
| glossary specificity, C5 (frame − other) | -0.78% [-0.94%, -0.65%]* | -0.0116 | 0.0016 | 555,327 |

### `inside`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -3.52% [-4.05%, -3.04%]* | -0.0316 | 0.0016 | 391,983 |
| in-context, P0 (P0 frame − P0) | -4.55% [-5.05%, -4.08%]* | -0.0744 | 0.0016 | 130,661 |
| channel (C5 − C0′) | -1.53% [-1.66%, -1.41%]* | -0.0137 | 0.0016 | 391,983 |
| channel given the glossary (C5 frame − C0′ frame) | -1.48% [-1.61%, -1.36%]* | -0.0128 | 0.0016 | 391,983 |
| glossary given the channel (C5 frame − C5) | -3.47% [-4.00%, -2.99%]* | -0.0306 | 0.0016 | 391,983 |
| both vs neither (C5 frame − C0′) | -4.94% [-5.48%, -4.46%]* | -0.0444 | 0.0016 | 391,983 |
| glossary specificity, C0′ (frame − other) | -3.76% [-4.28%, -3.28%]* | -0.0338 | 0.0016 | 391,983 |
| glossary specificity, C5 (frame − other) | -3.71% [-4.24%, -3.24%]* | -0.0329 | 0.0016 | 391,983 |

### `unlinked`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | +0.66% [+0.61%, +0.70%]* | +0.0138 | 0.0016 | 2,219,340 |
| in-context, P0 (P0 frame − P0) | +0.14% [+0.10%, +0.19%]* | +0.0034 | 0.0016 | 739,780 |
| channel (C5 − C0′) | -0.01% [-0.02%, -0.01%]* | -0.0003 | 0.0016 | 2,219,340 |
| channel given the glossary (C5 frame − C0′ frame) | -0.01% [-0.02%, -0.01%]* | -0.0002 | 0.0016 | 2,219,340 |
| glossary given the channel (C5 frame − C5) | +0.66% [+0.62%, +0.70%]* | +0.0138 | 0.0016 | 2,219,340 |
| both vs neither (C5 frame − C0′) | +0.64% [+0.60%, +0.69%]* | +0.0136 | 0.0016 | 2,219,340 |
| glossary specificity, C0′ (frame − other) | -0.04% [-0.06%, -0.03%]* | -0.0009 | 0.0016 | 2,219,340 |
| glossary specificity, C5 (frame − other) | -0.04% [-0.06%, -0.02%]* | -0.0008 | 0.0016 | 2,219,340 |

### `all`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | +0.39% [+0.34%, +0.44%]* | +0.0074 | 0.0016 | 3,142,656 |
| in-context, P0 (P0 frame − P0) | -0.45% [-0.54%, -0.37%]* | -0.0106 | 0.0016 | 1,047,552 |
| channel (C5 − C0′) | -0.18% [-0.20%, -0.17%]* | -0.0034 | 0.0016 | 3,142,656 |
| channel given the glossary (C5 frame − C0′ frame) | -0.17% [-0.18%, -0.16%]* | -0.0032 | 0.0016 | 3,142,656 |
| glossary given the channel (C5 frame − C5) | +0.41% [+0.36%, +0.46%]* | +0.0076 | 0.0016 | 3,142,656 |
| both vs neither (C5 frame − C0′) | +0.22% [+0.17%, +0.28%]* | +0.0042 | 0.0016 | 3,142,656 |
| glossary specificity, C0′ (frame − other) | -0.29% [-0.33%, -0.25%]* | -0.0055 | 0.0016 | 3,142,656 |
| glossary specificity, C5 (frame − other) | -0.28% [-0.32%, -0.24%]* | -0.0053 | 0.0016 | 3,142,656 |

`none` replays the final evaluations: max |window sum − eval_windows.npz| 5.96; counts equal in 7 of 7 runs.

## t5 · SmolLM2-360M

Models: C0' s1, C0' s2, C0' s3, C5 s1, C5 s2, C5 s3, P0 s1

Glossary cost: `frame` 1231 tokens per window on average (max 2210), entry coverage 100.0%, held-out entry coverage 100.0%; `other` 1443 tokens.

### `after_heldout`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -9.89% [-10.81%, -8.94%]* | -0.0858 | 0.0016 | 150,369 |
| in-context, P0 (P0 frame − P0) | -3.28% [-3.83%, -2.77%]* | -0.1099 | 0.0016 | 50,123 |
| channel (C5 − C0′) | -13.25% [-14.11%, -12.43%]* | -0.1150 | 0.0016 | 150,369 |
| channel given the glossary (C5 frame − C0′ frame) | -10.93% [-11.76%, -10.14%]* | -0.0855 | 0.0016 | 150,369 |
| glossary given the channel (C5 frame − C5) | -7.48% [-8.44%, -6.55%]* | -0.0563 | 0.0016 | 150,369 |
| both vs neither (C5 frame − C0′) | -19.74% [-20.80%, -18.71%]* | -0.1714 | 0.0016 | 150,369 |
| glossary specificity, C0′ (frame − other) | -9.91% [-10.84%, -8.98%]* | -0.0860 | 0.0016 | 150,369 |
| glossary specificity, C5 (frame − other) | -7.56% [-8.51%, -6.63%]* | -0.0570 | 0.0016 | 150,369 |

### `after_unseen`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -8.28% [-11.93%, -5.07%]* | -0.0729 | 0.0016 | 14,721 |
| in-context, P0 (P0 frame − P0) | -4.26% [-6.71%, -2.12%]* | -0.1463 | 0.0016 | 4,907 |
| channel (C5 − C0′) | -6.53% [-8.12%, -5.09%]* | -0.0575 | 0.0016 | 14,721 |
| channel given the glossary (C5 frame − C0′ frame) | -6.24% [-7.87%, -4.73%]* | -0.0504 | 0.0016 | 14,721 |
| glossary given the channel (C5 frame − C5) | -7.99% [-11.74%, -4.70%]* | -0.0658 | 0.0016 | 14,721 |
| both vs neither (C5 frame − C0′) | -14.00% [-17.57%, -10.76%]* | -0.1233 | 0.0016 | 14,721 |
| glossary specificity, C0′ (frame − other) | -8.31% [-11.90%, -5.24%]* | -0.0732 | 0.0016 | 14,721 |
| glossary specificity, C5 (frame − other) | -8.23% [-11.88%, -5.04%]* | -0.0679 | 0.0016 | 14,721 |

### `after_rare_seen`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -8.28% [-9.65%, -6.98%]* | -0.0690 | 0.0016 | 108,909 |
| in-context, P0 (P0 frame − P0) | -3.36% [-4.10%, -2.70%]* | -0.1120 | 0.0016 | 36,303 |
| channel (C5 − C0′) | -6.02% [-6.63%, -5.43%]* | -0.0502 | 0.0016 | 108,909 |
| channel given the glossary (C5 frame − C0′ frame) | -4.82% [-5.47%, -4.15%]* | -0.0368 | 0.0016 | 108,909 |
| glossary given the channel (C5 frame − C5) | -7.10% [-8.50%, -5.80%]* | -0.0556 | 0.0016 | 108,909 |
| both vs neither (C5 frame − C0′) | -12.70% [-13.96%, -11.48%]* | -0.1058 | 0.0016 | 108,909 |
| glossary specificity, C0′ (frame − other) | -8.61% [-9.93%, -7.39%]* | -0.0719 | 0.0016 | 108,909 |
| glossary specificity, C5 (frame − other) | -7.80% [-9.13%, -6.59%]* | -0.0615 | 0.0016 | 108,909 |

### `after_len3plus`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -6.18% [-6.56%, -5.79%]* | -0.0370 | 0.0016 | 1,447,566 |
| in-context, P0 (P0 frame − P0) | -3.32% [-3.52%, -3.13%]* | -0.1058 | 0.0016 | 482,522 |
| channel (C5 − C0′) | -7.56% [-7.79%, -7.32%]* | -0.0453 | 0.0016 | 1,447,566 |
| channel given the glossary (C5 frame − C0′ frame) | -6.49% [-6.74%, -6.25%]* | -0.0365 | 0.0016 | 1,447,566 |
| glossary given the channel (C5 frame − C5) | -5.10% [-5.50%, -4.71%]* | -0.0283 | 0.0016 | 1,447,566 |
| both vs neither (C5 frame − C0′) | -12.27% [-12.69%, -11.85%]* | -0.0735 | 0.0016 | 1,447,566 |
| glossary specificity, C0′ (frame − other) | -7.22% [-7.59%, -6.86%]* | -0.0438 | 0.0016 | 1,447,566 |
| glossary specificity, C5 (frame − other) | -6.43% [-6.80%, -6.07%]* | -0.0361 | 0.0016 | 1,447,566 |

### `inside`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -19.60% [-20.29%, -18.91%]* | -0.2005 | 0.0016 | 859,584 |
| in-context, P0 (P0 frame − P0) | -10.55% [-10.99%, -10.12%]* | -0.3562 | 0.0016 | 286,528 |
| channel (C5 − C0′) | +0.05% [-0.04%, +0.14%] | +0.0005 | 0.315 | 859,584 |
| channel given the glossary (C5 frame − C0′ frame) | +1.03% [+0.88%, +1.18%]* | +0.0085 | 0.0016 | 859,584 |
| glossary given the channel (C5 frame − C5) | -18.81% [-19.50%, -18.13%]* | -0.1926 | 0.0016 | 859,584 |
| both vs neither (C5 frame − C0′) | -18.77% [-19.46%, -18.09%]* | -0.1921 | 0.0016 | 859,584 |
| glossary specificity, C0′ (frame − other) | -18.44% [-19.14%, -17.76%]* | -0.1861 | 0.0016 | 859,584 |
| glossary specificity, C5 (frame − other) | -17.88% [-18.56%, -17.21%]* | -0.1810 | 0.0016 | 859,584 |

### `unlinked`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | +0.61% [+0.31%, +0.91%]* | +0.0031 | 0.0016 | 1,142,448 |
| in-context, P0 (P0 frame − P0) | -0.55% [-0.62%, -0.49%]* | -0.0199 | 0.0016 | 380,816 |
| channel (C5 − C0′) | -0.02% [-0.08%, +0.03%] | -0.0001 | 0.898 | 1,142,448 |
| channel given the glossary (C5 frame − C0′ frame) | -0.00% [-0.07%, +0.06%] | -0.0000 | 0.949 | 1,142,448 |
| glossary given the channel (C5 frame − C5) | +0.63% [+0.34%, +0.92%]* | +0.0032 | 0.0016 | 1,142,448 |
| both vs neither (C5 frame − C0′) | +0.61% [+0.32%, +0.90%]* | +0.0031 | 0.0016 | 1,142,448 |
| glossary specificity, C0′ (frame − other) | -1.60% [-1.76%, -1.44%]* | -0.0084 | 0.0016 | 1,142,448 |
| glossary specificity, C5 (frame − other) | -1.60% [-1.76%, -1.44%]* | -0.0084 | 0.0016 | 1,142,448 |

### `all`

| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |
|---|---|---|---|---:|
| in-context, C0′ (C0′ frame − C0′) | -8.34% [-8.67%, -8.00%]* | -0.0549 | 0.0016 | 3,142,656 |
| in-context, P0 (P0 frame − P0) | -3.65% [-3.80%, -3.49%]* | -0.1250 | 0.0016 | 1,047,552 |
| channel (C5 − C0′) | -3.19% [-3.30%, -3.09%]* | -0.0210 | 0.0016 | 3,142,656 |
| channel given the glossary (C5 frame − C0′ frame) | -2.62% [-2.73%, -2.51%]* | -0.0158 | 0.0016 | 3,142,656 |
| glossary given the channel (C5 frame − C5) | -7.79% [-8.13%, -7.46%]* | -0.0497 | 0.0016 | 3,142,656 |
| both vs neither (C5 frame − C0′) | -10.74% [-11.07%, -10.41%]* | -0.0707 | 0.0016 | 3,142,656 |
| glossary specificity, C0′ (frame − other) | -8.88% [-9.20%, -8.57%]* | -0.0589 | 0.0016 | 3,142,656 |
| glossary specificity, C5 (frame − other) | -8.53% [-8.85%, -8.22%]* | -0.0549 | 0.0016 | 3,142,656 |

`none` replays the final evaluations: max |window sum − eval_windows.npz| 5.99; counts equal in 7 of 7 runs.
