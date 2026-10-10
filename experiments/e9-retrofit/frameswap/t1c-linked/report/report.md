# E9 frame swap — is the own frame used?

Evaluation-time frame swap (`e9_frameswap`; pre-registration `experiments/e9-retrofit/preregistration-frameswap.md`). Each row is `variant − own` within the same trained model on its evaluation windows (positive = the variant is worse, i.e. the run's own rows help): Δ in nats per target token and relative to `own`'s loss, token-weighted and pooled over seeds per window, with 95% cluster-bootstrap intervals over windows; Holm over the four variants within a target set × stratum (* = Holm p < 0.05). Variants: `other` = another target entry's frame (derangement within the set), `other-any` = a random non-target entry's frame, `empty` = no injection, `mean` = the mean non-target row.

**Reading rule:** a held-out gain counts as ontology-specific only if `other − own` > 0 on `after_heldout` with Holm p < 0.05.

## t1c · SmolLM2-135M · full · C5

Seeds: 1 (single seed: intervals cover windows only).

### Target set `linked` (matched stratum `after_len3plus`)

5056 target entries (0 dropped: empty frame); windows holding a target span per seed: 2047.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_len3plus` | other | +0.1583 [+0.1524, +0.1643] | +7.72% [+7.36%, +8.10%] | 0.0006* | 265,185 |
| `after_len3plus` | empty | +0.1304 [+0.1254, +0.1354] | +6.36% [+6.05%, +6.67%] | 0.0006* | 265,185 |
| `after_len3plus` | mean | +0.1284 [+0.1235, +0.1334] | +6.26% [+5.96%, +6.57%] | 0.0006* | 265,185 |
| `after` | other | +0.1582 [+0.1527, +0.1637] | +7.48% [+7.16%, +7.82%] | 0.0006* | 457,057 |
| `after` | empty | +0.1345 [+0.1297, +0.1394] | +6.36% [+6.08%, +6.65%] | 0.0006* | 457,057 |
| `after` | mean | +0.1323 [+0.1276, +0.1371] | +6.26% [+5.98%, +6.54%] | 0.0006* | 457,057 |
| `unlinked` | other | +0.0132 [+0.0126, +0.0139] | +0.63% [+0.60%, +0.67%] | 0.0006* | 1,549,478 |
| `unlinked` | empty | +0.0116 [+0.0110, +0.0122] | +0.55% [+0.52%, +0.59%] | 0.0006* | 1,549,478 |
| `unlinked` | mean | +0.0114 [+0.0109, +0.0120] | +0.55% [+0.52%, +0.58%] | 0.0006* | 1,549,478 |
| `all` | other | +0.0446 [+0.0430, +0.0463] | +2.19% [+2.09%, +2.29%] | 0.0006* | 2,095,104 |
| `all` | empty | +0.0381 [+0.0368, +0.0396] | +1.87% [+1.79%, +1.96%] | 0.0006* | 2,095,104 |
| `all` | mean | +0.0375 [+0.0362, +0.0389] | +1.84% [+1.76%, +1.93%] | 0.0006* | 2,095,104 |
| `inside` | other | +0.0499 [+0.0472, +0.0527] | +6.26% [+5.88%, +6.66%] | 0.0006* | 126,863 |
| `inside` | empty | +0.0365 [+0.0340, +0.0389] | +4.57% [+4.25%, +4.90%] | 0.0006* | 126,863 |
| `inside` | mean | +0.0357 [+0.0333, +0.0380] | +4.48% [+4.16%, +4.80%] | 0.0006* | 126,863 |
| `after_len2` | other | +0.1798 [+0.1727, +0.1871] | +8.24% [+7.84%, +8.66%] | 0.0006* | 222,537 |
| `after_len2` | empty | +0.1571 [+0.1507, +0.1636] | +7.20% [+6.84%, +7.57%] | 0.0006* | 222,537 |
| `after_len2` | mean | +0.1535 [+0.1472, +0.1599] | +7.03% [+6.68%, +7.39%] | 0.0006* | 222,537 |
| `after_heldout` | other | +0.1099 [+0.1013, +0.1187] | +4.71% [+4.30%, +5.14%] | 0.0006* | 28,724 |
| `after_heldout` | empty | +0.0840 [+0.0762, +0.0919] | +3.60% [+3.24%, +3.97%] | 0.0006* | 28,724 |
| `after_heldout` | mean | +0.0800 [+0.0726, +0.0876] | +3.43% [+3.08%, +3.78%] | 0.0006* | 28,724 |
| `after_unseen` | other | +0.0194 [-0.0033, +0.0433] | +0.72% [-0.12%, +1.63%] | 0.286 | 924 |
| `after_unseen` | empty | -0.0034 [-0.0207, +0.0160] | -0.13% [-0.77%, +0.60%] | 1 | 924 |
| `after_unseen` | mean | -0.0041 [-0.0223, +0.0160] | -0.15% [-0.82%, +0.60%] | 1 | 924 |
| `after_rare_seen` | other | +0.0118 [+0.0010, +0.0225] | +0.41% [+0.03%, +0.80%] | 0.09 | 5,274 |
| `after_rare_seen` | empty | +0.0020 [-0.0075, +0.0114] | +0.07% [-0.26%, +0.40%] | 1 | 5,274 |
| `after_rare_seen` | mean | -0.0012 [-0.0106, +0.0079] | -0.04% [-0.37%, +0.28%] | 1 | 5,274 |

Per seed (`after_len3plus`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.0500 | +7.72% [+7.36%, +8.10%] | — | +6.36% [+6.05%, +6.67%] | +6.26% [+5.96%, +6.57%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 3.06 (counts equal).

## t1c · SmolLM2-360M · full · C5

Seeds: 1 (single seed: intervals cover windows only).

### Target set `linked` (matched stratum `after_len3plus`)

5056 target entries (0 dropped: empty frame); windows holding a target span per seed: 2047.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_len3plus` | other | +0.0701 [+0.0673, +0.0728] | +3.91% [+3.72%, +4.11%] | 0.0006* | 265,185 |
| `after_len3plus` | empty | +0.0576 [+0.0552, +0.0599] | +3.21% [+3.04%, +3.38%] | 0.0006* | 265,185 |
| `after_len3plus` | mean | +0.0559 [+0.0536, +0.0582] | +3.12% [+2.95%, +3.28%] | 0.0006* | 265,185 |
| `after` | other | +0.0751 [+0.0723, +0.0779] | +4.05% [+3.86%, +4.24%] | 0.0006* | 457,057 |
| `after` | empty | +0.0637 [+0.0612, +0.0662] | +3.43% [+3.27%, +3.60%] | 0.0006* | 457,057 |
| `after` | mean | +0.0611 [+0.0588, +0.0635] | +3.30% [+3.14%, +3.46%] | 0.0006* | 457,057 |
| `unlinked` | other | +0.0051 [+0.0049, +0.0054] | +0.28% [+0.26%, +0.29%] | 0.0006* | 1,549,478 |
| `unlinked` | empty | +0.0047 [+0.0044, +0.0049] | +0.25% [+0.24%, +0.27%] | 0.0006* | 1,549,478 |
| `unlinked` | mean | +0.0045 [+0.0042, +0.0047] | +0.24% [+0.23%, +0.26%] | 0.0006* | 1,549,478 |
| `all` | other | +0.0203 [+0.0195, +0.0211] | +1.13% [+1.08%, +1.18%] | 0.0006* | 2,095,104 |
| `all` | empty | +0.0174 [+0.0167, +0.0181] | +0.97% [+0.92%, +1.02%] | 0.0006* | 2,095,104 |
| `all` | mean | +0.0167 [+0.0161, +0.0174] | +0.93% [+0.89%, +0.98%] | 0.0006* | 2,095,104 |
| `inside` | other | +0.0178 [+0.0164, +0.0192] | +2.95% [+2.70%, +3.20%] | 0.0006* | 126,863 |
| `inside` | empty | +0.0129 [+0.0116, +0.0142] | +2.14% [+1.92%, +2.36%] | 0.0006* | 126,863 |
| `inside` | mean | +0.0124 [+0.0112, +0.0137] | +2.06% [+1.85%, +2.28%] | 0.0006* | 126,863 |
| `after_len2` | other | +0.0881 [+0.0840, +0.0922] | +4.58% [+4.33%, +4.84%] | 0.0006* | 222,537 |
| `after_len2` | empty | +0.0765 [+0.0730, +0.0802] | +3.98% [+3.76%, +4.21%] | 0.0006* | 222,537 |
| `after_len2` | mean | +0.0724 [+0.0690, +0.0758] | +3.77% [+3.56%, +3.98%] | 0.0006* | 222,537 |
| `after_heldout` | other | +0.0498 [+0.0441, +0.0557] | +2.46% [+2.16%, +2.77%] | 0.0006* | 28,724 |
| `after_heldout` | empty | +0.0275 [+0.0238, +0.0315] | +1.36% [+1.16%, +1.56%] | 0.0006* | 28,724 |
| `after_heldout` | mean | +0.0269 [+0.0233, +0.0306] | +1.32% [+1.14%, +1.52%] | 0.0006* | 28,724 |
| `after_unseen` | other | +0.0169 [+0.0020, +0.0334] | +0.70% [+0.08%, +1.40%] | 0.046* | 924 |
| `after_unseen` | empty | +0.0151 [+0.0031, +0.0294] | +0.62% [+0.13%, +1.23%] | 0.0276* | 924 |
| `after_unseen` | mean | +0.0119 [-0.0003, +0.0265] | +0.49% [-0.01%, +1.11%] | 0.0584 | 924 |
| `after_rare_seen` | other | +0.0075 [+0.0003, +0.0154] | +0.29% [+0.01%, +0.60%] | 0.123 | 5,274 |
| `after_rare_seen` | empty | +0.0039 [-0.0036, +0.0122] | +0.15% [-0.14%, +0.47%] | 0.322 | 5,274 |
| `after_rare_seen` | mean | +0.0063 [-0.0011, +0.0147] | +0.24% [-0.04%, +0.57%] | 0.198 | 5,274 |

Per seed (`after_len3plus`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 1.7926 | +3.91% [+3.72%, +4.11%] | — | +3.21% [+3.04%, +3.38%] | +3.12% [+2.95%, +3.28%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 2.99 (counts equal).

