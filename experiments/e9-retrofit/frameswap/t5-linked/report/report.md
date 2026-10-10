# E9 frame swap — is the own frame used?

Evaluation-time frame swap (`e9_frameswap`; pre-registration `experiments/e9-retrofit/preregistration-frameswap.md`). Each row is `variant − own` within the same trained model on its evaluation windows (positive = the variant is worse, i.e. the run's own rows help): Δ in nats per target token and relative to `own`'s loss, token-weighted and pooled over seeds per window, with 95% cluster-bootstrap intervals over windows; Holm over the four variants within a target set × stratum (* = Holm p < 0.05). Variants: `other` = another target entry's frame (derangement within the set), `other-any` = a random non-target entry's frame, `empty` = no injection, `mean` = the mean non-target row.

**Reading rule:** a held-out gain counts as ontology-specific only if `other − own` > 0 on `after_heldout` with Holm p < 0.05.

## t5 · SmolLM2-360M · full · C5

Seeds: 1, 2, 3.

### Target set `linked` (matched stratum `after_len3plus`)

3603 target entries (0 dropped: empty frame); windows holding a target span per seed: 1024, 1024, 1024.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_len3plus` | other | +0.1893 [+0.1856, +0.1929] | +34.16% [+33.36%, +34.98%] | 0.0006* | 1,447,566 |
| `after_len3plus` | empty | +0.1351 [+0.1323, +0.1380] | +24.39% [+23.82%, +24.99%] | 0.0006* | 1,447,566 |
| `after_len3plus` | mean | +0.1129 [+0.1105, +0.1153] | +20.38% [+19.87%, +20.90%] | 0.0006* | 1,447,566 |
| `after_len3plus_filler` | other | +0.8230 [+0.8075, +0.8383] | +64.52% [+62.38%, +66.75%] | 0.0006* | 268,752 |
| `after_len3plus_filler` | empty | +0.5528 [+0.5416, +0.5645] | +43.34% [+41.90%, +44.87%] | 0.0006* | 268,752 |
| `after_len3plus_filler` | mean | +0.4998 [+0.4900, +0.5098] | +39.18% [+37.90%, +40.55%] | 0.0006* | 268,752 |
| `after_len3plus_nonfiller` | other | +0.0448 [+0.0435, +0.0461] | +11.50% [+11.13%, +11.87%] | 0.0006* | 1,178,814 |
| `after_len3plus_nonfiller` | empty | +0.0399 [+0.0384, +0.0415] | +10.24% [+9.84%, +10.66%] | 0.0006* | 1,178,814 |
| `after_len3plus_nonfiller` | mean | +0.0247 [+0.0235, +0.0259] | +6.34% [+6.03%, +6.65%] | 0.0006* | 1,178,814 |
| `after` | other | +0.1892 [+0.1855, +0.1929] | +34.15% [+33.35%, +34.96%] | 0.0006* | 1,448,394 |
| `after` | empty | +0.1351 [+0.1323, +0.1380] | +24.38% [+23.82%, +24.99%] | 0.0006* | 1,448,394 |
| `after` | mean | +0.1129 [+0.1105, +0.1153] | +20.37% [+19.87%, +20.90%] | 0.0006* | 1,448,394 |
| `unlinked` | other | +0.0097 [+0.0090, +0.0104] | +1.88% [+1.74%, +2.02%] | 0.0006* | 1,142,448 |
| `unlinked` | empty | +0.0057 [+0.0051, +0.0062] | +1.10% [+0.99%, +1.20%] | 0.0006* | 1,142,448 |
| `unlinked` | mean | +0.0048 [+0.0044, +0.0053] | +0.94% [+0.85%, +1.03%] | 0.0006* | 1,142,448 |
| `all` | other | +0.0914 [+0.0896, +0.0931] | +14.33% [+14.02%, +14.64%] | 0.0006* | 3,142,656 |
| `all` | empty | +0.0643 [+0.0630, +0.0657] | +10.09% [+9.87%, +10.32%] | 0.0006* | 3,142,656 |
| `all` | mean | +0.0527 [+0.0515, +0.0539] | +8.26% [+8.07%, +8.46%] | 0.0006* | 3,142,656 |
| `inside` | other | +0.0057 [+0.0050, +0.0064] | +0.56% [+0.49%, +0.63%] | 0.0006* | 859,584 |
| `inside` | empty | +0.0020 [+0.0011, +0.0029] | +0.20% [+0.11%, +0.29%] | 0.0006* | 859,584 |
| `inside` | mean | -0.0036 [-0.0043, -0.0030] | -0.35% [-0.42%, -0.29%] | 0.0006* | 859,584 |
| `after_len2` | other | +0.1352 [+0.0614, +0.2005] | +23.46% [+8.94%, +48.90%] | 0.0012* | 930 |
| `after_len2` | empty | +0.1060 [+0.0331, +0.1657] | +18.39% [+5.96%, +36.37%] | 0.0028* | 930 |
| `after_len2` | mean | +0.0958 [+0.0380, +0.1494] | +16.62% [+5.71%, +35.88%] | 0.0012* | 930 |
| `after_heldout` | other | +0.1820 [+0.1750, +0.1892] | +24.17% [+22.98%, +25.42%] | 0.0006* | 150,369 |
| `after_heldout` | empty | +0.2133 [+0.2039, +0.2232] | +28.32% [+26.76%, +29.97%] | 0.0006* | 150,369 |
| `after_heldout` | mean | +0.1564 [+0.1491, +0.1640] | +20.76% [+19.56%, +22.03%] | 0.0006* | 150,369 |
| `after_unseen` | other | +0.2040 [+0.1748, +0.2349] | +24.79% [+20.57%, +29.87%] | 0.0006* | 14,721 |
| `after_unseen` | empty | +0.1426 [+0.1179, +0.1682] | +17.33% [+14.01%, +21.11%] | 0.0006* | 14,721 |
| `after_unseen` | mean | +0.1173 [+0.0989, +0.1365] | +14.25% [+11.72%, +17.19%] | 0.0006* | 14,721 |
| `after_rare_seen` | other | +0.1762 [+0.1660, +0.1868] | +22.51% [+20.98%, +24.14%] | 0.0006* | 108,909 |
| `after_rare_seen` | empty | +0.1322 [+0.1235, +0.1410] | +16.88% [+15.60%, +18.21%] | 0.0006* | 108,909 |
| `after_rare_seen` | mean | +0.1045 [+0.0970, +0.1122] | +13.34% [+12.29%, +14.45%] | 0.0006* | 108,909 |

Per seed (`after_len3plus`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 0.5568 | +34.00% [+33.16%, +34.86%] | — | +23.22% [+22.68%, +23.80%] | +19.77% [+19.28%, +20.29%] |
| 2 | 0.5535 | +36.07% [+35.14%, +37.00%] | — | +25.40% [+24.80%, +26.03%] | +21.01% [+20.48%, +21.55%] |
| 3 | 0.5521 | +32.40% [+31.61%, +33.22%] | — | +24.55% [+23.96%, +25.17%] | +20.36% [+19.85%, +20.89%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 1.77 (counts equal), s2 2.33 (counts equal), s3 1.51 (counts equal).

