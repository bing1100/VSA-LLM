# E9 frame swap — is the own frame used?

Evaluation-time frame swap (`e9_frameswap`; pre-registration `experiments/e9-retrofit/preregistration-frameswap.md`). Each row is `variant − own` within the same trained model on its evaluation windows (positive = the variant is worse, i.e. the run's own rows help): Δ in nats per target token and relative to `own`'s loss, token-weighted and pooled over seeds per window, with 95% cluster-bootstrap intervals over windows; Holm over the four variants within a target set × stratum (* = Holm p < 0.05). Variants: `other` = another target entry's frame (derangement within the set), `other-any` = a random non-target entry's frame, `empty` = no injection, `mean` = the mean non-target row.

**Reading rule:** a held-out gain counts as ontology-specific only if `other − own` > 0 on `after_heldout` with Holm p < 0.05.

## t7 · SmolLM2-360M · full · C5

Seeds: 1, 2, 3.

### Target set `linked` (matched stratum `after_len3plus`)

4383 target entries (0 dropped: empty frame); windows holding a target span per seed: 4094, 4094, 4094.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_len3plus` | other | +0.0221 [+0.0208, +0.0235] | +1.00% [+0.94%, +1.06%] | 0.0006* | 595,311 |
| `after_len3plus` | empty | +0.0215 [+0.0199, +0.0231] | +0.97% [+0.90%, +1.04%] | 0.0006* | 595,311 |
| `after_len3plus` | mean | +0.0178 [+0.0166, +0.0191] | +0.80% [+0.75%, +0.86%] | 0.0006* | 595,311 |
| `after` | other | +0.0219 [+0.0206, +0.0233] | +0.99% [+0.93%, +1.05%] | 0.0006* | 624,990 |
| `after` | empty | +0.0212 [+0.0196, +0.0228] | +0.95% [+0.88%, +1.02%] | 0.0006* | 624,990 |
| `after` | mean | +0.0175 [+0.0163, +0.0187] | +0.79% [+0.73%, +0.84%] | 0.0006* | 624,990 |
| `unlinked` | other | +0.0004 [+0.0004, +0.0005] | +0.02% [+0.02%, +0.02%] | 0.0006* | 11,654,319 |
| `unlinked` | empty | +0.0003 [+0.0003, +0.0004] | +0.02% [+0.01%, +0.02%] | 0.0006* | 11,654,319 |
| `unlinked` | mean | +0.0003 [+0.0003, +0.0004] | +0.02% [+0.01%, +0.02%] | 0.0006* | 11,654,319 |
| `all` | other | +0.0015 [+0.0014, +0.0016] | +0.07% [+0.06%, +0.07%] | 0.0006* | 12,570,624 |
| `all` | empty | +0.0013 [+0.0013, +0.0014] | +0.06% [+0.06%, +0.07%] | 0.0006* | 12,570,624 |
| `all` | mean | +0.0012 [+0.0011, +0.0012] | +0.05% [+0.05%, +0.06%] | 0.0006* | 12,570,624 |
| `inside` | other | +0.0054 [+0.0044, +0.0064] | +0.49% [+0.40%, +0.58%] | 0.0006* | 309,057 |
| `inside` | empty | +0.0019 [+0.0011, +0.0026] | +0.17% [+0.10%, +0.24%] | 0.0006* | 309,057 |
| `inside` | mean | +0.0035 [+0.0026, +0.0044] | +0.32% [+0.24%, +0.40%] | 0.0006* | 309,057 |
| `after_len2` | other | +0.0178 [+0.0130, +0.0229] | +0.75% [+0.55%, +0.97%] | 0.0006* | 30,471 |
| `after_len2` | empty | +0.0149 [+0.0087, +0.0215] | +0.63% [+0.37%, +0.91%] | 0.0006* | 30,471 |
| `after_len2` | mean | +0.0103 [+0.0064, +0.0145] | +0.44% [+0.27%, +0.61%] | 0.0006* | 30,471 |
| `after_heldout` | other | +0.0022 [+0.0004, +0.0040] | +0.10% [+0.02%, +0.18%] | 0.034* | 129,489 |
| `after_heldout` | empty | -0.0074 [-0.0103, -0.0045] | -0.34% [-0.47%, -0.20%] | 0.0006* | 129,489 |
| `after_heldout` | mean | +0.0020 [+0.0003, +0.0037] | +0.09% [+0.02%, +0.17%] | 0.034* | 129,489 |
| `after_rare_seen` | other | +0.0134 [+0.0106, +0.0163] | +0.57% [+0.45%, +0.70%] | 0.0006* | 64,869 |
| `after_rare_seen` | empty | +0.0155 [+0.0118, +0.0194] | +0.66% [+0.51%, +0.83%] | 0.0006* | 64,869 |
| `after_rare_seen` | mean | +0.0089 [+0.0065, +0.0115] | +0.38% [+0.28%, +0.49%] | 0.0006* | 64,869 |

Per seed (`after_len3plus`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.2167 | +1.04% [+0.96%, +1.11%] | — | +0.96% [+0.89%, +1.04%] | +0.80% [+0.74%, +0.86%] |
| 2 | 2.2168 | +0.94% [+0.87%, +1.01%] | — | +0.96% [+0.89%, +1.04%] | +0.76% [+0.71%, +0.82%] |
| 3 | 2.2163 | +1.02% [+0.94%, +1.09%] | — | +0.98% [+0.90%, +1.05%] | +0.85% [+0.79%, +0.91%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 3.39 (counts equal), s2 2.68 (counts equal), s3 3.03 (counts equal).

## t7 · SmolLM2-360M · full · C5sh

Seeds: 1, 2, 3.

### Target set `linked` (matched stratum `after_len3plus`)

4383 target entries (0 dropped: empty frame); windows holding a target span per seed: 4094, 4094, 4094.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_len3plus` | other | +0.0189 [+0.0176, +0.0201] | +0.85% [+0.79%, +0.91%] | 0.0006* | 595,311 |
| `after_len3plus` | empty | +0.0202 [+0.0188, +0.0218] | +0.91% [+0.85%, +0.98%] | 0.0006* | 595,311 |
| `after_len3plus` | mean | +0.0159 [+0.0148, +0.0171] | +0.72% [+0.67%, +0.77%] | 0.0006* | 595,311 |
| `after` | other | +0.0187 [+0.0175, +0.0199] | +0.84% [+0.78%, +0.90%] | 0.0006* | 624,990 |
| `after` | empty | +0.0200 [+0.0186, +0.0215] | +0.90% [+0.83%, +0.97%] | 0.0006* | 624,990 |
| `after` | mean | +0.0156 [+0.0145, +0.0167] | +0.70% [+0.65%, +0.75%] | 0.0006* | 624,990 |
| `unlinked` | other | +0.0003 [+0.0003, +0.0004] | +0.01% [+0.01%, +0.02%] | 0.0006* | 11,654,319 |
| `unlinked` | empty | +0.0003 [+0.0002, +0.0003] | +0.01% [+0.01%, +0.02%] | 0.0006* | 11,654,319 |
| `unlinked` | mean | +0.0003 [+0.0002, +0.0003] | +0.01% [+0.01%, +0.01%] | 0.0006* | 11,654,319 |
| `all` | other | +0.0012 [+0.0012, +0.0013] | +0.06% [+0.05%, +0.06%] | 0.0006* | 12,570,624 |
| `all` | empty | +0.0013 [+0.0012, +0.0013] | +0.06% [+0.05%, +0.06%] | 0.0006* | 12,570,624 |
| `all` | mean | +0.0010 [+0.0009, +0.0011] | +0.05% [+0.04%, +0.05%] | 0.0006* | 12,570,624 |
| `inside` | other | +0.0053 [+0.0044, +0.0062] | +0.48% [+0.39%, +0.57%] | 0.0006* | 309,057 |
| `inside` | empty | +0.0026 [+0.0018, +0.0034] | +0.23% [+0.16%, +0.31%] | 0.0006* | 309,057 |
| `inside` | mean | +0.0039 [+0.0030, +0.0047] | +0.35% [+0.28%, +0.43%] | 0.0006* | 309,057 |
| `after_len2` | other | +0.0147 [+0.0106, +0.0191] | +0.62% [+0.45%, +0.81%] | 0.0006* | 30,471 |
| `after_len2` | empty | +0.0140 [+0.0081, +0.0202] | +0.59% [+0.34%, +0.86%] | 0.0006* | 30,471 |
| `after_len2` | mean | +0.0088 [+0.0053, +0.0125] | +0.37% [+0.22%, +0.53%] | 0.0006* | 30,471 |
| `after_heldout` | other | +0.0041 [+0.0024, +0.0059] | +0.19% [+0.11%, +0.27%] | 0.0006* | 129,489 |
| `after_heldout` | empty | -0.0044 [-0.0067, -0.0023] | -0.20% [-0.31%, -0.10%] | 0.0006* | 129,489 |
| `after_heldout` | mean | +0.0036 [+0.0015, +0.0058] | +0.16% [+0.07%, +0.27%] | 0.0016* | 129,489 |
| `after_rare_seen` | other | +0.0088 [+0.0066, +0.0111] | +0.38% [+0.28%, +0.48%] | 0.0006* | 64,869 |
| `after_rare_seen` | empty | +0.0131 [+0.0097, +0.0166] | +0.56% [+0.41%, +0.71%] | 0.0006* | 64,869 |
| `after_rare_seen` | mean | +0.0060 [+0.0039, +0.0081] | +0.26% [+0.17%, +0.35%] | 0.0006* | 64,869 |

Per seed (`after_len3plus`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.2164 | +0.84% [+0.78%, +0.90%] | — | +0.93% [+0.86%, +1.00%] | +0.73% [+0.67%, +0.79%] |
| 2 | 2.2165 | +0.85% [+0.79%, +0.91%] | — | +0.91% [+0.84%, +0.98%] | +0.69% [+0.64%, +0.75%] |
| 3 | 2.2169 | +0.87% [+0.80%, +0.93%] | — | +0.90% [+0.83%, +0.97%] | +0.73% [+0.68%, +0.79%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 3.33 (counts equal), s2 2.22 (counts equal), s3 2.02 (counts equal).

