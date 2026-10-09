# SMOKE — E9 frame swap — T4 SmolLM2-360M, CPU smoke on 8 evaluation windows

Evaluation-time frame swap (`e9_frameswap`; pre-registration `experiments/e9-retrofit/preregistration-frameswap.md`). Each row is `variant − own` within the same trained model on its evaluation windows (positive = the variant is worse, i.e. the run's own rows help): Δ in nats per target token and relative to `own`'s loss, token-weighted and pooled over seeds per window, with 95% cluster-bootstrap intervals over windows; Holm over the four variants within a target set × stratum (* = Holm p < 0.05). Variants: `other` = another target entry's frame (derangement within the set), `other-any` = a random non-target entry's frame, `empty` = no injection, `mean` = the mean non-target row.

**Reading rule:** a held-out gain counts as ontology-specific only if `other − own` > 0 on `after_heldout` with Holm p < 0.05.

**SMOKE:** at least one input was scored on a window subset (`--limit-windows`); the numbers check the pipeline only.

## t4 · SmolLM2-360M · full · C5

Seeds: 1 (single seed: intervals cover windows only).

**Primary** (`other − own`, `after_heldout`): Δ -0.0041 [-0.0150, +0.0045], -0.23% [-0.94%, +0.25%], Holm p 1 → not shown ontology-specific.

### Target set `heldout` (matched stratum `after_heldout`)

850 target entries (0 dropped: empty frame); windows holding a target span per seed: 8.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_heldout` | other | -0.0041 [-0.0150, +0.0045] | -0.23% [-0.94%, +0.25%] | 1 | 341 |
| `after_heldout` | other-any | +0.0026 [-0.0082, +0.0108] | +0.14% [-0.44%, +0.61%] | 1 | 341 |
| `after_heldout` | empty | -0.0014 [-0.0076, +0.0072] | -0.07% [-0.47%, +0.43%] | 1 | 341 |
| `after_heldout` | mean | -0.0038 [-0.0100, +0.0023] | -0.21% [-0.61%, +0.13%] | 0.79 | 341 |
| `after_heldout_filler` | other | +0.0389 [-0.0135, +0.1437] | +1.93% [-1.56%, +3.32%] | 1 | 6 |
| `after_heldout_filler` | other-any | +0.0294 [-0.0276, +0.1433] | +1.46% [-3.20%, +3.31%] | 1 | 6 |
| `after_heldout_filler` | empty | +0.0148 [-0.0195, +0.0834] | +0.73% [-2.26%, +1.93%] | 1 | 6 |
| `after_heldout_filler` | mean | +0.0211 [-0.0250, +0.1131] | +1.04% [-2.89%, +2.61%] | 1 | 6 |
| `after_heldout_nonfiller` | other | -0.0049 [-0.0163, +0.0039] | -0.27% [-0.99%, +0.23%] | 0.873 | 335 |
| `after_heldout_nonfiller` | other-any | +0.0021 [-0.0104, +0.0114] | +0.11% [-0.56%, +0.64%] | 1 | 335 |
| `after_heldout_nonfiller` | empty | -0.0016 [-0.0081, +0.0069] | -0.09% [-0.48%, +0.42%] | 1 | 335 |
| `after_heldout_nonfiller` | mean | -0.0042 [-0.0111, +0.0022] | -0.23% [-0.66%, +0.13%] | 0.805 | 335 |
| `after` | other | -0.0003 [-0.0014, +0.0011] | -0.02% [-0.09%, +0.07%] | 1 | 2,130 |
| `after` | other-any | +0.0000 [-0.0019, +0.0022] | +0.00% [-0.11%, +0.14%] | 1 | 2,130 |
| `after` | empty | -0.0004 [-0.0014, +0.0007] | -0.03% [-0.08%, +0.04%] | 1 | 2,130 |
| `after` | mean | -0.0007 [-0.0018, +0.0003] | -0.04% [-0.10%, +0.02%] | 0.686 | 2,130 |
| `unlinked` | other | +0.0005 [+0.0000, +0.0012] | +0.02% [+0.00%, +0.06%] | 0.0984 | 5,465 |
| `unlinked` | other-any | +0.0002 [-0.0002, +0.0005] | +0.01% [-0.01%, +0.02%] | 0.844 | 5,465 |
| `unlinked` | empty | +0.0001 [-0.0001, +0.0004] | +0.00% [-0.01%, +0.02%] | 0.844 | 5,465 |
| `unlinked` | mean | +0.0002 [-0.0001, +0.0005] | +0.01% [-0.00%, +0.02%] | 0.569 | 5,465 |
| `all` | other | +0.0003 [-0.0002, +0.0009] | +0.02% [-0.01%, +0.05%] | 1 | 8,184 |
| `all` | other-any | +0.0002 [-0.0003, +0.0008] | +0.01% [-0.02%, +0.04%] | 1 | 8,184 |
| `all` | empty | +0.0000 [-0.0002, +0.0002] | +0.00% [-0.01%, +0.01%] | 1 | 8,184 |
| `all` | mean | -0.0000 [-0.0003, +0.0002] | -0.00% [-0.01%, +0.01%] | 1 | 8,184 |

Per seed (`after_heldout`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 1.8197 | -0.23% [-0.94%, +0.25%] | +0.14% [-0.44%, +0.61%] | -0.07% [-0.47%, +0.43%] | -0.21% [-0.61%, +0.13%] |

### Target set `unseen` (matched stratum `after_unseen`)

18735 target entries (0 dropped: empty frame); windows holding a target span per seed: 6.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_unseen` | other | +0.0166 [-0.0425, +0.0601] | +1.09% [-3.13%, +4.16%] | 0.565 | 56 |
| `after_unseen` | other-any | +0.0396 [+0.0095, +0.0644] | +2.61% [+0.55%, +4.73%] | 0.0248* | 56 |
| `after_unseen` | empty | +0.1652 [-0.0005, +0.4787] | +10.87% [-0.04%, +35.35%] | 0.155 | 56 |
| `after_unseen` | mean | +0.0735 [-0.0016, +0.1904] | +4.84% [-0.10%, +14.31%] | 0.155 | 56 |
| `after_unseen_filler` | other | +0.0235 [-0.0070, +0.0557] | +1.31% [-0.36%, +2.75%] | 0.874 | 25 |
| `after_unseen_filler` | other-any | +0.0662 [-0.0067, +0.1146] | +3.69% [-0.46%, +5.53%] | 0.31 | 25 |
| `after_unseen_filler` | empty | +0.0381 [-0.0382, +0.0972] | +2.13% [-2.28%, +4.90%] | 0.874 | 25 |
| `after_unseen_filler` | mean | +0.0411 [-0.0112, +0.0825] | +2.29% [-0.75%, +4.09%] | 0.509 | 25 |
| `after_unseen_nonfiller` | other | +0.0110 [-0.0760, +0.0689] | +0.85% [-6.55%, +9.08%] | 0.732 | 31 |
| `after_unseen_nonfiller` | other-any | +0.0182 [-0.0086, +0.0520] | +1.40% [-0.53%, +5.72%] | 0.411 | 31 |
| `after_unseen_nonfiller` | empty | +0.2676 [+0.0192, +0.8588] | +20.59% [+1.02%, +82.93%] | 0.0008* | 31 |
| `after_unseen_nonfiller` | mean | +0.0997 [+0.0001, +0.3347] | +7.67% [+0.01%, +32.31%] | 0.139 | 31 |
| `after` | other | +0.0009 [-0.0011, +0.0036] | +0.06% [-0.07%, +0.21%] | 0.564 | 2,130 |
| `after` | other-any | +0.0013 [-0.0000, +0.0032] | +0.08% [-0.00%, +0.18%] | 0.216 | 2,130 |
| `after` | empty | +0.0041 [-0.0003, +0.0109] | +0.25% [-0.02%, +0.74%] | 0.388 | 2,130 |
| `after` | mean | +0.0017 [-0.0003, +0.0045] | +0.11% [-0.02%, +0.31%] | 0.388 | 2,130 |
| `unlinked` | other | +0.0001 [-0.0001, +0.0005] | +0.01% [-0.01%, +0.02%] | 1 | 5,465 |
| `unlinked` | other-any | +0.0000 [-0.0002, +0.0002] | +0.00% [-0.01%, +0.01%] | 1 | 5,465 |
| `unlinked` | empty | +0.0001 [-0.0001, +0.0003] | +0.00% [-0.01%, +0.01%] | 1 | 5,465 |
| `unlinked` | mean | +0.0001 [-0.0001, +0.0003] | +0.00% [-0.00%, +0.01%] | 1 | 5,465 |
| `all` | other | +0.0003 [-0.0003, +0.0012] | +0.02% [-0.01%, +0.06%] | 0.561 | 8,184 |
| `all` | other-any | +0.0004 [-0.0001, +0.0010] | +0.02% [-0.00%, +0.05%] | 0.448 | 8,184 |
| `all` | empty | +0.0011 [-0.0002, +0.0030] | +0.06% [-0.01%, +0.16%] | 0.483 | 8,184 |
| `all` | mean | +0.0005 [-0.0001, +0.0014] | +0.03% [-0.01%, +0.07%] | 0.483 | 8,184 |

Per seed (`after_unseen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 1.5197 | +1.09% [-3.13%, +4.16%] | +2.61% [+0.55%, +4.73%] | +10.87% [-0.04%, +35.35%] | +4.84% [-0.10%, +14.31%] |

### Target set `rare_seen` (matched stratum `after_rare_seen`)

39010 target entries (0 dropped: empty frame); windows holding a target span per seed: 8.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_rare_seen` | other | +0.0004 [-0.0101, +0.0103] | +0.03% [-0.69%, +0.64%] | 1 | 338 |
| `after_rare_seen` | other-any | +0.0012 [-0.0075, +0.0148] | +0.08% [-0.48%, +1.00%] | 1 | 338 |
| `after_rare_seen` | empty | +0.0070 [-0.0092, +0.0178] | +0.44% [-0.62%, +1.11%] | 1 | 338 |
| `after_rare_seen` | mean | -0.0015 [-0.0143, +0.0060] | -0.09% [-0.93%, +0.39%] | 1 | 338 |
| `after_rare_seen_filler` | other | -0.0186 [-0.0902, +0.0513] | -1.02% [-5.27%, +2.47%] | 0.939 | 37 |
| `after_rare_seen_filler` | other-any | -0.0176 [-0.0636, +0.0219] | -0.97% [-3.54%, +1.09%] | 0.939 | 37 |
| `after_rare_seen_filler` | empty | +0.0173 [-0.0087, +0.0364] | +0.95% [-0.41%, +2.18%] | 0.685 | 37 |
| `after_rare_seen_filler` | mean | -0.0122 [-0.0512, +0.0099] | -0.67% [-2.69%, +0.53%] | 0.939 | 37 |
| `after_rare_seen_nonfiller` | other | +0.0027 [-0.0080, +0.0102] | +0.18% [-0.58%, +0.63%] | 1 | 301 |
| `after_rare_seen_nonfiller` | other-any | +0.0036 [-0.0080, +0.0192] | +0.23% [-0.52%, +1.33%] | 1 | 301 |
| `after_rare_seen_nonfiller` | empty | +0.0057 [-0.0114, +0.0174] | +0.37% [-0.80%, +1.11%] | 1 | 301 |
| `after_rare_seen_nonfiller` | mean | -0.0002 [-0.0141, +0.0073] | -0.01% [-1.00%, +0.47%] | 1 | 301 |
| `after` | other | +0.0001 [-0.0030, +0.0034] | +0.01% [-0.19%, +0.21%] | 1 | 2,130 |
| `after` | other-any | -0.0003 [-0.0029, +0.0024] | -0.02% [-0.18%, +0.14%] | 1 | 2,130 |
| `after` | empty | +0.0017 [-0.0009, +0.0052] | +0.11% [-0.05%, +0.32%] | 0.888 | 2,130 |
| `after` | mean | +0.0005 [-0.0013, +0.0025] | +0.03% [-0.08%, +0.15%] | 1 | 2,130 |
| `unlinked` | other | +0.0006 [-0.0003, +0.0019] | +0.03% [-0.01%, +0.09%] | 0.56 | 5,465 |
| `unlinked` | other-any | -0.0002 [-0.0009, +0.0004] | -0.01% [-0.05%, +0.02%] | 0.56 | 5,465 |
| `unlinked` | empty | +0.0004 [-0.0001, +0.0008] | +0.02% [-0.00%, +0.04%] | 0.269 | 5,465 |
| `unlinked` | mean | +0.0005 [-0.0000, +0.0012] | +0.03% [-0.00%, +0.06%] | 0.269 | 5,465 |
| `all` | other | +0.0005 [-0.0005, +0.0015] | +0.02% [-0.03%, +0.07%] | 0.703 | 8,184 |
| `all` | other-any | -0.0002 [-0.0010, +0.0008] | -0.01% [-0.05%, +0.04%] | 0.723 | 8,184 |
| `all` | empty | +0.0007 [-0.0000, +0.0015] | +0.04% [-0.00%, +0.08%] | 0.206 | 8,184 |
| `all` | mean | +0.0004 [-0.0000, +0.0009] | +0.02% [-0.00%, +0.04%] | 0.206 | 8,184 |

Per seed (`after_rare_seen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 1.5818 | +0.03% [-0.69%, +0.64%] | +0.08% [-0.48%, +1.00%] | +0.44% [-0.62%, +1.11%] | -0.09% [-0.93%, +0.39%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 2.39 (counts equal).

## t4 · SmolLM2-360M · full · C5sh

Seeds: 1 (single seed: intervals cover windows only).

**Primary** (`other − own`, `after_heldout`): Δ +0.0011 [-0.0061, +0.0102], +0.06% [-0.34%, +0.60%], Holm p 1 → not shown ontology-specific.

### Target set `heldout` (matched stratum `after_heldout`)

850 target entries (0 dropped: empty frame); windows holding a target span per seed: 8.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_heldout` | other | +0.0011 [-0.0061, +0.0102] | +0.06% [-0.34%, +0.60%] | 1 | 341 |
| `after_heldout` | other-any | -0.0011 [-0.0120, +0.0157] | -0.06% [-0.65%, +1.00%] | 1 | 341 |
| `after_heldout` | empty | +0.0009 [-0.0088, +0.0127] | +0.05% [-0.48%, +0.73%] | 1 | 341 |
| `after_heldout` | mean | -0.0005 [-0.0071, +0.0084] | -0.03% [-0.39%, +0.51%] | 1 | 341 |
| `after_heldout_filler` | other | +0.0056 [-0.0235, +0.0202] | +0.28% [-0.53%, +2.68%] | 1 | 6 |
| `after_heldout_filler` | other-any | -0.0100 [-0.0129, -0.0086] | -0.51% [-1.14%, -0.29%] | 0.000886* | 6 |
| `after_heldout_filler` | empty | -0.0023 [-0.0145, +0.0038] | -0.12% [-0.33%, +0.51%] | 1 | 6 |
| `after_heldout_filler` | mean | +0.0067 [+0.0056, +0.0088] | +0.34% [+0.20%, +0.74%] | 0.000886* | 6 |
| `after_heldout_nonfiller` | other | +0.0010 [-0.0065, +0.0105] | +0.06% [-0.35%, +0.61%] | 1 | 335 |
| `after_heldout_nonfiller` | other-any | -0.0009 [-0.0121, +0.0157] | -0.05% [-0.65%, +1.01%] | 1 | 335 |
| `after_heldout_nonfiller` | empty | +0.0009 [-0.0091, +0.0130] | +0.05% [-0.49%, +0.75%] | 1 | 335 |
| `after_heldout_nonfiller` | mean | -0.0006 [-0.0074, +0.0084] | -0.03% [-0.40%, +0.51%] | 1 | 335 |
| `after` | other | +0.0001 [-0.0014, +0.0014] | +0.00% [-0.09%, +0.09%] | 1 | 2,130 |
| `after` | other-any | +0.0001 [-0.0025, +0.0027] | +0.01% [-0.16%, +0.17%] | 1 | 2,130 |
| `after` | empty | -0.0002 [-0.0023, +0.0015] | -0.01% [-0.14%, +0.09%] | 1 | 2,130 |
| `after` | mean | -0.0003 [-0.0017, +0.0011] | -0.02% [-0.11%, +0.07%] | 1 | 2,130 |
| `unlinked` | other | +0.0002 [+0.0000, +0.0005] | +0.01% [+0.00%, +0.02%] | 0.0576 | 5,465 |
| `unlinked` | other-any | +0.0001 [-0.0000, +0.0003] | +0.01% [-0.00%, +0.01%] | 0.389 | 5,465 |
| `unlinked` | empty | -0.0000 [-0.0002, +0.0002] | -0.00% [-0.01%, +0.01%] | 1 | 5,465 |
| `unlinked` | mean | +0.0000 [-0.0001, +0.0002] | +0.00% [-0.01%, +0.01%] | 1 | 5,465 |
| `all` | other | +0.0002 [-0.0001, +0.0004] | +0.01% [-0.00%, +0.02%] | 0.682 | 8,184 |
| `all` | other-any | +0.0001 [-0.0005, +0.0008] | +0.01% [-0.03%, +0.04%] | 1 | 8,184 |
| `all` | empty | -0.0000 [-0.0004, +0.0003] | -0.00% [-0.02%, +0.02%] | 1 | 8,184 |
| `all` | mean | -0.0000 [-0.0003, +0.0003] | -0.00% [-0.02%, +0.01%] | 1 | 8,184 |

Per seed (`after_heldout`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 1.8275 | +0.06% [-0.34%, +0.60%] | -0.06% [-0.65%, +1.00%] | +0.05% [-0.48%, +0.73%] | -0.03% [-0.39%, +0.51%] |

### Target set `unseen` (matched stratum `after_unseen`)

18735 target entries (0 dropped: empty frame); windows holding a target span per seed: 6.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_unseen` | other | +0.0986 [+0.0003, +0.3120] | +6.34% [+0.02%, +21.36%] | 0.185 | 56 |
| `after_unseen` | other-any | -0.0082 [-0.0282, +0.0141] | -0.53% [-2.20%, +0.69%] | 0.416 | 56 |
| `after_unseen` | empty | +0.1181 [-0.0055, +0.3565] | +7.59% [-0.36%, +24.89%] | 0.251 | 56 |
| `after_unseen` | mean | +0.0414 [-0.0082, +0.1217] | +2.66% [-0.56%, +8.23%] | 0.29 | 56 |
| `after_unseen_filler` | other | -0.0005 [-0.0437, +0.0375] | -0.03% [-2.73%, +1.81%] | 1 | 25 |
| `after_unseen_filler` | other-any | -0.0031 [-0.0314, +0.0159] | -0.17% [-1.66%, +0.80%] | 1 | 25 |
| `after_unseen_filler` | empty | -0.0053 [-0.0569, +0.0389] | -0.29% [-3.24%, +1.80%] | 1 | 25 |
| `after_unseen_filler` | mean | -0.0041 [-0.0423, +0.0263] | -0.23% [-2.42%, +1.28%] | 1 | 25 |
| `after_unseen_nonfiller` | other | +0.1784 [-0.0026, +0.5987] | +13.42% [-0.34%, +49.46%] | 0.224 | 31 |
| `after_unseen_nonfiller` | other-any | -0.0124 [-0.0503, +0.0291] | -0.93% [-5.01%, +1.50%] | 0.536 | 31 |
| `after_unseen_nonfiller` | empty | +0.2176 [+0.0086, +0.6771] | +16.37% [+0.84%, +56.61%] | 0.0104* | 31 |
| `after_unseen_nonfiller` | mean | +0.0782 [-0.0093, +0.2357] | +5.88% [-0.96%, +19.59%] | 0.258 | 31 |
| `after` | other | +0.0027 [-0.0002, +0.0075] | +0.17% [-0.01%, +0.51%] | 0.486 | 2,130 |
| `after` | other-any | +0.0001 [-0.0009, +0.0010] | +0.01% [-0.06%, +0.06%] | 0.843 | 2,130 |
| `after` | empty | +0.0030 [-0.0003, +0.0081] | +0.19% [-0.02%, +0.56%] | 0.486 | 2,130 |
| `after` | mean | +0.0010 [-0.0004, +0.0029] | +0.06% [-0.03%, +0.20%] | 0.486 | 2,130 |
| `unlinked` | other | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.01%, +0.01%] | 1 | 5,465 |
| `unlinked` | other-any | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 5,465 |
| `unlinked` | empty | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.01%, +0.01%] | 1 | 5,465 |
| `unlinked` | mean | +0.0000 [-0.0001, +0.0002] | +0.00% [-0.00%, +0.01%] | 1 | 5,465 |
| `all` | other | +0.0007 [-0.0001, +0.0020] | +0.04% [-0.01%, +0.11%] | 0.682 | 8,184 |
| `all` | other-any | +0.0000 [-0.0002, +0.0003] | +0.00% [-0.01%, +0.02%] | 0.854 | 8,184 |
| `all` | empty | +0.0008 [-0.0002, +0.0022] | +0.04% [-0.01%, +0.12%] | 0.662 | 8,184 |
| `all` | mean | +0.0003 [-0.0002, +0.0009] | +0.02% [-0.01%, +0.05%] | 0.682 | 8,184 |

Per seed (`after_unseen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 1.5558 | +6.34% [+0.02%, +21.36%] | -0.53% [-2.20%, +0.69%] | +7.59% [-0.36%, +24.89%] | +2.66% [-0.56%, +8.23%] |

### Target set `rare_seen` (matched stratum `after_rare_seen`)

39010 target entries (0 dropped: empty frame); windows holding a target span per seed: 8.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_rare_seen` | other | -0.0183 [-0.0350, -0.0032] | -1.15% [-2.32%, -0.20%] | 0.0744 | 338 |
| `after_rare_seen` | other-any | -0.0160 [-0.0374, +0.0017] | -1.01% [-2.53%, +0.10%] | 0.221 | 338 |
| `after_rare_seen` | empty | -0.0079 [-0.0265, +0.0067] | -0.50% [-1.77%, +0.42%] | 0.308 | 338 |
| `after_rare_seen` | mean | -0.0146 [-0.0357, +0.0020] | -0.92% [-2.37%, +0.13%] | 0.221 | 338 |
| `after_rare_seen_filler` | other | -0.0464 [-0.1088, +0.0342] | -2.55% [-6.35%, +1.73%] | 0.962 | 37 |
| `after_rare_seen_filler` | other-any | -0.0359 [-0.1105, +0.0590] | -1.98% [-6.31%, +3.13%] | 0.962 | 37 |
| `after_rare_seen_filler` | empty | -0.0172 [-0.0615, +0.0461] | -0.94% [-3.46%, +2.50%] | 0.962 | 37 |
| `after_rare_seen_filler` | mean | -0.0346 [-0.0791, +0.0247] | -1.90% [-4.56%, +1.28%] | 0.962 | 37 |
| `after_rare_seen_nonfiller` | other | -0.0148 [-0.0361, -0.0005] | -0.95% [-2.44%, -0.03%] | 0.165 | 301 |
| `after_rare_seen_nonfiller` | other-any | -0.0136 [-0.0384, -0.0001] | -0.87% [-2.70%, -0.01%] | 0.165 | 301 |
| `after_rare_seen_nonfiller` | empty | -0.0067 [-0.0282, +0.0071] | -0.43% [-1.97%, +0.47%] | 0.458 | 301 |
| `after_rare_seen_nonfiller` | mean | -0.0122 [-0.0369, +0.0039] | -0.78% [-2.58%, +0.26%] | 0.356 | 301 |
| `after` | other | -0.0038 [-0.0072, -0.0007] | -0.23% [-0.43%, -0.04%] | 0.0424* | 2,130 |
| `after` | other-any | -0.0032 [-0.0062, -0.0004] | -0.20% [-0.37%, -0.03%] | 0.048* | 2,130 |
| `after` | empty | -0.0016 [-0.0039, +0.0006] | -0.10% [-0.23%, +0.04%] | 0.183 | 2,130 |
| `after` | mean | -0.0026 [-0.0056, +0.0002] | -0.16% [-0.33%, +0.01%] | 0.138 | 2,130 |
| `unlinked` | other | -0.0001 [-0.0003, +0.0002] | -0.00% [-0.01%, +0.01%] | 1 | 5,465 |
| `unlinked` | other-any | +0.0002 [-0.0005, +0.0011] | +0.01% [-0.02%, +0.05%] | 1 | 5,465 |
| `unlinked` | empty | +0.0001 [-0.0002, +0.0005] | +0.01% [-0.01%, +0.02%] | 1 | 5,465 |
| `unlinked` | mean | +0.0002 [-0.0001, +0.0005] | +0.01% [-0.01%, +0.02%] | 1 | 5,465 |
| `all` | other | -0.0010 [-0.0019, -0.0002] | -0.05% [-0.10%, -0.01%] | 0.0264* | 8,184 |
| `all` | other-any | -0.0006 [-0.0014, +0.0001] | -0.03% [-0.08%, +0.00%] | 0.322 | 8,184 |
| `all` | empty | -0.0003 [-0.0009, +0.0004] | -0.01% [-0.05%, +0.02%] | 0.45 | 8,184 |
| `all` | mean | -0.0005 [-0.0013, +0.0002] | -0.03% [-0.07%, +0.01%] | 0.399 | 8,184 |

Per seed (`after_rare_seen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 1.5882 | -1.15% [-2.32%, -0.20%] | -1.01% [-2.53%, +0.10%] | -0.50% [-1.77%, +0.42%] | -0.92% [-2.37%, +0.13%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 3.23 (counts equal).

