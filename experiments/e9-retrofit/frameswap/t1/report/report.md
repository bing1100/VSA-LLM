# E9 frame swap — is the own frame used?

Evaluation-time frame swap (`e9_frameswap`; pre-registration `experiments/e9-retrofit/preregistration-frameswap.md`). Each row is `variant − own` within the same trained model on its evaluation windows (positive = the variant is worse, i.e. the run's own rows help): Δ in nats per target token and relative to `own`'s loss, token-weighted and pooled over seeds per window, with 95% cluster-bootstrap intervals over windows; Holm over the four variants within a target set × stratum (* = Holm p < 0.05). Variants: `other` = another target entry's frame (derangement within the set), `other-any` = a random non-target entry's frame, `empty` = no injection, `mean` = the mean non-target row.

**Reading rule:** a held-out gain counts as ontology-specific only if `other − own` > 0 on `after_heldout` with Holm p < 0.05.

## t1 · SmolLM2-135M · full · C5

Seeds: 1 (single seed: intervals cover windows only).

**Primary** (`other − own`, `after_heldout`): Δ +0.0005 [-0.0003, +0.0014], +0.02% [-0.01%, +0.05%], Holm p 0.357 → not shown ontology-specific.

### Target set `heldout` (matched stratum `after_heldout`)

1976 target entries (0 dropped: empty frame); windows holding a target span per seed: 1750.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_heldout` | other | +0.0005 [-0.0003, +0.0014] | +0.02% [-0.01%, +0.05%] | 0.357 | 50,222 |
| `after_heldout` | other-any | +0.0001 [-0.0007, +0.0010] | +0.00% [-0.03%, +0.04%] | 0.778 | 50,222 |
| `after_heldout` | empty | -0.0010 [-0.0017, -0.0003] | -0.04% [-0.07%, -0.01%] | 0.012* | 50,222 |
| `after_heldout` | mean | -0.0012 [-0.0019, -0.0005] | -0.05% [-0.08%, -0.02%] | 0.004* | 50,222 |
| `after` | other | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.01%, +0.00%] | 1 | 508,381 |
| `after` | other-any | -0.0000 [-0.0002, +0.0001] | -0.00% [-0.01%, +0.00%] | 1 | 508,381 |
| `after` | empty | -0.0002 [-0.0003, -0.0001] | -0.01% [-0.01%, -0.00%] | 0.0132* | 508,381 |
| `after` | mean | -0.0002 [-0.0003, -0.0001] | -0.01% [-0.01%, -0.00%] | 0.0032* | 508,381 |
| `unlinked` | other | +0.0001 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.344 | 1,504,189 |
| `unlinked` | other-any | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | empty | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | mean | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `all` | other | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.478 | 2,095,104 |
| `all` | other-any | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 0.897 | 2,095,104 |
| `all` | empty | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.206 | 2,095,104 |
| `all` | mean | -0.0001 [-0.0001, -0.0000] | -0.00% [-0.01%, -0.00%] | 0.0216* | 2,095,104 |

Per seed (`after_heldout`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.4618 | +0.02% [-0.01%, +0.05%] | +0.00% [-0.03%, +0.04%] | -0.04% [-0.07%, -0.01%] | -0.05% [-0.08%, -0.02%] |

### Target set `unseen` (matched stratum `after_unseen`)

6157 target entries (0 dropped: empty frame); windows holding a target span per seed: 58.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_unseen` | other | +0.0038 [-0.0085, +0.0215] | +0.14% [-0.32%, +0.79%] | 1 | 664 |
| `after_unseen` | other-any | +0.0092 [-0.0024, +0.0255] | +0.34% [-0.09%, +0.92%] | 0.582 | 664 |
| `after_unseen` | empty | +0.0052 [-0.0045, +0.0200] | +0.19% [-0.17%, +0.73%] | 1 | 664 |
| `after_unseen` | mean | +0.0034 [-0.0059, +0.0183] | +0.13% [-0.23%, +0.66%] | 1 | 664 |
| `after` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | mean | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `unlinked` | other | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | other-any | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `all` | other | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | other-any | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | mean | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |

Per seed (`after_unseen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.6805 | +0.14% [-0.32%, +0.79%] | +0.34% [-0.09%, +0.92%] | +0.19% [-0.17%, +0.73%] | +0.13% [-0.23%, +0.66%] |

### Target set `rare_seen` (matched stratum `after_rare_seen`)

7395 target entries (0 dropped: empty frame); windows holding a target span per seed: 447.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_rare_seen` | other | +0.0013 [-0.0013, +0.0037] | +0.05% [-0.05%, +0.15%] | 0.968 | 6,341 |
| `after_rare_seen` | other-any | +0.0019 [-0.0010, +0.0049] | +0.08% [-0.04%, +0.19%] | 0.828 | 6,341 |
| `after_rare_seen` | empty | +0.0002 [-0.0017, +0.0021] | +0.01% [-0.07%, +0.08%] | 1 | 6,341 |
| `after_rare_seen` | mean | +0.0003 [-0.0017, +0.0024] | +0.01% [-0.07%, +0.09%] | 1 | 6,341 |
| `after` | other | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | other-any | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | empty | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | mean | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `unlinked` | other | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | mean | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `all` | other | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | mean | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |

Per seed (`after_rare_seen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.5204 | +0.05% [-0.05%, +0.15%] | +0.08% [-0.04%, +0.19%] | +0.01% [-0.07%, +0.08%] | +0.01% [-0.07%, +0.09%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 4.24 (counts equal).

## t1 · SmolLM2-360M · full · C5

Seeds: 1 (single seed: intervals cover windows only).

**Primary** (`other − own`, `after_heldout`): Δ +0.0000 [-0.0006, +0.0006], +0.00% [-0.03%, +0.03%], Holm p 1 → not shown ontology-specific.

### Target set `heldout` (matched stratum `after_heldout`)

1976 target entries (0 dropped: empty frame); windows holding a target span per seed: 1750.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_heldout` | other | +0.0000 [-0.0006, +0.0006] | +0.00% [-0.03%, +0.03%] | 1 | 50,222 |
| `after_heldout` | other-any | -0.0000 [-0.0006, +0.0006] | -0.00% [-0.03%, +0.03%] | 1 | 50,222 |
| `after_heldout` | empty | -0.0006 [-0.0011, -0.0001] | -0.03% [-0.05%, -0.00%] | 0.122 | 50,222 |
| `after_heldout` | mean | -0.0003 [-0.0008, +0.0001] | -0.02% [-0.04%, +0.01%] | 0.545 | 50,222 |
| `after` | other | +0.0001 [-0.0000, +0.0002] | +0.00% [-0.00%, +0.01%] | 1 | 508,381 |
| `after` | other-any | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.01%] | 1 | 508,381 |
| `after` | empty | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.01%, +0.00%] | 1 | 508,381 |
| `after` | mean | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `unlinked` | other | -0.0001 [-0.0001, -0.0000] | -0.00% [-0.00%, -0.00%] | 0.129 | 1,504,189 |
| `unlinked` | other-any | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.761 | 1,504,189 |
| `unlinked` | empty | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.407 | 1,504,189 |
| `unlinked` | mean | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.407 | 1,504,189 |
| `all` | other | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.558 | 2,095,104 |
| `all` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 0.872 | 2,095,104 |
| `all` | empty | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.558 | 2,095,104 |
| `all` | mean | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.558 | 2,095,104 |

Per seed (`after_heldout`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.2231 | +0.00% [-0.03%, +0.03%] | -0.00% [-0.03%, +0.03%] | -0.03% [-0.05%, -0.00%] | -0.02% [-0.04%, +0.01%] |

### Target set `unseen` (matched stratum `after_unseen`)

6157 target entries (0 dropped: empty frame); windows holding a target span per seed: 58.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_unseen` | other | +0.0004 [-0.0063, +0.0067] | +0.02% [-0.26%, +0.29%] | 1 | 664 |
| `after_unseen` | other-any | -0.0001 [-0.0047, +0.0043] | -0.01% [-0.20%, +0.19%] | 1 | 664 |
| `after_unseen` | empty | +0.0001 [-0.0045, +0.0048] | +0.00% [-0.19%, +0.20%] | 1 | 664 |
| `after_unseen` | mean | -0.0004 [-0.0051, +0.0040] | -0.02% [-0.22%, +0.17%] | 1 | 664 |
| `after` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.699 | 508,381 |
| `unlinked` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | other-any | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.777 | 1,504,189 |
| `unlinked` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `all` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | other-any | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.25 | 2,095,104 |
| `all` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.878 | 2,095,104 |

Per seed (`after_unseen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.3468 | +0.02% [-0.26%, +0.29%] | -0.01% [-0.20%, +0.19%] | +0.00% [-0.19%, +0.20%] | -0.02% [-0.22%, +0.17%] |

### Target set `rare_seen` (matched stratum `after_rare_seen`)

7395 target entries (0 dropped: empty frame); windows holding a target span per seed: 447.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_rare_seen` | other | +0.0006 [-0.0012, +0.0024] | +0.03% [-0.05%, +0.11%] | 0.786 | 6,341 |
| `after_rare_seen` | other-any | +0.0019 [+0.0001, +0.0036] | +0.08% [+0.00%, +0.16%] | 0.151 | 6,341 |
| `after_rare_seen` | empty | -0.0006 [-0.0021, +0.0008] | -0.03% [-0.10%, +0.04%] | 0.786 | 6,341 |
| `after_rare_seen` | mean | -0.0009 [-0.0024, +0.0006] | -0.04% [-0.11%, +0.03%] | 0.746 | 6,341 |
| `after` | other | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | other-any | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.678 | 508,381 |
| `after` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `after` | mean | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 508,381 |
| `unlinked` | other | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 0.806 | 1,504,189 |
| `unlinked` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `unlinked` | mean | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,504,189 |
| `all` | other | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 0.769 | 2,095,104 |
| `all` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 0.43 | 2,095,104 |
| `all` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |
| `all` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 2,095,104 |

Per seed (`after_rare_seen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.2392 | +0.03% [-0.05%, +0.11%] | +0.08% [+0.00%, +0.16%] | -0.03% [-0.10%, +0.04%] | -0.04% [-0.11%, +0.03%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 3.3 (counts equal).

