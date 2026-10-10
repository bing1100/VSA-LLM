# E9 frame swap — is the own frame used?

Evaluation-time frame swap (`e9_frameswap`; pre-registration `experiments/e9-retrofit/preregistration-frameswap.md`). Each row is `variant − own` within the same trained model on its evaluation windows (positive = the variant is worse, i.e. the run's own rows help): Δ in nats per target token and relative to `own`'s loss, token-weighted and pooled over seeds per window, with 95% cluster-bootstrap intervals over windows; Holm over the four variants within a target set × stratum (* = Holm p < 0.05). Variants: `other` = another target entry's frame (derangement within the set), `other-any` = a random non-target entry's frame, `empty` = no injection, `mean` = the mean non-target row.

**Reading rule:** a held-out gain counts as ontology-specific only if `other − own` > 0 on `after_heldout` with Holm p < 0.05.

## wordnet · SmolLM2-135M · full · C5

Seeds: 1 (single seed: intervals cover windows only).

**Primary** (`other − own`, `after_heldout`): Δ -0.0001 [-0.0006, +0.0004], -0.00% [-0.02%, +0.02%], Holm p 1 → not shown ontology-specific.

### Target set `heldout` (matched stratum `after_heldout`)

5900 target entries (0 dropped: empty frame); windows holding a target span per seed: 1008.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_heldout` | other | -0.0001 [-0.0006, +0.0004] | -0.00% [-0.02%, +0.02%] | 1 | 49,903 |
| `after_heldout` | other-any | +0.0000 [-0.0005, +0.0005] | +0.00% [-0.02%, +0.02%] | 1 | 49,903 |
| `after_heldout` | empty | +0.0002 [-0.0002, +0.0007] | +0.01% [-0.01%, +0.02%] | 1 | 49,903 |
| `after_heldout` | mean | +0.0001 [-0.0004, +0.0006] | +0.00% [-0.01%, +0.02%] | 1 | 49,903 |
| `after` | other | +0.0000 [-0.0001, +0.0002] | +0.00% [-0.00%, +0.01%] | 1 | 381,901 |
| `after` | other-any | -0.0000 [-0.0002, +0.0001] | -0.00% [-0.01%, +0.00%] | 1 | 381,901 |
| `after` | empty | -0.0001 [-0.0002, +0.0001] | -0.00% [-0.01%, +0.00%] | 1 | 381,901 |
| `after` | mean | +0.0001 [-0.0001, +0.0002] | +0.00% [-0.00%, +0.01%] | 1 | 381,901 |
| `unlinked` | other | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | other-any | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | empty | +0.0001 [-0.0000, +0.0002] | +0.00% [-0.00%, +0.01%] | 0.603 | 624,859 |
| `unlinked` | mean | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `all` | other | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | other-any | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | empty | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | mean | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |

Per seed (`after_heldout`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.7506 | -0.00% [-0.02%, +0.02%] | +0.00% [-0.02%, +0.02%] | +0.01% [-0.01%, +0.02%] | +0.00% [-0.01%, +0.02%] |

### Target set `unseen` (matched stratum `after_unseen`)

20549 target entries (0 dropped: empty frame); windows holding a target span per seed: 71.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_unseen` | other | +0.0008 [-0.0043, +0.0055] | +0.03% [-0.14%, +0.19%] | 1 | 836 |
| `after_unseen` | other-any | +0.0024 [-0.0028, +0.0074] | +0.08% [-0.09%, +0.26%] | 1 | 836 |
| `after_unseen` | empty | -0.0001 [-0.0049, +0.0044] | -0.00% [-0.17%, +0.15%] | 1 | 836 |
| `after_unseen` | mean | -0.0017 [-0.0067, +0.0026] | -0.06% [-0.23%, +0.09%] | 1 | 836 |
| `after` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `unlinked` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.735 | 624,859 |
| `unlinked` | other-any | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.735 | 624,859 |
| `unlinked` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.35 | 624,859 |
| `unlinked` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.56 | 624,859 |
| `all` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | other-any | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |

Per seed (`after_unseen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.9201 | +0.03% [-0.14%, +0.19%] | +0.08% [-0.09%, +0.26%] | -0.00% [-0.17%, +0.15%] | -0.06% [-0.23%, +0.09%] |

### Target set `rare_seen` (matched stratum `after_rare_seen`)

27961 target entries (0 dropped: empty frame); windows holding a target span per seed: 452.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_rare_seen` | other | +0.0012 [-0.0000, +0.0024] | +0.04% [-0.00%, +0.08%] | 0.18 | 7,742 |
| `after_rare_seen` | other-any | +0.0002 [-0.0011, +0.0015] | +0.01% [-0.04%, +0.05%] | 0.763 | 7,742 |
| `after_rare_seen` | empty | +0.0010 [-0.0001, +0.0023] | +0.04% [-0.01%, +0.08%] | 0.18 | 7,742 |
| `after_rare_seen` | mean | +0.0011 [+0.0000, +0.0023] | +0.04% [+0.00%, +0.08%] | 0.171 | 7,742 |
| `after` | other | +0.0001 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.485 | 381,901 |
| `after` | other-any | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.485 | 381,901 |
| `after` | empty | +0.0001 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.485 | 381,901 |
| `after` | mean | +0.0001 [+0.0000, +0.0002] | +0.00% [+0.00%, +0.01%] | 0.142 | 381,901 |
| `unlinked` | other | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.862 | 624,859 |
| `unlinked` | other-any | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.862 | 624,859 |
| `unlinked` | empty | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.862 | 624,859 |
| `unlinked` | mean | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.862 | 624,859 |
| `all` | other | +0.0000 [+0.0000, +0.0001] | +0.00% [+0.00%, +0.00%] | 0.2 | 1,047,552 |
| `all` | other-any | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.784 | 1,047,552 |
| `all` | empty | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.2 | 1,047,552 |
| `all` | mean | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 0.2 | 1,047,552 |

Per seed (`after_rare_seen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.8680 | +0.04% [-0.00%, +0.08%] | +0.01% [-0.04%, +0.05%] | +0.04% [-0.01%, +0.08%] | +0.04% [+0.00%, +0.08%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 3.45 (counts equal).

## wordnet · SmolLM2-360M · full · C5

Seeds: 1 (single seed: intervals cover windows only).

**Primary** (`other − own`, `after_heldout`): Δ +0.0002 [-0.0002, +0.0007], +0.01% [-0.01%, +0.03%], Holm p 0.715 → not shown ontology-specific.

### Target set `heldout` (matched stratum `after_heldout`)

5900 target entries (0 dropped: empty frame); windows holding a target span per seed: 1008.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_heldout` | other | +0.0002 [-0.0002, +0.0007] | +0.01% [-0.01%, +0.03%] | 0.715 | 49,903 |
| `after_heldout` | other-any | +0.0003 [-0.0002, +0.0007] | +0.01% [-0.01%, +0.03%] | 0.715 | 49,903 |
| `after_heldout` | empty | +0.0000 [-0.0004, +0.0005] | +0.00% [-0.01%, +0.02%] | 0.825 | 49,903 |
| `after_heldout` | mean | +0.0003 [-0.0001, +0.0007] | +0.01% [-0.00%, +0.03%] | 0.443 | 49,903 |
| `after` | other | +0.0000 [-0.0001, +0.0002] | +0.00% [-0.00%, +0.01%] | 1 | 381,901 |
| `after` | other-any | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.01%] | 1 | 381,901 |
| `after` | empty | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.01%] | 1 | 381,901 |
| `after` | mean | +0.0000 [-0.0001, +0.0002] | +0.00% [-0.00%, +0.01%] | 1 | 381,901 |
| `unlinked` | other | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | other-any | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | empty | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.01%, +0.00%] | 1 | 624,859 |
| `unlinked` | mean | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `all` | other | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | other-any | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | empty | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | mean | +0.0000 [-0.0000, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |

Per seed (`after_heldout`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.5260 | +0.01% [-0.01%, +0.03%] | +0.01% [-0.01%, +0.03%] | +0.00% [-0.01%, +0.02%] | +0.01% [-0.00%, +0.03%] |

### Target set `unseen` (matched stratum `after_unseen`)

20549 target entries (0 dropped: empty frame); windows holding a target span per seed: 71.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_unseen` | other | -0.0010 [-0.0044, +0.0024] | -0.04% [-0.16%, +0.09%] | 1 | 836 |
| `after_unseen` | other-any | -0.0011 [-0.0041, +0.0020] | -0.04% [-0.16%, +0.08%] | 1 | 836 |
| `after_unseen` | empty | -0.0014 [-0.0047, +0.0020] | -0.05% [-0.18%, +0.08%] | 1 | 836 |
| `after_unseen` | mean | -0.0033 [-0.0063, -0.0005] | -0.13% [-0.23%, -0.02%] | 0.0968 | 836 |
| `after` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `unlinked` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `all` | other | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | other-any | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | empty | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.863 | 1,047,552 |
| `all` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |

Per seed (`after_unseen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.6399 | -0.04% [-0.16%, +0.09%] | -0.04% [-0.16%, +0.08%] | -0.05% [-0.18%, +0.08%] | -0.13% [-0.23%, -0.02%] |

### Target set `rare_seen` (matched stratum `after_rare_seen`)

27961 target entries (0 dropped: empty frame); windows holding a target span per seed: 452.

| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |
|---|---|---|---|---|---:|
| `after_rare_seen` | other | +0.0009 [-0.0002, +0.0019] | +0.03% [-0.01%, +0.07%] | 0.364 | 7,742 |
| `after_rare_seen` | other-any | +0.0010 [+0.0000, +0.0020] | +0.04% [+0.00%, +0.08%] | 0.156 | 7,742 |
| `after_rare_seen` | empty | +0.0003 [-0.0007, +0.0013] | +0.01% [-0.03%, +0.05%] | 0.652 | 7,742 |
| `after_rare_seen` | mean | +0.0005 [-0.0005, +0.0015] | +0.02% [-0.02%, +0.06%] | 0.652 | 7,742 |
| `after` | other | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | other-any | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | empty | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `after` | mean | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 381,901 |
| `unlinked` | other | -0.0001 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.263 | 624,859 |
| `unlinked` | other-any | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | empty | -0.0000 [-0.0001, +0.0001] | -0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `unlinked` | mean | +0.0000 [-0.0001, +0.0001] | +0.00% [-0.00%, +0.00%] | 1 | 624,859 |
| `all` | other | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.251 | 1,047,552 |
| `all` | other-any | -0.0000 [-0.0001, +0.0000] | -0.00% [-0.00%, +0.00%] | 0.892 | 1,047,552 |
| `all` | empty | +0.0000 [-0.0000, +0.0000] | +0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |
| `all` | mean | -0.0000 [-0.0000, +0.0000] | -0.00% [-0.00%, +0.00%] | 1 | 1,047,552 |

Per seed (`after_rare_seen`):

| seed | own loss | other | other-any | empty | mean |
|---|---|---|---|---|---|
| 1 | 2.6154 | +0.03% [-0.01%, +0.07%] | +0.04% [+0.00%, +0.08%] | +0.01% [-0.03%, +0.05%] | +0.02% [-0.02%, +0.06%] |

`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed s1 3.17 (counts equal).

