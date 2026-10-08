# E12 T5 SmolLM2-360M seed 1 — Q1 and F1 pilot — PILOT

Pre-registration: `experiments/e12-self-query/preregistration.md`. Contrasts: units × seeds crossed model (Satterthwaite t, 95% CI, two-sided p; one seed: one-sample t), Holm within Q1 and within F1.

**PILOT: these numbers do not count toward the pre-registered endpoints.**

## SmolLM2-360M

Runs: {'P0': [1], 'C0p': [1], 'C5': [1], 'C5ut': [1]}

### Q1 — recall tool on role-swap twins (C5 host, contrast accuracy, choice)

| contrast | estimate |
|---|---|
| recall:own − none | +0.2825 [+0.2520, +0.3130] (p 1.6e-50, Holm 1.6e-50; 300 units × 1 seeds) |
| recall:own − recall:C5ut | +0.2950 [+0.2658, +0.3242] (p 1.33e-56, Holm 2.66e-56; 300 units × 1 seeds) |

Means: recall:own 0.790, none 0.507, recall:C5ut 0.495, symbolic 0.863; pairs fully decoded by the C5 recall: 0.857.

Reading: (a) the self-query reads the stored roles: recall from the bound store beats no recall and the role-blind store

### Twins by host × condition (choice)

| host model | condition | contrast | item | decode (filler) | all four slots | seeds |
|---|---|---:|---:|---:|---:|---|
| C0p | none | 0.488 | 0.507 | — | — | [1] |
| C0p | recall:C5 | 0.794 | 0.536 | 0.959 | 0.857 | [1] |
| C0p | symbolic | 0.868 | 0.540 | 1.000 | 1.000 | [1] |
| C0p | roleless:C5 | 0.498 | 0.495 | 0.959 | — | [1] |
| C5 | none | 0.507 | 0.506 | — | — | [1] |
| C5 | recall:own | 0.790 | 0.539 | 0.959 | 0.857 | [1] |
| C5 | recall:C5ut | 0.495 | 0.501 | 0.897 | — | [1] |
| C5 | symbolic | 0.863 | 0.546 | 1.000 | 1.000 | [1] |
| C5 | wrong:own | 0.228 | 0.462 | 0.000 | 0.000 | [1] |
| C5 | roleless:own | 0.497 | 0.498 | 0.959 | — | [1] |
| C5 | definition | 0.748 | 0.533 | — | — | [1] |
| C5 | recall:C5tr | 0.497 | 0.500 | 0.443 | 0.000 | [1] |
| C5ut | none | 0.500 | 0.501 | — | — | [1] |
| C5ut | recall:own | 0.485 | 0.499 | 0.897 | — | [1] |
| C5ut | recall:C5 | 0.798 | 0.545 | 0.959 | 0.857 | [1] |
| C5ut | symbolic | 0.864 | 0.548 | 1.000 | 1.000 | [1] |
| P0 | none | 0.492 | 0.496 | — | — | [1] |
| P0 | recall:C5@1 | 0.690 | 0.515 | 0.959 | 0.857 | [1] |
| P0 | symbolic | 0.730 | 0.515 | 1.000 | 1.000 | [1] |
| P0 | wrong:C5@1 | 0.311 | 0.485 | 0.000 | 0.000 | [1] |

### Twin secondaries

| contrast | estimate |
|---|---|
| C5: symbolic − recall:own | +0.0733 [+0.0536, +0.0931] (p 2.36e-12; 300 units × 1 seeds) |
| C5: definition − symbolic | -0.1158 [-0.1411, -0.0906] (p 2.3e-17; 300 units × 1 seeds) |
| C5: recall:own − roleless:own | +0.2933 [+0.2634, +0.3232] (p 1.59e-54; 300 units × 1 seeds) |
| C5: recall:own − wrong:own | +0.5625 [+0.5250, +0.6000] (p 1.1e-90; 300 units × 1 seeds) |
| C5: recall:own − recall:C5tr | +0.2933 [+0.2630, +0.3237] (p 1.86e-53; 300 units × 1 seeds) |
| C5ut: recall:own − none | -0.0150 [-0.0460, +0.0160] (p 0.342; 300 units × 1 seeds) |
| C5ut: recall:C5 − recall:own | +0.3133 [+0.2841, +0.3426] (p 3.98e-61; 300 units × 1 seeds) |
| C0p: recall:C5 − none | +0.3058 [+0.2761, +0.3356] (p 6.23e-58; 300 units × 1 seeds) |
| P0: recall:C5 − none | +0.1983 [+0.1672, +0.2295] (p 2.92e-29; 300 units × 1 seeds) |
| C0p: recall:C5 − roleless:C5 | +0.2958 [+0.2652, +0.3264] (p 1.91e-53; 300 units × 1 seeds) |
| C5: recall:own − 0.5 | +0.2900 [+0.2682, +0.3118] (p 2.7e-79; 300 units × 1 seeds) |
| C0p: recall:C5 − 0.5 | +0.2942 [+0.2724, +0.3159] (p 9.06e-81; 300 units × 1 seeds) |
| P0: recall:C5 − 0.5 | +0.1900 [+0.1682, +0.2118] (p 2.79e-46; 300 units × 1 seeds) |
| C5 recall:own − C0p recall:C5 (does the channel add to the tool?) | -0.0042 [-0.0189, +0.0106] (p 0.579; 300 units × 1 seeds) |
| C5 recall:own − P0 recall:C5 (does the channel add to the tool?) | +0.1000 [+0.0772, +0.1228] (p 3.31e-16; 300 units × 1 seeds) |
| C5 recall:own on pairs with decode_all = 1 | 0.838 (257 pairs) |
| C5 recall:own on pairs with decode_all = 0 | 0.506 (43 pairs) |

### F1 — faithfulness of decoding (channel route; new words; comprehensiveness net share, τ = 0.05)

| contrast | estimate |
|---|---|
| F1a: C5 − 0 | +0.3419 [+0.2994, +0.3845] (p 2.57e-41, Holm 5.15e-41; 300 units × 1 seeds) |
| F1b: C5 − C5ut | -0.0503 [-0.0983, -0.0024] (p 0.0397, Holm 0.0397; 300 units × 1 seeds) |

| secondary | estimate |
|---|---|
| C5 moved | +0.5993 [+0.5745, +0.6241] (p 2.14e-141; 300 units × 1 seeds) |
| C5ut moved | +0.6395 [+0.6115, +0.6675] (p 4.86e-135; 300 units × 1 seeds) |
| C5 sufficiency | +0.2660 [+0.2207, +0.3113] (p 8.66e-26; 300 units × 1 seeds) |
| C5ut sufficiency | +0.2986 [+0.2538, +0.3435] (p 2.51e-31; 300 units × 1 seeds) |
| C5 specificity | +0.2217 [+0.1996, +0.2437] (p 2.81e-56; 300 units × 1 seeds) |
| C5ut specificity | +0.2391 [+0.2159, +0.2623] (p 3.36e-58; 300 units × 1 seeds) |
| C5 gap | +0.3752 [+0.3177, +0.4327] (p 2.27e-30; 300 units × 1 seeds) |
| C5ut gap | +0.5379 [+0.4664, +0.6094] (p 1.43e-37; 300 units × 1 seeds) |
| C5 comprehensiveness_tau0 | +0.3636 [+0.3167, +0.4105] (p 2.72e-39; 300 units × 1 seeds) |
| C5ut comprehensiveness_tau0 | +0.4047 [+0.3562, +0.4533] (p 1.56e-43; 300 units × 1 seeds) |
| C5 comprehensiveness_tau0.02 | +0.3537 [+0.3083, +0.3992] (p 1.75e-39; 300 units × 1 seeds) |
| C5ut comprehensiveness_tau0.02 | +0.4020 [+0.3541, +0.4499] (p 5.07e-44; 300 units × 1 seeds) |
| C5 comprehensiveness_tau0.1 | +0.3220 [+0.2811, +0.3630] (p 4.28e-40; 300 units × 1 seeds) |
| C5ut comprehensiveness_tau0.1 | +0.3826 [+0.3360, +0.4292] (p 1.1e-42; 300 units × 1 seeds) |
| C5 comprehensiveness, undecoded edges | +0.0690 [-0.2306, +0.3685] (p 0.641; 29 units × 1 seeds) |
| C5 role specificity (twins) | -0.0358 [-0.0912, +0.0196] (p 0.204; 300 units × 1 seeds) |
| C5ut role specificity (twins) | +0.0083 [-0.0468, +0.0634] (p 0.766; 300 units × 1 seeds) |
| role specificity C5 − C5ut | -0.0442 [-0.1204, +0.0320] (p 0.255; 300 units × 1 seeds) |

Reading: {'F1a': '(a) decoded edges are causally used: removing one moves its filler more than removing another edge', 'F1b': 'binding changes edge-level faithfulness (C5 lower)'}

### F1 by model (seed-averaged term means; twins: role specificity per pair)

| model | comprehensiveness | moved | sufficiency | specificity | gap (nats) | decoded edges | role specificity | RS (nats) | seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C5 | +0.342 | 0.599 | +0.266 | 0.222 | +0.375 | 1497.000 | -0.036 | -0.021 | [1] |
| C5ut | +0.392 | 0.640 | +0.299 | 0.239 | +0.538 | 1470.000 | +0.008 | +0.001 | [1] |

