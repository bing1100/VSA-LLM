# E14 RD-4 / RD-7 — pooled report (item differences of every seed pooled)

## t4 heldout — SmolLM2-360M-full-C0p (1 seeds)

| condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| definition | 0.7248 | 0.5946 | 0.8280 |
| explicit-sym:linker | 0.5291 | 0.3808 | 0.6708 |
| explicit-sym:oracle | 0.4930 | 0.6339 | 0.8649 |
| explicit-sym:stated | 0.5610 | 0.3489 | 0.6536 |
| explicit-sym:typeprior | 0.5618 | 0.3686 | 0.6904 |
| noisy-explicit:0.1 | 0.4816 | 0.6290 | 0.8673 |
| noisy-explicit:0.3 | 0.4505 | 0.5971 | 0.8477 |
| noisy-explicit:0.5 | 0.4103 | 0.6020 | 0.8329 |
| none | 0.6175 | 0.5405 | 0.7666 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.1245 [-0.1720, -0.0745] (p 0.0005, Holm 0.0015) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.0934 [+0.0246, +0.1622] (p 0.0090, Holm 0.0090) | 407 |
| explicit-sym:oracle − none | entailment | +0.0983 [+0.0516, +0.1474] (p 0.0005, Holm 0.0015) | 407 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:linker − none | property | -0.0885 [-0.1278, -0.0483] (p 0.0005) | 407 |
| explicit-sym:linker − none | paraphrase | -0.1597 [-0.2187, -0.0958] (p 0.0005) | 407 |
| explicit-sym:linker − none | entailment | -0.0958 [-0.1474, -0.0467] (p 0.0015) | 407 |
| explicit-sym:typeprior − none | property | -0.0557 [-0.0934, -0.0172] (p 0.0050) | 407 |
| explicit-sym:typeprior − none | paraphrase | -0.1720 [-0.2310, -0.1106] (p 0.0005) | 407 |
| explicit-sym:typeprior − none | entailment | -0.0762 [-0.1302, -0.0270] (p 0.0030) | 407 |
| explicit-sym:stated − none | property | -0.0565 [-0.0942, -0.0180] (p 0.0060) | 407 |
| explicit-sym:stated − none | paraphrase | -0.1916 [-0.2482, -0.1351] (p 0.0005) | 407 |
| explicit-sym:stated − none | entailment | -0.1130 [-0.1671, -0.0614] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | property | +0.2318 [+0.1867, +0.2776] (p 0.0005) | 407 |
| definition − explicit-sym:oracle | paraphrase | -0.0393 [-0.1057, +0.0270] (p 0.2584) | 407 |
| definition − explicit-sym:oracle | entailment | -0.0369 [-0.0860, +0.0123] (p 0.1615) | 407 |
| definition − none | property | +0.1073 [+0.0721, +0.1450] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.0541 [-0.0025, +0.1130] (p 0.0670) | 407 |
| definition − none | entailment | +0.0614 [+0.0147, +0.1081] (p 0.0125) | 407 |

## t4 heldout — SmolLM2-360M-full-C5 (3 seeds)

| condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| both:oracle | 0.5130 | 0.6683 | 0.8731 |
| definition | 0.7289 | 0.6003 | 0.8305 |
| explicit-sym:linker | 0.5315 | 0.3563 | 0.6757 |
| explicit-sym:oracle | 0.5113 | 0.6642 | 0.8763 |
| explicit-sym:stated | 0.5550 | 0.3669 | 0.6724 |
| explicit-sym:typeprior | 0.5607 | 0.3620 | 0.7109 |
| explicit-vsa:linker | 0.5089 | 0.3251 | 0.6380 |
| explicit-vsa:oracle | 0.5168 | 0.6577 | 0.8862 |
| explicit-vsa:stated | 0.5471 | 0.3571 | 0.6429 |
| implicit:linker | 0.6197 | 0.5471 | 0.7748 |
| implicit:oracle | 0.6164 | 0.5274 | 0.7666 |
| implicit:stated | 0.6153 | 0.5397 | 0.7641 |
| implicit:typeprior | 0.6134 | 0.5340 | 0.7690 |
| noisy-explicit:0.1 | 0.4936 | 0.6511 | 0.8788 |
| noisy-explicit:0.3 | 0.4638 | 0.6069 | 0.8436 |
| noisy-explicit:0.5 | 0.4144 | 0.5954 | 0.8305 |
| noisy-implicit:0.1 | 0.6159 | 0.5291 | 0.7633 |
| noisy-implicit:0.3 | 0.6110 | 0.5405 | 0.7625 |
| noisy-implicit:0.5 | 0.6148 | 0.5414 | 0.7682 |
| none | 0.6143 | 0.5438 | 0.7666 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.1029 [-0.1310, -0.0737] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:oracle − none | paraphrase | +0.1204 [+0.0843, +0.1581] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:oracle − none | entailment | +0.1097 [+0.0819, +0.1368] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:oracle − implicit:oracle | property | -0.1051 [-0.1335, -0.0773] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.1368 [+0.0991, +0.1736] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1097 [+0.0827, +0.1368] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:stated − implicit:stated | property | -0.0603 [-0.0819, -0.0388] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:stated − implicit:stated | paraphrase | -0.1728 [-0.2056, -0.1392] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-sym:stated − implicit:stated | entailment | -0.0917 [-0.1212, -0.0622] (p 0.0005, Holm 0.0045) | 1221 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0022 [-0.0044, +0.0085] (p 0.5199) | 1221 |
| implicit:oracle − none | paraphrase | -0.0164 [-0.0303, -0.0033] (p 0.0170) | 1221 |
| implicit:oracle − none | entailment | +0.0000 [-0.0082, +0.0082] (p 1.0000) | 1221 |
| implicit:linker − none | property | +0.0055 [-0.0003, +0.0115] (p 0.0630) | 1221 |
| implicit:linker − none | paraphrase | +0.0033 [-0.0098, +0.0164] (p 0.6758) | 1221 |
| implicit:linker − none | entailment | +0.0082 [+0.0000, +0.0164] (p 0.0710) | 1221 |
| explicit-sym:linker − none | property | -0.0827 [-0.1051, -0.0606] (p 0.0005) | 1221 |
| explicit-sym:linker − none | paraphrase | -0.1876 [-0.2211, -0.1540] (p 0.0005) | 1221 |
| explicit-sym:linker − none | entailment | -0.0909 [-0.1204, -0.0614] (p 0.0005) | 1221 |
| explicit-sym:typeprior − none | property | -0.0535 [-0.0753, -0.0317] (p 0.0005) | 1221 |
| explicit-sym:typeprior − none | paraphrase | -0.1818 [-0.2170, -0.1458] (p 0.0005) | 1221 |
| explicit-sym:typeprior − none | entailment | -0.0557 [-0.0844, -0.0262] (p 0.0005) | 1221 |
| explicit-sym:stated − none | property | -0.0592 [-0.0808, -0.0374] (p 0.0005) | 1221 |
| explicit-sym:stated − none | paraphrase | -0.1769 [-0.2113, -0.1433] (p 0.0005) | 1221 |
| explicit-sym:stated − none | entailment | -0.0942 [-0.1229, -0.0639] (p 0.0005) | 1221 |
| explicit-vsa:oracle − explicit-sym:oracle | property | +0.0055 [-0.0019, +0.0131] (p 0.1590) | 1221 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | -0.0066 [-0.0230, +0.0098] (p 0.4679) | 1221 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | +0.0098 [-0.0016, +0.0221] (p 0.0900) | 1221 |
| explicit-vsa:oracle − none | property | -0.0975 [-0.1253, -0.0693] (p 0.0005) | 1221 |
| explicit-vsa:oracle − none | paraphrase | +0.1138 [+0.0762, +0.1523] (p 0.0005) | 1221 |
| explicit-vsa:oracle − none | entailment | +0.1196 [+0.0925, +0.1466] (p 0.0005) | 1221 |
| explicit-vsa:linker − none | property | -0.1054 [-0.1280, -0.0822] (p 0.0005) | 1221 |
| explicit-vsa:linker − none | paraphrase | -0.2187 [-0.2523, -0.1835] (p 0.0005) | 1221 |
| explicit-vsa:linker − none | entailment | -0.1286 [-0.1597, -0.0991] (p 0.0005) | 1221 |
| both:oracle − explicit-sym:oracle | property | +0.0016 [-0.0038, +0.0068] (p 0.5674) | 1221 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0041 [-0.0082, +0.0164] (p 0.5704) | 1221 |
| both:oracle − explicit-sym:oracle | entailment | -0.0033 [-0.0098, +0.0033] (p 0.3799) | 1221 |
| both:oracle − implicit:oracle | property | -0.1035 [-0.1321, -0.0756] (p 0.0005) | 1221 |
| both:oracle − implicit:oracle | paraphrase | +0.1409 [+0.1032, +0.1777] (p 0.0005) | 1221 |
| both:oracle − implicit:oracle | entailment | +0.1065 [+0.0794, +0.1335] (p 0.0005) | 1221 |
| definition − explicit-sym:oracle | property | +0.2176 [+0.1906, +0.2432] (p 0.0005) | 1221 |
| definition − explicit-sym:oracle | paraphrase | -0.0639 [-0.1024, -0.0270] (p 0.0005) | 1221 |
| definition − explicit-sym:oracle | entailment | -0.0459 [-0.0729, -0.0172] (p 0.0025) | 1221 |
| definition − none | property | +0.1147 [+0.0931, +0.1360] (p 0.0005) | 1221 |
| definition − none | paraphrase | +0.0565 [+0.0254, +0.0885] (p 0.0010) | 1221 |
| definition − none | entailment | +0.0639 [+0.0369, +0.0909] (p 0.0005) | 1221 |
| explicit-sym:linker − implicit:linker | property | -0.0882 [-0.1103, -0.0666] (p 0.0005) | 1221 |
| explicit-sym:linker − implicit:linker | paraphrase | -0.1908 [-0.2252, -0.1581] (p 0.0005) | 1221 |
| explicit-sym:linker − implicit:linker | entailment | -0.0991 [-0.1278, -0.0688] (p 0.0005) | 1221 |
| explicit-sym:typeprior − implicit:typeprior | property | -0.0527 [-0.0745, -0.0308] (p 0.0005) | 1221 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | -0.1720 [-0.2072, -0.1360] (p 0.0005) | 1221 |
| explicit-sym:typeprior − implicit:typeprior | entailment | -0.0581 [-0.0868, -0.0287] (p 0.0005) | 1221 |

## t5 new — SmolLM2-360M-full-C0p (1 seeds)

| condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| definition | 0.6655 | 0.5618 | 0.7283 |
| explicit-sym:linker | 0.3669 | 0.4402 | 0.5750 |
| explicit-sym:oracle | 0.5242 | 0.6351 | 0.6667 |
| explicit-sym:stated | 0.5242 | 0.6351 | 0.6667 |
| explicit-sym:typeprior | 0.5343 | 0.6383 | 0.6717 |
| noisy-explicit:0.1 | 0.4912 | 0.6220 | 0.6483 |
| noisy-explicit:0.3 | 0.4196 | 0.6043 | 0.6067 |
| noisy-explicit:0.5 | 0.3447 | 0.5749 | 0.5617 |
| none | 0.1884 | 0.3218 | 0.5100 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.3358 [+0.3100, +0.3620] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.3133 [+0.2799, +0.3460] (p 0.0005, Holm 0.0015) | 1529 |
| explicit-sym:oracle − none | entailment | +0.1567 [+0.1067, +0.2067] (p 0.0005, Holm 0.0015) | 600 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:linker − none | property | +0.1785 [+0.1550, +0.2021] (p 0.0005) | 1529 |
| explicit-sym:linker − none | paraphrase | +0.1184 [+0.0837, +0.1524] (p 0.0005) | 1529 |
| explicit-sym:linker − none | entailment | +0.0650 [+0.0117, +0.1183] (p 0.0170) | 600 |
| explicit-sym:typeprior − none | property | +0.3460 [+0.3205, +0.3718] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | paraphrase | +0.3165 [+0.2838, +0.3492] (p 0.0005) | 1529 |
| explicit-sym:typeprior − none | entailment | +0.1617 [+0.1117, +0.2150] (p 0.0005) | 600 |
| explicit-sym:stated − none | property | +0.3358 [+0.3100, +0.3620] (p 0.0005) | 1529 |
| explicit-sym:stated − none | paraphrase | +0.3133 [+0.2799, +0.3460] (p 0.0005) | 1529 |
| explicit-sym:stated − none | entailment | +0.1567 [+0.1067, +0.2067] (p 0.0005) | 600 |
| definition − explicit-sym:oracle | property | +0.1413 [+0.1220, +0.1602] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | paraphrase | -0.0733 [-0.1001, -0.0458] (p 0.0005) | 1529 |
| definition − explicit-sym:oracle | entailment | +0.0617 [+0.0266, +0.0983] (p 0.0025) | 600 |
| definition − none | property | +0.4771 [+0.4539, +0.4997] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.2400 [+0.2060, +0.2740] (p 0.0005) | 1529 |
| definition − none | entailment | +0.2183 [+0.1683, +0.2683] (p 0.0005) | 600 |

## t5 new — SmolLM2-360M-full-C5 (3 seeds)

| condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| both:oracle | 0.5295 | 0.6433 | 0.6789 |
| definition | 0.6421 | 0.5193 | 0.7267 |
| explicit-sym:linker | 0.3586 | 0.4332 | 0.5600 |
| explicit-sym:oracle | 0.5016 | 0.6098 | 0.6550 |
| explicit-sym:stated | 0.5016 | 0.6098 | 0.6550 |
| explicit-sym:typeprior | 0.5126 | 0.5991 | 0.6617 |
| explicit-vsa:linker | 0.3368 | 0.4334 | 0.5611 |
| explicit-vsa:oracle | 0.4989 | 0.6095 | 0.6533 |
| explicit-vsa:stated | 0.4989 | 0.6095 | 0.6533 |
| implicit:linker | 0.2218 | 0.3745 | 0.5228 |
| implicit:oracle | 0.2603 | 0.4079 | 0.5350 |
| implicit:stated | 0.2603 | 0.4079 | 0.5350 |
| implicit:typeprior | 0.2602 | 0.4022 | 0.5350 |
| noisy-explicit:0.1 | 0.4697 | 0.6008 | 0.6311 |
| noisy-explicit:0.3 | 0.4082 | 0.5908 | 0.5939 |
| noisy-explicit:0.5 | 0.3355 | 0.5404 | 0.5400 |
| noisy-implicit:0.1 | 0.2516 | 0.4088 | 0.5317 |
| noisy-implicit:0.3 | 0.2384 | 0.3966 | 0.5206 |
| noisy-implicit:0.5 | 0.2271 | 0.3966 | 0.5139 |
| none | 0.1896 | 0.3442 | 0.5156 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.3121 [+0.2966, +0.3271] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-sym:oracle − none | paraphrase | +0.2655 [+0.2459, +0.2847] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-sym:oracle − none | entailment | +0.1394 [+0.1089, +0.1689] (p 0.0005, Holm 0.0045) | 1800 |
| explicit-sym:oracle − implicit:oracle | property | +0.2413 [+0.2252, +0.2573] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-sym:oracle − implicit:oracle | paraphrase | +0.2019 [+0.1816, +0.2208] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-sym:oracle − implicit:oracle | entailment | +0.1200 [+0.0889, +0.1494] (p 0.0005, Holm 0.0045) | 1800 |
| explicit-sym:stated − implicit:stated | property | +0.2413 [+0.2252, +0.2573] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-sym:stated − implicit:stated | paraphrase | +0.2019 [+0.1816, +0.2208] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-sym:stated − implicit:stated | entailment | +0.1200 [+0.0889, +0.1494] (p 0.0005, Holm 0.0045) | 1800 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| implicit:oracle − none | property | +0.0707 [+0.0618, +0.0792] (p 0.0005) | 4587 |
| implicit:oracle − none | paraphrase | +0.0637 [+0.0488, +0.0785] (p 0.0005) | 4587 |
| implicit:oracle − none | entailment | +0.0194 [+0.0011, +0.0383] (p 0.0420) | 1800 |
| implicit:linker − none | property | +0.0323 [+0.0256, +0.0390] (p 0.0005) | 4587 |
| implicit:linker − none | paraphrase | +0.0303 [+0.0179, +0.0425] (p 0.0005) | 4587 |
| implicit:linker − none | entailment | +0.0072 [-0.0083, +0.0228] (p 0.3964) | 1800 |
| explicit-sym:linker − none | property | +0.1691 [+0.1553, +0.1825] (p 0.0005) | 4587 |
| explicit-sym:linker − none | paraphrase | +0.0889 [+0.0691, +0.1077] (p 0.0005) | 4587 |
| explicit-sym:linker − none | entailment | +0.0444 [+0.0144, +0.0744] (p 0.0040) | 1800 |
| explicit-sym:typeprior − none | property | +0.3231 [+0.3077, +0.3374] (p 0.0005) | 4587 |
| explicit-sym:typeprior − none | paraphrase | +0.2549 [+0.2344, +0.2743] (p 0.0005) | 4587 |
| explicit-sym:typeprior − none | entailment | +0.1461 [+0.1167, +0.1750] (p 0.0005) | 1800 |
| explicit-sym:stated − none | property | +0.3121 [+0.2966, +0.3271] (p 0.0005) | 4587 |
| explicit-sym:stated − none | paraphrase | +0.2655 [+0.2459, +0.2847] (p 0.0005) | 4587 |
| explicit-sym:stated − none | entailment | +0.1394 [+0.1089, +0.1689] (p 0.0005) | 1800 |
| explicit-vsa:oracle − explicit-sym:oracle | property | -0.0027 [-0.0073, +0.0017] (p 0.2449) | 4587 |
| explicit-vsa:oracle − explicit-sym:oracle | paraphrase | -0.0002 [-0.0087, +0.0078] (p 0.9708) | 4587 |
| explicit-vsa:oracle − explicit-sym:oracle | entailment | -0.0017 [-0.0106, +0.0083] (p 0.7573) | 1800 |
| explicit-vsa:oracle − none | property | +0.3094 [+0.2939, +0.3243] (p 0.0005) | 4587 |
| explicit-vsa:oracle − none | paraphrase | +0.2653 [+0.2453, +0.2845] (p 0.0005) | 4587 |
| explicit-vsa:oracle − none | entailment | +0.1378 [+0.1083, +0.1672] (p 0.0005) | 1800 |
| explicit-vsa:linker − none | property | +0.1473 [+0.1340, +0.1606] (p 0.0005) | 4587 |
| explicit-vsa:linker − none | paraphrase | +0.0892 [+0.0689, +0.1083] (p 0.0005) | 4587 |
| explicit-vsa:linker − none | entailment | +0.0456 [+0.0161, +0.0744] (p 0.0020) | 1800 |
| both:oracle − explicit-sym:oracle | property | +0.0279 [+0.0210, +0.0348] (p 0.0005) | 4587 |
| both:oracle − explicit-sym:oracle | paraphrase | +0.0336 [+0.0220, +0.0460] (p 0.0005) | 4587 |
| both:oracle − explicit-sym:oracle | entailment | +0.0239 [+0.0106, +0.0367] (p 0.0005) | 1800 |
| both:oracle − implicit:oracle | property | +0.2692 [+0.2535, +0.2846] (p 0.0005) | 4587 |
| both:oracle − implicit:oracle | paraphrase | +0.2354 [+0.2163, +0.2544] (p 0.0005) | 4587 |
| both:oracle − implicit:oracle | entailment | +0.1439 [+0.1133, +0.1733] (p 0.0005) | 1800 |
| definition − explicit-sym:oracle | property | +0.1405 [+0.1288, +0.1528] (p 0.0005) | 4587 |
| definition − explicit-sym:oracle | paraphrase | -0.0905 [-0.1075, -0.0739] (p 0.0005) | 4587 |
| definition − explicit-sym:oracle | entailment | +0.0717 [+0.0489, +0.0944] (p 0.0005) | 1800 |
| definition − none | property | +0.4526 [+0.4394, +0.4653] (p 0.0005) | 4587 |
| definition − none | paraphrase | +0.1751 [+0.1550, +0.1953] (p 0.0005) | 4587 |
| definition − none | entailment | +0.2111 [+0.1828, +0.2400] (p 0.0005) | 1800 |
| explicit-sym:linker − implicit:linker | property | +0.1368 [+0.1223, +0.1504] (p 0.0005) | 4587 |
| explicit-sym:linker − implicit:linker | paraphrase | +0.0586 [+0.0390, +0.0778] (p 0.0005) | 4587 |
| explicit-sym:linker − implicit:linker | entailment | +0.0372 [+0.0061, +0.0678] (p 0.0160) | 1800 |
| explicit-sym:typeprior − implicit:typeprior | property | +0.2525 [+0.2361, +0.2681] (p 0.0005) | 4587 |
| explicit-sym:typeprior − implicit:typeprior | paraphrase | +0.1969 [+0.1772, +0.2163] (p 0.0005) | 4587 |
| explicit-sym:typeprior − implicit:typeprior | entailment | +0.1267 [+0.0961, +0.1567] (p 0.0005) | 1800 |

