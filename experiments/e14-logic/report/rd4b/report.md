# E14 RD-4b — recall of the item's relation, context-free null (pre-registration §13)

PMI choices use the null prompt **without** the context; `raw` columns are uncalibrated choices; `(stated)`: items whose gold answer is in the `stated` reader's frame.

## t4 heldout — SmolLM2-360M-full-C0p (1 seeds; 131 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.7322 | 0.7248 | 0.8624 | 0.8690 | 0.9312 | 0.9593 | 0.9924 |
| explicit-rel-wrong:oracle | 0.5266 | 0.5504 | 0.6953 | 0.6904 | 0.8722 | 0.6743 | 0.7684 |
| explicit-rel:linker | 0.6667 | 0.6265 | 0.8305 | 0.7756 | 0.9042 | 0.8524 | 0.8651 |
| explicit-rel:oracle | 0.8575 | 0.8452 | 0.9238 | 0.9640 | 0.9902 | 0.9593 | 0.9847 |
| explicit-rel:stated | 0.7084 | 0.6757 | 0.8452 | 0.8256 | 0.9140 | 0.9746 | 1.0000 |
| explicit-rel:typeprior | 0.7117 | 0.6757 | 0.8550 | 0.8346 | 0.9238 | 0.9618 | 0.9949 |
| explicit-sym:oracle | 0.5749 | 0.8501 | 0.6536 | 0.6945 | 1.0000 | 0.7354 | 0.8295 |
| none | 0.6175 | 0.5405 | 0.7666 | 0.7346 | 0.8796 | 0.7608 | 0.7913 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.2400 [+0.2064, +0.2760] (p 0.0005, Holm 0.0030) | 407 |
| explicit-rel:oracle − none | paraphrase | +0.3047 [+0.2531, +0.3563] (p 0.0005, Holm 0.0030) | 407 |
| explicit-rel:oracle − none | entailment | +0.1572 [+0.1106, +0.2015] (p 0.0005, Holm 0.0030) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.2826 [+0.2424, +0.3260] (p 0.0005, Holm 0.0030) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | paraphrase | -0.0049 [-0.0442, +0.0344] (p 0.8868, Holm 0.8868) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.2703 [+0.2260, +0.3170] (p 0.0005, Holm 0.0030) | 407 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0426 [-0.0893, +0.0041] (p 0.0805) | 407 |
| explicit-sym:oracle − none | paraphrase | +0.3096 [+0.2531, +0.3661] (p 0.0005) | 407 |
| explicit-sym:oracle − none | entailment | -0.1130 [-0.1744, -0.0541] (p 0.0005) | 407 |
| explicit-rel:stated − none | property | +0.0909 [+0.0622, +0.1212] (p 0.0005) | 407 |
| explicit-rel:stated − none | paraphrase | +0.1351 [+0.0909, +0.1843] (p 0.0005) | 407 |
| explicit-rel:stated − none | entailment | +0.0786 [+0.0442, +0.1155] (p 0.0005) | 407 |
| explicit-rel:typeprior − none | property | +0.0942 [+0.0647, +0.1269] (p 0.0005) | 407 |
| explicit-rel:typeprior − none | paraphrase | +0.1351 [+0.0860, +0.1867] (p 0.0005) | 407 |
| explicit-rel:typeprior − none | entailment | +0.0885 [+0.0515, +0.1278] (p 0.0005) | 407 |
| explicit-rel:linker − none | property | +0.0491 [+0.0246, +0.0753] (p 0.0010) | 407 |
| explicit-rel:linker − none | paraphrase | +0.0860 [+0.0393, +0.1327] (p 0.0015) | 407 |
| explicit-rel:linker − none | entailment | +0.0639 [+0.0294, +0.1007] (p 0.0005) | 407 |
| explicit-rel-wrong:oracle − none | property | -0.0909 [-0.1220, -0.0590] (p 0.0005) | 407 |
| explicit-rel-wrong:oracle − none | paraphrase | +0.0098 [-0.0467, +0.0664] (p 0.7693) | 407 |
| explicit-rel-wrong:oracle − none | entailment | -0.0713 [-0.1179, -0.0246] (p 0.0030) | 407 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | property | +0.3309 [+0.2948, +0.3678] (p 0.0005) | 407 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | paraphrase | +0.2948 [+0.2457, +0.3464] (p 0.0005) | 407 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | entailment | +0.2285 [+0.1867, +0.2703] (p 0.0005) | 407 |
| definition − explicit-rel:oracle | property | -0.1253 [-0.1548, -0.0966] (p 0.0005) | 407 |
| definition − explicit-rel:oracle | paraphrase | -0.1204 [-0.1622, -0.0786] (p 0.0005) | 407 |
| definition − explicit-rel:oracle | entailment | -0.0614 [-0.0958, -0.0270] (p 0.0005) | 407 |
| definition − none | property | +0.1147 [+0.0876, +0.1450] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.1843 [+0.1351, +0.2310] (p 0.0005) | 407 |
| definition − none | entailment | +0.0958 [+0.0590, +0.1351] (p 0.0005) | 407 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:stated − none | property | +0.2137 [+0.1578, +0.2723] (p 0.0005) | 131 |
| explicit-rel:stated − none | paraphrase | +0.2901 [+0.2137, +0.3740] (p 0.0005) | 131 |
| explicit-rel:stated − none | entailment | +0.1220 [+0.0650, +0.1789] (p 0.0005) | 123 |
| definition − none | property | +0.1985 [+0.1450, +0.2545] (p 0.0005) | 131 |
| definition − none | paraphrase | +0.2977 [+0.2214, +0.3817] (p 0.0005) | 131 |
| definition − none | entailment | +0.1138 [+0.0569, +0.1707] (p 0.0005) | 123 |
| explicit-rel:stated − definition | property | +0.0153 [-0.0051, +0.0408] (p 0.2084) | 131 |
| explicit-rel:stated − definition | paraphrase | -0.0076 [-0.0382, +0.0153] (p 0.8223) | 131 |
| explicit-rel:stated − definition | entailment | +0.0081 [+0.0000, +0.0244] (p 0.7258) | 123 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.2293 [+0.1876, +0.2703] (p 0.0005) | 407 |
| explicit-rel:oracle − none | entailment | +0.1106 [+0.0811, +0.1425] (p 0.0005) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.2695 [+0.2285, +0.3120] (p 0.0005) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | -0.0098 [-0.0197, -0.0025] (p 0.0295) | 407 |

## t4 heldout — SmolLM2-360M-full-C5 (3 seeds; 131 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.7338 | 0.7224 | 0.8608 | 0.8676 | 0.9312 | 0.9576 | 0.9924 |
| explicit-rel-wrong:oracle | 0.5280 | 0.5618 | 0.6937 | 0.6918 | 0.8722 | 0.6743 | 0.7693 |
| explicit-rel:linker | 0.6609 | 0.6159 | 0.8280 | 0.7811 | 0.9009 | 0.8346 | 0.8711 |
| explicit-rel:oracle | 0.8624 | 0.8411 | 0.9279 | 0.9612 | 0.9902 | 0.9593 | 0.9847 |
| explicit-rel:stated | 0.7095 | 0.6896 | 0.8452 | 0.8250 | 0.9132 | 0.9771 | 1.0000 |
| explicit-rel:typeprior | 0.7169 | 0.6757 | 0.8567 | 0.8348 | 0.9230 | 0.9627 | 0.9949 |
| explicit-sym:oracle | 0.5782 | 0.8591 | 0.6618 | 0.6959 | 1.0000 | 0.7354 | 0.8329 |
| implicit:oracle | 0.6151 | 0.5274 | 0.7666 | 0.7325 | 0.8829 | 0.7583 | 0.7863 |
| none | 0.6143 | 0.5438 | 0.7666 | 0.7327 | 0.8837 | 0.7489 | 0.7880 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.2482 [+0.2285, +0.2675] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-rel:oracle − none | paraphrase | +0.2973 [+0.2695, +0.3260] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-rel:oracle − none | entailment | +0.1613 [+0.1360, +0.1859] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.2842 [+0.2604, +0.3085] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-rel:oracle − explicit-sym:oracle | paraphrase | -0.0180 [-0.0410, +0.0049] (p 0.1230, Holm 0.1230) | 1221 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.2662 [+0.2400, +0.2916] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-rel:oracle − implicit:oracle | property | +0.2473 [+0.2277, +0.2662] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-rel:oracle − implicit:oracle | paraphrase | +0.3137 [+0.2858, +0.3423] (p 0.0005, Holm 0.0045) | 1221 |
| explicit-rel:oracle − implicit:oracle | entailment | +0.1613 [+0.1360, +0.1867] (p 0.0005, Holm 0.0045) | 1221 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | -0.0360 [-0.0642, -0.0082] (p 0.0095) | 1221 |
| explicit-sym:oracle − none | paraphrase | +0.3153 [+0.2834, +0.3489] (p 0.0005) | 1221 |
| explicit-sym:oracle − none | entailment | -0.1048 [-0.1392, -0.0713] (p 0.0005) | 1221 |
| explicit-rel:stated − none | property | +0.0953 [+0.0775, +0.1127] (p 0.0005) | 1221 |
| explicit-rel:stated − none | paraphrase | +0.1458 [+0.1179, +0.1720] (p 0.0005) | 1221 |
| explicit-rel:stated − none | entailment | +0.0786 [+0.0590, +0.0999] (p 0.0005) | 1221 |
| explicit-rel:typeprior − none | property | +0.1026 [+0.0849, +0.1212] (p 0.0005) | 1221 |
| explicit-rel:typeprior − none | paraphrase | +0.1319 [+0.1032, +0.1605] (p 0.0005) | 1221 |
| explicit-rel:typeprior − none | entailment | +0.0901 [+0.0688, +0.1130] (p 0.0005) | 1221 |
| explicit-rel:linker − none | property | +0.0467 [+0.0322, +0.0612] (p 0.0005) | 1221 |
| explicit-rel:linker − none | paraphrase | +0.0721 [+0.0450, +0.0991] (p 0.0005) | 1221 |
| explicit-rel:linker − none | entailment | +0.0614 [+0.0410, +0.0827] (p 0.0005) | 1221 |
| explicit-rel-wrong:oracle − none | property | -0.0863 [-0.1048, -0.0680] (p 0.0005) | 1221 |
| explicit-rel-wrong:oracle − none | paraphrase | +0.0180 [-0.0147, +0.0500] (p 0.2774) | 1221 |
| explicit-rel-wrong:oracle − none | entailment | -0.0729 [-0.0991, -0.0467] (p 0.0005) | 1221 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | property | +0.3344 [+0.3129, +0.3557] (p 0.0005) | 1221 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | paraphrase | +0.2793 [+0.2506, +0.3079] (p 0.0005) | 1221 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | entailment | +0.2342 [+0.2105, +0.2580] (p 0.0005) | 1221 |
| definition − explicit-rel:oracle | property | -0.1286 [-0.1458, -0.1125] (p 0.0005) | 1221 |
| definition − explicit-rel:oracle | paraphrase | -0.1188 [-0.1425, -0.0958] (p 0.0005) | 1221 |
| definition − explicit-rel:oracle | entailment | -0.0672 [-0.0876, -0.0475] (p 0.0005) | 1221 |
| definition − none | property | +0.1196 [+0.1032, +0.1360] (p 0.0005) | 1221 |
| definition − none | paraphrase | +0.1785 [+0.1491, +0.2080] (p 0.0005) | 1221 |
| definition − none | entailment | +0.0942 [+0.0737, +0.1147] (p 0.0005) | 1221 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:stated − none | property | +0.2282 [+0.1942, +0.2621] (p 0.0005) | 393 |
| explicit-rel:stated − none | paraphrase | +0.2926 [+0.2443, +0.3435] (p 0.0005) | 393 |
| explicit-rel:stated − none | entailment | +0.1274 [+0.0949, +0.1626] (p 0.0005) | 369 |
| definition − none | property | +0.2087 [+0.1773, +0.2409] (p 0.0005) | 393 |
| definition − none | paraphrase | +0.2850 [+0.2392, +0.3333] (p 0.0005) | 393 |
| definition − none | entailment | +0.1192 [+0.0867, +0.1518] (p 0.0005) | 369 |
| explicit-rel:stated − definition | property | +0.0195 [+0.0068, +0.0331] (p 0.0010) | 393 |
| explicit-rel:stated − definition | paraphrase | +0.0076 [-0.0076, +0.0229] (p 0.3849) | 393 |
| explicit-rel:stated − definition | entailment | +0.0081 [+0.0000, +0.0190] (p 0.0950) | 369 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.2285 [+0.2050, +0.2523] (p 0.0005) | 1221 |
| explicit-rel:oracle − none | entailment | +0.1065 [+0.0893, +0.1245] (p 0.0005) | 1221 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.2654 [+0.2408, +0.2902] (p 0.0005) | 1221 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | -0.0098 [-0.0164, -0.0049] (p 0.0005) | 1221 |
| explicit-rel:oracle − implicit:oracle | property | +0.2288 [+0.2045, +0.2525] (p 0.0005) | 1221 |
| explicit-rel:oracle − implicit:oracle | entailment | +0.1073 [+0.0909, +0.1253] (p 0.0005) | 1221 |

## t5 new — SmolLM2-360M-full-C0p (1 seeds; 1229 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.8453 | 0.8162 | 0.8833 | 0.9333 | 0.9300 | 0.8702 | 0.9276 |
| explicit-rel-wrong:oracle | 0.1756 | 0.5513 | 0.4700 | 0.1972 | 0.5033 | 0.1839 | 0.2107 |
| explicit-rel:linker | 0.4438 | 0.6311 | 0.6883 | 0.4935 | 0.7350 | 0.4532 | 0.5024 |
| explicit-rel:oracle | 0.7950 | 0.7888 | 0.8033 | 0.8751 | 0.8550 | 0.8515 | 0.9166 |
| explicit-rel:stated | 0.7950 | 0.7888 | 0.8033 | 0.8751 | 0.8550 | 0.8515 | 0.9166 |
| explicit-rel:typeprior | 0.7603 | 0.7737 | 0.8017 | 0.8391 | 0.8400 | 0.8088 | 0.8727 |
| explicit-sym:oracle | 0.6776 | 0.7600 | 0.8017 | 0.7701 | 0.8583 | 0.6912 | 0.7531 |
| none | 0.1884 | 0.3218 | 0.5100 | 0.2155 | 0.5350 | 0.1851 | 0.2181 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.6066 [+0.5847, +0.6285] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-rel:oracle − none | paraphrase | +0.4670 [+0.4388, +0.4951] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-rel:oracle − none | entailment | +0.2933 [+0.2483, +0.3383] (p 0.0005, Holm 0.0030) | 600 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.1174 [+0.0981, +0.1377] (p 0.0005, Holm 0.0030) | 1529 |
| explicit-rel:oracle − explicit-sym:oracle | paraphrase | +0.0288 [+0.0059, +0.0517] (p 0.0135, Holm 0.0270) | 1529 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.0017 [+0.0000, +0.0050] (p 0.7218, Holm 0.7218) | 600 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.4892 [+0.4637, +0.5137] (p 0.0005) | 1529 |
| explicit-sym:oracle − none | paraphrase | +0.4382 [+0.4088, +0.4676] (p 0.0005) | 1529 |
| explicit-sym:oracle − none | entailment | +0.2917 [+0.2467, +0.3367] (p 0.0005) | 600 |
| explicit-rel:stated − none | property | +0.6066 [+0.5847, +0.6285] (p 0.0005) | 1529 |
| explicit-rel:stated − none | paraphrase | +0.4670 [+0.4388, +0.4951] (p 0.0005) | 1529 |
| explicit-rel:stated − none | entailment | +0.2933 [+0.2483, +0.3383] (p 0.0005) | 600 |
| explicit-rel:typeprior − none | property | +0.5719 [+0.5494, +0.5942] (p 0.0005) | 1529 |
| explicit-rel:typeprior − none | paraphrase | +0.4519 [+0.4232, +0.4801] (p 0.0005) | 1529 |
| explicit-rel:typeprior − none | entailment | +0.2917 [+0.2467, +0.3383] (p 0.0005) | 600 |
| explicit-rel:linker − none | property | +0.2554 [+0.2325, +0.2776] (p 0.0005) | 1529 |
| explicit-rel:linker − none | paraphrase | +0.3094 [+0.2780, +0.3407] (p 0.0005) | 1529 |
| explicit-rel:linker − none | entailment | +0.1783 [+0.1333, +0.2250] (p 0.0005) | 600 |
| explicit-rel-wrong:oracle − none | property | -0.0128 [-0.0304, +0.0049] (p 0.1485) | 1529 |
| explicit-rel-wrong:oracle − none | paraphrase | +0.2296 [+0.1982, +0.2616] (p 0.0005) | 1529 |
| explicit-rel-wrong:oracle − none | entailment | -0.0400 [-0.0833, +0.0050] (p 0.0910) | 600 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | property | +0.6194 [+0.5988, +0.6403] (p 0.0005) | 1529 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | paraphrase | +0.2374 [+0.2060, +0.2681] (p 0.0005) | 1529 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | entailment | +0.3333 [+0.2933, +0.3717] (p 0.0005) | 600 |
| definition − explicit-rel:oracle | property | +0.0504 [+0.0379, +0.0625] (p 0.0005) | 1529 |
| definition − explicit-rel:oracle | paraphrase | +0.0275 [+0.0065, +0.0484] (p 0.0095) | 1529 |
| definition − explicit-rel:oracle | entailment | +0.0800 [+0.0517, +0.1083] (p 0.0005) | 600 |
| definition − none | property | +0.6570 [+0.6367, +0.6759] (p 0.0005) | 1529 |
| definition − none | paraphrase | +0.4944 [+0.4663, +0.5226] (p 0.0005) | 1529 |
| definition − none | entailment | +0.3733 [+0.3317, +0.4167] (p 0.0005) | 600 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:stated − none | property | +0.6664 [+0.6428, +0.6904] (p 0.0005) | 1229 |
| explicit-rel:stated − none | paraphrase | +0.4841 [+0.4516, +0.5159] (p 0.0005) | 1229 |
| definition − none | property | +0.6851 [+0.6635, +0.7075] (p 0.0005) | 1229 |
| definition − none | paraphrase | +0.4931 [+0.4614, +0.5232] (p 0.0005) | 1229 |
| explicit-rel:stated − definition | property | -0.0187 [-0.0305, -0.0073] (p 0.0030) | 1229 |
| explicit-rel:stated − definition | paraphrase | -0.0090 [-0.0301, +0.0122] (p 0.4344) | 1229 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.6596 [+0.6367, +0.6818] (p 0.0005) | 1529 |
| explicit-rel:oracle − none | entailment | +0.3200 [+0.2717, +0.3683] (p 0.0005) | 600 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.1050 [+0.0837, +0.1262] (p 0.0005) | 1529 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | -0.0033 [-0.0083, +0.0000] (p 0.2669) | 600 |

## t5 new — SmolLM2-360M-full-C5 (3 seeds; 1229 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.8567 | 0.8413 | 0.8900 | 0.9267 | 0.9350 | 0.8881 | 0.9391 |
| explicit-rel-wrong:oracle | 0.1712 | 0.5858 | 0.4617 | 0.2041 | 0.5039 | 0.1781 | 0.2151 |
| explicit-rel:linker | 0.4682 | 0.6416 | 0.6828 | 0.5055 | 0.7161 | 0.4868 | 0.5263 |
| explicit-rel:oracle | 0.8450 | 0.8254 | 0.7994 | 0.8975 | 0.8556 | 0.9079 | 0.9585 |
| explicit-rel:stated | 0.8450 | 0.8254 | 0.7994 | 0.8975 | 0.8556 | 0.9079 | 0.9585 |
| explicit-rel:typeprior | 0.8107 | 0.8066 | 0.7978 | 0.8604 | 0.8461 | 0.8656 | 0.9121 |
| explicit-sym:oracle | 0.6872 | 0.7667 | 0.7978 | 0.7557 | 0.8572 | 0.7135 | 0.7676 |
| implicit:oracle | 0.2603 | 0.4079 | 0.5350 | 0.2842 | 0.5767 | 0.2267 | 0.2514 |
| none | 0.1896 | 0.3442 | 0.5156 | 0.2245 | 0.5350 | 0.1839 | 0.2262 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.6554 [+0.6433, +0.6672] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-rel:oracle − none | paraphrase | +0.4811 [+0.4644, +0.4971] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-rel:oracle − none | entailment | +0.2839 [+0.2567, +0.3117] (p 0.0005, Holm 0.0045) | 1800 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.1578 [+0.1461, +0.1692] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-rel:oracle − explicit-sym:oracle | paraphrase | +0.0586 [+0.0460, +0.0713] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.0017 [-0.0017, +0.0050] (p 0.4034, Holm 0.4034) | 1800 |
| explicit-rel:oracle − implicit:oracle | property | +0.5847 [+0.5712, +0.5981] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-rel:oracle − implicit:oracle | paraphrase | +0.4175 [+0.4003, +0.4345] (p 0.0005, Holm 0.0045) | 4587 |
| explicit-rel:oracle − implicit:oracle | entailment | +0.2644 [+0.2383, +0.2906] (p 0.0005, Holm 0.0045) | 1800 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-sym:oracle − none | property | +0.4976 [+0.4822, +0.5121] (p 0.0005) | 4587 |
| explicit-sym:oracle − none | paraphrase | +0.4225 [+0.4044, +0.4404] (p 0.0005) | 4587 |
| explicit-sym:oracle − none | entailment | +0.2822 [+0.2550, +0.3100] (p 0.0005) | 1800 |
| explicit-rel:stated − none | property | +0.6554 [+0.6433, +0.6672] (p 0.0005) | 4587 |
| explicit-rel:stated − none | paraphrase | +0.4811 [+0.4644, +0.4971] (p 0.0005) | 4587 |
| explicit-rel:stated − none | entailment | +0.2839 [+0.2567, +0.3117] (p 0.0005) | 1800 |
| explicit-rel:typeprior − none | property | +0.6211 [+0.6086, +0.6334] (p 0.0005) | 4587 |
| explicit-rel:typeprior − none | paraphrase | +0.4624 [+0.4456, +0.4787] (p 0.0005) | 4587 |
| explicit-rel:typeprior − none | entailment | +0.2822 [+0.2561, +0.3094] (p 0.0005) | 1800 |
| explicit-rel:linker − none | property | +0.2786 [+0.2654, +0.2922] (p 0.0005) | 4587 |
| explicit-rel:linker − none | paraphrase | +0.2974 [+0.2790, +0.3148] (p 0.0005) | 4587 |
| explicit-rel:linker − none | entailment | +0.1672 [+0.1417, +0.1928] (p 0.0005) | 1800 |
| explicit-rel-wrong:oracle − none | property | -0.0183 [-0.0280, -0.0085] (p 0.0005) | 4587 |
| explicit-rel-wrong:oracle − none | paraphrase | +0.2416 [+0.2226, +0.2590] (p 0.0005) | 4587 |
| explicit-rel-wrong:oracle − none | entailment | -0.0539 [-0.0800, -0.0267] (p 0.0005) | 1800 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | property | +0.6738 [+0.6617, +0.6855] (p 0.0005) | 4587 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | paraphrase | +0.2396 [+0.2226, +0.2572] (p 0.0005) | 4587 |
| explicit-rel:oracle − explicit-rel-wrong:oracle | entailment | +0.3378 [+0.3144, +0.3606] (p 0.0005) | 1800 |
| definition − explicit-rel:oracle | property | +0.0117 [+0.0039, +0.0193] (p 0.0035) | 4587 |
| definition − explicit-rel:oracle | paraphrase | +0.0159 [+0.0041, +0.0270] (p 0.0105) | 4587 |
| definition − explicit-rel:oracle | entailment | +0.0906 [+0.0739, +0.1072] (p 0.0005) | 1800 |
| definition − none | property | +0.6671 [+0.6552, +0.6783] (p 0.0005) | 4587 |
| definition − none | paraphrase | +0.4971 [+0.4807, +0.5134] (p 0.0005) | 4587 |
| definition − none | entailment | +0.3744 [+0.3494, +0.3994] (p 0.0005) | 1800 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:stated − none | property | +0.7240 [+0.7117, +0.7358] (p 0.0005) | 3687 |
| explicit-rel:stated − none | paraphrase | +0.5189 [+0.5015, +0.5365] (p 0.0005) | 3687 |
| definition − none | property | +0.7042 [+0.6912, +0.7163] (p 0.0005) | 3687 |
| definition − none | paraphrase | +0.5094 [+0.4920, +0.5270] (p 0.0005) | 3687 |
| explicit-rel:stated − definition | property | +0.0198 [+0.0134, +0.0263] (p 0.0005) | 3687 |
| explicit-rel:stated − definition | paraphrase | +0.0095 [-0.0011, +0.0198] (p 0.0945) | 3687 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.6730 [+0.6606, +0.6855] (p 0.0005) | 4587 |
| explicit-rel:oracle − none | entailment | +0.3206 [+0.2928, +0.3478] (p 0.0005) | 1800 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.1418 [+0.1303, +0.1536] (p 0.0005) | 4587 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | -0.0017 [-0.0044, +0.0006] (p 0.2569) | 1800 |
| explicit-rel:oracle − implicit:oracle | property | +0.6134 [+0.5997, +0.6269] (p 0.0005) | 4587 |
| explicit-rel:oracle − implicit:oracle | entailment | +0.2789 [+0.2517, +0.3061] (p 0.0005) | 1800 |

