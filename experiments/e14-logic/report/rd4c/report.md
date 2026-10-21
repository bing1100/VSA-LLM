# E14 RD-4c — routing without the item's label, Claude's frames, larger hosts (pre-registration §16)

PMI choices use the null prompt **without** the context; `raw` columns are uncalibrated choices; `(stated)`: items whose gold answer is in the `stated` reader's frame.

## t4 heldout — Qwen3-1.7B-Base-lora-C0p (1 seeds; 131 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.8124 | 0.7396 | 0.9337 | 0.8870 | 0.9558 | 0.9695 | 0.9949 |
| explicit-rel-wrong:oracle | 0.6159 | 0.5651 | 0.7322 | 0.7674 | 0.9165 | 0.7557 | 0.8575 |
| explicit-rel:oracle | 0.9009 | 0.8550 | 0.9115 | 0.9648 | 0.9926 | 0.9593 | 0.9822 |
| explicit-rel:teacher | 0.7649 | 0.6658 | 0.8771 | 0.8583 | 0.9287 | 0.9313 | 0.9720 |
| explicit-rel:typeprior | 0.7756 | 0.6781 | 0.8747 | 0.8698 | 0.9435 | 0.9644 | 1.0000 |
| explicit-route:lex:oracle | 0.8337 | 0.7690 | 0.9042 | 0.9222 | 0.9926 | 0.9517 | 0.9975 |
| explicit-route:lex:typeprior | 0.7764 | 0.6781 | 0.8747 | 0.8771 | 0.9484 | 0.9440 | 1.0000 |
| explicit-route:lm:oracle | 0.8272 | 0.7543 | 0.9165 | 0.9173 | 0.9754 | 0.9440 | 0.9898 |
| explicit-route:lm:teacher | 0.7723 | 0.6486 | 0.8796 | 0.8690 | 0.9312 | 0.9262 | 0.9695 |
| explicit-route:lm:typeprior | 0.7731 | 0.6732 | 0.8821 | 0.8763 | 0.9459 | 0.9415 | 1.0000 |
| explicit-route:top2:oracle | 0.8296 | 0.8157 | 0.8550 | 0.9140 | 0.9877 | 0.9517 | 0.9822 |
| explicit-route:top2:typeprior | 0.7772 | 0.6609 | 0.8821 | 0.8845 | 0.9435 | 0.9567 | 1.0000 |
| explicit-sym:oracle | 0.6052 | 0.8329 | 0.6609 | 0.7150 | 0.9951 | 0.7863 | 0.8626 |
| none | 0.7289 | 0.6388 | 0.8747 | 0.7920 | 0.9066 | 0.8244 | 0.8601 |

Router accuracy (the asked relation chosen, among items whose frame holds it): explicit-route:lex:oracle 0.792; explicit-route:lex:typeprior 0.890; explicit-route:lm:oracle 0.585; explicit-route:lm:teacher 0.801; explicit-route:lm:typeprior 0.781; explicit-route:top2:oracle 0.838; explicit-route:top2:typeprior 0.966

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.1720 [+0.1400, +0.2056] (p 0.0005, Holm 0.0030) | 407 |
| explicit-rel:oracle − none | paraphrase | +0.2162 [+0.1695, +0.2654] (p 0.0005, Holm 0.0030) | 407 |
| explicit-rel:oracle − none | entailment | +0.0369 [+0.0000, +0.0737] (p 0.0550, Holm 0.1100) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.2957 [+0.2531, +0.3399] (p 0.0005, Holm 0.0030) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | paraphrase | +0.0221 [-0.0147, +0.0615] (p 0.2844, Holm 0.2844) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.2506 [+0.2064, +0.2948] (p 0.0005, Holm 0.0030) | 407 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lm:typeprior − none | property | +0.0442 [+0.0172, +0.0729] (p 0.0025) | 407 |
| explicit-route:lm:typeprior − none | paraphrase | +0.0344 [-0.0098, +0.0811] (p 0.1540) | 407 |
| explicit-route:lm:typeprior − none | entailment | +0.0074 [-0.0221, +0.0393] (p 0.6948) | 407 |
| explicit-route:lm:typeprior − definition | property | -0.0393 [-0.0598, -0.0188] (p 0.0005) | 407 |
| explicit-route:lm:typeprior − definition | paraphrase | -0.0663 [-0.1057, -0.0295] (p 0.0005) | 407 |
| explicit-route:lm:typeprior − definition | entailment | -0.0516 [-0.0786, -0.0270] (p 0.0005) | 407 |
| explicit-rel:typeprior − none | property | +0.0467 [+0.0205, +0.0745] (p 0.0005) | 407 |
| explicit-rel:typeprior − none | paraphrase | +0.0393 [-0.0049, +0.0811] (p 0.0825) | 407 |
| explicit-rel:typeprior − none | entailment | +0.0000 [-0.0344, +0.0344] (p 1.0000) | 407 |
| explicit-rel-wrong:oracle − none | property | -0.1130 [-0.1441, -0.0827] (p 0.0005) | 407 |
| explicit-rel-wrong:oracle − none | paraphrase | -0.0737 [-0.1254, -0.0246] (p 0.0045) | 407 |
| explicit-rel-wrong:oracle − none | entailment | -0.1425 [-0.1843, -0.1031] (p 0.0005) | 407 |
| definition − none | property | +0.0835 [+0.0573, +0.1122] (p 0.0005) | 407 |
| definition − none | paraphrase | +0.1007 [+0.0590, +0.1450] (p 0.0005) | 407 |
| definition − none | entailment | +0.0590 [+0.0319, +0.0860] (p 0.0005) | 407 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − none | property | +0.1450 [+0.0967, +0.1959] (p 0.0005) | 131 |
| definition − none | paraphrase | +0.1679 [+0.0992, +0.2443] (p 0.0005) | 131 |
| definition − none | entailment | +0.0732 [+0.0325, +0.1220] (p 0.0010) | 123 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.1728 [+0.1360, +0.2113] (p 0.0005) | 407 |
| explicit-rel:oracle − none | entailment | +0.0860 [+0.0590, +0.1130] (p 0.0005) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.2498 [+0.2096, +0.2908] (p 0.0005) | 407 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | -0.0025 [-0.0098, +0.0049] (p 0.7923) | 407 |

## t4 heldout — SmolLM2-360M-full-C5 (3 seeds; 131 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.7338 | 0.7224 | 0.8608 | 0.8679 | 0.9312 | 0.9576 | 0.9924 |
| explicit-rel-wrong:oracle | 0.5280 | 0.5618 | 0.6937 | 0.6918 | 0.8722 | 0.6743 | 0.7693 |
| explicit-rel:oracle | 0.8624 | 0.8411 | 0.9279 | 0.9612 | 0.9902 | 0.9593 | 0.9847 |
| explicit-rel:teacher | 0.7032 | 0.6634 | 0.8354 | 0.8190 | 0.9115 | 0.9364 | 0.9618 |
| explicit-rel:typeprior | 0.7169 | 0.6757 | 0.8567 | 0.8348 | 0.9230 | 0.9627 | 0.9949 |
| explicit-route:lex:oracle | 0.7871 | 0.7854 | 0.9214 | 0.9083 | 0.9902 | 0.9440 | 0.9966 |
| explicit-route:lex:typeprior | 0.7142 | 0.6781 | 0.8714 | 0.8348 | 0.9287 | 0.9389 | 0.9924 |
| explicit-route:lm:oracle | 0.7797 | 0.7690 | 0.9009 | 0.8998 | 0.9590 | 0.9338 | 0.9890 |
| explicit-route:lm:teacher | 0.7032 | 0.6740 | 0.8419 | 0.8198 | 0.9140 | 0.9160 | 0.9567 |
| explicit-route:lm:typeprior | 0.7128 | 0.6740 | 0.8665 | 0.8367 | 0.9255 | 0.9389 | 0.9975 |
| explicit-route:top2:oracle | 0.7786 | 0.8092 | 0.8731 | 0.9004 | 0.9902 | 0.9423 | 0.9822 |
| explicit-route:top2:typeprior | 0.7123 | 0.6675 | 0.8780 | 0.8335 | 0.9378 | 0.9542 | 1.0000 |
| explicit-sym:oracle | 0.5782 | 0.8591 | 0.6618 | 0.6959 | 1.0000 | 0.7354 | 0.8329 |
| none | 0.6143 | 0.5438 | 0.7666 | 0.7327 | 0.8837 | 0.7489 | 0.7880 |

Router accuracy (the asked relation chosen, among items whose frame holds it): explicit-route:lex:oracle 0.792; explicit-route:lex:typeprior 0.890; explicit-route:lm:oracle 0.585; explicit-route:lm:teacher 0.798; explicit-route:lm:typeprior 0.781; explicit-route:top2:oracle 0.832; explicit-route:top2:typeprior 0.962

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lm:typeprior − none | property | +0.0986 [+0.0814, +0.1160] (p 0.0005, Holm 0.0030) | 1221 |
| explicit-route:lm:typeprior − none | paraphrase | +0.1302 [+0.1007, +0.1589] (p 0.0005, Holm 0.0030) | 1221 |
| explicit-route:lm:typeprior − none | entailment | +0.0999 [+0.0778, +0.1237] (p 0.0005, Holm 0.0030) | 1221 |
| explicit-route:lm:oracle − explicit-rel:oracle | property | -0.0827 [-0.0958, -0.0696] (p 0.0005, Holm 0.0030) | 1221 |
| explicit-route:lm:oracle − explicit-rel:oracle | paraphrase | -0.0721 [-0.0925, -0.0516] (p 0.0005, Holm 0.0030) | 1221 |
| explicit-route:lm:oracle − explicit-rel:oracle | entailment | -0.0270 [-0.0434, -0.0115] (p 0.0005, Holm 0.0030) | 1221 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lex:oracle − explicit-rel:oracle | property | -0.0753 [-0.0879, -0.0628] (p 0.0005) | 1221 |
| explicit-route:lex:oracle − explicit-rel:oracle | paraphrase | -0.0557 [-0.0745, -0.0360] (p 0.0005) | 1221 |
| explicit-route:lex:oracle − explicit-rel:oracle | entailment | -0.0066 [-0.0115, -0.0025] (p 0.0010) | 1221 |
| explicit-route:top2:oracle − explicit-rel:oracle | property | -0.0838 [-0.0977, -0.0707] (p 0.0005) | 1221 |
| explicit-route:top2:oracle − explicit-rel:oracle | paraphrase | -0.0319 [-0.0491, -0.0156] (p 0.0005) | 1221 |
| explicit-route:top2:oracle − explicit-rel:oracle | entailment | -0.0549 [-0.0713, -0.0385] (p 0.0005) | 1221 |
| explicit-route:lm:oracle − explicit-sym:oracle | property | +0.2015 [+0.1753, +0.2277] (p 0.0005) | 1221 |
| explicit-route:lm:oracle − explicit-sym:oracle | paraphrase | -0.0901 [-0.1147, -0.0647] (p 0.0005) | 1221 |
| explicit-route:lm:oracle − explicit-sym:oracle | entailment | +0.2391 [+0.2113, +0.2670] (p 0.0005) | 1221 |
| explicit-route:lm:typeprior − explicit-rel:typeprior | property | -0.0041 [-0.0139, +0.0057] (p 0.4149) | 1221 |
| explicit-route:lm:typeprior − explicit-rel:typeprior | paraphrase | -0.0016 [-0.0197, +0.0156] (p 0.8743) | 1221 |
| explicit-route:lm:typeprior − explicit-rel:typeprior | entailment | +0.0098 [-0.0041, +0.0246] (p 0.1890) | 1221 |
| explicit-route:lex:typeprior − none | property | +0.0999 [+0.0824, +0.1171] (p 0.0005) | 1221 |
| explicit-route:lex:typeprior − none | paraphrase | +0.1343 [+0.1057, +0.1630] (p 0.0005) | 1221 |
| explicit-route:lex:typeprior − none | entailment | +0.1048 [+0.0811, +0.1286] (p 0.0005) | 1221 |
| explicit-route:top2:typeprior − none | property | +0.0980 [+0.0794, +0.1158] (p 0.0005) | 1221 |
| explicit-route:top2:typeprior − none | paraphrase | +0.1237 [+0.0942, +0.1532] (p 0.0005) | 1221 |
| explicit-route:top2:typeprior − none | entailment | +0.1114 [+0.0885, +0.1351] (p 0.0005) | 1221 |
| explicit-route:lm:typeprior − definition | property | -0.0210 [-0.0341, -0.0079] (p 0.0010) | 1221 |
| explicit-route:lm:typeprior − definition | paraphrase | -0.0483 [-0.0704, -0.0262] (p 0.0005) | 1221 |
| explicit-route:lm:typeprior − definition | entailment | +0.0057 [-0.0098, +0.0205] (p 0.4759) | 1221 |
| explicit-rel:teacher − explicit-rel:typeprior | property | -0.0137 [-0.0229, -0.0044] (p 0.0055) | 1221 |
| explicit-rel:teacher − explicit-rel:typeprior | paraphrase | -0.0123 [-0.0303, +0.0049] (p 0.1780) | 1221 |
| explicit-rel:teacher − explicit-rel:typeprior | entailment | -0.0213 [-0.0336, -0.0106] (p 0.0010) | 1221 |
| explicit-rel:teacher − none | property | +0.0890 [+0.0726, +0.1059] (p 0.0005) | 1221 |
| explicit-rel:teacher − none | paraphrase | +0.1196 [+0.0925, +0.1450] (p 0.0005) | 1221 |
| explicit-rel:teacher − none | entailment | +0.0688 [+0.0483, +0.0901] (p 0.0005) | 1221 |
| explicit-route:lm:teacher − none | property | +0.0890 [+0.0729, +0.1051] (p 0.0005) | 1221 |
| explicit-route:lm:teacher − none | paraphrase | +0.1302 [+0.1032, +0.1564] (p 0.0005) | 1221 |
| explicit-route:lm:teacher − none | entailment | +0.0753 [+0.0532, +0.0975] (p 0.0005) | 1221 |
| explicit-route:lm:teacher − definition | property | -0.0306 [-0.0426, -0.0183] (p 0.0005) | 1221 |
| explicit-route:lm:teacher − definition | paraphrase | -0.0483 [-0.0713, -0.0246] (p 0.0005) | 1221 |
| explicit-route:lm:teacher − definition | entailment | -0.0188 [-0.0344, -0.0033] (p 0.0180) | 1221 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − none | property | +0.2087 [+0.1773, +0.2409] (p 0.0005) | 393 |
| definition − none | paraphrase | +0.2850 [+0.2392, +0.3333] (p 0.0005) | 393 |
| definition − none | entailment | +0.1192 [+0.0867, +0.1518] (p 0.0005) | 369 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lm:typeprior − none | property | +0.1040 [+0.0876, +0.1212] (p 0.0005) | 1221 |
| explicit-route:lm:typeprior − none | entailment | +0.0418 [+0.0270, +0.0573] (p 0.0005) | 1221 |
| explicit-route:lm:oracle − explicit-rel:oracle | property | -0.0614 [-0.0784, -0.0453] (p 0.0005) | 1221 |
| explicit-route:lm:oracle − explicit-rel:oracle | entailment | -0.0311 [-0.0418, -0.0221] (p 0.0005) | 1221 |

## t5 new — Qwen3-0.6B-Base-lora-C0p (2 seeds; 1229 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.7423 | 0.6838 | 0.8008 | 0.8703 | 0.8750 | 0.7929 | 0.8788 |
| explicit-rel-wrong:oracle | 0.1787 | 0.4598 | 0.4408 | 0.1947 | 0.5108 | 0.1853 | 0.2111 |
| explicit-rel:oracle | 0.7251 | 0.7155 | 0.8592 | 0.8432 | 0.9117 | 0.7954 | 0.8625 |
| explicit-rel:typeprior | 0.6929 | 0.6988 | 0.8608 | 0.8080 | 0.9083 | 0.7553 | 0.8181 |
| explicit-route:lex:oracle | 0.5291 | 0.6151 | 0.5133 | 0.5554 | 0.5800 | 0.6131 | 0.6402 |
| explicit-route:lex:typeprior | 0.5007 | 0.6024 | 0.5200 | 0.5263 | 0.5825 | 0.5781 | 0.6037 |
| explicit-route:lm:oracle | 0.2536 | 0.4333 | 0.5117 | 0.2650 | 0.5725 | 0.2716 | 0.2775 |
| explicit-route:lm:typeprior | 0.2521 | 0.4326 | 0.5133 | 0.2626 | 0.5717 | 0.2697 | 0.2744 |
| explicit-route:top2:oracle | 0.4160 | 0.5468 | 0.6017 | 0.4390 | 0.6533 | 0.4725 | 0.4984 |
| explicit-route:top2:typeprior | 0.3844 | 0.5196 | 0.5792 | 0.4089 | 0.6400 | 0.4331 | 0.4581 |
| explicit-sym:oracle | 0.7529 | 0.7243 | 0.8583 | 0.8491 | 0.9108 | 0.7862 | 0.8397 |
| none | 0.1964 | 0.3738 | 0.4633 | 0.2310 | 0.5433 | 0.1991 | 0.2317 |

Router accuracy (the asked relation chosen, among items whose frame holds it): explicit-route:lex:oracle 0.392; explicit-route:lex:typeprior 0.391; explicit-route:lm:oracle 0.198; explicit-route:lm:typeprior 0.205; explicit-route:top2:oracle 0.389; explicit-route:top2:typeprior 0.356

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.5288 [+0.5144, +0.5438] (p 0.0005, Holm 0.0030) | 3058 |
| explicit-rel:oracle − none | paraphrase | +0.3417 [+0.3205, +0.3640] (p 0.0005, Holm 0.0030) | 3058 |
| explicit-rel:oracle − none | entailment | +0.3958 [+0.3658, +0.4267] (p 0.0005, Holm 0.0030) | 1200 |
| explicit-rel:oracle − explicit-sym:oracle | property | -0.0278 [-0.0410, -0.0152] (p 0.0005, Holm 0.0030) | 3058 |
| explicit-rel:oracle − explicit-sym:oracle | paraphrase | -0.0088 [-0.0255, +0.0075] (p 0.3164, Holm 0.6328) | 3058 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.0008 [+0.0000, +0.0025] (p 0.7323, Holm 0.7323) | 1200 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lm:typeprior − none | property | +0.0558 [+0.0455, +0.0669] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − none | paraphrase | +0.0589 [+0.0396, +0.0785] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − none | entailment | +0.0500 [+0.0208, +0.0775] (p 0.0015) | 1200 |
| explicit-route:lm:typeprior − definition | property | -0.4902 [-0.5065, -0.4740] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − definition | paraphrase | -0.2511 [-0.2734, -0.2292] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − definition | entailment | -0.2875 [-0.3175, -0.2583] (p 0.0005) | 1200 |
| explicit-rel:typeprior − none | property | +0.4966 [+0.4820, +0.5113] (p 0.0005) | 3058 |
| explicit-rel:typeprior − none | paraphrase | +0.3250 [+0.3035, +0.3476] (p 0.0005) | 3058 |
| explicit-rel:typeprior − none | entailment | +0.3975 [+0.3675, +0.4275] (p 0.0005) | 1200 |
| explicit-rel-wrong:oracle − none | property | -0.0177 [-0.0278, -0.0075] (p 0.0025) | 3058 |
| explicit-rel-wrong:oracle − none | paraphrase | +0.0860 [+0.0651, +0.1073] (p 0.0005) | 3058 |
| explicit-rel-wrong:oracle − none | entailment | -0.0225 [-0.0550, +0.0117] (p 0.1955) | 1200 |
| definition − none | property | +0.5459 [+0.5312, +0.5607] (p 0.0005) | 3058 |
| definition − none | paraphrase | +0.3100 [+0.2871, +0.3326] (p 0.0005) | 3058 |
| definition − none | entailment | +0.3375 [+0.3075, +0.3675] (p 0.0005) | 1200 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − none | property | +0.5938 [+0.5783, +0.6090] (p 0.0005) | 2458 |
| definition − none | paraphrase | +0.3743 [+0.3487, +0.3991] (p 0.0005) | 2458 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.6122 [+0.5968, +0.6285] (p 0.0005) | 3058 |
| explicit-rel:oracle − none | entailment | +0.3683 [+0.3375, +0.3983] (p 0.0005) | 1200 |
| explicit-rel:oracle − explicit-sym:oracle | property | -0.0059 [-0.0183, +0.0067] (p 0.3884) | 3058 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.0008 [+0.0000, +0.0025] (p 0.7303) | 1200 |

## t5 new — Qwen3-1.7B-Base-lora-C0p (2 seeds; 1229 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.8108 | 0.6851 | 0.8825 | 0.8769 | 0.9342 | 0.8265 | 0.9022 |
| explicit-rel-wrong:oracle | 0.1553 | 0.4284 | 0.4275 | 0.1900 | 0.4883 | 0.1587 | 0.1998 |
| explicit-rel:oracle | 0.7966 | 0.6841 | 0.8775 | 0.8800 | 0.9167 | 0.8391 | 0.9392 |
| explicit-rel:typeprior | 0.7672 | 0.6681 | 0.8742 | 0.8468 | 0.9183 | 0.8037 | 0.8975 |
| explicit-route:lex:oracle | 0.5762 | 0.5546 | 0.5408 | 0.6202 | 0.5858 | 0.6709 | 0.7234 |
| explicit-route:lex:typeprior | 0.5507 | 0.5412 | 0.5383 | 0.5930 | 0.5900 | 0.6391 | 0.6890 |
| explicit-route:lm:oracle | 0.3004 | 0.4179 | 0.5483 | 0.3278 | 0.5917 | 0.3283 | 0.3592 |
| explicit-route:lm:typeprior | 0.2979 | 0.4173 | 0.5492 | 0.3275 | 0.5883 | 0.3267 | 0.3588 |
| explicit-route:top2:oracle | 0.4261 | 0.4771 | 0.6408 | 0.4644 | 0.6825 | 0.4870 | 0.5291 |
| explicit-route:top2:typeprior | 0.3960 | 0.4588 | 0.6133 | 0.4344 | 0.6675 | 0.4514 | 0.4929 |
| explicit-sym:oracle | 0.7794 | 0.7077 | 0.8775 | 0.8273 | 0.9175 | 0.7724 | 0.8255 |
| none | 0.1818 | 0.3636 | 0.4600 | 0.2266 | 0.5492 | 0.1753 | 0.2323 |

Router accuracy (the asked relation chosen, among items whose frame holds it): explicit-route:lex:oracle 0.392; explicit-route:lex:typeprior 0.391; explicit-route:lm:oracle 0.200; explicit-route:lm:typeprior 0.206; explicit-route:top2:oracle 0.397; explicit-route:top2:typeprior 0.366

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.6148 [+0.6020, +0.6277] (p 0.0005, Holm 0.0030) | 3058 |
| explicit-rel:oracle − none | paraphrase | +0.3205 [+0.2973, +0.3437] (p 0.0005, Holm 0.0030) | 3058 |
| explicit-rel:oracle − none | entailment | +0.4175 [+0.3875, +0.4475] (p 0.0005, Holm 0.0030) | 1200 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.0172 [+0.0046, +0.0298] (p 0.0075, Holm 0.0150) | 3058 |
| explicit-rel:oracle − explicit-sym:oracle | paraphrase | -0.0235 [-0.0409, -0.0059] (p 0.0045, Holm 0.0135) | 3058 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | +0.0000 [-0.0025, +0.0025] (p 1.0000, Holm 1.0000) | 1200 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lm:typeprior − none | property | +0.1161 [+0.1037, +0.1284] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − none | paraphrase | +0.0536 [+0.0337, +0.0736] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − none | entailment | +0.0892 [+0.0608, +0.1158] (p 0.0005) | 1200 |
| explicit-route:lm:typeprior − definition | property | -0.5129 [-0.5301, -0.4961] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − definition | paraphrase | -0.2678 [-0.2901, -0.2456] (p 0.0005) | 3058 |
| explicit-route:lm:typeprior − definition | entailment | -0.3333 [-0.3617, -0.3058] (p 0.0005) | 1200 |
| explicit-rel:typeprior − none | property | +0.5853 [+0.5724, +0.5989] (p 0.0005) | 3058 |
| explicit-rel:typeprior − none | paraphrase | +0.3044 [+0.2812, +0.3270] (p 0.0005) | 3058 |
| explicit-rel:typeprior − none | entailment | +0.4142 [+0.3833, +0.4450] (p 0.0005) | 1200 |
| explicit-rel-wrong:oracle − none | property | -0.0265 [-0.0365, -0.0167] (p 0.0005) | 3058 |
| explicit-rel-wrong:oracle − none | paraphrase | +0.0647 [+0.0438, +0.0853] (p 0.0005) | 3058 |
| explicit-rel-wrong:oracle − none | entailment | -0.0325 [-0.0650, -0.0008] (p 0.0455) | 1200 |
| definition − none | property | +0.6290 [+0.6159, +0.6416] (p 0.0005) | 3058 |
| definition − none | paraphrase | +0.3215 [+0.2989, +0.3434] (p 0.0005) | 3058 |
| definition − none | entailment | +0.4225 [+0.3925, +0.4525] (p 0.0005) | 1200 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − none | property | +0.6511 [+0.6371, +0.6656] (p 0.0005) | 2458 |
| definition − none | paraphrase | +0.3434 [+0.3190, +0.3678] (p 0.0005) | 2458 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:oracle − none | property | +0.6534 [+0.6385, +0.6679] (p 0.0005) | 3058 |
| explicit-rel:oracle − none | entailment | +0.3675 [+0.3358, +0.3975] (p 0.0005) | 1200 |
| explicit-rel:oracle − explicit-sym:oracle | property | +0.0526 [+0.0407, +0.0649] (p 0.0005) | 3058 |
| explicit-rel:oracle − explicit-sym:oracle | entailment | -0.0008 [-0.0042, +0.0017] (p 0.8098) | 1200 |

## t5 new — SmolLM2-360M-full-C5 (3 seeds; 1229 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.8568 | 0.8411 | 0.8900 | 0.9266 | 0.9344 | 0.8881 | 0.9390 |
| explicit-rel-wrong:oracle | 0.1712 | 0.5858 | 0.4617 | 0.2041 | 0.5039 | 0.1781 | 0.2151 |
| explicit-rel:oracle | 0.8450 | 0.8254 | 0.7994 | 0.8975 | 0.8556 | 0.9079 | 0.9585 |
| explicit-rel:typeprior | 0.8107 | 0.8066 | 0.7978 | 0.8604 | 0.8461 | 0.8656 | 0.9121 |
| explicit-route:lex:oracle | 0.6177 | 0.6904 | 0.6039 | 0.6416 | 0.5922 | 0.7166 | 0.7391 |
| explicit-route:lex:typeprior | 0.5886 | 0.6756 | 0.6039 | 0.6127 | 0.5911 | 0.6791 | 0.7022 |
| explicit-route:lm:oracle | 0.3065 | 0.5422 | 0.6128 | 0.3458 | 0.5928 | 0.3278 | 0.3712 |
| explicit-route:lm:typeprior | 0.3037 | 0.5404 | 0.6139 | 0.3434 | 0.5917 | 0.3247 | 0.3679 |
| explicit-route:top2:oracle | 0.4965 | 0.6492 | 0.6611 | 0.5446 | 0.6928 | 0.5724 | 0.6231 |
| explicit-route:top2:typeprior | 0.4518 | 0.6313 | 0.6533 | 0.5033 | 0.6789 | 0.5194 | 0.5728 |
| explicit-sym:oracle | 0.6872 | 0.7670 | 0.7978 | 0.7557 | 0.8572 | 0.7135 | 0.7676 |
| none | 0.1896 | 0.3442 | 0.5156 | 0.2245 | 0.5350 | 0.1839 | 0.2262 |

Router accuracy (the asked relation chosen, among items whose frame holds it): explicit-route:lex:oracle 0.392; explicit-route:lex:typeprior 0.391; explicit-route:lm:oracle 0.207; explicit-route:lm:typeprior 0.212; explicit-route:top2:oracle 0.488; explicit-route:top2:typeprior 0.440

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lm:typeprior − none | property | +0.1141 [+0.1028, +0.1258] (p 0.0005, Holm 0.0030) | 4587 |
| explicit-route:lm:typeprior − none | paraphrase | +0.1962 [+0.1777, +0.2143] (p 0.0005, Holm 0.0030) | 4587 |
| explicit-route:lm:typeprior − none | entailment | +0.0983 [+0.0739, +0.1228] (p 0.0005, Holm 0.0030) | 1800 |
| explicit-route:lm:oracle − explicit-rel:oracle | property | -0.5385 [-0.5518, -0.5253] (p 0.0005, Holm 0.0030) | 4587 |
| explicit-route:lm:oracle − explicit-rel:oracle | paraphrase | -0.2832 [-0.2993, -0.2679] (p 0.0005, Holm 0.0030) | 4587 |
| explicit-route:lm:oracle − explicit-rel:oracle | entailment | -0.1867 [-0.2117, -0.1617] (p 0.0005, Holm 0.0030) | 1800 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lex:oracle − explicit-rel:oracle | property | -0.2273 [-0.2381, -0.2162] (p 0.0005) | 4587 |
| explicit-route:lex:oracle − explicit-rel:oracle | paraphrase | -0.1349 [-0.1478, -0.1223] (p 0.0005) | 4587 |
| explicit-route:lex:oracle − explicit-rel:oracle | entailment | -0.1956 [-0.2211, -0.1694] (p 0.0005) | 1800 |
| explicit-route:top2:oracle − explicit-rel:oracle | property | -0.3485 [-0.3606, -0.3359] (p 0.0005) | 4587 |
| explicit-route:top2:oracle − explicit-rel:oracle | paraphrase | -0.1761 [-0.1910, -0.1615] (p 0.0005) | 4587 |
| explicit-route:top2:oracle − explicit-rel:oracle | entailment | -0.1383 [-0.1611, -0.1150] (p 0.0005) | 1800 |
| explicit-route:lm:oracle − explicit-sym:oracle | property | -0.3806 [-0.3973, -0.3635] (p 0.0005) | 4587 |
| explicit-route:lm:oracle − explicit-sym:oracle | paraphrase | -0.2248 [-0.2431, -0.2065] (p 0.0005) | 4587 |
| explicit-route:lm:oracle − explicit-sym:oracle | entailment | -0.1850 [-0.2106, -0.1600] (p 0.0005) | 1800 |
| explicit-route:lm:typeprior − explicit-rel:typeprior | property | -0.5070 [-0.5203, -0.4938] (p 0.0005) | 4587 |
| explicit-route:lm:typeprior − explicit-rel:typeprior | paraphrase | -0.2662 [-0.2821, -0.2505] (p 0.0005) | 4587 |
| explicit-route:lm:typeprior − explicit-rel:typeprior | entailment | -0.1839 [-0.2078, -0.1589] (p 0.0005) | 1800 |
| explicit-route:lex:typeprior − none | property | +0.3991 [+0.3849, +0.4137] (p 0.0005) | 4587 |
| explicit-route:lex:typeprior − none | paraphrase | +0.3314 [+0.3135, +0.3490] (p 0.0005) | 4587 |
| explicit-route:lex:typeprior − none | entailment | +0.0883 [+0.0633, +0.1133] (p 0.0005) | 1800 |
| explicit-route:top2:typeprior − none | property | +0.2623 [+0.2477, +0.2765] (p 0.0005) | 4587 |
| explicit-route:top2:typeprior − none | paraphrase | +0.2871 [+0.2686, +0.3056] (p 0.0005) | 4587 |
| explicit-route:top2:typeprior − none | entailment | +0.1378 [+0.1117, +0.1644] (p 0.0005) | 1800 |
| explicit-route:lm:typeprior − definition | property | -0.5531 [-0.5669, -0.5390] (p 0.0005) | 4587 |
| explicit-route:lm:typeprior − definition | paraphrase | -0.3006 [-0.3176, -0.2832] (p 0.0005) | 4587 |
| explicit-route:lm:typeprior − definition | entailment | -0.2761 [-0.2989, -0.2533] (p 0.0005) | 1800 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| definition − none | property | +0.7042 [+0.6912, +0.7163] (p 0.0005) | 3687 |
| definition − none | paraphrase | +0.5094 [+0.4920, +0.5270] (p 0.0005) | 3687 |

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-route:lm:typeprior − none | property | +0.1188 [+0.1073, +0.1297] (p 0.0005) | 4587 |
| explicit-route:lm:typeprior − none | entailment | +0.0567 [+0.0333, +0.0811] (p 0.0005) | 1800 |
| explicit-route:lm:oracle − explicit-rel:oracle | property | -0.5518 [-0.5653, -0.5378] (p 0.0005) | 4587 |
| explicit-route:lm:oracle − explicit-rel:oracle | entailment | -0.2628 [-0.2900, -0.2356] (p 0.0005) | 1800 |

