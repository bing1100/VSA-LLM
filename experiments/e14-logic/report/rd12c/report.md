# E14 RD-12c — the RD-12 cycle's read frames under selective recall (pre-registration §17)

PMI choices use the null prompt **without** the context; `raw` columns are uncalibrated choices; `(stated)`: items whose gold answer is in the `stated` reader's frame.

## t4 heldout — SmolLM2-360M-full-C5 (3 seeds; 61 stated property items per seed)

| condition | property | paraphrase | entailment | property raw | entailment raw | property (stated) | property raw (stated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| definition | 0.7298 | 0.7394 | 0.8342 | 0.8494 | 0.9188 | 0.9362 | 0.9945 |
| explicit-rel:oracle | 0.8804 | 0.8629 | 0.9272 | 0.9718 | 0.9898 | 0.9563 | 0.9836 |
| explicit-rel:rd12-cycle-r2 | 0.6926 | 0.6904 | 0.8240 | 0.8049 | 0.9036 | 0.9381 | 0.9508 |
| explicit-rel:rd12-single-r2 | 0.6870 | 0.6937 | 0.8223 | 0.8049 | 0.9019 | 0.9326 | 0.9508 |
| explicit-rel:typeprior | 0.6977 | 0.6633 | 0.8325 | 0.8088 | 0.9069 | 0.9654 | 0.9891 |
| explicit-sym:rd12-cycle-r2 | 0.7078 | 0.6836 | 0.8596 | 0.8071 | 0.9171 | 0.9308 | 0.9508 |
| explicit-sym:rd12-single-r2 | 0.6926 | 0.6836 | 0.8545 | 0.8037 | 0.9137 | 0.9217 | 0.9454 |
| none | 0.6204 | 0.5770 | 0.7124 | 0.7281 | 0.8680 | 0.7395 | 0.8051 |

Property accuracy by the item's relation:

| relation | definition | explicit-rel:oracle | explicit-rel:rd12-cycle-r2 | explicit-rel:rd12-single-r2 | explicit-rel:typeprior | explicit-sym:rd12-cycle-r2 | explicit-sym:rd12-single-r2 | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| has_functional_parent | 1.000 | 1.000 | 0.977 | 0.959 | 0.959 | 0.971 | 0.947 | 0.942 |
| has_role | 0.372 | 0.630 | 0.360 | 0.354 | 0.340 | 0.399 | 0.383 | 0.319 |
| is_a | 0.881 | 0.998 | 0.828 | 0.824 | 0.850 | 0.834 | 0.820 | 0.732 |

### Primary (Holm)

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:rd12-cycle-r2 − explicit-rel:rd12-single-r2 | property | +0.0056 [+0.0006, +0.0113] (p 0.0265, Holm 0.0795) | 591 |
| explicit-rel:rd12-cycle-r2 − explicit-rel:rd12-single-r2 | paraphrase | -0.0034 [-0.0152, +0.0085] (p 0.6963, Holm 1.0000) | 591 |
| explicit-rel:rd12-cycle-r2 − explicit-rel:rd12-single-r2 | entailment | +0.0017 [+0.0000, +0.0051] (p 0.7383, Holm 1.0000) | 591 |

### Secondary

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:rd12-cycle-r2 − none | property | +0.0722 [+0.0474, +0.0970] (p 0.0005) | 591 |
| explicit-rel:rd12-cycle-r2 − none | paraphrase | +0.1134 [+0.0745, +0.1523] (p 0.0005) | 591 |
| explicit-rel:rd12-cycle-r2 − none | entailment | +0.1117 [+0.0795, +0.1455] (p 0.0005) | 591 |
| explicit-rel:rd12-single-r2 − none | property | +0.0666 [+0.0412, +0.0914] (p 0.0005) | 591 |
| explicit-rel:rd12-single-r2 − none | paraphrase | +0.1168 [+0.0778, +0.1574] (p 0.0005) | 591 |
| explicit-rel:rd12-single-r2 − none | entailment | +0.1100 [+0.0778, +0.1421] (p 0.0005) | 591 |
| explicit-rel:rd12-cycle-r2 − explicit-rel:typeprior | property | -0.0051 [-0.0203, +0.0096] (p 0.5269) | 591 |
| explicit-rel:rd12-cycle-r2 − explicit-rel:typeprior | paraphrase | +0.0271 [+0.0017, +0.0541] (p 0.0495) | 591 |
| explicit-rel:rd12-cycle-r2 − explicit-rel:typeprior | entailment | -0.0085 [-0.0271, +0.0102] (p 0.3959) | 591 |
| explicit-rel:oracle − explicit-rel:rd12-cycle-r2 | property | +0.1878 [+0.1602, +0.2166] (p 0.0005) | 591 |
| explicit-rel:oracle − explicit-rel:rd12-cycle-r2 | paraphrase | +0.1726 [+0.1371, +0.2115] (p 0.0005) | 591 |
| explicit-rel:oracle − explicit-rel:rd12-cycle-r2 | entailment | +0.1032 [+0.0761, +0.1320] (p 0.0005) | 591 |
| definition − explicit-rel:rd12-cycle-r2 | property | +0.0372 [+0.0175, +0.0575] (p 0.0020) | 591 |
| definition − explicit-rel:rd12-cycle-r2 | paraphrase | +0.0491 [+0.0135, +0.0863] (p 0.0070) | 591 |
| definition − explicit-rel:rd12-cycle-r2 | entailment | +0.0102 [-0.0135, +0.0355] (p 0.4639) | 591 |
| explicit-sym:rd12-cycle-r2 − explicit-sym:rd12-single-r2 | property | +0.0152 [+0.0051, +0.0254] (p 0.0020) | 591 |
| explicit-sym:rd12-cycle-r2 − explicit-sym:rd12-single-r2 | paraphrase | +0.0000 [-0.0203, +0.0186] (p 1.0000) | 591 |
| explicit-sym:rd12-cycle-r2 − explicit-sym:rd12-single-r2 | entailment | +0.0051 [-0.0051, +0.0152] (p 0.4214) | 591 |

### Stated-only items

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|

### Primary contrasts on raw choices

| a − b | test | difference [95% CI] | n |
|---|---|---|---:|
| explicit-rel:rd12-cycle-r2 − explicit-rel:rd12-single-r2 | property | -0.0000 [-0.0028, +0.0028] (p 1.0000) | 591 |
| explicit-rel:rd12-cycle-r2 − explicit-rel:rd12-single-r2 | entailment | +0.0017 [+0.0000, +0.0051] (p 0.6978) | 591 |

