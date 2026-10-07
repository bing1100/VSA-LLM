**SMOKE TEST — not a result.**

# E11 read-to-learn — t5 new — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Item set `experiments/e11-read-to-learn/items/t5-new-smollm2-v1` (20 terms, 228 items); styles ['prose'] (primary `prose`); channel composes: True. No weight is updated except inside the gradient baseline, whose weights are restored after every term.

## Frames written by each reader

| style | reader | frames | empty | edges | precision | recall | F1 | stated recall | filler recall | relation acc. | relation acc. (ambiguous fillers) | cost / word |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| prose | host | 20 | 0 | 4.75 | 0.947 | 0.634 | 0.759 | 0.634 | 1.000 | 0.980 | 0.000 (2) | generated_tokens 96.0, prompt_tokens 571.7, seconds 0.7 |
| prose | linker | 20 | 0 | 4.10 | 0.744 | 0.430 | 0.545 | 0.430 | 1.000 | 0.741 | 0.300 (30) | forward_passes 15.2, forward_tokens 1734.2, seconds 0.3 |
| prose | linker-all | 20 | 0 | 7.10 | 0.817 | 0.817 | 0.817 | 0.817 | 1.000 | 0.830 | 0.333 (39) | forward_passes 15.2, forward_tokens 1734.2, seconds 0.3 |
| prose | linker-joint | 20 | 0 | 4.00 | 0.812 | 0.458 | 0.586 | 0.458 | 1.000 | 0.813 | 0.464 (28) | forward_passes 23.1, forward_tokens 2598.4, seconds 0.4 |
| prose | none | 20 | 20 | 0.00 | — | 0.000 | — | 0.000 | 1.000 | — | — (0) | — |
| prose | oracle | 20 | 0 | 7.10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (39) | — |
| prose | pattern | 20 | 0 | 1.00 | 1.000 | 0.141 | 0.247 | 0.141 | 1.000 | 1.000 | — (0) | — |
| prose | random | 20 | 0 | 7.10 | 0.063 | 0.063 | 0.063 | 0.063 | 1.000 | 1.000 | — (0) | — |
| prose | stated | 20 | 0 | 7.10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (39) | — |
| prose | typeprior | 20 | 0 | 7.10 | 0.901 | 0.901 | 0.901 | 0.901 | 1.000 | 0.905 | 0.641 (39) | — |

## Item tests (no definition in context unless the condition says `context`)

| condition | property | property_new | entailment | paraphrase | statement_accuracy | statement_loss |
|---|---:|---:|---:|---:|---:|---:|
| prose|context | 0.6064 | 0.6824 | 0.6250 | 0.4681 | 0.8511 | 2.1519 |
| prose|context+linker | 0.5957 | 0.6689 | 0.6250 | 0.4574 | 0.8617 | 2.2364 |
| prose|context+oracle | 0.6170 | 0.6959 | 0.6500 | 0.5426 | 0.8404 | 2.3393 |
| prose|gradient×1.0 | 0.1755 | 0.1757 | 0.4500 | 0.2979 | 0.1915 | 4.8422 |
| prose|host | 0.2287 | 0.2297 | 0.4500 | 0.3723 | 0.2340 | 4.5369 |
| prose|linker | 0.1862 | 0.1824 | 0.4750 | 0.2872 | 0.1596 | 4.6961 |
| prose|linker-all | 0.2234 | 0.2297 | 0.5000 | 0.3936 | 0.2234 | 4.5634 |
| prose|linker-joint | 0.2128 | 0.1959 | 0.5250 | 0.3404 | 0.2340 | 4.6812 |
| prose|none | 0.1915 | 0.1824 | 0.4250 | 0.2979 | 0.1809 | 4.8828 |
| prose|oracle | 0.2074 | 0.2162 | 0.5250 | 0.3298 | 0.2234 | 4.6178 |
| prose|pattern | 0.1968 | 0.1757 | 0.4750 | 0.3085 | 0.2128 | 4.8522 |
| prose|random | 0.1702 | 0.1689 | 0.3750 | 0.3191 | 0.1596 | 4.7874 |
| prose|stated | 0.2074 | 0.2162 | 0.5250 | 0.3298 | 0.2234 | 4.6178 |
| prose|typeprior | 0.2234 | 0.2230 | 0.5000 | 0.3617 | 0.2234 | 4.6007 |

| style | a − b | test | difference [95% CI] | n | p (Holm within pair) |
|---|---|---|---|---:|---:|
| prose | linker − none | property | -0.0053 [-0.0480, +0.0320] | 94 | 1.0000 |
| prose | linker − none | property_new | +0.0000 [-0.0473, +0.0405] | 74 | 1.0000 |
| prose | linker − none | entailment | +0.0500 [-0.0500, +0.1500] | 40 | 1.0000 |
| prose | linker − none | paraphrase | -0.0106 [-0.0957, +0.0745] | 94 | 1.0000 |
| prose | linker − none | statement_accuracy | -0.0213 [-0.0638, +0.0213] | 94 | 1.0000 |
| prose | linker − none | statement_loss | -0.1867 [-0.2934, -0.0947] | 94 | 0.0120 |
| prose | oracle − none | property | +0.0160 [-0.0427, +0.0798] | 94 | 1.0000 |
| prose | oracle − none | property_new | +0.0338 [-0.0270, +0.1014] | 74 | 1.0000 |
| prose | oracle − none | entailment | +0.1000 [-0.0500, +0.2250] | 40 | 1.0000 |
| prose | oracle − none | paraphrase | +0.0319 [-0.0641, +0.1383] | 94 | 1.0000 |
| prose | oracle − none | statement_accuracy | +0.0426 [-0.0213, +0.1064] | 94 | 1.0000 |
| prose | oracle − none | statement_loss | -0.2650 [-0.4378, -0.1247] | 94 | 0.0120 |
| prose | linker − oracle | property | -0.0213 [-0.0745, +0.0266] | 94 | 1.0000 |
| prose | linker − oracle | property_new | -0.0338 [-0.1014, +0.0203] | 74 | 1.0000 |
| prose | linker − oracle | entailment | -0.0500 [-0.1750, +0.0750] | 40 | 1.0000 |
| prose | linker − oracle | paraphrase | -0.0426 [-0.1277, +0.0319] | 94 | 1.0000 |
| prose | linker − oracle | statement_accuracy | -0.0638 [-0.1170, -0.0213] | 94 | 0.0240 |
| prose | linker − oracle | statement_loss | +0.0783 [-0.0456, +0.2253] | 94 | 1.0000 |
| prose | linker − typeprior | property | -0.0372 [-0.0904, +0.0106] | 94 | 0.5115 |
| prose | linker − typeprior | property_new | -0.0405 [-0.1081, +0.0135] | 74 | 0.5115 |
| prose | linker − typeprior | entailment | -0.0250 [-0.1250, +0.0750] | 40 | 0.8352 |
| prose | linker − typeprior | paraphrase | -0.0745 [-0.1596, +0.0000] | 94 | 0.4396 |
| prose | linker − typeprior | statement_accuracy | -0.0638 [-0.1170, -0.0213] | 94 | 0.0360 |
| prose | linker − typeprior | statement_loss | +0.0954 [-0.0305, +0.2288] | 94 | 0.5115 |
| prose | linker − random | property | +0.0160 [-0.0319, +0.0638] | 94 | 1.0000 |
| prose | linker − random | property_new | +0.0135 [-0.0405, +0.0676] | 74 | 1.0000 |
| prose | linker − random | entailment | +0.1000 [+0.0000, +0.2250] | 40 | 0.8152 |
| prose | linker − random | paraphrase | -0.0319 [-0.1489, +0.0745] | 94 | 1.0000 |
| prose | linker − random | statement_accuracy | +0.0000 [-0.0426, +0.0426] | 94 | 1.0000 |
| prose | linker − random | statement_loss | -0.0912 [-0.2505, +0.0529] | 94 | 1.0000 |
| prose | linker − pattern | property | -0.0106 [-0.0691, +0.0427] | 94 | 1.0000 |
| prose | linker − pattern | property_new | +0.0068 [-0.0608, +0.0676] | 74 | 1.0000 |
| prose | linker − pattern | entailment | +0.0000 [-0.1000, +0.1000] | 40 | 1.0000 |
| prose | linker − pattern | paraphrase | -0.0213 [-0.1066, +0.0638] | 94 | 1.0000 |
| prose | linker − pattern | statement_accuracy | -0.0532 [-0.1064, -0.0106] | 94 | 0.0480 |
| prose | linker − pattern | statement_loss | -0.1561 [-0.3098, -0.0210] | 94 | 0.0599 |
| prose | linker − host | property | -0.0426 [-0.0959, +0.0106] | 94 | 0.4016 |
| prose | linker − host | property_new | -0.0473 [-0.1081, +0.0069] | 74 | 0.4016 |
| prose | linker − host | entailment | +0.0250 [-0.1000, +0.1500] | 40 | 0.8571 |
| prose | linker − host | paraphrase | -0.0851 [-0.1809, +0.0109] | 94 | 0.3916 |
| prose | linker − host | statement_accuracy | -0.0745 [-0.1279, -0.0316] | 94 | 0.0120 |
| prose | linker − host | statement_loss | +0.1592 [+0.0222, +0.3062] | 94 | 0.1499 |
| prose | linker − linker-all | property | -0.0372 [-0.0957, +0.0160] | 94 | 0.4795 |
| prose | linker − linker-all | property_new | -0.0473 [-0.1149, +0.0068] | 74 | 0.4795 |
| prose | linker − linker-all | entailment | -0.0250 [-0.1250, +0.0750] | 40 | 0.8312 |
| prose | linker − linker-all | paraphrase | -0.1064 [-0.1915, -0.0106] | 94 | 0.1359 |
| prose | linker − linker-all | statement_accuracy | -0.0638 [-0.1170, -0.0213] | 94 | 0.0240 |
| prose | linker − linker-all | statement_loss | +0.1327 [+0.0333, +0.2418] | 94 | 0.0500 |
| prose | stated − none | property | +0.0160 [-0.0427, +0.0798] | 94 | 1.0000 |
| prose | stated − none | property_new | +0.0338 [-0.0270, +0.1014] | 74 | 1.0000 |
| prose | stated − none | entailment | +0.1000 [-0.0500, +0.2250] | 40 | 1.0000 |
| prose | stated − none | paraphrase | +0.0319 [-0.0641, +0.1383] | 94 | 1.0000 |
| prose | stated − none | statement_accuracy | +0.0426 [-0.0213, +0.1064] | 94 | 1.0000 |
| prose | stated − none | statement_loss | -0.2650 [-0.4378, -0.1247] | 94 | 0.0120 |
| prose | linker-joint − none | property | +0.0213 [-0.0266, +0.0745] | 94 | 0.9590 |
| prose | linker-joint − none | property_new | +0.0135 [-0.0405, +0.0745] | 74 | 0.9590 |
| prose | linker-joint − none | entailment | +0.1000 [+0.0250, +0.2000] | 40 | 0.1299 |
| prose | linker-joint − none | paraphrase | +0.0426 [-0.0319, +0.1170] | 94 | 0.9351 |
| prose | linker-joint − none | statement_accuracy | +0.0532 [+0.0000, +0.1170] | 94 | 0.2717 |
| prose | linker-joint − none | statement_loss | -0.2016 [-0.3548, -0.0797] | 94 | 0.0240 |
| prose | linker-joint − typeprior | property | -0.0106 [-0.0638, +0.0426] | 94 | 1.0000 |
| prose | linker-joint − typeprior | property_new | -0.0270 [-0.0743, +0.0203] | 74 | 1.0000 |
| prose | linker-joint − typeprior | entailment | +0.0250 [-0.0750, +0.1500] | 40 | 1.0000 |
| prose | linker-joint − typeprior | paraphrase | -0.0213 [-0.1277, +0.0747] | 94 | 1.0000 |
| prose | linker-joint − typeprior | statement_accuracy | +0.0106 [-0.0426, +0.0638] | 94 | 1.0000 |
| prose | linker-joint − typeprior | statement_loss | +0.0805 [-0.0277, +0.1860] | 94 | 0.7912 |
| prose | linker-joint − oracle | property | +0.0053 [-0.0374, +0.0479] | 94 | 1.0000 |
| prose | linker-joint − oracle | property_new | -0.0203 [-0.0676, +0.0203] | 74 | 1.0000 |
| prose | linker-joint − oracle | entailment | +0.0000 [-0.1250, +0.1250] | 40 | 1.0000 |
| prose | linker-joint − oracle | paraphrase | +0.0106 [-0.0851, +0.0957] | 94 | 1.0000 |
| prose | linker-joint − oracle | statement_accuracy | +0.0106 [-0.0532, +0.0745] | 94 | 1.0000 |
| prose | linker-joint − oracle | statement_loss | +0.0634 [-0.0437, +0.1670] | 94 | 1.0000 |
| prose | linker-joint − linker | property | +0.0266 [-0.0266, +0.0851] | 94 | 1.0000 |
| prose | linker-joint − linker | property_new | +0.0135 [-0.0473, +0.0878] | 74 | 1.0000 |
| prose | linker-joint − linker | entailment | +0.0500 [-0.0500, +0.1500] | 40 | 1.0000 |
| prose | linker-joint − linker | paraphrase | +0.0532 [-0.0322, +0.1489] | 94 | 1.0000 |
| prose | linker-joint − linker | statement_accuracy | +0.0745 [+0.0319, +0.1277] | 94 | 0.0120 |
| prose | linker-joint − linker | statement_loss | -0.0149 [-0.1518, +0.1055] | 94 | 1.0000 |
| prose | context − linker | property | +0.4202 [+0.3403, +0.5000] | 94 | 0.0120 |
| prose | context − linker | property_new | +0.5000 [+0.4122, +0.5878] | 74 | 0.0120 |
| prose | context − linker | entailment | +0.1500 [-0.0500, +0.3750] | 40 | 0.2078 |
| prose | context − linker | paraphrase | +0.1809 [+0.0426, +0.3085] | 94 | 0.0320 |
| prose | context − linker | statement_accuracy | +0.6915 [+0.6061, +0.7766] | 94 | 0.0120 |
| prose | context − linker | statement_loss | -2.5442 [-2.8824, -2.2373] | 94 | 0.0120 |
| prose | context+linker − context | property | -0.0106 [-0.0426, +0.0160] | 94 | 1.0000 |
| prose | context+linker − context | property_new | -0.0135 [-0.0473, +0.0203] | 74 | 1.0000 |
| prose | context+linker − context | entailment | +0.0000 [+0.0000, +0.0000] | 40 | 1.0000 |
| prose | context+linker − context | paraphrase | -0.0106 [-0.0745, +0.0532] | 94 | 1.0000 |
| prose | context+linker − context | statement_accuracy | +0.0106 [+0.0000, +0.0319] | 94 | 1.0000 |
| prose | context+linker − context | statement_loss | +0.0845 [+0.0170, +0.1664] | 94 | 0.1199 |
| prose | context − none | property | +0.4149 [+0.3351, +0.4947] | 94 | 0.0120 |
| prose | context − none | property_new | +0.5000 [+0.4122, +0.5811] | 74 | 0.0120 |
| prose | context − none | entailment | +0.2000 [-0.0250, +0.4250] | 40 | 0.1079 |
| prose | context − none | paraphrase | +0.1702 [+0.0426, +0.2872] | 94 | 0.0160 |
| prose | context − none | statement_accuracy | +0.6702 [+0.5745, +0.7553] | 94 | 0.0120 |
| prose | context − none | statement_loss | -2.7309 [-3.0780, -2.4037] | 94 | 0.0120 |
| prose | gradient×1.0 − linker | property | -0.0106 [-0.0532, +0.0319] | 94 | 1.0000 |
| prose | gradient×1.0 − linker | property_new | -0.0068 [-0.0541, +0.0405] | 74 | 1.0000 |
| prose | gradient×1.0 − linker | entailment | -0.0250 [-0.1250, +0.0750] | 40 | 1.0000 |
| prose | gradient×1.0 − linker | paraphrase | +0.0106 [-0.0745, +0.1064] | 94 | 1.0000 |
| prose | gradient×1.0 − linker | statement_accuracy | +0.0319 [-0.0106, +0.0745] | 94 | 1.0000 |
| prose | gradient×1.0 − linker | statement_loss | +0.1461 [+0.0535, +0.2505] | 94 | 0.0240 |
| prose | gradient×1.0 − none | property | -0.0160 [-0.0372, +0.0000] | 94 | 0.5295 |
| prose | gradient×1.0 − none | property_new | -0.0068 [-0.0203, +0.0000] | 74 | 1.0000 |
| prose | gradient×1.0 − none | entailment | +0.0250 [-0.0500, +0.1250] | 40 | 1.0000 |
| prose | gradient×1.0 − none | paraphrase | +0.0000 [-0.0426, +0.0426] | 94 | 1.0000 |
| prose | gradient×1.0 − none | statement_accuracy | +0.0106 [+0.0000, +0.0319] | 94 | 1.0000 |
| prose | gradient×1.0 − none | statement_loss | -0.0405 [-0.0648, -0.0206] | 94 | 0.0120 |

## Gradient baseline

lr 0.001, sgd; general-text loss before 2.3152.

- ×1.0: 5.0 steps / term (573 training tokens), general-text Δ loss mean -0.0004 (max +0.0004), 47s

## Locality

`{"applicable": true, "entries": 2000, "max_abs_row_change": 1.4901161193847656e-07}`

Timings (s): `{"readers": 26.5, "persistence": 110.6, "context": 137.8, "gradient": 47.3}`; total 323s.