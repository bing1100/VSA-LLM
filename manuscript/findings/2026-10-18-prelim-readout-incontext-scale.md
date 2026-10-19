# Preliminary results, 2026-10-18: readout arms, uncapped in-context glossary, 125M scaling seeds

**Recorded:** 2026-10-18 by workplace-fd. These are preliminary. Two of the three items come from finished runs whose
official report jobs are held back on the CPU lane by 26 failed jobs (§4).

## 1. Readout arms: models trained to unbind (decision 60 step 2)

**Status:** owned by the binding program, not this session. The numbers are read from the per-run outputs under
`runs/t5/SmolLM2-360M-full-U5*-s{1,2,3}/{role-twins-t5-smollm2-v1, readout}/report.md` and will be superseded by the
official readout report.

**Measures:**
- **twin contrast:** role-swapped twins, `choice`, the model's own store; chance 0.5;
- **role-query accuracy:** chance 0.043;
- **read MRR:** with the model's own query; **oracle MRR:** with the gold role;
- **off − on:** loss after held-out terms with the readout off minus on.

All values are means over 3 seeds.

| Arm | Twin contrast | Role-query accuracy | Read MRR | Oracle MRR | Off − on, loss after held-out terms |
|---|---:|---:|---:|---:|---:|
| U5 (learned HRR) | 0.501 | 0.123 | 0.125 | 0.980 | +0.015 |
| U5u (unitary) | 0.490 | 0.103 | 0.107 | 0.990 | +0.015 |
| U5sb (bounded spectrum) | 0.497 | 0.103 | 0.108 | 0.990 | +0.016 |
| U5bu (block unitary) | 0.494 | 0.068 | 0.078 | 0.988 | +0.012 |
| U5sl (slotted) | 0.494 | **0.196** | 0.196 | 0.971 | +0.006 |
| U5rf (fixed random) | 0.498 | 0.107 | 0.111 | 0.990 | +0.015 |
| U5tr (translation, control) | 0.488 | 0.120 | 0.115 | 0.938 | +0.007 |
| U5ut (untyped, control) | 0.495 | 0.114 | 0.124 | 0.978 | +0.014 |

**Reading:**
- **R1** ("with a trained readout, binding matters") **looks null.** Every arm is at chance on role-swapped twins,
  including the controls.
- **The store is readable:** oracle MRR 0.94–0.99.
- **The model's own role query is mostly wrong:** 0.07–0.20, against chance 0.043.
- **The readout still lowers loss a little** after held-out terms (+0.006 to +0.016 absolute).
- **Bearing on E14:** this is the P1 null again. A model does not learn to query its store by role through language-model
  training, even with a readout trained to unbind. It strengthens the case for explicit recall (E12, RD-4) and for
  external executors (E14), over implicit use.

## 2. In-context glossary without the cap (addendum 2 §1; report `experiments/e9-retrofit/incontext-full/report/`)

**Setup:** a 4,096-token glossary covering 98.8% of each window's entries, against 45% (T4) and 79% (T5) at the
1,024-token cap.

| Contrast (relative loss) | T4 `after_heldout`, capped → uncapped | T4 `after_len3plus` | T5 `after_heldout` |
|---|---|---|---|
| **method on top of the glossary** (C5 frame − C0′ frame) | −0.36%* → −0.36%* | −1.01%* → −1.01%* | −10.76%* → −10.93%* |
| glossary for the plain model (C0′ frame − C0′) | −0.66%* → −0.25% (n.s.) | −0.52%* → −0.14% (n.s.) | — |

- **The "adds on top of the definition" claim survives full coverage unchanged.** The pre-registration predicted it
  would shrink; it did not.
- **The plain model gains *less* from the longer glossary.** That prediction (the glossary's gain grows with coverage)
  failed. A longer prefix dilutes what a 360M model can use.

## 3. Scaling screen, 125M seeds 1–3 (addendum 2 §3)

Final evaluation losses of `runs/scale-v1/125M-{C0,C5}-s{1,2,3}`; C5 − C0, relative:

| Stratum | s1 | s2 | s3 | Mean (SD) | Pre-registered prediction |
|---|---:|---:|---:|---:|---:|
| all | −0.29% | −0.34% | −0.36% | **−0.33% (0.04)** | about −0.3% |
| after_len3plus | −1.13% | −1.24% | −1.27% | **−1.21% (0.07)** | about −1.1% |

- **Both are as predicted and consistent across seeds.**
- **The phase-2 rule is unchanged** (computed on seed 1): not requested.
- **The report needs a pooled-seed mode:** `e4_scale` still analyses one seed per size.

## 4. Queue

- **26 failed jobs** (2026-10-17) hold back every CPU-lane report: 24 readout-arm `edit-v2` evaluations, and 2 T1
  `freqrows`.
  - The `edit-v2` jobs need `--alias-table …/t5.json`; without it they fall back to WordNet.
  - The `freqrows` jobs need `skip_empty_frames` for empty T1 frames.
  - Reported to workplace-9a, whose area this is.
- **E14 tier P and the RD-12 pilot** start after about 20 GPU-h of the current band, around the evening of 2026-10-19.
