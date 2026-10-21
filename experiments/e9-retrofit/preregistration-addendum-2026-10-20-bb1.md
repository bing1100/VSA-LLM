# E9 — addendum of 2026-10-20 (B-B1): the bit-matched unstructured side module under 4-bit PTQ

**Status:** pre-registration. Written on 2026-10-20, **before any non-smoke run of the arm**; it is to be committed before the queue script runs (the agent that wrote it did not commit). Labelled CPU smokes
ran before this file was written (§9). They checked the code path and change nothing below. Any later change goes in §11, with
its date and reason.

**Origin:** the author approved this on 2026-10-20 ("approve all … run them and queue"). It is item 7 of the 2026-10-11
addendum, which the author deferred on 2026-10-11 and 2026-10-15 (`preregistration-addendum-2026-10-11.md`, header). It is
also control B-B1 of the novelty check (`manuscript/novelty-check-2026-10.md` §3.4). Every claim-B write-up names it as
pending: `manuscript/claims.md` E9.B1, `manuscript/status/10-20-2026/07-e9-quantization-and-editing.md` §07.1 and
`manuscript/paper-readiness.md` (B stays out of the abstract until B-B1–B-B3 pass).

**Ownership:** R9, `claims.md`, `draft.md` and the quantization write-ups belong to workplace-9a. This block writes its own
report folders (`report/<stage>-bb1`, `report/bb1`) and edits none of those files.

## 1. Question

Under INT4 weight-only PTQ of the host, C5 keeps its advantage over C0′ on T5, keeps part of it on T4, and gains more on T7
(R9 "Dimension 2", "T4 at 3 seeds", "T7"). A published mechanism predicts this for *any* module outside the quantized linear
weights:
- 4-bit PTQ snaps small full-fine-tuning deltas back to the original weights (Zhang et al., ICLR 2025);
- an update held in a separate LoRA module survives (Abitante et al., IJCNN 2026).

Does the **structured** channel keep more of its advantage under INT4 than a module with the **same number of parameters and
the same precision but no structure**? Or is claim B only "extra higher-precision capacity next to a quantized host"?

## 2. Design, taken from the sources

Novelty check B-B1 lists three parts:
- **(a)** C0′ plus an FP16 LoRA, or an FP16 free per-concept table, with the same byte budget as the channel, trained on the
  same schedule;
- **(b)** C2 at the channel's precision in each variant (A: both FP16; B: both INT4, same group size);
- **(c)** the channel's bytes spent on host precision instead (salient FP16 columns, or one layer kept at INT8).

The claims ledger names the same: "FP16 LoRA or free table at equal bytes; C2 at the channel's precision".

- **What already exists.** C2 is a free per-entry table matched to C5's parameters (`e4_plan.matched_sizes`). It is already
  scored at variant A (FP16 table next to the INT4 host) and variant B (table at INT4) on every stage (`quant/<stage>`). Its
  claim-B rescorings (RTN, HQQ, NF4, GPTQ, AWQ, channel off) also exist for T5, T4 and T7 at 360M, seeds 1–3. So part (b) and
  the free-table half of (a) are done. They enter the analysis below as the comparison `C2 vs C0L` and as R9's `C5 vs C2`.
- **What is missing is the unkeyed module of (a).** C2 is still keyed to the linked terms (one row per entry, injected at the
  term's last subtoken). The arm that tests "extra capacity outside the quantized weights" without any term structure is a LoRA
  beside the fully trained host.
- **Part (c) is not run.** It asks a different question, about host precision rather than a separate module. It is left for
  later.

**The arm, `C0L`** (`e9_plan.SIDE_ARMS`; `training.lm` key `model.side_lora`). The simplest faithful version, with each choice
the sources left open:

| choice | fixed as | why |
|---|---|---|
| host | C0′'s recipe: SmolLM2-360M fully fine-tuned, host lr 3e-5, the same 50M tokens, batches (a function of data seed and step), windows and seeds 1–3 | "C0′ plus …"; pairs every window with C0′, C2 and C5 |
| module | LoRA adapters on every default projection of every decoder layer (q, k, v, o, gate, up, down: 224 adapters), unmerged | B-B1 names an FP16 LoRA; the published mechanism (Abitante et al.) is a LoRA; no ontology, no composition, not keyed to terms |
| size | C5's channel's floating-point parameters on the same track and host, counted from the stage's own C5 config (`e9_plan.channel_parameters`, identical to `e4_quant`'s channel bytes ÷ 2): rank r on every adapter, r + 1 in the first k layers, the closest count (`side_lora_spec`) | "same byte budget"; one uniform rank misses by up to 8% (T4: rank 4 = 0.92×, rank 5 = 1.15×) |
| scale | α = rank (update `B A`, scale 1); A ~ N(0, 1/in), B = 0 | the rank pattern does not change the update scale; identity at the start, as every LoRA here |
| optimizer | the channel's parameter group: `train.lr` 1e-3, weight decay 0.1, the same warmup and cosine schedule | "trained on the same schedule" as the channel it replaces; at the host's 3e-5 a zero-initialised adapter would barely move, and the control would be a strawman |
| precision | as the channel: fp32 master weights under bf16 autocast, 16 bits in the byte count, next to the INT4 host (variant A); quantized with the channel's variant-B scheme (asymmetric RTN, the host's group size; `e4_quant.quantize_channel`) in `int4-B` | B-B1 (a) and (b) |
| integer bytes | not matched | the channel's frame schedule (68 KB on T5, 1.2 MB on T4) is the ontology itself, which the control by design lacks |

The resulting patterns (from the stages' C5 configs, 2026-10-20):

| stage | C5 channel parameters | C0L rank pattern | C0L parameters | ratio |
|---|---:|---|---:|---:|
| T5 | 1,159,803 | 2, + 1 in layers 0–3 | 1,153,280 | 0.994 |
| T4 | 2,360,107 | 4, + 1 in layers 0–10 | 2,357,440 | 0.999 |
| T7 | 1,104,891 | 2, + 1 in layer 0 | 1,102,400 | 0.998 |

**Evaluation:** only the claim-B rescoring (`e9_rescore` profile `side`), on the run's own evaluation windows:
- `ref` and `ref-off` (side module off: adapters scaled to 0);
- `int4-A` (torchao RTN, group 64, side module FP16) and `int4-A-off`;
- `int4-B` (side module quantized too);
- `int4-hqq`, `int4-nf4`, `int4-gptq` and `int4-awq` (simulated; GPTQ / AWQ calibrated on 64 training windows).

No zero-shot or editing items: the arm has no channel. P0, C0′, C2 and C5 already have their `controls` rescorings. Variant B of
C2 and C5 is read from the stage's `e4_quant` windows (same windows, checked by start and count).

**Code** (all opt-in; recorded runs are unchanged):
- `integrations.transformers`: `add_side_lora`, `lora_units`, `side_lora_plan`, `side_lora_off`, `LoRALinear.side_module`;
- `training.lm`: the `model.side_lora` key, its optimizer group and the manifest record;
- `evaluation.quantization.merge_lora`: side adapters are never merged;
- `e4_quant`: variant B quantizes the adapters' own tensors, and the byte count follows;
- `e9_rescore`: profile `side`, and `-off` switches the side module off;
- `e9_plan`: `SIDE_ARMS`, `channel_parameters`, `side_lora_spec`, batch tag `bb1`, estimates;
- new module `e9_bb1`: the analysis and report;
- tests: `tests/test_e9_pq.py` (`test_side_*`, `test_e9_plan_side_arm`, `test_bb1_report_on_synthetic_rescores`).

## 3. Tracks and seeds

T5, T4 and T7, SmolLM2-360M, seeds 1–3, as in R9's claim-B tables. These are the three stages with 3-seed rescorings of P0,
C0′, C2 and C5. The cost is moderate (§8), so nothing is trimmed.

## 4. Statistic

For a candidate c, a reference r and a quantizer q:
- **DiD** = (c − r)@q − (c − r)@bf16, in nats per token.
- Window sums are pooled over the common seeds; the 95% cluster bootstrap runs over evaluation windows (`e4_quant.retention`,
  as R9's claim-B tables).
- Positive DiD means c's advantage over r shrinks under q.
- The four absolute cells are reported with n (B-B6).

Comparisons (`e9_bb1.COMPARISONS`):
- `C5 vs C0L`;
- `C5 vs C0′` (R9's DiD, reproduced from the same files);
- `C0L vs C0′` (the side module's own INT4 robustness);
- `C2 vs C0L` (keyed table against unkeyed module) and `C5 vs C2`;
- with modules off: `C0L off vs C0′` (the C0L-trained host weights alone), `C0L vs C0L off` (the side module's own
  contribution) and `C5 off vs C0L off` (the two co-trained hosts without their modules), under `int4-A`.

## 5. Primary

**B-B1 primary:** DiD(C5 vs C0L) under `int4-A`, on `after_heldout`, `after_rare_seen` and `after_len3plus`, for T5, T4 and T7.
That is 9 tests, Holm-adjusted together. `int4-A` is the configuration of R9's claim-B headline. The strata are R9's term
strata; `after_unseen` is too small on T7.

Secondaries (Holm over quantizers within each stage × comparison × stratum):
- the same DiD under GPTQ and AWQ;
- the same DiD under `int4-B`;
- DiD(C0L vs C0′) on `all` and `unlinked`;
- the modules-off comparisons.

## 6. Predictions and decision reading

**Predictions** (from the mechanism the sources cite and R9's numbers):
- **M (mechanism):** C0L is more INT4-robust than C0′ on domain text: DiD(C0L vs C0′) < 0 on `all` and on `unlinked` under
  `int4-A`, on each track. The adapters train at the channel's learning rate and hold a share of the domain update that RTN does
  not touch. **Mb:** under `int4-B` (adapters quantized) that DiD moves towards 0.
- **B1 (primary, expected outcome):** no primary cell reads "structure".
  - **T5 `after_heldout`:** DiD(C5 vs C0L) > 0. R9 already has DiD(C5 vs C0′) = +0.049*, and C0L's robustness adds to it.
  - **T4:** DiD(C5 vs C0L) ≥ 0. R9's C5 − C0′ DiDs there are ≥ 0.
  - **T7 `after_rare_seen` and `after_len3plus`:** the only candidates. R9 has DiD(C5 vs C0′) = −0.0077* / −0.0080*. The
    prediction is that DiD(C5 vs C0L) is closer to 0 than these, with a CI that includes 0. Confidence is low.
- **B2 (GPTQ):** where RTN's growth vanished under GPTQ (T7), DiD(C5 vs C0L) under `int4-gptq` is ≥ −0.002 with a CI that
  includes 0.

**Reading, per primary cell, fixed now** (`e9_bb1.reading` prints it):

| reading | condition | consequence |
|---|---|---|
| **structure** | DiD(C5 vs C0L) < 0, Holm p < 0.05 in the primary family | C5's advantage over a bit-matched unstructured module grows under RTN on that track and stratum. Claim B may say "more robust than a bit-matched unstructured module (RTN)" there. The "RTN" qualifier may be dropped only where GPTQ or AWQ agree in sign with a CI excluding 0 (novelty check §3.5: B-B1–B-B3) |
| **separate module** | DiD(C5 vs C0′) < 0 (CI excludes 0) but DiD(C5 vs C0L) has a CI including 0 | the growth over C0′ is what any separate module gives. No structure claim: claim B stays a measurement, as now |
| **shrinks** | DiD(C5 vs C0L) > 0, Holm p < 0.05 | C5 keeps less of its advantage over the side module than over C0′. Claim B's "preserved" wording is weakened on that track; report it |
| **kept** | otherwise (CI includes 0) | no change of C5's advantage over the side module; nothing new for claim B |

**Validity gate.** C0L must be a real competitor.
- If C0L's side module contributes nothing at bf16 (`C0L vs C0L off` at `ref`: |gain| < 0.1% on `all`), the control is inert on
  that track.
- The same holds if mechanism M fails while the adapters' contribution is also ≈ 0.
- An inert control makes B-B1 **inconclusive** for that track, not a pass for claim B. A rerun at a higher side-module
  learning rate would need a new addendum.

## 7. Reports

- Per stage: job `<stage>-report-s1-2-3-bb1-SmolLM2-360M` (`e9_bb1 report --stages <stage>`) → `report/<stage>-bb1`.
- All three stages: job `bb1-report-t5-t4-t7` → `report/bb1`. This report carries the primary family's Holm adjustment.
- Both run on the CPU lane (names contain `-report`). Neither writes into `report/<stage>`, `quant/<stage>` or any rescoring
  folder of another arm.

## 8. Cost

The GPU-hours below come from measured durations in `.jobs/*.json` for the same stages, hosts and windows.

| block | jobs | basis | GPU-h |
|---|---:|---|---:|
| C0L training, T5 × 3 | 3 | T5 C0′ / C2 / C5 360M: median 66.6 min; × 1.05 assumed adapter overhead (not yet measured on the GPU) | 3.5 |
| C0L training, T4 × 3 | 3 | T4: median 67.1 min × 1.05 | 3.5 |
| C0L training, T7 × 3 | 3 | T7 (4,096 windows): median 75.9 min × 1.05 | 4.0 |
| C0L rescoring (9 variants) | 9 | the C5 `controls` rescorings (9 variants): T5 7.2, T4 7.5, T7 23.9 min | 1.9 |
| reports | 4 | CPU lane | 0 |
| **total** | **22** | | **≈ 12.9 GPU-h** (`e9_plan --dry-run`: 12.9) |

## 9. Smokes before this commit (CPU only, `CUDA_VISIBLE_DEVICES=`, labelled, not results)

- **Unit tests** (`tests/test_e9_pq.py`, the toy GPT-2 world). They check:
  - the adapters, the rank pattern and the parameter count;
  - side adapters are never merged, while host-mode LoRA still is;
  - `-off` changes the logits;
  - C0L trains with its adapters in the channel's group and the host at `host_lr`, and `load_final` rebuilds the adapters;
  - rescoring on the CPU (`ref`, `ref-off`, `int4-rtn`, `int4-rtn-off`, `int4-gptq`; ref check paired);
  - variant B quantizes only the adapters' tensors, and the byte count follows;
  - the plan writes C0′'s config plus `model.side_lora`; the queue gives training, rescoring and the `-bb1-` report, with no
    `pq` items;
  - `e9_bb1` recovers known DiDs on synthetic rescorings.
  - The whole `tests/test_e9*.py` suite passes.
- **The real T5 stage on SmolLM2-135M** (`e9_plan.write_stage` into a scratch root, overridden to 128-token sequences,
  micro-batch 2, 4 steps, 4 evaluation windows):
  - **Config:** the same code matched C0L to C5's 135M channel. C5 has 1,057,659 parameters; C0L is rank 3, + 1 in layers
    0–13, 210 adapters, 1,058,304 parameters (1.0006×).
  - **Training:** 4 steps; evaluation loss over all tokens 4.479 → 4.402; 4.0 GB peak RSS.
  - **Rescoring** (`e9_rescore score`, 4 calibration windows; 6:10 min, 2.5 GB): `ref` 4.4020 (equal to the trainer's final
    evaluation; ref check paired, maximum window difference 0.0), `ref-off` 4.4150, `int4-rtn` 4.5976, `int4-rtn-off` 4.6168,
    `int4-gptq` 4.4903, `int4-hqq` 4.6205. Every scheme left the adapters unmerged (`lora_merged` 0).
  - **`e4_quant` INT8** on the CPU (torchao INT4 needs CUDA, so `int4-A` / `int4-B` were not run here): all 210 wrapped host
    projections were quantized, and the 210 adapters were kept. Adapter bytes are 2,116,608 at FP16 in variant A and 1,370,800
    quantized in variant B; `int8-A` 4.4653, `int8-B` 4.4653.
- **`e9_bb1 report` on the real T5 / T4 / T7 rescorings of P0, C0′, C2 and C5** (without C0L, 2,000 resamples, 53 s). It
  reproduces R9's DiD(C5 vs C0′) from the same files:
  - T5 `after_heldout`: +0.0488 [+0.0433, +0.0542] (RTN), −0.0006 (GPTQ), +0.0173 (AWQ);
  - T4 `after_heldout`: +0.0020 [−0.0002, +0.0042];
  - T7 `after_rare_seen` / `after_len3plus`: −0.0077 / −0.0080.

  It also reads variant B of C5 from the `e4_quant` windows (T5 held-out +0.0487, ≈ A, as R9 reports).
- **Dry runs:** `e9_plan --dry-run` for T5, T4 and T7 with `--models C0L` wrote the nine configs
  (`experiments/e9-retrofit/configs/{t5,t4,t7}/SmolLM2-360M-full-C0L-s{1,2,3}.yaml`; identical to C0′'s apart from
  `model.side_lora`) and listed 7 jobs per stage (≈ 3.9 / 3.9 / 5.1 GPU-h).
- **Not smoked:** torchao INT4 (`int4-A`, `int4-B`) on the wrapped projections, because it needs CUDA and the GPU was busy. The
  INT8 path above uses the same wrapping and the same `quantize_` call. A failure would show in the first C0L rescoring job,
  which keeps its finished variants on retry (`--resume`). The 1.05 training overhead is an assumption until that job's time
  is known.

## 10. Queue

`experiments/e9-retrofit/queue-commands-bb1.sh` (not run by the agent that wrote it): `e9_plan --models C0L --seeds 1 2 3` on
T5, T4 and T7 at `$PRIO_BB1`, then the cross-stage report at `$PRIO_BB1_REPORT`.

## 11. Changes after this commit

(none)
