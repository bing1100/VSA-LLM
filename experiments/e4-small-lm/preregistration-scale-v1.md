# E4 scaling screen, phase 1 (`scale-v1`): does the composed span channel change from-scratch scaling?

**Status:** design and pre-registration. Written on 2026-10-10 and committed **before any non-smoke `scale-v1` run**. The
labelled CPU smokes (§9) ran before this commit on tiny hosts; they check the pipeline and change nothing in §§1–8. Any later
change is listed in §10 with its date and reason. Phase 1 is **seed 1 only and exploratory**: it decides whether phase 2 is
worth requesting (§6), not whether the channel works.

**Origin:** decision 65 (author, 2026-10-10). Does training from scratch with the ontology-composed span channel change the
scaling behaviour (loss vs tokens and vs model size) compared with raw from-scratch training, and do HRRBERT's HRRAdd / HRRCat
hybrids do better? Evidence so far: the reduced operator screen (`experiments/e4-small-lm/analysis/opscreen-v1/report.md`;
GPT-2-style 50M from scratch, 100M tokens, 3 seeds) found C5 − C0 = −0.44%* on `all`, −0.14%* `unlinked`, −1.32%*
`after_len3plus`, −2.47%* `after_rare`, −0.37%* `after_heldout`; C2 (free table, matched parameters) −0.30% on `all`; the
operators tie; data multiplier k = D_C0(L)/D_cond(L) on `all` 1.024 [1.020, 1.028], `after` 1.057, `after_rare_seen` ≈ 1.16;
projected gaps at 2.5B / 7B tokens with seed CIs spanning 0. In pretrained hosts (E9) WordNet gave no gain at all, so the
from-scratch regime is where the channel helps in-distribution text, as HRRBERT reported (HRRAdd / HRRCat ≈ +2% MLM accuracy
over an unstructured embedding; HRRBase, the structure alone, −17%).

**Code:** `src/vsa_embed/experiments/e4_plan.py` (`--stage scale-v1`: configs, `matched_hybrid_widths`);
`src/vsa_embed/training/lm.py` (`model.size: 20M`; channel modes `compose_add`, `compose_cat`);
`src/vsa_embed/span_channel.py` (`SpanChannel`, `HYBRID_MODES`); `src/vsa_embed/experiments/e4_scale.py` (the report);
tests `tests/test_span_channel.py`, `tests/test_lm_training.py`, `tests/test_e4_plan.py`, `tests/test_e4_scale.py`. Configs
`experiments/e4-small-lm/configs/scale-v1/`; runs `experiments/e4-small-lm/runs/scale-v1/`; report
`experiments/e4-small-lm/analysis/scale-v1/`; queue commands `experiments/e4-small-lm/queue-commands-scale-v1.sh` (not
executed by the author of this document).

## 1. Questions

- **Q1 (primary).** Does C5's data multiplier against C0 shrink as the host grows from 20M to 50M to 125M parameters at a
  fixed 500M-token budget? (The E4 escalation rule's size-trend branch, here with three sizes and a slope.)
- **Q2.** Do HRRBERT's hybrids — a free per-entry row added to (HRRAdd) or concatenated with (HRRCat) the composed row — do
  better than C5, and than C0?
- **Q3.** How much of C5's gain does the free table C2 reach (C2's share), is the gain specific to the ontology's frames at
  50M (C5 vs C5sh, shuffled frames), and does the channel help unlinked text (HRRBERT's in-distribution claim)?

## 2. Design

**Data (as the opscreen).** C3 corpus `~/data/vsa-llm/c3/wordnet-gpt2-v1`: WordNet-linked FineWeb-Edu, GPT-2 BPE, 1.1B
training tokens, ℓ_min 2; ontology of 101,500 link entries over 8,192 atomics and 16 relations, 5,900 entries held out (never
linked in training). Evaluation: the opscreen's 1,024 fixed windows of 1,024 tokens (≈ 1.05M targets), identical across sizes
and conditions, per-window losses saved at every evaluation.

**Hosts** (GPT-2 architecture, tied 50,257-row embedding, context 1,024):

| Size | Layers × width × heads | Parameters (C0) | Non-embedding (N) | Channel share of N (C5) |
|---|---|---:|---:|---:|
| 20M | 6 × 384 × 6 | 30,339,456 | 10,647,552 | 20.7% |
| 50M | 8 × 512 × 8 | 51,475,968 | 25,220,096 | 8.9% |
| 125M | 12 × 768 × 12 | 124,439,808 | 85,056,000 | 2.7% |

"20M" is a nominal label (the token table alone holds 19.3M); N (blocks plus final layer norm) is the size variable of every
trend below.

**Recipe** (the frozen 50M recipe, gates.md D4.0, at every size): 32,768 tokens per optimizer step (micro-batch 32 × 1;
125M 16 × 2, its measured memory limit), AdamW (β 0.9 / 0.95, weight decay 0.1, clip 1.0), bf16 autocast, cosine to 0.1× of
the peak, 500M tokens (15,258 steps; the end is 499,974,144 tokens), seed 1, data seed 1234 (batches are a function of
(data seed, step, micro-step), so conditions are paired within a size; 125M's second micro-batch draws other windows, so data
order differs partly across sizes). **Peak lr ∝ 1/width**, anchored at the frozen 50M value: 2.667e-3 (20M), 2e-3 (50M),
1.333e-3 (125M). The plan's own 50M → 125M ratio (D4.8 configs: 1e-3 → 6e-4) is 0.6, as GPT-3's and Pythia's; 1/width gives
0.67, and 1.333e-3 lies inside the 125M check range {1e-3, 2e-3} that was planned but never run. **Warmup 10M tokens** at
every size (the plan's convention for the 300M / 500M budgets; 2% of the run). Evaluations at 5, 10, 20, 40, 80, 160, 320M
tokens and the end; checkpoints every 10 minutes (bit-exact resume).

**Conditions** (configs written by `e4_plan --stage scale-v1`):

- **C0** no channel. **C5** the opscreen's: HRR operator, composed width 256, attentive composition (key width 8) with the P1
  context (window 8). **C2** free per-entry table of width 22 + projector (`matched_sizes`, as the opscreen).
- **HRRAdd** (`compose_add`) injects `P(c_j + U f_j)`; **HRRCat** (`compose_cat`) injects `P[f_j ; c_j]`: `c_j` is C5's
  attentive composition (same keys, context, operator, frames) at **half C5's width, 128**; `f_j` is a free per-entry row of
  width 11 (C2's table and initialization); `U` (HRRAdd only) lifts it to width 128 (near-isometric at initialization); `P`
  projects to the model width; the gate is C5's. **Held-out entries** have no trained free row and read C2's fallback, the mean
  of the trained rows; their composed part is composed zero-shot, as in C5.
- **C5sh** (50M only): C5 with shuffled frames (every entry reads another entry's frame; matched parameters, wrong structure).

**Parameter matching** (channel parameters, P1 context excluded — it is identical for C5 and the hybrids):

| Size | C5 | C2 (width 22) | HRRAdd (128 + 11) | HRRCat (128 + 11) |
|---|---:|---:|---:|---:|
| 20M | 2,204,611 | 2,242,218 (+1.71%) | 2,220,695 (+0.73%) | 2,223,511 (+0.86%) |
| 50M | 2,237,635 | 2,245,290 (+0.34%) | 2,237,335 (−0.01%) | 2,241,559 (+0.18%) |
| 125M | 2,303,683 | 2,251,434 (−2.27%) | 2,270,615 (−1.44%) | 2,277,655 (−1.13%) |

The free width is the integer that brings the hybrid closest to C5 (one unit of free width is ≈ 4.5% of the budget, so exact
matching is impossible at a fixed composed width); the hybrids are within 1.5% of C5 at every size, closer than C2. Each part
holds about half the budget. HRRBERT's equal halves (d/2 ‖ d/2) were rejected: with a free table over 101,500 entries they
leave both parts ≈ 20-dimensional, i.e. a 20-dimensional HRR. Note that with a trained projector after the join, HRRAdd and
HRRCat are close reparameterizations of one function class (HRRCat's free block of `P` vs HRRAdd's `P·U`); a difference
between them would be an optimization effect.

**Grid and cost** (GPU-hours at the opscreen's measured 50M throughput — C0 69.1k, C2 69.0k, C5 67.7k tokens/s on the RTX
3090 — scaled by 6N with N the total parameter count; hybrids and C5sh at C5's rate; nine evaluations included):

| Block (queue order) | Runs | GPU-h per run | GPU-h |
|---|---|---|---:|
| 20M | C0, C5, C2, HRRAdd, HRRCat | 1.19–1.22 | 6.0 |
| 50M | C0, C5, C2, HRRAdd, HRRCat, C5sh | 2.02–2.06 | 12.3 |
| 125M | C0, C5, C2, HRRAdd, HRRCat | 4.89–4.99 | 24.8 |
| **Total** | 16 | | **43.1** |

The 6N scaling is conservative at 125M (measured C4 throughput 36k tokens/s → 3.9 h per run, 19.4 h for the block, 37.7 h in
total) and probably optimistic at 20M (small matrices run below the 50M utilization). Disk: ≈ 4.5 GB of `final.pt` in all,
≤ 1.5 GB of checkpoint at a time (finished runs delete theirs).

## 3. Measurements and statistics

**Strata** (trainer definitions): `all`, `unlinked`, `after` (8 tokens after a linked span), `after_len3plus` (after spans of
≥ 3 subtokens), `after_rare_seen` (entries linked 1–9 times in training), `after_heldout`, `inside` (subtokens 2..ℓ).

**Data multiplier** `k = D_ref(L) / D_cond(L)`, tokens to reach loss L interpolated in log-tokens between evaluations
(`convergence.tokens_to_loss`), at **the lowest common loss target** (10% of the way from the worse final loss to the better
first loss: `common_targets`' first, the target the E4 escalation rule reads) and, secondary, `k_end` at the worse final loss.
**Single-seed CIs:** 95% percentile cluster bootstrap over the evaluation windows (2,000 resamples, seed 0): each resample
reweights the windows, rebuilds both curves from the per-window loss sums of every evaluation and recomputes target and k.
The same window weights are used at every size (the windows are shared), so the trend's bootstrap is joint. **Seed-inflated
CIs** `⟨ ⟩` add, in quadrature on ln k, the seed SD measured on the opscreen's 3 seeds at 50M × 100M (`--seed-reference`;
assumed size- and budget-independent): C5 vs C0 (used for C5, C5sh, HRRAdd, HRRCat vs C0) 0.0016 on `all`, 0.0015 on
`after_len3plus`; C2 vs C0 0.0084 / 0.0072; between channel conditions the pooled C5 operator variants vs C5, 0.011 / 0.012.
(For comparison, the window-bootstrap SD of ln k for one opscreen seed is 0.0011 on `all` and 0.0037 on `after_len3plus`.)

**Final gaps:** (cond − ref) / ref at the end, the E4 analysis' window bootstrap (10,000 resamples), Holm over the condition
grid within each size and stratum (`e4_report`; references C0, C2, C5sh, C5).

**Size trend:** the OLS slope `b` of ln k on ln N over the three sizes, with its window-bootstrap percentile CI and the
seed-inflated CI `b ± 1.96·√(var_window(b) + s²·Σc_i²)` (`c_i` the OLS weights, `s` the seed SD of ln k above).

**Other quantities:** compute-adjusted k (× measured tokens/s ratio; × the 6N ratio of total parameters, counting every
channel parameter as dense compute — an upper bound on the channel's cost); C2's share of C5's final gap
`(L_C0 − L_C2)/(L_C0 − L_C5)` (window bootstrap; reported only where C5's own gap is Holm-significant); `L(D) = E + B·D^−β`
fits and projected gaps at 2.5B / 7B tokens (point estimates: one seed gives no seed CI).

## 4. Primary endpoint

Per primary stratum **`all`** and **`after_len3plus`**, C5 vs C0: the slope `b` of ln k on ln N (20M, 50M, 125M) with its
seed-inflated 95% CI. Reading: **k shrinks** with size if the CI's upper bound is < 0; **k grows** if its lower bound is > 0;
otherwise no trend is detectable at this precision. Alongside, the **E4 escalation rule (iii)** at 125M (experiments.md §0.13):
the projected gap's CI excludes 0 (not estimable with one seed, so this branch is false in phase 1) or k does not shrink from
the smaller to the larger model (k_125M ≥ k_50M, point estimates, `convergence.escalation_verdict`). Rules (i) (k's CI low
> 1.1, here the seed-inflated window CI) and (ii) (C2 beaten, window-bootstrap final difference) are reported, not used.

## 5. Secondary endpoints

- **S1 hybrids:** HRRAdd − C5 and HRRCat − C5 (final relative gap, Holm, and k against C5) per size on every trend stratum;
  HRRAdd / HRRCat vs C0 (k, slope, final gap).
- **S2 C2's share** of C5's final gap per size (`all`, `after_len3plus`) and C2's k and slope vs C0: how much of the gain
  per-entry capacity alone buys, and whether that changes with size.
- **S3 specificity at 50M:** C5 − C5sh on `after_len3plus`, `after`, `after_rare_seen`, `after_heldout`, `all`.
- **S4 unlinked text** (HRRBERT's in-distribution claim): C5 / HRRAdd / HRRCat − C0 on `unlinked` per size, and its slope.
- **S5 descriptive:** `k_end` and its slope; the trends on `after`, `after_rare_seen`, `after_heldout`, `inside`;
  compute-adjusted k; projected gaps; the per-size E4 report (fits, gate items, figures).

## 6. Phase-2 decision rule

Phase 2 — any of: 350M from scratch (C0, C5 and a qualifying hybrid; ≈ 11k tokens/s measured, ≈ 13 GPU-h per 500M-token
run), seeds 2–3 at 125M (≈ 20 GPU-h for C0 and C5), or the same grid on a natural-domain corpus — is **requested from the
author only if**, on at least one primary stratum (`all`, `after_len3plus`):

1. C5 beats C0 at 125M (final relative difference below 0, Holm-significant), **and**
2. rule (iii) holds at 125M (k_125M ≥ k_50M, or a projected-gap CI excluding 0), **and**
3. shrinkage is not established: the seed-inflated CI of the slope `b` has an upper bound ≥ 0.

Otherwise phase 2 is not requested and the result is reported as a negative scaling result for the channel ("the from-scratch
gain shrinks with model size"). A hybrid joins phase 2 only if it beats C5 (final difference, Holm) on a primary stratum at
≥ 2 of the 3 sizes; otherwise phase 2, if any, carries C5 only. The report computes this verdict mechanically
(`e4_scale.phase2_verdict`); approval stays with the author.

## 7. Predictions (from the opscreen; written before any run)

- **Primary:** k shrinks with size on `all` — the channel adds ≈ 2.2M parameters, 21% of the 20M host's N but 2.7% of the
  125M's — with roughly k_all ≈ 1.04 / 1.025 / 1.01 at 20M / 50M / 125M (slope ≈ −0.014 per e-fold of N, CI below 0). On
  `after_len3plus` the shrinkage is slower (≈ 1.10 / 1.07 / 1.05) and may not be established. Rule (iii) at 125M: false on
  `all`, uncertain on `after_len3plus`. Predicted phase-2 verdict: not requested on `all`; `after_len3plus` decides.
- **Final gaps at 50M × 500M:** C5 − C0 ≈ −0.3% on `all` (−0.44% at 100M tokens; the gap narrows with tokens) and ≈ −1.0% on
  `after_len3plus`; the gaps narrow from 20M to 125M.
- **S1:** HRRAdd and HRRCat within ±0.1% of C5 on `all` and not significantly better than C5 at ≥ 2 sizes (half-width
  composition plus a C2-like table should land between C2 and C5); both beat C0 at every size; HRRAdd ≈ HRRCat. HRRBERT's
  hybrid advantage is not predicted to replicate against the composed channel.
- **S2:** C2's share ≈ 0.6–0.7 on `all` (opscreen 0.68) and ≈ 0.6 on `after_len3plus` (0.58), roughly constant over size.
- **S3:** C5 better than C5sh at 50M on `after_len3plus` (≈ −0.5%, about C5 − C2 in the opscreen) and on `after_rare_seen`.
- **S4:** C5 − C0 on `unlinked` ≈ −0.1% at 50M, shrinking toward 0 at 125M.

## 8. Limitations fixed in advance

One seed per cell: the window CIs measure evaluation noise only, and the seed-inflated CIs borrow seed variance from another
budget (100M tokens) and one size (50M). The recipe is not tuned per size (lr ∝ 1/width; no lr check at 20M or 125M), and a
mistuned size could change k. Curves are read at intermediate checkpoints of one cosine schedule per size, not at the end of
budget-specific schedules. The hybrids follow HRRBERT's structure but not its setting (BERT MLM with token-level embedding
replacement; here span-level injection into a causal LM through a gated projector, matched to C5's parameters with a
half-width composer). The FLOPs adjustment is an upper bound on the channel's cost. 500M tokens is below compute-optimal for
125M (≈ 2.5B), so the size trend is measured at a fixed, sub-optimal budget.

## 9. Smoke checks (labelled SMOKE; not results)

On 2026-10-10, CPU only, on the real C3 corpus and ontology (the GPU was busy): every phase-1 condition (C0, C5, C2,
HRRAdd, HRRCat, C5sh) at sizes `tiny` and `20M`, sequence length 128, 12 steps of 4 windows, 16 evaluation windows, configs
written by the `scale-v1` planner with smoke overrides; each run trained, evaluated at the start and at six later points with
per-window losses and wrote its manifest (channel parameters as in §2's table at width 384; the `tiny` width-64 hybrids are 2.1% below C5). The report
(`e4_scale`) ran on these 12 runs with the opscreen as the seed reference and produced every section (verdict: not
requested — 12 steps train nothing). The seed reference on the real opscreen reproduces the opscreen report's k (C5 vs C0
`all` mean 1.024; per-seed window-curve k equals the metrics-curve k to 1e-7). Outputs stayed in the session scratchpad.

## 10. Amendments

None.
