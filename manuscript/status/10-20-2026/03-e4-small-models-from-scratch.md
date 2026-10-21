# 03 — E4: small GPT-style models trained from scratch with the method

> **Update, 20 October (evening).** C5teach was approved on 20 October. Its training is scripted, and the teacher
> build (cap USD 82.40) waits for the author to start it; see the digest, §5.

**Overview.** E4 is the program's pre-registered core. It asks whether a language model trained from scratch learns
better when every linked multi-token term gets a vector composed from its ontology entry. We follow it through five
explorations:
- **03.1:** the 50M-parameter screens that fixed the recipe and found a small gain at 100M tokens (3 seeds).
- **03.2:** the scaling screen at 20M, 50M and 125M parameters × 500M tokens (seed 1). By its pre-registered rule we do
  not scale further.
- **03.3:** seeds 2–3 at 125M, which confirm the 125M gain.
- **03.4:** two toolkit arms at 50M. A growing dictionary (C5dev) adds nothing. A Claude-read ontology (C5teach) was
  never built.
- **03.5:** what the original E4 plan promised and never ran.

The bottom line: from scratch, the method gives a small, steady gain. It is about 0.3% of loss overall and 1.1–1.4%
after long terms. A model without it needs 1.7–4.5% more training data to match it overall. The gain does not grow
from 50M to 125M, and terms held out of training gain nothing at 500M tokens.

**Terms used throughout.**
- **C0:** the plain baseline, the same GPT-2-style model trained from scratch without the method.
- **C5:** the full method. At the last subtoken of each linked term we add a gated vector. We compose it from the term's
  ontology frame (its relation–filler edges) over a small shared dictionary trained with the model, with attention and
  a local context query.
- **HRR:** holographic reduced representation; C5 binds relation and filler by circular convolution.
- **C2:** a free per-term lookup table with C5's parameter budget (about 2.2–2.3M channel parameters) and no ontology.
- **C5sh:** C5 with shuffled frames (each term reads another term's frame).
- **HRRAdd / HRRCat:** hybrids after the HRRBERT paper. A half-width composed vector is combined with a free per-term
  vector, by addition (HRRAdd) or concatenation (HRRCat). Both are within 1.5% of C5's parameters.
- **C5rf / C5tr / C5ut:** C5 with a fixed random operator, with translation (x + r), or with no binding (untyped).
- **C1:** fixed random span vectors, the span-boundary control.
- **C5dev:** C5 with a growing dictionary (atomics split when their gradients show two usages).
- **C5teach:** C5 with frames that Claude writes from WordNet glosses.
- **Data:** the C3 corpus, FineWeb-Edu text linked to WordNet. It has GPT-2 BPE tokens and 1.1B training tokens. The
  ontology holds 101,500 entries over 8,192 atomics and 16 relations. 5,900 entries are held out (never linked in
  training).
- **Strata** (token subsets of 1,024 fixed evaluation windows, about 1.05M targets):
  - `all`;
  - `unlinked` (locality);
  - `after`: 8 tokens after a linked term;
  - `after_len3plus`: after terms of 3 or more subtokens ("long terms");
  - `after_heldout`: after held-out terms, whose vectors are composed zero-shot;
  - `after_rare_seen`: after terms seen 1–9 times in training (only 643 targets).
- **Numbers:** relative final-loss difference (condition − reference) / reference. **Negative = better.**
  - `[a, b]` is a 95% cluster bootstrap over evaluation windows. It covers evaluation noise only.
  - `⟨a, b⟩` adds the seed variance measured in 03.1.
  - `*` marks Holm-significant differences.
- **k (data multiplier):** tokens C0 needs to reach a loss, divided by tokens the condition needs. k = 1.045 means C0
  needs 4.5% more data.
- **N:** non-embedding parameters. The size trend is the slope of ln k on ln N.

---

# 03.1 — The 50M screens: pilot, recipe and operator screen

**Status:** done · **When:** 2026-10-02 to 2026-10-08 · **Sources:** `reports/R1-sanity-pilot.md`;
`resources/plan-improvement/gates.md` (D4.0); `resources/plan-improvement/execution.md` (decision 54);
`experiments/e4-small-lm/analysis/opscreen-v1/report.md`; `manuscript/update-2026-10-12.md` (E4-1, E4-2)

## Research gap
HRRBERT showed that ontology-composed embeddings help a masked language model trained from scratch. Nobody had tested a
causal model with span-level injection. Before the committed budget, we needed three things: any signal, a sound recipe,
and whether the HRR operator matters.

## Intuition
A model trained from scratch sees a rare term only a few times. If terms share learned atoms through their frames, a
rare term borrows from its relatives, and loss after it drops. If binding carries the gain, other operators do worse.

## Methodology
- **Model:** GPT-2-style 50M (8 layers × 512), context 1,024, the C3 corpus.
- **S0 pilot (R1):** C0, C1, C2, C3 (static bundle), C5; 100M tokens; 2 seeds; 262k tokens per step, lr 1e-3. It was
  exploratory and is never pooled.
- **D4.0 recipe sweep:** C0 only, tokens per step × peak learning rate, seed 1.
- **Operator screen (decision 54, fixed before any run):**
  - arms C0, C2, C5, C5rf, C5tr, C5ut × seeds 1–3, 100M tokens, the frozen recipe;
  - primary: C5 − arm on `after_heldout` and `after` (paired window bootstrap, seeds pooled, Holm over C5ut and C5rf);
  - reading: if C5 does not beat both C5ut and C5rf, the paper says "compositional parameter sharing", not binding;
  - secondary: Spearman correlation on CARD-660 (a rare-word similarity benchmark) for held-out words.

## Experiment
- S0 pilot: 10 runs, about 4.2 GPU-h (2 Oct).
- Recipe sweep: 11 C0 runs at 50M × 100M tokens (2–4 Oct).
- Operator screen: 18 runs at 50M × 100M tokens, about 8 GPU-h (7–8 Oct).

## Results
**Recipe.** The pilot recipe was under-trained. Final C0 loss was 5.63 at 262k tokens per step against 4.22 at the
frozen recipe (32,768 tokens per step, lr 2e-3). The pilot's effect sizes are therefore superseded.

**Operator screen** (3 seeds, 50M × 100M tokens):

| Stratum | C5 − C0 | C2 − C0 | C5 − C2 |
|---|---:|---:|---:|
| all | −0.44% [−0.46, −0.41]* | −0.30% [−0.32, −0.28]* | −0.14% [−0.16, −0.12]* |
| unlinked | −0.14% [−0.16, −0.11]* | −0.10% [−0.12, −0.07]* | −0.04% [−0.06, −0.01]* |
| after_len3plus | −1.32% [−1.40, −1.23]* | −0.76% [−0.83, −0.68]* | −0.56% [−0.64, −0.49]* |
| after_heldout | −0.37% [−0.47, −0.26]* | −0.19% [−0.27, −0.10]* | −0.18% [−0.29, −0.07]* |
| after_rare_seen | −2.50% [−3.97, −1.03]* | −0.90% [−2.27, +0.38] | −1.61% [−2.49, −0.68]* |

- k (C5 vs C0) on `all` is 1.024 [1.020, 1.028]. C5 trains 2.1% slower.
- **The operators tie, so the pre-registered reading holds.** Against C2 after held-out terms: C5 −0.18%*, C5rf −0.25%*,
  C5tr −0.13%*, C5ut −0.21%*. On `all`, C5rf is numerically best (−0.20%* vs C2). C5 beats neither C5rf nor C5ut, so
  the wording is "compositional parameter sharing".
- The mechanical E4 gate items (not a gate at this size) pass on item 1 (held-out vs C2) and item 3 (locality). Item 2
  needs C1, which this screen did not run.

## Limitations
- 100M tokens is a short budget. The pre-registered gate sits at 125M × 500M tokens.
- The report does not print the pre-registered paired C5 − arm intervals. The CARD-660 secondary is in no report,
  although per-run `probes.json` files exist.
- `analysis/opscreen-v1/` and `runs/opscreen/` are untracked in git as of 20 Oct.

## Next steps
- **Done:** the screen led to the scaling screen (decision 65; 03.2).
- **Open:** print the C5 − arm intervals and the CARD-660 secondary (asked for in update-2026-10-12, E4-2), and commit
  the report.

---

# 03.2 — Scaling screen, phase 1a: 20M, 50M and 125M parameters × 500M tokens (seed 1)

**Status:** done · **When:** pre-registered 2026-10-10; runs 2026-10-11 to 10-13; report 2026-10-13 · **Sources:**
`experiments/e4-small-lm/preregistration-scale-v1.md`; `experiments/e4-small-lm/analysis/scale-v1/{report.md,
e4-report.md}`; `manuscript/update-2026-10-15.md` §7; `resources/plan-improvement/execution.md` (decision 65)

## Research gap
03.1 measured one size at 100M tokens, and its projected gaps at 2.5B and 7B tokens had seed intervals spanning 0. In
pretrained hosts (E9), WordNet gave no gain at all. Two questions were open. Does the from-scratch gain grow, hold or
fade with model size? Do HRRBERT's hybrids, which beat a plain composed embedding in BERT, do better here?

## Intuition
The channel adds a fixed 2.2M parameters. That is 20.7% of the 20M host's N but only 2.7% of the 125M host's. A larger
model can learn more rare terms on its own, so we expected k to shrink with size.

## Methodology
- **Hosts** (GPT-2 architecture): 6 × 384 ("20M", N 10.6M), 8 × 512 (50M, N 25.2M), 12 × 768 (125M, N 85.1M).
- **Recipe:** the frozen 50M recipe at every size.
  - 32,768 tokens per step and 500M tokens (15,258 steps), cosine schedule, 10M warmup tokens.
  - Peak learning rate ∝ 1/width: 2.667e-3, 2e-3, 1.333e-3.
  - Seed 1. Batches are paired across conditions within a size.
- **Arms:** C0, C5, C2, HRRAdd, HRRCat at every size; C5sh at 50M (16 runs).
- **Primary endpoint** (on `all` and `after_len3plus`): the slope b of ln k (C5 vs C0) on ln N, with the seed-inflated
  95% interval. An upper bound below 0 means k shrinks; a lower bound above 0 means k grows.
- **Escalation rule (iii) at 125M:** k at 125M ≥ k at 50M (point estimates).
- **Phase-2 rule:** request 350M, more 125M seeds or a natural-domain corpus only if, on a primary stratum, C5 beats C0
  at 125M (Holm) **and** rule (iii) holds **and** shrinkage is not established.
- **Secondaries:** hybrids (S1), C2's share of C5's gain (S2), C5 − C5sh at 50M (S3), unlinked text (S4).

## Experiment
16 runs at seed 1, planned at 38–43 GPU-h. C5 runs 2.9%, 2.3% and 1.5% slower than C0 at 20M, 50M and 125M.

## Results
**C5 − C0 by size** (seed 1):

| | 20M | 50M | 125M |
|---|---:|---:|---:|
| all | −0.24% [−0.28, −0.20]* | −0.38% [−0.43, −0.34]* | −0.29% [−0.33, −0.24]* |
| after_len3plus | −1.20% [−1.35, −1.05]* | −1.40% [−1.56, −1.25]* | −1.13% [−1.29, −0.97]* |
| after_heldout | +0.02% [−0.16, +0.21] | −0.05% [−0.25, +0.14] | +0.25% [+0.05, +0.45] |
| unlinked | +0.09% [+0.04, +0.14]* | −0.11% [−0.16, −0.06]* | −0.04% [−0.09, +0.01] |
| k, all | 1.017 | 1.045 | 1.031 |
| k, after_len3plus | 1.142 | 1.147 | 1.122 |

**Primary endpoint and phase-2 verdict:**

| Stratum | Slope b ⟨seed-inflated CI⟩ | Pre-registered reading | Rule (iii) at 125M | Phase 2 |
|---|---|---|---|---|
| all | +0.0053 ⟨+0.0019, +0.0087⟩ | k grows | fails (1.031 < 1.045) | not requested |
| after_len3plus | −0.0092 ⟨−0.0178, −0.0006⟩ | k shrinks | fails (1.122 < 1.147) | not requested |

**Secondaries:**
- **Hybrids (S1)** beat C5 on `all`, and their edge shrinks with size. HRRAdd − C5: −0.13%*, −0.11%*, −0.06%*. HRRCat −
  C5: −0.19%*, −0.06%*, −0.03% [−0.07, +0.01].
- **C2's share of C5's gain (S2):** on `all` 1.25, 0.72, 0.68; on `after_len3plus` 1.06, 0.82, 0.72. At 20M the free
  table matches or beats C5.
- **Specificity at 50M (S3):** C5 − C5sh is −0.16%* on `all`, −0.50%* on `after_len3plus`, −0.36%* on
  `after_heldout`, and −0.81% [−2.33, +0.70] on `after_rare_seen`. Correct relations matter from scratch.
- **Cost-adjusted k on `all`:** 0.988, 1.021, 1.015 by wall clock; 0.948, 1.001, 1.012 by FLOPs (an upper bound on
  cost). At 20M the method does not pay for itself on all text.

**Pre-registered predictions** (§7, written before any run):

| Prediction | Observed | Verdict |
|---|---|---|
| k on `all` shrinks with size (≈ 1.04 / 1.025 / 1.01; slope ≈ −0.014) | 1.017 / 1.045 / 1.031; slope +0.0053 | failed |
| `after_len3plus` shrinks more slowly (≈ 1.10 / 1.07 / 1.05), maybe not established | 1.142 / 1.147 / 1.122; shrinkage established | partly |
| Rule (iii) at 125M false on `all`, uncertain on `after_len3plus` | false on both | passed |
| Phase 2 not requested on `all`; `after_len3plus` decides | not requested on either | passed |
| 50M gaps ≈ −0.3% (`all`), ≈ −1.0% (`after_len3plus`); gaps narrow from 20M to 125M | −0.38%, −1.40%; largest at 50M | partly |
| S1: hybrids within ±0.1% of C5, not better at ≥ 2 sizes; both beat C0 at every size | both beat C5 (Holm) at ≥ 2 sizes; both beat C0 everywhere | failed (small hybrid advantage replicated) |
| S2: C2's share ≈ 0.6–0.7 (`all`), ≈ 0.6 (`after_len3plus`), constant over size | not constant; ≥ 1 at 20M | partly (50M and 125M on `all`) |
| S3: C5 beats C5sh at 50M after long terms (≈ −0.5%) and after rare terms | −0.50%*; rare n.s. | passed; rare not established |
| S4: unlinked ≈ −0.1% at 50M, near 0 at 125M | −0.11%*; −0.04% (n.s.) | passed |

## Limitations
- One seed per cell. The `⟨ ⟩` intervals borrow seed variance from 50M × 100M tokens.
- The recipe was not tuned per size (no learning-rate check at 20M or 125M). The low k at 20M drives the positive slope
  on `all`. At the end-of-run loss target (k_end) the `all` slope is −0.0013 ⟨−0.0100, +0.0075⟩, so the "k grows"
  reading depends on the target.
- The sources word this differently. update-2026-10-15 says the benefit "does not grow with size". That matches rule
  (iii) (k at 125M < k at 50M), not the pre-registered slope reading on `all`, which says "k grows". Both rest on the
  same numbers; k peaks at 50M.
- k is read at intermediate checkpoints of one cosine schedule. 500M tokens is below compute-optimal for 125M (about
  2.5B).
- No C1 control and no probes in this cohort.
- `analysis/scale-v1/` and `runs/scale-v1/` are untracked in git as of 20 Oct.

## Next steps
- **Closed by the pre-registered rule:** phase 2 (350M, a natural-domain corpus). The author can override.
- **Done:** 125M seeds 2–3 (03.3).
- **Queued (E9):** the hybrid design in pretrained hosts (`H5add`, addendum 2 §4).
- **Open:** a pooled-seed mode for the `e4_scale` report (findings 2026-10-18 §3).

---

# 03.3 — Scaling screen: 125M seeds 2–3

**Status:** done (preliminary: not yet in the claims ledger) · **When:** pre-registered 2026-10-15; runs 2026-10-18;
report 2026-10-18, re-run by the queue 2026-10-20 · **Sources:**
`experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §3;
`manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §3;
`experiments/e4-small-lm/analysis/scale-v1-seeds/{report.md, e4-report.md}`

## Research gap
Phase 1a's 125M cell had one seed. The 125M gap is the number a paper would quote, so it needed a seed spread.

## Intuition
In 03.1 the seed SD of C5 − C0 on `all` was about 0.05 percentage points. Two more seeds should pin the 125M gap to
within about ±0.1 points.

## Methodology
- C0 and C5 at 125M × 500M tokens, seeds 2 and 3, with phase 1a's recipe and windows.
- **Primary:** C5 − C0 at 125M on `all` and `after_len3plus`, pooled over seeds 1–3, with the seed spread.
- **Prediction:** about −0.3% (`all`) and −1.1% (`after_len3plus`), both significant.
- **Reading:** the new seeds do not reopen the phase-2 rule, which used seed 1 as pre-registered.

## Experiment
Four runs (planned at about 16 GPU-h). We generated the `e4_scale` report over all 20 runs by hand on 18 Oct. The
official queue job re-ran it on 20 Oct and produced an identical `report.md`.

## Results

| Stratum | s1 | s2 | s3 | Mean (SD) | Pooled [window CI] | Prediction | Verdict |
|---|---:|---:|---:|---:|---|---:|---|
| all | −0.29% | −0.34% | −0.36% | −0.33% (0.04) | −0.33% [−0.36, −0.30]* | ≈ −0.3% | passed |
| after_len3plus | −1.13% | −1.24% | −1.27% | −1.21% (0.07) | −1.21% [−1.33, −1.10]* | ≈ −1.1% | passed |

Other 125M results (C0 and C5 pooled over 3 seeds; C2 seed 1):
- C5 − C0: `after` −0.78% [−0.84, −0.73]*; `unlinked` −0.09% [−0.12, −0.06]*; `after_heldout` +0.18% [+0.04, +0.32]
  (not significant after Holm); `after_rare_seen` −0.09% [−1.38, +1.06].
- k: `all` 1.031 [1.029, 1.034]; `after_len3plus` 1.125 [1.114, 1.135].
- C5 − C2: `all` −0.09% [−0.13, −0.05]*; `after_len3plus` −0.32% [−0.47, −0.17]*; `after_heldout` +0.22% [+0.02,
  +0.41] (not significant). C2's share of C5's gain on `all` falls to 0.59 [0.47, 0.71].
- **Size trend:** `all` +0.0055 ⟨+0.0024, +0.0086⟩; `after_len3plus` −0.0079 ⟨−0.0153, −0.0005⟩. Phase 2 is still not
  requested. Rule (iii) now holds on `after_len3plus` through the projected-gap interval, but shrinkage there is
  established.
- **Mechanical E4 gate at 125M** (not the formal gate: C2 has one seed, and C1 and probes are absent):
  - item 1 (held-out vs C2) fails: C5 − C2 = +0.0079 [+0.0006, +0.0148] in absolute loss, C5 worse;
  - item 3 (locality) passes: unlinked −0.09%, upper bound −0.06%, margin 0.5%;
  - items 2 and 4 are not available.

## Limitations
- The pooled interval covers evaluation windows only. The seed SDs (0.04 and 0.07 points) come from 3 seeds.
- C2, HRRAdd and HRRCat have one seed at 125M, so every comparison against them is single-seed.
- The report still prints the "single seed per size" banner. Its size trend mixes a 3-seed 125M cell with 1-seed 20M
  and 50M cells. The 18 Oct findings note says `e4_scale` "still analyses one seed per size", yet the report pools the
  three 125M seeds in its gaps and k. It averages the seeds within each window, and its interval resamples windows only.
- The newer report supersedes seed-1 values. C2's share of the 125M gain on `all` was 0.68 at seed 1 ("about 70%" in
  update-2026-10-15) and is 0.59 with three C5 seeds. After held-out terms, C5 − C0 was +0.25% at seed 1 and is +0.18%
  with three seeds. We use the newer values.

## Next steps
- **Done:** the 125M gap is firm.
- **Open:**
  - a pooled-seed mode for `e4_scale`;
  - entering the result in `manuscript/claims.md`, where H-A.1 still reads "pending (S0 pilot running)".
- **Closed:** phase 2, by the pre-registered rule.

---

# 03.4 — Toolkit arms at 50M: a growing dictionary (C5dev) and a Claude-read ontology (C5teach)

**Status:** partly done: C5dev done (seed 1); C5teach not run · **When:** pre-registered 2026-10-10; C5dev ran and was
reported on 2026-10-13 · **Sources:** `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md`;
`experiments/e4-small-lm/analysis/scale-v1-toolkit/report.md`;
`experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5dev-s1/cards.json`;
`experiments/e4-small-lm/scale-v1-toolkit/teacher-pilot/`; `resources/plan-improvement/execution.md` (decision 65)

## Research gap
Two toolkit ideas had not been tried in from-scratch training:
- **Passive learning:** can the dictionary grow new atomic senses from language-model gradients? E0 recovered planted
  senses; E3 failed on WordNet with frozen GPT-2 anchors.
- **Learning from reading:** can frames that Claude writes from glosses replace curated WordNet frames, above all for
  new words?

## Intuition
Gradients may show when one atomic serves two senses, and splitting it should help the terms that use it. Teacher
frames that agree with WordNet should carry the same gain. For held-out words, that would show reading substituting for
curation.

## Methodology
- Each arm is phase 1a's 50M C5 with one change, at 50M × 500M tokens, seed 1. Both are read against phase 1a's 50M C5
  and C0.
- **C5dev:** developmental dictionary on atomics.
  - A screen every 250 steps and a permutation test (999 permutations, p < 0.002).
  - Growth budget of at most 409 splits. The parent keeps unobserved usages, including held-out entries.
  - Consolidation merges sibling pairs at cosine ≥ 0.98.
- **C5teach (hybrid):** Claude (`claude-opus-5-5`) writes frames for the 5,900 held-out entries and 5,900 matched trained
  entries. It reads each sense's lemma, part of speech and gloss. All other entries keep WordNet frames.
- **Statistic:** final relative difference, window bootstrap, Holm over four comparisons. Differences below 0.2% (1.5% on
  `after_rare_seen`) are not interpreted.
- **Decision rules:**
  - C5dev advances only if C5dev − C5 ≤ −0.2% with the interval below 0 on `after` or `after_len3plus`, `after_heldout`
    is non-inferior (upper bound ≤ +0.3%), and at least 10 splits were accepted.
  - C5teach advances if it beats C5 by at least 0.2% on `after_heldout` or the teacher-trained stratum, or if it is
    non-inferior to C5 and beats C0 on `after_heldout`.

## Experiment
- **C5dev:** one run on 13 Oct (about 3.9 h wall clock).
- **C5teach:** not run. Building the teacher ontology needs the author's approval of the spend: about USD 73 for the
  hybrid and USD 342 for the full variant. Its jobs are commented out of the queue script and were never queued.
- **Teacher pilot** (before the pre-registration): 400 entries, 829 senses, USD 2.48 of a USD 5 cap.

## Results
**C5dev** (50M × 500M tokens, seed 1):

| Stratum | C5 − C0 | C5dev − C0 | C5dev − C5 (derived, no CI) |
|---|---:|---:|---:|
| all | −0.38% [−0.43, −0.34]* | −0.36% [−0.41, −0.32]* | +0.02% |
| unlinked | −0.11% [−0.16, −0.06]* | −0.06% [−0.11, −0.01]* | +0.06% |
| after | −0.90% [−0.98, −0.82]* | −0.94% [−1.02, −0.86]* | −0.04% |
| after_len3plus | −1.40% [−1.56, −1.25]* | −1.45% [−1.61, −1.29]* | −0.04% |
| after_heldout | −0.05% [−0.25, +0.14] | −0.18% [−0.38, +0.02] | −0.13% |
| after_rare_seen | −3.01% [−4.50, −1.39]* | −2.59% [−4.23, −0.84]* | +0.43% |

The last column is our own arithmetic on the report's final losses. The report gives no paired C5dev − C5 interval.

- **Growth:** `cards.json` lists 13 accepted splits and 1 merge. Nine of the 13 splits came by step 1,750 of 15,258.
- **k vs C0** (C5 in brackets): `all` 1.044 (1.045); `after` 1.106 (1.106); `after_heldout` 1.031 (1.024);
  `after_rare_seen` 1.260 (1.306).
- **Cost:** 68,372 tokens/s against C5's 68,019, so no measurable cost. The CPU upper bound had been 4.8%.

| Prediction (C5dev) | Observed | Verdict |
|---|---|---|
| \|C5dev − C5\| < 0.2% on every stratum | ≤ 0.13% on five strata; +0.43% on `after_rare_seen`, within its 1.5% bar | passed (point estimates) |
| `after_heldout` non-inferior (upper bound ≤ +0.3%) | no paired interval computed | not evaluable |
| fewer than 100 splits, most in the first third, few merges | 13 splits, 9 by step 1,750; 1 merge | passed |
| k(C5dev) = k(C5) ± 5% | within 3.5% on all four strata | passed |
| keeps ≥ 75% of C5's gain over C0 after terms; no worse than C0 on unlinked text | yes; unlinked −0.06%* | passed |

**Decision:** C5dev does not advance. Its difference from C5 after terms is −0.04%, far from the −0.2% bar. With 13
splits (≥ 10), growth was tested, and the rule does not follow up a null at seed 1. No source records this verdict
formally; update-2026-10-15 calls C5dev "about the same as the method".

**Teacher pilot** (a measurement of the teacher, not a model run):
- 97.3% of 2,710 proposed edges map onto the ontology.
- Frames hold 5.06 edges per entry against WordNet's 5.20.
- Agreement with WordNet (precision / recall): 0.78 / 0.75 on all edges, 0.59 / 0.56 on pointer edges, 0.56 / 0.52 on
  held-out entries.
- Meronyms are under-produced (recall 0.28–0.33), and entailment is never produced.

## Limitations
- One seed and one size, so the intervals reflect evaluation noise only.
- The report omits the pre-registered C5dev − C5 comparisons and their Holm correction.
- C5dev used 13 of 409 budgeted splits, so the screen settings may have limited growth. The rule flags this case only
  below 10 splits.
- C5teach is untested. The pilot measures agreement with WordNet, not usefulness to the model.

## Next steps
- **Closed:** C5dev, by the pre-registered rule.
- **Awaiting the author:** the C5teach spend (about USD 73 for the hybrid).
- **Open:** add the C5dev − C5 table to the toolkit report.

---

# 03.5 — The original E4 core: planned, not run

**Status:** not run (replaced by the reduced screens) · **When:** planned in September; set aside by decision 54 on
2026-10-07 · **Sources:** `resources/plan-improvement/experiments.md` (E4); `resources/plan-improvement/execution.md`
(decisions 54, 65); `manuscript/draft.md` §5.5; `manuscript/claims.md` (H-A.1, H-B.1, H-B.2);
`manuscript/paper-readiness.md`

## Research gap
The pre-registered E4 gate is defined at 125M × 500M tokens with 3 seeds. It needs controls that the screens lacked.

## Intuition
Only the full grid can separate four explanations of the gain: composition (C5), span-boundary information (C1), any
extra lookup memory (C1h), and simpler composition (C3t, C3, C4). It also tests growth (C6) and downstream probes.

## Methodology
As planned:
- **Conditions** with matched non-embedding parameters:
  - baselines and controls C0, C1, C1h and C2;
  - composition variants C3t, C3, C4, C5, C6 and C7;
  - an operator sweep, C8.
- **Gate** at 125M × 500M tokens, 3 seeds:
  1. C5 or C6 beats C2 on held-out terms (interval excluding 0) and matches C2 on seen rare terms;
  2. it beats C1 on every linked stratum;
  3. unlinked loss stays within 0.5% of C0;
  4. at least two probes improve (WiC, WSD, CARD-660, Rare Words, BLESS, HyperLex, LAMBADA).
- **Stop rules:**
  - failing item 2 makes the channel a "boundary detector";
  - failing item 1 while passing item 2 leaves a seen-concept efficiency result with no zero-shot claim.
- **Extensions:** ℓ_min feasibility (E4.7; ℓ_min is the minimum number of subtokens a linked term must have), INT8/INT4
  quantization (E4.4), table compression (E4.5) and continued pretraining (E4.6, now done as E9).

## Experiment
The 125M three-seed grid never ran:
- Decision 54 (author, 7 Oct) replaced the E4 core with the reduced operator screen and kept "the full E4 core
  unqueued". The sources give no further reason. The readiness audit of 3 Oct priced the core at about 143 GPU-h.
- Decision 65 then ran the scaling screen (03.2) instead.
- C1 ran only in the under-trained S0 pilot.
- Table-compression configs exist (`configs/compress/`) but have no runs.

## Results
None at gate scale. The nearest evidence is 03.3: C0 and C5 at 125M × 500M with 3 seeds, but C2 has one seed and C1 is
absent. On that evidence item 1 fails mechanically and item 3 passes (03.3).

## Limitations
- Without C1 we cannot rule out that part of the gain is span-boundary information. The pilot's C1 was worse than C0
  (+0.26%* after a term), but it was trained with the under-trained recipe.
- `manuscript/draft.md` §5.5 still shows the D4.1 table as pending and "S0 pilot running". The claims ledger rows H-A.1,
  H-B.1 and H-B.2 are still "pending". The readiness audit (3 Oct) says E4 has no formal pre-registration file. That
  was true then; `preregistration-scale-v1.md` has existed since 10 Oct.

## Next steps
- **Closed by the author for now** (decision 54): the full grid, the probes, quantization and compression from scratch.
- **Open (housekeeping):** rewrite draft §5.5 around 03.1–03.3 and update the claims ledger.

---

# Sources

- Pre-registrations: `experiments/e4-small-lm/preregistration-scale-v1.md`,
  `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md`,
  `experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §3.
- Reports: `experiments/e4-small-lm/analysis/{opscreen-v1, scale-v1, scale-v1-seeds, scale-v1-toolkit}/report.md`
  (and `e4-report.md` in `scale-v1` and `scale-v1-seeds`); `reports/R1-sanity-pilot.md`.
- Run outputs: `experiments/e4-small-lm/runs/scale-v1-toolkit/50M-C5dev-s1/cards.json`;
  `experiments/e4-small-lm/scale-v1-toolkit/teacher-pilot/`.
- Plain-language summaries: `manuscript/update-2026-10-12.md` (E4-1, E4-2), `manuscript/update-2026-10-15.md` §7,
  `manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §3.
- Plan and decisions: `resources/plan-improvement/experiments.md` (E4), `resources/plan-improvement/gates.md` (D4.0),
  `resources/plan-improvement/execution.md` (decisions 54, 65).
- Ledger and draft: `manuscript/claims.md`, `manuscript/draft.md` §5.5, `manuscript/paper-readiness.md`.
- Dates: run manifests, `.jobs/` entries (read only), `git log -- experiments/e4-small-lm`.
