# E14 pre-registration — addendum 12 (2026-10-21, before any host run of it): RD-4e, routing that generalises

Companion to `experiments/e14-logic/preregistration.md` (conventions in its header; RD-4b §13, RD-4c §16, RD-4d §19).
Written and committed before any GPU job of this block was queued. The author's instruction of 2026-10-21, *"do these
steps"*, on the routing block ("register the 'no routing for frames ≤ 4 lines, few-shot router otherwise' rule and a
paraphrase-augmented few-shot router for T5; one small block"), is the go-ahead.

**Status:** pre-registration. Any change goes in the last section with its date and reason. Labelled smokes may run before
a queue entry; they change nothing here.

**Code:** `src/vsa_embed/experiments/e14_reading.py` (`--family rd4e`; routers `fewshot-para`, `fewshot-abstain`,
`fewshot-top2` in `RouteCache`; the condition `explicit-size:L:<router>:<reader>` and its derived twin
`derived-size:L:…`; `--max-length`; `size-rule-offline`; `recall-report --family rd4e`); tests `tests/test_e14_rd4e.py`;
queue `experiments/e14-logic/queue-commands-addendum12-routing.sh`; the offline analysis in
`experiments/e14-logic/report/rd4e/offline-size-rule-{t4,t5}.md`.

**Conventions:** as the main file: paired bootstrap, 4,000 resamples, 95% intervals, items paired, the item differences of
every seed pooled (the R9 convention), Holm within the primary family of each track. The null prompt is read without the
context (`--null plain`, §13).

---

## 1. Origin, and a correction found while preparing this block

RD-4d (§19) left routing solved on T4 (few-shot router: asked relation chosen 0.935–0.945 of the time; routed gold
recall 0.845 against labelled 0.862) and unsolved on T5 (few-shot 0.358, `lex` 0.392, every learned router below word
overlap; routed gold recall 0.400 against labelled 0.845). The memorising router reaches the ceiling, so the T5 gap is
generalisation across question wordings. RD-4c (§16) found whole-frame recall nearly harmless on T5's short frames for
the Qwen3 hosts but costly for SmolLM2-360M (0.687 against 0.845), and RD-4b (§13) found wrong lines cost more than no
lines (0.528 against 0.614 on T4).

**Correction (found 2026-10-21 while assembling §2; stated before any RD-4e run).** The scoring adapter cuts every
prompt at `max_length` = 256 tokens (`evaluation/probes.py`, `_run_batch`, `truncation=True`), and the recall runs
never raised it. A prompt longer than that loses its candidate continuation, so every candidate scores 0 and the raw
choice is candidate 0. The whole-frame contexts were long enough for this to matter:

- T4 `explicit-sym:oracle`: 42.3% of the property / entailment prompts exceed 256 tokens (0% for frames of ≤ 6
  lines, 8.8% at 7–8 lines, 44% at 9–12, 98% at 13+). On the frames of 15+ lines the raw accuracy of
  `explicit-sym:oracle` in RD-4c equals the share of items whose gold is candidate 0 exactly (0.226 = 0.226): the
  signature of the cut.
- T5 `explicit-sym:oracle`: 17.5% (0% up to 8 lines, 47% at 9–12 lines).
- Selective recall (`explicit-rel`, `explicit-route`), the definition text and the routed conditions stay under the
  limit (longest T5 definition prompt 232 tokens), so RD-4b/c/d's main results are unaffected; **the whole-frame numbers
  of tier P, RD-4b, RD-4c (including the Qwen3 hosts' T4 0.605) and the "whole frame hurts on T4" reading are partly
  an artefact and must be re-measured.** RD-4e does so (`--max-length 1280`, §4), and every condition now records
  `truncated_share` in its diagnostics.

The offline analysis below is therefore read only where whole-frame prompts fit: L ≤ 6 on T4 and L ≤ 8 on T5 are clean;
larger L and the whole-frame rows carry the artefact (marked).

## 2. The size rule, offline first (derived analysis; no new scoring)

**Rule.** "Recall the whole frame when the reader's frame has ≤ L lines, route otherwise", L ∈ {2, 3, 4, 6} (the
author's grid; 8, 10, 12 and ∞ shown for the shape). **Combination:** per item, the `explicit-sym:<reader>` row of
RD-4c (SmolLM2-360M C5, seeds 1–3, plain null) when the reader's frame for the item's concept has ≤ L lines, else the
`explicit-route:<router>:<reader>` row of RD-4d (same host, seeds, items, null); differences against RD-4d's routed and
labelled (`explicit-rel:<reader>`) rows, paired by item, seeds pooled. Frame sizes from the runs' own `frames.jsonl`.
Oracle frames only: no plain-null `explicit-sym:typeprior` exists yet (tier P's used the context null), so the type-prior
version is measured in the run (§4). Command: `e14_reading size-rule-offline` (files above).

**T5, few-shot router (property; n = 4,587):** routed 0.400; L = 3: 0.403 (+0.003); L = 4: 0.416 (+0.016
[+0.012, +0.020]); **L = 6: 0.500 (+0.100 [+0.090, +0.110])**; L = 8: 0.608 (+0.208); whole frame 0.687 (artefact
above 8 lines). Paraphrase: +0.065 at L = 6; entailment +0.048. Against labelled recall (0.845) every L in the grid stays
far below: −0.345 at L = 6. With `lex`: 0.618 → 0.680 at L = 6 (+0.062), 0.738 at L = 8.

**T4, few-shot router (property; n = 1,221):** routed 0.845; L = 3: −0.000; L = 4: −0.001; **L = 6: −0.001
[−0.003, +0.000]**; L = 8: −0.007; whole frame 0.578 (artefact). The rule changes nothing on T4 where routing already
works; with `lex` it adds +0.006 at L = 6.

**L\* = 6**, fixed now: the largest L of the author's grid, which is also the L with the largest T5 gain within the grid
and the largest L whose whole-frame prompts never exceed 256 tokens on either track (so the offline number is clean).
The shape says more than the grid: on T5 the hybrid rises monotonically up to whole-frame recall, i.e. for SmolLM2-360M
on T5 the frame size at which routing beats whole-frame recall is above 8 lines — if it exists at all once truncation
is removed. The run therefore also assembles L ∈ {2, 3, 4, 6, 8} from its own untruncated rows (`derived-size:…`).

## 3. The routers

All few-shot routers share RD-4d's prompt ("Each question asks for one kind of fact." followed by `Question: <wording
with X> ...` / `Fact: X <relation phrase>` demonstrations, then the item's question with the term and `Fact: <term>`;
the option continues as ` <phrase>`, scored by length-normalised log-probability). Candidates are the relations the
reader's frame holds. Each item is routed by **one** of its templates, `question_template` (crc32 of the item id, as RD-4d);
that wording is the held-out wording q.

- **`fewshot`** — RD-4d's router unchanged: up to two item templates per candidate relation, excluding q exactly. It is
  the baseline of (b) and (c). Note, seen while writing the filter below: on T4 `has_role` its demonstrations include
  "{x} has a role as" when q is "{x} has a role as a" (a near-duplicate); RD-4d's T4 result had that help.
- **`fewshot-para`** — the demonstrations of each candidate relation come from `demonstration_bank`: the relation's item
  templates plus **hand-written paraphrases** (`ROUTE_PARAPHRASES`; 5 per relation for the 3 T4 and 12 T5 relations,
  written by hand on 2026-10-21, no model call; a relation without an entry gets five template-grammar carriers of its
  statement phrase), minus q and every near-duplicate of q, without repeated content. Up to `PARA_SHOTS` = 5 wordings per
  relation are shown, rotated by a stable per-item offset so every wording of the bank is used across items (on T5: the
  one or two other templates and 5 paraphrases → 6–7 wordings, 5 shown; on T4 has_role: 2 templates + 5 paraphrases).
  The prompt holds up to 12 relations × 5 = 60 demonstrations (≈ 700 tokens on the largest T5 frames), so the
  demonstrations are run once per item and cached (`e11.PrefixCache`; `--route-scoring prefix`, the same quantity as
  RD-4d's path, tested close on the CPU world); the plain `fewshot` uses the same path within the run.
- **The hold-out guarantee.** Two wordings are *near-duplicates* when their content-word sets (words minus `{x}` and a
  fixed stop list) are equal, or one is inside the other with one word of difference. This covers "{x} has a role as" /
  "… as a" / "… as an" (equal content), "{x} is part of" / "{x} was delivered as part of" / "{x} forms part of" (one word
  inside), while keeping real rewordings ("{x} is owned by" / "{x} is run by"). `demonstration_bank` drops q and its
  near-duplicates and asserts that none survives; `tests/test_e14_rd4e.py` checks on the committed T4 and T5 item files
  that for every relation and every held-out template q: q is absent from the demonstrations, no demonstration is a
  near-duplicate of q, no hand paraphrase is a near-duplicate of any item template (so the paraphrases are always
  available), the demonstrations have pairwise distinct content, and the per-item rotation never shows q. The routed
  wording's own template is never used as a calibration query either (next item).
- **`fewshot-abstain`** — `fewshot`'s choice unless its **margin** (best minus second-best normalised score, nats per
  token) is below τ; then nothing is recalled (the recall call with "(nothing recalled)", the same form as an item without
  lines in `explicit-rel`). **τ is fixed by rule, computed inside each job on its own host from the T4 training
  templates:** every T4 held-out item whose relation is in its oracle frame contributes its *training* templates t ≠ q
  as calibration queries, routed by the plain few-shot prompt whose demonstrations exclude both q (held out as ever) and
  t itself (leave-one-out, so a query never sees its own wording); **τ = the 20th percentile of the margins of the
  correctly routed calibration queries** (`ABSTAIN_QUANTILE`). T5 jobs load the T4 items and frames for this
  (`--calibration-items`, `--calibration-frames`): T4 wording, the job's own host and scale. T5's own training templates
  cannot calibrate (one training wording per relation, so leave-one-out leaves none). RD-4d saved no margins, so τ could not
  be a number written here; the rule is. τ, the calibration accuracy and the abstention share are recorded per run, and
  `routing.jsonl` keeps every item's scores and margins for later analysis.
- **`fewshot-top2`** — `fewshot`'s two best relations.
- **`lex`** — RD-4c's word-overlap router (the best T5 router so far).

## 4. Design

**Items, hosts, scoring:** RD-4d's: `t4-heldout-smollm2-v1` (341 terms, 814 items) and `t5-new-smollm2-v1` (300 words,
3,658 items); SmolLM2-360M C5, seeds 1–3; E11's property / paraphrase / entailment tests, plain null; `--max-length 1280`
(the context budget stays 1,024 tokens, so no prompt is cut; the longest T4 whole-frame prompt is 604 tokens).
Qwen3-1.7B C0′ on T5 is **not** queued: RD-4c's Qwen3-1.7B T5 job took 58 min for 12 conditions, so one seed of this
family would add ≥ 1.5 GPU-h on its own, more than the block's budget.

**Conditions** (`--family rd4e`, 18 scored):
- `none`, `explicit-rel:{oracle,typeprior}` (labelled), `explicit-sym:{oracle,typeprior}` (whole frame, now untruncated),
  `definition`;
- `explicit-route:<router>:{oracle,typeprior}` for `lex`, `fewshot`, `fewshot-para`, `fewshot-abstain`, `fewshot-top2`;
- `explicit-size:6:fewshot:{oracle,typeprior}`: the size rule at L\*, scored as a real condition.
- **Derived in the run, no scoring** (`derived-size:L:<router>:<reader>` for L ∈ {2, 3, 4, 6, 8}, routers `fewshot`,
  `fewshot-para`, `lex`, both readers): §2's combination rule applied to the run's own `explicit-sym` and
  `explicit-route` rows. `derived-size:6:fewshot:oracle` must reproduce `explicit-size:6:fewshot:oracle` item for item
  (same prompts, same null); the report prints the agreement.

**Primary** (per track; Holm over 7 contrasts × 3 tests):
- **(a)** `P − explicit-rel:typeprior` for each realistic pipeline P with type-prior frames in the fixed set
  {`explicit-route:fewshot-para:typeprior`, `explicit-route:fewshot-abstain:typeprior`, `explicit-route:fewshot-top2:typeprior`,
  `explicit-size:6:fewshot:typeprior`, `explicit-route:lex:typeprior`}. "The best realistic pipeline" is the member with the
  highest pooled property mean; its contrast is read after Holm over the whole family, which pays for the selection.
- **(b)** `explicit-route:fewshot-para:oracle − explicit-route:fewshot:oracle` (the question of this block, on T5).
- **(c)** `explicit-size:6:fewshot:oracle − explicit-route:fewshot:oracle`.

**Secondary (unadjusted):** `fewshot-abstain` and `fewshot-top2` against `fewshot` (oracle frames); `fewshot-para` against
labelled and against whole-frame recall; RD-4d's realistic pipeline `fewshot:typeprior` against labelled type-prior
recall (the reference for (a)); `fewshot-para:typeprior` against the definition and against `none`; the re-measured
whole-frame contrasts `explicit-sym:oracle − explicit-rel:oracle` and `explicit-sym:typeprior − explicit-rel:typeprior`
and `explicit-sym:oracle − fewshot:oracle`; the derived-against-real check; `derived-size:{4,8}:fewshot:oracle` against
routing; `derived-size:6:fewshot-para:oracle` against `fewshot-para`; `lex − fewshot-para` (oracle). Descriptive: router
accuracy, abstention share and τ, calibration accuracy, `truncated_share` per condition, the full derived-size table.

## 5. Predictions (written before any run; the offline analysis of §2 and the CPU smokes were seen)

1. **T4.** Routing is already within 0.05 of labelled type-prior recall (RD-4d: 0.724 against 0.717), so (a) holds for
   `fewshot-para:typeprior` and the size rule, |difference| ≤ 0.03; **"routing solved" on T4.** (b) is about 0 on T4
   (|Δ| ≤ 0.02; the stricter near-duplicate filter may cost a little on `has_role`). (c) is within [−0.01, +0.005]
   (offline −0.001). Re-measured whole-frame recall `explicit-sym:oracle` rises from 0.578 to ≥ 0.75 on property and its
   cost against labelled recall shrinks to within 0.10 (RD-4c's −0.284 was mostly the cut).
2. **T5.** (b) is positive on property by ≥ +0.15 (router accuracy from 0.36 to ≥ 0.55), but `fewshot-para:oracle`
   stays ≥ 0.10 below labelled recall (0.845), so **"routing not solved" on T5 for SmolLM2-360M.** (c) is +0.10 ± 0.02
   on property (offline +0.100 [+0.090, +0.110]) and the derived row agrees with the scored one on ≥ 0.99 of items.
   Whole-frame recall, re-measured, is ≥ 0.70 and remains above every routed pipeline unless `fewshot-para` reaches
   router accuracy ≥ 0.75. `fewshot-top2 − fewshot` is positive (+0.10 to +0.20: two chances, still ≤ 2 lines).
   `fewshot-abstain`: in the CPU smoke the T5 host routed the T4 calibration queries at only 0.665 (the T4 host: 0.93 in
   RD-4d), and the 20th percentile of its correct margins (τ ≈ 2.2 nats per token) abstained on two thirds of the 3
   words' items while raising the accuracy of the routes it kept (0.70 against 0.47). Prediction: τ between 1.5 and 3 on
   the T5 hosts (the T4 host's smoke gave τ 1.81 at calibration accuracy 0.683 and abstained on 11 of 12 items — the
   leave-one-out queries have one demonstration less than the router, so τ may be generous), abstention share 0.4–0.8
   on both tracks; `fewshot-abstain − fewshot` within [−0.05, +0.05] on T5 property (fewer wrong lines, but also fewer
   right ones) and negative on T4, where the dropped routes were mostly right. `lex` stays near 0.62.
3. Across both tracks, `truncated_share` is 0 for every condition at `--max-length 1280`.

**Alternative stated in advance:** if (b) is not positive on T5, more wordings in the demonstrations do not help this
host route; the remaining levers are whole-frame recall on short frames (the derived table says how short) or a larger
host (RD-4c: the Qwen3 hosts lose little to whole-frame recall on T5).

## 6. Decision rule

For each track: **"routing solved"** iff the best realistic routed pipeline (the member of the set in (a) with the
highest pooled property mean, type-prior frames) is within 0.05 of labelled type-prior recall on property
(difference ≥ −0.05, the interval reported). Expected: solved on T4, not on T5. If T5 is not solved and (b) is positive,
the manuscript states that paraphrase-augmented demonstrations close part of the wording gap and names the remaining
gap; if (b) is not positive, the alternative above. The derived size-rule table (untruncated) replaces §2's as the
reference for the frame size at which routing beats whole-frame recall on each track. The truncation correction is
reported with the re-measured whole-frame numbers wherever earlier whole-frame results are cited.

## 7. Smokes seen before this registration (labelled; not results)

- `tests/test_e14_rd4e.py` on the tiny E9 world (CPU): the hold-out guarantee on the committed T4 and T5 item files, the
  size-rule assembly, the abstention rule, prefix-cached against brute few-shot scoring (agree to 1e-3), and the whole
  `--family rd4e` path with its report.
- A CPU `--limit 3` smoke of the real T5 and T4 pipelines on SmolLM2-360M C5 s1 (outputs under the session scratchpad,
  labelled SMOKE), to check the T4 calibration set on the T5 host and the prefix path on the extended adapter: 814
  calibration queries, `truncated_share` 0 everywhere at `--max-length 1280`, `derived-size:6` identical to
  `explicit-size:6` on every item of the smoke, no prompt trimmed.

## 8. Queue and cost

`queue-commands-addendum12-routing.sh`: six GPU jobs `e14-rd4e-{t4-heldout,t5-new}-SmolLM2-360M-C5-s{1,2,3}` at priority
**54.49599975** (`--min-free-gb 10`), the CPU-lane report `e14-rd4e-report` at **54.49599988** (writes
`experiments/e14-logic/report/rd4e/`). Estimate from RD-4d's and RD-4c's measured durations (`.jobs/*.json`): a T4 RD-4d job
took 4.8 min for 14 conditions (0.3–0.5 min per condition), a T5 RD-4d job 18.3 min (plain conditions 0.9 min, whole
frame 2.4 min, few-shot routing + scoring 3.0 min). RD-4e scores 18 conditions, two of them whole-frame, routes four
prompts per item (two readers × plain / paraphrase, cached demonstrations) and runs the calibration once: about
8–10 min per T4 job and 28–35 min per T5 job, **≈ 1.9–2.2 GPU-h in all**, above the block's ~1.5 GPU-h target by the
type-prior copies of the secondary routers and the untruncated whole-frame conditions; the report is seconds. If the
budget must hold, cancel `e14-rd4e-t5-new-SmolLM2-360M-C5-s3` (the primaries pool items over seeds and remain paired).

## 9. Changes after this commit

(none)
