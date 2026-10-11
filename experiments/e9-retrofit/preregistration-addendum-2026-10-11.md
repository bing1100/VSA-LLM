# E9 — addendum of 2026-10-11: six follow-up experiments

**Status:** pre-registration. Written on 2026-10-11 and committed **before any non-smoke run of these items**. One labelled CPU
smoke ran before this commit (§3, 4 windows of one T7 run). It checked the pipeline and the glossary's cost, and it changes nothing
below. Any later change goes in §8, with its date and reason.

**Origin:** author request, 2026-10-11. After the session's review of the E9 results (T5, T4 and T7 at 3 seeds, the frame swap,
T7 INT4), the author asked to queue the six proposed follow-ups 1–6. Item 7, the bit-matched unstructured side module (claim B,
B-B1), is to be re-evaluated later. Each item closes a specific gap in R9's readings:

| # | Item | Gap it closes |
|---|---|---|
| 1 | Claim-B controls on T7 (GPTQ, AWQ, HQQ, NF4, channel off, quantized embedding) | Is T7's INT4 growth of C5's gain (1.63× after seen terms) an RTN artefact or a property of the host weights? |
| 2 | `C2w`: C2's free table with 256-wide rows (C5's composed width), T4 and T7 | The "compositional parameter sharing" reading compares C5sh with a capacity-matched C2 whose rows are 34 wide |
| 3 | In-context frame glossary for the loss (`e9_incontext`), T4, T5, T7, T7-ROOD | Required baseline A-B3 (novelty check §7, P0), which has no loss-endpoint implementation |
| 4 | `C6gi`: TransE trained without held-out entries, held-out rows folded in, on T7-ROOD | C6g's TransE is transductive (held-out triples in its training graph), so its ties with C5 favour it |
| 5 | T4 SmolLM2-135M, seeds 2–3 (C0′, C2, C5) | "The smaller host gains more on natural text" rests on 135M seed 1 |
| 6 | `C5ld`: C5 with link dropout 0.3 in training, T7 | Can training remove the held-out cost, given that injecting at names the host learned unlinked costs loss (T4 frame swap, T7 +0.48%)? |

**Code:**
- `training.lm.drop_links` (opt-in `channel.link_dropout`; evaluation links every span);
- `e9_rowsource` kind `kge_inductive` (`kge_graph(exclude=…)`, `fold_in_transe`);
- `e9_plan` arms `C2w`, `C5ld`, `C6gi`;
- `e9_report` arm labels;
- new module `e9_incontext`;
- tests in `tests/test_e9_pq.py`.

The recipe, the windows, the strata, the seeds and the 360M host are those of the base batches unless a row below says otherwise.

## 1. T7 claim-B controls

**Runs:** `e9_rescore` with the `controls` variants (`ref ref-off int4-A int4-A-off int4-A-emb int4-hqq int4-nf4 int4-gptq int4-awq`)
on T7 P0 (s1) and on C0′, C2 and C5 (s1–3). Queue priority 53.95. The official `t7-report-s1-2-3-pq` and `t7-report-s1-2-3`
(priority 54) read them, through `e9_report`'s claim-B section.

**Primary:** C5 vs C0′, `DiD` on `after_rare_seen` and `after` (strata where RTN gave a negative DiD, i.e. growth) under
`int4-gptq`, and `DiD off`, the same comparison with the channel off at both precisions.

**Predictions:**
- (a) GPTQ removes most of the growth (`int4-gptq` DiD closer to 0 than `int4-A`). RTN is the noisiest quantizer.
- (b) With the channel off, the DiD keeps the sign of `int4-A`. As on T5 for all tokens, the robustness is in the C5-trained
  host weights.

**Reading:**
- If `DiD off` ≈ `DiD` (CI overlap), the T7 growth belongs to the jointly trained host, not to the channel at inference.
- If `channel DiD` < 0 (Holm), the quantized host reads the channel better than the bf16 host did.
- If GPTQ's DiD CI includes 0, the growth is an RTN effect.

In every case claim B stays a measurement.

## 2. `C2w`: the free table at C5's width

**Runs:** T4 and T7, SmolLM2-360M, seeds 1–3, C2's recipe with `free_dimension: 256` (rows 256 wide, projected to the host
like C2's), with no capacity matching. Parameter counts, computed from the ontologies:
- T4 (67,545 entries): the C2w table has 17.3M parameters, against C2's 34-wide rows (2.3M) and C5's 2.35M channel budget.
- T7 (8,184 entries): the C2w table has 2.1M parameters, against C2's 119-wide rows (0.97M) and C5's 1.09M budget.

So the width question is mainly T4's. Evaluations use the `pq` chain. Priorities 54.33 (T4) and 54.331 (T7). Reports `t4-report-addendum` and
`t7-report-addendum` (§7).

**Primary:** `C5sh − C2w` on `after_len3plus` (T4), point estimate and 95% window bootstrap pooled over seeds. Negative means
the wrong-frame composed code is better.

**Prediction:** C2w beats C2, but stays behind C5sh on T4 (C5sh − C2w < 0).

**Reading:**
- If C5sh − C2w < 0 with a CI excluding 0, sharing atoms (not row width) carries the gain. The "compositional parameter
  sharing" wording stays.
- If the CI includes 0 or C2w is better, the wording becomes "a per-term vector of sufficient width". The sharing claim is
  withdrawn for natural text.

T7 is the replication; there C5sh and C2 were already close (−0.56% vs −0.46%).

## 3. In-context frame glossary for the loss (`e9_incontext`)

**Runs:** evaluation only. P0 (s1), C0′ and C5 (s1–3) on T4, T5 and T7 at priority 54.34. T7-ROOD runs get the same at
54.49716, after their training. Variants:
- `none` (replays the final evaluation);
- `frame` (a `Glossary:` prefix with one line `<name>: <verbalized frame>` per entry linked in the window, in order of first
  occurrence, capped at 1,024 tokens);
- `other` (the same number of lines for entries not in the window).

Only the window's own targets count. The verbalized frames are the C6d texts, `e9_rowsource`.

**Primary (per track):** "channel given the glossary", `C5 frame − C0′ frame`, on the strata where C5 − C0′ is negative:
- T5: `after_heldout`;
- T4: `after_unseen`, `after_rare_seen`, `after_len3plus`;
- T7: `after_rare_seen`, `after_len3plus`.

Secondaries:
- in-context benefit for C0′ (`C0′ frame − C0′`) against the channel (`C5 − C0′`);
- the glossary given the channel (`C5 frame − C5`);
- specificity (`frame − other`);
- the glossary's token cost.

Holm is applied over the comparisons within a stratum.

**Predictions:**
- (a) The glossary helps C0′ at least as much as the channel does, on every stratum where the channel helps. A verbalized
  frame in context is the upper bound for text evidence, as E11's reading (d) anticipates.
- (b) `C5 frame − C0′ frame` is small: |relative| < 0.5% on T4/T7, and smaller than C5 − C0′ on T5.
- (c) `frame − other` < 0 (Holm) wherever the glossary helps.

**Reading:**
- If (a) and (b) hold, the channel's case is persistence at zero context tokens. The paper states the token cost per window
  next to the loss.
- If `C5 frame − C0′ frame` < 0 (Holm) on a primary stratum, the channel adds to an in-context definition there.
- If the glossary does not help C0′, in-context frames are no competitor on that track.

## 4. `C6gi`: an inductive TransE row source on T7-ROOD

**Table:** `e9_rowsource build --kind kge_inductive`. TransE (C6g's recipe) is trained on the T7-ROOD ontology triples, leaving
out every triple whose head or tail is a held-out entry. Each held-out entry is then folded in, with everything else frozen, by
the exact L1 fit `argmin_h Σ_k ‖h + r_k − t_k‖₁` over its own triples: the per-dimension median of `t_k − r_k`. The rows are
whitened over non-held-out entries, as for every row source.

**Runs:** T7-ROOD, SmolLM2-360M, seeds 1–3. Priorities:
- the table build and training at 54.49705, after the T7-ROOD base trainings at 54.497;
- the `pq` evaluations at 54.49706;
- the frame swap (held-out and linked sets) at 54.4971.

The existing reports `t7rood-report-s1-2-3-pq` (54.4973) and `t7rood-frameswap-report` (54.4972) read them.

**Primary:** `C5 − C6gi` on `after_heldout` (T7-ROOD). Negative means zero-shot composition is better.

**Prediction:** C5 is ahead of or tied with C6gi on `after_heldout`, and C6gi is at least as good as C5 on seen strata (as C6g
was on T4).

**Reading:**
- If C5 − C6gi < 0 (Holm within the row-source arms), composing a frame zero-shot beats folding a new entry into a fixed KG
  embedding. This is the composition-specific zero-shot claim on natural text, conditional on T7-ROOD's primary P1.
- A tie means any frame-derived vector does as well; the claim wording follows.

## 5. T4 SmolLM2-135M, seeds 2–3

**Runs:** C0′, C2 and C5, seeds 2–3, with the full evaluation chain and the claim-B rescoring. P0 is seed-independent.
Priority 54.32. `t4-report-s2-3` regenerates `report/t4` from all T4 runs (a superset of the official report).

**Prediction:** 135M's seed-1 gains replicate: 135M gains more than 360M after unseen, rare and 3+-subtoken terms.

**Reading:** the size comparison enters R9 only if the 3-seed intervals of 135M − 360M (per stratum, C5 − C0′) exclude 0.

## 6. `C5ld`: link dropout in training

**Runs:** T7, SmolLM2-360M, seeds 1–3. C5 with `channel.link_dropout: 0.3`: each linked span of a training batch is not
injected, with probability 0.3, deterministically per (run seed, step, micro-step). Evaluation links every span. Priorities:
- training at 54.332;
- the `pq` chain at 54.3321;
- the frame swap, held-out (`heldout unseen rare_seen`) and linked sets, at 54.3322;
- reports at 54.3399.

**Primary:** `C5ld − C5` on `after_heldout`. Negative means link dropout reduces the held-out cost.

**Secondaries:**
- `C5ld − C0′` on `after_heldout`;
- `C5ld − C5` on `after_rare_seen` and `after_len3plus` (the price on seen terms);
- the frame swap's `empty − own` for C5ld.

**Prediction (low confidence):** the held-out cost shrinks (C5ld − C5 < 0 on `after_heldout`), at a seen-term price under
0.1%. The host learns that the same name occurs with and without its vector, so an injection at a name it learned unlinked is
less disruptive.

**Reading:**
- A held-out gain with no seen-term price makes link dropout part of the recipe.
- A held-out gain bought with a seen-term price is a trade-off to report.
- No held-out change means the held-out cost is not a train/test linking mismatch of this kind.

## 7. Reports and multiplicity

- `t4-report-addendum` and `t7-report-addendum` (`e9_report` on all runs of the track; `report/<track>-addendum`) at 54.3399.
  The arm sections compare C5 with C2w, C5ld and C6gi, Holm over the arms present within a stratum.
- `incontext-report` (`e9_incontext report`, T4/T5/T7) at 54.3402, and `t7rood-incontext-report` at 54.49736.
- `t7-addendum-frameswap-report` (`frameswap/t7` and `frameswap/t7-linked` re-reported with C5ld) at 54.3399.

Each item's primary is read on its own, at its own family's Holm correction; no item's reading depends on another's.

## 8. Changes after this commit

(none)
