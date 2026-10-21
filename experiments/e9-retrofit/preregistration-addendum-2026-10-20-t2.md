# E9 — addendum of 2026-10-20 (T2): the developer-tools track in the retrofit

**Status:** pre-registration. Written on 2026-10-20, **before any non-smoke run on T2**; it is to be committed before the queue script runs (the agent that wrote it did not commit). Two labelled CPU smokes
ran before this file was written (§8): four training steps of the T2 C5 configuration on SmolLM2-135M, and the T2 zero-shot
evaluation on a 24-item subset. They checked the pipeline only and change nothing below. Any later change goes in §10, with its
date and reason.

**Origin:** author, 2026-10-20 ("approve all … run them and queue"). The research status of 20 October
(`manuscript/status/10-20-2026/08-e9-new-vocabulary-and-clinical.md` §08.4) lists T2 as "built and feasible but not queued", and
the only unrun track that can test the method on both the held-out and the rare stratum.

## 1. Question

Does the composed embedding (C5) lower the loss after **API symbols the host has never read** when the domain text is
developer documentation rather than T5's glossary prose? T2's private libraries are invented (contamination-free, like T5),
but its documents are API references, module sources, type stubs, examples, tutorials, Q&A threads, issue reports, changelogs
and code reviews, mixed with real-library documentation (10% of the domain characters) and FineWeb-Edu (50% of the tokens).
Held-out symbols never occur in training text (`experiments/t2-developer-tools/README.md`, step 5), so they are a clean
zero-shot test on the same terms as T5 and T7-ROOD.

## 2. Track, host and recipe

- **Corpus:** `runs/v1` of `experiments/t2-developer-tools/` (SmolLM2 tokenizer; 100M training tokens, 50M domain / 50M
  general; the E9 recipe reads the first 50M, as on every track). Ontology: 55,227 entries (55,412 concepts), 16 relations,
  8,192 atomics, 9.0 edges per entry; alias normalization `identifier` (README "Linking code symbols").
- **Holdout:** 1,078 symbols (899 private, 179 real), pinned `75ee2217be97bd9d…`; plus the 600 zero-shot symbols of
  `items/synthetic_concepts.jsonl` (defined only by schemas, in no document), which enter the ontology as held-out entries.
- **Evaluation:** the domain split `eval` on **3,072 windows** of 1,024 tokens — the fewest that meet WP-T1's strict bar at
  ℓ_min 2 (`runs/v1/feasibility_strict.json`; C7's criterion already holds at 512 windows). On these windows (counted on
  2026-10-20, before any run): 164,767 linked spans (96.6% private); **after held-out: 6,720 occurrences of 920 entries (449
  with ≥ 5)**, 93% private; rare-seen 9,696 (2,205 entries), unseen 5,082. General text: `eval-general` (C3's 5,000
  evaluation documents).
- **Host and arms (the T4 core set):** SmolLM2-360M, full fine-tuning; **P0** (the original host, evaluation only), **C0′**
  (plain fine-tuning), **C2** (free per-entry table matched to C5's parameters), **C5** (the method: HRR, 256 dimensions,
  attentive, key 8, P1 window 8); seeds 1–3 (P0 once). Recipe unchanged from `e9_plan`: 50M tokens, 64 × 1,024 tokens per
  step, host lr 3e-5, channel lr 1e-3, gate bias 0, warmup 2.5M tokens.
- **Code:** `e9_tracks.TRACKS["t2"]` (corpus root, windows, WP-C7 items, holdout list, relation choices) and the T2 branch of
  `track_lexicon` (the track's own `tracks.devtools.ZEROSHOT` wordings; fillers by `t2_atom_text`); tests in
  `tests/test_e9.py` (`test_t2_*`). The evaluation alias table `~/data/vsa-llm/e9/alias-tables/t2.json` replays the build
  exactly (digest equal to the ontology's and to the build's `alias_table.json`; `identifier` mode kept); it was written on
  2026-10-20 by the item build below.

## 3. Evaluations (the standard E9 chain, as on T4)

Per run (P0 included): channel probes at bf16 and INT4; the WP-C7 zero-shot items (`items/zeroshot_property.jsonl`,
`zeroshot_entailment.jsonl`: the 600 zero-shot and the held-out symbols) at bf16 and INT4; the dimension-3 items (new words,
one-relation edits) at bf16 and INT4; the claim-B rescoring (`--rescore all`: filler / non-filler strata and the quantizer
controls). Then `e4_quant` on `eval` and on `eval-general`, and the R9 report (`report/t2`).

**Dimension-3 items** (built for T2 on 2026-10-20 with the standard builder, `e9_tracks items --track t2`, seeds 0 / name seed
11 / ℓ_min 2 / 20M contamination tokens, as T4, T7 and T8; written to `experiments/e9-retrofit/items/` by the queue script,
which checks these hashes):

| folder | counts | `items.jsonl` sha256 | `concepts.jsonl` sha256 |
|---|---|---|---|
| `new-words-t2-smollm2-v1` | 300 invented symbols, 4,102 items (1,751 property, 1,751 statement, 600 entailment) | `51ad2bc69f78023e…` | `2bec1f0111e4ef49…` |
| `edits-t2-smollm2-v1` | 199 edits (137 return type, 53 API status, 9 category), 796 items | `0f96d856950027ff…` | `51dc4c1ebd7200a1…` |

New words keep the donor's kind, library, language and owner and resample the rest (category = `kind`). An edit changes the
return type where a symbol has exactly one, else the category, else the status, to a filler of the same atom type.

## 4. Primary

**P1:** C5 − C0′ < 0 on `after_heldout` (relative loss, SmolLM2-360M, seeds 1–3 pooled, 95% cluster bootstrap over windows,
Holm over the references C0′ and C2 within the stratum, as R9 dimension 1).

## 5. Predictions (from the README's novelty and feasibility numbers)

- **Novelty is the highest of any track.** The decision-55 screen (`experiments/e9-retrofit/novelty/candidates.md`; P0 =
  SmolLM2-360M, 512 windows) measured 25.8 nats per linked span inside T2's terms: 4.84× T4 and 1.69× T5 (15.3), with 7.29
  subtokens per span and no occurrence in general text (general/domain 0.0000). The README adds that the identifiers are
  built from stems found in none of 269,377 forbidden words.
  - **N1:** the novelty index C0′ − P0 after held-out terms is at least T4's (−40%), likely T5-like (≤ −60%).
- **P1 holds** (C5 − C0′ < 0 after held-out, CI excluding 0). The R9 calibration (gain follows novelty: T5 −13.3%, T4
  −0.4%, T1 / WordNet 0) and held-out symbols that are never in training text, as on T5, point the same way. **Magnitude:**
  between T4 and T5, point guess −5% (low confidence). Two things pull it below T5. First, 3.4% of the linked spans in the
  windows are real symbols that the host may know. Second, the frames are larger and partly copied from the same library, so
  a held-out symbol's vector is less distinctive.
- **P2:** C5 − C2 < 0 after held-out (the free table has no held-out rows; on T5 C2 is worse than C0′ there).
- **P3 (rare and unseen):** C5 − C0′ < 0 after rare-seen and after unseen terms.
- **P4 (the copy concern):** most of the gain after rare and unseen terms falls on filler tokens (filler share ≥ 0.5,
  `e9_rescore` split). The private stream is templated (README "How much domain text is realistic": 11–28% new word 8-grams
  per 20M-character chunk), as on T5, where 86–90% of the rare/unseen gain was filler copying.
- **P5 (dimension 3):** on the 600 zero-shot symbols, C5 with its own frame scores above `random_frame` and `none` on property
  selection (as on T5: +0.125).
- **Locality:** no cost on `unlinked` or general text beyond ±0.2%.

## 6. Decision reading (fixed now)

- **P1 holds with N1:** T2 is the second contamination-free track with a held-out gain, in documentation text with a real
  long tail. Claim A's zero-shot sentence may cite T2 next to T5, with the decision-28 caveat (templated synthetic text, the
  repetition statistic reported) and P4's filler share next to the number. T2 is not natural text and does not replace
  T7-ROOD as the real-world zero-shot test.
- **P1 fails (CI includes 0 or positive):** the T5 held-out gain is specific to the glossary generator. The zero-shot wording
  narrows to T5, and R9 records T2 as a second synthetic track without the effect.
- **P1 holds but N1 fails** (low novelty): this contradicts the P0 screen. Report it and do not use T2 in the novelty ordering.
- **No ontology-specificity claim from this block.** Without C5sh (rule (a) of decision 64) the gain cannot be called
  frame-specific; C5sh and C6d on T2 are a follow-up for the author to decide, not queued here.

## 7. Cost

The GPU-hours below come from measured durations of the analogous T4 SmolLM2-360M jobs in `.jobs/*.json` (2026-10-03 to 10-11).
They are scaled to T2's 3,072 windows and item counts.

| block | jobs | basis | GPU-h |
|---|---:|---|---:|
| training C0′ / C2 / C5 × 3 | 9 | T4: 66.5–67.9 min per run (median 67.1), + 2,048 extra windows at 7 evaluation points (24 s per 1,024 windows) ≈ 72.7 min | 10.9 |
| P0 (evaluation only) | 1 | T4 0.5 min, × 3 for the windows | 0.03 |
| probes bf16 / INT4 | 20 | T4 medians 2.1 / 4.0 min (track-independent probe sets) | 1.0 |
| zero-shot bf16 / INT4 | 20 | T4 medians 1.7 / 3.8 min × 2.8 (20,617 T2 rows vs 7,425 T4 rows) | 2.5 |
| dimension 3 bf16 / INT4 | 20 | T4 medians 1.1 / 2.3 min × 1.7 (4,102 + 796 items vs 2,022 + 800) | 1.0 |
| claim-B rescoring | 10 | T4: P0 / C0′ 5.9, C2 / C5 7.5 min at 1,024 windows, × 2.5 for 3,072 windows (window factor 0.25 + 0.75 · w/1,024, which reproduces T7's 4,096-window jobs: 17.9 / 23.9 min) | 2.9 |
| `e4_quant` on `eval` and on `eval-general` | 2 | 4 / 7.2 min per base / channel run (`EVAL_MINUTES_360M`, T1/T4-measured) × 2.5 for the windows | 4.9 |
| R9 report | 1 | CPU lane (`-report`; T4: 23–64 min) | 0 |
| **total** | **83** | | **≈ 23 GPU-h** |

`e9_plan --dry-run` (with `--rescore all`) gives 83 jobs and ≈ 18.8 GPU-h. It does not scale `e4_quant`, zero-shot or editing by
windows or items, so the table above is the planning number. With 1,024 windows (as T4), the block would cost ≈ 18 GPU-h. But
`after_heldout` would then have 2,147 occurrences of 611 entries (114 with ≥ 5), instead of 6,720 / 920 / 449.

## 8. Smokes before this commit (CPU only, `CUDA_VISIBLE_DEVICES=`, labelled, not results)

- `e9_plan` configs of T2 for SmolLM2-135M (C5, C2, C0′), overridden to 128-token sequences, micro-batch 2, 4 steps, 4 evaluation
  windows. C5 trained 4 steps: training loss 3.28 → 3.58 (4 batches), evaluation loss 3.453 → 3.410 over all tokens, all strata
  present (held-out spans found in 4 windows), 4.0 GB peak RSS, 66 s.
- `e9_tracks zeroshot` on that run with 12 property groups and 12 entailment pairs (6 synthetic + 6 held-out each), all four
  sources (`own`, `none`, `mean_row`, `random_frame`): every surface linked (identifier normalization), 5:49 min on CPU.
- The alias-table replay (4.5 s) and the two item builds (37 s and 16 s; 2.2 / 1.1 GB). Rebuilt once more: byte-identical.
- The T2 filler table (`e9_rescore fillers --track t2`, 5 s; built at job time by the first rescoring), and `e9_rescore score`
  on the smoke run (`ref`, `ref-off`, `int4-rtn`): ref check paired, maximum window difference 0.0, filler strata present
  (20 of 169 targets after linked terms are filler tokens in the 4 windows).

## 9. Queue

`experiments/t2-developer-tools/queue-commands-e9.sh` (not run by the agent that wrote it): it builds the dimension-3 items
(CPU, checked against §3's hashes), then `e9_plan --track t2 --hosts SmolLM2-360M --models P0 C0p C2 C5 --seeds 1 2 3
--rescore all --priority $PRIO_T2 --level-step 0.0001 --queue`. The report job's name contains `-report` (CPU lane).

## 10. Changes after this commit

(none)
