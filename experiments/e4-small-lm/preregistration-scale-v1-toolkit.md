# From-scratch scaling screen, phase 1b — the toolkit arms C5dev and C5teach (pre-registration addendum)

**Status:** design and pre-registration, written on 2026-10-10 and committed **before any non-smoke run of these arms**.
What ran before this commit: two labelled CPU SMOKEs of the developmental tracker and the trainer (§7; plumbing and cost
only) and the teacher pilot (§5: 400 C3 entries read by Claude, USD 2.48 spent of the USD 5 cap; a measurement of the
teacher, not a language-model run). Neither changes §§1–4. Any later change is listed in §9 with its date and reason.

**Origin:** decision 65 (author, 2026-10-10). Phase 1a (another package; stage `scale-v1`, priority 54.3) trains C0, C2,
C5, HRRAdd, HRRCat and C5sh from scratch at 20M / 50M / 125M parameters on 500M tokens of the C3 corpus (FineWeb-Edu linked
with WordNet: `~/data/vsa-llm/c3/wordnet-gpt2-v1`, 101,500 entries, 8,192 atomics, 16 relations, 5,900 held-out entries)
and has its own pre-registration. Phase 1b adds the author's toolkit arms at **50M × 500M tokens, seed 1**, same corpus,
recipe and evaluation windows, read against phase 1a's 50M C5 and C0 (seed 1). **Phase 1b is exploratory** (one seed, one
size); it decides only whether an arm goes to phase 2 (§4).

**Code:** `src/vsa_embed/experiments/scale_toolkit.py` (configs, overhead SMOKE, queue script),
`src/vsa_embed/experiments/c3_teacher.py` (teacher ontology: sample, teach, evaluate, ontology, reference, project),
`src/vsa_embed/developmental.py` (opt-in periodic consolidation, below), `e4_report --pool-ontologies` (C5teach shares
C5's cohort); tests `tests/test_scale_toolkit.py`, `tests/test_c3_teacher.py`. Configs
`experiments/e4-small-lm/configs/scale-v1-toolkit/`; queue script `experiments/e4-small-lm/queue-commands-scale-v1-toolkit.sh`
(not executed by the agent that wrote it).

## 1. Arms

Both arms are phase 1a's 50M C5 with exactly one change (`tests/test_scale_toolkit.py` checks the config differences).

**Base config.** Phase 1a's `configs/scale-v1/50M-C5-s1.yaml` did not exist in `main` when this was written, so the base is
`configs/opscreen/50M-C5-s1.yaml` (the frozen D4.0 recipe: GPT-2-style 50M, 32,768 tokens per step, peak lr 2e-3, warmup
5M, cosine to 0.1×; channel HRR, attentive, key dimension 8, P1 context window 8, dimension 256) with 500M tokens
(15,258 steps) and log-spaced evaluations at 5M, 10M, 20M, 40M, 80M, 160M, 320M and 500M tokens (`eval.first_tokens` 5M)
on the same 1,024 windows of 1,024 tokens, with per-window losses. **Rule:** before the C5dev job is queued, the configs are
regenerated from phase 1a's 50M C5 config if it exists then (`python -m vsa_embed.experiments.scale_toolkit configs`
copies it and changes only the arm's keys), so the arms always share phase 1a's recipe; any difference this makes is
recorded in §9.

**C5dev — passive learning (developmental dictionary).** `channel.developmental` (`vsa_embed.developmental`, M3): EMAs of
each atomic's gradient give coherence and activity; every `screen_every` steps the least coherent active atomics become
candidates, their per-usage (per-entry) gradients are recorded until the next screen, and a candidate is split along the
top between-usage direction when its split gain beats a permutation null; the parent keeps unobserved usages (held-out
entries among them) and tracks the usage-weighted mean of its children; sibling pairs that stay near-identical are merged.

| setting | value | why |
|---|---|---|
| target | atomics | E3 split no relation on WordNet; atomic senses are the hypothesis |
| route_unobserved / sync_parent | parent / true | the G1 default (gates.md, 2026-09-30): held-out cosine not hurt |
| screen_every | 250 steps (≈ 8.2M tokens; 61 rounds) | records ≈ 250 steps of per-usage gradients per candidate |
| activity_percentile / coherence_threshold | 0.25 / 0.6 | E3's WordNet screen |
| max_candidates / max_splits_per_round | 16 / 4 | bounded work per round |
| min_usages / min_contributions | 4 / 20 | E3 |
| permutations / p_value | 999 / 0.002 | ≈ 980 tests over the run: ≈ 2 false splits expected under the global null (≈ 10 at E3's p < 0.01) |
| growth_budget | 0.05 (≤ 409 splits; ≤ 818 new rows) | E3 |
| cooldown | 1,000 steps | a split pair is not re-tested for ≈ 4 rounds |
| consolidate_every / merge_min_age / merge_cosine | 1,000 / 1,000 / 0.98 | new, below |
| epsilon / beta / test / seed | 0.05 / 0.9 / permutation / 1 | defaults |

*Consolidation.* E0 showed the screen alone over-splits (raw false-split rate 0.38) and that consolidation is required, but
the LM trainer never called `consolidate()`. `DevelopmentalConfig` gains `consolidate_every` and `merge_min_age` (default 0
= off: every earlier config is unchanged); with them `grow()` merges sibling pairs at cosine ≥ 0.98 that were split at least
`merge_min_age` steps ago (new children start at cosine ≈ 0.995, so a younger pair would merge at once), and a
parent-fallback link moves the retired child's usage weight to the kept child.

*Cost.* §7: the tracker's work, forced to its maximum (16 candidates every round, parent sync with the budget used up),
adds ≈ 4.8% of the 50M step time (≈ 23 ms per 0.485 s step); the queue script budgets C5dev at C5's 2.09 GPU-h × 1.048
≈ 2.19 GPU-h.

**C5teach — learning from reading (Claude as teacher).** The same C3 entries, relation types, atomics, held-out entries and
training frequencies; only the edges differ. Claude (`claude-opus-5-5`, `claude -p`, effort `medium`, prompt
`c3-teacher-v1`, 25 senses per call) reads each WordNet sense of an entry — its first lemma, part of speech and gloss
(definition and examples; never the synset id or the synonym list) — and writes WordNet-style (relation, filler) facts with
fillers named as WordNet synsets. The parser (`c3_teacher.FillerMapper`) maps a filler to its atomic (`exact`), an existing
synset outside the dictionary to its nearest in-dictionary hypernym (the rule the WordNet builder applied to WordNet's own
pointers: `ancestor`), a wrong sense number or a bare word to the word's first atomic sense of the part of speech the
relation expects (`lemma`, `lemma_ancestor`), `lexname` / `pos` fillers to their closed vocabularies, and records every
unmapped filler with its reason. A sense frame keeps one lexname and one pos and ≤ 16 edges (C3's cap); an entry's frame is
the union of its senses' frames (C3's rule). Two variants (§6):

- **C5teach (hybrid; recommended):** teacher frames for the 5,900 held-out entries and 5,900 trained entries matched on
  evaluation-corpus span-count band × sense-count band (seen in training), WordNet frames elsewhere; the run also scores
  the reference strata of the two teacher entry sets (`ref_after_teacher_heldout` = `after_heldout`;
  `ref_after_teacher_trained`). Phase 1a's 50M C5 and C0 (seed 1) are rescored on the same reference strata by
  evaluation-only jobs (`C5@teachref`, `C0@teachref`; `train.eval_only`, `init_from` their `final.pt`).
- **C5teachF (full):** teacher frames for every entry.

The teacher ontology is built and frozen (its `teacher` record holds the parse statistics and the sha256 of the answers
file) before any C5teach run; the build needs the author's approval of the spend, so the C5teach jobs are commented out.

## 2. Comparisons, strata and statistics

**Comparisons** (seed 1): C5dev − C5, C5teach − C5, C5dev − C0, C5teach − C0, all against phase 1a's 50M runs.
**Strata:** `all`, `unlinked`, `after`, `after_len3plus`, `after_rare_seen`, `after_heldout` (the trainer's; `after_*` =
the 8 targets after a linked span); for the hybrid C5teach also `ref_after_teacher_trained`. Note the sizes on the 1,024
windows (opscreen): `after_heldout` 53,768 targets, `after_len3plus` 90,491, `after_rare_seen` only 643 (wide intervals).

**Statistic:** relative loss difference `(arm − reference) / reference` at the final evaluation (500M tokens),
token-weighted, with a 95% cluster bootstrap over the evaluation windows (10,000 resamples, seed 0;
`e4_report`, `statistics.paired_ratio_bootstrap`), negative = the arm is better. Holm over the four comparisons within
each stratum. C5teach is on another ontology of the same entries, so it is analysed with `e4_report --pool-ontologies`
(runs: `runs/scale-v1` 50M seed 1 and `runs/scale-v1-toolkit`). With one seed the interval reflects window sampling only,
not seed variance (opscreen at 100M, C5 seeds 1–3, range of the final loss: `after_heldout` 0.07%, `after` 0.10%,
`after_len3plus` 0.12%, `after_rare_seen` 1.1% relative); **differences smaller than 0.2% relative (1.5% on
`after_rare_seen`) are not interpreted** whatever their interval.

**Data multiplier** `k = D_C0(L) / D_arm(L)` per stratum (`convergence.data_multiplier` on the log-spaced curves 5M…500M, at
the hardest loss target the curves share), for C5dev, C5teach and C5 against C0; a point estimate (one seed), reported
next to k(C5). opscreen (100M, 3 seeds): k(C5) ≈ 1.06 on `after`, 1.02 on `after_heldout`.

**Diagnostics** (reported, not tested): C5dev — splits and merges per round (`cards.json`), dictionary size, parent sync,
training throughput vs C5; C5teach — the ontology's edges per entry, agreement with WordNet over every teacher entry,
parse rate and spend.

## 3. Predictions (recorded before any run)

- **C5dev − C5:** no difference worth reading: |Δ| < 0.2% on every stratum, `after_heldout` non-inferior (upper bound
  ≤ +0.3%). Fewer than 100 accepted splits, most in the first third of training, and few merges. Basis: M3 recovered planted
  senses only in the synthetic teacher world (G1); on WordNet with frozen anchors (E3) it failed (precision 0.07, false-split
  rate 0.19) and test MRR equalled uniform enlargement; LM gradients are noisier still. k(C5dev) = k(C5) ± 5%.
- **C5teach − C5 (hybrid):** on `after_heldout` within ±0.3% (teacher frames for new words carry the composition gain about
  as well as WordNet's), on `ref_after_teacher_trained` within ±0.3%; strata outside the teacher sets unchanged (|Δ| < 0.1%).
  Basis: the pilot's teacher frames have WordNet's density (5.1 vs 5.2 edges per entry), agree on 78% of edges overall and
  on 56–59% of the pointer edges (the rest are mostly neighbouring senses or ancestors), so the composed rows should be
  close; the opscreen arms that changed the operator moved `after_heldout` by ≤ 0.1%.
- **C5teach − C5 (full variant, if run instead):** the same bounds on `after_heldout`, `after` and `after_len3plus`.
- **vs C0:** both arms keep at least 75% of C5's gain over C0 on `after`, `after_len3plus` and `after_heldout` (opscreen
  at 100M: C5 − C0 = −0.99%, −1.32%, −0.37%), and are no worse than C0 on `unlinked` (locality).

## 4. Decision rule for phase 2

Phase 2 = seeds 2–3 at 50M plus the 125M size, with phase 1a's statistics (3 seeds, Holm, gate-style reading).

1. **C5dev advances** only if, at seed 1, C5dev − C5 ≤ −0.2% with the 95% interval below 0 on `after` or on
   `after_len3plus`, **and** `after_heldout` is non-inferior (upper bound ≤ +0.3%), **and** at least 10 splits were accepted.
   Otherwise it is dropped (a null at seed 1 is not followed up; a split count below 10 means growth was not tested and the
   screen settings, not the idea, are revisited first).
2. **C5teach advances** if, at seed 1, (a) C5teach − C5 ≤ −0.2% with the interval below 0 on `after_heldout` or on the
   teacher-trained stratum (reading beats the curated ontology), **or** (b) C5teach is non-inferior to C5 on both
   (upper bounds ≤ +0.5%) **and** C5teach − C0 < 0 with the interval below 0 on `after_heldout` (reading substitutes for
   curation). Phase 2 of an advancing hybrid arm builds the full teacher ontology first (C5teachF), subject to the author's
   approval of that spend. If C5teach − C5 ≥ +0.2% with the interval above 0 on `after_heldout`, the arm is dropped.
3. Anything else is inconclusive: no phase 2 by default; the author decides. Phase 1b results are always reported as
   exploratory (one seed), whatever the rule says.

## 5. Teacher pilot (ran before this commit; `experiments/e4-small-lm/scale-v1-toolkit/teacher-pilot/`)

Sample (`sample.json`, seed 20261010): 200 held-out entries, 40 per evaluation-corpus span-count band (0, 1–2, 3–9, 10–29,
30+; held-out entries have no training frequency), and 200 trained entries, 40 per training-frequency band (0, 1–9, 10–99,
100–999, 1000+); 829 senses. HARD cap USD 5 (cumulative over the ledger, enforced before each call; a per-call
`--max-budget-usd`; failed calls charged).

| measure | value |
|---|---|
| spend | USD 2.41 for the 400 entries (34 calls, none failed) + USD 0.07 for one calibration call = **USD 2.48** |
| USD per entry / per sense | 0.0060 / 0.0029 (median USD 0.071 per call of 25 senses; ≈ 2,100 output tokens) |
| wall time | 132 s for 33 calls at 4 workers (+ 16 s for the first call) ≈ 0.37 s per entry; median call 15.8 s; 0.65 call-s per sense |
| answers / parse rate | 829 / 829 senses answered; 2,710 proposed edges, **97.3% mapped** (exact 690, ancestor 284, lemma 4, lexname 829, pos 829); unmapped: no in-dictionary ancestor 61, unknown word 10, self 3 |
| edges per entry | teacher 5.06 vs WordNet 5.20 (without lexname / pos: 2.34 vs 2.47) |
| agreement with WordNet (micro P / R) | all edges 0.78 / 0.75; pointer edges 0.59 / 0.56 (held-out 0.56 / 0.52, trained 0.65 / 0.62); fillers regardless of relation 0.60 / 0.58 |
| per relation (P / R) | hypernym 0.59 / 0.59, instance_hypernym 0.70 / 0.76, part_holonym 0.73 / 0.60, member_holonym 0.78 / 0.60, similar_to 0.82 / 0.60, antonym 0.59 / 0.58, topic_domain 0.23 / 0.38, lexname 0.89 / 0.88, pos 1.00 / 1.00; meronyms under-produced (recall 0.28–0.33), entailment never produced |

Ten examples with both frames are in `teacher-pilot/report.md`. Typical disagreements are a neighbouring sense number
(`set.n.01` for `set.n.02`), a sibling or ancestor hypernym (`begin.v.03` for `get_down.v.07`), extra topic domains and
missing meronyms; the frames are otherwise WordNet-like.

## 6. Projection and recommendation

From the pilot's USD and call-seconds per sense (`teacher-pilot/projection.json`; 4 parallel workers):

| variant | senses | USD | teacher hours |
|---|---:|---:|---:|
| full (every entry; C5teachF) | 117,659 | ≈ 342 | ≈ 5.3 |
| hybrid (5,900 held-out + 5,900 matched trained entries; C5teach) | 25,282 | ≈ 73 | ≈ 1.1 |
| held-out entries only | 12,731 | ≈ 37 | ≈ 0.6 |

**Recommendation: the hybrid.** It costs a fifth of the full build and carries the two comparisons the arm is about: new
words learned by reading their glosses (`after_heldout`, every held-out entry teacher-framed) and seen words with teacher
instead of curated frames (the matched stratum, same evaluation-span counts, so the same power), while every other entry
keeps C5's frames, so a difference is attributable to those entries' edges. It is also the toolkit's deployment case (new
words read into an existing ontology). The full build answers a different question (an ontology built entirely by
reading, its atomics trained only on teacher frames) and is worth its cost only once the hybrid passes rule 4.2. Cost of
the hybrid's extra machinery: two evaluation-only rescoring jobs (≈ 0.02 GPU-h each).

## 7. SMOKEs (CPU, before this commit; not results)

- `scale-v1-toolkit/smoke/overhead.json` (SMOKE): the tracker at the 50M channel's real size on CPU — real C3 training
  batches (32 × 1,024, ℓ ≥ 2, held-out masked), the real composer (8,192 atomics, 256-d HRR, attentive, P1 context), a
  stand-in host (frozen random embeddings, next-token dot-product loss) — with the coherence threshold forced above 1 so 16
  candidates are recorded and tested every round, 760 steps (three screens, two test rounds), 6 CPU threads. Per step:
  `observe` with candidates 2.9 ms, parent sync with all 409 budgeted splits 3.8 ms; per test round (16 candidates, 999
  permutations; the largest candidate had 1,629 contributions) 2.8–5.6 s, i.e. 16.7 ms per step at one round per 250
  steps. Total ≈ 23 ms per step = **4.8%** of the 50M GPU step (32,768 tokens at 67.6k tokens/s), an upper bound on the
  tracker's work; on the GPU the per-step part adds a device-to-host copy of the 8 MB atomic gradient and the parent sync
  becomes ≈ 1,200 small kernels at full budget (a few ms more). One split was accepted (stand-in loss; not meaningful).
- `scale-v1-toolkit/smoke/trainer-smoke.json` (SMOKE): C5 and C5dev through `training.lm` at the tiny size (2 layers,
  width 64, 120 steps, screen every 10 steps) on the real corpus: both train, evaluate every stratum and write their
  manifests; C5dev accepted one split (step 110, a lexicographer-file atomic: its 63 observed usages went to the two
  children, the parent keeps the ≈ 13,800 unobserved ones)
  and logged its card; the two runs' final losses agree to 4 decimals (the split came 10 steps before the end); wall time
  43 s vs 37 s on CPU at this size (the tracker's fixed costs dominate a tiny model; not an estimate for 50M).

## 8. Files

Configs `configs/scale-v1-toolkit/50M-{C5dev,C5teach,C5teachF,C5@teachref,C0@teachref}-s1.yaml`; runs (when queued)
`runs/scale-v1-toolkit/`; teacher store `~/data/vsa-llm/c3-teacher/wordnet-gpt2-v1/` (`answers.jsonl`, `ledger.jsonl`,
`runs.jsonl`, per-call cache `calls/`, later `ontology-{hybrid,full}.pt` and `reference-hybrid-1024x1024.npz`); the pilot's
answers, frames, summary and report are copied to `scale-v1-toolkit/teacher-pilot/` (WordNet is public; no licensed text
was sent to the teacher).

## 9. Amendments

None yet.
