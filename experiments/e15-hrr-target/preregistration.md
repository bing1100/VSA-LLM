# E15 — HRR as a training target ("HT"): pre-registration, stage 1 (T5)

**Status:** pre-registration, written on 2026-10-22 and committed **before any GPU job of E15 is queued**. The only
runs before this commit are the labelled CPU smokes of §5 (loss scales, a few steps of SmolLM2-135M) and §14 (pipeline
checks on that smoke run); they fix λ and change nothing else here. Any later change goes in §15 with its date and reason.

**Origin:** the author's approval of 2026-10-22 — *"Run and queue this experiment with high priority — since they can
change the trajectory of remaining experiments"* — with the design fixed in that request (host, recipe, losses, arms,
splits, endpoints, predictions, decision rule, priorities). Stage 2 (T4) is registered in §11 and **not** queued.

**Background read before writing:** `manuscript/plain-language-summary-2026-10-22.md` §2–§5;
`manuscript/status/10-20-2026/09-binding-programme.md` §09.4 (the readout arms: hook at the input of decoder layer
"third" = 11 of 32, LM loss only, R1 and R2 null) and `06-e9-retrofit-pretrained-models.md` (C5 ≈ C5sh ≈ C2w on T5;
the T5 recipe); `experiments/e14-logic/report/rd4b/report.md` (T5 new words, SmolLM2-360M C5, 3 seeds: property
`none` 0.1896, `implicit:oracle` 0.2603, `explicit-rel:oracle` 0.8450).

## 1. Question

Can a pretrained language model learn to **read the HRR algebra** of a composed term vector when it is given a training
signal that only the algebra can satisfy — and, if it can decode the facts, does it then **use** them (the implicit
route) on terms it has never seen in text?

## 2. Diagnosis being tested

In C5 the composed vector `c = N(Σ_e T_{r_e}(a_e))` is only an input feature at the term's last subtoken. No loss ever
touches its structure from the model's side, so the model learns "this vector identifies the term": shuffled frames cost
almost nothing (C5sh), a free table at the same width matches it (C2w), and a readout head trained by the LM loss alone
does not use binding (09.4, R1 null). HT adds two losses on the hidden state at the term that cannot be satisfied by
memorising an identity: the hidden state must **write** the term's bundle, and the bundle must **read** back the right
filler under each role.

## 3. Design (fixed)

- **Host and recipe:** SmolLM2-360M, full fine-tuning, exactly E9's T5 recipe (`experiments/e9-retrofit/configs/t5/
  SmolLM2-360M-full-C5-s{1,2,3}.yaml`: 50M tokens, host lr 3e-5, method lr 1e-3, micro-batch 4 × accumulation 16 ×
  1,024 tokens, warm-up 2.5M tokens, cosine to 0.1, data seed 1234, run seeds 1–3, 1,024 evaluation windows with
  per-window losses). The existing T5 runs are the baselines: C5, C2, C5sh and C0′ (`SmolLM2-360M-full-{C5,C2,C5sh,
  C0p}-s{1,2,3}` under `experiments/e9-retrofit/runs/t5/`, each with `final.pt` and `eval_windows.npz`; verified
  2026-10-22). Nothing in those folders is modified.
- **Where the losses read:** the hidden state `h` at the term's last subtoken (the injection position) at the **input
  of decoder layer ℓ = 11 of 32** — `readout.readout_layer("third")`, the layer the readout arms of 09.4 used — captured
  by a forward pre-hook that changes nothing in the forward (`ChannelLM._capture_target`).
- **Write loss.** A linear map `W` (host width 960 → store dimension 256, no bias) must reproduce the composer's
  static frame bundle of the term, `b = N(Σ_e T_{r_e}(a_e))` (`FrameComposer.raw_bundle(uniform=True)`, the store the
  09.4 readout read): `L_write = 1 − cos(W h, b)`.
- **Read loss.** For every edge `(r, f)` of the term's frame, `û = unbind_r(W h)` with the store's own operator
  (`FrameComposer.unbind`: HRR correlation), and `L_read = CE(softmax(N(û) · Aᵀ / τ), f)` over the **whole**
  unit-normalised atom table `A` (3,493 atoms; chance = ln 3493 = 8.16 nats, top-1 chance 1/3493 = 0.00029), τ = 0.1
  fixed. This is Plate's cleanup memory as a trained objective.
- **Total:** `L_LM + λ_w L_write + λ_r L_read`, averaged over the linked occurrences (write) and their edges (read) of
  each micro-batch. λ fixed in §5.
- **The target is detached** (`detach_target: true`): the bundle, the roles and the atoms carry no gradient from the
  two losses. Justification: the host and `W` must learn to match the store, not the store collapse onto whatever the
  host emits; the store still trains through the LM loss exactly as in C5, and the two losses still reach the channel
  through the injected row (`h` depends on it). This keeps the hypothesis clean — "can the host learn to produce and
  decode the store's code?" — at the price that a store that is hard to decode is not repaired by the losses.
- **Facts used for the losses:** the edges of the frame the run **injects** (the composer's schedule). For C5sh+WR
  that is the shuffled frame: the losses teach the model to decode what it is given, so a decodable-but-wrong store
  separates "decodes" from "uses" (§8, secondary S1).
- **C2+R's decoder.** A free table has no composer. C2+R decodes through a **frozen random HRR code** over the
  ontology's frames (`hrr_target.ontology_decoder`: random roles and atoms, dimension 256, seeded on its own, never
  trained): the host must learn to write its free row into a fixed code. For a seen term this is possible by
  memorisation; for a term never seen the free row is C2's fallback (the mean trained row), which carries no structure
  to generalise from. No write loss (λ_w = 0): a free row has no bundle to match.
- **Initialisation discipline:** `W` and the frozen code draw from their own generators (`torch.random.fork_rng`), so
  every other initialisation of a run equals the corresponding E9 run's (regression test §14).

## 4. Arms (3 seeds each, T5, SmolLM2-360M)

| Arm | Channel | λ_w | λ_r | Config |
|---|---|---:|---:|---|
| **C5+WR** | C5 (HRR, attentive composition, 256-wide) | 0.20 | 0.035 | `configs/t5/SmolLM2-360M-full-C5WR-s{1,2,3}.yaml` |
| **C5+R** | C5 | 0 | 0.035 | `…-C5R-s{k}.yaml` |
| **C2+R** | C2 (free table, 222 wide, parameter-matched) + frozen random HRR decoder | 0 | 0.035 | `…-C2R-s{k}.yaml` |
| **C5sh+WR** | C5 with shuffled frames | 0.20 | 0.035 | `…-C5shWR-s{k}.yaml` |
| C5, C2, C5sh, C0′ | E9's T5 runs, reused | — | — | `experiments/e9-retrofit/configs/t5/…` |

Each config equals its E9 base config plus the `channel.target` block (and the `experiment` / `e15_arm` labels); the
test `test_the_twelve_configs_differ_from_their_e9_base_only_by_the_target_block` enforces this.

## 5. λ from the CPU smoke (SMOKE, not a result)

**Smoke:** SmolLM2-135M (the 360M host does not fit a CPU smoke in the time available; the loss scales it fixes are
those of the LM loss, the write cosine and the read cross-entropy, none of which depends on host size at
initialisation beyond L_LM), the T5 data, C5+WR and C2+R configs with λ_w = λ_r = 1 so the raw terms are logged,
micro-batch 2 × 512 tokens, 4 steps, `CUDA_VISIBLE_DEVICES=` (outputs
`/tmp/claude-1000/-home-bhux-workplace/71272da5-02b7-47b1-a057-5887c33fb83d/scratchpad/e15/run-{C5WR,C2R}/metrics.jsonl`).

| step | L_LM | L_write | L_read | note |
|---:|---:|---:|---:|---|
| 1 | 2.863 | 0 | 0 | no linked term in the 2 × 512 batch (T5 is 50% FineWeb-Edu) |
| 2 | 3.462 | 1.001 | 8.369 | ≈ cos 0 and ≈ chance (8.16) at initialisation, as expected |
| 3 | 2.847 | 0.923 | 8.057 | |
| 4 | 3.201 | 0.835 | 7.726 | C2+R: 1.012 / 8.402 / 7.996 / 7.664 — same scales through the frozen code |

**Rule applied:** the added terms should be 10–20% of L_LM at step 0. With L_LM ≈ 2.9–3.0 (SmolLM2-360M starts
slightly lower on T5, ≈ 2.8), L_write ≈ 1.0 and L_read ≈ 8.3:

- **λ_w = 0.20, λ_r = 0.035** → 0.20 · 1.0 + 0.035 · 8.3 = 0.49 ≈ **16–17% of L_LM** for the WR arms (write 0.20, read
  0.29: the two terms start within a factor 1.5 of each other);
- the read-only arms (C5+R, C2+R) add 0.29 ≈ **10% of L_LM**.

τ = 0.1 as given. These values are fixed here and written into the 12 configs before any run.

## 6. Held-out splits (fixed; `splits/t5-v1/splits.json`, sha256 `c9cd4af169b269ddf425f72b2dfb64bf2d6c9aa6111fc18a3c4ce1270e209c7f`)

Built by `e15_eval split` from the T5 ontology (`/home/bhux/data/vsa-llm/tracks/t5-glossary/v1/ontology.pt`, alias
table sha256 `5cae33ad…dd25415`) and E11/E14's T5 new-word set `experiments/e11-read-to-learn/items/t5-new-smollm2-v1`
(300 invented words, 3,658 items; the set RD-4b scored). "Training frames" are the 3,640 non-held-out entries' frames:
5,373 distinct (role, filler) edges over 3,350 of the 3,493 atoms.

**Does the T5 "new" set satisfy the compositional hold-out by construction?** Its construction rule (base items'
manifest) is: *"category edges of a real donor kept; every other filler resampled (frequency-weighted, same relation) so
that no existing entry has the category edge together with all resampled edges and no entry has the same frame"*. So
every frame is a **new combination** by construction (300 of 300 frames differ, as edge sets, from every training
frame), the fillers are drawn from the fillers observed under the same relation (hence almost always training atoms),
but the rule does not guarantee that every atom was trained. The check:

| set | n | every atom in a training frame | every (role, filler) edge in a training frame | frame (as a set) new | mean share of edges seen |
|---|---:|---:|---:|---:|---:|
| T5 new words (v1) | 300 | **285** | 264 | 300 | 0.985 |
| ontology held-out entries | 560 | 411 | 173 | 560 | 0.874 |

**Reading of the design's definition.** "Every role and filler atom occurs in some training term's frame but the
(role, filler) combination is new" is taken at the level of the **frame**: the frame is a new combination of trained
atoms (an edge-level reading — every single edge unseen while its atoms are seen — is impossible on T5, where the
category relations have 3–11 fillers each and every one is seen under its role). The splits are therefore:

- **(i) seen** — 300 training entries linked ≥ 10 times, a seeded sample (seed 0), read under the frame the run injects;
- **(ii) compositional** — the 285 new words whose every atom is a training filler and whose frame is new. **Primary
  split.** `compositional_strict` (264): the subset whose every edge also occurs in some training frame (pure
  recombination of seen edges) — secondary;
- **(iii) unseen_filler** — 164 terms with at least one atom no training frame holds: the other 15 new words and 149
  of the ontology's held-out entries (secondary; expected to fail for every arm);
- **heldout_ontology** — all 560 held-out entries (360 real, 200 synthetic), read under their gold frames (secondary;
  no QA items in stage 1, read accuracy only).

The QA items of the compositional split are the `t5-new-smollm2-v1` items of its 285 concepts (≈ 1,450 property,
≈ 570 entailment items; paraphrase = the property items' consistency across templates).

## 7. Evaluation

Every trained run (the 12 new ones and the 9 reused C5 / C2 / C5sh runs) gets two evaluation jobs:

- **QA, the implicit route** — `e14_reading recall --family e15 --null plain` on `t5-new-smollm2-v1` with the oracle
  frames (`SmolLM2-360M-full-C5-s1/e11-t5-new/frames.jsonl`, identical gold frames for every run): conditions `none`,
  `explicit-rel:oracle` and, for compose channels, `implicit:oracle` — the held-out word's composed vector injected and
  **no text about it** in the prompt. The free-table arms (C2, C2+R) have no vector for a new word beyond the fallback
  row, so their implicit route **is** their `none` condition (recorded as `implicit_condition` in the report). Tests:
  property, paraphrase, entailment; PMI against the plain null (RD-4b's protocol). Output
  `qa/t5/SmolLM2-360M-full-<arm>-s<k>/`.
- **Read accuracy** — `e15_eval read`: for each split the term alone after a neutral prefix (`Note: {x}`; new words
  inserted zero-shot with their gold frame as E14 does, held-out entries read their gold frame, seen entries read the
  injected frame), the hidden state at the term's last subtoken at layer 11, `W h` unbound by each role and cleaned up
  by top-1 cosine over all 3,493 atoms. Reported per split: edge accuracy, per-concept accuracy with a bootstrap
  interval, MRR, typed top-1 (atoms observed under the role; diagnostic), and a **post-hoc ridge probe** fitted on the
  seen split's states (in-sample on `seen`; out-of-sample on the other splits) — the only read accuracy the baselines
  without `W` can have, and a check of whether the structure sits in `h` without the losses. Output
  `read/t5/SmolLM2-360M-full-<arm>-s<k>/read.json`.
- **Loss endpoints** from the runs' standard evaluation (`eval_windows.npz`, final point): relative change against C5
  on `after_heldout` (loss after held-out terms), `after`, `unlinked` (general and unlinked text) and `all`, paired by
  evaluation window, summed over seeds, cluster bootstrap over windows (the E9 convention).

**Statistics:** item differences paired within seed (arm A's seed k against arm B's seed k), pooled over the three
seeds, percentile bootstrap with 4,000 resamples, 95% intervals, two-sided bootstrap p; Holm within the primary family.

## 8. Endpoints

**Primary (Holm over the three):**

1. **P1** — implicit property on the compositional split, **C5+WR − C5**.
2. **P2** — implicit property on the compositional split, **C5+WR − C2+R** (C2+R's implicit route = its `none`).
3. **P3** — read accuracy on the compositional split for **C5+WR**: against chance 1/|A| = 0.00029 (reported with its
   interval) and **C5+WR − C2+R** paired by concept and seed (the Holm-adjusted test).

**Secondary (unadjusted):** S1 C5sh+WR − C5+WR (implicit property, compositional); S2 C5+R − C5+WR (same); S3 paraphrase
and entailment versions of P1; S4 P1 on `compositional_strict` and on the 15 unseen-filler new words; S5 explicit recall
unchanged: `explicit-rel:oracle` property, C5+WR − C5; S6 `none` property, C5+WR − C5 and C2+R − C2 (does the training
change the no-information baseline?); S7 loss after held-out terms, C5+WR − C5 (relative); S8 general-text loss cost
(`unlinked`, `all`), each arm − C5; S9 read accuracy on seen terms, on `heldout_ontology` and on `unseen_filler`, for
every arm, trained `W` and the probe; S10 C5+WR − C5sh (implicit property).

## 9. Predictions (as given by the author; none changed by the smoke)

- T5 read accuracy on the compositional split ≥ **0.85** for C5+WR (chance 0.0003).
- Implicit property on the compositional split rises from C5's ≈ 0.19–0.26 (RD-4b: 0.26 with the vector, 0.19 without)
  to ≥ **0.55** for C5+WR.
- C2+R stays ≤ **0.30** on the compositional split (implicit = `none`).
- C5sh+WR below C5+WR by ≥ **0.2** on implicit property.
- Loss after the term (after held-out) within **±0.3%** of C5.
- General-text loss cost ≤ **0.2%** (`unlinked`).
- Read-only C5+R: between C5 and C5+WR on both reading and use (no number given; descriptive).
- Unseen-filler split: read accuracy near chance for every arm (the atom was never a filler; nothing to recover).

## 10. Decision rule (stage 1)

- **P1 ≥ +0.15 and P3 (C5+WR read accuracy) ≥ 0.80** → *the algebra is learnable with the right signal*: stage 2 (T4,
  §11) is queued.
- **P3 ≥ 0.80 but P1 < +0.15** → *"decodes but does not use"* (a reading–use gap): the host can recover the facts from
  the vector but the later layers do not act on them. Reported as such; stage 2 is not queued without the author.
- **P3 < 0.60** → *the algebra is not learnable by this host at this scale*; the implicit route is closed and the
  manuscript's statement ("the model never reads the structure") stands with this stronger test behind it.
- 0.60 ≤ P3 < 0.80: no pre-registered reading; reported as inconclusive with the numbers.

The report (`e15_eval report`, `report/t5/report.md`) prints the reading from these thresholds.

## 11. Stage 2 (T4), registered now, **not queued**

Gated on stage 1's first reading. Host SmolLM2-360M, E9's T4 recipe (`experiments/e9-retrofit/configs/t4/
SmolLM2-360M-full-{C5,C2}-s{1,2,3}.yaml` as bases, 100M tokens), arms **C5+WR** and **C2+R** (3 seeds each; λ as in §5,
re-checked by one CPU smoke of the T4 loss scales before queueing), reused baselines C5, C2 (and C0′). Evaluation: the
E11 `t4-heldout-smollm2-v1` items (real ChEBI held-out entries, style `chebi`; RD-4b's T4 group, where C5's
`implicit:oracle` property is 0.615 and `none` 0.614), the same `--family e15 --null plain` QA and `e15_eval read` on a
T4 split file built the same way (seen, compositional = held-out entries whose atoms are all training fillers, unseen
filler). **Primary:** implicit property on T4's compositional held-out, C5+WR − C5, and read accuracy of C5+WR.
**Prediction:** 0.61 → ≥ 0.70. **Rule:** C5+WR − C5 ≥ +0.05 (Holm) with read accuracy ≥ 0.80 → the implicit route
works on real text and the manuscript's implication 1 is revised; otherwise the T5 result is a synthetic-only
capability and is reported as such.

## 12. Compute

Measured: every T7-ROOD SmolLM2-360M training job (same host, 50M tokens, `.jobs/t7rood-SmolLM2-360M-full-*.json`
started → finished) took **1.24–1.26 GPU-h**; E14's RD-4b T5 new-word QA jobs 0.13–0.16 GPU-h each.

- 12 training jobs × 1.25 h ≈ **15 GPU-h**;
- 21 QA jobs (7 arms × 3 seeds) × 0.15 h ≈ 3.2 GPU-h;
- 21 read-accuracy jobs × ≈ 0.1 h ≈ 2 GPU-h (five splits, 1,469 prompts of a few tokens each, one ridge fit);
- report: CPU lane.

**Total ≈ 20 GPU-h**, queued at 54.4969 (training), 54.49692 (evaluations) and 54.49695 (report), ahead of the pending
T7-ROOD block at 54.497 (`queue-commands-t5.sh`). The runner's lower-first order runs the training jobs before the
evaluations that read them.

## 13. Limitations

- One host (360M), one synthetic track in stage 1. T5's text is generated from the ontology the method reads, so a
  positive result is an upper bound; stage 2 exists for this reason.
- The losses read one layer (11 of 32), chosen from 09.4, not tuned. A model could carry the algebra at another depth.
- λ is set from loss scales at initialisation on a 135M smoke, not by a sweep. If the read loss collapses to chance
  while the write loss falls, the report says so (the train rows log `lm_loss`, `write_loss`, `read_loss`).
- The read loss is over all atoms; the typed cleanup (atoms observed under the role) is reported only as a diagnostic.
- "Compositional" is a frame-level notion on T5 (§6); 264 of the 285 frames are pure recombinations of seen edges, 21
  contain an edge whose atoms were both seen but never together under that role — the strict subset is a secondary.
- C2+R decodes a frozen random code rather than the trained store; this is the fair control for "structure in the
  input", not for "a trainable store". A C2 variant with a trainable decoder is not run.
- The post-hoc probe is in-sample on the seen split (300 rows, 960 features): its seen-split number says little; it is
  read out of sample on the other splits.
- The implicit-route QA uses the RD-4b protocol (`--max-length 256`, plain null); the prompts here are short, so the
  truncation correction of 2026-10-21 does not apply (the report records `truncated_share`).
- Seen-term QA has no item set (E11's T5 sets are new words or held-out entries); seen-term behaviour is read through
  read accuracy and the loss endpoints only.

## 14. Files, code and tests

- Code (additive; no behaviour change for configs without `channel.target`): `src/vsa_embed/hrr_target.py` (the
  losses, the frozen code, `target_from_config`), `src/vsa_embed/integrations/transformers.py` (`_capture_target` hook,
  spans handed to the target), `src/vsa_embed/training/lm.py` (`channel.target` built in `build_channel`; the loss
  terms and the `lm_loss` / `write_loss` / `read_loss` columns of train rows), `src/vsa_embed/experiments/e14_reading.py`
  (`--family e15`), `src/vsa_embed/experiments/e15_eval.py` (`split`, `read`, `report`).
- Tests `tests/test_e15.py` (10 tests, CPU, log `…/scratchpad/e15/pytest-e15.log`): loss shapes and gradient flow (W and
  the host receive gradient, the detached store does not, flags restored); a perfect `W` reads every planted filler at
  accuracy 1.0 and MRR 1.0 while a random map reads at chance; the frozen code is seeded and frozen; the compositional
  split check on a toy ontology; the committed T5 splits match their hash and sizes; the 12 configs differ from their E9
  bases only by the target block; training with the target (compose and free) logs the losses, saves `W` (and the
  code) and reloads; **regression:** a config without the block builds no target, has byte-identical parameters, the
  same LM loss on the same batch, and no extra metric columns; read accuracy runs end to end on a toy run with the
  probe.
- Smokes (SMOKE-labelled, scratchpad `e15/`): §5's loss scales; `e15_eval read --limit 8 --smoke` and
  `e14_reading recall --family e15 --limit 3 --smoke` on the 135M smoke run (pipeline only; their numbers mean nothing).
- Queue: `experiments/e15-hrr-target/queue-commands-t5.sh`; job names `e15-t5-SmolLM2-360M-full-<arm>-s<k>`
  (`-qa`, `-read`), report `e15-t5-report`.

## 15. Changes after the first commit

- **2026-10-22 (evening), stage 1b registered** (§16 below), at the author's approval *"Register and queue 1b"*, after
  5 of the 12 stage-1 training runs had finished and before any stage-1 evaluation job had run. Reason: a
  SMOKE-level preliminary read (CPU, 80 terms per split, seed 1, scratchpad `e15-prelim/`; not a result) showed the
  read loss still falling at the end of the 50M tokens (7.7 → 3.6–3.8) while the write loss had converged (0.83 →
  0.02), compositional read accuracy 0.47 (C5+WR) / 0.38 (C5+R), unseen-filler 0.53, loss after held-out ≈ −2% vs
  C5. The author's reading: the 0.8 read bar of §10 was set from first principles and the objective is undertrained.
  **Nothing in §1–§14 changes:** stage 1 runs and is read exactly as registered; stage 1b adds length. Code added for
  it is opt-in (`train.save_at_tokens`, `e15_eval report --stage 1b`); the stage-1 configs, runs, jobs and rule are
  untouched (the stage-1 config test now selects the 12 stage-1 files by name, since the five 1b configs sit in the
  same folder).
- **2026-10-23, stage 1b: the matched-length loss control `C5-long` added** (§16.1 arm (d); queued by
  `queue-commands-t5-stage1c.sh`), at the author's request ("Queue these two"), **before any stage-1b evaluation job had
  run** (four of the five 1b training runs had finished, the fifth, lr0.10, was running). Reason: §16.3's loss endpoints
  compare 200M-token arms with E9's 50M-token C5, so a loss change confounds the objective with the tokens. `C5-long` is
  E9's T5 C5 seed-1 config with `train.total_tokens: 200000000` and `train.save_at_tokens [50M, 100M, 150M, 200M]` and
  nothing else (`configs/t5/SmolLM2-360M-full-C5-long-s1.yaml`; test
  `test_the_c5_long_control_config_differs_from_e9_c5_only_by_length_and_checkpoints`). It gets §16.2's read accuracy
  (probe only: it trains no `W`) and implicit-route QA at 200M. `report --stage 1b` now shows every loss endpoint against
  `C5-long` as well as against C5 (the three C5WR-long seeds each paired with the one control run; `C5-long − C5` is
  shown too, as the effect of length alone), plus two secondaries at 200M (seed 1): C5WR-long − C5-long on implicit
  property and C5WR-long's trained-`W` read accuracy against C5-long's probe. **Nothing else in §16 changes**: the
  endpoints, the decision rule and the predictions are read as registered; the control only sharpens S4. The stage-1b
  config test now skips the `C5-long` file (it is not one of the five 1b arms).
- **2026-10-23, stage 1c registered** (§17 below), at the author's approval of the same request, **before any run of
  it**, and before any stage-1 or stage-1b QA evaluation of the long runs had been read. Reason: the stage-1 report
  (`report/t5/report.md`): C5+WR reads 0.48 of a held-out word's facts from the vector while implicit QA on the same
  words rose only +0.030 (0.264 → 0.294) against an explicit-recall ceiling of 0.82, and a decodable-but-shuffled store
  (C5sh+WR, 0.45 read) gives no gain — the read loss (atom space, external head) is not aligned with the test (a
  token-space judgement by the LM head). Code added for it is opt-in (`channel.distill`, `channel.verbalize`,
  `model.max_positions`, `report --stage 1c`); stage-1 and 1b configs, runs, jobs and rules are untouched.

## 16. Stage 1b (registered 2026-10-22, before any run of it)

**Question.** Is the read objective of §3 *undertrained* at 50M tokens — does compositional read accuracy keep rising
with length, and does the implicit route open once it does — or is it *flat*, so that the stage-1 reading stands?

### 16.1 Arms (T5, SmolLM2-360M, E9's T5 recipe; everything not named here identical to stage 1)

| Arm | Base (stage-1 config) | `train.total_tokens` | λ_w | λ_r | Seeds | Config |
|---|---|---:|---:|---:|---|---|
| **(a) C5WR-long** | `SmolLM2-360M-full-C5WR-s{k}` | 200M (4× stage 1) | 0.20 | 0.035 | 1–3 | `configs/t5/SmolLM2-360M-full-C5WR-long-s{1,2,3}.yaml` |
| **(b) C2R-long** | `…-C2R-s1` | 200M | 0 | 0.035 | 1 | `…-C2R-long-s1.yaml` — the memorisation control at matched length |
| **(c) C5WR-long-lr0.10** | `…-C5WR-s1` | 200M | 0.20 | **0.10** | 1 | `…-C5WR-long-lr0.10-s1.yaml` — exploratory: weight against length |

Each config equals its stage-1 base plus `train.total_tokens: 200000000`, `train.save_at_tokens: [50M, 100M, 150M,
200M]` and, for (c) only, `read_weight: 0.10` (the labels `experiment`, `e15_arm`, `e15_stage`, `e15_base_config`
aside); the test `test_the_five_long_configs_differ_from_their_stage1_base_only_in_length_checkpoints_and_lambda_r`
enforces this. Same τ = 0.1, layer 11, detached target, data (`data.seed` 1234), run seeds, micro-batch 4 × 16 ×
1,024, lr 1e-3 / host lr 3e-5, 1,024 evaluation windows, and the same splits file `splits/t5-v1/splits.json`
(sha256 `c9cd4af1…209c7f`).

**Schedule at 200M.** The trainer's step is 65,536 tokens: 200M → 3,051 steps (199,950,336 tokens; stage 1: 762 steps,
49,938,432 tokens). `train.warmup_tokens` stays at E9's 2,500,000 (38 steps); the cosine to `min_lr_ratio` 0.1 then
stretches over the remaining 3,013 steps instead of 724. So at the 50M checkpoint the long run's learning rate is still
0.88 × peak (stage 1 ended there at the floor, 0.1 × peak): the 50M checkpoint of (a) and the stage-1 C5WR runs have
the same tokens but a different schedule (secondary S7). The run's own loss evaluations fall at the log-spaced points
2.5, 5, 10, 20, 40, 80, 160M and the end; the loss endpoints below use the end (200M).

### 16.2 Checkpoints and evaluation

- **Named checkpoints** (`train.save_at_tokens`, opt-in, added for this stage): at the first optimizer step whose
  token count reaches 50M, 100M, 150M the run writes `checkpoint-<tokens>.pt` in `final.pt`'s format (steps 763, 1,526,
  2,289 — 50,003,968 / 100,007,936 / 150,011,904 tokens; the file records the actual step and tokens); the 200M entry
  is never reached before the last step (199,950,336 < 200,000,000) and is written at the end as a hard link to
  `final.pt`, so `checkpoint-200000000.pt` **is** the final state. Nothing changes for a config without the key
  (regression test).
- **Read accuracy** — `e15_eval read --checkpoint checkpoint-<tokens>.pt` on the full splits, exactly §7's procedure,
  at **every checkpoint for the three C5WR-long seeds** and at 200M for (b) and (c). Output
  `read/t5/SmolLM2-360M-full-<arm>-s<k>/ckpt-<tokens>/read.json`.
- **Implicit-route QA** — `e14_reading recall --family e15 --null plain --checkpoint checkpoint-<tokens>.pt`, §7's
  protocol (same items, same oracle frames, conditions `none`, `implicit:oracle`, `explicit-rel:oracle`), at every
  checkpoint for the three C5WR-long seeds and at 200M for (b) (whose implicit route is its `none`, as in §7). Output
  `qa/t5/SmolLM2-360M-full-<arm>-s<k>/ckpt-<tokens>/`. **The curve is the result.**
- **Loss endpoints** from `eval_windows.npz` at the end of each run, paired by window as in §7.
- Baselines: stage 1's C5 QA (`qa/t5/SmolLM2-360M-full-C5-s{k}`, the stage-1 evaluation jobs) and the stage-1 C5WR
  read outputs; nothing is re-run.

Statistics as §7 (items paired within seed, pooled over the three seeds, percentile bootstrap, 4,000 resamples, 95%
intervals, two-sided bootstrap p; Holm within the primary family). Report: `e15_eval report --stage 1b`
(`report/t5-stage1b/report.md`).

### 16.3 Endpoints

**Primary (Holm over the three):**

1. **P1** — implicit property on the compositional split, **C5WR-long − C5**, at the **best checkpoint**, where "best"
   is **chosen by compositional read accuracy** (the checkpoint with the highest per-concept mean pooled over the three
   seeds) — **never by QA**, so the choice cannot select on the outcome. The QA contrast at every checkpoint is also
   reported, as the curve.
2. **P2** — compositional read accuracy of **C5WR-long at 200M** (per-concept mean over the three seeds, against chance
   1/3,493, with its interval), read against the thresholds of §16.4.
3. **P3** — the **slope** of compositional read accuracy over the four checkpoints: per concept and seed the
   least-squares slope of accuracy against tokens (in accuracy per 50M tokens; paired across checkpoints), bootstrap
   over concepts; **and** the **150M → 200M increment**, paired per concept, reported with its interval and read as
   "still rising" when the interval excludes 0 (**the increment, not the slope, drives the decision rule**).

**Secondary (unadjusted):** S1 unseen-filler and strict-compositional read accuracy at every checkpoint; S2 C2R-long at
200M against C5WR-long at 200M (read accuracy paired per concept with seed 1; implicit property, C2R-long's `none`
against C5WR-long's `implicit:oracle`); S3 lr0.10 against seed 1 of (a) on compositional read accuracy at 200M (paired
per concept) and on LM loss (`unlinked`, `all`: the cost); S4 loss after held-out terms, after seen terms (`after`) and
general text (`unlinked`, `all`), C5WR-long against C5 and against the stage-1 C5WR runs (relative, paired windows);
S5 the QA means (`none`, `implicit:oracle`, `explicit-rel:oracle`) at every checkpoint; S6 paraphrase and entailment
versions of P1's curve; S7 the 50M checkpoint of (a) against the stage-1 C5WR runs on compositional read accuracy
(same tokens, different schedule — a consistency check, paired per concept and seed).

### 16.4 Decision rule (stage 1b)

- **QA gate:** P1's contrast (implicit property, compositional, C5WR-long − C5) **≥ +0.15 at any checkpoint** → stage 2
  (T4, §11, already registered) is queued.
- **Read ≥ 0.80 at 200M** → the algebra is learnable to the original standard (§10's bar).
- **0.60 ≤ read < 0.80 at 200M and the 150M → 200M increment > 0 with an interval excluding 0** → *still rising*:
  register a **stage 1c at 500M tokens** before any verdict on the implicit route.
- **Read < 0.60 and flat** (the increment's interval includes 0) → the implicit route is closed; the manuscript's
  statement stands with this stronger test behind it.
- Any other combination (e.g. below 0.60 but still rising; 0.60–0.80 and flat) has no pre-registered reading and is
  reported as such with the numbers.

The QA gate and the read gates are read independently (both can hold). The report prints the reading.

### 16.5 Predictions (the author's, written before any run)

- Compositional read accuracy **0.6–0.7 at 200M and still rising** (increment > 0).
- The unseen-filler split **keeps pace** with the compositional split (the mechanism is vector preservation to layer 11
  plus linear unbinding, not memorised edges).
- The QA gain (implicit property, C5WR-long − C5) **crosses +0.15 by 100M** tokens.
- **C2R-long ≤ 0.30** on compositional held-out read accuracy at 200M.
- **lr0.10 reads higher** than seed 1 of (a) by **≥ 0.05** at 200M but **costs ≥ 0.1%** general-text loss (`unlinked`).
- Loss after held-out terms stays at about **−2%** against C5.

### 16.6 Compute, queue and disk

- Author's estimate: 1.25 GPU-h per 50M tokens at 360M → (a) 3 × 5 h + (b) 5 h + (c) 5 h = 25 GPU-h training;
  evaluations 12 read (≈ 0.1 h) + 12 QA (≈ 0.15 h) for (a) + 2 read + 1 QA ≈ 3.5 h; **≈ 29 GPU-h**.
- Measured on the stage-1 runs that have finished (`.jobs/e15-t5-SmolLM2-360M-full-C5WR-s{1,2}.json`,
  `…-C5R-s1.json`): 68 min each for 50M tokens including seven loss evaluations → **1.13 GPU-h per 50M**. A 200M run
  then takes ≈ 4.5 h (four blocks plus eight evaluations and three 1.45 GB checkpoint writes) → 5 runs ≈ **22.5 GPU-h**;
  27 evaluation jobs (14 read × ≈ 0.1 h + 13 QA × ≈ 0.15 h) ≈ 3.4 GPU-h; **≈ 26 GPU-h in all**, the author's 29 the
  upper bound.
- **Queue** (`queue-commands-t5-stage1b.sh`): the five training jobs `e15-t5-SmolLM2-360M-full-<arm>-s<k>` at
  **54.49696** (after stage 1's report at 54.49695, before T7-ROOD at 54.497; `--min-free-gb 20` and the default
  `--resume` retry as stage 1's training jobs), the 27 evaluation jobs `…-ckpt<tokens>-{read,qa}` at **54.49697**
  (`--no-resume`), the CPU-lane report `e15-t5-stage1b-report` at **54.49698**. **Ordering guarantee:** the queue
  orders jobs by `(priority, created)`, a runner takes the first pending job of its lane only when no job of that lane
  is live, and a job blocked on disk stays first in line; so every 54.49696 training job has finished (or failed)
  before any 54.49697 evaluation starts, and the CPU lane starts the report only once every job of lower priority is
  done. The resume convention makes a restarted training job continue from `checkpoint.pt`; a named checkpoint already
  on disk is never rewritten.
- **Disk:** each named checkpoint is 1.45 GB (the 200M one is a hard link to `final.pt`): 5 runs × 3 ≈ 22 GB plus 5 ×
  1.45 GB of `final.pt` ≈ 29 GB in all; the time-based `checkpoint.pt` (≈ 4.4 GB, deleted at the end) is transient.
  At registration the disk had 62 GB free and stage 1 still had six `final.pt` (≈ 9 GB) to write; the training jobs'
  `--min-free-gb 20` guard blocks a job (it stays first in line) when free space falls below 20 GB — the author may
  need to free space before the last 1b runs.
- **Addendum 2026-10-23 (§15):** arm **(d) `C5-long`** — E9's C5 seed 1 at 200M tokens, the matched-length loss control
  (`…-C5-long-s1.yaml`), queued at 54.49696 with its 200M read and QA at 54.49697 by `queue-commands-t5-stage1c.sh`,
  so the 1b report (54.49698) can use it; ≈ 4.5 GPU-h training + 3 × 1.45 GB checkpoints.

### 16.7 Files

Configs (five, §16.1); `queue-commands-t5-stage1b.sh`; code: `src/vsa_embed/training/lm.py` (`train.save_at_tokens`,
`_save_named_checkpoint`, `_link_or_copy`), `src/vsa_embed/experiments/e15_eval.py` (`report --stage 1b`:
`run_report_1b`, `per_item_slopes`, `read_curve`, `render_report_1b`); `--checkpoint` already existed on `e15_eval
read` and `e14_reading recall` and is used unchanged. Tests `tests/test_e15_stage1b.py` (three tests: the configs,
the named checkpoints end to end on the toy world with the regression that an absent key changes nothing, the stage-1b
report on synthetic folders with the three decision readings); log `…/scratchpad/e15-1b/pytest-e15-1b.log`. CPU
smoke (SMOKE-labelled, `…/scratchpad/e15-1b/`): the 135M smoke config with `save_at_tokens` writes the named files and
`e15_eval read` / `e14_reading recall --family e15` load one through `--checkpoint`.

## 17. Stage 1c — aligned objectives (registered 2026-10-23, before any run of it)

**Question.** Stage 1 separated *decoding* from *using*: a model trained to read the HRR algebra of a composed vector
recovers the facts (0.48 compositional read accuracy at 50M tokens, prelim 0.72 at 200M) but does not act on them in a
question (+0.030 implicit property against an explicit-recall ceiling of 0.82). The stage-1 read loss lives in atom
space behind an external head `W`; the test is a token-space judgement by the LM head. Does an objective **aligned with
the test** — one that asks the model, in token space, to behave as if it had read the facts — open the implicit route
on words it has never seen in text? And does it do so by *using the structure* (a free table under the same objective
can only memorise)?

### 17.1 The two objectives (`src/vsa_embed/aligned_targets.py`; opt-in config blocks read by `training.lm`)

**(1) Route distillation, `channel.distill` (arms C5RD, C2RD).** For every linked term occurrence in a micro-batch, a
*teacher* forward pass of the **same model** — current parameters, `torch.no_grad()`, **no EMA** (the teacher's
advantage is its context, not its weights; an EMA copy would double the state in every checkpoint and add a decay
constant to tune) — sees the term's **gold frame rendered as the E14 explicit-recall lines** exactly as `explicit-rel`
renders them (`self_query.RecallWriter` through the T5 track lexicon, confidence shown: `recall(<term>):` then
`- <term> is owned by the Rolk Desk. (1.00)` …), **prepended to the window** that contains the term. This is the
simplest faithful variant of "before the segment containing the term": the window *is* the segment; attention is
causal, so every position after the term sees the lines. One recall block per distinct entry of the window, in order
of first occurrence, at most `max_entries` = 8 blocks and `max_prefix_tokens` = 384 tokens (a block that would exceed
the cap is dropped; the smoke's prefixes were 253–377 tokens); the vector is injected as in C5 at the shifted span
positions (every span of the window, prefixed or not). The *student* pass is the ordinary C5 pass on the trainer's
`ids`, vector only. Loss: `L_LM + λ_rd · KL(teacher ‖ student)` on the next-token distributions over the **K = 8
positions after each prefixed term** (the trainer's `after` stratum: positions e..e+7 predict tokens e+1..e+8; positions
of overlapping occurrences counted once), temperature 1, averaged over the positions of the micro-batch (zero when no
term is linked). Occurrences whose entry did not fit in the prefix are not distilled. **No leak:** the teacher's input
is a separate tensor assembled in `AlignedObjectives.teacher_batch`; the student forward receives the trainer's `ids`
unchanged, and the test decodes both (`recall(` and `(1.00)` absent from the student's text, the teacher's input equal
to prefix + window, spans shifted by the prefix length). The teacher runs under `torch.random.fork_rng`, so the
student's RNG stream equals a run without the block. The teacher's recall is the **ontology's gold frame** (for C2RD as
for C5RD; a C5shRD variant would also read gold text — the teacher knows the truth, the student's vector may not).
Cost: one extra no-gradient forward of (1024 + ≤ 384) tokens per micro-batch that holds a linked term (≈ 60% of T5's
micro-batches).

**(2) Fact verbalisation, `channel.verbalize` (arms C5FV, C2FV).** An auxiliary cloze loss in token space. For each
linked occurrence, prompts of the form `"<term>: <field label>:" → " <filler name>"`: the term's **own tokens from the
window** (no re-encoding), the suffix of a template drawn from `FV_TEMPLATES` (below), and the filler's name as the
run's lexicon writes it (`RecallWriter.atom_text`: "the Rolk Desk", "plans headcount", "finance") as the **target
tokens** — cross-entropy over the target tokens only, with the vector injected at the term's last subtoken (the
prompt's spans: start 0, inject = end = last term token). One prompt per fact of **the frame the run injects** (the
composer's schedule; for a free table, the ontology's frame), **capped at 6 per occurrence** (a seeded subset of a
larger frame) and at **64 prompts per micro-batch** (a seeded subset; deterministic per run seed, step and micro-step),
scored as separate right-padded short sequences in a second forward of the same micro-batch.
`L = L_LM + λ_fv · L_cloze`.

*Templates and their disjointness from the E14 items.* `FV_TEMPLATES` holds two field-style wordings per T5 relation
(23 relations, 46 wordings: `{x}: owning team:`, `{x}: accountable custodian:`, `{x}: upstream dependency:`,
`{x}: health indicator:`, `{x}: governing policy:`, …), every one with at least two content words. The check follows
RD-4d/RD-4e's wording hold-out: no FV wording equals, or is a `near_duplicate` of (content-word sets equal, or one
inside the other with one word of difference), **any question template of any T5 item set** (property, statement and
entailment templates of `experiments/e9-retrofit/items/*t5*` and `experiments/e11-read-to-learn/items/t5-*` — 52
distinct wordings including the T5 lexicon's three prompts per relation) **nor any RD-4e router wording**
(`ROUTE_PARAPHRASES`, a stricter extra). Test `test_fv_templates_are_disjoint_from_the_e14_qa_item_templates`.
Wordings that failed the check while drafting and were replaced: `{x}: owned by:`, `{x}: depends on:`, `{x}: output:`,
`{x}: status:`, `{x}: run frequency:` ("{x} is run by"), `{x}: metric:`, `{x}: owner:`, `{x}: function:`. The cloze
format ("term: label: filler") is also unlike the E14 question form ("<template with the term> <candidate>").

### 17.2 Arms (T5, SmolLM2-360M, E9's T5 recipe: 50M tokens, everything as stage 1 §3)

| Arm | Base (E9 config) | Block | λ | Seeds | Config |
|---|---|---|---:|---|---|
| **C5RD** | `SmolLM2-360M-full-C5-s{k}` | `distill` {k 8, max_entries 8, max_prefix_tokens 384} | λ_rd 1.0 | 1–3 | `configs/t5/SmolLM2-360M-full-C5RD-s{1,2,3}.yaml` |
| **C5FV** | `…-C5-s{k}` | `verbalize` {max_facts 6, max_prompts 64} | λ_fv 0.05 | 1–3 | `…-C5FV-s{1,2,3}.yaml` |
| **C2RD** | `…-C2-s1` (free table, 222 wide) | `distill` | 1.0 | 1 | `…-C2RD-s1.yaml` — memorisation control |
| **C2FV** | `…-C2-s1` | `verbalize` | 0.05 | 1 | `…-C2FV-s1.yaml` — memorisation control |
| C5, C2, C5WR | E9's T5 runs; stage 1's C5WR QA | — | — | — | reused |

Each config equals its E9 base plus the one block (and the labels `experiment`, `e15_arm`, `e15_base_arm`,
`e15_stage: 1c`, `e15_base_config`); the test `test_the_eight_stage1c_configs_differ_from_their_e9_base_only_by_the_block`
enforces this, and that one λ_rd and one λ_fv serve every arm and seed. No `channel.target`: these arms train no `W`.
The free-table controls can memorise a seen term's facts under either objective but have no vector for a held-out
word beyond the fallback row, so they cannot generalise to the compositional split by construction. *Optional, not
queued:* `C5shRD` seed 1 (shuffled frames + distillation; needs the author's go-ahead and ≈ 1.7 GPU-h).

### 17.3 λ from the CPU smoke (SMOKE, not a result; scratchpad `e15-1c/`, `README-SMOKE.md`, `init-scales.log`)

SmolLM2-135M, T5 data, micro-batch 2 × 512, `CUDA_VISIBLE_DEVICES=`. Two checks: (i) four optimizer steps of each of
C5, C5RD, C5FV, C2RD, C2FV with λ = 1 (the raw terms logged; pipeline, resume-free); (ii) the terms at initialisation
over 12 micro-batches (data seed 1234, steps 0–11; 7 of them hold a linked term — T5 is 47% domain text):

| term | value at initialisation | note |
|---|---:|---|
| L_LM | 3.08 (mean of 12; 2.69–3.46) | 360M starts ≈ 2.8 on T5 |
| KL(teacher ‖ student), 8 positions after each prefixed term | **0.66 mean / 0.24 median** over the 7 linked micro-batches (0.13–2.36); 0.38 averaged over all 12 (the trainer's logged convention) | heavy-tailed: large when the window continues with the fact sentences the recall lines name |
| L_cloze (filler name after "term: label:") | **8.78** (8.53–8.94) | chance ln 49,152 = 10.8 |
| prefix | 253–377 tokens (cap 384), 64 prompts per linked micro-batch (the cap) | |

**Rule applied (stage 1's):** the added term ≈ 10–20% of L_LM at step 0, read on the linked micro-batches as §5 did:

- **λ_rd = 1.0** → 0.66 nats ≈ 21% of L_LM on the linked-micro-batch mean (8% at its median; 12% averaged over all
  micro-batches as the train rows will show it). 0.75 would give 16% / 6% / 9%; 1.0 is chosen so the median micro-batch
  still carries a visible signal, accepting the upper edge of the band on the mean.
- **λ_fv = 0.05** → 0.44 nats ≈ 14% of L_LM (8% over all micro-batches).

Cost measured on the smoke (CPU, per step): C5RD 1.45× C5 (teacher forward 1.0–1.4 s against a 0.75 s student
forward), C5FV 1.75× at 64 prompts per 1,024 tokens — on the real micro-batch (4 × 1,024 tokens, the same 64-prompt cap)
≈ 1.3×. Peak RSS 5.7–7.6 GB.

### 17.4 Evaluation (as stage 1, §7)

Every stage-1c run gets the two evaluation jobs of §7 unchanged: **implicit-route QA** (`e14_reading recall --family
e15 --null plain`, conditions `none` / `implicit:oracle` / `explicit-rel:oracle`; the free-table arms' implicit route
is their `none`) and **read accuracy** (`e15_eval read`) — for these arms **only the post-hoc ridge probe applies**
(no trained `W`; `trained_map: false` in `read.json`), reported for every split and compared with C5's probe (0.158 on
the compositional split in stage 1). **Loss endpoints** from `eval_windows.npz` against C5 as §7. In addition the
report reads the runs' train rows: the **teacher–student KL** (`kl_loss`; `cloze_loss` for FV) at the end of training
relative to its start (mean of the last 5% of train rows over the mean of the first 5%). Statistics as §7; Holm within
the primary family. Report: `e15_eval report --stage 1c` (`report/t5-stage1c/report.md`), tests on synthetic folders.

### 17.5 Endpoints

**Primary (Holm over the three), implicit property on the compositional split:**

1. **P1** — **C5RD − C5** (3 seeds, items paired within seed).
2. **P2** — **C5FV − C5** (3 seeds).
3. **P3** — **C5RD − C2RD** (structure against memorisation; seed 1 against seed 1, C2RD's implicit route = its `none`).

**Secondary (unadjusted):** S1 paraphrase and entailment versions of P1 and P2; S2 C5RD − C5WR (stage 1's QA) and
C5FV − C5WR; S3 C5RD − C5FV; S4 C5FV − C2FV (seed 1); S5 explicit recall unchanged? (`explicit-rel:oracle` property,
C5RD − C5 and C5FV − C5); S6 `none` property, each arm against its base (C5RD − C5, C5FV − C5, C2RD − C2, C2FV − C2);
S7 P1 and P2 on `compositional_strict` and on the 15 unseen-filler new words; S8 loss after held-out terms, after seen
terms (`after`), general text (`unlinked`, `all`), each arm against C5 (relative, paired windows); S9 probe read
accuracy on every split, C5RD − C5 and C5FV − C5 paired per concept and seed; S10 the end-of-training KL (and cloze)
relative to its start — did the student converge to the teacher?

### 17.6 Decision rule (stage 1c)

Read on the **level** of C5RD's implicit property on the compositional split (3 seeds pooled) and on P3:

- **≥ 0.50** (i.e. ≥ +0.24 over C5's 0.264) **and P3 ≥ +0.15** → *the model learns to USE composed facts when the
  objective asks for it*; stage 2 (T4, §11) is queued **with the RD objective**.
- **0.35–0.50** → *partial*: reported with the gap to the teacher (explicit recall, 0.82).
- **< 0.35** → *aligned objectives do not close the gap at this scale*: the implicit route is closed, and the
  manuscript keeps the loss-domain result (C5+WR −1.9% vs C5sh+WR +8.7% after held-out terms) as the structural-use
  finding.
- ≥ 0.50 with P3 < +0.15 has no pre-registered reading (memorisation not excluded) and is reported with the numbers.

The report prints the reading from these thresholds.

### 17.7 Predictions (the author's, written before any run)

- C5RD **0.50–0.60** implicit property on the compositional split (it cannot exceed the teacher's 0.82).
- C5FV **0.40–0.50**.
- C2RD and C2FV **≤ 0.30** (implicit = `none`).
- Loss after held-out terms: **C5RD ≤ C5** (the teacher's distribution is better there).
- Explicit recall (`explicit-rel:oracle`) **unchanged within 0.02**.
- The end-of-training KL falls to **< 20% of its initial value**.

### 17.8 Compute, queue and disk

- Measured: a stage-1 50M-token SmolLM2-360M job took 68 min (1.13 GPU-h; `.jobs/e15-t5-SmolLM2-360M-full-C5WR-s1.json`);
  QA 1.8 min and read 6 s per run (`…-C5-s1-{qa,read}.json`). With the smoke's ratios (RD ≈ 1.45×, FV ≈ 1.3× on the real
  micro-batch): C5RD 3 × 1.65 h + C5FV 3 × 1.5 h + C2RD 1.65 h + C2FV 1.5 h ≈ **12.6 GPU-h training** (the author's
  estimate ≈ 15 the upper bound), 16 evaluations ≈ 0.3 h; the C5-long control of §15 adds ≈ 4.5 h + 0.05 h. **≈ 17.5
  GPU-h in all.**
- **Queue** (`queue-commands-t5-stage1c.sh`): C5-long training at **54.49696** and its 200M read/QA at **54.49697**
  (so the 1b report at 54.49698 can use them); the 8 stage-1c training jobs `e15-t5-SmolLM2-360M-full-<arm>-s<k>` at
  **54.49699** (`--min-free-gb 20`, default resume), their 16 evaluations (`-qa`, `-read`) at **54.496992**
  (`--no-resume`), the CPU-lane report `e15-t5-stage1c-report` at **54.496995**; all before T7-ROOD at 54.497. Ordering
  as §16.6: jobs sorted by (priority, created), one live job per lane, the CPU lane waits for every lower-priority job.
- **Disk:** 8 × 1.45 GB `final.pt` ≈ 11.6 GB plus C5-long's 3 named checkpoints and `final.pt` ≈ 5.8 GB; the disk had
  63 GB free at registration with the last 1b run still to write; the 20 GB guard applies.

### 17.9 Files, code and tests

- Code (additive; no behaviour change without the blocks): `src/vsa_embed/aligned_targets.py` (`FV_TEMPLATES`,
  `AlignedObjectives.teacher_batch` / `teacher` / `distill_loss` / `cloze_batch` / `cloze_loss`, `aligned_from_config`,
  `recall_writer`), `src/vsa_embed/training/lm.py` (`channel.distill` / `channel.verbalize` in the loop — teacher pass
  before the student pass, the `kl_loss` / `cloze_loss` columns; `model.max_positions`, a longer GPT-2 position table for
  the toy tests), `src/vsa_embed/experiments/e15_eval.py` (`report --stage 1c`: `run_report_1c`, `kl_ratio`,
  `render_report_1c`; the `C5-long` contrasts of `report --stage 1b`). Eight configs + the C5-long control;
  `queue-commands-t5-stage1c.sh`.
- Tests `tests/test_e15_stage1c.py` (13 tests, CPU, log `…/scratchpad/e15-1c/pytest-e15-1c.log`): template
  disjointness; the teacher's input (recall lines + window), the untouched student input, the shifted spans, the
  distilled positions, KL shape / non-negativity / zero against itself / gradient to host and channel; the cloze
  prompts (term tokens, template suffix, filler name as the only labels, caps, determinism); tiny end-to-end training
  with each block on compose and free channels (columns, `loss = lm + λ·term`, no parameters added, reload); the
  regression that a config without the blocks has identical parameters, identical LM loss, no extra columns and
  byte-identical repeated runs; block validation; the configs; the stage-1c report on synthetic folders with the three
  decision readings.
