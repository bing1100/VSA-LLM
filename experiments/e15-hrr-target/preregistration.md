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

(none)
