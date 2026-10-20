# 13 — E13: the learning cycle — read, learn, write, reason, learn again (decision 63)

**Overview.** E13 sends one model once around the read–learn–write cycle. The model trains passively with a seed
ontology (stage 0), learns edges the ontology hid (1), and writes never-seen words from their definitions (2). It then
reasons about those words through recall (3), and trains on new text in a second round, where we count the tokens it
needs to reach a target loss (4). Stage 5 repeats the write and the second round on a frozen 4-bit host, against QLoRA.
E13 is the only experiment that tests the toolkit as a whole, and the only one that can support a training-efficiency
claim. **It has no results yet:** all 148 jobs are pending, in the T5 band from 54.4985 and the T7-ROOD band from
54.4996. A projection made today puts T5 done around Oct 25–26 and T7-ROOD around Oct 30, ±30%. The design was amended
three times before any run. Amendment 3, approved today, adds recall of only the asked relation (L5b), after E14 showed
that it works on real chemicals. This file covers the research gap, the design and the status of each part. The CPU
smoke runs check the pipeline; they are not results.

**Terms used throughout.**
- **Tracks.** **T5:** a synthetic company glossary of invented words. **T7-ROOD:** MeSH 2026 records in 2025–26 PubMed
  text, split by the records' entry date, with round-2 names removed from round-1 text (below). **T4:** real chemistry (ChEBI and PubMed); cited only as prior evidence.
- **Rounds.** **Round 1:** the terms the model trains on first. **Round 2:** new terms, kept out of all round-1 text.
- **Seed ontology:** the ontology the model starts with. It has no frames for round-2 terms, and 20% of round-1 edges are
  erased. The erased edges are the **erased gold** that stage 1 should recover.
- **Models.** **C5:** the method (a vector composed from the term's ontology frame, added at the term's last subtoken).
  **C0′:** fine-tuned without it. **C2:** a free per-term table. **Hosts:** SmolLM2-360M (primary, seeds 1–3) and
  Qwen3-1.7B-Base (LoRA, seed 1; descriptive).
- **Readers** (from E11) turn a definition into a frame:
  - **linker:** the trained model picks each found partner's relation by the definition's likelihood; it is the `read`
    arm;
  - **linker-joint:** the same choice inside the type prior's frame;
  - **type prior:** the relation each partner most often fills;
  - **oracle:** the gold frame;
  - **random:** the gold frame's shape, with frequency-weighted wrong partners.
- **Routes.** **Implicit:** written rows act through the composed vector (stages 2, 4, 5). **Explicit:** the store's
  decoded frame is written as text before a question (stage 3). `recall:own` gives the whole read frame;
  `recall-rel:own` gives only the asked relation's lines.
- **Strata.** **ref_round2:** next-token loss on the 8 tokens after a round-2 term, in fixed evaluation windows that are
  never trained on. **ref_round1:** the same after a round-1 term.
- **TTC (tokens to criterion):** the round-2 training tokens an arm needs to reach the `noread` arm's final ref_round2
  loss.
- **4-bit schemes.** **RTN int4:** round-to-nearest, group 128 (primary). **NF4:** QLoRA's 4-bit format (secondary).
  **QLoRA:** LoRA adapters trained on the frozen 4-bit host.
- **PMI scoring** (stage 3, E12's scorer): a candidate's log-probability after the prompt, minus its log-probability after
  a context-free null prompt; the highest wins.
- **Endpoints L1–L5b** are defined in the tables below.
- **Sign conventions.** Loss differences are a − b in nats, **negative = a has lower loss (better)**. Accuracy
  differences: **positive = better**. L2 is a ratio: **below 1 = reading first is faster**.
- **Statistics:** two-way cluster bootstrap (windows or anchors × seeds), 2,000 resamples. Holm runs across the seven
  primary tests of each track at α = 0.05; a test that cannot be evaluated counts as p = 1.

---

# 13.1 — Write new words, recover hidden edges, then learn faster (stages 0–2 and 4; L1, L2, L4)

**Status:** queued (no results) · **When:** pre-registered 2026-10-08 (commit 1cf9dfb); amendment 1 (T7-ROOD)
2026-10-08; amendment 2 (decision 64) 2026-10-09; CPU smokes 2026-10-08, 2026-10-09 and 2026-10-20 · **Sources:**
`experiments/e13-learning-cycle/preregistration.md` §§1–12; `experiments/e13-learning-cycle/README.md`;
`experiments/e13-learning-cycle/{t5,t7-rood}.yaml`; `experiments/e13-learning-cycle/smoke*/`;
`manuscript/toolkit-methodology-2026-10.md` §§0, 2, 4

## Research gap
The methodology note rates *read* the strongest tool and *learn* the weakest: rule mining matches it, its acceptance
test is uncalibrated, and E10.9a found passive plus active learning no faster at matched data. *Write* is mixed: gold frames
help on invented words but not on real chemistry, and E11 later found that read frames recover under half of the
invented-word gain. When E13 was designed, no experiment chained learn,
write and train in one model, or measured whether writing a word first makes the next round of training cheaper. (E14's
RD-12 pilot later tested a smaller read → train → read loop on T4; see file 14, exploration 14.7.)

## Intuition
A model that writes rows for new words before reading text about them starts round 2 already knowing something. It
should then reach a target loss in fewer tokens. Stage 1 asks whether the model can also recover structure that the
seed ontology hid.

## Methodology
**Tracks and rounds:**

| | T5 (controlled upper bound; runs first) | T7-ROOD (main real-text track) |
|---|---|---|
| Round 1 | 3,640 training terms | 5,714 MeSH records introduced before 2013-01-26 |
| Round 2 | 360 held-out terms, absent from all training text (leakage audit 0) | 2,470 records introduced later; no round-1 training document names one |
| Round-2 text | new T5 documents (≈ 5M tokens) plus as much FineWeb-Edu | 21.3M tokens: the 24,882 abstracts round 1 dropped, plus general text, 50/50 |
| Seed ontology | 716 edges to round-2 terms dropped; inverses `owns` (3,240 edges) and `has_part` (690) added; 6,024 of 30,121 round-1 edges erased | E10.L's rule: 1,966 of 15,594 edges of 2,933 seen entries erased |
| Definitions read | E11's prose definitions (from the gold frame, in held-out wording) | 906 MeSH scope notes |
| Stage-1 proposer | rule closure, accepted by an LM held-out utility test with Holm and null-world proposals | TK-L's learn tool (decompose + closure, Holm plus a decoy threshold); rule closure secondary |

**Stages:**
- **0:** the E9 C5 recipe, 50M tokens, on the seed ontology.
- **1:** propose missing edges; accept each only if it passes a test whose false-acceptance rate a null world measures.
- **2:** the linker reads the round-2 definitions, and their rows are inserted, with no gradient step.
- **4:** each arm continues training for 10M round-2 tokens, evaluated at 13 points from 0 to 10M tokens.

**Stage-4 arms** (all start from the same stage-0 model and see the same batches):
- `read` (linker frames), `noread` (no row), `gold` (oracle frames), `random`;
- `fvt` (trainable subtoken-mean rows: standard vocabulary expansion);
- `defs` (definitions prepended to the training text, compute-matched);
- C0′ and C2 references (T5 only).

**Endpoints** (primary host; kill criteria from the pre-registration):

| Code | Question | Met iff | If not met |
|---|---|---|---|
| **L1** (write) | Do written rows lower the loss after round-2 terms before any training? | `read − noread` **and** `read − random` (amendment 2) at step 0: each Holm p < 0.05 with upper CI < 0 | no zero-shot write claim; if only the first passes: "rows written for new terms lower the loss, but not specifically through their read frames" |
| **L2** (efficiency) | Does reading first cut the tokens to criterion? | TTC ratio `read / noread`, with reading cost charged (forward tokens ÷ 3): Holm p < 0.05, upper CI < 1 | **no training-efficiency claim** |
| **L4** (learn) | Are accepted edges right, and is the test calibrated? | precision ≥ 0.80 against erased gold, null false-acceptance ≤ 0.05, binomial Holm p < 0.05 | stage 1 dropped and reported as negative |

**Named secondaries:** `read − definition` at step 0, from the `context` jobs (definitions prepended to the evaluation
windows), with token costs; the other arms against `noread`; area under the learning curve; forgetting (ref_round1);
general-text locality.

**Expectations (pre-registration §7):**
- L1 is large on T5 with gold frames, and smaller with the linker (E11 frame F1 0.55 against 1.00).
- On real text, `read − noread` may pass while `read − random` fails.
- Stage 1 adds little speed but writes correct edges. On T5's seed ontology, rule closure proposes 1,302 edges, all
  erased gold (a CPU check without a model).

## Experiment
**Queue status** (read-only check, 2026-10-20; every job `pending`):

| Part | T5 level (jobs) | T7-ROOD level (jobs) |
|---|---|---|
| Preparation, stage 0, learn, write | 54.4985 (14) | 54.4996 (18) |
| Stage 4 | 54.4986 (27); Qwen3 `random` 54.49865 (1) | 54.49965 (21); 54.49967 (1) |
| L1 definition comparator (`context`) | 54.49885 (4) | 54.49977 (4) |
| Report (CPU lane) | 54.4989 (1) | 54.49979 (1) |

Stages 3 and 5 are in 13.2 and 13.3. In all: T5 has 75 jobs (≈ 38.4 GPU-h) and T7-ROOD 73 (≈ 48.2 GPU-h), plus about
2 GPU-h per track from amendment 3. About 235 jobs sat ahead of the T5 block on 2026-10-20.

## Results
**None yet.**

**Smoke** (SMOKE, not results: SmolLM2-135M on the CPU, 8 windows of 128 tokens, 3 stage-0 steps, 2 round-2 steps):
- **Every stage ran end to end** on both tracks, after each amendment.
- **Learn.** On T5, rule closure made 24 proposals, all erased gold, plus 24 null proposals. Six were testable on the
  tiny validation text, and none was accepted. On T7-ROOD, TK-L's pipeline reproduced the 1,966 erased edges; rule
  closure proposed nothing on T7's relations.
- **Write.** The linker's frame F1 was 0.55 on the 4 round-2 terms in the smoke windows. On the stage-2 items, the
  definition in context scored 0.72 against 0.43 for none; written rows changed the scores but not the choices.
- **L1** `read − noread` was −0.009 nats on 4 windows (T5).

## Limitations
- **No results yet.** Every statement about outcomes is an expectation.
- **A weak reader in the `read` arm.** E11 found linker F1 0.51–0.53 on T5 new words and 0.02–0.03 on T4.
- **T7-ROOD's notes rarely name partners:** about 2% of gold partners on T7's held-out records (E11 §§15–16), so many
  linker frames may be empty.
- **The random control is not content-free on T7-ROOD.** Random frames keep 36% of the gold edges there (9% on T5), so
  `read − random` measures content beyond the type and branch prior.
- **The real-text gate.** §7 ties T7-ROOD's L1 and L2 to T7 showing a gain. T7 failed its held-out prediction (+0.48%,
  worse; 12 Oct update), which is why T7-ROOD exists.
- **Missing comparators.** Qwen3 has one seed; T7-ROOD has no C0′ or C2 references; C6d-style rows are not an arm.
- **Source differences** (the pre-registration is authoritative):
  - the README still says nothing is queued, and lists amendments 1–2 only;
  - the methodology note estimated 40–55 GPU-h; the queued plan is ≈ 86.6 GPU-h (38.4 + 48.2) before amendment 3;
  - the 15 Oct update projected E13 for ≈ Oct 24; today's projection is later.

## Next steps
- **Queued:** all parts (table above). Projection: T5 ≈ Oct 25–26, T7-ROOD ≈ Oct 30, ±30%.
- **Done:** the pre-registration, three amendments and four smoke runs.
- **Rule:** any further change must be a dated amendment, written before the affected jobs run (§9).

---

# 13.2 — Reason about new words with recall (stage 3; L5, L5b)

**Status:** queued (no results) · **When:** pre-registered 2026-10-08; token accounting and definition comparator
added 2026-10-09 (amendment 2); selective recall added 2026-10-20 (amendment 3, commit 2d935d6) · **Sources:**
`experiments/e13-learning-cycle/preregistration.md` §§4–5, 12 (amendments 2–3);
`experiments/e13-learning-cycle/amendment-proposal-2026-10-20.md`; `experiments/e13-learning-cycle/smoke-amendment3/`;
`manuscript/findings/2026-10-20-e14-followups.md` §1

## Research gap
E12 showed that recall works on gold-built stores. We had not tested it on a store that the model wrote itself, from
reading, for words it had never trained on. When E13 was designed, recall's token cost against a plain definition was
also unmeasured. E12's 3c later measured it on gold-built T5 stores: 182.9 tokens per item for the decoded text,
against 115.6 for the definition (file 12, exploration 12.4).

## Intuition
If the written frames hold the right partners, recall turns them into text that the model can read. E14's RD-4b showed
that the lines must be selective. Recall of only the asked relation raised property accuracy from 0.614 to 0.862 on T4
(+0.248\*) and from 0.190 to 0.845 on T5 (+0.655\*). Whole-frame recall did worse.

## Methodology
- **Items:** relation, reverse and two-hop questions about round-2 anchors. T7-ROOD has no two-hop items, and uses a
  seeded sample of 600 anchors.
- **Conditions:** `none`, `recall:own`, `recall-rel:own`, `symbolic` (the gold frame as text) and `definition` (T5:
  E12's gold-derived prose; T7-ROOD: the anchor's scope note).
- **Endpoints:**

| Code | Statistic | Met iff |
|---|---|---|
| **L5** | `recall:own − none` per anchor (anchors × seeds) | Holm p < 0.05 and lower CI > 0 |
| **L5b** (amendment 3) | `recall-rel:own − none`, same statistic | Holm p < 0.05 and lower CI > 0 |

- **Cost criterion** (a fixed-sequence secondary after L5): L5 met, `recall:own − definition` non-inferior at δ = 0.05,
  and fewer tokens. Only if all three hold may we claim "a cheaper substitute for the definition".
- **Secondaries:** `recall-rel:own − recall:own`; stage 3 rescored with the type prior's frames written
  (`recall-rel:own@typeprior`).
- **Predictions** (amendment 3): L5b > 0 on T5; L5b ≥ L5; the type-prior pass ≥ the linker pass.

## Experiment
Queued: 4 reason jobs per track (T5 54.4988; T7-ROOD 54.49975). They read the amended configuration at run time.

## Results
**None yet.**

**Smoke** (SMOKE, not results):
- **Prompt tokens added per item.** T5: recall 45, definition 140, symbolic 158. T7-ROOD: 60, 14 and 116. At smoke
  scale, T7-ROOD's notes are cheaper than recall text, so the cost criterion may favour the definition there.
- **Amendment-3 smoke.** Selective and whole-frame recall gave identical results on the smoke items (difference 0.000).
  So did the type-prior pass. Every verdict was False, as expected.
- **A bug found and fixed:** the report's L3 block overwrote L5's margin δ. A test now covers it.

## Limitations
- **The linker writes the recalled frames.** In RD-4b, linker frames recovered +0.28 of the oracle's +0.66 on T5
  property, against +0.62 for the type prior. The type-prior pass is only a secondary.
- **L5b is told the asked relation.** E14 RD-4c (preliminary; file 14, exploration 14.6) found that choosing it without
  the label costs about 8 points with full oracle frames, and nothing with short type-prior frames.
- **Weaker test on T7-ROOD.** Only about 37% of its sampled anchors are expected to have a note.
- **The draft and the final amendment differ.** The draft restricted two-hop items to the anchor's and the bridge's
  lines. The final amendment leaves two-hop and reverse items as under `recall:own`, since they already recall only
  their relations. It also renames the type-prior secondary `recall-rel:own@typeprior`.

## Next steps
- **Queued:** the reason jobs above.
- **Done:** amendment 3, implemented, tested and committed before any reason job.

---

# 13.3 — A frozen 4-bit host against QLoRA (stage 5; L3)

**Status:** queued (no results) · **When:** pre-registered 2026-10-08 · **Sources:**
`experiments/e13-learning-cycle/preregistration.md` §§4–5; `experiments/e13-learning-cycle/README.md` (design notes);
`experiments/e13-learning-cycle/smoke-amendment3/report/report.md`; `manuscript/toolkit-methodology-2026-10.md` §4

## Research gap
QLoRA is the standard way to adapt a frozen 4-bit model. We had not compared it with writing rows into a side store,
with no gradient on the host, on the same new-word text. E9 measured only post-training quantization: the method's
advantage roughly survives 4-bit compression, by an amount that depends on track and quantizer (12 Oct update, E9-9).

## Intuition
On a frozen 4-bit host, only the channel and its rows can change. If written rows carry what the new words mean, they
could replace part of the adapter training.

## Methodology
- **Setup:** stages 2–4 repeated with the stage-0 host quantized and frozen. The quantization is simulated: the weights
  hold the dequantized values a 4-bit kernel would compute with. The output head and input embedding stay at 16 bits.
- **Arms:**
  - `q4-read` and `q4-noread` (the channel trains);
  - `qlora` (LoRA r 64 plus the channel);
  - `qlora-read` (RTN only).

  NF4 runs the first three. Qwen3 runs the RTN `q4-read`, `q4-noread` and `qlora`.
- **L3:** `q4-read − q4-noread` on ref_round2 loss at 10M tokens (RTN): Holm p < 0.05 and upper CI < 0. Non-inferiority
  to QLoRA (upper CI of `q4-read − qlora` below 1% of QLoRA's loss) is reported with it.
- **Kill criterion:** if QLoRA wins clearly, the claim is only "zero-gradient".
- **Secondaries:** NF4 under L3's statistics; `qlora-read − qlora` (do rows add to adapters?); nominal host bytes and
  trainable parameters of every frozen arm. Text, tokens and evaluations match stage 4.

## Experiment
Queued: 24 stage-5 jobs per track (T5 54.4987; T7-ROOD 54.4997).

## Results
**None yet.**

**Smoke** (SMOKE, not results; SmolLM2-135M): the channel trained while the 4-bit host stayed frozen, and `qlora` saved
only its adapters. Nominal host bytes were equal across arms (4.5 bits per weight). `qlora` trained 20.7M parameters against
1.2M for `q4-read`.

## Limitations
- **No memory saving.** Because the quantization is simulated, memory is not saved and the reported bytes are nominal.
- **Scheme drift.** The methodology note proposed GPTQ and plain rounding; the pre-registration uses RTN and NF4.
- **No specificity gate.** No `q4-random` arm exists, so L3 cannot show that the read content matters.

## Next steps
- **Queued:** stage 5 on both tracks.
- **Done:** the stage-5 pipeline, checked in every smoke run.

---

# Sources

- `experiments/e13-learning-cycle/preregistration.md` (§§1–12; amendments 1–3)
- `experiments/e13-learning-cycle/README.md`; `experiments/e13-learning-cycle/amendment-proposal-2026-10-20.md`
- `experiments/e13-learning-cycle/t5.yaml`; `experiments/e13-learning-cycle/t7-rood.yaml`
- Smokes: `experiments/e13-learning-cycle/{smoke,smoke-t7-rood,smoke-amendment2/t5,smoke-amendment2/t7-rood,smoke-amendment3}/`
- `manuscript/toolkit-methodology-2026-10.md` §§0, 1, 2 (M2, M3), 4, 6
- `manuscript/update-2026-10-12.md` (E9-7, E9-9, E11); `manuscript/update-2026-10-15.md` (timeline)
- Cross-references: `manuscript/findings/2026-10-20-e14-tier-p-gpu.md` §1; `manuscript/findings/2026-10-20-e14-followups.md`
  §1; `manuscript/status/10-20-2026/14-e14-reading-real-text.md` (14.5–14.6); `11-e11-read-to-learn.md`;
  `12-e12-explicit-recall.md`
- Queue state: a read-only listing of `.jobs/e13-*.json` on 2026-10-20 (148 pending)
- Dates: `git log -- experiments/e13-learning-cycle` (1cf9dfb 2026-10-08 … 2d935d6 2026-10-20)
