# VSA-LLM — state of the research (20 October 2026; updated 21 October with the overnight results)

**For:** the program lead and collaborators. A human-readable digest of every experiment done so far, with one file per
area. The files are numbered in program order (01–16).

**How each file is built.** Every exploration follows the same seven sections:
1. research gap;
2. intuition;
3. methodology;
4. experiment;
5. results;
6. limitations;
7. next steps.

Each file opens with an overview and a list of the codes it uses, and closes with its sources (repo-relative paths).
Numbers are copied from the committed reports and findings records. Preliminary or single-seed results are labelled as
such.

**Conventions.**
- Loss changes are relative to the plain fine-tuned baseline C0′, and negative means better.
- Accuracy differences are in points, and positive means better.
- An asterisk (\*) marks significance after the pre-registered multiple-test correction.
- Intervals are 95%.

**Writing.** The prose follows the five audit passes of the *sciwrite* skill (L. A. Barba, CC BY 4.0,
github.com/labarba/sciwrite), which is based on K. Sainani's *Writing in the Sciences*:
1. cut clutter;
2. active voice;
3. keep subjects close to their verbs;
4. one term per concept;
5. consistent numbers.

---

## 1. The state of the research in one page

**The method.** The project adds one vector to a language model at each multi-token term. The vector is composed from
the term's ontology entry using a vector-symbolic algebra (holographic reduced representations, HRR). Five questions
organise the evidence.

### Q1. Does the composed vector help a language model predict text?

**Yes, but modestly on real text, and only where the words are new to the model.**
- **Invented words (synthetic glossary, T5):** large gains. Loss after held-out terms falls 13.3%\* at 360M and
  17.7–20.2% on Qwen3 (file 06.1). This is an upper bound.
- **Real text:** the gain is 1–2% and follows novelty.
  - Chemistry (T4): −2.29% after unseen terms and −1.09% after long terms.
  - Barely-new 2026 medical names (T7) and known vocabulary (T1-open, WordNet): no gain (06.2).
  - Clinical notes (T1c) show the same shape: −0.94% after long terms at 360M, nothing for new words (08.2).
- **Smaller hosts gain more on real text:** at 135M, −3.62% after unseen T4 terms (06.11).
- **The vector adds information beyond a definition in the prompt,** and costs no prompt space (06.8). This is the
  strongest real-text point in the method's favour.
- **Trained from scratch (E4):** a small, steady gain of −0.33%\* overall and −1.21%\* after long terms at 125M,
  3 seeds. It does not grow from 50M to 125M, so we do not scale further (file 03).

### Q2. Why does it help? What carries the gain?

**Not the binding operator, and on real text not composition as such.**
- Every binding operator ties, on every track and both training regimes (02.2, 03.1, 06.3). The allowed wording is
  "compositional parameter sharing".
- On real text, a wide enough free per-term vector matches the method after long terms (06.6).
- The "shared building blocks" explanation was withdrawn on 15 October (06.5).
- Composition still needs about 7.5× fewer parameters, and wins on unseen and rare terms.
- **On real text, a vector for a held-out name the model already saw unlinked is a net cost** (+0.48%\* on T7; 06.9).
  The clean test, T7-ROOD (new names with leakage audited to zero), has not started (08.1).

### Q3. Does the HRR store hold structure the model can use?

**The store holds it. The model does not use it on its own, but it can when the store is decoded into the prompt.**
- **The store is readable:** unbinding recovers related concepts almost perfectly, MRR +0.77 over a frequency guess
  (09.1).
- **The model ignores roles by itself:** on role-swapped twins, every arm scores at chance (09.2). Even arms trained
  with an unbinding readout score at chance (R1 null; 09.4).
- **Decoding the store into the prompt (explicit recall) works:**
  - twins rise from 0.50 to 0.78\* on T5 (12.1);
  - role assignment on real chemistry rises from 0.55 to 0.88 (12.3).
- **Selective recall is the strongest result of the week** (14.5). Recalling only the lines for the relation a
  question asks about lifts:
  - real-chemistry property accuracy from 0.61 to 0.86 (+0.25\*);
  - new-word property accuracy from 0.19 to 0.85 (+0.66\*).

### Q4. Can a model learn new knowledge by reading?

**Partly. Reading is limited by what text states and by how facts are selected, not by reader intelligence.**
- **Definitions state only about 12% of a term's ontology facts** (14.1). Bigger dictionaries do not help, and
  reading does not snowball.
- **Labelling relations is not the bottleneck:** a frontier model (Claude) beats a simple type prior by only +0.03 F1
  (14.2).
- **Frames read into the term's vector help invented words a little** (+0.031\*) and not real chemistry (11.1).
- **Frames recalled as text help a lot** (14.5).
- **The realistic pipeline helps** (14.6): a real reader plus the model's own choice of relation gives +0.10\* on real
  chemistry and +0.11\* on new words.
- **Choosing the relation was the next bottleneck. It is solved on real chemistry and not on new words** (14.8). A
  few-shot router costs 0.018 against the labelled ceiling on T4, and the realistic pipeline matches the definition
  text (0.724 against 0.734). On T5 every learned router fails (−0.445\*): the held-out wording does not resemble the
  demonstrations. RD-4e (addendum 12) tests routers that generalise.
- **More text and inferred facts do not help on T4** (14.10, 14.11). A Wikipedia page states 0.120 of a term's
  facts against 0.116 for the definition, and a store read from it hurts (−0.107\*). Facts inferred by neighbour
  voting help only when they are right, so they need a precision gate; the predictor's hit rate (0.16) sits at the
  break-even point.
- **Frames written by a model are not more precise than the ontology's own statistics,** on any track tried. A
  Claude-read ontology in a 50M model from scratch (C5teach, 03.4) changes the loss inside the terms it describes
  (−0.34\* against C5) and nothing after them, at one seed. E7's self-authored frames failed their gate (05.2).
- **Selective recall holds on Qwen3 hosts** (+0.17 to +0.62\*).
- **The self-learning loops are negative so far:**
  - E10's active plus passive learning is not faster (10.3);
  - the read → train → read cycle (RD-12) adds nothing through the loss, even with a passive phase that no longer drifts
    (RD-12b);
  - the cycle's extra facts help only a little through selective recall (+0.015\*) (14.7);
  - E13 tests the full cycle from about 25 October.

### Q5. Can logic run inside the store, and does it help a model?

**The store executes logic exactly. Models can relay its conclusions. It is not better than symbolic logic.**
- **Execution is exact at width ≥ 1,024:** chains, joins, negation, numbers and decision trees (15.1).
- **Concatenating blocks removes the algebra's hard limits,** and laying trees out by structure is exact to depth 12
  (15.2).
- **In the "soft" settings the store only ties symbolic baselines** (gate G2 fails; 15.3).
- **Qwen3 hosts relay the executor's conclusions** (+0.42 to +0.47\* at every depth). Hosts around 360M fail true/false
  questions even with the answer stated, and succeed in fill-in-the-blank form (15.4–15.5).
- **A 1.7B model guides a sound prover** on hard problems (+0.11 to +0.15\* over random search). Soundness comes from
  the tools (15.7).
- **The best prover is a hybrid:** a symbolic heuristic orders the steps and the model breaks ties, beating either alone
  (+0.14\* over set of support at budget 10).
- **A tree cannot be read from prose at ≤ 1.7B** (15.8): leaf accuracy 0.24 on paraphrased trees for the best host,
  against 0.82 on templated ones. The executor needs its tree structured.
- **The executor keeps a chain of what-if edits consistent** (15.9): 99.7% of chains right at every step, against
  3–5% in context (+0.72 to +0.76\* at k = 8). The caveat: in-context answering is at chance from k = 0, so this is a
  claim about execution, not about drift.
- **Under noisy records the executor's path still beats the tree in text** (+0.67\* / +0.49\*), and the model follows
  it even when it is wrong (15.10). X7's ensemble gain was width (15.10).

### Also

- **4-bit quantization keeps the gain to a degree that depends on track and quantizer.** Report it as a measurement,
  not a claim (07.1).
- **Editing the ontology changes predictions on invented words only** (07.2).
- **Results reproduce bit for bit on the replacement machine,** and every result is pre-registered (16.1, 16.10).

## 2. What the manuscript can claim now

| Can claim (with evidence) | Cannot claim | Pending a queued test |
|---|---|---|
| A small, steady gain from scratch that does not grow with size (03) | that binding matters for loss (06.3, 03.1) | zero-shot vectors for genuinely new names (T7-ROOD, 08.1) |
| Retrofit gains that follow novelty, larger for small hosts (06.2, 06.11) | "sharing building blocks" as the mechanism on real text (06.5) | richer ontologies (06.12) |
| Adds on top of a definition in the prompt, at zero prompt cost (06.8) | soft reasoning better than symbols (15.3) | the learning cycle (E13, 13) |
| The store is readable; decoded explicitly it supplies roles (09.1, 12.1, 12.3); whole-frame recall needs no router (RD-4e, 14.8) | that models use stored roles unaided (09.2, 09.4) | — |
| Selective explicit recall works on real and new terms (14.5) | self-learning that "learns faster" (10.3, 14.7) | the cycle under selective recall (14.7: RD-12b, RD-12c) |
| An exact executor with a model-readable interface; sound LLM-guided proving (15) | quantization as a headline claim (07.1) | Wikidata entities (T8, 08.3) |
| The executor keeps counterfactual chains consistent, 99.7% against 3–5% in context (15.9); stated as execution, not drift, because in-context answering is at chance at k = 0 | self-authored or Claude-written frames as a source of structure on real text: E7's G3 failed (05.2), C5teach adds nothing after terms (03.4); both closed negatives | |
| A structure-only vector for an unseen concept lifts property 0.52 → 0.67 and entailment 0.72 → 0.85 at 125M from scratch (E5.4, 04.2); the gain is the frame's content, which a graph projection matches | trees read from prose at ≤ 1.7B (15.8); more text or inferred facts as a lever on T4 (14.10, 14.11) | |

## 3. Index (program order)

| File | Area | Status | One-line verdict |
|---|---|---|---|
| [01](01-mechanism-checks.md) | VSA capacity; frozen-row factorisation; E0 planted structure | done | Width 512 holds about 16 pairs. Typed HRR loses to an untyped sum on frozen rows. E0's gate G1 passed. |
| [02](02-frozen-host-experiments.md) | E1, E2 (+F.1), E3 on frozen hosts | done, closed | The context query fits better but does not pick senses. Operators tie. F.1 refuted. The split test does not transfer to WordNet. |
| [03](03-e4-small-models-from-scratch.md) | E4: models trained from scratch, scaling | done (3 seeds at 125M; C5teach 1 seed, 21 October) | About −0.3% overall and −1.2% after long terms. No growth with size. No held-out benefit. A Claude-read ontology (C5teach) changes nothing after terms. |
| [04](04-e5-explainability.md) | E5: explainability | done on the 125M scaling checkpoints (amendment 1; 21 October) | Zero-shot insertion is the clear positive (0.52 → 0.67). Attention is weakly faithful. Senses and frequency are no cleaner than baselines. The judge is weakly positive against C0 only. |
| [05](05-e7-self-authoring.md) | E7: self-authoring | D7.1 done; G3 failed 21 October; D7.2/D7.3 not queued; closed on this track | Self-authored and teacher edges for masked concepts are no more precise than degree-matched random ones (0.062 and 0.069 against a random high of 0.243). |
| [06](06-e9-retrofit-pretrained-models.md) | E9: retrofit into pretrained models (core) | done; follow-ups queued | Large on invented words; 1–2% on real new words; mechanism unresolved. |
| [07](07-e9-quantization-and-editing.md) | E9: 4-bit quantization; ontology editing | done | Quantization is a measurement. Editing is a proof of concept on synthetic data. The size-matched control B-B1 is queued. |
| [08](08-e9-new-vocabulary-and-clinical.md) | E9: T7/T7-ROOD, T1c, T8, other tracks | partly done; decisive tests queued | T7 failed for diagnosed reasons. T1c has the same shape as chemistry. T7-ROOD, T8 and T2 (approved 20 October) are pending. |
| [09](09-binding-programme.md) | Binding programme, including the readout arms | done | The store is readable and the algebra behaves as designed, but behaviour does not follow it. |
| [10](10-e10-self-learning-ontology.md) | E10: self-learning ontology | done (negative); learn tool pending | Baselines explain most of it. Nothing transfers to WordNet. Active plus passive learning is not faster. |
| [11](11-e11-read-to-learn.md) | E11: read to learn | partly done | Read frames help invented words a little and real text not at all. Text routes win. |
| [12](12-e12-explicit-recall.md) | E12: explicit recall | done | Decoding the store into the prompt resolves roles, on synthetic and real text. |
| [13](13-e13-learning-cycle.md) | E13: the learning cycle | queued; amendment 3 approved | No results yet. T5 about 25–26 October, T7-ROOD about 30 October. |
| [14](14-e14-reading-real-text.md) | E14: reading facts from real text | done | Text states about 12% of facts. Selective recall works (+0.25 / +0.66) and holds on Qwen3. Routing is solved on T4 (−0.018) and open on T5. More text and inferred facts do not help. The cycle adds little. |
| [15](15-e14-logic-in-the-store.md) | E14: logic in the store | done | An exact executor with a readable interface, consistent under what-if chains (99.7% against 3–5%). Not better than symbols. Trees cannot be read from prose at ≤ 1.7B. A model plus a heuristic proves best. |
| [16](16-infrastructure-and-validation.md) | Reproduction, benchmarks, judge, toolkit, queue | done / partly queued | Results reproduce. The harnesses are validated. The toolkit write and learn evaluations are queued. |

## 4. What is running, and when it lands

**Projected from measured job durations** (one GPU, one job at a time; ±30%):

| Block | Lands about |
|---|---|
| E14 follow-ups (RD-4c, RD-12b, RD-12c, the hybrid prover) | **landed** 20 October; `manuscript/findings/2026-10-20-e14-evening.md` |
| Approved 20 October, run first: E14 tier F (learned routers, strict ReAct, equal-size store, noisy trees), E7 D7.1, E5 on the 125M checkpoints, C5teach | **landed** 21 October; `manuscript/findings/2026-10-21-overnight.md` |
| Added 20 October (evening): tree-text, RD-13 retrieval, LG-CF, RD-14 | **landed** 21 October; same findings file |
| RD-4e (addendum 12): routers that generalise across wordings | **landed** 22 October; `manuscript/findings/2026-10-22-rd4e.md` |
| Qwen3 retrofit evaluations; the T7-ROOD block (the decisive new-vocabulary test) | 24 October |
| E13 on T5 | 25–26 October |
| T8 (Wikidata) | 27 October |
| E10's learn tool | 28 October |
| E13 on T7-ROOD | 30 October |
| Approved 20 October, run after the toolkit: T2 (developer tools), B-B1 (side-module control) | 4–6 November |
| Everything queued, including T1c-ROOD, the richer-ontology study and the editing baselines | about 7 November |

The whole queue is about 425 GPU-h. The blocks approved on 20 October add about 44 GPU-h. The blocks that ran first
took about one third of a day, as projected, and the dates above are otherwise unchanged. D7.2's 44 GPU-h are not
queued (G3 failed), so nothing moves later for it.

## 5. Decisions approved on 20 October

The author approved all six decisions on 20 October. Everything that makes no Claude calls was queued at once. The
three steps that spend money on Claude were held first, because this session's permission check refused the C5teach
teacher spend. **The author approved the Claude spend later that evening, and all three ran:**
- the C5teach teacher build is done (24,670 synsets, USD 68.05, 0 errors); the 50M run and its 2 reports landed on
  21 October (file 03, 03.4);
- the E5 judge grade jobs ran after the E5 item builders (cap USD 25); the held-out neighbours study was refused by
  the cap with USD 4.32 left (file 04, 04.2);
- the watcher ran E7's follow-ups after D7.1 (teacher on 999 concepts, USD 9.07) and the G3 check. **G3 failed**, so
  D7.2 was not queued. The judge (cap USD 50; informational) was interrupted at USD 22.66 by a session restart and
  resumed from its cache (file 05, 05.2).

The last column of the table below lists what was held before that approval.

| # | Decision | Pre-registration | Queued | Held, then started (Claude spend) |
|---|---|---|---|---|
| 1 | C5teach | `e4-small-lm/preregistration-scale-v1-toolkit-status-2026-10-20.md` | 3 GPU jobs (about 2.5 GPU-h) and 2 reports, queued after the teacher build | the teacher build (projection USD 73; cap USD 82.40, cumulative, including USD 2.41 already spent) |
| 2 | E14 tier F | `e14-logic/preregistration.md` §19 | 19 jobs, about 3.7 GPU-h: RD-4d (learned routers), RD-3b (strict ReAct), X7b (equal-size store, untrained controls), LG-D2 (noisy trees). X8b (irregular trees) and LG-P3b (premise retrieval at scale) already ran on the CPU; results in `manuscript/findings/2026-10-20-e14-evening.md` §5 | — |
| 3 | E5 on the 125M checkpoints | `e5-explainability/preregistration-amendment-1-scale-v1.md` | 41 jobs, about 3.3 GPU-h and 1.7 CPU-h: E5.1, E5.2, E5.4, E5.5, the E5.3 item builders and the R5 report | the 3 E5.3 grade jobs (Claude judge; cap USD 25) |
| 4 | E7 rescheduled | `e7-self-authoring/preregistration-2026-10-20-reschedule.md` (new G3 rule) | the 9 D7.1 jobs, about 1.1 GPU-h | the follow-ups (Claude teacher, cap USD 15) and the judge (cap USD 50); D7.2 (about 44 GPU-h) is queued only if G3 passes, and D7.3 after it |
| 5 | T2 (developer tools) | `e9-retrofit/preregistration-addendum-2026-10-20-t2.md` | 83 jobs, about 23 GPU-h | — |
| 6 | B-B1 (side-module control) | `e9-retrofit/preregistration-addendum-2026-10-20-bb1.md` | 22 jobs, about 12.9 GPU-h | — |

**Added later on 20 October (author: tree test, retrieval tool, cheaper tests; pre-registration §20, addenda 9–11).**
Seventeen E14 jobs at 54.4959997 (about 2.5 GPU-h), reports at 54.49599985–7, landing 21 October:
- **tree-text** (addendum 9): can a host read a decision tree from prose into the executor's form? Decision rule:
  worth more only if the best host reaches 0.80 leaf accuracy on the paraphrased style.
- **RD-13, retrieve-then-read** (addendum 10): Wikipedia pages for 332 of 341 T4 concepts (MediaWiki API, cached by
  revision; CC BY-SA). **Already known from the CPU stage:** the page states 0.120 of the gold edges, the definition
  0.116; more text is not the lever on T4, and no dump is fetched. The queued arms compare the store with the same
  text in the prompt.
- **LG-CF** (addendum 11): chains of up to 8 what-if edits, executor relay against in-context answering.
- **RD-14** (addendum 11): facts inferred by neighbour voting (precision 0.55), recalled marked or unmarked.

**Results of the 21 October blocks.** Every job above ran without failure. The results are written up in
`manuscript/findings/2026-10-21-overnight.md` and in files 03 (03.4), 04 (04.2), 05 (05.2), 14 (14.8–14.11) and 15
(15.8–15.10). In one line each: routing is solved on T4 and not on T5; the strict reader still loses to the type
prior; trees cannot be read from prose at ≤ 1.7B; the executor keeps what-if chains consistent (99.7% against 3–5%);
Wikipedia states no more facts than the definition, and a store read from it hurts; inferred facts need a precision
gate; E5.4 zero-shot insertion is the clear E5 positive; C5teach changes nothing after terms; E7's G3 failed.
- **Stopped:** trees from prose at ≤ 1.7B; retrieval over more text on T4; recall of inferred facts; E7 D7.2 and
  D7.3; C5teachF.
- **Continuing:** E15 HT, "HRR as a training target" (approved 22 October, high priority; `experiments/e15-hrr-target/`); E13 (lands 25–30
  October). The consistency result (15.9) and E5.4 (04.2) go to the claims ledger.
- **Caveats to carry:** LG-CF's in-context arm is at chance at k = 0, so the consistency claim is about execution,
  not drift; tree-text, LG-CF and RD-14 are one item set each; C5teach is one seed; E5.3's held-out study did not
  run.

**Placement.** Lower numbers run first.
- The small blocks run before the toolkit band (54.496–54.4999): E14 GPU jobs at 54.4959990–91, E7 D7.1 at
  54.4959993, E5 GPU jobs at 54.4959994, the E5 reports at 54.4959998, the E14 reports at 54.4959999.
- T2 (54.5001–54.5004) and B-B1 (54.501–54.5015) run after the toolkit band and before the editing baselines (54.6).
- A CPU-lane report waits for every job with a lower number. A failed report therefore holds back every later
  report. The E5 grade jobs are wrapped, so that a study cut by the judge cap is recorded as INCOMPLETE and does
  not fail the queue.

**The commands for those steps**, kept for the record (from the repository root,
`PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python`):

1. C5teach. Build the teacher answers, then the ontology, then queue the run:

   ```bash
   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher teach --scope hybrid --max-usd 82.40
   PYTHONPATH=src $PY -m vsa_embed.experiments.c3_teacher ontology --scope hybrid
   PY=$PY PRIO_TRAIN=54.4959992 PRIO_REPORT=54.4959996 bash experiments/e4-small-lm/queue-commands-c5teach.sh
   ```

2. E5 judge. Add the 3 grade jobs and a final R5 report:

   ```bash
   JUDGE_ONLY=1 PRIO_GPU=54.4959994 PRIO_REPORT=54.4959998 bash experiments/e5-explainability/queue-commands-scale-v1.sh
   ```

3. E7, after the 9 D7.1 jobs are done. Run the follow-ups and the judge, then queue D7.2 if G3 passes:

   ```bash
   bash experiments/e7-self-authoring/queue-commands-d72-d73.sh followups
   bash experiments/e7-self-authoring/queue-commands-d72-d73.sh judge
   PRIO_D72=54.5020 bash experiments/e7-self-authoring/queue-commands-d72-d73.sh d72
   ```

   54.5020 puts D7.2's 44 GPU-h after B-B1, so it does not delay the toolkit. A lower number would run it sooner and
   push the toolkit back.

**Correction, 21 October (afternoon).** While building RD-4e we found that the probe adapter truncates prompts at
256 tokens. Whole-frame recall prompts often exceeded that (42% on T4), so the whole-frame numbers in files 12 and 14
(e.g. T4 0.578; Qwen3-1.7B 0.605) and the reading "whole-frame recall hurts on real chemistry" are partly an artefact.
Selective, routed and definition prompts were not affected, so every primary result stands; E13's amendment 3 rests
on selective recall beating no recall, which is unaffected. RD-4e re-measured whole-frame recall untruncated on 22 October: it matches selective recall (gold frames 0.867 vs 0.872 on
T4), so the realistic pipeline needs no router. Details:
`manuscript/findings/2026-10-21-overnight.md` §10b and `experiments/e14-logic/preregistration.md` §20.

## 6. Housekeeping and inconsistencies found while writing this digest

We left the source documents unchanged; the items below are for whoever maintains them.

**Stale ledger entries.**
- `manuscript/claims.md`:
  - H-A.1 still says "pending (S0 pilot running)";
  - E9.B1 says the quantized advantage "does not grow", although it grew on T7;
  - E9.C2 quotes 0.25 vs 0.19, where the evidence says 0.259 vs 0.189.
- `draft.md` contribution 7 uses E10's superseded wording.
- `reports/R9` says the gain "grows with model size". That holds on T5 only; on real text smaller hosts gain more.

**Superseded statements still in the text.**
- `status-2026-10-08` says scrambled relations "destroy the gain". That holds on T5 only.
- `update-2026-10-12` says binding wins "only for reverse lookup". On T4 it also wins for chaining (09.3).
- The 01b and 01c dossier READMEs keep superseded headline numbers above their own errata.

**Uncommitted reports cited by the summaries.**
- E4 analysis folders: `analysis/opscreen-v1`, `scale-v1`, `scale-v1-toolkit`.
- E9 `report/{t4,t7}-addendum2`, `report/{t5,t4,t7}-understanding`, the working-tree `report/t7`, and
  `report/t5-binding`.
- Most E11 and E12 report folders.

These should be committed, apart from any clinical outputs, which follow the T1c rules.

**An unflagged preliminary result.** The pre-registered understanding primary on T5 held-out terms is −0.024
[−0.030, −0.018]. That is significant and opposite to the prediction, and the affordance items drive it. No document
interprets it yet (06.7).

**Missing analyses.**
- The pre-registered C5dev − C5 contrast is not printed by the toolkit report (03.4).
- The hashed-memory paired contrast is not printed (06.5).
- There is no C5 run on the edit-v2 items, so the pre-registered v2 comparison cannot be made (09.6).

**Stale planning documents.** `resources/plan-improvement/queue.md` and `manuscript/reproducibility.md` predate many
status changes. `gates.md` cites a different commit for E2's run than its manifest records.

Each file's Limitations section records the source disagreements it found, and says which source it followed: always
the most recent.
