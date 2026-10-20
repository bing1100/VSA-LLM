# 11 — E11: read-to-learn — can a model learn a new word from one reading of its definition?

**Overview.** E11 asks whether a trained model can read a term's definition once, turn it into a frame of ontology
edges, write that frame into its store, and use the term later, with the definition gone from the prompt and no weight
change. Exploration 11.1 tests one word at a time against two pre-registered endpoints: invented words (P1, track T5)
and real chemistry (P2, track T4). It also compares the written frame with two text routes (the definition in the
prompt; a few gradient steps on it) and with a dense encoding of the definition. Exploration 11.2 asks what changes when
hundreds of new words compete for a fixed context. In short, a read frame helps a little on invented words (P1 passed),
but it recovers under half of the gold frame's gain, and the model's own choice of relation is worse than a frequency
table. On real chemistry, reading does nothing and even gold frames slightly raise the loss (P2 failed). Text routes win
by wide margins. Frames win only on cost per use, and only once many words compete for the context. The real-text
follow-up on T7-ROOD is queued.

**Terms used throughout.**
- **Tracks.** **T5:** a synthetic company glossary of invented words. **T4:** real chemistry (ChEBI entries and PubMed
  text). **T7:** 2026 MeSH names in 2025–26 PubMed abstracts. **T7-ROOD:** T7 retrained with every abstract that names a
  held-out record removed, so held-out names are truly unseen.
- **Models** (SmolLM2-360M unless stated):
  - **C5:** fine-tuned with the method, which adds a vector composed from the term's ontology frame at the term's last
    subtoken;
  - **C0′:** fine-tuned on the same text without the method;
  - **P0:** the untouched pretrained model; **C2:** a free per-term vector table;
  - **C6d:** a same-site arm whose vector is the frozen host's encoding of a text (the "definition encoder").
- **Frame:** a term's set of (relation, partner) edges, e.g. (*owned by*, *the Zash Team*). **Gold frame:** the
  ontology's own frame.
- **Readers** turn a definition into a frame. A model-free concept finder first marks which known concepts the text
  names.
  - **oracle:** the gold frame (upper bound);
  - **stated:** the gold edges whose partner the text names (the readable ceiling);
  - **type prior:** each found concept gets the relation it most often fills in the ontology;
  - **pattern:** cue phrases ("owned by") and Hearst patterns;
  - **linker** (pre-registered primary): for each found concept, the trained model picks the relation that makes the
    definition most likely, and keeps the edge only if it beats no row (the "self-test");
  - **linker-joint:** the same choice made inside the type prior's frame (declared after the smoke test; secondary);
  - **host:** the same model, prompted to write "relation: partner" lines;
  - **teacher:** Claude; built but not approved for E11, so not run;
  - **random:** the gold frame's shape with frequency-weighted wrong partners; **linker-random:** the linker frame's
    shape with wrong partners; **none:** no row.
- **Routes.** **Implicit:** the frame is composed into the term's vector; this is E11's only frame route. **Explicit:**
  the frame is written as text lines before the question (E12, E14). **Text routes:** the definition in the prompt
  ("in context"), or Adam steps on the definition ("gradient"), compute-matched to the linker's forward passes.
- **Item tests** (definition not in the prompt unless stated): **property**, pick the term's partner among 5 candidates
  (chance 0.20); **entailment**, a true against a corrupted statement (chance 0.50); **paraphrase**, the same choice
  across rewordings.
- **PMI scoring:** a candidate's log-probability after the item prompt, minus its log-probability after the same prompt
  with the term's name replaced by "this". The highest wins. In context conditions, the null prompt also carries the
  definition.
- **Loss after the term:** next-token loss on the 8 tokens after each occurrence of a read term, in evaluation documents
  other than the term's own definition document.
- **Sign conventions.** Accuracy differences are a − b, **positive = better**. Loss differences are a − b in percent,
  **positive = the loss went up (worse)**.
- **Statistics:** seeds 1–3 pooled; paired percentile bootstrap with 2,000 resamples (over items, or over term clusters
  for loss); Holm over P1 and P2; secondaries unadjusted.

---

# 11.1 — One word, one reading: does a read frame help? (P1, P2)

**Status:** partly done · **When:** pre-registered 2026-10-07, amended 2026-10-07 and 2026-10-09 (all before any run);
pooled report 2026-10-11; T7-ROOD jobs queued · **Sources:** `experiments/e11-read-to-learn/preregistration.md`;
`experiments/e11-read-to-learn/report/report.md`; per-run reports
`experiments/e9-retrofit/runs/{t5,t4,t7}/SmolLM2-360M-full-*-s*/e11-*/report.md`; `manuscript/update-2026-10-12.md`
(E11-1 to E11-3)

## Research gap
E9 showed that a hand-written gold frame for an invented word raises property accuracy by 4–7 points. We had not tested
whether a trained model can write a usable frame itself, from one reading. E10 had no real-model test of a self-chosen
relation or of a self-acceptance test. Nor did we know whether an explicit write beats the known text routes (in-context
editing, a gradient step) at matched compute.

## Intuition
The channel turns any frame into a vector. A reader that maps a definition to edges would then give the model a
persistent memory of the word that costs no prompt tokens. Letting the model pick each relation by its own likelihood
would make the reading "self-reflective".

## Methodology
- **Data.**
  - **T5-N:** 300 invented words. Definitions are written from the gold frame; the primary `prose` style uses held-out
    wordings and no training template.
  - **T4-H:** 341 held-out ChEBI entities, read as `<name>: <ChEBI definition>`. The concept finder sees 14.9% of their
    gold partners (100% on T5).
- **Models:** C5 runs every reader; C0′ runs the text routes; C6d runs the encoder (amendment §16).
- **Gradient route:** about 5 (T5) or 11 (T4) Adam steps per term, at lr 1e-4 chosen on 40 dev words (the middle of the
  grid); weights restored after each term.
- **Endpoints:**
  - **P1:** T5-N property, linker − none;
  - **P2:** T4-H loss after the term, linker − none. The 2026-10-09 amendment requires linker to beat none, random and
    linker-random; P2's p is then the largest of the three (intersection–union).
- **Pre-registered readings:**
  - (a) "learns from one reading": P1 > 0, linker − random > 0, and recovery share R = (linker − none) / (oracle − none)
    ≥ 0.5;
  - (b) the same on natural text;
  - (c) self-reflective: linker beats the type prior;
  - (d)–(f): in context, a gradient step or the encoder does as well.

## Experiment
- **Ran:** T5-N and T4-H on C5 and C0′, seeds 1–3; C6d seeds 1–3; T7-H (T7 held-out records) at seed 1.
- **Not run:** SmolLM2-135M, C2, P0, and the T5-H, T4-N and T1-H sets (tiers 2–3, which wait for the author).
- **Queued:** T7-ROOD-H, 11 jobs at 54.4979–54.49792.

## Results
**Primary endpoints:**

| Endpoint | Estimate [95% CI] | p (Holm) | Verdict |
|---|---|---:|---|
| P1: T5-N property, linker − none | **+0.0312** [+0.0246, +0.0376] | 0.0020 | **passed** |
| P2: T4-H loss, linker − none | **+0.13%** [+0.05, +0.22] (worse) | 0.9535 | **failed** |
| P2 component: linker − random | −0.47% [−1.06, −0.10] | | |
| P2 component: linker − linker-random | +0.00% [−0.07, +0.08] | | |

**Readings:**
- **(a) not met.** Linker − random passes (+0.0160 [+0.0081, +0.0241]), but R ≈ 0.44 (+0.031 / +0.070; derived, since
  the report does not print R). The bar was 0.5.
- **(b) failed.** On T4-H even the gold frame raises the loss: oracle − none +0.18% [+0.03, +0.38]. The channel cannot use
  frames for these terms at all.
- **(c) failed.** Linker − type prior on property is −0.0377 [−0.0453, −0.0303]. On ambiguous partners, the linker picks
  the right relation 0.265–0.328 of the time (per seed), against 0.700 for the type prior.
- **(d) not met as worded.** The definition in context beats the frame: context − linker +0.4208 [+0.4075, +0.4344].
  The frame still adds a little on top: context+linker − context +0.0078 [+0.0017, +0.0137].
- **(e) holds:** gradient − linker +0.2734 [+0.2595, +0.2880] on the same C5 model. The frame keeps a locality
  advantage: rows of 2,000 other entries move by at most 1.8e-7. At seed 1, each per-term update raises general-text
  loss by +0.022 nats (T5) and +0.058 to +0.060 nats (T4-H), on C0′ and C5. That is about 1% and 2.5% of the 2.31-nat
  baseline (derived), above E4's 0.5% locality margin.
- **(f) not met:** C6d encoder − C5 linker on property is −0.0232 [−0.0298, −0.0162] (T5) and −0.0150
  [−0.0224, −0.0074] (T4); on T4 loss −0.03% [−0.36, +0.22]. The encoder writes 25–50× more cheaply (73 or 39 forward
  tokens per term, against 1,816 or 1,923).

**Frame quality** (edge F1 against gold; seeds 1–3; T5-N prose, T4-H ChEBI):

| Reader | T5-N F1 | T4-H F1 | T4-H precision |
|---|---:|---:|---:|
| stated | 1.000 | 0.210 | 1.000 |
| type prior | 0.913 | 0.148 | 0.313 |
| host | 0.744–0.747 | 0.124–0.125 | 0.276–0.279 |
| linker | 0.509–0.532 | 0.021–0.027 | 0.047–0.060 |
| pattern | 0.239 | 0.006 | 0.533 |

**Other results:**
- **T4-H item tests barely move:** linker − none on property +0.0055 [−0.0003, +0.0115]; oracle − none +0.0022
  [−0.0041, +0.0085].
- **T7-H** (seed 1): the notes state 2.1% of gold edges, and 371 of 487 linker frames are empty. Linker − none on loss is
  +0.03% [−0.00, +0.08].

**Cross-reference (E14, file 14, explorations 14.4–14.5):** E14 later wrote the same E11 frames as text lines before the
question (explicit route), restricted to the asked relation. With the linker's frames, property accuracy then rises from
0.614 to 0.661 on T4 and from 0.190 to 0.468 on T5 (RD-4b).

## Limitations
- **Untracked numbers.** The pooled report omits R and the frame metrics; we took them from the per-run reports. The
  report folders are not committed to git.
- **Context under-credited.** The PMI null prompt carries the definition. E14 showed this works against context
  conditions, so the in-context advantage is, if anything, understated.
- **T4 is not a clean zero-shot test.** Its held-out terms appear unlinked in training (E9-6). T7-ROOD is the clean test.
- **Two anomalies are unexplained:**
  - the definition in context raises T4-H loss after the term (C5 +0.77% [+0.53, +1.05]);
  - per-term gradient steps raise it by +79.36% [+52.31, +117.48], while general-text loss rises only 0.06 nats.

  The 12 Oct update flags the +79% for a sanity check.
- **Scope.** One host size. The host reader has no pooled contrast. The teacher was not run.
- **Mislabel in the 12 Oct update.** It calls `linker-random` a "random-relation frame". Per pre-registration §16.1,
  `linker-random` keeps the linker's relations and draws wrong partners.

## Next steps
- **Done:** P1, P2, the encoder competitor (T5-N, T4-H) and T7-H at seed 1. "Self-reflective reading" is dropped as a
  claim (12 Oct update).
- **Queued:** E11 on T7-ROOD-H (54.4979–54.49792; projected ≈ 21–22 Oct in the 15 Oct update). Pre-registration §16.1
  expects a null there: the notes name about 2% of the gold partners, so most linker frames will be empty.
- **Proposed, not queued:** tiers 2–3 (T5-H, T4-N, T1-H, 135M). Their commands must be re-printed with `plan`, and the
  author must approve them.
- **Superseded direction:** E14 (14.4–14.5) and E13 amendment 3 test the same frames through selective explicit recall
  instead of the implicit route.

---

# 11.2 — Many words, one reading each (E11-M)

**Status:** done · **When:** pre-registered 2026-10-07 (pre-registration §14); report 2026-10-11 · **Sources:**
`experiments/e11-read-to-learn/preregistration.md` §14; `experiments/e11-read-to-learn/report-many/report.md`;
`manuscript/update-2026-10-12.md` (E11-4)

## Research gap
The single-word smoke test showed that one definition in context beats every frame (property 0.61 against ≤ 0.23). Real
use, though, involves many new words competing for a fixed context. We had not measured where, if anywhere, a
zero-token frame overtakes reading in context.

## Intuition
In-context reading degrades once definitions no longer fit the window. Frames cost nothing per use, and rows do not
interfere. The frame gain should therefore stay flat as the number of words N grows, while the in-context gain falls.

## Methodology
- **Sets:**
  - **T5-M:** 700 invented words (prose definitions, mean 110 tokens), in nested episodes of N = 25 to 400;
  - **T4-M:** the 242 T4-H terms found in evaluation text, N up to 242.
- **Routes:**
  - frames from the oracle, linker, linker-joint or type prior;
  - in context, with the first N definitions cut at 2,048 or 7,168 tokens (only 17–22 T5 definitions fit in 2,048);
  - BM25 retrieval of the top 1 or 3 definitions;
  - sequential compute-matched gradient steps.
- **Models:** C5 runs every route; C0′ runs the text routes. Seeds 1–3.
- **Primary M1:** T5-M, N = 200, property, frame:linker − context at 2,048 tokens; its own family, α = 0.05.

## Experiment
Six T5-M jobs and the T4-M jobs (C5 and C0′, seeds 1–3), plus the pooled report.

## Results
**T5-M property accuracy, C5:**

| N | linker frame − none | context (2,048) − none | linker frame − context (2,048) | oracle frame − context (2,048) | retrieval k=1 − linker frame |
|---:|---:|---:|---:|---:|---:|
| 25 | +0.035 | +0.359 | −0.324 | −0.279 | +0.508 |
| 100 | +0.037 | +0.112 | −0.075 | −0.036 | +0.464 |
| 200 | +0.038 | +0.048 | **−0.0102** [−0.0271, +0.0062] | **+0.0325** [+0.0148, +0.0495] | +0.459 |
| 400 | +0.037 | +0.030 | +0.0075 [−0.0036, +0.0180] | +0.045 | +0.452 |

**Readings:**
- **M1 is null** (p 0.257), so reading M-a ("frames beat context") is not met. Reading M-c ("context still wins") is not
  met either.
- **M-b is met:** the oracle frame beats context at N = 200. The reader is the bottleneck.
- **M-d (crossover N\*, point estimates):** at 2,048 tokens, N\* = 200 for the oracle and 400 for the linker. At 7,168
  tokens, the linker stays below context up to N = 400 (−0.0322 [−0.0440, −0.0209]).
- **M-e is met:** retrieval beats the linker frame by +0.45 to +0.51 at every N.
- **M-f is not evaluable:** the report lacks the forgetting numbers. At N = 200, gradient − linker frame is −0.0153
  [−0.0322, +0.0002].

**Predictions (written before any run):**
- M1 ≈ 0: **passed**.
- Context − none ≈ +0.04 at N = 200 and ≈ +0.02 at N = 400: **passed** (observed +0.048 and +0.030).
- Retrieval stays near the single-definition level: **passed**.
- An oracle crossover only near N ≈ 400: **failed**; it came earlier, at N = 200.

**T4-M:**
- Frames ≈ none at every N: linker −0.006 to +0.007; oracle −0.004 to +0.007.
- On C5, context at 2,048 tokens beats none by +0.09 to +0.15 for N ≥ 50.
- On C5, sequential gradient steps lower property accuracy by 0.21–0.26 for N ≥ 100.

## Limitations
- One word order only.
- The report omits interference, forgetting, cost and the loss after mentions in passages and windows. So the
  pre-registered check "passages and windows agree with the items" cannot be read yet.
- Retrieval queries with the term's own prompt, which favours it.
- The PMI caveat of 11.1 applies to every context route.

## Next steps
- **Done:** T5-M and T4-M at 3 seeds.
- **Proposed:** print the missing interference, forgetting and loss-after-mention numbers from the existing outputs (no
  GPU needed).
- **Closed:** the defensible claim is narrow. Frames are a zero-token way to keep many new words available; retrieval is
  the strongest route.

---

# Sources

- `experiments/e11-read-to-learn/preregistration.md` (§§1–16, including the smoke addenda)
- `experiments/e11-read-to-learn/report/report.md` and `report.json`; `experiments/e11-read-to-learn/report-many/report.md`
- `experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-{C5,C0p,C6d}-s{1,2,3}/e11-t5-new/`;
  `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-{C5,C0p,C6d}-s{1,2,3}/e11-t4-heldout/`;
  `experiments/e9-retrofit/runs/t7/SmolLM2-360M-full-{C5,C0p}-s1/e11-t7-heldout/` (per-run reports and summaries)
- `experiments/e11-read-to-learn/dev/SmolLM2-360M/gradient_lr.json`
- `manuscript/update-2026-10-12.md` (E11-1 to E11-4); `manuscript/update-2026-10-15.md`;
  `manuscript/toolkit-methodology-2026-10.md` §1.3
- Cross-references: `manuscript/findings/2026-10-20-e14-tier-p-gpu.md` §1; `manuscript/findings/2026-10-20-e14-followups.md`
  §1; `manuscript/status/10-20-2026/14-e14-reading-real-text.md` (14.4–14.5)
- Dates: `git log -- experiments/e11-read-to-learn` (0eff4a9 2026-10-07 … a5e5229 2026-10-09)
