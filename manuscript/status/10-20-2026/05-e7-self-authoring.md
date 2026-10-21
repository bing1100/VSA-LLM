# 05 — E7: self-authored ontologies on pretrained hosts

> **Update, 21 October.** D7.1 ran on 20–21 October with the Claude follow-ups (USD 9.07). **Gate G3 failed:** on the
> general track, neither the host's nor the teacher's edges for masked concepts are more precise than degree-matched
> random ones. D7.2 was not queued, and D7.3 depends on it. Self-authoring is closed on this track (05.2). Full
> record: `manuscript/findings/2026-10-21-overnight.md` §10.

**Overview.** E7 asks whether a small pretrained model can read text, write ontology frames for the concepts it meets,
keep only the frames that help on held-out text, and then improve through the method. We built its reading corpora and
hidden gold on 2 Oct (05.1). D7.1, authoring quality, waited in the queue from 2 Oct until the author moved it to the
front on 20 Oct; it ran on 20–21 Oct and failed the new gate G3 (05.2). D7.2 (one self-improvement round) and D7.3
(cross-authoring into a from-scratch model) were therefore not queued.

On 8 Oct, decision 63 recast self-authoring as one of three proposal sources for the toolkit's *learn* step. **The one
E7 result is negative:** for masked WordNet concepts, the host's edges have precision 0.062 (16 verifiable edges) and
the teacher's 0.069 (277), against a degree-matched random high of 0.243. Evidence from other experiments (05.3)
agrees: small hosts write poor frames, and on real text even Claude's frames are no more precise than the ontology's
own statistics:
- a 360M host added 0 facts from 4 notes in a smoke test;
- on real chemistry definitions the host reader recovered 8.0% of gold edges, against 9.7% for a type prior;
- Claude's frames agree with WordNet at precision 0.78 and recall 0.75.

**Terms used throughout.**
- **C5:** the full method, a gated vector composed from a term's ontology frame (relation–filler edges) over a shared
  dictionary and added at the term's last subtoken. **HRR** (holographic reduced representation) is its binding
  operator. **C0′:** the pretrained host fine-tuned without the method.
- **M5:** the self-authoring loop. The host finds candidate concepts, writes frames, and verifies each edge by its
  held-out utility. Accepted edges enter the linker and dictionary, and the host trains on them.
- **Host:** the pretrained model (SmolLM2-135M or -360M, Qwen2.5-0.5B). **Teacher:** Claude (`claude-opus-5-5`) called
  headless with `claude -p`.
- **Masked concepts / hidden gold:** WordNet entries removed from the visible ontology. Their gold frames are kept apart
  and used only for auditing.
- **D7.0–D7.3:** the work packages (corpora, authoring quality, round 0 → 1, cross-authoring).
- **Controls:**
  - **cm:** compute-matched continued pretraining (CPT);
  - **EntiGraph:** synthetic-text CPT;
  - **SPA:** synthetic QA augmentation;
  - **notes:** self-written free-text notes through the same filter;
  - **verbal:** the verified frames rendered as text, with no channel;
  - **selfnv:** self-authoring without verification;
  - **random:** random frames of matched degree.
- **Signs:** for loss, negative = better. Precision, recall and accuracy: higher = better.

---

# 05.1 — Reading corpora and hidden gold (D7.0)

**Status:** done · **When:** 2026-10-02 · **Sources:** `experiments/e7-self-authoring/{general,devtools}.yaml`;
`experiments/e7-self-authoring/runs/{general-v1, devtools-v1}/{report.md, summary.json, manifest.json}`;
`resources/plan-improvement/execution.md` (decisions 31–34)

## Research gap
To score self-authored frames, we need concepts the host has never seen in its ontology, with gold frames it cannot
see. We also need reading text that no host or C3 model was trained on.

## Intuition
Mask a fifth of the linked WordNet entries and keep their frames hidden. Then give the host fresh text that mentions
them. Whatever correct structure it writes, it must have read.

## Methodology
- **General track:** FineWeb-Edu documents in C3's shard order, SmolLM2 tokenizer, terms of 2 or more subtokens.
  - **Masking:** 20% of the WordNet entries linked in the SmolLM2 host corpus, with the C3 holdout excluded. Because
    entries share concepts, an 11.9% draw yields 20.2% of eligible entries (decision 31).
  - **Reading slice:** starts at document 1,070,634, right after the GPT-2 C3 training corpus. It is disjoint from
    every C3 and host training corpus (decision 34).
- **Dev-tools track:** the documentation of the synthetic private library (C6 v1) is the reading text. 20% of its
  symbols are masked, and their schemas are the gold.
- **Clinical track** (PubMed and MIMIC-IV with a held-out SNOMED CT subgraph): planned in experiments.md, but no corpus
  was prepared.

## Experiment
Two preparation runs on 2 Oct, from clean commit `903b9cd`.

## Results

| General track | Value |
|---|---:|
| masked concepts / entries | 26,368 / 15,890 |
| visible aliases | 117,680 |
| aliases excluded (C3 holdout, C6 held-out) | 8,118 |
| channel-training corpus (round 0) | 60.0M tokens, 56,394 documents |
| replay corpus (round 1) | 6.25M tokens |
| reading corpus D_read | 25.0M tokens, 1.37M term matches |
| held-out test contexts | 4.0M tokens |

| Dev-tools track | Value |
|---|---:|
| masked symbols | 64 of 321 eligible |
| visible / excluded aliases | 293 / 57 |
| reading documents | 2,950 |

The gold frames live only under `~/data/vsa-llm/e7/{general,devtools}-v1/gold/` and are not committed.

## Limitations
- C6's alias table cannot link snake_case symbols, so a dev-tools channel round first needs a linker fix (noted in
  `devtools.yaml`).
- No clinical reading corpus exists.
- The masked set is WordNet. Hosts may know many masked concepts from pretraining, which is a contamination caveat for
  the general track.

## Next steps
- **Done.** The corpora are ready for D7.1.
- **Not planned:** a clinical E7 corpus.

---

# 05.2 — Authoring quality, one self-improvement round and cross-authoring (D7.1–D7.3)

**Status:** D7.1 done; G3 failed 21 October; D7.2 and D7.3 not queued; self-authoring closed on this track · **When:**
plan 2026-10-03; D7.1 ran 2026-10-20 to 2026-10-21 ·
**Sources:** `experiments/e7-self-authoring/preregistration-2026-10-20-reschedule.md`;
`experiments/e7-self-authoring/runs/general-v1/gate/g3-d72.json`; `manuscript/findings/2026-10-21-overnight.md` §10;
`resources/plan-improvement/experiments.md` (E7); `experiments/e7-self-authoring/runs/general-v1/{plan/d71.json,
round.json}`; `.jobs/e7-general-v1-*.json` (read only); `manuscript/paper-readiness.md`; `manuscript/claims.md`
(H-G.1–H-G.3); `manuscript/toolkit-methodology-2026-10.md` §1.2, §3.2, §5; `resources/plan-improvement/execution.md`
(decisions 30, 63)

## Research gap
Self-improvement from self-written knowledge has precedents: Evontree, SEAL and Active Reading. No work has a small
model write *typed frames* for its own poorly modelled terms, accept edges by held-out utility, and consume them through
a non-textual channel, while beating compute-matched and synthetic-text controls.

## Intuition
If structuring what it reads helps the model, then self-written frames should beat simply reading the text again (cm)
and frames of the same size with random content. If self ≈ cm, structure adds nothing. If self ≈ random, the gain is
just capacity.

## Methodology
As planned:
- **E7.1 (authoring quality):** the hosts read D_read and write frames for candidate concepts found by a surprisal
  screen.
  - **Metrics:** edge precision, recall and F1 against the hidden gold; relation-type accuracy; discovery recall;
    LLM-judged plausibility of edges with no gold.
  - **Baselines:** Hearst patterns, direct prompting (OLLM-style), the teacher, random frames.
- **E7.2 (self-improvement):** round 0 → 1 on SmolLM2-360M, 25M reading tokens, 3 seeds.
  - **Arms:** gold → host, self → self, selfnv, teacher → host, random, cm, EntiGraph, plus notes, SPA and verbal (added
    after the literature re-check).
  - **Primary endpoint:** loss on held-out contexts of authored and masked-gold concepts, against cm.
  - **cm's budget** counts discovery, authoring and verification FLOPs, about 27M tokens on top of 25M (decision 32).
- **E7.3 (cross-authoring):** the best host's ontology becomes the linker ontology of a 50M from-scratch model, against
  the curated ontology and no channel.
- **Gate (G5):**
  - self → self beats cm and random on the primary endpoint, intervals excluding 0 over 3 seeds;
  - locality within 0.5%;
  - verification improves edge precision over selfnv.
- **Round settings** (`round.json`): C5 with HRR. Verification uses a utility quantile of 0.05, at least 6 validation
  contexts and at most 8 edges per concept. Decision 30 moves the host from frozen to LoRA and picks the channel
  condition at G3.
- **Gate G3 (the reschedule of 20 Oct):** each author's edge precision on the masked gold is compared with the upper
  bound of a degree-matched random author's precision (the "random high"). An arm passes only with at least 30
  verifiable edges, coverage ≥ 0.1 and precision above the random high. D7.2 is queued only if an arm passes.

## Experiment
- **D7.1:** 9 jobs (discover, author and direct-prompt for each of the three hosts), about 1.1 GPU-h in total. They
  waited at priority 60 from 2 Oct, behind 1,104 jobs, until the reschedule of 20 Oct moved them to 54.4959993. They
  ran on 20–21 Oct, followed by the CPU follow-ups: Hearst patterns, random frames, and the Claude teacher on 999
  concepts (USD 9.07).
- **D7.2 and D7.3:** not queued. The reschedule replaced the E4 gate G3 (never held; decision 54, file 03, 03.5) with
  an E7 gate G3 on D7.1's output, which failed.
- **Budget:** D7.1's 1.1 GPU-h and the follow-ups' USD 9.07 were spent; D7.2 (about 44 GPU-h) was not queued.
- **Smoke runs:** three labelled CPU smoke folders (2 Oct; `smoke/`, no manifests) used slices of about 0.1–0.25M
  tokens or 300 documents. The general smoke ran every stage, D7.0 through the R6 report. They test plumbing and are not
  results.

## Results
**Gate G3 (21 October): FAIL on both arms.**

| Arm | Masked concepts authored | Verifiable edges | Precision [95% CI] | Random high | Coverage | Verdict |
|---|---:|---:|---|---:|---:|---|
| self (Qwen2.5-0.5B) | 64 | 16 | 0.062 [0.011, 0.283] | 0.243 | 0.282 | FAIL (too few edges; not above random) |
| teacher (Claude) | 64 | 277 | 0.069 [0.044, 0.105] | 0.243 | 0.751 | FAIL (not above random) |

- The self-authoring host writes almost no edges for masked concepts: 36 edges over 26 concepts, 16 of them
  verifiable against the gold (findings §10).
- The teacher's edges on this general WordNet track are no more precise than degree-matched random ones. Its recall
  of the masked gold is 0.232 (19 of 82).
- Baselines: Hearst patterns 0 of 8 edges right; direct prompting of SmolLM2-360M, precision 0.049 on 833 edges.
- Discovery recall at 1,000 candidates is 0.010–0.014 for the three hosts.
- D7.2 was not queued; D7.3 depends on it. The Claude judge (cap USD 50) is informational and not part of G3. A
  session restart interrupted it at USD 22.66, and it resumed from its cache.
- H-G.1 (self-authored frames beat cm and random), H-G.2 (verification raises precision) and H-G.3 (beats Hearst and
  OLLM-style extraction) were "pending" when this file was written. The ledger should now record H-G.3's
  authoring-quality part as failed on the general track, and H-G.1 and H-G.2 as untested, since D7.2 never ran.

**Change of role (decision 63, 8 Oct).** The toolkit methodology makes self-authoring one of three proposal sources for
the *learn* step, next to decomposing learned vectors and rule closure. Each proposal must pass the same null-calibrated
acceptance test. The *learn* tool (`vsa_embed.learn`) lists E7 cards as an input source.

## Limitations
- D7.1 measured authoring quality only. The self → self round (D7.2) never ran, so H-G.1's primary endpoint (loss
  against cm) is untested rather than refuted.
- The self arm has 16 verifiable edges, so its precision interval is wide (0.011–0.283). The verdict rests on the edge
  count as much as on the precision.
- Hosts may know masked WordNet concepts from pretraining (05.1). That would favour the authors, not hurt them.
- The design's fairness rests on two plan decisions, the LoRA host and cm's FLOPs accounting (decisions 30, 32).
- `manuscript/draft.md` §5.10 is a table of placeholders.

## Next steps
- **Closed on this track:** G3 failed on 21 October; D7.2 and D7.3 are not queued.
- **Informational:** the Claude judge of authored edges, resumed from its cache. It does not change G3.
- **Carried by other experiments:** the *learn* step of the toolkit (E13) and the E14 reading diagnostics carry the
  "can the model write its own structure?" question on real tracks (see the E14 status files). Their evidence agrees
  with G3: on real text, frames written by a model, small or large, are not more precise than the ontology's own
  statistics (RD-11, RD-3, C5teach).

---

# 05.3 — Adjacent evidence: can a model write frames?

**Status:** done elsewhere (E4 phase 1b pilot, E14 RD-1a, E11) and smoke only (addendum 2 §7) · **When:** 2026-10-10
to 2026-10-17 · **Sources:** `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md` §5;
`experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §7; `manuscript/findings/2026-10-17-e14-tier-p-cpu.md`
(RD-1a); `manuscript/update-2026-10-12.md` (E11-1); `manuscript/update-2026-10-15.md`

## Research gap
Before D7.1 ran (05.2) we lacked its central number: how good are a host's and a teacher's frames? Four other
experiments measured parts of it, and they agree with D7.1.

## Intuition
If small hosts cannot write usable frames even from definitions, E7.2's self → self arm reduces to the teacher → host
arm. The plan foresaw this risk (tasks.md: "E7 reduces to teacher → host").

## Methodology
- **Teacher on WordNet glosses** (E4 phase 1b pilot): Claude writes WordNet-style facts for 400 C3 entries (829 senses)
  from each sense's lemma, part of speech and gloss. We score them against WordNet.
- **Host and readers on real definitions** (E14 RD-1a, CPU): 341 held-out ChEBI definitions (T4 chemistry) against
  2,744 gold edges. The readers are E11's string linker, the host reader and a type prior.
- **Host relation choice** (E11-1, T5 synthetic glossary, SmolLM2-360M C5, 3 seeds): the model picks each relation by
  which choice makes the definition most likely.
- **Host and teacher reading MeSH notes** (addendum 2 §7, smoke): the T7 360M C5 model, or Claude, reads MeSH record
  notes and writes facts.

## Experiment
- Teacher pilot: 34 calls plus 1 calibration call, USD 2.48 (10 Oct).
- E14 RD-1a: CPU (17 Oct).
- Addendum 2 smokes: host 4 notes (CPU); teacher 2 notes, 1 call, USD 0.06 (before 15 Oct).

## Results
- **Teacher (Claude) on glosses:**
  - 97.3% of 2,710 proposed edges map onto the ontology;
  - 5.06 edges per entry against WordNet's 5.20;
  - agreement with WordNet (precision / recall) 0.78 / 0.75 on all edges, 0.59 / 0.56 on pointer edges;
  - meronyms under-produced (recall 0.28–0.33), entailment never produced.
- **Readers on real chemistry definitions:** only 11.7% of gold edges are stated in the text. The linker recovers
  1.7%, the host reader 8.0% and the type prior 9.7%.
- **Host relation choice (E11-1):** the model's own choices are worse than a simple type-frequency table. The update
  concludes that "the 'self-reflective' reading fails".
- **Host reading MeSH notes (SMOKE):** 0 edges from 4 notes; the completions repeat a relation label or echo the prompt.
  The teacher smoke added 1 edge from 2 notes. The host stage is gated (it trains only if at least 10% of entries gain
  an edge) and is expected to be skipped.

## Limitations
- None of these is E7.1:
  - the pilot reads glosses, not running text;
  - RD-1a and E11 read definitions;
  - the MeSH reads are smokes on 2–4 notes.
- They measure agreement or recall against gold, not usefulness through the channel.
- The host results come from SmolLM2-360M only. Qwen2.5-0.5B, an E7 author, is untested.

## Next steps
- **Pending in the queue (E9):** the teacher enrichment of T7-ROOD, the clean held-out track of new 2026 MeSH names.
  Claude reads MeSH notes under a USD 60 cap at priority 54.49995. The host-read stage waits behind its 10% gate.
- **Done (21 October):** C5teach, Claude-read frames in a from-scratch 50M model (file 03, 03.4; USD 68.05), the
  closest relative of E7.3. At one seed it changes the loss inside the terms it describes and nothing after them.

---

# Sources

- E7 configs and run folders: `experiments/e7-self-authoring/{general.yaml, devtools.yaml}`;
  `experiments/e7-self-authoring/runs/{general-v1, devtools-v1}/`; `experiments/e7-self-authoring/smoke/` (SMOKE only).
- Reschedule and gate: `experiments/e7-self-authoring/preregistration-2026-10-20-reschedule.md`;
  `experiments/e7-self-authoring/runs/general-v1/gate/g3-d72.json`; `manuscript/findings/2026-10-21-overnight.md` §10.
- Queue state (read only): `.jobs/e7-general-v1-*.json`.
- Plan and decisions: `resources/plan-improvement/experiments.md` (E7); `resources/plan-improvement/execution.md`
  (decisions 30–34, 54, 63; TK-L); `resources/plan-improvement/queue.md` (item 50); `resources/plan-improvement/tasks.md`
  (D7 risks).
- Ledger, draft and audits: `manuscript/claims.md` (H-G.1–H-G.3); `manuscript/draft.md` §5.10;
  `manuscript/paper-readiness.md`; `manuscript/toolkit-methodology-2026-10.md`; `manuscript/related-work-recheck-2026-10.md`.
- Adjacent evidence: `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md` §5;
  `experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §7;
  `manuscript/findings/2026-10-17-e14-tier-p-cpu.md`; `manuscript/update-2026-10-12.md` (E11-1);
  `manuscript/update-2026-10-15.md`.
- Dates: `git log -- experiments/e7-self-authoring`; run manifests.
