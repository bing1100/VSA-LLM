# 05 — E7: self-authored ontologies on pretrained hosts

**Overview.** E7 asks whether a small pretrained model can read text, write ontology frames for the concepts it meets,
keep only the frames that help on held-out text, and then improve through the method. We built its reading corpora and
hidden gold on 2 Oct (05.1). The rest has not run (05.2):
- **D7.1, authoring quality:** queued on 2 Oct at priority 60. It has never reached the GPU; 1,104 pending jobs rank
  ahead of it.
- **D7.2, one self-improvement round, and D7.3, cross-authoring into a from-scratch model:** never queued. Both wait for
  the E4 gate G3, which was never held.

On 8 Oct, decision 63 recast self-authoring as one of three proposal sources for the toolkit's *learn* step. **There
are no E7 results.** Evidence from other experiments (05.3) suggests that small hosts write poor frames while Claude
writes WordNet-like ones:
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

**Status:** D7.1 queued (since 2026-10-02, not reached); D7.2 and D7.3 not queued · **When:** plan 2026-10-03 ·
**Sources:** `resources/plan-improvement/experiments.md` (E7); `experiments/e7-self-authoring/runs/general-v1/{plan/d71.json,
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

## Experiment
- **D7.1:** 9 jobs (discover, author and direct-prompt for each of the three hosts), about 1.1 GPU-h in total. All 9
  have been pending at priority 60 since 2026-10-02. As of 20 Oct, 1,104 pending jobs rank ahead of them (E9, the
  toolkit band 54.496–54.4999, addendum 2).
- **D7.2 and D7.3:** never queued. Both wait for G3 (`queue.md` item 50), and G3 was never held: decision 54 replaced
  the E4 core with reduced screens (file 03, 03.5).
- **Budget committed:** about 40 GPU-h. None has been spent.
- **Smoke runs:** three labelled CPU smoke folders (2 Oct; `smoke/`, no manifests) used slices of about 0.1–0.25M
  tokens or 300 documents. The general smoke ran every stage, D7.0 through the R6 report. They test plumbing and are not
  results.

## Results
None. H-G.1 (self-authored frames beat cm and random), H-G.2 (verification raises precision) and H-G.3 (beats Hearst
and OLLM-style extraction) all remain "pending".

**Change of role (decision 63, 8 Oct).** The toolkit methodology makes self-authoring one of three proposal sources for
the *learn* step, next to decomposing learned vectors and rule closure. Each proposal must pass the same null-calibrated
acceptance test. The *learn* tool (`vsa_embed.learn`) lists E7 cards as an input source.

## Limitations
- Nothing has been measured.
- The design's fairness rests on two plan decisions, the LoRA host and cm's FLOPs accounting (decisions 30, 32).
- `manuscript/draft.md` §5.10 is a table of placeholders.

## Next steps
- **Queued:** D7.1 (about 1.1 GPU-h). It runs only when the queue reaches priority 60.
- **Blocked:** D7.2 and D7.3, on G3.
- **Superseded in part:** the *learn* step of the toolkit (E13) and the E14 reading diagnostics now carry the "can the
  model write its own structure?" question on real tracks (see the E14 status files).

---

# 05.3 — Adjacent evidence: can a model write frames?

**Status:** done elsewhere (E4 phase 1b pilot, E14 RD-1a, E11) and smoke only (addendum 2 §7) · **When:** 2026-10-10
to 2026-10-17 · **Sources:** `experiments/e4-small-lm/preregistration-scale-v1-toolkit.md` §5;
`experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §7; `manuscript/findings/2026-10-17-e14-tier-p-cpu.md`
(RD-1a); `manuscript/update-2026-10-12.md` (E11-1); `manuscript/update-2026-10-15.md`

## Research gap
E7.1 has not run, so we lack its central number: how good are a host's and a teacher's frames? Four other experiments
measured parts of it.

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
- **Awaiting the author:** C5teach, Claude-read frames in a from-scratch 50M model (file 03, 03.4; about USD 73). It is
  the closest relative of E7.3.

---

# Sources

- E7 configs and run folders: `experiments/e7-self-authoring/{general.yaml, devtools.yaml}`;
  `experiments/e7-self-authoring/runs/{general-v1, devtools-v1}/`; `experiments/e7-self-authoring/smoke/` (SMOKE only).
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
