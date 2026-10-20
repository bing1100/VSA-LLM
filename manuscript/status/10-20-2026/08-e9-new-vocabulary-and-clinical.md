# 08 — E9 tracks: new vocabulary (T7, T7-ROOD), clinical notes (T1c), Wikidata (T8) and the other domains

This file covers the E9 test domains beyond the synthetic glossary and chemistry: where each stands, what it showed, and what is queued. T7 was built to be the real domain whose words are new to the host. It was not: plain fine-tuning lowers the loss after its terms by only 7–10%. The method helps a little after rare and long names, and it *hurts* after held-out names, because those names occur unlinked in the training text. T7-ROOD removes every training document that names a held-out entry. It is the pre-registered clean zero-shot test on real text, and all 253 of its jobs are still pending on 20 October. Clinical notes (T1c, licensed; aggregate numbers only) repeat the chemistry pattern at 3 seeds: a gain after and inside long terms, larger on the smaller host, and none for held-out or rare terms. The clinical coding task (T1c-F) finished in the queue on 16 October, but no committed document reports its results. T1c-ROOD is queued. T8 (Wikidata entities in web text) is built and pre-registered, with all 117 jobs pending; its pre-registration expects little gain, because the host already knows these entities. T1-open and WordNet are null at seed 1. T2 is built but has no E9 runs; T3 and T6 failed the feasibility criterion.

**Codes used in this file** (as in `manuscript/update-2026-10-12.md`, "How to read the numbers", and `src/vsa_embed/experiments/e9_plan.py`):

| Code | Meaning |
|---|---|
| host | the pretrained language model: SmolLM2-135M or -360M (full fine-tuning), Qwen3-1.7B (LoRA r 64) |
| entry, frame | an ontology record (e.g. a MeSH supplementary record); its frame is the set of its relation–filler edges |
| **C5** | the method: a vector composed from the entry's frame over a shared, jointly trained dictionary, added at the last subtoken of each linked multi-token term. Binding: **HRR** (holographic reduced representation, circular convolution) |
| **P0** | the original host, not trained further |
| **C0′** | the host fine-tuned on the same text and tokens without the method; the fair baseline |
| **C2** | a free lookup table, one learned vector per entry, no ontology, matched to C5's parameters |
| C5sh | C5 with shuffled frames: every entry reads another entry's frame |
| C5ut / C5tr | C5 without binding (a plain sum of fillers) / with translation (relation added, x + r) |
| C6d / C6gi | a frozen per-entry vector through a trained, parameter-matched projector at C5's position: the definition encoder (the frozen host's encoding of the frame written as text) / TransE trained without the held-out entries, which are folded in afterwards |
| held-out terms | kept out of the method's training; at test time their vector is composed from the frame alone ("zero-shot"). On T7, T4 and T1c they still occur in the training text, unlinked. On T7-ROOD, T8 and T1c-ROOD they do not ("ROOD": really out of distribution, HRRBERT's protocol) |
| unseen / rare / long terms | in the ontology but absent from the training text / seen 1–9 times / three or more subtokens |
| T1–T8 | tracks. **T1-open**: MeSH descriptors in PubMed; **T1c**: clinical notes (MIMIC-III, licensed) with SNOMED CT; **T1c-F**: assignment of rare ICD (International Classification of Diseases) codes from T1c's discharge summaries; **T1c-ROOD**: T1c's clean held-out test; **T2**: developer tools; **T3**: product catalogues; **T4**: chemistry (ChEBI, PubMed); **T5**: synthetic company glossary; **T6**: legal (EuroVoc, EU law); **T7**: PubMed 2025–26 abstracts with 2026 MeSH supplementary names; **T7-ROOD**: T7 with held-out names removed from training; **T8**: Wikidata entities in FineWeb-Edu; **WordNet**: general web text (negative control) |
| novelty index | C0′ − P0 after a track's terms. Large = the words are new to the host |
| ℓ_min | the minimum number of subtokens for a term to be linked (2 here) |

**Numbers.** Unless stated, a number is the relative change in next-token loss on the 8 tokens after a term, C5 − C0′. **Negative means better.** Values pool 3 training seeds, with 95% cluster-bootstrap intervals over evaluation windows. `*` = significant after Holm correction. "Seed 1" marks a single run. "Preliminary" marks numbers from generated reports that no committed summary has checked yet.

---

# 08.1 — New vocabulary: T7 (2026 MeSH names) and the T7-ROOD design

**Status:** partly done (T7 at 360M, 3 seeds, and at 135M; T7-ROOD queued, not started) · **When:** T7 built 2026-10-07; T7-ROOD built and pre-registered 2026-10-08, amended 2026-10-09 (decision 64) and 2026-10-11 (C6gi, in-context glossary); T7 results 2026-10-10/11 · **Sources:** `experiments/t7-new-vocabulary/preregistration-rood.md`; `experiments/t7-new-vocabulary/ROOD.md`; `reports/R9-retrofit-quantization.md` ("T7"); `manuscript/status-2026-10-08.md` §2; `manuscript/update-2026-10-12.md` (E9-7); `manuscript/update-2026-10-15.md` §5; `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §§3–4

## Research gap
Every real-text gain so far was small. The 8 October status named the missing piece: a real domain whose vocabulary is genuinely new to the host, with a clear gain at 3 seeds. T5 alone cannot carry a paper.

## Intuition
Names that entered MeSH only in 2026 should be new to a host pretrained earlier. Removing every training document that names a held-out entry (T7-ROOD) leaves the composed vector as the host's only source for what such a name means.

## Methodology
- **T7:** 8,184 MeSH 2026 supplementary concept records (drugs, chemicals, diseases, organisms).
  - Frames run over 3,295 descriptor dictionary entries: 7 relation types, 5.2 edges per entry, 75% of them tree or record-class codes.
  - Names were chosen as absent from 300,000 FineWeb-Edu documents.
  - Text: PubMed 2025–26 abstracts that mention the names, mixed 50/50 with FineWeb-Edu (57.27M SmolLM2 tokens).
  - 908 held-out entries; evaluation on 4,096 `eval-pubmed` windows; no unseen stratum.
- **T7-ROOD changes one thing:** every training document that names a held-out record (3,647 candidate names, whole-word match) is dropped. That removes 15,530 of 68,264 mention-bearing abstracts (6.76M tokens), refilled from 16,347 abstracts that mention no selected name.
  - Held fixed: ontology, alias table, held-out set, evaluation windows, token budget. Leakage audit: 0 training documents contain a held-out name.
  - Power: 5,515 held-out occurrences of 302 entries with ≥ 5 occurrences in the windows.
  - Side effect, stated in advance: seen entries lose 6.7% of their training spans, which works against C5.
- **Pre-registration (before any T7 or T7-ROOD result):**
  - **P1:** C5 − C0′ < 0 after held-out terms on T7-ROOD (360M, seeds 1–3, Holm).
  - **P2:** the T7-ROOD gain is at least T7's (paired by window).
  - **P3:** C0′'s own gain over P0 after held-out terms is smaller on T7-ROOD.
  - **Rule (a), decision 64:** the gain counts as ontology-specific only if C5 − C5sh < 0.
  - **Rule (b), decision 64:** "better than standard new-word vectors" only if C5 − C6d < 0.
  - **Addendum 1:** C5 − C6gi (prediction: C5 ahead or tied) and the in-context glossary.
- **Decision readings, fixed in advance:** a significant T7-ROOD gain after a null T7 would mean text exposure masked the method, and T7-ROOD would carry the real-world claim. Both null would mean the E13 learning cycle (read → learn → write) runs on T5 only. The table assumed T7 would show a gain or a null; T7's significant *cost* matches no row exactly.

## Experiment
- **T7, done:** P0, C0′, C2, C5, C5sh and C6d at 360M, seeds 1–3, plus four further arms analysed in file 06 (a wide free table, link dropout, a hybrid and a hashed memory); C0′, C2 and C5 at 135M, seeds 1–3.
- **T7-ROOD, queued (253 jobs, none started on 2026-10-20):** 360M P0, C0′, C2, C5, C5sh and C6d at 54.497; C6gi at 54.49705; evaluations 54.4971; reports 54.4973; in-context glossary 54.49736; the P2/P3 paired comparison 54.4974; Qwen3-1.7B (seed 1) 54.4977; E11 and E12 follow-ups 54.4979–54.49795; 135M base runs and richer-ontology arms 54.499935–54.49999 (06.12).

## Results
T7 (360M, 3 seeds):

| Stratum | C5 − C0′ | C2 − C0′ | Novelty (C0′ − P0) |
|---|---|---|---|
| after held-out | **+0.48% [+0.35, +0.62]*** (worse) | −0.02% | −10.2% [−10.6, −9.8] |
| after rare | −0.50% [−0.65, −0.36]* | −0.18% | −7.2% |
| after long | −0.57% [−0.62, −0.51]* | −0.46% | −9.9% |
| inside | −0.28% [−0.34, −0.22]* | −0.25% | −34.1% |

- **The recorded prediction failed** (decision 55: a gain at least T4's).
- **Two causes:** the held-out names occur unlinked in 22.7% of training abstracts, and only the names themselves are new to the host (inside −34.1%; the text after them only −10%).
- **Copying plays no role:** frame fillers carry about 0% of the gain.
- **Every term-specific vector costs after held-out names** (C5sh +0.31%*, C6d +0.26%*). The free table, which injects its mean row, is neutral.
- **At 135M** (3 seeds, preliminary) the held-out cost disappears (+0.10% [−0.08, +0.27]), and the gains after rare and long terms roughly double (06.11).
- **Quantization and editing on T7:** see 07.1 and 07.2.
- **T7-ROOD:** no result yet. P1–P3 and rules (a)–(b) are open.

## Limitations
- T7's novelty premise was wrong. T7-ROOD fixes the exposure confound, but not the low novelty of the surrounding text.
- T7-ROOD's refill text has no linked spans, so seen entries get fewer method updates.
- **Sources disagree on timing.** `update-2026-10-12.md` dated T7-ROOD ≈ 20–21 Oct and `update-2026-10-15.md` ≈ 19–20 Oct. On 2026-10-20 nothing has started: the E14 band (reading and logic experiments, priority 54.4959981) and the Qwen3 T4 evaluations (54.4961–54.4963) run first.

## Next steps
- **Queued:** all of the T7-ROOD block above.
- **Queued:** Qwen3-1.7B on T7 v1 (54.4975).
- **Done:** T7 at 135M, seeds 2–3 (to be written into R9).

---

# 08.2 — Clinical notes (T1c), aggregate results only

**Status:** partly done (language-model loss at 3 seeds, both hosts; coding task finished in the queue but unreported; T1c-ROOD queued) · **When:** seed 1 by 2026-10-12; seeds 2–3 by 2026-10-15; T1c-F jobs finished 2026-10-16; T1c-ROOD pre-registered 2026-10-08 · **Sources:** `manuscript/update-2026-10-12.md` (E9-11); `manuscript/update-2026-10-15.md` §6; `manuscript/status-2026-10-08.md` (§2 and frame swap); `manuscript/toolkit-methodology-2026-10.md` (holdout H2, T1c-F)

**Data rule.** T1c uses licensed MIMIC-III notes. This section uses only aggregate numbers from committed manuscript files. We read no note text and no MIMIC-derived file, and we did not open the uncommitted T1c reports.

## Research gap
T1c is a third real domain and the closest to a medical application. Only there does the plan have a downstream task (rare ICD codes), not just next-token loss.

## Intuition
Clinical note style is new to a web-pretrained host, so the method might help as it does on chemistry. The coding task asks whether a loss gain matters for a real decision.

## Methodology
- **Data:** MIMIC-III notes linked to SNOMED CT.
- **Hosts and arms:** SmolLM2-135M and -360M; P0, C0′, C2 and C5; seeds 1–3; the E9 recipe and strata.
- **T1c-F:** assign rare ICD codes from discharge summaries, with and without the method. 382 codes are held out as labels only; the admissions stay in training.
- **T1c-ROOD (holdout H2):** train on MIMIC-III (ICD-9 era) and test on MIMIC-IV ICD-10 admissions. ICD-10 codes get frames from SNOMED through the UMLS (Unified Medical Language System) map. ICD-10-only codes never occur in the training data (HRRBERT's "ROOD-unseen" protocol). Decision 64 adds the C5sh and C6d reading rules.
- **Pre-registration:** the committed manuscript records no directional loss prediction for T1c. The frame-swap amendment registered none either.

## Experiment
- Language-model loss: both hosts, seeds 1–3.
- Linked frame swap: seed 1.
- T1c-F: the 360M encode, train and analysis jobs (P0, C0′ and C5, seeds 1–3, plus one P0 variant) finished in the queue by 2026-10-16, after a command-line bug in the second analysis pass was fixed.

## Results

| T1c, 3 seeds | 360M | 135M |
|---|---|---|
| after long terms | −0.94%* | −1.84%* |
| inside terms | −0.53%* | −1.23%* |
| after unseen terms | −0.48%* | +0.02% |
| after held-out terms | −0.02% | −0.13% |
| after rare terms | −0.02% | +0.04% |
| text without terms | −0.06%* | −0.10%* |

- **The seed-1 picture holds:** a reliable gain after and inside long terms, about twice as large on the smaller host. There is no gain after held-out or rare terms; at 360M there is a small gain after unseen terms.
- **Novelty is high:** −37 to −41%. The note style is new to the host.
- **The free table does nothing** (C2 − C0′ ≈ 0, seed 1).
- **The host relies on each term's own vector** (linked frame swap, seed 1, after long terms): another term's vector costs +3.91%, no vector +3.21%.
- **T1c-F:** no committed result exists.

## Limitations
- The 3-seed numbers come without intervals in the manuscript.
- The unseen and rare strata are small (about 900 and 5,000 target tokens), so their intervals are wide.
- R9 has no T1c section yet.
- **Newer numbers supersede older ones.** At seed 1 (`update-2026-10-12.md`), the gain after unseen terms at 360M was −0.39% (n.s.); at 3 seeds (`update-2026-10-15.md`) it is −0.48%*.

## Next steps
- **Done, unreported:** T1c-F. Its results need a committed aggregate write-up.
- **Queued:** T1c-ROOD, 63 jobs at priority 54.4998–54.49984. The updates dated it ≈ 25–26 Oct.
- **Proposed:** the T1c section of R9.

---

# 08.3 — Wikidata entities (T8): status

**Status:** queued (built and pre-registered; no run) · **When:** built and pre-registered 2026-10-08; amended 2026-10-09 (decision 64) · **Sources:** `experiments/t8-wikidata/preregistration.md`; `manuscript/update-2026-10-12.md`; `manuscript/update-2026-10-15.md` ("Next experiments"); `manuscript/toolkit-methodology-2026-10.md` (M1, M5)

## Research gap
The toolkit's general relation benchmarks (BEAR, LRE, TwoHopFact, PopQA, Entity Inferences) are built on Wikidata entities, while our ontologies were MeSH, ChEBI and SNOMED. T8 gives those entities frames, so that the benchmarks can be scored with a trained store. It is also an E9 track with the strict document-exclusion holdout.

## Intuition
If composition works zero-shot, a held-out entity that never appears in the fine-tuning text should still benefit from a vector composed from its Wikidata statements.

## Methodology
- **Ontology:** a Wikidata snapshot of 2026-10-08 (CC0). 20,000 entities named ≥ 5 times in FineWeb-Edu training text; 88 relations, 8,192 filler dictionary entries; 7.2 edges per entity.
- **Holdout:** 1,680 entities. They are node-disjoint and alias-disjoint, and every training document naming one (5,305 names) is dropped. Leakage audit: 0.
- **Corpora:** training 100M tokens (SmolLM2 and Qwen3); `eval-entities` 28.8M tokens, scored on 8,192 windows.
- **Arms:**
  - SmolLM2-360M: P0, C0′, C2, C5, C5sh and C6d at seeds 1–3; C5ut and C5tr at seed 1 only (seeds 2–3 withdrawn, decision 64);
  - Qwen3-1.7B: P0, C0′ and C5 at seed 1.
- **Primary:** C5 − C0′ after held-out entities (360M, seeds 1–3).
- **Readings:**
  - a CI excluding 0 = a real-world zero-shot result, read with the caution of toolkit methodology M5 (the host already knows many of these facts);
  - a CI including 0 = no gain over the host's own knowledge, **the expected outcome for well-known entities**;
  - rules (a) C5 − C5sh < 0 and (b) C5 − C6d < 0, as on T7-ROOD.

## Experiment
Nothing has run. Queue (2026-10-20): 117 jobs pending.
- SmolLM2 training at 54.499, evaluations at 54.4991, frame swap and quantization at 54.4992, reports at 54.4993;
- Qwen3 at 54.4994–54.49943.

## Results
None yet. Build numbers only:
- 129,325 realized training documents with 0 held-out names;
- 4,865 held-out occurrences of 319 entities with ≥ 5 occurrences in the 8,192 windows;
- benchmark items mapped onto T8 (e.g. BEAR 23,193 items, 33% with a subject in T8).

## Limitations
- The entities are well known, and the hosts were pretrained on text containing them. A T8 result is about properties of the store, not about knowledge coverage.
- The linker matches some in-word prefixes ("Javan|ese"). These are counted as linker artefacts, not mentions.
- **Sources disagree on timing.** The updates dated T8 ≈ 24 Oct (12 Oct) and ≈ 23 Oct (15 Oct). It runs after T7-ROOD in the queue.

## Next steps
- **Queued:** the whole block.
- **Not decided** (pre-registration §6): v2 editing items, a Wikipedia text stream, raising the entity cap to all 30,596, and running the understanding items on T8.

---

# 08.4 — The other domain tracks

**Status:** T1-open and WordNet done (seed 1, null); T2 built, not run; T3 and T6 built, infeasible, not run; T4 and T5 active (file 06) · **When:** corpora built by early October; T1-open and WordNet results 2026-10-05 · **Sources:** `reports/R9-retrofit-quantization.md` ("T1-open", "WordNet", "Across tracks"); `experiments/t2-developer-tools/README.md`; `experiments/t3-product-catalogues/README.md`; `experiments/t4-chemistry/README.md`; `experiments/t5-enterprise-glossary/README.md`; `experiments/t6-legal-regulatory/README.md`; `manuscript/paper-readiness.md` (E8 rows); `manuscript/update-2026-10-15.md` (richness table)

## Research gap
The E8 plan (application tracks) asked for one track per application family. We need to know which tracks can test the method at all, and which ones did.

## Intuition
A track can test the method only if it has enough held-out and rare terms in its evaluation text, and only if its words are new to the host.

## Methodology
- **Feasibility:** each track uses the C3 holdout procedure and a fixed criterion. The held-out and seen-rare strata must both be powered at a given ℓ_min within ≤ 1,024 evaluation windows. Thresholds were to be harmonized at gate G2 (decision 17, still pending in `paper-readiness.md`, 7 Oct).
- **Runs:** the E9 recipe of 06.1–06.2 wherever a track ran.

## Experiment
Only T1-open and WordNet ran besides T4, T5, T7 and T1c. Both used SmolLM2-135M and -360M at seed 1.

## Results

| Track | Ontology and text | Built | Feasible | E9 runs | Finding |
|---|---|---|---|---|---|
| T1-open | MeSH (30,915 descriptors; 5 relation types, 4.4 edges per term, mostly tree codes) in PubMed | yes | yes | seed 1, both hosts (74 jobs done; none queued) | **null:** every term stratum within ±0.15%; held-out +0.02% [−0.01, +0.04] (360M). Frame swap, INT4 and editing all null. Novelty −5 to −8% |
| WordNet | WordNet multi-word terms in FineWeb-Edu | yes | yes | seed 1, both hosts | **null, as predicted:** within ±0.06%; held-out −0.00% [−0.02, +0.01] (360M) |
| T2 | API symbols as concepts: 16 invented libraries (contamination-free) plus 20 real Python packages and Node.js (55,412 concepts) | yes (SmolLM2 100M tokens; GPT-2 300M) | yes, at ℓ_min 2 with 512 windows (SmolLM2) | none queued | — |
| T3 | Google Product Taxonomy with Shopify attributes, in Amazon ESCI product texts | yes | **no**: the seen-rare stratum is empty (≈ 38 rare entries); held-out and locality only (148 held-out entries) | none | the author's call whether to run held-out only |
| T6 | EuroVoc 4.24 in English EU legislation (MultiEURLEX) | yes | **no**: seen-rare nearly empty (18 entries in 256 windows); held-out and locality only (59 held-out entries) | none | — |
| T4, T5 | ChEBI chemistry; synthetic glossary | yes | yes | active: T4 137 jobs pending (richness stage 1, Qwen3 evaluations), T5 253 pending (larger editing sets, editing baselines, Qwen3 seed 3, Qwen3-4B) | file 06 |

- **Known vocabulary gives no gain.** T1-open and WordNet confirm the novelty ordering of 06.2.
- **Only T2 among the unrun tracks can test the method on both strata.** Its private libraries are contamination-free by construction, like T5's glossary.

## Limitations
- T1-open and WordNet are seed 1 only.
- T3 and T6 cannot test the rare-term claim under the fixed criterion.
- T2's E9 value is untested. Its invented symbols are synthetic, like T5's, but its text is real documentation.

## Next steps
- **Closed:** T1-open and WordNet; no further seeds are queued.
- **Proposed in the plan, not queued:** T2 (the README plans SmolLM2-135M continued training); a decision on held-out-only runs for T3 and T6; Qwen3 on T1-open (a config exists, `t1-qwen3.yaml`).
- **Queued:** the T4 and T5 work listed in file 06 and 07.

---

# Sources

- Plain-language summaries: `manuscript/update-2026-10-12.md` (E9-2, E9-7, E9-11, "What is still missing"), `manuscript/update-2026-10-15.md` (§§5–6, "Next experiments"), `manuscript/status-2026-10-08.md`.
- Ledgers and methodology: `manuscript/claims.md`, `manuscript/paper-readiness.md`, `manuscript/toolkit-methodology-2026-10.md`.
- Report: `reports/R9-retrofit-quantization.md`.
- Pre-registrations: `experiments/t7-new-vocabulary/preregistration-rood.md`, `experiments/t8-wikidata/preregistration.md`, `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md`, `experiments/e9-retrofit/preregistration-addendum-2026-10-15.md`.
- Track documentation: `experiments/t7-new-vocabulary/ROOD.md`; `experiments/{t2-developer-tools,t3-product-catalogues,t4-chemistry,t5-enterprise-glossary,t6-legal-regulatory}/README.md`.
- Generated reports (**uncommitted, preliminary**): working-tree `experiments/e9-retrofit/report/t7/report.md` (T7 at 135M, 3 seeds). No T1c report was opened.
- Queue state: `python -m vsa_embed.jobqueue status`, read-only, 2026-10-20 (T7-ROOD 253 pending; T8 117 pending; T1c 237 done, 63 pending in T1c-ROOD).
