# E14 pre-registration, addendum 10 (2026-10-20, evening; before any host run of it): RD-13, retrieve-then-read

Companion to `preregistration.md` (§§13, 16 and 19 are the RD-4b / RD-4c / RD-4d blocks this builds on). Written and
committed before any GPU job of it was queued. The author's go-ahead: "Retrieval tool over a Wikipedia dump could be
cool" (2026-10-20, evening).

## 1. Question

RD-1 found that a ChEBI definition states only 11.7% of a concept's gold facts, and RD-4b/4c that a fact which *is*
stated is used well once read and recalled selectively (property accuracy 0.96 on the stated items against 0.75 with
nothing, SmolLM2-360M C5). So the lever is the share of facts the text states. Does more text per concept, retrieved
from Wikipedia, raise that share — and does the whole pipeline (read → store → selective recall with the model's own
router) rise with it? The retrieved text in the prompt (RAG) is also the strongest baseline for the same information,
so the store must be compared against it, not only against "nothing".

## 2. The tool and the cache (built before this registration; no host run)

`src/vsa_embed/experiments/e14_retrieve.py` (`fetch`, `stats`, `build`). **API, not dump:** the MediaWiki API of
en.wikipedia.org, as the E11 Wiktionary survey did (`DATA_SOURCES.md` §15b); a full dump (~20 GB compressed) was not
downloaded (disk; the API covered 97% of the concepts, above the 70% fallback threshold set for this decision).

- **Queries:** the concept's item surface (its ChEBI name or alias), then up to 3 ChEBI 255 synonyms (shortest first;
  formulae and all-caps codes excluded), each sent to `list=search` (5 hits, namespace 0). Synonyms are only queried
  while fewer than k = 3 plausible pages are in hand; at most 8 candidate pages per concept are fetched.
- **Pages:** `prop=extracts` (plain text, `exsectionformat=wiki`, whole page) with `prop=revisions` (revision id,
  timestamp) and `prop=pageprops` (the disambiguation flag), one request per page.
- **Disambiguation rule (recorded per concept in `concepts.jsonl`):** disambiguation pages, missing pages and empty
  extracts are dropped; the rest are ordered by (i) an exact title match to one of the queries (case-insensitive),
  (ii) a chemistry cue word in the first 1,500 characters of the intro (`CHEMISTRY_CUES`: compound, acid, enzyme, drug,
  …), (iii) the query's rank, (iv) the search rank. The first is the *primary* page; a concept is *plausibly covered*
  when its primary page has an exact title or a cue. The top-3 pages are kept per concept.
- **Cache:** `~/data/vsa-llm/wikipedia/t4-heldout-v1/` — `pages.jsonl` (1,547 pages: id, title, revision id,
  timestamp, fetch time, extract), `searches.jsonl`, `concepts.jsonl` (queries, candidates with their flags and
  reasons, top-3), `requests.jsonl` (1,976 requests, 820 s, 2.4 requests/s on average, cap 4/s), `source.json`.
  Licence CC BY-SA 4.0 (attribution by title and revision id in every row); the text is not committed (the §15b
  precedent). User-Agent `VSA-LLM-research/0.1 (bingxuhu@gmail.com) E14 retrieve-then-read`. `DATA_SOURCES.md` §17
  and `SHA256SUMS` under `~/data/vsa-llm/` record it.

  | file | sha256 at registration |
  |---|---|
  | `concepts.jsonl` | `ba610583b98716c6c050c6676b344c594e5bc36aa090c8e1ac590fe4cc75ff77` |
  | `pages.jsonl` | `22aa21c9703cf631ba1688999ddd28eb1bb9a45c3aa47bda7698344e7051b566` |
  | `searches.jsonl` | `9db76707272ea4eca430b5c721563e138d53660ea23ef4429a0a4471cd764c10` |
  | `requests.jsonl` | `a82662b45c9c4e133b9b0065896edcc0612d3973553ddf4ad86d543658b29de0` |

- **Coverage (`stats`, seen before this registration):** 341 T4 held-out concepts (`t4-heldout-smollm2-v1`, the RD-4b/4c
  set); 332 with a page (0.974), 331 plausible (0.971; 196 by exact title, 135 by cue only); 22 primary pages came from
  a synonym query; 0.07 candidates per concept dropped as disambiguation pages. Whole extracts: median 1,010 words
  (p10 142, p90 3,189; min 15, max 11,023); intros median 116 words. The 10 uncovered concepts are long IUPAC-style
  names and role classes (e.g. "anticoronaviral agent", "nonasaccharide"). Spot checks: most chemicals land on their
  own page ("glyzin" → Glycine, "fosfinas" → Phosphine, "bile salts" → Bile acid); role and class concepts are the
  weak spot ("protective agent" → Personal protective equipment, "organic anionic group" → Fries rearrangement).

## 3. The texts and the readers (`build`; CPU)

- **Retrieved text:** the primary page's extract with headings removed and navigational sections dropped, cut to
  whole sentences within **300 words** (≈ 516 SmolLM2 tokens in the smoke; the definitions average 60), written as
  "<headword>: <text>" like the ChEBI style. The same text is what the readers read and what the RAG arm puts in the
  prompt. The whole extract is read too (`wiki-full`, stated reader only) as the ceiling on what more text could state.
- **Readers:** E11's CPU readers on the retrieved text — `stated` (gold edges whose partner the concept finder found:
  the stated rate's reader and an oracle-labelled ceiling) and `typeprior` (every found mention with its most frequent
  relation). **The ReAct reader is not used:** on T4 the type prior is the best reader we have short of Claude (RD-3,
  stated F1 0.573 against 0.337 for `react`, which over-proposes partners), and reading with it costs no GPU. The
  readers are wired into `e14_reading recall` as `--retrieved DIR` (the readers `wiki-stated`, `wiki-typeprior` and the
  condition `retrieved`), the smaller change against a second item directory: one flag, one condition kind, no copy of
  the item set.
- **Output:** `experiments/e14-logic/items/t4-heldout-wiki/` — `frames.jsonl` (committed), `coverage.json`,
  `report.md`, `manifest.json` (sha256s), and the git-ignored `retrieved.jsonl` (rebuildable from the cache).

## 4. Arms (`--family rd13`; conditions of one run, scored as RD-4b/4c with the context-free null)

| arm | condition | what the model sees |
|---|---|---|
| (1) nothing | `none` | the item alone |
| (2) definition | `definition` | the ChEBI definition in the prompt (the existing arm) |
| (3) RAG | `retrieved` | the retrieved text (300 words) in the prompt |
| (4) store, realistic | `explicit-route:lm:wiki-typeprior` | the type prior's lines from the retrieved text, the relation chosen by the `lm` router |
| (4′) store, oracle-routed | `explicit-rel:wiki-typeprior` | the same lines, the asked relation given |
| (5) reference pipeline | `explicit-route:lm:typeprior`, `explicit-rel:typeprior` | RD-4c's pipeline on the definition text |
| ceilings | `explicit-rel:oracle`, `explicit-rel:wiki-stated`, `explicit-route:lm:wiki-stated` | gold lines; the gold lines the retrieved text states |

Concepts without a page (9 of 341) have no retrieved text: their items fall out of the paired contrasts that involve a
retrieved arm (the existing conditions are scored on every item; contrasts are paired over the items both arms hold).

## 5. Hosts, metrics, families

- **Hosts:** SmolLM2-360M C5 seeds 1–3 (the RD-4c main hosts; their own E11 frames for the definition readers);
  Qwen3-1.7B-Base C0′ s1 (SmolLM2 C5 s1's E11 frames, as in RD-4c). Items `t4-heldout-smollm2-v1`: 407 property,
  407 entailment; paraphrase is the property items' consistency.
- **Metrics:** property, paraphrase, entailment accuracy (PMI with the plain null; raw choices also recorded); the
  stated rate of the retrieved text against the definition text (`coverage.json`: share of gold edges whose partner the
  finder found, per relation, for `chebi`, `wiki` and `wiki-full`); router accuracy and lines per item as diagnostics.
- **Primary family (per host group: C5 pooled over seeds; C0′ alone; Holm over 3 contrasts × 3 tests):**
  - (a) `explicit-route:lm:wiki-typeprior − none` — (4) − (1);
  - (b) `explicit-route:lm:wiki-typeprior − retrieved` — (4) − (3);
  - (c) `retrieved − definition` — (3) − (2).
- **Secondary (unadjusted):** (4′) − none, (4′) − retrieved, (4′) − `explicit-rel:typeprior`, (4) − (5), (4) −
  definition, the `wiki-stated` ceilings − none and − retrieved, retrieved − none, definition − none, oracle − retrieved.
- **Stated-only subset (descriptive, with intervals):** items whose gold answer is in the `wiki-stated` frame:
  (4′) − none, retrieved − none, (4′) − retrieved.
- **The stated rate** is descriptive (one number per style, with the per-relation table); the prediction below is
  what makes it a test of the premise.

## 6. Predictions (written before `build` ran on the full set and before any host run)

- **Stated rate.** Definition text 0.117 (RD-1). Retrieved text at 300 words: **0.20–0.30**; the whole extract
  0.30–0.40. The structural relations (`contains_element`, `charge`, `branch`, 42% of the edges) stay near 0 in both,
  so the ceiling is about 0.58. The 2-concept smoke saw 0.19 → 0.33 (not a result).
- **(c) retrieved − definition, property:** in **[−0.05, +0.03]** on SmolLM2-360M, not significant; the ChEBI
  definition is short and written for the gold facts, the Wikipedia text is 8× longer and about the parent compound as
  often as the concept. On Qwen3-1.7B closer to 0 or slightly positive (it uses long context better). Both `retrieved −
  none` and `definition − none` are positive.
- **(a) (4) − none, property:** **positive but small, +0.02 to +0.06**, and it may miss significance after Holm: the
  type prior reads the 300 words into ~15 edges at precision ~0.15 (smoke), and RD-4b showed wrong lines cost (−0.11
  for all-wrong lines). The gain stays below the definition pipeline's +0.10 (RD-4c (a) on T4).
- **(b) (4) − retrieved, property:** **negative, −0.02 to −0.08.** RAG is the stronger way to use this text for this
  model; the reader's precision is the loss, not the recall route (RD-4b showed the route works when the lines are
  true).
- **Secondary:** (4′) − none positive (+0.05 to +0.12); the `wiki-stated` ceiling − none clearly positive and above
  `explicit-rel:typeprior − none`, and on the stated-only subset near 0.95 property as in RD-4b; the lm router's
  accuracy on the wiki frames is below its 0.78 on the definition frames (more relations per frame).

**Decision rule.**
- If the stated rate on retrieved text is **< 0.20**, more text per concept is not the lever on T4: the dump is not
  worth the disk, and the next step is relation discovery (RD-11) or structure readers (RD-6), not retrieval.
- If (b) is negative (Holm) while (a) is positive, the store's realistic pipeline loses to RAG on the same text, and
  the loss is reading precision: the next step is a precision filter on the wiki reader (RD-3b's) and routing fixes
  (RD-4d), not more text.
- If (a) and (b) are both ≥ 0 and (c) ≥ 0, retrieval feeds the store usefully; a full-dump build is then worth
  proposing (author's decision; disk).

## 7. Limitations (known before the run)

- Page choice by cue word mislabels some role and class concepts (135 of 331 primary pages were chosen by cue only;
  examples in §2); the rule and every choice are recorded, nothing is hand-corrected.
- For stereo-specific concepts the page is the parent compound's ("(R)-lipoic acid" → Lipoic acid), so some "stated"
  partners are facts about the parent; the reader excludes only the ChEBI headword, and the page title can be read as
  a partner.
- The 300-word cut is by words, not host tokens (≈ 516 SmolLM2 tokens; Qwen3 similar); `fit_context`'s 1,024-token cap
  may cut the longest recall contexts (the wiki type prior's lines, ~15 per item) and never the text itself.
- The hosts have seen Wikipedia in pretraining; the RAG arm partly restates what they know. The items were built from
  ChEBI, not Wikipedia, so no item wording is in the text. Revisions are of September–October 2026; ChEBI is release
  255.
- One track (T4), one retrieval source, one reader family (no host reader on the wiki text).

## 8. Compute, queue, smokes

- **GPU estimate: about 0.5–0.8 GPU-h** in all, under the 2 GPU-h cap, so no restriction of hosts or seeds was
  needed. Basis: RD-4c's T4 runs took ~5 min of scoring on SmolLM2-360M and ~15 min on Qwen3-1.7B for 14 conditions
  with 26–278-token contexts; RD-13 has 10 conditions, two of them with ~500-token contexts (the smoke's `retrieved`
  took 4× the definition arm's time), so ~8–10 min per SmolLM2 seed and ~25–30 min for Qwen3-1.7B. Reading is CPU.
- **Queue:** `queue-commands-addendum10-retrieve.sh` — 4 GPU jobs `e14-rd13-t4-heldout-*` at **54.4959997** and the
  CPU-lane `e14-rd13-report` (writes `experiments/e14-logic/report/rd13/`) at **54.49599986**, after the §19 jobs.
- **Smokes seen before this registration (labelled SMOKE; not results):** `build` and `recall --family rd13` on the
  first 2 concepts (8 items), SmolLM2-360M C5 s1 on CPU, to check the code paths and the stated-rate computation;
  outputs under the session scratchpad only.
- **Tests:** `tests/test_e14_retrieve.py` (cache round-trip through a fake API, the disambiguation rule, the
  truncation, the `retrieved` condition).
