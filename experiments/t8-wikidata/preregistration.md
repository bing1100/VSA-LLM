# T8 — Wikidata-framed natural track: pre-registration (E9 on T8)

**Status:** pre-registration, written on 2026-10-08 (decision 63, WP TK-B2) and committed **before any training or
evaluation on T8**. The corpus build (`runs/v1`) and the item builds are CPU data preparation; they read no trained
checkpoint. Any later change is listed in §8 with its date and reason.

## 1. Purpose

Methodology M5 (`manuscript/toolkit-methodology-2026-10.md`): the general relation benchmarks of the toolkit (BEAR, LRE
relations, TwoHopFact, PopQA, Entity Inferences) are built on Wikidata entities, and our ontologies were MeSH, ChEBI and
SNOMED. T8 gives those entities frames — Wikidata truthy statements of a fixed property list — so that the *read*,
*write* and *meta* benchmarks can be scored with a trained store. T8 is also an E9 natural track in its own right, with
the stricter real-world holdout of methodology M1 ("ROOD-text").

**What T8 can and cannot show (M5's caution).** The entities are mostly well known, the text is FineWeb-Edu, and the
hosts were pretrained on FineWeb-Edu (SmolLM2) or on web text that contains these entities (Qwen3). The host already
knows many of the facts. A T8 result is therefore about properties of the store (fixed-width vectors composed from a
frame; zero-shot composition for held-out entities; reverse and chained queries in vector space), not about knowledge
coverage. The contamination-free claims stay with the fictitious and artificial sets (reversal, ALCUNA, COMPS-WUGS).

## 2. Data (frozen before any run)

- **Entities:** BEAR (60 relations, QIDs given), PopQA (QIDs given), TwoHopFact (e1/e2/e3 QIDs given), LRE relations
  (26 factual relations; names resolved with `wbsearchentities`, disambiguated by the relation's Wikidata property, then
  by type), Entity Inferences / ECBD (names; ECBD by its Wikipedia title). Resolution counts:
  `~/data/vsa-llm/wikidata/t8/entities.json`.
- **Wikidata snapshot** (CC0; retrieved 2026-10-08 through WDQS and the API, raw responses cached under
  `~/data/vsa-llm/wikidata/cache/`): `records.jsonl.gz`, sha256 pinned in `t8.yaml` (`wikidata.records_sha256`).
- **Selection** (`screen/selection.tsv`, sha256 pinned as `ontology.selection_sha256`): at most 20,000 entities,
  those named ≥ 5 times in the training-side FineWeb-Edu text first; alias policy as in `t8.yaml` (names must be written
  with the entity's capitals in ≥ 90% of their occurrences; single-word aliases, shared keys and labels a comparably known
  other item also bears do not link).
- **Frames:** one edge per (property, value), ≤ 3 values per property (P31: 2), ≤ 16 edges, over the 8,192 most-used
  filler items; the property list and the benchmark relation each property serves are in `ontologies/wikidata.py`.
- **Text:** FineWeb-Edu sample-10BT shards 002 and 000 (C3's), after C3's 5,000 evaluation documents. Documents whose
  id's sha256 bucket is < 1,000 / 10,000 are evaluation-only. Training = documents that mention a selected name, mixed
  50/50 by tokens with documents that mention none.

## 3. Holdout (M1, ROOD)

1. **Node-disjoint:** a held-out entity fills no frame of the track (no other concept names it), and it is a
   single-concept entry.
2. **Alias-disjoint:** T1-open's closure — no held-out alias is a training alias or a whole-word part of one.
3. **Document exclusion:** every training document (both streams) that names a held-out entity under *any* of its
   Wikidata names (≥ 3 characters) is dropped; those documents feed the evaluation split `eval-entities` instead.
   Eligibility caps the exclusion cost (an entity's names occur in ≤ 300 training-side screen documents).
4. **Leakage audit** (in the build; must be 0 on every count): held-out names in the realized training documents;
   training documents that are also `eval-entities` documents; held-out spans when the training documents are linked
   with the full alias table.
5. The held-out list is pinned (`data.expected_holdout_sha256`) by the first full build, before any training.

Build numbers (`runs/v1/report.md`): see §7.

## 4. E9 design

Recipe: the E9 recipe of the other natural tracks (50M tokens, 64 × 1,024 tokens per step, warmup 2.5M; SmolLM2 full
fine-tuning at host lr 3e-5, channel lr 1e-3, gate bias 0; Qwen3 LoRA r64 at host lr 2e-4 with `channel.scale_to_host`),
evaluation on `eval-entities` with 8,192 windows (the fewest that meet the held-out and rare criteria at ℓ_min 2;
4,096 windows reach 2,418 held-out occurrences but only 130 held-out entries with ≥ 5) and per-window losses.

| Host | Models | Seeds | Priority |
|---|---|---|---|
| SmolLM2-360M | P0, C0′, C2, C5 | 1–3 | 54.499 (level step 0.0001) |
| SmolLM2-360M | WP-PQ1 arms C5ut (no binding), C5tr (translation) | 1–3 | 54.499 (level step 0.0001) |
| Qwen3-1.7B-Base | P0, C0′, C5 (LoRA r64) | 1 | 54.4992 (level step 0.0001) |

## 5. Endpoints

**Primary (pre-registered):** C5 − C0′ relative loss on the `after_heldout` stratum of `eval-entities` (the 8 tokens
after a held-out entity's span: held-out entities never occur in training text), ℓ_min 2, the runs' final evaluation on
the 8,192 windows, SmolLM2-360M, seeds 1–3 pooled, paired window bootstrap (95% CI), at bf16 — the E9 readout ("C0′
lowers every stratum, so C5 − C0′ is the readout").

**Secondary** (Holm within each family): C5 − C0′ on `after_unseen`, `after_rare_seen`, `after`; C5 − C2 on the same
strata; the operator controls C5 − C5ut and C5 − C5tr on `after_heldout` (rule 7: a binding claim needs C5 better than
both); INT4 retained gain (E9's quantization readout); dimension 3 (new words, edits) as in E9; Qwen3-1.7B seed 1
C5 − C0′ on `after_heldout` (descriptive; seeds 2–3 only if seed 1 shows a clear effect, M6).

**Readings fixed now.**
- C5 − C0′ < 0 on `after_heldout` with the CI excluding 0: the composed store helps on held-out entities whose text the
  host never saw in fine-tuning — a real-world zero-shot result on famous-entity text, read with M5's caution.
- CI including 0: no gain over the host's own knowledge on these entities (the expected outcome for well-known entities,
  as on T1-open and WordNet); the benchmark endpoints then test store properties only.
- C5 not better than C5ut and C5tr: "compositional parameter sharing", not binding (rule 7).

The benchmark endpoints (BEAR, LRE, TwoHopFact, PopQA, Entity Inferences on the trained store) are pre-registered by
TK-B1; T8 provides their items mapped onto its concepts (`~/data/vsa-llm/benchmarks/t8-items/`, manifest
`experiments/t8-wikidata/items-manifest.json`).

## 6. Not decided here

- Whether to grow the dimension-3 items to v2 sizes (decision 56) — only if T8 continues past the first block.
- A Wikipedia text stream: not built (FineWeb-Edu coverage, §7).
- The concept cap: 20,000 of 30,596 qualifying entities (the methodology's guide); raising it to all 30,596 would raise
  PopQA and TwoHopFact subject coverage but needs a new selection, holdout and build (before any run only).
- Whether the WP-UB understanding items run on T8 (built; queue command commented in `queue-commands.sh`).

## 7. Build record (before any run)

- **Entities** (`t8_benchmarks entities`): 201,497 references, 78,759 entities. LRE subjects: 8,719 verified by the
  relation's property, 330 by type, 617 top hit, 30 unresolved; Entity Inferences: 108 typed, 30 top hit, 8 unresolved,
  24 fictitious (no QID); ECBD: 936 rows by their Wikipedia title.
- **Snapshot:** 341,772 records (78,748 benchmark entities, 56,452 fillers with ≥ 10 sitelinks and statements, 206,572
  label-only fillers), 1,378,033 statements, 100 redirects; sha256 `7ad8fa65…d7a8`.
- **Screen:** 652,402 names counted in 1,303,465 training-side and the evaluation-side FineWeb-Edu documents; homonyms of
  73,185 labels; 30,596 entities qualify (≥ 5 mentions); **20,000 selected**, 29,993 names; sha256 `4916a8c5…be77`.
  FineWeb-Edu coverage (any Wikidata name ≥ 5 times): BEAR subjects 4,144 / 7,518, LRE subjects 3,773 / 8,473, PopQA
  subjects 8,571 / 12,244, TwoHopFact e1 7,796 / 26,614 — enough that no Wikipedia stream was added.
- **Ontology:** 88 relations, 8,192 filler atomics, 144,367 edges (7.2 per entity; 69,628 edges to fillers outside the
  dictionary dropped), 99 empty frames.
- **Holdout:** 1,680 entries (1,645 chosen of 8,217 eligible with ≥ 20 screen occurrences, 35 by the closure; 13,590
  ROOD-eligible entries; excluded: 3,585 frame fillers, 2,739 by exclusion cost, 86 by closure); 5,305 exclusion names,
  naming ≤ 9.8% of the training-side documents; sha256 `99312697…291d` (pinned).
- **Leakage audit:** 0 held-out names in the 129,325 realized training documents, 0 shared documents, 0 held-out
  mentions with the full alias table (402 in-word prefix matches such as "Javan|ese" and 1 offset-shift match are linker
  artefacts, not mentions; the training table never links held-out aliases).
- **Corpora:** train 100.0M tokens (SmolLM2; 50.6% entity documents) and 100.0M (Qwen3); `eval-entities` 28.8M tokens
  (5,000 evaluation-side + 8,000 ROOD documents); `eval-general` 5.0M; `eval` 10.2M.
- **Feasibility (ℓ_min 2, SmolLM2):** 8,192 windows — held-out 4,865 occurrences, 319 entries ≥ 5; rare 8,289 / 3,357
  entries (feasible). Whole split: 16,028 / 927; 28,778 / 5,663.
- **Items:** dimension 3 — 300 new words (4,050 items), 194 edits (SmolLM2; 193 for Qwen3); WP-UB understanding items
  (19,219; not in the pre-registered block). Benchmark items on T8 (`items-manifest.json`):

| set | items | subject in T8 | held-out subject | answer in the subject's frame |
|---|---:|---:|---:|---:|
| BEAR (3 templates per instance) | 23,193 | 7,566 (33%) | 609 | 6,057 |
| LRE relations (factual) | 9,696 | 3,656 (38%) | 274 | 1,571 |
| PopQA | 14,267 | 1,429 (10%) | 57 | 798 |
| TwoHopFact 2-hop (subject e1) | 45,595 | 4,358 (10%) | 293 | 1,627 (e3 already in e1's frame: a shortcut) |
| TwoHopFact hop 1 (e1 → e2) | 45,595 | 4,358 (10%) | 293 | 2,045 |
| TwoHopFact hop 2 (e2 → e3) | 45,595 | 19,120 (42%) | 1,102 | 11,603 |
| Entity Inferences | 170 | 12 (7%) | 0 | 0 |

## 8. Changes

**8.1 (2026-10-09, decision 64, author; before any T8 training or evaluation).**

- **Timing.** Every T8 job is pending: SmolLM2-360M training at 54.499, evaluations at 54.4991, reports at 54.4993. No T8
  number exists.
- **Why.** On T4 (real chemistry, SmolLM2-360M, 3 seeds), C5 − C0′ on `after_heldout` is −0.40% [−0.54, −0.24], but
  shuffled frames keep that gain (C5 − C5sh +0.01% [−0.08, +0.10]) and the definition encoder beats C5 there (C5 − C6d
  +0.17% [+0.05, +0.29]). The operator arms tie on T4 (C5 − C5rf and C5 − C5ut n.s.), on T5 (untyped ties learned HRR,
  translation is better) and in decision 54's operator screen.
- **Arms added** (SmolLM2-360M, seeds 1–3, the §4 recipe; configs in `experiments/e9-retrofit/configs/t8/`; queued by the
  coordinator at 54.499, `pq` chain at 54.4991; row-source table job `t8-rowsource-definition-SmolLM2-360M` at 54.499):
  - **C5sh**: shuffled frames (every entry reads another entry's frame, a derangement);
  - **C6d**: the definition encoder (the frozen host's mean-pooled hidden state of the entry's verbalized frame, through a
    trained projector at C5's site; held-out entities get their rows from the same source).
- **Two reading rules** for the primary endpoint (C5 − C0′ on `after_heldout`), read in the arm batch report
  `report/t8-pq` (`e9_report`'s arm table: C5 − arm on the 8,192 final-evaluation windows, pooled over common seeds,
  paired window bootstrap):
  - **(a) Ontology specificity.** A held-out gain counts as **ontology-specific** only if C5 − C5sh < 0 on
    `after_heldout`, with the Holm-adjusted p < 0.05 and the CI excluding 0. On T8 the operator-arm family within the
    stratum holds C5sh, C5ut and C5tr (the last two at seed 1), so Holm runs over three. Otherwise §5's first reading
    becomes "the store helps on held-out entities; the frame's content is not shown to be the source".
  - **(b) Against standard new-word vectors.** "Better than standard new-word vectors" is claimed only if C5 − C6d < 0
    on `after_heldout`, with Holm p < 0.05 (the row-source family: C6d alone) and the CI excluding 0.
- **C5ut / C5tr seeds 2–3 withdrawn.** The coordinator moved the jobs `t8-SmolLM2-360M-full-C5ut-s2`, `-C5ut-s3`,
  `-C5tr-s2` and `-C5tr-s3` to `.jobs/cancelled`. Seed 1 of each is kept as a replication.
  - Rule 7's binding claim on T8 is **no longer sought**: the operator is null on T5, on T4 and in the operator screen.
  - The secondary contrasts C5 − C5ut and C5 − C5tr on `after_heldout` remain, as **descriptive single-seed
    replications**: their CIs cover evaluation windows only.
  - §5's reading "C5 not better than C5ut and C5tr: compositional parameter sharing, not binding" stays the default. If
    both seed-1 contrasts favour C5, that is reported as a single-seed observation, not as a binding claim.
- **Within-model specificity.** The held-out frame-swap rescore (`e9_frameswap`, pre-registered separately in
  `experiments/e9-retrofit/preregistration-frameswap.md`, being built) swaps held-out entities' frames at evaluation on
  the trained C5 itself. It is the within-model test that complements C5sh. Its rules are its own.
- **Unchanged:** the primary endpoint and its test, the other secondaries, the Qwen3 block (P0, C0′, C5 at seed 1, no
  controls), the benchmark endpoints and §6.
