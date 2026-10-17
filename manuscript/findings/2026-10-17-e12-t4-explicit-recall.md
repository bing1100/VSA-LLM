# Finding record — E12 explicit recall on real chemistry text (T4), three seeds

**Recorded:** 2026-10-17, for later processing into R9 / the E12 write-up, the claims ledger (E12 rows) and the draft. The
title, abstract and outline belong to workplace-ba/fd; the E12 section belongs to its owner.

**Status:** official report plus derived intervals.
- **Official report:** `experiments/e12-self-query/report/t4-s1-2-3/report.md` and `analysis.json` (job `t4-report-e12`,
  run 2026-10-17, committed with this note).
- **What the report gives:** per-condition means over seeds 1–3 (360M) and seed 1 (135M). Its formal Q1/F1 contrasts
  cover only the T5 twin set, so for these natural chemistry items it gives **no confidence intervals**.
- **Derived numbers:** the intervals and the subset and frame-size breakdowns in §3 are **derived by workplace-fd** from
  the same per-unit outputs. They are paired bootstraps over the 715 anchors, scores averaged over seeds 1–3, 4,000
  resamples. They are **not part of the official report** and should be recomputed by the report code before they are cited.
- **Earlier sources:** the seed-1 numbers in `manuscript/status-2026-10-08.md` §3.4 and `report/t4-s1`. This record
  supersedes them at three seeds.
- **Provenance note:** the report first failed (it targeted the stage's base folder, which `e12_report` refuses,
  amendment 12.4). It was re-run into `report/t4-s1-2-3` on 2026-10-17. The numbers are unaffected.

---

## 1. The experiment in plain words

**Question.** The method stores what the ontology says about a term inside one composed vector (relations bound to their
partners). The binding test P1 showed that a model does not use these stored *roles* on its own: it cannot tell apart
two new words whose relations are swapped. E12 asks whether decoding the store into text, in the prompt, lets the model use them.

**Items** (`experiments/e9-retrofit/items/role-natural-t4-smollm2-v1`, 715 anchors × 4 templates):
- **Anchors:** ChEBI chemicals with two relations to two other chemicals, e.g. *is a* F and *is the conjugate base of* G.
- **Ambiguity:** F and G are each attested elsewhere under the *other* relation, so either could plausibly fill either
  role. General knowledge of the partners is not enough; the model must assign roles.
- **Subsets:** 300 seen terms, 300 rare and 115 held-out (vectors composed zero-shot).
- **Relation pairs:** mostly `is_a` against `has_functional_parent` or the conjugate acid/base relations.
- **Score:** a role-contrast accuracy (chance 0.5). Both a choice and a cloze format.

**Method ("recall").** For each anchor, each relation is unbound from the term's composed vector and the result is
snapped to the nearest known concept. The decoded edges go into the prompt as lines like "X — is a: F (0.61)", then the
question follows.

**Conditions:**
- nothing;
- decoded from the model's **own** store;
- decoded from a **no-binding** store (C5ut: the same partners summed, no roles);
- decoded from a **translation** store (C5tr);
- **roleless** decode (the decoded partners listed without their relation labels);
- **gold** relations as text (upper bound);
- a **prose definition** written from the gold frame.

The decoded store can also be read by a *different* host (the "recall:C5" rows).

## 2. Official results (choice format, contrast accuracy)

**SmolLM2-360M, seeds 1–3.** The "decode" column is the share of edges decoded correctly from the store.

| Host | Prompt content | Accuracy | Decode |
|---|---|---:|---:|
| C5 | nothing | 0.554 | — |
| C5 | **decoded from its own store** | **0.882** | 0.908 |
| C5 | gold relations as text | 0.897 | 1.000 |
| C5 | prose definition | 0.915 | — |
| C5 | decoded partners without role labels | 0.479 | 0.909 |
| C5 | decoded from a no-binding store (C5ut) | 0.500 | 0.882 |
| C5 | decoded from a translation store (C5tr) | 0.511 | 0.625 |
| C5rf (fixed random operator) | decoded from its own store | 0.881 | **0.960** |
| C5sh (shuffled frames) | decoded from its own store | 0.559 | 0.439 |
| C0′ (no method) | reading C5's store | 0.877 | 0.908 |
| C2, C6d, C6g, C6m, C5ut hosts | reading C5's store | 0.877–0.882 | 0.908 |
| P0 (untouched model; seed 1) | reading C5's store / gold / definition | 0.857 / 0.887 / 0.822 | 0.921 |

- **Cloze format** (same items, 360M): nothing 0.540; own store 0.794; gold 0.822; definition 0.856; no-binding store
  0.518; translation store 0.502. A C0′ host reading C5's store scores 0.811. Same pattern, about 0.08 lower.
- **SmolLM2-135M** (seed 1, choice): nothing 0.565; own store 0.894; gold 0.924; definition 0.875; roleless 0.481.
  C0′ reading C5's store scores 0.901.
- **Two-hop and reverse items** (360M, seed 1, C5 host): two-hop accuracy nothing 0.230, own-store decode 0.623, gold
  0.843. Reverse lookup: nothing 0.554, decode **0.477**, gold 0.766. **Decoded reverse lookups do not help.**

## 3. Derived intervals and breakdowns (workplace-fd; see Status)

**Paired differences, 360M C5 host, all 715 anchors, seed-averaged:**

| Contrast | Difference [95% CI] |
|---|---|
| own-store decode − nothing | **+0.328 [+0.302, +0.355]** |
| own-store decode − no-binding store | **+0.382 [+0.357, +0.407]** |
| own-store decode − translation store | +0.371 [+0.349, +0.393] |
| own-store decode − gold text | −0.015 [−0.023, −0.006] |
| own-store decode − prose definition | −0.033 [−0.049, −0.017] |
| C0′ reading C5's store − C5 reading its own | −0.005 [−0.011, +0.001] |

**By subset:**

| Subset | n | nothing | own store | gold | definition | own − nothing | own − no-binding |
|---|---:|---:|---:|---:|---:|---|---|
| seen | 300 | 0.551 | 0.867 | 0.887 | 0.918 | +0.317 [+0.274, +0.358] | +0.354 [+0.316, +0.393] |
| rare | 300 | 0.498 | 0.881 | 0.894 | 0.895 | +0.383 [+0.342, +0.425] | +0.423 [+0.385, +0.462] |
| **held-out** (zero-shot vectors) | 115 | 0.707 | **0.925** | 0.929 | 0.959 | +0.217 [+0.168, +0.270] | +0.345 [+0.280, +0.410] |

**By frame size** (edges in the anchor's frame; the capacity question):

| Edges | n | own store | gold | gap to gold |
|---|---:|---:|---:|---:|
| 0–7 | 176 | 0.932 | 0.929 | +0.003 |
| 8–12 | 397 | 0.891 | 0.905 | −0.014 |
| 13–24 | 142 | 0.796 | 0.833 | −0.037 |

## 4. Findings

1. **The store is readable and useful on real text.** Decoding a term's composed vector into text lifts role-assignment
   accuracy from 0.55 to 0.88 (+0.33), within 0.015 of the gold relations. All three seeds agree (0.880–0.884).
2. **Binding is what makes it work.**
   - Without roles the decoded partners mislead. Partners without labels score 0.479 (below chance), and the no-binding
     store 0.500, although the no-binding store decodes the partners almost as well (0.882).
   - The translation store decodes poorly (0.625) and gives chance.
   - This is the clearest evidence in the program that HRR-style binding, not just composition, earns its place: for
     **explicit, role-specific read-out**.
3. **Learning the operator is unnecessary.**
   - A fixed random operator (C5rf) gives the same accuracy (0.881) and decodes *better* (0.960 against 0.908).
   - Shuffled frames destroy the store (decode 0.439, accuracy 0.559), so the content must be the right ontology.
4. **The knowledge lives in the store, not in the model trained with it.**
   - Every host reading C5's store scores about 0.88: plain fine-tuning, the free table, the row-source arms, even the
     untouched pretrained model (0.857, seed 1).
   - The store is a portable, plug-in knowledge source.
5. **Zero-shot vectors read back as well as gold.** For held-out terms, whose vectors are composed from the ontology
   without training, decode accuracy is 0.925 against gold 0.929. (Their no-prompt baseline is also higher, 0.707.)
6. **Capacity shows at large frames.** The store falls further behind gold as frames grow: −0.037 at 13–24 edges,
   against +0.003 at 0–7. This fits the capacity sweep (one 256-wide vector holds about 8 relations cleanly).
7. **A prose definition is still slightly better** (−0.033). For the untouched model and at 135M, though, the decoded
   store *beats* the definition (P0 0.857 against 0.822; 135M 0.894 against 0.875).
8. **Multi-step use is partial.** Decoded two-hop chains help (0.23 → 0.62, gold 0.84). Decoded reverse lookups do not
   (0.48 against gold 0.77). Seed 1 only.

## 5. Implications for the paper

- **Framing:** "VSA" belongs with the **explicit store**: a compact, fixed-size, decodable record of role-specific facts,
  usable by any host through its decoded text. It does not belong with language-model loss, where simpler operators match
  binding.
- **The strongest real-text result for binding in the program,** now at three seeds.
- **What not to claim:**
  - that the store beats a written definition (it is slightly behind with the fine-tuned model);
  - that the store is cheaper in tokens (not measured);
  - that the model uses roles implicitly (it does not; P1 null).
- **Practical framing:** a definition needs curated text per term; the store needs only the ontology, and works for
  terms composed zero-shot.

## 6. Caveats on fairness

- **Storage, not learning:** the stores are built from the curated ontology, so this tests storage and read-out
  fidelity, not discovery. E11: reading facts from real definitions does not work yet.
- **Matched conditions:** the same items, templates, prompts and scoring in every condition. The no-binding store holds
  the same partners, so the role contrast is clean.
- **The "definition" is generated from the gold frame,** not a real ChEBI definition.
- **Hosts:** 360M has 3 seeds; 135M and P0 have one. The two-hop and reverse items have one seed.
- **Intervals:** the ones in §3 are derived, not official (see Status).

## 7. Open items and next steps

- [ ] Add natural-item contrasts with intervals to `e12_report` (or recompute §3 there) before citing them.
- [ ] Token accounting: decoded-store text against definition length per item.
- [ ] A capacity follow-up: recall against frame size at composed width 512. The ontology-richness stage-1 arm `C5w512`
      trains such a model on T4, around Oct 27–28; run E12 on it.
- [ ] Qwen3 hosts (queued: `t5-qwen3-report-e12`, now writing `report/t5-qwen3-e12`), and T7-ROOD understanding
      (`e12-t7rood-understanding-report`).
- [ ] Readout arms (models trained to unbind themselves): around Oct 17–19.
- [ ] Fold into: the claims ledger E12 rows; the draft's E12 subsection; R9/E12 section; the abstract (owner
      workplace-ba/fd).

## Sources

- **Official report:** `experiments/e12-self-query/report/t4-s1-2-3/{report.md,analysis.json}`.
- **Per-run outputs:** `experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-*-s{1,2,3}/self-query-role-natural-t4-smollm2-v1/summary.json`.
- **Items:** `experiments/e9-retrofit/items/role-natural-t4-smollm2-v1/` (concepts.jsonl: subset, degree).
- **Pre-registration:** `experiments/e12-self-query/preregistration.md` (natural items §SQ6; amendment 12.4).
