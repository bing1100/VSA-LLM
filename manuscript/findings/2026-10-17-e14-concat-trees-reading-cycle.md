# Finding record — E14 addenda 1–2: concatenation instead of superposition, tree layouts, and multi-round reading

**Recorded:** 2026-10-17 by workplace-fd, for later processing into the E14 write-up and proposal
`manuscript/proposals/2026-10-17-vsa-predicate-logic.md` (§2.6, §9.6).

**Status:** descriptive results, pre-registered in `experiments/e14-logic/preregistration.md` §10 (addendum 1, with
the X8 note) and §11 (addendum 2), each before its runs. All CPU; RD-11 used Claude as the reader ($11.22).

**Pending:** X2 (chains under each layout) and X3 (joins) are still running and will be appended.

**Sources:** `experiments/e14-logic/runs/{lga-concat, lga-concat-x8, rd11-cycle-t4, rd11-cycle-t1}/`.

---

## 1. Concatenate instead of superpose (the HRRCat principle; addendum 1)

**Origin:** the author: "something like HRR cat to combine multiple to reduce noise … higher dimensions … concatenation".
HRRCat in the author's HRRBERT paper concatenates an HRR part with a free part. Here the same principle is tested on the
logic store, against one wide HRR of equal total width. 3 seeds; X7 seed 1.

### X1: rules as vectors under each layout (k = 10 examples unless stated)

| Rule | Wide HRR (1,024 → 16,384) | Redundant blocks ×4 | **By role** [pred \| arg1 \| arg2 \| arg3] |
|---|---:|---:|---:|
| inverse (renamed predicate) | 1.00 | 1.00 | 1.00 |
| rotate three arguments | 1.00 (0.75 at 1,024 with k = 3) | 1.00 | 1.00 |
| **symmetric** (same predicate, arguments swapped) | 0.03–0.12 | 0.03–0.08 | **1.00** (k = 3: 1.00) |
| **sub-relation** (predicate changed, arguments kept) | 0.00 | 0.00 | **1.00** (k = 3: 1.00) |
| any of the four, trained T5 fillers (256 per block) | 0.60 (inverse, wide 256) | — | **0.99–1.00** |

**Rule applied to a whole frame of 8 facts:**

| Layout | 1,024 | 4,096 | 8,192 | 16,384 |
|---|---:|---:|---:|---:|
| wide (all 8 facts superposed) | ≤ 0.02 | ≤ 0.02 | ≤ 0.02 | ≤ 0.02 |
| one block per fact (wide HRR inside); role-moving rules | 0.05–0.28 | 0.80–1.00 | 1.00 | 1.00 |
| one block per fact (wide HRR inside); fixed-point rules | ≤ 0.21 | ≤ 0.21 | ≤ 0.21 | ≤ 0.21 |
| **one block per fact, role blocks inside** | **1.00** | **1.00** | **1.00** | **1.00** |

**Predictions:** X-P1, X-P2 and X-P3 are confirmed.
- Redundant concatenation equals the wide HRR at equal width (X-P1).
- Role concatenation makes the fixed-point rules expressible (X-P2). The rule becomes a block permutation plus a
  per-block map.
- One block per fact makes frames rule-ready (X-P3).

### X7: combining the three trained T4 stores (seeds 1–3; X-P6 confirmed)

Per-slot decode accuracy (top-m against the gold frame):

| | s1 | s2 | s3 | **ensemble of 3 (cosines averaged)** |
|---|---:|---:|---:|---:|
| held-out (zero-shot) | 0.918 | 0.906 | 0.891 | **0.979** |
| seen | 0.920 | 0.912 | 0.896 | **0.981** |

Per relation, held-out, single stores → ensemble:
- `is_a` 0.94–0.95 → 1.00;
- `has_role` 0.91–0.94 → 0.99;
- conjugate acid 0.81–0.89 → **1.00**;
- enantiomer 0.73–0.77 → 0.96.

Three independently trained 256-wide stores, combined, remove most of the decode noise on real chemistry.

### X4–X6: higher widths

- **Nested role–filler terms without clean-up:** depth 8 is 0.86 at 4,096 and 1.00 at 8,192.
- **One nested vector per decision tree:** **still fails at any width**. Depth 8 is 0.23 at 4,096, 0.32 at 8,192 and
  0.22 at 16,384 (X-P5 confirmed). The noise is structural (two subtrees superposed per level), not a capacity problem.
- **Negation as failure** at 32–64 facts per vector: ≥ 0.999 from 4,096.
- **k-of-n matching AUC:** ≥ 0.998.
- **Numbers (fractional power encoding):**
  - 100 levels decode exactly (1.00) at 4,096+;
  - 1,000 levels decode exactly only 0.33 (4,096) / 0.45 (8,192), but `x ≤ t` by a range bundle stays 0.99–1.00.
- **Resonator (our implementation):**
  - 2 factors × 1,000 codewords: 0.54 / 0.91 / 0.99 at 2,048 / 4,096 / 8,192;
  - 3 factors: ≤ 0.30. This is below published capacities, so the implementation should be checked before tier F.

## 2. Trees stored by their structure (X8; the author's idea)

**Setup:** complete binary trees, 20 features × 8 values, 10 classes; 5 trees × 100 inputs per depth; 3 seeds. The
input record is fixed at 4,096 for every layout.

| Depth | One nested vector (16,384) | Node memory, 64 / 256 blocks | **Heap concatenation (node i in block i), 64 / 256** | Level concatenation, L = 1,024 / 4,096 |
|---:|---:|---:|---:|---:|
| 2 | 0.89 | 0.97 / 1.00 | **1.00 / 1.00** | 1.00 / 1.00 |
| 4 | 0.72 | 0.91 / 1.00 | **1.00 / 1.00** | 1.00 / 1.00 |
| 6 | 0.40 | 0.86 / 1.00 | **1.00 / 1.00** | 0.98 / 1.00 |
| 8 | 0.11 | 0.64 / 1.00 | **1.00 / 1.00** | 0.57 / 0.95 |
| 10 | 0.09 | 0.23 / 1.00 | **1.00 / 1.00** | 0.25 / 0.53 |
| 12 | — | 0.14 / 1.00 | **1.00 / 1.00** | 0.12 / 0.19 |

**Total widths:**
- heap concatenation: (2^(d+1) − 1) × b, e.g. 0.52M (b = 64) and 2.1M (b = 256) at depth 12;
- node memory: about the same;
- level concatenation: (d + 1) × L;
- nested vector: D.

**Reading:**
- **Concatenating by the tree's structure is exact at every depth, even with 64-wide blocks.** Navigation is index
  arithmetic, and nothing is superposed except a node's own test.
- **Node memory** is the associative version of the same storage. It needs blocks ≥ 256, or the addresses of 2^d nodes
  collide.
- **Level concatenation** (width linear in depth) holds until a level holds about L/8 nodes.
- **Recommendation:** heap layout for complete or near-complete trees; a preorder layout with subtree sizes for
  irregular trees (not yet tested); heap for the top levels plus node memory for deep sparse parts.

## 3. Multi-round active reading with relation discovery (RD-11; addendum 2)

**Design:**
- **Rounds:** read twice (the relation order shuffled), keep the facts both readings agree on, reflect (cluster `NEW`
  proposals into relation types, accept ≥ 8 support), and repeat for 3 rounds.
- **Reader:** Claude.
- **Analysis:** the tables are from `reanalysis.json`. Every relation outside the round-0 set counts as discovered,
  which corrects the first-pass analysis that treated rediscovered names as visible.

### T4 (160 ChEBI definitions; 4 gold relations hidden at the start; $5.82)

| Round | Relations | Facts per definition (consolidated) | `NEW` share | Visible P / R | Gold recall, all relations (ceiling 0.232) |
|---:|---:|---:|---:|---|---:|
| 0 | 6 | 1.71 | 0.447 | 0.745 / 0.180 | 0.157 |
| 1 | 10 | 1.82 | 0.098 | 0.706 / 0.185 | 0.162 |
| 2 | 10 | 1.83 | 0.082 | 0.720 / 0.185 | 0.162 |

**Accepted after round 0:** `has_functional_parent` (26 supporting proposals), `has_part` (22), `found_in_organism` (10),
`is_metabolite_of` (9).

**Rediscovering the hidden relations:**
- **Two of four were rediscovered by name.**
  - `has_functional_parent`: recall of its 27 gold edges 0.33, precision 0.23–0.26 (Jaccard 0.17).
  - `has_part`: recall 0.13, precision 0.12 (Jaccard 0.07).
- **The two conjugate relations were not rediscovered:** definitions rarely mention conjugates (RD-1a: 5–17% stated).

**Against the predictions:**
- (a) Gold recall rises 0.157 → 0.162 → 0.162. That is a rise, but a small one, reaching 70% of the reachable ceiling.
- (b) **Not met:** two hidden relations were found, but with Jaccard < 0.3. The reader applies the rediscovered
  relations to pairs the gold frames do not hold.
- (c) **Not met:** consolidation adds +0.002 precision on T4 and +0.037 on T1. Two readings by the same strong reader
  mostly agree.

### T1 (120 MeSH scope notes; $5.40)

| Round | Relations | Facts per definition (consolidated) | `NEW` share | Gold recall (MeSH relations; ceiling 0.140) |
|---:|---:|---:|---:|---:|
| 0 | 5 | 2.04 | 0.720 | 0.059 |
| 1 | 9 | 2.18 | 0.409 | 0.056 |
| 2 | 11 | 2.40 | 0.323 | 0.063 |

- **Accepted:** `has_manifestation` (30), `affects_site` (13), `leads_to` (12), `transports` (11) after round 0;
  `acts_on_substrate` (9), `has_member` (9) after round 1.
- **(d) confirmed:** the `NEW` share falls 0.72 → 0.41 → 0.32, and captured facts per note rise 2.04 → 2.40.
- **Gold recall cannot rise here.** The new knowledge lies outside the gold ontology's relations, exactly the "gold may
  not be possible without the relationships existing" case.
- **Its value has to be measured downstream** (explicit recall on questions about those facts; tier F, RD-12).

### Reading

1. **The reflect step works as a relation-discovery mechanism.** It names relations that the ontology really has
   (T4), and on MeSH it adds six relation types that absorb more than half of the previously inexpressible statements.
2. **The gains toward gold are small** on T4. The text is the limit: the ceiling is 0.232, and 0.162 is reached.
3. **Precision of rediscovered relations against gold is low** (0.12–0.26). New relations need a check step (typing,
   consistency) and, in tier F, the passive rounds' feedback, so that commitments can be tested and retracted. E10.9
   saw zero retractions in 19 false commitments.
4. **The next question is whether the knowledge helps a model.** That is RD-4's route (queued) for gold frames, and
   RD-12 for the cycle's own frames.
