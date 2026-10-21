# E14 pre-registration — addendum 11 (2026-10-20, before any queue entry): two cheap tests, LG-CF and RD-14

Companion to `experiments/e14-logic/preregistration.md` (conventions in its header; RD-4b §13, LG-B verdicts §14, RD-4c
§16, LG-D2 §19). Written and committed before either study was queued. The author's instruction of 2026-10-20 (evening),
*"Let's also do some of the cheaper tests."*, is the go-ahead.

**Status:** pre-registration. Any change goes in the last section with its date and reason. Labelled smokes may run
before a queue entry; they change nothing here.

**Code:** `src/vsa_embed/logic/world.py` (`counterfactual_trees`, `cf_edit`), `logic/store.py` (`StableTreeStore`),
`experiments/e14_items.py` (`counterfactual`), `experiments/e14_score.py` (`cf-report`), `experiments/e14_inferred.py`
(RD-14, reusing `e14_reading`'s scoring); tests `tests/test_e14_addendum11.py`; queue
`experiments/e14-logic/queue-commands-addendum11-cheap.sh`.

**Conventions:** as the main file: paired bootstrap, 4,000 resamples, 95% intervals, Holm within each primary family;
LG-CF's items are paired by chain state, RD-14's by item with seeds pooled (the R9 convention). Each study's primary is
read on its own.

---

## 1. LG-CF: consistency under a chain of "what if" edits

### Question

An exact executor over the store answers every edited state of a decision problem exactly, so a chain of "what if"
questions stays consistent by construction. Does a frozen language model that answers the same chain in context drift
as the edits accumulate, and does relaying the executor remove the drift on the hosts of E14?

### Items (`e14_items counterfactual`, set `trees-cf`, seed 1; built 2026-10-20)

- **Trees:** LG-D's generator (`random_tree`: 8 invented attributes, categorical with 3–5 values or numeric 0–100;
  tests "its A is v" and "its A is at most t"; 6 outcomes "send it to the X desk") under its own seed key
  (`trees-cf:1`) and namer (`Namer(61)`); depths 2, 3, 4, 6 × 75 = **300 chains**, half with numeric attributes (odd
  indices, as in LG-D). The tier-P sets, `trees-noisy` and `world.trees` are unchanged.
- **Chains:** each tree and its record receive **8 successive edits** (`cf_edit`, seeded per chain). With probability
  1/2 an **input edit**: one attribute (on the current path with probability 0.7) takes another value (categorical:
  another of its values; numeric: another integer 0–100). Otherwise a **policy edit** on an internal node whose test
  text is unique in the current tree (so the edit's sentence names it without ambiguity): `retarget` (the test's value
  or threshold is replaced) or `swap` (its two branches are exchanged), each with probability 1/2; without a unique
  test, an input edit; a `swap` needs two different branches (a swap of equal branches would be no edit). Built: 1,199 input,
  606 retarget, 595 swap edits.
- **States and gold:** state k is the tree and record after the first k edits. Gold = the exact executor's leaf on
  state k (`reference.run_tree`). **States scored: k = 0, 1, 2, 4, 8** (k = 0 is the unedited item), 1,500 rows.
  Gold differs from state 0 in 28% / 44% / 58% / 64% of chains at k = 1 / 2 / 4 / 8.
- **Options:** the chain's 6 outcomes, shuffled once per chain, so every executor answer is an option (LG-D used 4).
- **Stores:** `StableTreeStore`, LG-D's `TreeStore` with the SHA-256-seeded codebook, so the set rebuilds byte-
  identically in any process (checked in two processes with different hash seeds: file sha256 `7943555f…`; rows
  sha256 `4cba3f98…`, in `items/manifest-cf-s1.json`).
- **Executor ceilings (no model):** the VSA executor's own answer equals the gold on 0.993 of the chains at k = 0, 0.997
  at k = 1, 2, 4 (one or two read errors out of 300) and 1.000 at k = 8; the symbolic executor 1.000 everywhere. Condition (c) of the author's
  design, "executor alone", is this ceiling; it is not scored by a host.

### Conditions (per state k; `e14_score score --sets trees-cf`)

- **`chain-text` (a):** the original record, the original policy as LG-D's `tree-text`, then "Changes so far:" and
  the edits 1 … k as a numbered list of sentences ("Its A is now v."; "In the policy, the test "…" now reads "…"."; "In
  the policy, the two branches of the test "…" are exchanged: …"). The model answers the chain in context. At k = 0
  this is LG-D's `tree-text` with 6 options.
- **`chain-relay` (b):** the edited record and the exact executor's path on the edited state (trace lines with read
  confidences and the "Conclusion:" line), i.e. LG-D's `path-vsa` interface, no history. The model relays the executor.
- **`chain-text+exec`:** `chain-text`, then the executor's path (the verdict inside the full history).
- **`chain-text+conclusion`:** `chain-text`, then the executor's conclusion line only.
- Question: "Under the policy, what happens to E?" (k = 0) or "After these changes, under the policy, what happens
  to E?"; forced choice over the 6 options by length-normalised log-probability, as LG-D.

**On the "verdict" interface.** §14 showed that the SmolLM2 hosts cannot relay an executor's verdict written as a
true/false claim (G1′ failed; they answer from surface cues). The relay here is therefore the multiple-choice path
interface, which SmolLM2-360M copied at 0.997 in LG-D (record + path, no policy text); the executor's answer is always
among the options. Whether the relay also survives a long history is what `chain-text+exec` and `chain-text+conclusion`
measure; the smoke below is why they are secondary.

### Hosts

H-360 (SmolLM2-360M), H-q06 (Qwen3-0.6B-Base), H-q17 (Qwen3-1.7B-Base), frozen, as in E14. (H-135 only in the smoke.)

### Metrics

- **Per-step accuracy** by condition and k: the drift curve.
- **Consistency:** the share of chains whose scored states k = 1, 2, 4, 8 are all answered correctly, per condition.
- **Anchoring:** at k = 8, the share of answers equal to the original state's gold and to the previous scored state's
  gold; also on the chains whose gold changed.
- **Follow rate:** how often the host picks its executor's answer (and when the executor is wrong, if ever).

### Primary (per host; Holm over the 3 hosts)

`chain-relay − chain-text` at **k = 8**, paired by chain (n = 300 per host).

### Secondary (unadjusted)

- the same contrast at k = 0, 1, 2, 4; its consistency version (chain level);
- `chain-text+exec − chain-text` and `chain-text+conclusion − chain-text` at k = 8; `chain-relay − chain-text+exec`
  and `chain-text+exec − chain-text+conclusion` (what the history and the trace cost the relay);
- the primary on chains whose gold changed from the start, on unchanged chains, on chains with a policy edit, on chains
  with input edits only, and by depth;
- the drift curve and anchoring (descriptive).

### Predictions (written before any host scored these items; the 135M smoke below was seen)

1. `chain-text` falls with k on every host: accuracy at k = 8 is at least 0.05 below k = 1. LG-D's `tree-text` was
   0.37–0.38 with 4 options on all three hosts; with 6 options, k = 0 is expected near 0.30 and k = 8 within 0.10 of
   chance (0.167) on SmolLM2-360M and Qwen3-0.6B. Qwen3-1.7B also declines, less steeply.
2. `chain-relay` is ≥ 0.95 at every k on every host (LG-D's `path-vsa`: 0.997 on all hosts).
3. **The primary is positive (Holm) on all three hosts, by at least +0.50.**
4. Consistency: `chain-relay` ≥ 0.90; `chain-text` ≤ 0.10 on every host.
5. The long-history relay is worse than the short one on SmolLM2-360M (`chain-relay − chain-text+exec` > 0) and
   about 0 (|·| ≤ 0.03) on the Qwen3 hosts, which relayed in every format of §14. `chain-text+conclusion` is at or
   below `chain-text+exec`.
6. Anchoring: at k = 8, on chains whose gold changed, `chain-text` picks the original state's gold more often than the
   true one on the SmolLM2 host (the in-context answer sticks to the unedited tree).

**Alternative stated in advance:** if `chain-text` on Qwen3-1.7B does not decline (k = 8 within 0.05 of k = 1), the
claim "a model answering in context drifts" is not supported at this scale; the executor then merely matches the model
and the positive primary (if any) reflects only the base gap of LG-D, not drift.

### Decision rule

- If the primary is positive (Holm) on all three hosts **and** prediction 1 holds on at least the two smaller hosts, the
  manuscript may state that an exact executor keeps a chain of counterfactual questions consistent where in-context
  answering drifts, for hosts up to 1.7B, with the executor's ceiling (1.000) as the reference.
- If the primary is positive but `chain-text` does not drift (the alternative), the result is stated as "the executor's
  advantage on decision trees persists under edits" only.
- If `chain-relay` is below 0.95 on a host, the relay interface fails on that host and the primary is not read as
  evidence about drift for it; §14's conclusion (small hosts need a multiple-choice interface) is extended with "and a
  short context".

### Smoke seen before this registration (labelled; not a result)

SmolLM2-135M, the first 25 rows (5 chains, depth 2) of the set as first built (before the no-op-swap rule, which
changed the edits), CPU, 600 sequences, 121 s. `chain-relay` 1.000 at every k, the
executor followed at 1.000; `chain-text` 0.2–0.6; `chain-text+exec` and `chain-text+conclusion` followed the executor
only 0.52 of the time. This is what made the short relay interface the primary and the long ones secondary, before any
host of the study was run. Mean context tokens (135M tokenizer): `chain-relay` 164, `chain-text` 299,
`chain-text+conclusion` 320, `chain-text+exec` 377.

### Cost

1,500 rows × 4 conditions × 6 options = 36,000 sequences, 1.17× LG-D's, with about twice LG-D's context tokens
(`tree-text` ≈ 150). Scaling LG-D's measured durations (0.058 / 0.083 / 0.166 h for H-360 / H-q06 / H-q17, from
`.jobs/`) by 2.3 gives about 0.14 / 0.19 / 0.39 GPU-h, **≈ 0.7 GPU-h**. The report is CPU lane.

---

## 2. RD-14: completion under selective recall — do inferred facts help or hurt when recalled?

### Question

Selective recall of the asked relation's gold facts helps on real chemicals (§13: +0.24 property on T4). A self-learning
route will not always hold the fact: it may have to **infer** it from the term's other edges (link prediction). Does a
recalled inferred fact help, hurt, or neither, and does marking it as inferred change that?

### Items, frames, scoring

- **Items:** RD-4b's T4 held-out set (`t4-heldout-smollm2-v1`: 341 terms; 407 property and 407 entailment items per
  seed; paraphrase from the property items' rewordings).
- **Gold frames:** each run's own E11 oracle frames (as RD-4b).
- **Scoring:** RD-4b's — selective recall by the item's own relation (**the oracle router**, so routing is held fixed),
  the context-free null (`--null plain`), raw choices recorded, PMI choices read.
- **The link predictor (`infer_frames`, no model):** for each relation r of a term's gold frame, r's fillers are predicted
  from the frame's *other* edges: the ontology entries that share those edges (held-out and synthetic entries excluded)
  are scored by the summed idf of the shared edges; the 20 best vote for their own fillers of r, weighted by their
  score; with no neighbour, r's most frequent fillers. m_r lines are kept, m_r = the mean number of r-edges per frame
  holding r (rounded, 1–3): a per-relation constant, so the gold count is not leaked. The type prior's pools (P(r | atom))
  are the ontology statistics this uses; the typing itself adds nothing for a relation fixed in advance.
- **Inductive KGE:** C6gi (`kge_inductive`, E9) is TransE trained without the held-out entries and folded in; no trained
  inductive table exists on disk for T4 (`kge-transe-inductive-d*.pt` absent), and training one is not a cheap test.
  Skipped; noted here.
- **The predictor's scores on the 341 terms (computed before the predictions below were written; no model):**

  | relation | precision | recall | lines |
  |---|---:|---:|---:|
  | is_a | 0.119 | 0.123 | 2 |
  | has_role | 0.375 | 0.237 | 2 |
  | has_functional_parent | 0.209 | 0.198 | 1 |
  | contains_element / charge / branch | 0.816 / 0.972 / 0.997 | 0.888 / 0.972 / 0.997 | 3 / 1 / 1 |
  | all edges | 0.547 | 0.584 | — |

  The items ask is_a (238), has_role (130) and has_functional_parent (39). **The inferred line names the gold answer
  for 0.157 of the property items** (is_a 0.122, has_role 0.215, has_functional_parent 0.179). A wrong inferred line
  never names one of the item's three distractors (0 of 343): wrong lines are plausible fillers outside the candidate
  set, as RD-4b's `explicit-rel-wrong` lines were.

### Conditions (`e14_inferred recall`; 9)

- `none` (iv);
- **(i) `explicit-rel:oracle`:** the gold lines of the asked relation;
- **(ii) `explicit-rel:minus`:** the gold frame without the asked relation, under selective recall: an empty recall
  (`recall(X):` / `- (nothing recalled)`);
- **(iii) `explicit-rel:infer-nn`:** the predictor's lines for the asked relation, written as gold lines read
  (confidence tag "(1.00)"); **`explicit-rel-marked:infer-nn`:** the same lines with the tag replaced by "(inferred)";
- secondary, the known edges shown: `explicit-known:oracle` (the gold lines of the other relations; the model must infer
  by itself), `explicit-known-infer:infer-nn` and `explicit-known-infer-marked:infer-nn` (those, then the inferred lines);
- `definition`.

### Hosts

SmolLM2-360M C5, seeds 1–3, T4 held-out (the RD-4c main hosts).

### Primary (Holm over 3 contrasts × 3 tests: property, paraphrase, entailment)

- **(a)** `explicit-rel:infer-nn − explicit-rel:minus`: inferred lines against nothing;
- **(b)** `explicit-rel-marked:infer-nn − explicit-rel:infer-nn`: marking;
- **(c)** `explicit-rel:infer-nn − explicit-rel:oracle`: inference against the truth.

### Secondary (unadjusted)

- `explicit-rel:minus − none` (the empty recall header), `explicit-rel:oracle − none`, `explicit-rel:infer-nn − none`,
  `explicit-rel-marked:infer-nn − explicit-rel:minus`;
- `explicit-known:oracle − none` and `− explicit-rel:minus`; `explicit-known-infer:infer-nn − explicit-known:oracle`;
  `explicit-known-infer-marked − explicit-known-infer`; `definition − explicit-rel:infer-nn`;
- the primary contrasts on the items whose inferred line names the gold answer ("inferred right") and on the others
  ("inferred wrong"); the primary on raw choices.

### Predictions (written after the predictor's scores above were known, before any model scored these conditions)

1. **(a) is not positive on property.** 16% of the items get the gold answer named (about +0.24 on those, RD-4b's
   oracle gain); 84% get a wrong line, and RD-4b's wrong lines cost −0.09 on property. Expected about −0.04 on property
   (95% interval likely covering 0), about −0.04 on entailment, about 0 on paraphrase.
2. **(b) is about 0** (|·| ≤ 0.02 on every test, not significant): a 360M host does not read the confidence tag (§14:
   it answers from surface cues).
3. **(c) is negative on property and entailment,** about −0.25 and −0.15 (the oracle gains of RD-4b, lost).
4. Splits: on "inferred right" items (a) is about +0.20 on property; on "inferred wrong" items about −0.09.
5. `explicit-known:oracle − none` is negative on property (about −0.05: 10–15 lines of real chemical names, RD-4b's
   dilution), and adding the inferred lines to the known ones changes little (|·| ≤ 0.03).

**Alternative stated in advance:** (a) positive (Holm) on property would mean that a line of the asked relation helps
even when it is wrong more often than right, i.e. the model uses the line's *form* (which relation, which kind of filler)
rather than its content; the "inferred wrong" split would then also be ≥ 0.

### Decision rule

- **If (a) is positive (Holm) on property,** inferred facts at this precision may be recalled unmarked in the
  self-learning route; an E13 amendment adding the predictor to the recall step is drafted for the author (not queued).
- **Otherwise,** inferred facts are not recalled without a precision gate. The "inferred right" split estimates the gain
  a better predictor could reach, and (c) the cost of the present one; the precision at which (a) would turn positive is
  reported from the two splits (the mixture point), as the target for a predictor.
- **If (b) is positive (Holm),** inferred lines are marked in any future use. If (b) is about 0, marking is free but
  ineffective on this host, and is kept as the format for larger hosts to test (RD-4c's Qwen3 hosts).

### Smoke seen before this registration (labelled; not a result)

SmolLM2-360M C5 s1 on the first 6 terms (20 items), CPU, all 9 conditions, context-free null; the predictor's lines,
the empty recall, the marked tag and the known-edge recalls rendered as designed (`explicit-known*` ≈ 240–290 context
tokens, `explicit-rel:infer-nn` ≈ 60). Its numbers are not results.

### Cost

RD-4b's T4 jobs took 0.042–0.047 GPU-h each (`.jobs/`, 8–9 conditions). RD-14 has 9 conditions, two of them long
(the `explicit-known*` recalls): about 0.07 GPU-h per seed, **≈ 0.2 GPU-h**; the report is CPU lane.

---

## 3. Queue, outputs, total

- `queue-commands-addendum11-cheap.sh`: GPU jobs at priority **54.4959997** (`e14-lgcf-SmolLM2-360M`,
  `e14-lgcf-Qwen3-0.6B`, `e14-lgcf-Qwen3-1.7B`, `e14-rd14-t4-heldout-SmolLM2-360M-C5-s1..3`), CPU-lane reports at
  **54.49599987** (`e14-lgcf-report`, `e14-rd14-report`).
- Outputs: `experiments/e14-logic/runs/lgcf/<host>/` (predictions under `~/data/vsa-llm/e14/runs/lgcf/`),
  `report/lgcf/`; `runs/rd14/t4-heldout/SmolLM2-360M-C5-s{1,2,3}/`, `report/rd14/`.
- **Total: about 0.9 GPU-h** (0.7 + 0.2), under the 2 GPU-h asked for.
- Items on disk: `~/data/vsa-llm/e14/items/trees-cf-s1.jsonl.gz`; manifest `experiments/e14-logic/items/manifest-cf-s1.json`.

## 4. Changes after this commit

None yet.
