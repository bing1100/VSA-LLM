# Results of the night of 2026-10-20/21: routers, readers, trees from prose, counterfactual chains, retrieval, inferred facts, E5, C5teach, E7's gate

**Recorded:** 2026-10-21 (morning) by workplace-8e.

**Sources:** official reports from the queue's report jobs:
- `experiments/e14-logic/report/{rd4d, rd3b, treetext, lgcf, rd14, rd13, lgd2}/`, `runs/x7b/`;
- `experiments/e5-explainability/report/scale-v1/`;
- `experiments/e4-small-lm/analysis/scale-v1-toolkit-c5teach{,-teachref}/`;
- `experiments/e7-self-authoring/runs/general-v1/gate/g3-d72.json`.

**Pre-registrations:** `experiments/e14-logic/preregistration.md` §19 and §20 (addenda 9–11);
`experiments/e5-explainability/preregistration-amendment-1-scale-v1.md`;
`experiments/e4-small-lm/preregistration-scale-v1-toolkit-status-2026-10-20.md`;
`experiments/e7-self-authoring/preregistration-2026-10-20-reschedule.md`.

**Conventions:** accuracy differences, positive = better; \* = Holm-significant within the pre-registered family;
paired bootstrap over items, seeds pooled unless stated. All jobs ran without failure.

## 1. RD-4d: learned relation routers, the routed wording held out (§19)

**SmolLM2-360M C5, 3 seeds. Property accuracy.**

| Condition | T4 real chemicals | T5 new words |
|---|---:|---:|
| none | 0.614 | 0.190 |
| labelled selective recall, gold frames (ceiling) | 0.862 | 0.845 |
| **few-shot router**, gold frames | **0.845** | 0.400 |
| naive-Bayes router, gold frames | 0.829 | 0.262 |
| `lex` router, gold frames | 0.787 | 0.618 |
| `lm` router, gold frames | 0.780 | 0.307 |
| memorising router, gold frames (upper bound) | 0.862 | 0.839 |
| **few-shot router + type-prior frames (realistic)** | **0.724** | 0.378 |
| `lex` router + type-prior frames | 0.714 | 0.589 |
| definition text | 0.734 | 0.857 |

**Router accuracy** (the asked relation chosen, among items whose frame holds it): few-shot 0.935–0.945 on T4,
0.358 on T5; naive Bayes 0.278 on T5; `lex` 0.79–0.89 on T4, 0.39 on T5; `lm` 0.21 on T5; memo 0.997.

**Primary (Holm over 6 per track):**
- T4: few-shot routed − labelled: property **−0.018\***, paraphrase −0.015\*, entailment +0.003 (n.s.). Realistic
  pipeline − none: +0.109\*, +0.136\*, +0.103\*.
- T5: few-shot routed − labelled: **−0.445\***, −0.235\*, −0.130\*. Realistic − none: +0.188\*, +0.237\*, +0.156\*.

**Secondary:** on T4 the realistic pipeline is 0.010 below the definition text on property (p 0.10) and 0.043 below
on paraphrase\*. Few-shot beats `lm` by +0.065\* with gold frames on T4.

**Reading:**
- **Routing is solved on real chemistry.** The routing cost fell from −0.083 (RD-4c) to −0.018. The realistic
  pipeline now matches the definition text on property.
- **On T5 every learned router fails.** The held-out question wording does not resemble the demonstrations, and the
  routers pick a generic relation. The memorising router reaches the ceiling, so the gap is generalisation across
  wordings, not router capacity. Word overlap remains the best T5 router.
- Next (RD-4e, addendum 12, registered 2026-10-21): paraphrase-augmented demonstrations, abstention, top-2, and the
  size rule "whole frame when the frame is short".

## 2. RD-3b: a ReAct reader with a precision filter (§19)

| T4 stated-edge F1 (precision) | SmolLM2-360M | Qwen3-1.7B |
|---|---:|---:|
| `react` | 0.337 (0.194) | 0.320 (0.192) |
| **`react-strict`** | **0.433 (0.307)** | **0.446 (0.351)** |
| type prior | 0.573 (0.367) | 0.573 (0.367) |
| Claude teacher | 0.603 (0.672) | — |

**Primary (Holm per track over hosts × 2):** `react-strict − react` +0.096\* (360M) and +0.126\* (1.7B) on T4;
`react-strict − typeprior` −0.141\* and −0.127\*. On T5 `react-strict` is the best reader (0.923 / 0.942; +0.026\*
over the type prior on Qwen3-1.7B).

**Predictions:** `react-strict > react` on T4, **yes**; still below the type prior, **yes**; precision ≥ 0.28, **yes**.

**Reading:** the filter does what it was built for and the reader still loses to a frequency table on real text.
Labelling is not the T4 bottleneck; this line is closed.

## 3. Trees read from prose (addendum 9) — not worth queuing more

72 synthetic decision trees (depths 2–4), each rendered as templated prose and as paraphrased prose; the host emits
the executor's structured form; the decoded tree is executed on 24 records and compared with the gold tree.

| Host | Style | Parse rate | Exact tree | **Leaf accuracy** |
|---|---|---:|---:|---:|
| SmolLM2-360M | templated | 0.986 | 0.250 | 0.451 [0.369, 0.534] |
| SmolLM2-360M | paraphrased | 0.458 | 0.000 | 0.095 [0.054, 0.142] |
| Qwen3-0.6B | templated | 0.903 | 0.528 | 0.667 [0.575, 0.755] |
| Qwen3-0.6B | paraphrased | 0.806 | 0.042 | 0.218 [0.150, 0.289] |
| Qwen3-1.7B | templated | 0.917 | 0.722 | **0.822 [0.741, 0.899]** |
| Qwen3-1.7B | paraphrased | 0.694 | 0.042 | **0.237 [0.168, 0.310]** |

Baselines: a regex reader of the templated style 1.000; the first leaf 0.231. By depth, Qwen3-1.7B templated: 1.00 /
0.85 / 0.62 at depths 2 / 3 / 4; paraphrased 0.38 / 0.28 / 0.05.

**Decision rule** (≥ 0.80 paraphrased leaf accuracy with CI ≥ 0.70 for the best host): **not met; verdict "not
worth queuing more".** Predictions: the templated numbers for Qwen3-1.7B (> 0.9) and the paraphrased range
(0.6–0.8) were both too optimistic; the SmolLM2 predictions held.

**Reading:** these hosts imitate a format; they do not read logic from prose. The line "decode trees from text, then
reason with the executor" stops at ≤ 1.7B. The executor remains useful when the tree arrives structured (written by
a larger model or a person).

## 4. LG-CF: consistency under chains of what-if edits (addendum 11)

300 trees (depths 2, 3, 4, 6) with 8 successive edits each (an input attribute changes, a test is retargeted, or two
branches are swapped). Gold at state k is the exact executor's leaf. Six answer options. Conditions: `chain-text`
(the original record, the policy, and the k edits listed; the model answers in context); `chain-relay` (the edited
record and the executor's path on the edited state; the model relays); `chain-text+exec` and `+conclusion` (the full
history plus the executor's path or its one conclusion line).

| Accuracy at k = 0 / 1 / 2 / 4 / 8 | SmolLM2-360M | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|
| executor (no model) | 0.993 / 0.997 / 0.997 / 0.997 / 1.000 | same | same |
| `chain-relay` | 0.993 / 0.997 / 0.997 / 0.997 / 1.000 | same | same |
| `chain-text` | 0.247 / 0.277 / 0.270 / 0.250 / 0.243 | 0.233 / 0.277 / 0.253 / 0.250 / 0.273 | 0.240 / 0.307 / 0.273 / 0.257 / 0.280 |
| `chain-text+conclusion` | 0.847 / … / 0.890 | 0.990 / … / 1.000 | 0.993 / … / 1.000 |
| `chain-text+exec` | 0.797 / … / 0.783 | 0.983 / … / 0.997 | 0.993 / … / 0.997 |
| **consistency, all of k = 1, 2, 4, 8 right** | relay 0.997, text 0.050 | relay 0.997, text 0.033 | relay 0.997, text 0.037 |

**Primary, `chain-relay − chain-text` at k = 8, Holm over hosts:** **+0.757\*** [+0.710, +0.803], **+0.727\***
[+0.673, +0.777], **+0.720\*** [+0.667, +0.770] (p 0.0015 each; n = 300).

**Predictions:**

| Prediction | Result |
|---|---|
| 1. `chain-text` falls with k (k = 8 at least 0.05 below k = 1) | **no**: it is at chance (0.167–0.31) from k = 0 on, with no slope |
| 2. `chain-relay` ≥ 0.95 everywhere | **yes** (0.993–1.000) |
| 3. primary positive on all hosts by ≥ +0.50 | **yes** (+0.72 to +0.76) |
| 4. consistency relay ≥ 0.90, text ≤ 0.10 | **yes** (0.997 vs 0.03–0.05) |
| 5. long-history relay worse on SmolLM2, ≈ 0 on Qwen3 | **yes**: SmolLM2 −0.217\* with the full history; Qwen3 +0.003 |
| 6. SmolLM2 anchors to the original leaf | **no**: 0.24 of answers, near the 1/6 chance rate |

**Decision rule.** The primary is positive on all three hosts. Prediction 1 did not hold as worded: `chain-text`
does not *decline*, because it is already at chance at k = 0 (these hosts cannot run a tree given in prose, as LG-D
found: 0.37 with four options). The pre-registered alternative therefore applies: the result is stated as **"the
executor's advantage on decision trees persists under edits, and keeps a chain of what-if questions consistent
(99.7%) where in-context answering is inconsistent (3–5%)"**, not as "in-context answering drifts with chain
length". The stronger drift claim would need a host that can run the tree at k = 0 at all (a ≥ 4B host).

**Reading:**
- The consistency gap is real and large, but its source is that the model never executes the policy in context, at
  any k. The executor supplies the whole of the computation; the model supplies the relay.
- On the Qwen3 hosts a single conclusion line appended to the full history is enough (0.99+). SmolLM2-360M needs the
  short path interface (0.78–0.89 with the long history).
- Every host follows the executor at 0.999–1.000 when it is wrong (n = 5), so the executor's soundness is the claim's
  load-bearing part.

## 5. RD-14: inferred facts under selective recall (addendum 11)

Missing facts predicted by an idf-weighted neighbour vote over the ontology (precision 0.547 / recall 0.584 overall;
`is_a` 0.119, `has_role` 0.375; structural relations 0.82–1.00), recalled with the oracle router on T4 (SmolLM2-360M
C5, 3 seeds). The inferred line names the gold answer for 15.7% of property items and never names a distractor.

| Condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| none | 0.614 | 0.544 | 0.767 |
| gold minus the asked relation (empty recall) | 0.650 | 0.615 | 0.813 |
| **inferred, unmarked** | **0.659** | 0.613 | 0.853 |
| inferred, marked "(inferred)" | 0.655 | 0.618 | 0.852 |
| gold (labelled selective recall) | 0.862 | 0.841 | 0.928 |

**Primary (Holm over 9):** inferred − empty: property +0.009 (n.s.), paraphrase −0.002 (n.s.), **entailment
+0.040\***; marked − unmarked: |Δ| ≤ 0.004 (n.s.); inferred − gold: −0.204\*, −0.228\*, −0.075\*.

**Split by whether the inference was right:** right (n 192): +0.234\* property; wrong (n 1,029): −0.033\*. The
break-even precision on the asked relation is therefore about 0.12–0.15; the predictor's 0.16 hit rate sits at it.

**Decision rule:** (a) not positive on property → **no recall of inferred facts without a precision gate; no E13
amendment.** Marking changes nothing, so it is kept as a format only.

## 6. RD-13: retrieve-then-read with Wikipedia (addendum 10)

**Tool:** MediaWiki API, cached by revision id (`~/data/vsa-llm/wikipedia/t4-heldout-v1/`, CC BY-SA 4.0); pages for
332 of 341 T4 held-out concepts (median 1,010 words). No dump.

**Stated rate (CPU, string-matching finder, lower bound):** ChEBI definition **0.116**; Wikipedia, 300 words
**0.087**; whole page **0.120**. Structural relations 0% in every text; `is_a` 0.35 → 0.18; `has_role` 0.05 → 0.15.
**The pre-registered rule fired: more text is not the lever on T4.**

**GPU arms** (SmolLM2-360M C5 ×3; Qwen3-1.7B C0′ ×1; property):

| Condition | SmolLM2-360M | Qwen3-1.7B |
|---|---:|---:|
| none | 0.614 | 0.729 |
| definition text | 0.734 | 0.812 |
| 300 words of Wikipedia in the prompt (RAG) | — | **0.166** |
| store from Wikipedia, type-prior reader, `lm` router (realistic) | 0.524 | 0.622 |
| store from Wikipedia, stated facts only, labelled | 0.692 | 0.749 |
| store from the definition, type-prior reader, `lm` router (reference) | 0.713 | 0.773 |

**Primary (Qwen3-1.7B, Holm over 9):** store − none **−0.107\*** (property), −0.096\* (entailment); store − RAG
+0.456\*; RAG − definition **−0.646\***.

**Reading:**
- The Wikipedia store hurts because its frames are wrong: the type prior on Wikipedia text reads 10.7 edges per
  concept at precision 0.05, and wrong lines cost more than none (RD-4b).
- 300 words of encyclopaedia text in the prompt collapses the PMI-scored property choice (0.17): the long context
  swamps the term. Paraphrase, which the PMI null does not penalise the same way, rises (+0.12).
- Only the stated Wikipedia facts, recalled selectively, are neutral to slightly positive (+0.02 to +0.03).
- **Closed.** On T4 the lever is the precision of what is read, not the amount of text. If retrieval returns, the
  finder (alias coverage for common names) comes first.

## 7. LG-D2 and X7b (§19)

**LG-D2, trees on noisy records** (SmolLM2-135M; 10% / 30% of record fields missing or wrong): the executor's path
beats the tree in text by **+0.670\*** / **+0.488\***; soft matching adds +0.002 / +0.003 (n.s.). The model follows
the executor 1.000, including when the executor is wrong on a corrupted field. Noise costs the executor exactly what
it costs on affected records (−0.42 at 10%) and nothing elsewhere.

**X7b, one trained store as wide as the ensemble:** slot decode accuracy, held-out entries: single 256-wide stores
0.891–0.918; three-store ensemble 0.979; **one 768-wide trained store 0.992** (+0.013\* over the ensemble); an
*untrained* 768-wide store 0.998; untrained 256-wide 0.927. **X7's ensemble gain was width**, as the smoke suggested
(findings 2026-10-20 §6). Training costs decoding −0.006 at width 768 and −0.022 at 256.

## 8. E5 on the 125M checkpoints (amendment 1)

- **E5.1 faithfulness (H-F):** removing the k attention-top edges costs more than removing k random ones: +0.021 /
  +0.028 / +0.036 nats at k = 1 / 2 / 4 (C5, 3 seeds, tight intervals), but the sufficiency contrast is negative
  (−0.026 to −0.041) and no seed reaches Holm significance under the pre-registered per-seed test. **Partial.**
- **E5.2 sense alignment (H-C):** attention picks the gold sense 0.444 against 0.845 for WordNet's first sense and
  0.812 for SemCor's most frequent sense (held-out 0.382 vs 0.783). **Fails.**
- **E5.3 judge (Claude, cap USD 25):** neighbour pairs, rare words: C5 8/40 against C2 4/40 (p 0.17) and C0 2/40
  (p 0.044); head-to-head preference C5 over C0 8–1 (p 0.020), over C2 7–2 (p 0.090); edge explanations 5/40 against
  5/40 (null). κ 0.82–0.89, calibration 0.8–1.0. The held-out neighbours study was refused by the cap (USD 4.32 left
  of 25). **Weakly positive against C0, null against the free table.**
- **E5.4 zero-shot insertion (H-E):** a structure-only vector for an unseen concept lifts property 0.518 → 0.672 and
  entailment 0.721 → 0.846 on the held-out set (3/3 seeds Holm-significant for property and entailment); on the
  synthetic set 0.213 → 0.518 and 0.408 → 0.715. The graph-projection baseline matches it (0.674 / 0.832), so the gain
  is the frame's content, not the composer. C0 cannot be given a vector (0.532). **The clearest E5 positive.**
- **E5.5 frequency (H-D):** a ridge probe recovers log frequency from C5's concept rows with R² 0.189 against 0.128
  from the surface baseline and 0.117 at initialisation; C2's rows 0.052. **Fails**: training adds frequency
  information to the rows.

**Reading:** E5 confirms E9's picture at 125M: the composed vector carries usable content for new concepts (E5.4),
the attention over edges is only weakly faithful (E5.1), and nothing about senses or frequency is cleaner than the
baselines (E5.2, E5.5).

## 9. C5teach: a Claude-read ontology at 50M (toolkit phase 1b, 1 seed)

Teacher build: 24,670 synsets, 987 calls, 0 errors, USD 68.05; ontology of 11,800 teacher entries, parse rate 0.969.

| (condition − C0) / C0 | C5 | C5dev | **C5teach** |
|---|---:|---:|---:|
| all | −0.38%\* | −0.36%\* | −0.41%\* |
| inside terms | −1.23%\* | −1.49%\* | −1.56%\* |
| after terms | −0.90%\* | −0.94%\* | −0.90%\* |
| after held-out terms | −0.05% | −0.18% | +0.02% |
| after rare terms | −2.65%\* | −2.52%\* | −3.12%\* |

C5teach − C5: all −0.02%, inside **−0.34%\*** [−0.67, −0.01], after +0.00%, after held-out +0.07%. On the teacher's
own reference strata: after teacher-trained terms −1.32% (C5teach) against −1.23% (C5@teachref); after
teacher-held-out terms +0.02% against −0.05%.

**Reading:** the Claude-read ontology changes the loss inside the terms it describes and nothing after them, at one
seed. It does not move any gate stratum. Consistent with RD-4c, where Claude's frames did not beat the type prior's.
**C5teachF (the full variant, about USD 342) is not worth building.**

## 10. E7: gate G3 failed; D7.2 and D7.3 are not queued

D7.1 ran (9 jobs) with the CPU follow-ups (Hearst, random frames, Claude teacher: 999 concepts, USD 9.07). The new
G3 rule compares each author's edge precision on the masked gold with a degree-matched random high:

| Arm | Masked concepts authored | Verifiable edges | Precision [95% CI] | Random high | Coverage | Verdict |
|---|---:|---:|---|---:|---:|---|
| self (Qwen2.5-0.5B) | 64 | 16 | 0.062 [0.011, 0.283] | 0.243 | 0.282 | FAIL (edges, above-random) |
| teacher (Claude) | — | 277 | 0.069 [0.044, 0.105] | 0.243 | 0.751 | FAIL (above-random) |

**Both arms are at or below the random high.** The self-authoring host writes almost no edges for masked concepts
(36 edges over 26 concepts), and the teacher's edges on this general WordNet track are no more precise than
degree-matched random ones. D7.2 was not queued; D7.3 depends on it. The Claude judge (cap USD 50; informational, not
part of G3) was interrupted at USD 22.66 by a session restart and resumed from its cache.

**Reading:** self-authoring is closed on this track. Together with RD-11 (reading in rounds), RD-3 (Claude only
+0.03 F1 over the type prior) and C5teach, the evidence is consistent: on real text, frames written by a model —
small or large — are not more precise than the ontology's own statistics.

## 11. What follows

- **Continue:** RD-4e (routing that generalises; addendum 12, registered and queued 2026-10-21); E13 (lands 25–30
  October); the consistency result and E5.4 go to the claims ledger (numbers sent to workplace-9a).
- **Stop:** trees from prose at ≤ 1.7B; retrieval over more text on T4; recall of inferred facts; E7 D7.2/D7.3;
  C5teachF.
- **Caveats to carry:** LG-CF's in-context arm is at chance at k = 0, so the consistency claim is about execution,
  not drift; tree-text, LG-CF and RD-14 are one item set each; C5teach is one seed; E5.3's held-out study did not run.
