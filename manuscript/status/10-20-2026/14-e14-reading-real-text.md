# 14 — E14, part 1: reading facts from real text, and using them

> **Update, 20 October (evening).** Tier F was approved on 20 October. RD-4d (learned routers) and RD-3b (a stricter
> ReAct reader) are queued and should land on 21 October; see the digest, §5.

**Overview.** This file follows one question through seven explorations: why does reading facts from real definitions
fail, and what would make it work? We split a fact's path into four stages:
1. **stated** — the text says it;
2. **found** — the reader locates the partner concept;
3. **labelled** — the reader picks the right relation;
4. **used** — the language model benefits from the stored fact.

Explorations 14.1–14.3 locate the losses at stages 1–3. Explorations 14.4–14.6 test stage 4. Exploration 14.7 tests
the full read–train–read cycle.

**Terms used throughout.**
- **T4:** real chemistry (ChEBI definitions and PubMed text). **T5:** a synthetic glossary of invented words.
- **Frame:** the set of (relation, partner) facts for a term, e.g. (*is a*, *monoterpene*).
- **Gold frame / oracle reader:** the ontology's true frame. **Readers** build frames from text:
  - **linker** — string matching;
  - **type prior** — partner types plus relation statistics;
  - **host** — the language model itself;
  - **teacher** — Claude.
- **Implicit route:** the frame is composed into the term's vector (the method, C5).
- **Explicit route (recall):** the frame is written as short lines of text before the question.
- **C5:** SmolLM2-360M fine-tuned with the method. **C0′:** fine-tuned without it.
- **Items:** E11's held-out test questions.
  - **property:** pick the term's partner for a relation;
  - **paraphrase:** the same pick, consistent across rewordings;
  - **entailment:** a fact implied by the frame.
- **Statistics:** differences are accuracy points (positive = better), paired bootstrap over items, seeds pooled.
  An asterisk (\*) marks significance after Holm correction within the pre-registered family.

Everything here was pre-registered in `experiments/e14-logic/preregistration.md` before it ran.

---

# 14.1 — Where reading loses facts: coverage, dictionary and structure (RD-1, RD-2, RD-5, RD-6)

**Status:** done (CPU) · **When:** 2026-10-17 · **Sources:** `manuscript/findings/2026-10-17-e14-tier-p-cpu.md`; `experiments/e14-logic/runs/{rd1a-audit, rd1b-audit-t4, rd1b-audit-t1, rd2-dictionary, rd5-bootstrap, rd6-structure}/`

## Research gap
E11 found that frames read from real ChEBI definitions do not help the model. We did not know why. The candidates were
an ontology that is too small, a reader that is not smart enough, coverage too narrow to "pick up steam", or a text
that simply does not state the facts.

## Intuition
If a definition never states a fact, no reader can recover it. Measuring how much of each term's gold frame the text
states puts a ceiling on every reader. Measuring what the text *does* say tells us whether the ontology is the limit.

## Methodology
Four descriptive audits, with no confirmatory test:
- **RD-1a:** an edge counts as stated if its partner's name or alias appears in the definition.
- **RD-1b:** Claude labels every statement in a definition into one of six categories (expressible by a relation,
  structural, a property no relation covers, …).
- **RD-2:** the partner finder runs with larger dictionaries; also a shrinking curve.
- **RD-5:** a simulation of "snowball" reading, in which read terms become linkable partners for later reading.
- **RD-6:** structure readers take facts from structured fields (formula, charge, ancestry).

## Experiment
- 341 T4 held-out definitions with 2,744 gold edges.
- RD-1b: 100 definitions each from T4 and T1 (MeSH scope notes), $2.19 of Claude calls.
- RD-2: 33,941 surfaces, then 65,514 with ChEBI synonyms.
- RD-5: 5,000 definitions × 5 rounds × 3 seeds.

## Results
- **Stage 1 is the main loss.** Definitions state only **11.7%** of gold edges:
  - `is_a` 35%;
  - `has_role` 5%;
  - `contains_element`, `charge` and `branch` 0%. These three are 42% of all edges.
- **The ontology is too narrow on medical text, not on chemistry.**

  | | T4 (ChEBI) | T1 (MeSH) |
  |---|---:|---:|
  | Statements a relation can express | 54% | 30% |
  | Properties no relation covers | 6% | 42% |
- **A bigger dictionary does not help:** gold partners found 0.116 → 0.117. Recall scales linearly with the share of
  partners the text names.
- **No snowball:** the dictionary grows 0.25 → 0.28, then stalls. Edge precision is 0.22–0.28. Read terms are rarely
  other terms' partners.
- **Structure readers supply the structural 42%** at precision and recall 1.00. This holds by construction: the
  ontology was built from the same fields.

## Limitations
- String matching under-counts paraphrased facts, so 11.7% is a lower bound.
- RD-1b's labels come from one model, with no human agreement check.
- RD-5 tests one snowball mechanism with one reader.

## Next steps
Closed as diagnostics. They motivate relation discovery (14.3), better readers (14.2, 14.6) and RD-4b's stated-only
items (14.5).

---

# 14.2 — Is labelling the relation the bottleneck? A ReAct-style reader (RD-3)

**Status:** done · **When:** 2026-10-19 (GPU) · **Sources:** `experiments/e14-logic/report/rd3/`; `manuscript/findings/2026-10-20-e14-tier-p-gpu.md` §2

## Research gap
Once a reader finds a partner, it must pick the relation, e.g. *has role* rather than *is a*. We asked whether a smarter,
step-by-step reader, of the kind ReAct agents use, labels relations better than string matching or statistics.

## Intuition
A model that first finds candidate mentions, then rejects bad ones ("none of these"), then chooses a relation and checks
its type should make fewer labelling errors than a one-shot matcher.

## Methodology
- Readers compared on every *stated* gold edge:
  - linker, type prior and host (from E11's frames);
  - `mc-rel`: scored choice over relations, with a relation prior;
  - `react`: exact mentions → fuzzy candidate spans with "none of these" → relation → typing check;
  - Claude as teacher (T4 only).
- Metric: stated-edge F1, per term.
- Primary contrasts: `react − linker` and `react − typeprior`, Holm per track.
- Hosts for the new readers: SmolLM2-360M and Qwen3-1.7B-Base.

## Experiment
248 T4 terms with at least one stated edge; 300 T5 words.

## Results

| Reader | T4 F1 (precision) | T5 F1 |
|---|---|---|
| type prior | 0.573 (0.37) | 0.916 |
| linker | 0.101 (0.06) | 0.518 |
| `react`, 360M / 1.7B | 0.337 / 0.320 (0.19) | 0.917 / 0.936 |
| Claude (teacher) | **0.603** (0.67) | — |

- `react` beats the linker everywhere: +0.22 to +0.42 F1\*.
- `react` loses to the type prior on T4 (−0.24\*) and ties it on T5. Its fuzzy step over-proposes partners, so its
  precision is low.
- **A 5× larger host does not help, and Claude gains only +0.03 F1 over the type prior.** Labelling is not the T4
  bottleneck; coverage is (14.1).

## Limitations
Our `react` implementation lacks a precision filter. The hosts are base models, not instruction-tuned.

## Next steps
Proposed, not queued: a typing filter for `react`, and an instruction-tuned host.

---

# 14.3 — Reading in rounds, with relation discovery (RD-11)

**Status:** done (CPU, Claude as reader, $11.22) · **When:** 2026-10-17 · **Sources:** `manuscript/findings/2026-10-17-e14-concat-trees-reading-cycle.md` §3; `experiments/e14-logic/runs/rd11-cycle-{t4,t1}/`

## Research gap
The author's hypothesis: reading is iterative. Repeated passes, reflection on the available relations and new relation
types should grow knowledge over rounds, even when gold is out of reach at first.

## Intuition
A reader that may propose `NEW` relations, and a reflection step that turns recurring proposals into relation types,
should recover relations the ontology hides from it. It should also capture facts the ontology cannot express.

## Methodology
Each round:
1. Claude reads every definition twice, with the relation order shuffled.
2. Facts both readings agree on are kept.
3. Reflection clusters the `NEW` proposals and accepts relation types with ≥ 8 supporting proposals.

There are three rounds. On T4, four gold relations are hidden at the start, as a rediscovery test.

## Experiment
160 T4 ChEBI definitions; 120 T1 MeSH scope notes.

## Results
- **T4:**
  - 2 of 4 hidden relations rediscovered by name: `has_functional_parent` (recall 0.33, precision 0.23–0.26) and
    `has_part` (recall 0.13);
  - gold recall 0.157 → 0.162, against a text ceiling of 0.232.
- **T1:**
  - six new relation types accepted;
  - the share of `NEW` statements fell 0.72 → 0.32, and captured facts per note rose 2.04 → 2.40.
- **Consolidation (two readings agreeing) added little:** +0.002 precision on T4, +0.037 on T1.

## Limitations
- Gold recall measures only knowledge the ontology already has, so new relations need a downstream test.
- Name matching to the hidden relations is conservative.
- The first-pass analysis wrongly counted rediscovered names as visible; we corrected it before the write-up.

## Next steps
The use of these frames is tested in 14.7 (RD-12, RD-12b, RD-12c).

---

# 14.4 — Can stored facts be used? Explicit recall against implicit injection (RD-4, RD-7)

**Status:** done · **When:** 2026-10-19 (GPU) · **Sources:** `experiments/e14-logic/report/rd4/`; `manuscript/findings/2026-10-20-e14-tier-p-gpu.md` §1; `experiments/e14-logic/report/diagnostics-2026-10-20.md`

## Research gap
This is stage 4, the decisive question for self-learning. E11 showed that composing even the gold frame into a term's
vector does not help on real chemistry. We did not know whether the facts are useless to the model, or only the
injection route is.

## Intuition
A model may use facts it can *read* (text before the question) far better than facts hidden in a vector. This was the
lesson of E12's explicit recall on role-swapped twins.

## Methodology
- **Conditions:**
  - `none`;
  - `implicit:<reader>`;
  - `explicit-sym:<reader>` (the frame as recall lines);
  - `explicit-vsa` (the frame stored in the model's own VSA store and decoded);
  - `both`;
  - `definition`;
  - noisy frames (10/30/50% wrong partners).
- **Primary contrasts** (Holm over 9 per track): (a) `explicit-sym:oracle − none`, (b) `− implicit:oracle`, (c) the
  same with stated frames.
- **Decision rule:** if (b) is positive on T4 property, recommend explicit recall for self-learning.

## Experiment
SmolLM2-360M C5 (3 seeds) and C0′ (1 seed), on T4 held-out (341 terms; 407 property items per seed) and T5 new words
(300; 1,529 property items per seed).

## Results
- **New words (T5):** explicit recall helps a great deal. Property +0.31\* against `none` and +0.24\* against implicit
  injection; implicit alone gives +0.07. The decoded VSA store matches the symbolic record (±0.01).
- **Real chemicals (T4):** explicit recall *lowered* property accuracy (−0.10\*) but raised paraphrase (+0.12\*) and
  entailment (+0.11\*).
- **The decision rule did not fire.**
- **Exploratory diagnosis:**
  - the recall named the gold answer for 399 of 407 items;
  - the scoring baseline prompt (PMI null) also read the recall, cancelling part of its evidence;
  - a 15-line list of chemical names diluted what the model knew.

## Limitations
The scoring penalised every context condition, and whole-frame recall confounded "the facts" with "a long list". C0′
had one seed.

## Next steps
Tested directly in 14.5.

---

# 14.5 — Selective recall: only the relation the question asks about (RD-4b)

**Status:** done · **When:** 2026-10-20 · **Sources:** `experiments/e14-logic/report/rd4b/`; `manuscript/findings/2026-10-20-e14-followups.md` §1

## Research gap
14.4 left two explanations for the T4 drop: a scoring artefact and list dilution. We needed to separate them, and to
test a version of recall built to avoid both.

## Intuition
If the model is distracted by irrelevant facts, giving it only the lines for the asked relation should help. If the null
prompt cancels evidence, a null without the context should restore it.

## Methodology
- **Conditions:**
  - `explicit-rel:<reader>`: only the asked relation's lines;
  - a wrong-partner control (same lines, wrong answers);
  - whole-frame recall;
  - implicit;
  - definition.
- **Scoring:** PMI against a null prompt without context.
- **Stated-only items:** questions whose answer the definition states.
- **Primary** (Holm over 9 per track): (a) `explicit-rel:oracle − none`, (b) `− explicit-sym:oracle`,
  (c) `− implicit:oracle`.
- **Decision rule:** if (a) and (c) are positive on T4 property, recommend read → store → selective recall, and draft
  an E13 amendment.

## Experiment
As 14.4 (C5 × 3 seeds, C0′ × 1 seed, T4 and T5).

## Results

| Property accuracy | T4 real chemicals | T5 new words |
|---|---:|---:|
| none | 0.614 | 0.190 |
| whole frame | 0.578 | 0.687 |
| **only the asked relation (oracle frames)** | **0.862** (+0.248\*) | **0.845** (+0.655\*) |
| same lines, wrong partners | 0.528 | 0.171 |
| type-prior frames, asked relation | 0.717 | 0.811 |
| definition text | 0.734 | 0.857 |

- **The decision rule fired.** The scoring artefact explained about two-thirds of 14.4's T4 drop; list dilution
  explained the rest (whole frame −0.036 with the fixed null, p 0.0095).
- **The model reads the content:** wrong partners pull accuracy below `none`.
- **On stated-only T4 items,** recall of the stated facts (0.977) slightly beats the definition (0.958; +0.020,
  p 0.001).
- **We drafted the E13 amendment, and the author approved it** (selective recall endpoint L5b; see file 13).

## Limitations
- Oracle frames are gold. With the type prior's frames the T4 gain falls to +0.10, and the definition still slightly
  beats it.
- The item's relation label chose the lines, which a deployed system cannot know (tested in 14.6).
- One host size.

## Next steps
14.6 removes the label and tests better readers and larger hosts.

---

# 14.6 — Without the label: routing, Claude's frames and larger hosts (RD-4c)

**Status:** done · **When:** 2026-10-20 · **Sources:** `experiments/e14-logic/report/rd4c/`; `manuscript/findings/2026-10-20-e14-evening.md` §1; preregistration §16

## Research gap
Selective recall (14.5) was given the item's relation label, which a deployed system does not have. We asked three
things:
- whether a router can choose the relation without the label;
- whether a better reader (Claude) closes the gap to gold frames;
- whether the effect holds on larger models.

## Intuition
Questions signal their relation ("has a role as", "is a kind of"). The model, or simple word overlap, should pick that
relation from the ones the term's frame holds.

## Methodology
- **Routers** choose among the frame's relations, never using the label:
  - `lex`: word overlap between the relation's phrase and the question;
  - `lm`: the host scores each relation phrase as a continuation of the question;
  - `top2`: the two best relations under `lm`.
- **Readers:** oracle, type prior, Claude (T4 only).
- **Primary** (Holm over 6 per track):
  - (a) the realistic pipeline (`lm` router + type-prior frames) − none;
  - (b) `lm` router + oracle frames − labelled oracle (the cost of routing).
- **Larger hosts:** Qwen3-0.6B and 1.7B C0′, explicit route only. Primary: labelled oracle recall − none, and − whole
  frame.

## Experiment
- SmolLM2-360M C5, 3 seeds, on T4 and T5.
- Qwen3-1.7B C0′ on T4 (1 seed); Qwen3-0.6B and 1.7B C0′ on T5 (2 seeds each).
- 11 GPU jobs.

## Results

| Property accuracy, SmolLM2-360M | T4 real chemicals | T5 new words |
|---|---:|---:|
| none | 0.614 | 0.190 |
| labelled selective recall, oracle frames | 0.862 | 0.845 |
| `lm` router + oracle frames | 0.780 | 0.307 |
| `lex` router + oracle frames | 0.787 | 0.618 |
| **`lm` router + type-prior frames (realistic)** | **0.713** | **0.304** |
| `lex` router + type-prior frames | 0.714 | 0.589 |
| definition text | 0.734 | 0.857 |

- **(a) The realistic pipeline helps on both tracks:** property +0.099\* (T4) and +0.114\* (T5).
- **(b) Routing costs a lot:** −0.083\* on T4 and **−0.538\*** on T5. The prediction (≥ −0.05) failed on both tracks.
- **Router accuracy:**
  - `lm`: 0.585 on T4 with oracle frames, 0.781 with type-prior frames; 0.21 on T5;
  - `lex`: 0.79–0.89 on T4; 0.39 on T5.

  On T5 the model's router almost always picks a generic relation. Word overlap does far better there: +0.399 over
  `none` with type-prior frames.
- **Claude's frames do not beat the type prior's** on T4 (−0.014\*). That prediction failed.
- **Larger hosts keep the effect:**
  - labelled selective recall − none is +0.172\* (Qwen3-1.7B, T4), +0.529\* (Qwen3-0.6B, T5) and +0.615\* (Qwen3-1.7B,
    T5);
  - on T5's short frames, the larger hosts are barely hurt by whole-frame recall (−0.028 to +0.017).

## Limitations
- The routers are simple and untrained.
- T5's question wording rarely matches the relation phrases, which favours `lex` over `lm` for reasons specific to the
  track.
- Claude's frames are sparse (RD-3 recall 0.66).
- The larger hosts have one or two seeds.

## Next steps
**Routing, not reading, is now the bottleneck of the realistic pipeline.** A learned router is approved (tier F,
2026-10-20) and being built.

---

# 14.7 — The read → train → read cycle (RD-12, RD-12b, RD-12c)

**Status:** done (RD-12, RD-12b, RD-12c) · **When:** 2026-10-20 · **Sources:** `experiments/e14-logic/report/{rd12, rd12b, rd12c}/`; `manuscript/findings/2026-10-20-e14-evening.md` §§2–3; preregistration §§12, 17, 18

## Research gap
The author's hypothesis: self-learning alternates passive learning (gradient descent on text) with active reading (the
model reads, reflects and writes facts), and knowledge accumulates over rounds.

## Intuition
Each passive phase should make the model better at using the facts it read. Each reading round should add facts,
including new relation types (14.3).

## Methodology
- **Arms:**
  - `cycle`: RD-11's reading with reflection, then passive training, three rounds;
  - `single`: one reading, reused;
  - `gold`: the gold frames;
  - `passive`: no frames.
- **Passive phase:** 2M tokens per round on the definitions mixed with domain text.
- **Primary:** loss on held-out text after the read terms, `cycle − passive` and `cycle − single`.

## Experiment
T4, SmolLM2-360M C5 seed 1, 160 definitions. A first attempt failed twice; both causes were fixed and logged.

## Results
- **No arm helped,** gold included:
  - `cycle − passive` +0.003 (n.s.);
  - `cycle − single` −0.001 (n.s.);
  - `gold − passive` +0.010 (p 0.04, i.e. *worse*).
- **Every arm drifted worse** (held-out loss 1.783 → 1.815) while training loss fell (2.04 → 1.86). That is the
  signature of overfitting to the narrow passive mix: definitions repeated 40× and a learning-rate restart.
- The cycle still grew its frames: gold recall 0.155 → 0.176, frames per term 0.88 → 1.32. It retracted nothing.

### Reruns (RD-12b, RD-12c)

**RD-12b** (the gentle passive phase, with no-training controls):
- **The recipe gate passes:** the drift of `passive − still-none` is +0.08%, against a +0.5% limit.
- **On loss, nothing changes:**
  - `cycle − passive` +0.0007 (n.s.);
  - `cycle − single` +0.0002 (n.s.);
  - training with gold frames adds nothing over writing them (`gold − still-gold` +0.0036, n.s.).
- **Under selective recall the cycle's extra facts help a little:** `cycle − single` on property is +0.0152\*
  (Holm 0.018).
- Gold frames add +0.23 whether or not the model trained with them.

**RD-12c** (the pilot's frames on the base models, 3 seeds):
- `cycle − single` on property is +0.0056 (Holm 0.080, n.s.).
- It is largest on `has_functional_parent`, the rediscovered relation (0.977 against 0.959).

**Reading:** repeated reading adds a few usable facts, and passive training adds nothing on top of explicit recall.

## Limitations
- The pilot (not the reruns) measured use only through the implicit route, which does nothing on T4 even with gold frames (14.4).
- The passive recipe itself caused drift.
- One seed.

## Next steps
Closed for T4.
- Reading's ceiling is the text (14.1) and routing (14.6), not the number of rounds.
- A T5 version, where the implicit route works, remains possible.
