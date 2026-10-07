# E9 — frequency bias, and understanding beyond copying

**Status:** design and pre-registration. Written on 2026-10-07 and committed **before any evaluation of these measures on
a trained checkpoint** (GPU or CPU). The labelled smoke tests (§9) ran after this commit; they check that the pipeline
works and estimate cost, and change nothing in §§1–8. Any later change is listed in §10 with its date and reason.

**Origin:** author request of 2026-10-07, after R9's "T5 controls" section and its copy-concern table
(`reports/R9-retrofit-quantization.md`):

1. Copying filler words is not necessarily bad: part of it measures **frequency bias**. Measure the frequency-bias
   improvement properly, in the tradition of the group's HRRBERT paper (`resources/vsa-paper.md`: codes binned by log
   frequency, per-bin tests against the unstructured control, t-SNE showing that unstructured embeddings encode frequency).
2. Build test items for **understanding**, whose correct answer is *not* a filler word of the term's frame, so copying
   cannot solve them.

**A first look** (existing metrics; T5, SmolLM2-360M, 3 seeds; relative loss vs C0′) motivated the design:

| Training frequency of the preceding term | C2 (free table) | C5 |
|---|---|---|
| held-out | +2.2% | −13.3% |
| unseen | +2.2% | −6.6% |
| 1–9 | +1.7% | −6.0% |
| 10–99 | +0.8% | −7.2% |
| ≥ 100 | −2.1% | −6.2% |

The free table is frequency-biased; C5 helps every bin. On Qwen3-1.7B, C5 cuts the gap between rare and frequent terms
by 17%. T4 shows small, flat gains; T1 and WordNet are null.

**Code:**
- `src/vsa_embed/experiments/e9_freqbias.py`: fine-bin and strict rescoring (`score`), exclusion tables (`exclusions`), the
  row probe and t-SNE (`rows`), analysis functions, queue;
- `src/vsa_embed/experiments/e9_understanding.py`: item builders (`items`), evaluation (`evaluate`), analysis, queue;
- `src/vsa_embed/experiments/e9_report.py`: opt-in `--freqbias` and `--understanding ITEMS` sections (with `--item-seed`);
- tests: `tests/test_e9_understanding.py` (toy glossary, fake host, CPU).

**Items** (committed with this document; `items.jsonl.gz` is gzip with mtime 0, so the digests are reproducible):

| Item set | anchors (seen / rare / heldout / new) | items | sha256 `concepts.jsonl` | sha256 `items.jsonl.gz` |
|---|---|---:|---|---|
| `items/understanding-t5-smollm2-v1` | 300 / 300 / 560 / 700 | 48,786 | `a35dd11f…57dd` | `f1b998ee…5751` |
| `items/understanding-t5-qwen3-v1` | 300 / 300 / 560 / 700 | 48,786 | `408f6057…35f1` | `f1b998ee…5751` (identical items) |
| `items/understanding-t4-smollm2-v1` | 300 / 300 / 686 / 300 | 17,058 | `afb8a5c8…e4f0` | `af8f22b5…7f92` |
| `items/understanding-t7-smollm2-v1` | 300 / 300 / 908 / 300 | 16,684 | `e7dd2def…5c2d` | `a0117a4b…6adb` |

Full digests are in each `manifest.json`. Exclusion tables (derived, not committed; `~/data/vsa-llm/e9/exclusion-tables/`,
rebuilt identically by `e9_freqbias exclusions`): `t5-smollm2.pt` `7e2b4ea1…2136`, `t5-qwen3.pt` `b38f94c4…8352`,
`t4-smollm2.pt` `b8d1cf89…8b94`, `t1-smollm2.pt` `e0238274…1954`, `wordnet-smollm2.pt` `c4419b1e…96b8`,
`t7-smollm2.pt` `f15c93d8…7e3f`. T1c's table is built by its own job under `~/data/vsa-llm/t1c/exclusion-tables/`
(licensed; never in the repository).

## 1. Questions and families

| Family | Question | Unit | Primary endpoint |
|---|---|---|---|
| **A1** frequency bias of the loss | Does the channel narrow the loss gap between rare and frequent terms? | evaluation window (term-cluster check) | Δgap, C5 − C0′ |
| **A2** frequency decodability of concept rows | Do the per-concept vectors encode training frequency (HRRBERT's t-SNE observation, made quantitative)? | entry | R²(C2 free row) − R²(C5 composed vector) |
| **B1** understanding items | Does the composed row carry knowledge that copying a filler cannot use? | anchor concept × seed | composite, C5 − C0′, held-out terms |
| **B2** strict non-copy loss | Does the loss gain after a term survive when every copyable token is removed? | evaluation window | `after_heldout_strict`, C5 − C0′ |

Each family has one primary endpoint, tested at two-sided α = 0.05. The four answer different questions, so no
correction is made across families (stated in advance). Secondary tests are Holm-adjusted within the family named.

**Primary setting (all four):** T5 synthetic glossary, SmolLM2-360M, seeds 1–3, bf16 (as trained). P0 has no training
randomness and pairs with every seed. Everything else replicates or generalizes (§6).

## 2. A1 — frequency bias of the loss

**Rescoring** (`e9_freqbias score`; output `RUN/freqbias/`). A finished run is evaluated on its own evaluation windows
with the trainer's per-token losses (bf16 autocast, `training.lm.evaluate`'s batches). Every after-span target gets the
training frequency of the term(s) it follows (the trainer's union rule: a target in the 8-token window after two spans is in
both spans' strata).

- **Bins (`halfdecade`, primary):** 1–3, 4–9, 10–31, 32–99, 100–316, 317–999, 1,000–3,162, ≥ 3,163, plus the trainer's
  `after_unseen` (frequency 0, not held out) and `after_heldout`. The bins are half-decades chosen so that their unions are
  exactly the trainer's `after_rare_seen` (1–9), `after_mid` (10–99) and `after_frequent` (≥ 100). This is checked on
  every batch. The `log2` scheme (1, 2–3, 4–7, …, ≥ 1,024) is a sensitivity analysis only.
- **Exact replay** (amended in §10.1): the per-window outputs contain the trainer's strata. `ref_check` must report
  `max |Δ window sum| = 0` against the run's `eval_windows.npz` at its final evaluation. A run that fails this is
  excluded from A1/B2, and the failure is reported.
- **Term table** (`terms.npz`): per evaluation window × entry, the sums over the union of that entry's own after-span
  windows. This is the unit of the term-cluster bootstrap. Its estimand counts a target once per distinct entry it
  follows, so its point estimate can differ slightly from the window estimand.

**Measures per model m** (seeds pooled per window):
- `L_m(b)`: token-weighted loss in bin b.
- **Slope** `S_m`: the OLS slope of `L_m(b)` on `x_b`, over the bins with ≥ 1,000 targets and ≥ 20 distinct entries in
  one seed's windows. Bins are unweighted. `x_b` is the target-weighted mean log2 frequency of the bin's entries.
  Units: nats per doubling of training frequency. Support is model-independent.
- **Gap** `G_m = L_m(after_rare_seen) − L_m(after_frequent)`, in nats.
- **Relative gap** `G_m / L_m(after_frequent)`.
- **Gap cut** `1 − G_C5 / G_C0′`.

**Primary endpoint A1.** `ΔG = G_C5 − G_C0′`, from a paired window bootstrap (10,000 resamples; the same resamples for
both models), with a two-sided percentile p.

**Prediction (written before running).** ΔG < 0 but small (≈ −0.02 to −0.03 nats): the first look shows C5 lowering
rare and frequent losses by nearly the same relative amount (−6.0% vs −6.2%), so the absolute gap shrinks roughly in
proportion to the loss. The relative gap change is ≈ 0. C2's ΔG > 0.

**Decision rule.**

| Reading | Conditions |
|---|---|
| **(a) "The channel reduces frequency bias, frequency-specifically"** | ΔG < 0 (CI excludes 0) **and** Δrelative gap < 0 (CI excludes 0) |
| **(b) "The channel narrows the rare–frequent gap in nats, in proportion to the loss"** | ΔG < 0 (CI excludes 0) and the Δrelative-gap CI includes 0. A uniform relative gain gives rare terms more nats; it is not a frequency-specific correction |
| **(c) "The gain is frequency-neutral"** | the ΔG CI includes 0 |
| **(d) "The channel adds frequency bias"** | ΔG > 0 (CI excludes 0) |
| downgrade | If the term-cluster bootstrap's ΔG CI includes 0 while the window CI does not, the reading is labelled "window-level only" |

**Key secondaries** (95% CIs; Holm within each row's family):

| # | Contrast |
|---|---|
| A1-S1 | C2's ΔG (prediction > 0: the free table adds frequency bias, HRRBERT's unstructured-embedding result), and `G_C5 − G_C2` |
| A1-S2 | Δslope C5 − C0′ (prediction ≥ 0: flatter) and C2 − C0′ (prediction < 0) |
| A1-S3 | Per bin: C5 − C0′ and C2 − C0′ relative loss (the HRRBERT per-bin presentation; the R9 references in place of Dunnett's many-to-one test), Holm over bins within each comparison; window and term-cluster intervals |
| A1-S4 | The arms: C5rf, C5ut, C5tr, C5sh, C6m, C6d, C6g − C0′ ΔG (prediction: C5sh ΔG ≥ 0; the C6 arms like C2) |
| A1-S5 | The gap on filler and on strict targets (`after_f*_filler`, `after_f*_strict`). Prediction: most of the gap reduction is on filler targets, which is the author's "copying measures frequency bias" |
| A1-S6 | `ref-off` (C5 with its channel switched off): is any change in bias a property of the C5-trained host weights? |

## 3. A2 — frequency decodability of concept rows

**Probe** (`e9_freqbias rows`, CPU; output `RUN/freqrows/`).
- **Representations:**
  - `table`, the channel's own per-concept vector before the projector: C5 and the C5 arms give the composed
    vector; C2 gives the free row; C6m/C6d/C6g give the frozen source row;
  - `row`, the injected row after the projector;
  - `host_subtoken_mean`, the host's input embedding averaged over each alias's subtokens (every model);
  - `degree` and `frame_graph` (a 64-rank spectral embedding of the frame graph): model-free references.
- **Features:** each representation's coordinates plus its L2 norm and log norm. A linear probe cannot read a norm, and
  a rarely updated free row differs from a frequent one mostly in norm.
- **Target and entries:** log2 training frequency, over entries that are not held out and have frequency ≥ 1. The `all`
  subset adds the unseen entries, with target log2(1 + f).
- **Fit:** a 5-fold cross-validated ridge (strength by GCV inside each training fold; fixed fold assignment, seed 0). R²
  is computed on the out-of-fold predictions.
- **Intervals:** an entry bootstrap (2,000 resamples). The same entry resamples are used for every run and
  representation, so differences are paired. Seeds are averaged.

**Primary endpoint A2.** `R²(C2 table) − R²(C5 table)`, entries with f ≥ 1.

**Prediction.** > 0, and large. On T5 each term's Zipf rank is drawn at random, independently of its frame, so a composed
vector can carry frequency only through frame features (R²(frame_graph) ≈ 0). A free row is updated in proportion to its
term's frequency.

**Decision rule.**
- > 0 with the CI excluding 0: HRRBERT's observation replicates in E9. Unstructured per-concept rows encode training
  frequency; composed rows do not.
- CI including 0, or < 0: it does not replicate. Both encode frequency alike, or neither does. The t-SNE figure is then
  descriptive only.

**Secondaries.**
- `R²(C5 table) − R²(frame_graph)`: composition adds no frequency information beyond the frame. Prediction ≤ 0.05.
- The C6m/C6g source rows: frozen and frequency-free by construction for C6g; C6m inherits the host's subtoken
  frequencies.
- `host_subtoken_mean` of P0 vs C0′ vs C5: does continued training write frequency into the host's embedding rows?
- The `all` subset.
- T4, T1, WordNet and Qwen3. On natural tracks frames correlate with frequency, so `frame_graph` is the reference there.

**Figure** (descriptive): exact t-SNE (perplexity 30, 750 iterations) of 1,500 entries sampled equally per integer log2
frequency, coloured by log2 frequency. One panel per representation; C2 and C5 side by side, as in HRRBERT. Licensed
tracks (T1c) get no figure.

## 4. B1 — understanding items (the answer is not a filler of the term's frame)

**Construction** (`e9_understanding items`; one directory per track and tokenizer family).

**Subsets.**

| Subset | Definition | T5 count |
|---|---|---|
| `seen` | training frequency ≥ 10; seeded sample | 300 |
| `rare` | training frequency 1–9; seeded sample | 300 |
| `heldout` | held-out and generator zero-shot terms; rows composed zero-shot | 560 (360 + 200) |
| `new` | the E9 new words (`new-words-t5-<family>-v2`, decision 56; v1 on T4 and T7); frames inserted at evaluation | 700 |

Anchors are single-concept entries whose shown surface links to them (≥ ℓ_min subtokens). On T5 they have a
non-structural type (not team, department or committee). The shown surface is the concept's own name (generator zero-shot
terms without the `synthetic:` prefix; on T4 the ChEBI name), else the canonical alias.

**Lexical overlap rule** (all families except two-hop's own answer rule): an item is dropped if any option shares a
content-word stem with any verbalization of the anchor within two hops. Content words have ≥ 3 letters and are not
stopwords. The verbalizations are the anchor's aliases, its frame's fillers, and the fillers of the frames of its concept
fillers (the exclusion table).

**Families** (T5 wording; T4 and T7 in `e9_understanding.T4_SPEC` / `T7_SPEC`):

| Family | Item | Options (chance) | Why copying cannot solve it |
|---|---|---|---|
| `two_hop` | "X is owned by a team that reports to" → the owner's department. Paths: owned_by→reports_to, governed_by→approved_by, governed_by→owned_by, part_of→sponsored_by, measured_by→owned_by, depends_on→owned_by, uses→owned_by. T4: functional parent / conjugate / class → role | answer + 4 fillers of the second relation (0.2) | the answer is not in X's frame (as an atom or as a text); no option is a filler of X's frame |
| `affordance` | "Most people interact with X by" → " logging in to it" (system) / " querying its rows" (dataset) / …; "Last week, X" → " crashed and had to be restarted" / … T4: the electrode an ion migrates to, from its charge. T7: what is done with a chemical / disease / organism | one phrase per type (T5: 0.125) | a consequence of the type, worded with no frame word |
| `paraphrase` | "X is mainly concerned with" → " money, budgets and accounting" (area finance); status, cadence, tier and type by definitions. T4: element symbol → element name, charge → electron balance. T7: MeSH tree category by name | gold + up to 4 other values (0.2–0.33) | no option shares a word with any verbalization |
| `reverse` | "Of X and Y, the one owned by the Zash Team is" → " X" (both orders) | the two names (0.5) | the filler is the cue; the answer is a term. PMI against "the one we mean" |
| `comparison` | "X and Y are owned by" → " the same team" / " two different teams"; area; type (T4: chemical class, net charge; T7: heading, record class) | (0.5), balanced same/different | the answer compares two frames |
| `negation` | "X is not owned by" → the false filler; scored with its affirmative twin | the true and a false filler (pair consistency, 0.25) | copying the filler fails the negated item |

**References** (not in the composite): `hop1` (X's own first-hop filler, a copy item) and `bridge` (the second-hop
question asked of the filler itself: does the host know the second hop?). Both are scored under `own` only.

**Scoring.**
- **Method:** as E5.4 — PMI of each option against a null prompt (the term replaced by "this"/"that"; for `reverse` the cue
  replaced by "we mean"), argmax per template. An item's score is the mean over its two templates.
- **Unit:** an anchor concept.
- **Family score per anchor:** the mean of (correct − chance) over its items. For `negation`, pair consistency − 0.25.
- **Composite per anchor:** the mean over the six families present for it.

**Sources.**
- C5: `own`, `none` (the anchors' rows zeroed) and `random_frame` (the anchors' frames replaced by random frames of equal
  degree; new words: their own random frames).
- Every other model: `own` (C0′ and P0 have no channel; C2, C6* and the C5 arms read their own rows).

**Primary endpoint B1.**
- **Contrast:** composite, `heldout`, C5 − C0′.
- **Model:** the concepts × seeds crossed random-effects model (`statistics.crossed_components`, Satterthwaite t, two-sided).
  The two-way cluster bootstrap is reported as the check. The seed-averaged anchor bootstrap is also reported.

**Predictions** (written before running; T5, 360M, held-out).
- Composite: C5 − C0′ > 0.
- `paraphrase`, `reverse`, `comparison`: clearly positive. The composed row states the facts these items ask about.
- `two_hop`: small. It needs the host to know the second hop, and `bridge` accuracy bounds it.
- `affordance`: positive on new words, whose names carry no type. Smaller on held-out terms, whose head nouns ("Ledger",
  "Index") give every model the type.
- `negation` consistency: near chance for every model (small LMs ignore "not").

**Decision rule.**

| Reading | Conditions |
|---|---|
| **(a) "The composed row carries usable knowledge beyond copying a filler"** | primary > 0 with p < 0.05, **and** the C5 `own − random_frame` composite > 0 on held-out terms (CI excludes 0) |
| **(b) "…and it is the frame's content"** | (a) **and** C5 − C5sh > 0 on the held-out composite (CI excludes 0) |
| **(c) "A generic channel effect"** | primary > 0, but `own − random_frame` has a CI including 0 |
| **(d) "No evidence of understanding beyond copying"** | the primary CI includes 0. The dimension-3 gains are then filler selection |

**Refutation readings.**

| # | Observation | Reading |
|---|---|---|
| U1 | `two_hop` ≈ 0 while `bridge` is well above chance | the host knows the second hop but does not chain it through the composed row |
| U2 | `two_hop` ≈ 0 and `bridge` ≈ chance | two-hop is untestable here (the host lacks the second hop), not refuted |
| U3 | `paraphrase` ≈ 0 while R9's filler `property` gain is large | the gain is lexical, i.e. copying |
| U4 | `reverse` ≈ 0 while forward `property` gains | a reversal-curse analogue through the channel |
| U5 | C2 ≈ C5 on held-out terms | impossible by construction (C2 gives held-out terms the mean row); if observed, a scoring bug |

**Secondaries.**
- Per family, C5 − C0′, Holm over the six families within each subset.
- C5 − C2, C5 − P0 and C5 − each arm (C5sh first).
- The `seen`, `rare` and `new` subsets.
- `affirm` and `negate` separately.
- `two_hop` among items whose `bridge` is correct (descriptive).
- Replications: SmolLM2-135M, Qwen3-0.6B/1.7B, T4 (natural text; the T4 dimension-3 null predicts small effects) and T7.

## 5. B2 — the strict non-copy loss stratum

**Definition.** `X_strict` keeps the targets of after-span stratum `X` that belong to no occurrence, starting after the
span, of a token sequence verbalizing any of:
- the term itself (its aliases);
- any filler of its frame;
- any filler of the frame of a concept that fills its frame (two hops);
- relation wording.

**What counts as a verbalization.**
- A concept: its aliases and lexicon text, and each content word of them.
- Relation wording: the content words of the relation names, of the track lexicon's relation templates and, on T5, of
  the generator's fact templates.
- Matching: as written, lower-cased and capitalized, with and without a leading space.

The strict set is a subset of `X_nonfiller`, and its share of the stratum is reported. A target after two spans is
excluded if it is a copy candidate for either.

**Primary endpoint B2.** The relative loss difference C5 − C0′ on `after_heldout_strict`, from a paired window bootstrap
(10,000 resamples).

**Prediction.** < 0, but smaller than R9's −19.4% on held-out non-filler targets. Relation words ("owned", "depends") are
removed, and they are predictable from the composed row's edges.

**Decision rule.**
- < 0 with the CI excluding 0: the gain survives without any copyable token (loss-level understanding: e.g. the
  type-specific document structure).
- CI including 0: R9's non-filler held-out gain was carried by two-hop names and relation wording.

**Secondaries** (Holm over the strict tests): `after_rare_seen_strict`, `after_unseen_strict`, `after_len3plus_strict` and
`after_strict` against C0′, C2, P0 and every arm (C5sh first).

## 6. Runs and order

Evaluation only, on finished `final.pt` checkpoints (main checkout, `experiments/e9-retrofit/runs/…`). Two kinds of job:
- GPU: the frequency rescoring, plus the understanding items;
- CPU: the row probe.

| Block | Runs | A1/B2 rescoring | A2 probe | B1 items |
|---|---|---|---|---|
| primary | T5 SmolLM2-360M: P0, C0′, C2, C5, C5rf/ut/tr/sh, C6m/d/g × s1–3 | yes | yes | yes (C5 three sources; others `own`) |
| size | T5 SmolLM2-135M: P0, C0′, C2, C5 × s1–3 | yes | yes | yes |
| hosts | T5 Qwen3-0.6B/1.7B: P0, C0′, C2, C5 × s1–2 | yes | yes | yes (`understanding-t5-qwen3-v1`) |
| natural | T4, T1, WordNet: SmolLM2-360M/135M seed 1 | yes | yes | T4 only (`understanding-t4-smollm2-v1`) |
| later | T4 s2–3 and arms, T7 (once trained) | yes | yes | T4, T7 |
| licensed | T1c (once trained) | yes, aggregates only | yes, no figure | — (no understanding items: the item files would hold SNOMED names) |

## 7. Exclusions

- A run whose `ref_check` fails is excluded from A1 and B2.
- Anchors whose surface does not link to their entry are excluded. Counts are reported per run.
- Bins below the support rule enter the tables but not the slope.

## 8. What this does not test

- Generation: every item is forced choice or likelihood.
- Natural-text understanding at scale: T4 and T7 are the natural tracks here, and T4's dimension-3 null (R9) predicts
  small effects.
- The clinical track's items: T1c has no understanding items, for licence reasons.

## 9. Smoke tests

Smoke tests run after the commit of this document and are labelled SMOKE. They use short subsets on the CPU, plus at most
4 GB and 5 min of GPU. They report timing and pipeline checks only, never a result.

### §9 addendum — smoke tests (2026-10-07; SMOKE, not a result)

The GPU was shared with a running queue job (100% utilization, 20.5 GB in use), so GPU timings are upper bounds. Outputs
went to a scratch directory, not to run folders.

- **A1 / B2 rescoring** (`score --windows N`), T5 SmolLM2-135M seed 1.
  - CPU, C5, 8 windows, `ref` and `ref-off`: 19 s. The masks replay the trainer's strata exactly (`counts_equal`). The
    sums differ by 0.3%, because the CPU runs fp32 against the trainer's GPU bf16.
  - GPU, 32 windows:
    - C0′ replays bit-exactly (max |Δ window sum| 0.0).
    - C5's large strata replay to 9 × 10⁻⁵ relative, closer than `e9_rescore`'s committed `ref` on the same windows
      (10⁻⁴); see §10.2.
  - Masks cost ≈ 0.01 s per window on T5 and T4: ≈ 15 s per 1,024-window run.
  - Every bin nests the coarse strata on every batch.
- **A2 row probe**, T5 SmolLM2-135M seed 1, CPU.
  - C5 took 3 min 43 s and 2.2 GB of RAM; C2 took about the same.
  - The pipeline produces R² for every representation, `oof.npz`, `tsne.npz` and `tsne.png` (the figure renders, with
    one panel per representation).
- **B1 items**, T5 SmolLM2-135M C5 seed 1, three sources.
  - CPU, 10 anchors per subset: 8 min 19 s. Every concept links (381 of 381) and every family and reference test
    populates.
  - GPU, 20 anchors per subset (695 concepts, all linked), with the final scorer (§10.3): 152 s, about 255 sequences/s on
    the shared GPU.
- **Cost basis for the plan.** The committed dimension-3 evaluations ran on the queue GPU without contention:
  - throughput: SmolLM2-360M ≈ 1,200 sequences/s, 135M ≈ 2,100, Qwen3-0.6B ≈ 450, Qwen3-1.7B ≈ 180, with the slower shared
    scorer;
  - the plan assumes ×1.5 for the continuation-only scorer (×2.5 on Qwen3, whose 152k-row head dominated);
  - item volume: T5 has 333k sequences under `own` and 258k under each other source (the reference tests are `own`
    only), plus ≈ 24k null prompts; T4 has 87k (+ 25k null).

## 10. Deviations and changes after commit

### 10.1 The replay rule (2026-10-07, before any evaluation of these measures on a checkpoint)

**Problem.** §2 and §7 required `max |Δ window sum| = 0` against the trainer's final evaluation. The committed PQ1
rescorings (`RUN/rescore`, 41 runs; read from their files, no new evaluation) show that this holds only for runs without a
composer:
- P0, C0′, C2 and C6* replay bit-exactly on the GPU.
- C5 and the C5 arms differ by at most **2.1 × 10⁻⁴ relative per stratum total**. Their masks are identical
  (`counts_equal`) in every run.

The composer's CUDA `index_add_` sums in no fixed order under bf16, which is documented in `e9_ontology_edit`. Under the
original rule every composing run would have been excluded.

**New rule** (`e9_freqbias.prefix_reference_check`, `replay_ok`):
- The masks must be identical (`counts_equal`), for every run.
- The sums must be bit-exact for runs without a composer.
- For composing runs, the sums must agree within **10⁻³ relative per stratum total**. That is 5× the largest observed
  difference, and far below the percent-level effects measured.

`e9_report --freqbias` excludes runs that fail and lists them (`excluded_replay`).

**Consequence.** Rescoring jobs keep the trainer's evaluation batch, so the queue never passes `--eval-batch`: a different
batch changes bf16 kernel shapes. Smoke runs on the CPU (fp32) check the masks only.

### 10.2 The replay tolerance applies to large strata (2026-10-07, after the first GPU smoke; no endpoint computed)

**What the smoke showed.** The labelled GPU smoke (§9: SmolLM2-135M C5 s1, the first 32 windows) reproduced the trainer's
large strata to 9 × 10⁻⁵ relative (`all`, `after`, `inside`, `unlinked`, `after_frequent`, `after_len3plus`). On the
same 32 windows, `e9_rescore`'s committed `ref` reproduces them to 1–3 × 10⁻⁴. The masks were identical, but a stratum with
a few hundred targets differed by 0.75%. In a small stratum, one window's composer noise is a large fraction of the total.

**New rule.** For a composing run, the 10⁻³ relative tolerance of §10.1 is checked on the strata with ≥ 5,000 targets (the
largest stratum when none has as many). Those strata identify the model; the masks are the replay of the strata and stay
exact for every stratum and every run. Runs without a composer must still replay bit-exactly. The C0′ GPU smoke did:
max |Δ window sum| = 0.0.

`ref_check` records `max_relative_stratum_diff` (the checked strata), `max_relative_stratum_diff_any` (every stratum) and
the list of checked strata.

### 10.3 The B1 continuation scorer (2026-10-07, after the B1 smoke; no endpoint computed)

**Change.** B1 scores options with `e9_understanding.continuation_logprob`:
- the same batch forward, spans and autocast as the shared adapter;
- output logits only at the positions that predict a continuation token, with texts batched by length (E11's
  `continuation_scores`, §13.2 of the E11 pre-registration);
- the output head applied in **float32**.

**Why.** It is 5.5× faster on the CPU. Under bf16 autocast, the head's rounding depends on the matrix shape: in the smoke,
the shared and the continuation-only scorers flipped a few near-tie items on the GPU. Applying the head in float32 removes
that rounding source.

**What does not change.** On the CPU (no autocast), it equals `channel_probes.continuation_logprob` to 6 × 10⁻⁵ nats (a
smoke check on 512 real prompts) and to 10⁻⁴ in the toy test. The PMI, the argmax, the families and the endpoints are
unchanged.

### 10.4 Job settings (2026-10-07; no endpoint computed)

- **Rescoring variants.** `ref` for every run, plus `ref-off` for C5 (A1-S6). The jobs keep the trainer's evaluation batch
  (§10.1).
- **Row probe figure.** Seed 1 only (the figure is descriptive). Every seed gets its probe.
- **B1 sources.** On Qwen3, C5 gets `own` and `none` (no `random_frame`) to halve its cost. The `own − random_frame`
  specificity condition of reading (a) is tested on SmolLM2, where all three sources run.
