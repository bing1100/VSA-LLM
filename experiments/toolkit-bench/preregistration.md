# Toolkit benchmarks, part 1: *write* benchmarks on existing E9 checkpoints (COMPS-WUGS, ALCUNA)

**Status:** design and pre-registration, written on 2026-10-08 and committed **before any full run**. The labelled CPU
smoke test (§10) runs after this commit; it checks the pipeline and measures cost, and changes nothing in §§1–9. Any
later change is listed in §11 with its date and reason.

**Disclosure.** While the harness was being built, one development smoke ran on the first 40 COMPS-WUGS pairs with
SmolLM2-135M C5 (CPU, not kept). Its numbers (all CIs include 0 except the two in-context routes) changed no endpoint,
contrast or reading below.

**Origin:** author decision 63 (`resources/plan-improvement/execution.md`, 2026-10-08), work package TK-B1;
`manuscript/toolkit-methodology-2026-10.md` §3.3 (*write*: Entity Inferences, COMPS-WUGS, ALCUNA) and §6 step 2
(*write* benchmarks on existing checkpoints).

**Code:** `vsa_embed.benchmarks.ranking` (harness), `vsa_embed.benchmarks.adapters` (item sets),
`vsa_embed.benchmarks.sources` (raw files), `vsa_embed.benchmarks.write_bench` (jobs, contrasts, smoke); tests
`tests/test_rank_benchmark.py`, `tests/test_benchmark_adapters.py`. Item format: `src/vsa_embed/benchmarks/README.md`.

## 1. Question

Does writing a never-seen word into the store as a frame (no gradient step, no text in context) let the model use the
word on a **public** benchmark of learning new words, and how does that compare with giving the same information as
text in the prompt?

| # | Sub-question | Measured by |
|---|---|---|
| Q1 | **Write with a gold frame.** Does an is-a frame ("a wug is a kind of mussel") make the model attribute the parent's properties to the nonce word? | COMPS-WUGS minimal-pair accuracy, `store:oracle − none` |
| Q2 | **Write by reading.** Does the frame the model reads from the one-sentence definition do the same? | `store:linker − none`; `store:typeprior` |
| Q3 | **Content, not presence.** Is the effect the frame's content or the presence of a row? | `store:oracle − store:random` |
| Q4 | **Route.** Store vs the same information in context (verbalized frame; the definition) and both together | in-context conditions |
| Q5 | **Second set.** The same on artificial taxa (ALCUNA ranking forms) | ALCUNA `store:oracle − none` |

**Expectation (stated before any full run).** These checkpoints are the E9 **WordNet track**, where the channel was null:
SmolLM2 already models general words (inside-span loss 0.92), C5 matched C0′, and the E9 new-word property test on
360M C5 gave `own − none` +0.0005 [−0.0035, +0.0045] (`runs/wordnet/SmolLM2-360M-full-C5-s1/edit/report.md`). We
therefore expect W1 and W2 near 0 and the in-context routes far above `none`. A null here is a negative-control-track
result; it does not test the store on new vocabulary (T5, T7), which the T8 Wikidata track (TK-B2) and E13 address.

## 2. Mechanism and conditions

Per item and option, the harness scores `log p(option | prefix)` on a trained run (evaluation only; nothing trains):

| Condition | What the model gets |
|---|---|
| `none` | The run as trained. The nonce word's span is dropped (no row): exactly E9's `none` source (tested equal) |
| `channel-off` | No span injected at all (known words unlinked too) |
| `store:<reader>` | The nonce word's row is composed by the channel from the reader's frame (`SpanChannel.add_entries`; every other row unchanged, tested exactly on CPU). Known words in the context are linked by the run's linker |
| `frame-in-context:oracle` | The gold frame verbalized ("wug is a kind of bivalve. wug is a kind of animal.") and prepended; no row |
| `definition-in-context` | The set's definition prepended (COMPS: "A wug is a mussel." — this reproduces the original COMPS-WUGS prompt exactly); no row |
| `store:oracle+definition-in-context` | Both |

**Readers** (frame of the nonce word): `oracle` (§3), `oracle:parent` (the parent's own trained entry frame: the
polysemous union of the senses its name links to), `random` (equal degree: same relations, fillers drawn
frequency-weighted from each relation's fillers; seeded per term), `typeprior` (concept finder + the atom's most
frequent relation; no model), `linker` (E11's pre-registered reader: the run scores each single-edge candidate frame
by the definition's likelihood after the headword and keeps the best relation per mention if it beats no row).

**Per-model conditions** (`write_bench.CONDITIONS`): C5 all of the above; C2 `none`, `channel-off`, `store:oracle`,
`store:random`, in-context routes (its free table gives every new entry the same fallback row, a frame-independent
control); C0′ and P0 `none` and the in-context routes (no channel).

## 3. Item sets (built before any run; `adapters build`, seed 0)

**COMPS-WUGS** `items/comps-wugs-wordnet-v1` (committed; Apache-2.0, COMPS commit `5a9dc6c4`):
- 13,896 pairs; 4 nonce words (wug, blicket, dax, fep; each ≥ 2 SmolLM2 subtokens, so the linker links them at
  ℓ_min = 2); 4 negative types × 3,474 (taxonomic, overlap, co-occurrence, random).
- Item = one pair: context `Therefore, a wug`, both options ` <property>.`, option 0 under "a wug is <acceptable>",
  option 1 under "a wug is <unacceptable>"; answer 0. Joiner " " (definition-in-context = the COMPS prompt).
- **Frame** of the nonce = `hypernym` → the parent's WordNet synset (COMPS's own sense keys) if it is one of the
  track's 8,142 synset atomics, else its nearest is-a ancestor that is (breadth-first over hypernyms and instance
  hypernyms; ties: most frequent filler) + the parent's `lexname` and `pos` edges.
- **Coverage:** 521 / 521 parent concepts map; 224 exactly (43%), 243 at depth 1, 43 at 2, 14 at 3–4. 2,940 pairs have
  both parents exact; **620 pairs (4.5%) get identical frames on both sides** (the store cannot separate them; they
  score 0.5 under every store condition and are kept).

**ALCUNA** `items/alcuna-wordnet-v1` (committed; MIT, Google Drive files of the ALCUNA repository, sha256 in the
manifest):
- Ranking forms only: multiple choice (the four listed options, scored as answer text) and boolean (Yes / No; the 8,962
  "I don't know" items and fill-in-the-blank are left out). Context `Question: <stem>\nAnswer:`.
- Frame = `hypernym` → the WordNet taxon of the entity's parent (full name, else the genus of a binomial, else a
  one-word higher taxon; animal and plant senses), nearest atomic as above, + `lexname` / `pos`. **Coverage:**
  1,429 of 3,554 entities (40%) map (621 full name, 808 genus); unmapped entities are left out.
- Sample: 1,000 items per (form, question type), seeded: `alcuna-mc` 3,000 (Knowledge Understanding, Differentiation,
  Association) and `alcuna-bool` 2,000 (Understanding, Differentiation); the entity name occurs in every question.
- Definition = the artificial entity's own properties verbalized (≤ 8 values per property).

## 4. Runs

E9 WordNet-track checkpoints, seed 1 (the track has no seeds 2–3), evaluation only:
`experiments/e9-retrofit/runs/wordnet/SmolLM2-{360M,135M}-{full-C5,full-C2,full-C0p,frozen-P0}-s1`. One job per run ×
set (16 jobs) + one pooled report. Outputs: `RUN/toolkit-bench-<set>/`; report `experiments/toolkit-bench/report/`.

## 5. Metric

Accuracy by the **summed log-probability** of the option (`sum`); a tie containing the answer scores 1/(tied options)
(identical requests are scored once, so ties are exact and `none` is exactly 0.5 on COMPS). Per-byte and per-token
normalizations are reported alongside (identical to `sum` on COMPS, whose two continuations are the same text).

## 6. Endpoints and statistics

**Primary** (Holm over both, α = 0.05, two-sided):
- **W1:** COMPS-WUGS, SmolLM2-360M C5, accuracy `store:oracle − none`.
- **W2:** COMPS-WUGS, SmolLM2-360M C5, accuracy `store:linker − none`.

Statistics: paired over items (one seed), percentile bootstrap, 2,000 resamples, seed 0; p = two-sided bootstrap p
(`e5_zeroshot.paired_difference`). **Power:** 13,896 pairs; the per-item difference is in {−0.5, 0, +0.5} (SD ≤ 0.5),
so the minimal detectable effect is ≈ 0.012 (80%, two-sided).

**Secondary** (CI and unadjusted p, labelled; `write_bench.contrast_spec`, also written to `contrasts.json`), all
360M C5 on COMPS-WUGS unless stated:
S1 ALCUNA `store:oracle − none` (a: multiple choice; b: Yes/No) · S2 `store:oracle − store:random` (content control) ·
S3 `store:random − none` · S4 `channel-off − none` · S5 W1 on the 2,940 pairs whose parents are both exact atomics ·
S6 `store:oracle:parent − none` · S7 `store:typeprior − none` · S8 `store:linker − store:typeprior` ·
S9 `definition-in-context − none` · S10 `frame-in-context:oracle − store:oracle` · S11 `store:oracle+definition −
definition` · S12 C5 `store:oracle` − C2 `store:oracle` (composition vs a free table) · S13 C5 − C0′ under
`definition-in-context` · S14–S15 W1, W2 at 135M · S16 W1 by negative type. Also reported, not tested: every
condition's accuracy for every run (P0 `definition-in-context` is the host's published-style COMPS-WUGS score), the
readers' frame precision / recall against the oracle frame and their cost, context tokens per condition.

## 7. Decision rule (fixed now)

| Reading | Conditions |
|---|---|
| **(a) "Writing an is-a frame transfers the parent's properties to a new word"** (WordNet track) | W1 > 0 with Holm p < 0.05 **and** S2 > 0 (CI excludes 0) |
| **(b) "…also when the model reads the frame itself"** | (a) and W2 > 0 with Holm p < 0.05 |
| **(c) "The store matches text in context"** | S10 CI includes 0 or is < 0. Otherwise (S10 > 0): in-context reading is better; the store's only advantage here is zero context tokens per use |
| **(d) "The store adds to the definition"** | S11 > 0 (CI excludes 0) |

**Refutation readings** (pre-written):

| # | Observation | Reading |
|---|---|---|
| R1 | W1 CI includes 0 | Writing a frame does not transfer properties on this track; consistent with the WordNet-track null of R9 (the host already models these words, the channel is not used). Not evidence about new vocabulary |
| R2 | W1 > 0 but S2 CI includes 0 | A row signals "a known noun is here", not its content (as E9.C1's refutation) |
| R3 | W1 > 0, W2 ≈ 0 | The reader is the bottleneck: one sentence names the parent, and the single-edge linker chooses relations poorly (E11 §13.3); report its frame precision |
| R4 | S5 > W1 clearly | The ancestor fallback dilutes the frame; exact-atomic parents carry the effect |
| R5 | S4 ≠ 0 | Linking known words in the context matters by itself (channel on vs off), independent of the new word |

## 8. Data and licences

Raw files under `~/data/vsa-llm/benchmarks/<source>/raw/` (pinned commits / dataset revisions, sha256 twice;
`DATA_SOURCES.md` §TK-B). COMPS (Apache-2.0) and ALCUNA (MIT) derived items are committed with attribution in their
manifests. Entity Inferences (no licence stated), the reversal set, LRE, BEAR (CC BY-SA 4.0) and PopQA are built as
items locally; only their manifests (digests) are committed; they are inputs to TK-B2 (Wikidata frames), not part of
this pre-registration.

## 9. What would change the plan

- If a job fails on memory, its batch size is halved; nothing else changes.
- Seeds: the WordNet track has seed 1 only. If WordNet seeds 2–3 are ever trained, they are a replication, not a change
  of W1/W2.

## 10. Smoke test and commands

Added below after this document was committed (§10 addendum).

### §10 addendum — CPU smoke test (2026-10-08; SMOKE, not a result)

**Setup.** `write_bench smoke`: SmolLM2-135M seed 1, every model (C5, C2, C0′, P0), 40 items per set spread evenly over
the file (ALCUNA: 24 multiple choice, 16 Yes/No), the planned conditions, CPU, 4 threads (the GPU belongs to the
queue). Outputs: `experiments/toolkit-bench/smoke/<set>-SmolLM2-135M-<model>-s1/`, costs in `smoke/cost.json`.
Peak resident memory ≈ 5 GB.

**Checks passed.**
- Every bound new term linked in every scored text (`linked` 1.00); `channel-off` 0.00.
- `none` is exactly 0.5 on every COMPS pair (exact ties); C2's store conditions equal its `none` (frame-independent
  fallback row), as expected.
- No COMPS text was truncated. ALCUNA: 8 long definition-in-context texts exceeded 1,024 tokens and were cut on the
  left (question and options kept); the GPU jobs use `--max-length 2048`.
- Unit tests (`tests/test_rank_benchmark.py`): store rows equal E9's `own`, `none` equals E9's zero row (1e-5), the
  scorer equals the shared continuation scorers, every other row and the entry count are restored exactly.

**Smoke numbers** (40 items, 135M, one seed: not evidence for anything; accuracy by summed log-probability):

| Set | Condition (C5 unless stated) | Accuracy | − none [95% CI] |
|---|---|---|---|
| COMPS-WUGS | none | 0.500 | — |
| COMPS-WUGS | store:oracle | 0.525 | +0.025 [−0.126, +0.176] |
| COMPS-WUGS | store:linker | 0.412 | −0.087 [−0.213, +0.038] |
| COMPS-WUGS | store:typeprior / store:random / store:oracle:parent | 0.537 / 0.550 / 0.500 | CIs include 0 |
| COMPS-WUGS | frame-in-context:oracle | 0.700 | +0.200 [+0.025, +0.325] |
| COMPS-WUGS | definition-in-context (= the COMPS prompt); P0: 0.725 | 0.675 | +0.175 [+0.050, +0.325] |
| ALCUNA (all) | store:oracle | 0.300 | +0.000 |
| ALCUNA (all) | definition-in-context | 0.550 | +0.250 [+0.049, +0.450] |

Readers on the COMPS definitions ("A wug is a mussel."): `typeprior` edge precision 0.85 (recall 0.15: it reads only
the hypernym), `linker` 0.05 — the single-edge linker rarely picks `hypernym` (as E11 §13.3). On ALCUNA's property
lists `typeprior` writes many wrong edges (precision 0.001).

**Cost and commands.** `experiments/toolkit-bench/queue-commands.sh` (from `write_bench plan`): 16 GPU jobs at priority
54.498 (gpu lane) and `tk-bench-write-report` at 54.4981 (cpu lane), added through the queue's Python API
(`write_bench queue`; the queue CLI takes integer priorities only). Estimated idle-GPU hours (`write_bench.job_hours`:
the smoke's padded tokens and scored texts per item, 39k / 80k tokens/s for 360M / 135M, 50 / 35 ms per batch, CPU-side
seconds, × 1.5):

| Host | C5 | C2 | C0′ | P0 |
|---|---|---|---|---|
| SmolLM2-360M (COMPS / ALCUNA) | 0.36 / 0.41 | 0.15 / 0.27 | 0.08 / 0.17 | 0.08 / 0.17 |
| SmolLM2-135M (COMPS / ALCUNA) | 0.31 / 0.29 | 0.14 / 0.21 | 0.07 / 0.11 | 0.07 / 0.12 |
| **Total** | | | | **≈ 3.0 GPU-h** |

## 11. Deviations and changes after commit

All made on 2026-10-08, after the commit of this document and **before any full run**. None changes W1, W2, the
secondaries' definitions or the readings.

### 11.1 ALCUNA runs without `store:linker` and `store:oracle:parent`

- `store:linker` on ALCUNA would score every single-edge candidate (≈ 10² per entity: many found atoms × 14 synset
  relations) over a ≈ 10²-token property list per entity. No pre-registered contrast uses it on ALCUNA (S1 is
  `store:oracle − none`), so it is not run there (`write_bench.SET_DROP`).
- ALCUNA items carry no parent-entry frame, so `store:oracle:parent` would equal `none`; it is not run there.
- COMPS-WUGS keeps every condition of §2.

### 11.2 Implementation details settled by the smoke test (numerically equal)

- The output head is applied to scored positions in chunks of 2,048 (`ranking.HEAD_CHUNK`), bounding memory when the
  linker scores whole definitions. Same numbers.
- GPU jobs batch up to 256 texts under a 32,768-token budget (short COMPS texts are latency-bound).
- GPU jobs read up to 2,048 tokens (`--max-length 2048`, was 1,024 in the smoke): a few ALCUNA definitions in context
  exceed 1,024 tokens (8 smoke texts were cut on the left). Longer texts are still cut on the left and counted
  (`truncated` in `summary.json`).

### 11.3 Secondary S17: the joint linker reader (added 2026-10-08, before any full run)

- In the 40-item development smoke (SmolLM2-135M, not evidence), the `linker` reader's edge precision on COMPS was 0.05,
  against 0.85 for `typeprior`. The same weakness is in E11 §13.3.
- The C5 COMPS jobs therefore also run `store:linker-joint`, and **S17 = `store:linker-joint − none`** (360M) is added as a
  secondary.
- W2 stays `store:linker − none` as committed. No reading changes.
- ALCUNA does not run `linker-joint`, for the same reason as §11.1.

