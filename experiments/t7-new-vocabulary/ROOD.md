# T7-ROOD and the E13 round split (decision 63, WP TK-H1)

**What this is.** T7 v1 holds 908 entries out of the linker only. Their channel rows are never trained, but their names
still occur, unlinked, in the training text, so the host can learn the word from text. HRRBERT kept its held-out codes
out of every training document. This folder adds that "really out of distribution" holdout (methodology M1, holdout H1):

- **T7-ROOD** (`t7rood`): T7 v1 with every training document that mentions a held-out record dropped. Everything else
  is held fixed. It gets an E9 block (`preregistration-rood.md`, `queue-commands-rood.sh`).
- **T7 rounds** (for E13): T7's records split by entry date into round 1 and round 2. The round-2 records' documents are
  dropped from round 1 and form the round-2 training text.

Code: `src/vsa_embed/experiments/t7_rood.py`, enabled by the opt-in `holdout:` section of a `t1_open_corpus` config
(three small hooks in `t1_open_corpus.run`). Tests: `tests/test_t7_rood.py`. Without the section every build is
unchanged: T7 v1 rebuilt with this code into a scratch root is byte-identical (§6).

## 1. Builds

| Config | Run folder (committed) | Data root (`~/data/vsa-llm/tracks/t7-newvocab/`) | Tokenizers |
|---|---|---|---|
| `t7-rood.yaml` | `runs/rood-v1` | `rood-v1/` | SmolLM2 |
| `t7-rood-qwen3.yaml` | `runs/rood-v1-qwen3` | `rood-v1/hosts/qwen3/` (`e9_tracks.QWEN3_ROOTS["t7rood"]`) | + Qwen3 |
| `t7-qwen3.yaml` | `runs/v1-qwen3` | `v1/hosts/qwen3/` (`QWEN3_ROOTS["t7"]`, declared before but not built) | + Qwen3 |
| `t7-rounds.yaml` | `runs/rounds-v1` | `rounds-v1/` and `rounds-v1/hosts/qwen3/` | SmolLM2, Qwen3 |

Each is built by `PYTHONPATH=src python -m vsa_embed.experiments.t1_open_corpus --config <config> --output <run folder>`
(CPU, 6 workers, on 2026-10-08: T7-ROOD 17 min, its Qwen3 relink 15 min, T7's Qwen3 relink 5 min, the rounds with both
tokenizers 48 min; logs in `~/data/vsa-llm/logs/t7-rood/`). The Qwen3 relinks reuse the
SmolLM2 corpora; a relink now rewrites `ontology.pt` only when its bytes would change (`save_if_changed`, atomically), so
T7 v1's `ontology.pt`, which queued T7 jobs read, was not touched.

## 2. The exclusion rule

- **Names.** Every candidate name of a held-out record under T7's selection policy (`mesh_novel.candidate_aliases`: all
  its MeSH terms except bare abbreviations and names under 4 characters), whether it was selected for linking or not,
  plus its linked aliases. For T7's 908 held-out records: 3,647 candidate names, 1,351 linked aliases, 3,646 matcher keys
  (sha256 `e03b683f…`). The held-out records have no abbreviation terms, so "every MeSH term" gives the same set.
- **Matcher** (`NameMatcher`). The T7 screen's `MentionCounter`: case-insensitive, whole alphanumeric tokens with single
  punctuation marks, i.e. word boundaries. "venetoclax-resistant" mentions venetoclax; "fascinating" does not mention
  the held-out protein *fascin*.
  - A left-boundary-only rule (a name followed by anything, the linker's own rule) would add 793 abstracts and 1,254 of
    the first 40,000 general documents. All of these were other words: "fascin" in "fascinating", "calibra" in
    "calibration", "zapa" in "zapateado". The rule is therefore whole-word, as the decision specifies.
  - Prefix matches like these are still *linked* in the evaluation corpora, as on every track (the prefix-causal linker
    does not check the right boundary); that is unchanged from T7 v1.
- **Training only.** The rule applies to every training document: PubMed abstracts, refill abstracts and general
  documents. The evaluation streams (`eval-pubmed`, `eval`, `eval-general`) are untouched.
- **Budget.** T7 v1's training text is every mention-bearing training abstract (the stream ends there), mixed 50/50 with
  FineWeb-Edu. The tokens of the dropped abstracts are refilled from training-side abstracts of the same PubMed files that
  mention no selected name (`top_up: domain`), in stream order, until the SmolLM2 corpus reaches T7 v1's 57,268,320 tokens
  (the document that crosses the budget is kept, as `build_corpus` does). The stream is then a fixed document count, so
  every host tokenizer reads exactly the same documents.

## 3. T7-ROOD: numbers and the leakage audit

| | T7 v1 | T7-ROOD |
|---|---:|---:|
| training documents | 95,044 | 96,112 |
| training tokens (SmolLM2) | 57,268,320 | 57,268,470 |
| mention-bearing abstracts | 68,264 (28,620,116 tokens) | 52,734 (21,861,364 tokens) |
| refill abstracts (no selected name) | 0 | 16,347 (6,497,623 tokens) |
| general documents | 26,780 (28,648,204 tokens) | 27,031 (28,909,483 tokens) |
| PubMed token share | 0.4998 | 0.4952 |
| linked spans in `train` (any length) | 151,665 | 141,455 |
| entries with a training span (ℓ_min 2) | 7,263 | 7,231 |
| held-out set | 908 entries, `5e133b5f…` | the same (`holdout_concepts.txt` identical) |
| alias table | `5df73251…` | the same |

- **Dropped:** 15,530 of the 68,264 mention-bearing training abstracts (22.7%; 6,758,752 tokens). No general document and
  no refill abstract mentioned a held-out name.
- **Against T7 v1, by document text:** 79,514 documents (50.5M tokens) are in both. The 15,530 only in T7 v1 are exactly
  the dropped abstracts (the matcher flags all 15,530). Of the 16,586 only in T7-ROOD, 16,347 are refill abstracts and 239
  are the general documents that follow T7 v1's.
- **Leakage audit:** every written training document was decoded and matched again. **0 training documents contain a
  held-out name**, for SmolLM2 and for Qwen3. The build fails otherwise (`rood_audit_failed.json`).
- **Evaluation corpora** (`eval`, `eval-pubmed`, `eval-general`): identical to T7 v1's: the same `tokens.bin` bytes, span
  arrays and manifests (apart from the track label). The 4,096 `eval-pubmed` windows hold 5,515 held-out occurrences of
  302 entries with ≥ 5 occurrences (the whole split: 8,786 and 411). Feasible at ℓ_min 2, as T7.
- **Side effect:** the dropped abstracts also mention seen records, so seen entries lose 6.7% of their training spans and
  32 entries lose all of them. The refill text has no linked span. Both work against C5 on T7-ROOD.

**Qwen3 relinks** (uint32 ids, NFC decode check, tokenizer fingerprint recorded):

| | T7 v1 (`v1/hosts/qwen3`) | T7-ROOD (`rood-v1/hosts/qwen3`) |
|---|---:|---:|
| `train` | 56,670,205 tokens, 95,044 documents (T7 v1's) | 56,645,735 tokens, 96,112 documents (T7-ROOD's SmolLM2 documents) |
| mention-bearing / refill / general tokens | — | 21,863,997 / 6,464,244 / 28,317,494 |
| `eval-pubmed` | 6,536,229 tokens | identical to T7's Qwen3 relink |
| held-out in the 4,096 windows (ℓ_min 2) | 5,531 occurrences, 310 entries ≥ 5: feasible | the same |
| leaked training documents | — | 0 |

The T7-ROOD Qwen3 training corpus shares 79,514 documents with T7's Qwen3 relink; the 15,530 only in T7's are exactly
the dropped abstracts.

## 4. The E9 track `t7rood`

- `e9_tracks.TRACKS["t7rood"]`: data `rood-v1`, `eval-pubmed`, 4,096 windows, T7's lexicon and relation choices,
  `holdout_names` `runs/rood-v1/holdout_concepts.txt`. Qwen3: `QWEN3_ROOTS["t7rood"]`. Alias table
  `~/data/vsa-llm/e9/alias-tables/t7rood.json` (the same digest as `t7.json`).
- Dimension-3 items (`experiments/e9-retrofit/items/`, built by `e9_tracks items --track t7rood --kind new|edits
  [--family qwen3]`): `new-words-t7rood-{smollm2,qwen3}-v1` and `edits-t7rood-{smollm2,qwen3}-v1`. The new words are
  byte-identical to T7's. Of the 200 edits, the 100 on held-out entries are identical to T7's (327 of 800 items shared); the
  seen half is sampled by training frequency, which differs on T7-ROOD (edited status frequent / mid / rare / held-out:
  2 / 34 / 64 / 100, T7: 3 / 53 / 44 / 100).
- T7 on Qwen3: `new-words-t7-qwen3-v1` and `edits-t7-qwen3-v1` (their `items.jsonl` byte-identical to the SmolLM2 sets),
  `understanding-t7-qwen3-v1` (`e9_understanding items --track t7 --family qwen3`: 16,705 items over 300 seen, 300 rare,
  908 held-out and 300 new anchors; SmolLM2: 16,684). Exclusion table `~/data/vsa-llm/e9/exclusion-tables/t7-qwen3.pt`.
- Configs: `experiments/e9-retrofit/configs/{t7rood,t7-qwen3,t7rood-qwen3}/`; queue commands: `queue-commands-rood.sh`.
  They differ from the T7 configs only in data paths and names.

## 5. E13: the round split and its interface

### 5.1 The split

- **Date.** MeSH `DateIntroduced` of each record (YYYY-MM-DD, from `supp2026.gz`). It is the record-level date that
  replaced `DateCreated` in the 2025+ MeSH XML (the term-level `DateCreated` is a different thing). Fallback: the
  publication year of the first training-side PubMed mention, as YYYY-12-31; no T7 record needs it.
- **Rule** (`t7_rood.date_holdout`): sort the 8,184 entries by date; the cut is the date of the last entry of the
  earliest 70%; round 2 = every entry introduced after the cut (ties at the cut stay in round 1), plus the holdout closure
  (a round-1 entry with an alias that contains a round-2 alias as a whole word moves to round 2, so the linker stays
  alias-disjoint).
- **Exclusion.** Round 2 is the held-out set of a ROOD build with the rule of §2: no round-1 training document mentions a
  round-2 name (audited), and round 1 is refilled to T7 v1's size.

**The split** (`runs/rounds-v1`, 2026-10-08; round-2 UI list sha256 `03f35409…`, pinned in `t7-rounds.yaml`):

- **Cut:** 2013-01-26. Round 1 has 5,714 entries (introduced 1968 to 2013-01-26). Round 2 has 2,470 entries (30.2%):
  2,455 introduced after the cut and 15 moved by the closure.
- **Round-2 dates by year:** 692 are from 2020 and 418 from 2021 (the COVID-era records); 141 from 2022 and 266 from 2023–26.
- Every date is a `DateIntroduced`; the fallback was never used.

| SmolLM2 tokens (Qwen3 in brackets) | documents | tokens | linked spans | PubMed share |
|---|---:|---:|---:|---:|
| `train` (round 1) | 96,437 | 57,268,379 (56,624,295) | 108,417 | 0.50 |
| … of which mention-bearing abstracts / refill / general | 43,382 / 25,888 / 27,167 | 17,905,426 / 10,297,681 / 29,065,272 | | |
| `train-round2` | 34,547 (24,882 abstracts + 9,665 general) | 21,326,939 (21,103,066) | 83,223 | 0.502 |
| `eval-round2` | 5,628 abstracts | 2,417,494 (2,413,915) | 18,010 | — |
| `eval-round1` | 9,971 abstracts | 4,121,210 (4,122,314) | 23,930 | — |
| `eval-pubmed` (all) | 15,599 abstracts | 6,538,704 | 41,940 | — |

- **Dropped from round 1:** 24,882 of the 68,264 mention-bearing training abstracts (10,714,690 tokens), refilled by
  25,888 abstracts without a selected name. **Leakage audit: 0 round-1 training documents contain a round-2 name**
  (SmolLM2 and Qwen3; 6,416 matcher keys over the candidate names of 2,470 records).
- **Coverage.** 5,663 of the 5,714 round-1 entries have a training span in round 1 (ℓ_min 2). 2,467 of the 2,470
  round-2 entries occur in `train-round2` (74,963 spans at ℓ_min 2).
- **Round-2 evaluation.** The whole `eval-round2` split holds 2,359 windows of 1,024 tokens (Qwen3: 2,356). In them:
  16,337 round-2 occurrences (ℓ_min 2) of 1,526 entries, 706 of them with ≥ 5. Use every window
  (`eval.windows: 2359`). The feasibility verdict "rare stratum underpowered" concerns round-1 rare entries and does not
  matter here.
- **Definitions.** 1,060 round-2 records have an SCR note; 907 keep ≥ 3 words once the bibliographic parts are removed.
  906 have a linkable headword (`round2-items-smollm2-v1`); 532 of these occur in `eval-round2` (8,519 occurrences).
- **Evaluation corpora.** `eval`, `eval-pubmed` and `eval-general` have the same tokens and span arrays as T7 v1's. Their
  manifests differ only in `alias_table_sha256`, because the held-out set differs, so the build's comparison prints
  "no".

### 5.2 Files

Data root `~/data/vsa-llm/tracks/t7-newvocab/rounds-v1/` (Qwen3: the same names under `hosts/qwen3/`). Entry ids are
T7's in every build (T7 v1, T7-ROOD, rounds; the same alias table and ontology), so frames, items and spans line up.

| Path | What | Linked with |
|---|---|---|
| `train/` | **round-1 training corpus** (50/50 PubMed/general, 57.27M SmolLM2 tokens); `train/sources.json` (per-source tokens), `train/rood.json` (document budget, stream tally, audit) | round-1 aliases only (round-2 names never occur) |
| `train-round2/` | **round-2 training corpus**: every training-side abstract that mentions a round-2 name (the documents round 1 dropped), mixed 50/50 with the general documents after round 1's; `sources.json` | all aliases |
| `eval-round2/` | **round-2 evaluation text**: the evaluation abstracts (PMID bucket < 2,000) that mention a round-2 name | all aliases |
| `eval-round1/` | the evaluation abstracts without a round-2 name (forgetting on round-1 text) | all aliases |
| `eval-pubmed/`, `eval/`, `eval-general/` | T7's evaluation corpora (identical to T7 v1's); `eval-general` for general-text locality | all aliases |
| `ontology.pt` | round-1 channel ontology: `heldout_entries` = round 2; `train_frequency` from round 1 | |
| `ontology-round2.pt` | round-2 view: `heldout_entries` = [], `round2_entries` = round 2, `train_frequency` = round 1 + round 2, plus `train_frequency_round1`, `train_frequency_round2`, `round: "2"`; the same frames, atomics and `alias_table_sha256` | |

Run folder `experiments/t7-new-vocabulary/runs/rounds-v1/` (committed):

| File | What |
|---|---|
| `round_split.tsv` | every entry: `entry, ui, name, introduced, date_source, round (1/2), reason (date / closure / empty)` |
| `round2_records.jsonl` | every round-2 entry: `entry, ui, name, introduced, reason, aliases` (linked forms), `frame` (gold: `[relation, atom, filler name]`), `note` (the SCR `<Note>`), `definition` (the note without its bibliographic parts, `e11_read_to_learn.scr_definition`, or null), `occurrences_train-round2`, `occurrences_eval-round2` (SmolLM2 spans, ℓ_min 2) |
| `round2-items-smollm2-v1/` | the E11-format held-out set (`e11_read_to_learn.build_heldout_set`, the code of E11's `items/t7-heldout-smollm2-v1`): `concepts.jsonl` (entry, headword, gold and random frames), `definitions.jsonl` (`'<headword>: <note>'`, style `scr`), `manifest.json`. Only entries with an informative note and a linkable headword |
| `holdout_concepts.txt`, `holdout_entries.tsv` | the round-2 UIs (sha256 pinned in `t7-rounds.yaml`) |
| `summary.json` → `rood.rounds` | the cut date, counts, per-tokenizer manifests and shares of the round corpora, round-2 feasibility on `eval-round2` (windows needed per ℓ_min), the file digests; `rood.audit` = the round-1 leakage audit |

### 5.3 How to load them

```python
import json, torch
from pathlib import Path
from vsa_embed.data.corpus import TokenCorpus

root = Path("~/data/vsa-llm/tracks/t7-newvocab/rounds-v1").expanduser()      # Qwen3: root / "hosts" / "qwen3"
round1 = TokenCorpus.open(root / "train")                  # .tokens (memmap), .spans (start/end/inject/entry/length), .manifest
onto1 = torch.load(root / "ontology.pt", weights_only=False)        # heldout_entries = round 2
onto2 = torch.load(root / "ontology-round2.pt", weights_only=False) # heldout_entries = [], round2_entries
round2_entries = set(onto2["round2_entries"])
records = [json.loads(l) for l in open("experiments/t7-new-vocabulary/runs/rounds-v1/round2_records.jsonl")]
```

Trainer configs (`training.lm`, the E9 recipe):

- **Stage 0, passive round 1:** `data.train = <root>/train`, `data.ontology = <root>/ontology.pt`, `data.eval =
  <root>/eval-round2` (the trainer's `after_heldout` stratum is then the loss after round-2 names, zero-shot) or
  `<root>/eval-pubmed`.
- **Stage 4, passive round 2:** `data.train = <root>/train-round2`, `data.ontology = <root>/ontology-round2.pt`, starting
  from the round-1 checkpoint (`train.init_from`). In the round-2 ontology nothing is held out (the channel trains the
  round-2 rows), so the "after a round-2 term" curve must be read with `round2_entries` (e.g. from `eval_windows.npz` with a
  reference-strata file, or by rescoring). The trainer's own `after_heldout` stratum is empty there.
- **Forgetting:** `<root>/eval-round1`. **Locality:** `<root>/eval-general`.
- **Read to learn (stage 2):** the definitions in `round2-items-smollm2-v1/definitions.jsonl` (or `definition` in
  `round2_records.jsonl` for every round-2 entry), the gold frames for the oracle reader, the random frames for the
  content control. Feed them to the E11 reader as `e11_read_to_learn evaluate --items .../round2-items-smollm2-v1`.

## 6. Checks run

- `tests/test_t7_rood.py` (9 tests): the section is opt-in and T7's config is unchanged; `save_if_changed`; SCR dates;
  word-boundary matching; drop, refill and the exact document budget on a toy stream; the date split with ties and the
  closure; the first-mention fallback; the audit (leaks, composition, overlap); the `t7rood` track and its plan at the
  fractional priority.
- **T7 v1 reproduction.** `t7.yaml` rebuilt with this code into a scratch data root (2026-10-08, 7.6 min). It was
  identical to `tracks/t7-newvocab/v1`: `tokens.bin`, span arrays, `manifest.json` and `sources.json` of `presample`,
  `train`, `eval`, `eval-pubmed`, `eval-general`, and `ontology.pt` byte for byte. (`spans.npz` is a zip whose member
  timestamps differ between any two builds, so its arrays are compared.) The scratch copy was deleted.

## 7. Follow-ups (2026-10-08, after the merge; nothing queued)

Queue commands: `queue-commands-rood-followups.sh`, printed by `python -m vsa_embed.experiments.t7_rood_queue`.

| Block | Priority | Jobs | GPU-h (idle) | What |
|---|---|---:|---:|---|
| paired comparison | 54.4974 | 1 | CPU | `t7rood-vs-t7-report`: `t7_rood_compare` on `runs/t7rood` vs `runs/t7` (SmolLM2-360M, seeds 1–3) → `e9-retrofit/report/t7rood-vs-t7` |
| paired comparison, Qwen3 | 54.4978 | 1 | CPU | `t7rood-vs-t7-qwen3-report`: the same on `t7rood-qwen3` vs `t7-qwen3` (seed 1; flagged single seed) |
| E11 read-to-learn | 54.4979 | 8 | ≈ 3.0 | the T7-ROOD held-out records read from their SCR notes by C5 (every reader; frames, gradient, windows, locality) and C0′ (gradient, windows), seeds 1–3; its own gradient-lr job on the T5 dev words (`dev-t7rood/`, so it does not wait for E11's dev job at 62); the E11 report `report-t7rood` |
| E12 recall | 54.49795 | 11 | ≈ 2.7 | every T7-ROOD SmolLM2-360M run on `understanding-t7rood-smollm2-v1`: the reverse family and the relation families (paraphrase, negation, affordance), core conditions `none` / recall / `symbolic`; the E12 report `e12-self-query/report/t7rood-understanding` |

- **Paired comparison** (`src/vsa_embed/experiments/t7_rood_compare.py`): the pre-registered P2 and P3. It computes each
  track's relative difference Σ (candidate − reference) / Σ reference over the final-evaluation windows, pooled over
  seeds, and Δ = T7-ROOD − T7 v1. The windows are the same on both tracks, and the script refuses windows that do not
  pair. One 95% cluster bootstrap over windows (10,000 resamples) uses the same resampled windows for both tracks. The
  readings are the pre-registered ones.
- **E11** (`e11_read_to_learn`: the `t7rood` track): `experiments/e11-read-to-learn/items/t7rood-heldout-smollm2-v1`, built
  by `e11_read_to_learn items --kind heldout --track t7rood`. It has the same 487 records, definitions and evaluation
  occurrences as T7's `t7-heldout-smollm2-v1`; only the concept ids differ (`t7roodh-…`). The held-out set and the
  evaluation corpus are the same.
- **E12** (`e12_self_query`). T7 has no role-swap twins and no two-hop items (MeSH heading fillers have no frames). The
  Q1-style test therefore reads:
  - the **reverse** family (reverse lookup over every store, as for T5);
  - the opt-in **relation families** `RELATION_FAMILIES` = paraphrase, negation, affordance (new; `--families`). They ask
    about one relation of the anchor's own frame, so the anchor's whole-frame recall (slot-aware, as for new words) goes in
    context. `slot_correct` records whether that recall stated the gold filler of the item's relation.

  Without `--families` E12 scores two-hop and reverse items as before. The `definition` condition is left out: there is
  no definition writer for MeSH frames.
- **Understanding items** (`e9_understanding`: `t7rood` uses T7's spec): `understanding-t7rood-smollm2-v1` (16,697 items)
  and `understanding-t7rood-qwen3-v1` (16,730 items). Each has 300 seen, 300 rare, 908 held-out and 300 new anchors. The
  seen and rare anchors are sampled by T7-ROOD's training frequencies. Exclusion tables:
  `~/data/vsa-llm/e9/exclusion-tables/t7rood-{smollm2,qwen3}.pt`.
- **SMOKE** (not a result; `smoke-followups/`, `smoke_cpu.py`). A T7-ROOD stage was trained on CPU (SmolLM2-135M LoRA,
  P0 / C0′ / C5, 4,096 tokens on the real `rood-v1` corpora, 16 evaluation windows). Every follow-up then ran on it with
  small limits: E11 on C5 and C0′, the E11 report, E12 on C5 and C0′ with the relation families, the E12 report, and the
  comparison against a copy of the stage (Δ = 0 exactly). The outputs are in `smoke-followups/outputs/`.

## 8. Decisions (author, 2026-10-08)

- **Rounds and H1.** The rounds stay as built. Round 2 is the only held-out set there; the 908 T7-ROOD held-out records
  are ordinary round-1 or round-2 records and are not excluded a second time.
- **Refill text.** The PubMed refill stays: abstracts that mention no selected name. `execution.md` discloses that seen
  terms lose 6.7% of their training spans. T7-ROOD vs T7 v1 is a secondary comparison; C5 − C0′ within T7-ROOD is the
  primary.
