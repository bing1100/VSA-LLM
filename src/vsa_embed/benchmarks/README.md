# Ranking benchmarks: item format and harness (`rank-items/1`)

Decision 63 (toolkit methodology, `manuscript/toolkit-methodology-2026-10.md` §3): every *read*, *write* and *meta*
benchmark that small hosts can run is scored by **log-probability ranking**. One harness
(`vsa_embed.benchmarks.ranking`) scores every set on any trained E9 run; adapters (`vsa_embed.benchmarks.adapters`)
turn public sets into items. Other work packages (TK-B2 Wikidata track, TK-H3L ICD-10-CM / MedConceptsQA items)
write items in the same format.

## Item format

One JSON object per line (`.jsonl`, or `.jsonl.gz`); ids unique within a file.

```json
{"id": "comps-wugs-1", "set": "comps-wugs", "kind": "rank",
 "context": "Therefore, a wug",
 "options": [" attaches to rocks.", " attaches to rocks."],
 "answer": 0,
 "terms": [],
 "option_terms": [[{"surface": "wug", "concept": null, "frame": [["hypernym", "synset:bivalve.n.01"], ["lexname", "lexname:noun.animal"], ["pos", "pos:n"]], "definition": "A wug is a mussel."}],
                  [{"surface": "wug", "concept": null, "frame": [["hypernym", "synset:clam.n.01"], ["lexname", "lexname:noun.animal"], ["pos", "pos:n"]], "definition": "A wug is a clam."}]],
 "joiner": " ",
 "meta": {"comps_id": 1, "negative_sample_type": "taxonomic"}}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | unique item id |
| `set` | yes | set name; reports break results down by set |
| `kind` | no (default `rank`) | only `rank` is supported |
| `context` | yes (may be `""`) | text before every option |
| `options` | yes, ≥ 2 | continuations; option *i* is scored as `log p(options[i] \| prefix)`. **Spacing is the writer's job**: `context + option` must be the full text (options usually start with a space) |
| `answer` | yes | index of the correct option |
| `terms` | yes (may be `[]`) | the terms the item is about (shared by all options) |
| `option_terms` | no | one list of terms per option, in force only while that option is scored: minimal pairs over *what is known about a term* (COMPS: the same continuation under "a wug is a mussel" vs "a wug is a clam") |
| `joiner` | no (default `"\n"`) | separates reading material (definitions, verbalized frames) from the context |
| `meta` | no | anything; scalar fields are copied to the per-item results so reports can filter on them (`where`). Keep entity ids here (`wikidata_qid`, `type_qid`, …) |

A **term**:

| Field | Meaning |
|---|---|
| `surface` | the string the term appears as in the context / options (linked case-insensitively at word boundaries, ≥ the run's ℓ_min subtokens) |
| `concept` | `null` for a term the store must **write** (a new word); otherwise an id of a concept the store already knows (WordNet `synset:dog.n.01`, a Wikidata QID, an ICD code …) |
| `insert` | optional; default `concept is null`. `true`: a row is written for the term under `surface` (an existing alias with the same surface is shadowed for the item, counted in the output). `false`: the run's own linker links it as in training and nothing is written |
| `frame` | gold frame `[[relation, atomic], …]` by **name** in the run's ontology (`ontology.pt`: `relation_names`, `atomic_names`); `null` if none. Names unknown to the run make the frame unusable for that run (counted as `unresolved_frames`) |
| `alt_frames` | optional `{name: frame}`: alternative gold frames, read by the reader `oracle:<name>` |
| `definition` | reading material (`null` if none): prepended by `definition-in-context`, read into a frame by the definition readers (it must contain `surface`) |

Write items with `ranking.write_items(path, items)` (deterministic gzip) and give the set a `manifest.json` next to
`items.jsonl.gz`: schema, source, licence, pinned commit, file digests, build rule, counts, coverage (see
`adapters.build`).

## Conditions and readers

`ranking evaluate --conditions` takes `+`-joined parts: `none` (new terms get no row), `channel-off` (no span at all),
`store:<reader>` (new terms' rows composed from the reader's frame), `frame-in-context:<reader>` (the frame verbalized
and prepended), `definition-in-context`. Readers: `oracle`, `oracle:<alt>`, `random` (equal-degree random frame),
`none`, and the E11 definition readers `typeprior`, `pattern`, `stated`, `linker`, `linker-all`, `linker-joint`.
Runs without a channel skip `store` and `channel-off`; the model readers need a composing channel (C5-type).

## Scores and statistics

Per option: summed log-probability (`sum`, primary), per UTF-8 byte (`per_byte`, lm-evaluation-harness `acc_norm`)
and per token (`per_token`). A tie that contains the answer earns 1/(tied options). Identical requests are scored
once, so ties are exact. Outputs per run × set: `items.jsonl.gz` (per item and condition: option scores, token and
byte counts, correctness under each normalization, context tokens added, link rate), `readers.jsonl.gz` (frames
written by each reader, precision/recall against the gold frame, cost), `summary.json`, `report.md`, provenance
(`manifest.json`, `resolved_config.yaml`). `ranking report` pools outputs over seeds (items as clusters) and models,
evaluates contrasts given as JSON (`{"primary": [...], "secondary": [...]}`, each `{name, set?, where?, a: {model,
condition}, b: {...}}`; `model` = `<host>-<model>`, e.g. `SmolLM2-360M-C5`) with paired percentile bootstraps
(2,000 resamples) and Holm over the primaries.

## Sets built here (`adapters build --set …`; raw files: `sources fetch`)

| Set | Items | Frames | Licence | In the repository |
|---|---|---|---|---|
| `comps-wugs-wordnet-v1` | 13,896 minimal pairs | WordNet is-a (all 521 parents mapped; 224 exact atomics) | Apache-2.0 | items + manifest |
| `alcuna-wordnet-v1` | 5,000 (3,000 multiple choice, 2,000 Yes/No) | WordNet is-a of the parent taxon (1,429 / 3,554 entities map) | MIT | items + manifest |
| `entity-inferences-v1` | 170 probes | none yet (TK-B2) | none stated | manifest (digests) |
| `reversal-v1` | 1,200 (4 × 300) | none yet | MIT | manifest |
| `lre-v1` | 11,089 (47 relations) | none yet | MIT | manifest |
| `bear-v1` | 23,193 (60 relations × templates) | none yet | CC BY-SA 4.0 | manifest |
| `popqa-v1` | 14,267 | none yet (subject QIDs kept) | none stated | manifest |

Raw files and full item files live under `~/data/vsa-llm/benchmarks/<source>/`; `~/data/vsa-llm/DATA_SOURCES.md`
records URLs, pinned commits, licences and sha256.
