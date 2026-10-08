# Toolkit *learn* data and holdout H3 (decision 63, TK-H3L)

Evaluation data for the *learn* tool (place records that are new in a later release of an ontology into the earlier
release) and for holdout H3 (ICD-10-CM FY2027 codes, created after every host's training data). Plan:
`manuscript/toolkit-methodology-2026-10.md` §2 M1 (H3), §3.2 (learn benchmarks), §3.3 (FY2027 codes).
No model has been run on any of it.

Code: `src/vsa_embed/benchmarks/learn_data.py` (formats, splits, snapshots, pinned downloads, CLI),
`learn_mesh.py`, `learn_icd10cm.py`, `learn_public.py`. Tests: `tests/test_toolkit_learn_data.py` (synthetic only).

```bash
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
PYTHONPATH=src $PY -m vsa_embed.benchmarks.learn_data fetch            # download or verify every pinned source (sha256)
PYTHONPATH=src $PY -m vsa_embed.benchmarks.learn_data icd10cm          # H3 items + FY2026 snapshot            (~10 s)
PYTHONPATH=src $PY -m vsa_embed.benchmarks.learn_data icd10cm-umls     # licensed UMLS map, local only          (~1 min)
PYTHONPATH=src $PY -m vsa_embed.benchmarks.learn_data mesh             # MeSH time split + PubMed evidence      (~4 min, 6 workers, 3 GB)
PYTHONPATH=src $PY -m vsa_embed.benchmarks.learn_data medconceptsqa    # (~1 min)
PYTHONPATH=src $PY -m vsa_embed.benchmarks.learn_data oet              # SNOMED, local only                     (~1 min)
PYTHONPATH=src $PY -m vsa_embed.benchmarks.learn_data taxo             # TaxoExpan / TMN, local only           (~1 min)
```

Every build is deterministic; `manifest.json` in each item folder pins the sha256 of every file (committed or local).

## Sets

`items/` = committed (`experiments/toolkit-learn/items/<set>-v1/`); `local` = `~/data/vsa-llm/toolkit-learn/<set>-v1/`.

| Set | Before → after | Items (dev / test) | Gold | Where | Licence |
|---|---|---|---|---|---|
| `mesh-2025-2026` | MeSH 2025 → 2026 | primary 179 (26 / 153): 120 descriptors, 59 SCRs; low-evidence 520; new edges 584 | descriptor: tree parents; SCR: HeadingMappedTo + pharmacological action | items; snapshots local | public domain (NLM terms: acknowledge, mark modified) |
| `icd10cm-fy2027` (H3) | ICD-10-CM FY2026 → FY2027 | 238 placement (48 / 190); 476 choice (96 / 380) | nearest FY2026 ancestor + block + chapter | items; snapshot and UMLS map local | public domain (CMS/NCHS); UMLS map licensed |
| `medconceptsqa-icd10cm` | — (ranking) | 264,256 test + 12 dev (local); dev + 1,511-item frozen sample committed | answer option | items (dev, sample); full local | Apache-2.0 |
| `oet-snomed-{disease,cpp}` | SNOMED CT US 2014-09-01 → 2017-03-01 | disease 66 / 48 concepts; cpp 143 / 97 | 2014 parents (atomic or complex) + children | local only (mode 700); manifest committed | SNOMED-derived (licence held) |
| `taxoexpan-semeval-{noun,verb}` | WordNet 3.0 + SemEval-2016 T14 lemmas | noun 349 / 516; verb 51 / 84 (official split) | parents in the full taxonomy | local only; manifest committed | no explicit data licence |
| `tmn-wordnet-{noun,verb}` | same files, TMN protocol | 1,000 / 1,000 each (**re-drawn split**) | nearest kept parents and children | local only; manifest committed | no explicit data licence |

## Formats

**Placement item** — one per new record (`learn_data.validate_placement`):

```json
{"id": "mesh-2025-2026:D000099097", "set": "mesh-2025-2026", "record": "D000099097", "name": "Aerogels",
 "aliases": ["Aero-Gels", "Aerogel", "Silica Aerogels", …], "definition": "A synthetic porous ultralight solid …",
 "gold_parents": ["D005782"], "gold_relations": [["parent", "D005782"]], "candidates": null,
 "evidence": {"pubmed_2025_26_docs": 678, "pubmed_2025_26_eval_docs": 170, "pubmed_2025_26_pmids": […], "pubmed_indexed_docs": 37, …},
 "meta": {"kind": "descriptor", "split": "test", "primary": true, "t7_selected": false, "t7_heldout": false, …}}
```

- `candidates: null` means every node of the set's before snapshot (MeSH: the 2025 *descriptors*; ICD: every FY2026
  code, block and chapter; OET: the 2014 catalogue incl. complex classes; taxonomies: the training taxonomy).
- A prediction is a hit if it is any of `gold_parents` (Hits@k, MRR over the candidates). `gold_relations` holds the
  typed edges the frame should contain (`parent`, `mapped_to`, `pharmacological_action`, `see_also`, `block`,
  `chapter`, `child`).
- `meta.split` is `dev` or `test` (`learn_data.split_of`: sha256 of `toolkit-learn-v1:<record>`, 20% dev), except where
  a benchmark has its own split (OET, TaxoExpan; TMN re-drawn), which is kept and named in `meta`.

**Ranking item** (`learn_data.validate_rank`; ICD choice items, MedConceptsQA): `{"id", "set", "kind": "rank",
"context", "options", "answer", "terms": [{"surface", "concept"}], "meta"}`. Score each option by log-likelihood after
`context`; `meta.mcq_prompt` holds the lettered MedConceptsQA prompt for prompting evaluations.

**Before snapshot** (`learn_data.write_snapshot` / `load_snapshot`): `nodes.jsonl.gz` (`{"id", "name", "aliases",
"definition", "kind", "meta"}`), `edges.jsonl.gz` (`{"source", "relation", "target"}` = source --relation--> target; for
`parent`, target is the parent) and `snapshot.json` (counts, sha256). MeSH also gets `ontology.pt` in the trainer's
channel-ontology format (`t1_open_corpus.channel_ontology`; T1 alias policy, 30,763 entries, 8,192 atomics, zero
training frequencies) — the 2025 descriptors as the frame ontology a model would start from.

**New edges** (`mesh-2025-2026/new-edges.jsonl`): `{"id", "set", "source", "kind", "relation", "target",
"removed_in_2026", "meta": {"split"}}` — edges added in 2026 between records that both exist in 2025 (descriptor
parent 53, see-also 6, pharmacological action 1; SCR mapped heading 430, pharmacological action 94): the "update the
mapping" gold of §3.2.

## Per-set notes

**MeSH 2025 → 2026** (new benchmark). Before: NLM's 2025 archive (`desc2025.gz`, `supp2025.gz`, Last-Modified
2026-01-08). After: the T7 files (`desc2026.gz`, `supp2026.gz`, 2026-08-12). 160 new descriptors, 539 new SCRs (534
introduced 2026; NLM now creates few SCRs), 6 / 449 deleted.
- Evidence: abstracts of T7's local PubMed text (20 newest baseline + 80 newest update files, PMID ≥ 41,025,505,
  1,773,541 abstracts) that mention a record by any name (T7 matcher and alias policy; names shared with another 2026
  record not counted). `pubmed_2025_26_eval_docs` = abstracts on T7's evaluation side. Descriptors also carry
  `pubmed_indexed_docs` (NLM indexing). Primary = ≥ 5 mentioning abstracts and a 2025 gold parent.
- 93 of the 160 new descriptors were 2025 SCRs with the same name (`meta.promoted_from_scr_2025`): new as headings,
  not as names. 265 new SCRs map to a heading that is itself new in 2026 (`meta.mapped_new`); their gold is that
  heading's nearest 2025 ancestors (only 2 of them are in the primary set).
- **T7 overlap.** 72 of T7's 8,184 selected records are new in 2026 (11 in T7's holdout). In the primary set, 53 of
  the 59 SCRs are T7-selected (7 held out): T7 checkpoints were trained with those names linked to their gold frames.
  On T7 models, evaluate SCRs only where `t7_selected` is false or `t7_heldout` is true; the 120 descriptors are
  unaffected (T7 links SCRs only).

**ICD-10-CM FY2027 (H3).** 190 new billable codes + 48 new headers = 238 new code strings (this is the "190–238" of
the plan). The April 1, 2026 update changed no codes, so FY2026 is one code set (tabular XML of April 2026).
- Gold parent = nearest FY2026 ancestor: 224 codes, 14 blocks (new categories such as `K6A`). 161 of the 238 have a new
  immediate parent; 44 gold parents were billable in FY2026 (an existing code subdivided). FY2027 blocks are mapped to
  FY2026 blocks (`K65-K6A` → `K65-K68`).
- Definition = title + tabular inclusion terms + includes notes (32 codes have inclusion terms, used as aliases).
- Choice items: `code2title` (MedConceptsQA wording) and `title2code`; 3 distractors = ≤ 2 siblings + same-chapter
  codes, never an ancestor or descendant, seeded per code.
- UMLS 2022AB map (local, `umls-2022ab/`; ICD10CM_2023, MSH2023, SNOMEDCT_US 2022-09-01): of the 41 distinct gold-parent
  codes, 40 have a CUI, 8 reach a MeSH descriptor and 13 a SNOMED concept; exact title matches give a CUI for 38 new
  codes (MeSH 12, SNOMED 17). Frames for E13 will mostly have to come from the ICD hierarchy itself.

**MedConceptsQA.** ICD-10-CM easy / medium / hard (dev 4 each; test 94,576 / 81,753 / 88,009 rows); 82 test rows with
duplicate options are skipped. `meta.in_fy2026` / `in_fy2027` mark codes deleted since (≈ 0.3%). No H3 code can occur.

**OET (SNOMED time split).** The Zenodo ver4 zip is self-contained: both SNOMED CT US releases as OWL plus the 2014
entity and edge catalogues — nothing beyond the zip is needed (our SNOMED CT International 2022-05-31 is not used).
Rebuilding OET from raw releases would need SNOMED CT **US Edition** 20140901 and 20170301 RF2, UMLS 2017AA and
MedMentions, which we do not hold. Splits: `valid-NIL` → dev, `test-NIL` → test. OET splits *mentions*: 20 (disease)
and 38 (cpp) test concepts also occur in dev (`meta.also_in_dev`; filter for a concept-disjoint test). 4 gold parents
are new complex classes absent from the 2014 catalogue (`meta.gold_parents_not_in_snapshot`). The official protocol
ranks edges `<parent, child>` from `snapshot-2014/candidate_edges.jsonl.gz`; gold edges are in `meta.positions`;
MedMentions contexts in `meta.contexts`.

**TaxoExpan / TMN.** TaxoExpan's "SemEval-Noun/Verb" = WordNet 3.0 noun/verb taxonomies + the SemEval-2016 Task 14
lemmas; its split is read from the dataset pickle (stub unpickler, nothing executed). Test = the 600 SemEval test
lemmas (leaves, no definitions in the files). TMN's WordNet-Noun/Verb are the same files (83,073 / 13,936 nodes), but
TMN's split downloads are gone (Google Drive 404), so the TMN protocol (1,000 + 1,000 random non-root nodes, parents
bridged to children) is re-drawn with seed 20210202 — **not the official split**; results are not comparable to TMN's
table. Glosses for synset nodes come from NLTK WordNet. MAG sets skipped (MAG retired).
Leakage (test split; siblings in training are allowed by both protocols):

| Set | test | with a training sibling | mean training siblings | name = a training node | parent bridged |
|---|---:|---:|---:|---:|---:|
| taxoexpan-semeval-noun | 516 | 79% | 11.4 | 68 | 0 |
| taxoexpan-semeval-verb | 84 | 69% | 12.5 | 12 | 0 |
| tmn-wordnet-noun | 1,000 | 92% | 32.9 | 290 | 29 |
| tmn-wordnet-verb | 1,000 | 87% | 20.6 | 646 | 120 |

"Name = a training node" is mostly WordNet polysemy (another sense of the same lemma). Every TMN test synset (99%) is
also a concept of the repository's WordNet track: models trained there have seen its edges.

## For TK-L and TK-E13

- Real-data gold for *learn*: placement of new concepts (all sets) and new edges for known concepts
  (`mesh-2025-2026/new-edges.jsonl`). Tune on `dev`, report `test` once.
- Contamination: H3 codes postdate every host. The MeSH records are new as MeSH entries, but most names occur in
  2025–26 PubMed text and 93 new descriptors reuse 2025 SCR names, so hosts trained on recent text may know the words;
  `evidence.pubmed_2025_26_pmids` is a sample of pointers, not an exclusion list (document exclusion is TK-H1's
  T7-ROOD). OET contexts are 2017-era PubMed abstracts (likely in every host's pretraining).
- Licensed outputs (OET, UMLS map) must stay under `~/data/vsa-llm/toolkit-learn/` (decision 58); `.gitignore` guards
  the item folders.
