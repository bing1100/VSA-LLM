# T4 chemistry (E8, task C7)

ChEBI as the ontology, IUPAC and trivial names as multi-token aliases, ChEBI entry texts and PubMed
chemistry abstracts as the domain corpus — the purest compositionality test (IUPAC names are
compositional, functional groups and elements are natural atomics).

## Sources and licences (details and hashes in `~/data/vsa-llm/DATA_SOURCES.md`, section "C7 application tracks T3–T6")

- **ChEBI release 255** (2026-09-09), `chebi.obo.gz` from the EBI FTP, CC BY 4.0
  ("ChEBI data is from https://www.ebi.ac.uk/chebi — version 255"). sha256 `c2765eb4599fbcbb…`.
- **PubMed 2026 baseline** files `pubmed26n1315`–`pubmed26n1326` (NLM; abstracts are provided under
  NLM's terms, copyright of the publishers), downloaded by the T1 work package to
  `~/data/vsa-llm/pubmed/baseline-2026/` and only read here (md5-verified against NCBI's `.md5` files
  at build time); no PubMed file was downloaded twice.
- **FineWeb-Edu** (ODC-By) as the general half of the training stream.
- PubChem compound descriptions were not downloaded: PubChem has no bulk description file, and its
  ChEBI-sourced descriptions are generated from the same ChEBI fields, so the ChEBI entry texts below
  play that role (substitute, documented).

## Recipe

1. **Ontology** (`ontologies/chebi.py`): the 62,418 non-obsolete 3-star entities; frames (≤ 24 edges)
   = `is_a`, the ChEBI relationships (has_functional_parent, has_parent_hydride, has_part,
   conjugate acid/base, tautomer, enantiomer, substituent group; up to 8 `has_role` before the
   atoms below), `contains_element` (formula elements except H), `charge`, `branch`. 8,192 atomics
   (the most-used filler entities plus element/charge/branch atoms; 35,804 out-of-dictionary fillers
   replaced by their nearest in-dictionary ancestor). 252,825 aliases (names + EXACT/RELATED
   synonyms: IUPAC names, INNs, brand names; formula strings, stopwords, ≤ 4-letter synonym
   abbreviations and aliases > 120 characters dropped) in 67,545 link entries.
2. **Corpus.** ChEBI entry texts (name, definition and PubChem-style relation sentences; 51,260
   entities, 10% by hash for evaluation) and PubMed abstracts kept if they have a MEDLINE
   ChemicalList or link ≥ 3 distinct ChEBI entities through long aliases (122,189 of 315,143
   abstracts with text; 3% by hash for evaluation). Domain training tokens 50.5M + FineWeb-Edu 49.5M =
   100.0M SmolLM2 tokens (domain fraction 0.51); domain evaluation 2.0M tokens.
3. **Holdout** (C3 procedure on a 5M-token presample): 10% of the 3,641 entries with ≥ 5 occurrences
   and an alias of ≥ 2 subtokens, stratified by log-frequency → 429 concepts, 551 entries (every
   entry sharing an alias with a held-out concept is held out too). `holdout_concepts.txt`, sha256
   `b58e504f8b060726…`. Held-out names still occur unlinked in training text (the C3 linker-holdout
   protocol); the strictly contamination-free test is the synthetic set.
4. **Synthetic compounds** (E5.4 "new compounds by IUPAC name"): 299 ring-substituted derivatives of
   15 aromatic parents ("6-bromo-8-methoxycoumarin", lowest-locant numbering on symmetric rings),
   absent from every ChEBI name/synonym (all stars) and from the text samples; frames carry the family
   class, substituent classes, functional parent, elements and one **invented role**, so role
   questions can only be answered from the frame (PubChem could not be searched offline, so a name may
   exist there; the role facts are fictional by design).
5. **Feasibility** (`runs/v1/report.md`): **feasible at ℓ_min = 2** with 256 evaluation windows (211
   held-out entries / 1,202 spans, 784 rare entries / 1,048 spans, covered fraction 0.15); ℓ_min = 3
   needs 512 windows, ℓ_min = 4 needs 1,024.
6. **Items**: `class_role_probe` (IUPAC name in a neutral sentence → frequent direct classes and
   roles; linear probe; 1,060 train / 136 held-out / 299 synthetic), `property_cloze` (multiple choice
   over is_a / has_role / functional parent / conjugate / parent hydride / enantiomer; 2 paraphrases),
   `zeroshot_property` (3 paraphrases) and `zeroshot_entailment` on synthetic and held-out compounds.

## Build

```bash
PYTHONPATH=src python -m vsa_embed.experiments.track_corpus \
  --config experiments/t4-chemistry/t4.yaml --output experiments/t4-chemistry/runs/v1
```

≈ 7 min with 3 tokenizer workers, 3.0 GB peak RSS (main process; each worker holds a ≈ 0.6 GB linker).
Outputs in `~/data/vsa-llm/tracks/t4-chemistry/v1/`.
