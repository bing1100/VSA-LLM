# T6 legal / regulatory (E8, task C7)

EuroVoc (the EU's multilingual thesaurus) as the ontology, its English preferred and non-preferred
terms as aliases, English EU legislation (MultiEURLEX) as the corpus.

## Sources and licences (details and hashes in `~/data/vsa-llm/DATA_SOURCES.md`, section "C7 application tracks T3–T6")

- **EuroVoc 4.24** SKOS core export (Publications Office Cellar, published 2026-07-08; sha256
  `91bdb24e833ba431…`), © European Union, reuse authorised (Commission reuse policy).
- **MultiEURLEX** (Chalkidis, Fergadiotis & Androutsopoulos 2021; CC BY-SA 4.0), English texts and EuroVoc
  labels streamed from the Hugging Face archive (`coastalcph/multi_eurlex` revision `2020d035…`; the
  archive's sha256 matched the hub LFS oid — the previous machine's download corruption did not recur).
- **FineWeb-Edu** (ODC-By) for the general-text locality set (and the training fill, if any).

## Recipe

1. **Ontology** (`ontologies/eurovoc.py`): 7,515 descriptors; frames = broader (BT), top term,
   microthesaurus (127 atoms), domain (21 atoms), related (≤ 6 RT); 5,120 atomics; 17,431 aliases
   (preferred labels + "UF" non-preferred terms) in 7,541 link entries.
2. **Corpus**: training documents = MultiEURLEX train + development splits (60,000 acts), evaluation
   documents = the test split (5,000 acts), each in a seeded order (hash of the CELEX id).
3. **Holdout**: C3 procedure on a 5M-token presample (10% of entries with ≥ 5 occurrences and an
   alias of ≥ 2 subtokens, stratified by log-frequency); see `items/holdout.json`.
4. **Synthetic descriptors** (E5.4 "new EuroVoc descriptors"): invented names placed under an existing
   descriptor (`invented`: two invented stems; `headed`: an invented stem + the broader term's head
   word), with broader / top term / microthesaurus / domain / related edges.
5. **Items**: `eurovoc_probe` (document-level EuroVoc classification, linear probe: the first 600
   characters of an act with its domains, microthesauri and descriptors; probe-train from MultiEURLEX
   train, probe-test from its test split), `defined_term_cloze` (definitions extracted from the acts,
   "‘X’ means …", true definition opening vs three others; 2 paraphrases), `zeroshot_property` and
   `zeroshot_entailment` (broader / microthesaurus / domain of synthetic and held-out descriptors).

Results (cardinality, feasibility verdict, item counts): `runs/v1/report.md`.

## Build

```bash
PYTHONPATH=src python -m vsa_embed.experiments.track_corpus \
  --config experiments/t6-legal-regulatory/t6.yaml --output experiments/t6-legal-regulatory/runs/v1
```
