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
   characters of an act with its domains, microthesauri and descriptors; 2,000 probe-train acts from
   MultiEURLEX train, 1,000 probe-test acts from its test split), `defined_term_cloze` (definitions extracted from the acts,
   "‘X’ means …", true definition opening vs three others; 2 paraphrases), `zeroshot_property` and
   `zeroshot_entailment` (broader / microthesaurus / domain of synthetic and held-out descriptors).

## Results (`runs/v1/report.md`)

- Training stream 100.0M SmolLM2 tokens (99.6M EUR-Lex + 0.4M FineWeb-Edu fill); domain evaluation 3.0M
  tokens of the test acts; general-text locality set 2.2M tokens.
- Holdout: 10% of the 1,847 eligible entries → 186 descriptors, 189 entries (sha256 `8148fa06fe76929f…`);
  300 synthetic descriptors (sha256 `6f9243110d1c27ce…`).
- Cardinality (SmolLM2, 500 evaluation acts): ℓ_min = 2 links 1,602 distinct entries, 27,576 spans,
  6.2% of tokens, 10.4 distinct entries per 1,024 tokens.
- **Feasibility: infeasible under the fixed criterion** — the seen-rare stratum is nearly empty
  (18 entries / 31 spans in 256 windows, 65 / 114 in 1,024 at ℓ_min = 2): 100M tokens of EU law link
  almost every EuroVoc term that the test acts use ≥ 10 times. The held-out stratum and locality
  alone are powered at ℓ_min = 2 with the default 256 windows (59 held-out entries, 1,314 spans).
- Items: `eurovoc_probe` 3,000 (1,014 with a held-out descriptor), `defined_term_cloze` 4,000 rows
  (2,000 terms × 2 paraphrases; 96 held-out-linked, 1,494 training-linked, 2,410 unlinked),
  `zeroshot_property` 4,308, `zeroshot_entailment` 2,872.

## Build

```bash
PYTHONPATH=src python -m vsa_embed.experiments.track_corpus \
  --config experiments/t6-legal-regulatory/t6.yaml --output experiments/t6-legal-regulatory/runs/v1
```

≈ 7 min from scratch with 3 tokenizer workers, 2.3 GB peak RSS. Outputs in `~/data/vsa-llm/tracks/t6-legal/v1/`.
