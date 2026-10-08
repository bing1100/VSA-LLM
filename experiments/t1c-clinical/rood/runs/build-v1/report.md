# T1c-ROOD LM corpus — clinical ROOD (SNOMED CT + MIMIC-III; ROOD patients and mentions excluded)

Aggregates only (licensed data stay under the data root). Preregistration: `experiments/t1c-clinical/rood/preregistration.md`.

Concept holdout (the ROOD closure): 125 concepts, 123 entries, 268 aliases; sha256 `34fc579eec1e1658654953dc6130cbb63e955be64a84b7ca0d383f2187fdcfc2`.

| corpus | tokens | documents | spans |
|---|---:|---:|---:|
| train | 130,005,024 | 181,832 | 6,372,565 |
| eval-rood | 4,983,507 | 1,039 | 321,266 |
| eval-general | 4,998,786 | 5,000 | 207,825 |
| eval-mimic | 22,352,802 | 39,993 | 1,311,829 |

## Feasibility of the held-out (ROOD) stratum

| split | ℓ_min | occurrences (split) | entries | entries ≥ 5 | verdict (split) | E9 windows: occ. / entries ≥ 5 | verdict | windows needed |
|---|---:|---:|---:|---:|---|---|---|---:|
| eval-rood | 2 | 1,560 | 29 | 16 | feasible (entries counted with ≥ 1 occurrence) | 676 / 12 (2,048) | exploratory-feasible | not reached |
| eval-rood | 3 | 1,540 | 29 | 16 | feasible (entries counted with ≥ 1 occurrence) | 670 / 11 (2,048) | exploratory-feasible | not reached |
| eval-mimic | 2 | 269 | 14 | 9 | infeasible | 18 / 1 (2,048) | infeasible | not reached |
| eval-mimic | 3 | 269 | 14 | 9 | infeasible | 18 / 1 (2,048) | infeasible | not reached |

## Leakage audit

- training spans of held-out entries: 0
- stream counts: `{"train_notes": {"yielded": 122109, "dropped_excluded_subject": 30006, "dropped_by_filter": 392}, "train_general": {"yielded": 61699, "dropped_by_filter": 51}}`
- decoded training corpus: `{"documents": 181832, "documents_mentioning": 0, "mentions": 0}`
- note identity (training-side notes vs decoded training documents): `{"rood_patient_training_side_notes": 372399, "rood_note_texts": 361994, "rood_note_texts_shared_with_other_patients": 1437, "rood_only_note_texts_in_training": 0, "shared_note_texts_in_training": 394, "other_patients_note_texts_in_training": 119815}`
- decoded eval-rood: `{"documents": 1039, "documents_mentioning": 1039, "mentions": 1572}`
- failures: `{}`
