**SMOKE TEST — not a result.**

# E11 read-to-learn — t4 heldout — C6d seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Item set `experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1` (8 terms, 28 items); styles ['chebi'] (primary `chebi`); channel composes: False. No weight is updated except inside the gradient baseline, whose weights are restored after every term.

## Item tests (no definition in context unless the condition says `context`)

| condition | property | paraphrase | entailment |
|---|---:|---:|---:|
| chebi|encoder | 0.5000 | 0.5000 | 0.8571 |
| chebi|none | 0.4762 | 0.3571 | 0.8571 |
| chebi|oracle | 0.4762 | 0.4286 | 0.7857 |

| style | a − b | test | difference [95% CI] | n | p (Holm within pair) |
|---|---|---|---|---:|---:|
| chebi | oracle − none | property | +0.0000 [+0.0000, +0.0000] | 14 | 1.0000 |
| chebi | oracle − none | paraphrase | +0.0714 [+0.0000, +0.2143] | 14 | 1.0000 |
| chebi | oracle − none | entailment | -0.0714 [-0.2143, +0.0000] | 14 | 1.0000 |
| chebi | encoder − none | property | +0.0238 [+0.0000, +0.0714] | 14 | 1.0000 |
| chebi | encoder − none | paraphrase | +0.1429 [+0.0000, +0.3571] | 14 | 0.6269 |
| chebi | encoder − none | entailment | +0.0000 [+0.0000, +0.0000] | 14 | 1.0000 |
| chebi | encoder − oracle | property | +0.0238 [+0.0000, +0.0714] | 14 | 1.0000 |
| chebi | encoder − oracle | paraphrase | +0.0714 [+0.0000, +0.2143] | 14 | 1.0000 |
| chebi | encoder − oracle | entailment | +0.0714 [+0.0000, +0.2143] | 14 | 1.0000 |

## Loss after the read terms in the evaluation text

5 windows, 7 occurrences (0 inside the term's own definition document, reported separately).

| condition | loss (other documents) | targets | loss (own document) | unlinked-token loss |
|---|---:|---:|---:|---:|
| chebi|none | 1.9485 | 56 | — | 2.1210 |
| chebi|encoder | 1.9610 | 56 | — | 2.1214 |
| chebi|oracle | 1.9637 | 56 | — | 2.1214 |

| condition − reference | part | relative change [95% CI] | Δ nats/token | p |
|---|---|---|---:|---:|
| chebi|encoder − none | other | +0.64% [-0.27, +1.38] | +0.0125 | 0.1436 |
| chebi|oracle − none | other | +0.78% [+0.54, +1.03] | +0.0152 | 0.0103 |
| chebi|encoder − oracle | other | -0.14% [-1.17, +0.34] | -0.0027 | 0.5436 |

## Definition-encoder route (C6d; §16.2)

Frozen host `HuggingFaceTB/SmolLM2-360M` (float32), statistics from 512 reference entries (sample of 512, the table's stored verbalizations); re-encoded rows against the stored table: cosine mean 0.9985, min 0.9945.
- chebi: 8 terms, one pass of 65.1 tokens per term (≈ 4.71e+10 FLOPs), 0 context tokens per use.

Timings (s): `{"readers": 0.0, "encoder": 817.1, "encoder_items": 5309.7, "windows": 131.8}`; total 6280s.