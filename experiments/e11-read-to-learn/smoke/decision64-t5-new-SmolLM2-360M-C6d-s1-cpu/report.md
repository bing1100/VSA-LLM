**SMOKE TEST — not a result.**

# E11 read-to-learn — t5 new — C6d seed 1 (HuggingFaceTB/SmolLM2-360M/train)

Item set `experiments/e11-read-to-learn/items/t5-new-smollm2-v1` (5 terms, 58 items); styles ['prose'] (primary `prose`); channel composes: False. No weight is updated except inside the gradient baseline, whose weights are restored after every term.

## Item tests (no definition in context unless the condition says `context`)

| condition | property | property_new | entailment | paraphrase | statement_accuracy | statement_loss |
|---|---:|---:|---:|---:|---:|---:|
| prose|encoder | 0.2292 | 0.2632 | 0.6000 | 0.4167 | 0.1667 | 4.9707 |
| prose|none | 0.2292 | 0.2368 | 0.7000 | 0.5000 | 0.1667 | 5.0161 |
| prose|oracle | 0.1875 | 0.2105 | 0.6000 | 0.3750 | 0.1667 | 4.9938 |

| style | a − b | test | difference [95% CI] | n | p (Holm within pair) |
|---|---|---|---|---:|---:|
| prose | oracle − none | property | -0.0417 [-0.1458, +0.0417] | 24 | 1.0000 |
| prose | oracle − none | property_new | -0.0263 [-0.1316, +0.0789] | 19 | 1.0000 |
| prose | oracle − none | entailment | -0.1000 [-0.3000, +0.0000] | 10 | 1.0000 |
| prose | oracle − none | paraphrase | -0.1250 [-0.2917, +0.0417] | 24 | 1.0000 |
| prose | oracle − none | statement_accuracy | +0.0000 [+0.0000, +0.0000] | 24 | 1.0000 |
| prose | oracle − none | statement_loss | -0.0223 [-0.1434, +0.0959] | 24 | 1.0000 |
| prose | encoder − none | property | +0.0000 [-0.0625, +0.0625] | 24 | 1.0000 |
| prose | encoder − none | property_new | +0.0263 [+0.0000, +0.0789] | 19 | 1.0000 |
| prose | encoder − none | entailment | -0.1000 [-0.3000, +0.0000] | 10 | 1.0000 |
| prose | encoder − none | paraphrase | -0.0833 [-0.2083, +0.0000] | 24 | 1.0000 |
| prose | encoder − none | statement_accuracy | +0.0000 [+0.0000, +0.0000] | 24 | 1.0000 |
| prose | encoder − none | statement_loss | -0.0454 [-0.1254, +0.0219] | 24 | 1.0000 |
| prose | encoder − oracle | property | +0.0417 [-0.0417, +0.1250] | 24 | 1.0000 |
| prose | encoder − oracle | property_new | +0.0526 [-0.0263, +0.1316] | 19 | 1.0000 |
| prose | encoder − oracle | entailment | +0.0000 [-0.2025, +0.3000] | 10 | 1.0000 |
| prose | encoder − oracle | paraphrase | +0.0417 [-0.0844, +0.1667] | 24 | 1.0000 |
| prose | encoder − oracle | statement_accuracy | +0.0000 [+0.0000, +0.0000] | 24 | 1.0000 |
| prose | encoder − oracle | statement_loss | -0.0230 [-0.0773, +0.0383] | 24 | 1.0000 |

## Definition-encoder route (C6d; §16.2)

Frozen host `HuggingFaceTB/SmolLM2-360M` (float32), statistics from 256 reference entries (sample of 256, the table's stored verbalizations); re-encoded rows against the stored table: cosine mean 0.9968, min 0.9943.
- prose: 5 terms, one pass of 76.6 tokens per term (≈ 5.54e+10 FLOPs), 0 context tokens per use.

Timings (s): `{"readers": 0.0, "encoder": 66.5, "encoder_items": 109.8}`; total 177s.