# E12 3c — critique inputs, C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train) — SMOKE

Calibrator: {'edges': 20670, 'correct': 0.9855345911949686, 'own_models': [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 19], 'pooled_relations': [4, 12, 15, 16, 17, 18, 20, 21, 22]}. Seen-entry calibration: ECE 0.001, Brier 0.008, AUROC 0.992.

| set | items | none | recall:own | null | evidence | symbolic | definition | belief (real) correct | belief (null) = null option |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| twins | 8 | 0.500 | 0.500 | 0.500 | — | — | — | 1.000 | 1.000 |
| new | 19 | 0.211 | 0.789 | 0.053 | — | 0.789 | 0.842 | 1.000 | 1.000 |
| heldout | 9 | 0.333 | 1.000 | 0.000 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 |

Prompt tokens each context adds per item (mean; added once to each of the item's prompts; the store's rule loop reads the decode without the host: none):

| set | recall:own | null | evidence | symbolic | definition |
|---|---:|---:|---:|---:|---:|
| twins | 228.0 | 228.0 | — | — | — |
| new | 172.8 | 173.6 | — | 172.8 | 109.9 |
| heldout | 156.3 | 157.0 | 17.5 | 157.7 | 98.7 |
