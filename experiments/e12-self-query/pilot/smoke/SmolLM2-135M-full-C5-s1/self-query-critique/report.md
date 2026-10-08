# E12 3c — critique inputs, C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train) — SMOKE

Calibrator: {'edges': 20670, 'correct': 0.9855345911949686, 'own_models': [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 19], 'pooled_relations': [4, 12, 15, 16, 17, 18, 20, 21, 22]}. Seen-entry calibration: ECE 0.001, Brier 0.008, AUROC 0.992.

| set | items | none | recall:own | null | evidence | belief (real) correct | belief (null) = null option |
|---|---:|---:|---:|---:|---:|---:|---:|
| twins | 16 | 0.562 | 0.500 | 0.500 | — | 1.000 | 1.000 |
| new | 41 | 0.341 | 0.829 | 0.098 | — | 1.000 | 1.000 |
| heldout | 16 | 0.250 | 0.938 | 0.062 | 0.929 | 1.000 | 1.000 |
