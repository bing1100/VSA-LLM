# E12 3b — arm S on C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train) — SMOKE

Training: 2 steps (3 planned), 3456 tokens (1550 in the loss), 143 tokens/s; loss 2.859288454055786 → 2.4324193000793457. Sequences per epoch [16, 16, 16].

Recall in the T traces: {'gold': 14, 'neither option': 2}.

| test | value |
|---|---:|
| twins none choice | 0.417 |
| twins none cloze | 0.333 |
| twins recall:own choice | 0.917 |
| twins recall:own cloze | 1.000 |
| new words none | 0.200 |
| two-hop none | 0.158 |
| two-hop recall:own | 0.921 |
| twins question format (no tool) | 0.667 |
