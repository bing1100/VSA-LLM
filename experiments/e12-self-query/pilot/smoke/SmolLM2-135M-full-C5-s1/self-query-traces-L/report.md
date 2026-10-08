# E12 3b — arm L on C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train) — SMOKE

Training: 2 steps (3 planned), 4160 tokens (4128 in the loss), 191 tokens/s; loss 1.80660879611969 → 1.279495358467102. Sequences per epoch [16, 16, 16].

Recall in the T traces: {'gold': 14, 'neither option': 2}.

| test | value |
|---|---:|
| twins none choice | 0.333 |
| twins none cloze | 0.333 |
| twins recall:own choice | 1.000 |
| twins recall:own cloze | 1.000 |
| new words none | 0.200 |
| two-hop none | 0.132 |
| two-hop recall:own | 0.921 |
| twins question format (no tool) | 1.000 |
