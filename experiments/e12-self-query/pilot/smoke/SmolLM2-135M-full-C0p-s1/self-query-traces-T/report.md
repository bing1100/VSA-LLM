# E12 3b — arm T on C0p seed 1 (HuggingFaceTB/SmolLM2-135M/train) — SMOKE

Training: 1 steps (3 planned), 2072 tokens (775 in the loss), 6 tokens/s; loss 2.7693792581558228 → 2.7693792581558228. Sequences per epoch [16, 16, 16].

Recall in the T traces: {'gold': 14, 'neither option': 2}.

| test | value |
|---|---:|
| twins none choice | 0.583 |
| twins none cloze | 0.333 |
| twins recall:own choice | 0.917 |
| twins recall:own cloze | 1.000 |
| new words none | 0.167 |
| two-hop none | 0.000 |
| two-hop recall:own | 0.812 |
| twins question format (no tool) | 0.667 |
| agent: format ok | 0.500 |
| agent: twin contrast | 1.000 |
