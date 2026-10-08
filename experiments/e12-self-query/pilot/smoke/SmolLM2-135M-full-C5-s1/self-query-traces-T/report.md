# E12 3b — arm T on C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train) — SMOKE

Training: 2 steps (3 planned), 4144 tokens (1550 in the loss), 23 tokens/s; loss 2.859288454055786 → 2.3872450590133667. Sequences per epoch [16, 16, 16].

Recall in the T traces: {'gold': 14, 'neither option': 2}.

| test | value |
|---|---:|
| twins none choice | 0.333 |
| twins none cloze | 0.333 |
| twins recall:own choice | 0.917 |
| twins recall:own cloze | 1.000 |
| new words none | 0.200 |
| two-hop none | 0.158 |
| two-hop recall:own | 0.947 |
| twins question format (no tool) | 1.000 |
| agent: format ok | 0.750 |
| agent: twin contrast | 1.000 |
