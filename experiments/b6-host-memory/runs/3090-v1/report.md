# Host memory and throughput on the local GPU (B6)

Sequence length 1024, bf16 autocast, fused AdamW, attentive span channel (20k entries, 8,192 atomics, d=256).

| Host | Mode | Checkpointing | Micro-batch | Peak GiB | Tokens/s |
|---|---|---|---:|---:|---:|
| scratch-50M | train | False | 8 | 5.8 | 66309 |
| scratch-50M | train | False | 16 | 10.2 | 69271 |
| scratch-50M | train | False | 32 | 18.8 | 71220 |
| scratch-125M | train | False | 8 | 10.2 | 34038 |
| scratch-125M | train | False | 16 | 18.0 | 35789 |
| scratch-125M | train | True | 16 | 6.1 | 28505 |
| scratch-125M | train | True | 32 | 9.9 | 28784 |
| scratch-350M | train | True | 4 | 6.5 | 10091 |
| scratch-350M | train | True | 8 | 7.7 | 10548 |
| scratch-350M | train | True | 16 | 10.1 | 11024 |
| HuggingFaceTB/SmolLM2-135M | frozen | False | 8 | 7.1 | 35315 |
| HuggingFaceTB/SmolLM2-135M | frozen | False | 16 | 12.8 | 36430 |
| HuggingFaceTB/SmolLM2-135M | lora | False | 8 | 9.5 | 26450 |
| HuggingFaceTB/SmolLM2-360M | frozen | False | 4 | 7.2 | 17272 |
| HuggingFaceTB/SmolLM2-360M | frozen | False | 8 | 11.5 | 18564 |
| HuggingFaceTB/SmolLM2-360M | frozen | True | 16 | OOM | — |
| Qwen/Qwen2.5-0.5B | frozen | False | 2 | 8.2 | 14410 |
| Qwen/Qwen2.5-0.5B | frozen | False | 4 | 11.4 | 15995 |
| Qwen/Qwen2.5-0.5B | frozen | True | 8 | 18.9 | 15028 |
