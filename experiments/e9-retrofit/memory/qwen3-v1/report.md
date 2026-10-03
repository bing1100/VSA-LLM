# E9 Qwen3 memory and throughput (C5, LoRA r = 64, sequence 1024, bf16 autocast)

GPU: NVIDIA GeForce RTX 3090 (23.6 GiB); budget 22.1 GiB (peak reserved). T5 Qwen3 batches; 3 timed optimizer steps per micro-batch after one warm-up step (fused AdamW, the trainer's parameter groups); channel injection scaled to the host (`scale_to_host`).

| Host | dtype | checkpointing | micro-batch | peak GiB | reserved GiB | tokens/s |
|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B-Base | float32 | False | 1 | 7.6 | 8.1 | 6,802 |
| Qwen3-0.6B-Base | float32 | False | 2 | 11.3 | 12.3 | 7,546 |
| Qwen3-0.6B-Base | float32 | False | 4 | 16.6 | 19.4 | 8,172 |
| Qwen3-0.6B-Base | float32 | False | 8 | 21.3 | 22.6 | OOM |
| Qwen3-1.7B-Base | float32 | False | 1 | 15.4 | 15.6 | 3,930 |
| Qwen3-1.7B-Base | float32 | False | 2 | 20.3 | 21.3 | 4,338 |
| Qwen3-1.7B-Base | float32 | False | 4 | 23.0 | 23.1 | OOM |
| Qwen3-1.7B-Base | float32 | True | 1 | 9.8 | 10.1 | 2,469 |
| Qwen3-1.7B-Base | float32 | True | 2 | 11.8 | 12.9 | 2,718 |
| Qwen3-1.7B-Base | float32 | True | 4 | 14.0 | 16.6 | 2,815 |
| Qwen3-1.7B-Base | float32 | True | 8 | 18.4 | 21.7 | 3,023 |
| Qwen3-4B-Base | float32 | False | 1 | 23.1 | 23.1 | OOM |
| Qwen3-4B-Base | float32 | True | 1 | 19.5 | 20.3 | 1,215 |
| Qwen3-4B-Base | float32 | True | 2 | 21.6 | 23.1 | 1,285 |
| Qwen3-4B-Base | float32 | True | 4 | 21.4 | 23.0 | OOM |
| Qwen3-4B-Base | bfloat16 | False | 1 | 16.8 | 17.4 | 2,349 |
| Qwen3-4B-Base | bfloat16 | False | 2 | 22.6 | 23.0 | OOM |
| Qwen3-4B-Base | bfloat16 | True | 1 | 11.0 | 11.8 | 1,381 |
| Qwen3-4B-Base | bfloat16 | True | 2 | 12.9 | 14.1 | 1,410 |
| Qwen3-4B-Base | bfloat16 | True | 4 | 14.5 | 17.3 | 1,465 |
| Qwen3-4B-Base | bfloat16 | True | 8 | 17.6 | 20.9 | 1,503 |

Evaluation forwards (no gradient, optimizer state resident):

| Host | dtype | checkpointing | eval batch | peak GiB | reserved GiB | ok |
|---|---|---|---:|---:|---:|:-:|
| Qwen3-0.6B-Base | float32 | False | 2 | 5.1 | 6.4 | yes |
| Qwen3-0.6B-Base | float32 | False | 4 | 5.7 | 6.9 | yes |
| Qwen3-0.6B-Base | float32 | False | 8 | 5.7 | 7.7 | yes |
| Qwen3-0.6B-Base | float32 | False | 16 | 6.1 | 9.5 | yes |
| Qwen3-1.7B-Base | float32 | False | 2 | 9.7 | 11.4 | yes |
| Qwen3-1.7B-Base | float32 | False | 4 | 10.3 | 11.9 | yes |
| Qwen3-1.7B-Base | float32 | False | 8 | 10.4 | 13.2 | yes |
| Qwen3-1.7B-Base | float32 | False | 16 | 11.2 | 15.2 | yes |
| Qwen3-1.7B-Base | float32 | True | 2 | 9.7 | 11.1 | yes |
| Qwen3-1.7B-Base | float32 | True | 4 | 10.3 | 11.3 | yes |
| Qwen3-1.7B-Base | float32 | True | 8 | 10.4 | 11.8 | yes |
| Qwen3-1.7B-Base | float32 | True | 16 | 10.5 | 12.8 | yes |
| Qwen3-4B-Base | float32 | True | 2 | 19.2 | 21.1 | yes |
| Qwen3-4B-Base | float32 | True | 4 | 19.8 | 21.2 | yes |
| Qwen3-4B-Base | float32 | True | 8 | 19.9 | 21.9 | yes |
| Qwen3-4B-Base | float32 | True | 16 | 20.0 | 22.3 | yes |
| Qwen3-4B-Base | bfloat16 | False | 2 | 11.4 | 12.8 | yes |
| Qwen3-4B-Base | bfloat16 | False | 4 | 12.0 | 13.4 | yes |
| Qwen3-4B-Base | bfloat16 | False | 8 | 12.1 | 14.7 | yes |
| Qwen3-4B-Base | bfloat16 | False | 16 | 12.8 | 16.9 | yes |
| Qwen3-4B-Base | bfloat16 | True | 2 | 11.4 | 12.6 | yes |
| Qwen3-4B-Base | bfloat16 | True | 4 | 12.0 | 12.6 | yes |
| Qwen3-4B-Base | bfloat16 | True | 8 | 12.1 | 13.1 | yes |
| Qwen3-4B-Base | bfloat16 | True | 16 | 12.1 | 14.3 | yes |

## Recommendation (`recommendation.json`, read by `e9_plan`)

| Host | micro-batch × accumulation | checkpointing | dtype | eval batch | tokens/s | GPU-h per 50M-token run |
|---|---|---|---|---:|---:|---:|
| Qwen3-0.6B-Base | 4 × 16 | False | float32 | 16 | 8,172 | 1.7 |
| Qwen3-1.7B-Base | 2 × 32 | False | float32 | 16 | 4,338 | 3.2 |
| Qwen3-4B-Base | 1 × 64 | True | float32 | 8 | 1,215 | 11.4 |
