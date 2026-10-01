# B8 probe and quantization validation

| Host | LAMBADA acc | WiC prompt | WiC probe | WiC majority | CARD-660 ρ | RW ρ | PPL bf16 | PPL INT8 | PPL INT4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | — | — | — | — | — | — | 37.279 | 37.270 | 41.685 |
| HuggingFaceTB/SmolLM2-135M | — | — | — | — | — | — | 26.948 | 27.354 | 36.971 |
| HuggingFaceTB/SmolLM2-360M | — | — | — | — | — | — | 22.456 | 22.679 | 27.206 |

Reference: GPT-2 small LAMBADA-OpenAI accuracy ≈ 0.325 (lm-evaluation-harness).
