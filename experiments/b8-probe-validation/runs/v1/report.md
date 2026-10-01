# B8 probe and quantization validation

| Host | LAMBADA acc | WiC prompt | WiC probe | WiC majority | CARD-660 ρ | RW ρ | PPL bf16 | PPL INT8 | PPL INT4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 0.310 | 0.552 | 0.563 | 0.500 | 0.084 | 0.107 | 37.279 | 40.360 | error: ImportError: Requires m |
| HuggingFaceTB/SmolLM2-135M | 0.428 | 0.596 | 0.577 | 0.500 | 0.215 | 0.262 | 26.948 | 27.431 | error: ImportError: Requires m |
| HuggingFaceTB/SmolLM2-360M | 0.532 | 0.610 | 0.582 | 0.500 | 0.168 | 0.208 | 22.456 | 22.775 | error: ImportError: Requires m |

Reference: GPT-2 small LAMBADA-OpenAI accuracy ≈ 0.325 (lm-evaluation-harness).
