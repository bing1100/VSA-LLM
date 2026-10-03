# Linear-attention fast-path smoke (tiny, cuda:0)

Batch 2 × 512 tokens, bf16 autocast on CUDA, fp32 weights, LoRA r = 8 on every projection (`lora_targets: linear_attention`, 31 adapters, uncovered token mixers: none); median of 5 steps after one warm-up step.

Bound by transformers: fast path yes ({'fla': '0.5.2', 'causal_conv1d': '1.7.0'}).

| path | forward s | forward+backward s | tokens/s | peak GiB | calls |
|---|---:|---:|---:|---:|---|
| reference | 0.0144 | 0.0381 | 26,880 | 0.20 | {'causal_conv1d_fn': {'reference': 18}, 'torch_chunk_gated_delta_rule': {'reference': 18}} |
| fast | 0.0118 | 0.0241 | 42,512 | 0.15 | {'causal_conv1d_fn': {'fast': 18}, 'torch_chunk_gated_delta_rule': {'fast': 18}} |

Speed-up of the fast path (forward+backward): **1.6×** (forward only 1.2×).

Agreement fast vs reference: |Δloss| 1.03e-04, final hidden max |Δ| 6.775e-02 (relative RMS 1.21e-02), LoRA gradient cosine 0.99985 (min per tensor 0.99903), relative gradient difference 1.75e-02.
