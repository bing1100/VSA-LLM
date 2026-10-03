# Linear-attention fast-path smoke (tiny, cuda:0)

Batch 1 × 1024 tokens, bf16 autocast on CUDA, fp32 weights, LoRA r = 8 on every projection (`lora_targets: linear_attention`, 31 adapters, uncovered token mixers: none); median of 5 steps after one warm-up step.

Bound by transformers: fast path yes ({'fla': '0.5.2', 'causal_conv1d': '1.7.0'}).

| path | forward s | forward+backward s | tokens/s | peak GiB | calls |
|---|---:|---:|---:|---:|---|
| reference | 0.0371 | 0.1109 | 9,233 | 1.37 | {'causal_conv1d_fn': {'reference': 18}, 'torch_chunk_gated_delta_rule': {'reference': 18}} |
| fast | 0.0229 | 0.0487 | 21,036 | 1.00 | {'causal_conv1d_fn': {'fast': 18}, 'torch_chunk_gated_delta_rule': {'fast': 18}} |

Speed-up of the fast path (forward+backward): **2.3×** (forward only 1.6×).

Agreement fast vs reference: |Δloss| 5.53e-04, final hidden max |Δ| 1.129e-01 (relative RMS 1.94e-02), LoRA gradient cosine 0.99985 (min per tensor 0.99850), relative gradient difference 2.49e-02.
