# Qwen3.5 environment for E9 (WP-Qwen35, 2026-10-03)

E9 on **Qwen3.5-2B-Base** and **Qwen3.5-0.8B-Base** needs transformers ≥ 5 (`qwen3_5`; the pinned `vsa-repro` env has
4.54) and, for usable speed, the Gated DeltaNet kernels. Everything for these hosts runs in a separate environment;
the pinned env is not changed. Lock file: [`qwen35-requirements.txt`](qwen35-requirements.txt).

## Environment

- **Interpreter:** `~/venvs/vsa-qwen35/bin/python` (`cpt_plan.QWEN35_PYTHON`), a `venv --system-site-packages` over
  `~/anaconda3/envs/vsa-repro` (Python 3.12.4). Torch 2.11.0+cu128, Triton 3.6.0 and torchao 0.18.0 are inherited
  unchanged, so the CUDA runtime (12.8) and the INT8/INT4 PTQ path are those of the pinned env.
- **Installed in the venv only** (`pip install --no-deps`, nothing inherited upgraded):

| Package | Version | Source | Role |
|---|---|---|---|
| transformers | 5.18.0 | PyPI | `qwen3_5` (text-only `Qwen3_5ForCausalLM`), Qwen3.5 tokenizer |
| tokenizers / huggingface_hub / safetensors | 0.23.2 / 1.33.0 / 0.8.0 | PyPI | required by transformers 5.18 |
| flash-linear-attention + fla-core | 0.5.2 | PyPI (pure Python + Triton) | chunked / fused-recurrent gated delta rule |
| einops | 0.8.1 | PyPI | fla-core dependency |
| causal-conv1d | 1.7.0 | **local build** from the PyPI sdist (below) | depthwise causal conv of the Gated DeltaNet |
| ninja | 1.13.0 | PyPI | build tool (causal-conv1d) |
| annotated-doc, anyio, h11, httpcore, httpx, markdown-it-py, mdurl, rich, shellingham, typer | see lock file | PyPI | huggingface_hub 1.x dependencies |

- **Not installed:** `kernels` (Hugging Face hub kernels). Without it transformers binds the installed packages
  directly and never downloads kernel code.

## causal-conv1d build (CUDA 12.8, sm_86)

The 1.7.0 release has prebuilt wheels up to torch 2.10 only, so it was compiled for this torch:

1. **Toolkit:** a separate conda prefix `~/venvs/cuda-12.8` (conda-forge, `cuda-version=12.8`: cuda-nvcc-tools /
   cuda-nvcc-impl 12.8.93, cuda-cudart-dev 12.8.90, cuda-cccl 12.8.90, libcublas-dev 12.8.5.5, libcusparse-dev 12.5.8.93,
   libcusolver-dev 11.7.3.90, libcurand-dev 10.3.9.90, cuda-nvtx-dev, cuda-profiler-api), used at build time only.
   `~/venvs/cuda-12.8/cuda_home` is a symlink layout (`bin`, `include`, `lib64`, `nvvm`, `targets`) used as `CUDA_HOME`.
   Host compiler: the system gcc 13.3.
2. **Source:** `causal_conv1d-1.7.0.tar.gz` from PyPI, sha256
   `3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b` (kept in `~/venvs/wheels/`).
3. **Patch** ([`patch_causal_conv1d.py`](patch_causal_conv1d.py)): gencode `sm_86` only (the sdist lists 75/80/87/90/100/120)
   and `nvcc --threads 1`.
4. **Build** (no GPU needed; 4 compile jobs; ≈ 2.5 min):

```bash
CUDA_HOME=~/venvs/cuda-12.8/cuda_home PATH=~/venvs/cuda-12.8/cuda_home/bin:~/venvs/vsa-qwen35/bin:/usr/bin:/bin \
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6 \
  ~/venvs/vsa-qwen35/bin/pip wheel --no-deps --no-build-isolation -w ~/venvs/wheels <unpacked, patched sdist>
~/venvs/vsa-qwen35/bin/pip install --no-deps ~/venvs/wheels/causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl
```

   Wheel sha256 `25dd9d29548a32d2b805476eab09a8bdbfd326e38f95eb8013eda41ba8fcd08f` (sm_86 code only).

## How the fast path is used and checked

- transformers binds `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` to
  `fla.ops.gated_delta_rule` and `causal_conv1d_fn` / `causal_conv1d_update` to `causal_conv1d` at import, whatever
  the device. With the kernels installed a CPU forward then fails (Triton cannot read CPU tensors).
  `vsa_embed.integrations.linear_attention.install_device_dispatch` (called by `training.lm.build_model` for
  linear-attention hosts) routes CUDA tensors to the kernels and CPU tensors to the PyTorch reference, and counts
  the calls of each implementation.
- Every Qwen3.5 run prints and records `linear_attention_kernels` in its manifest: what transformers bound
  (`fast_path_bound`), package versions, the interpreter, and the calls per implementation. The memory probe
  (`e9_memory`) records `kernel_calls` per training row and times one micro-batch-1 step on the reference path
  (`train-reference`) for the full-size speed comparison.
- `linear_attention_smoke` runs the same forward/backward on both paths (random 4-layer Qwen3.5 text model: 3 Gated
  DeltaNet + 1 full-attention layer; bf16 autocast, fp32 weights, LoRA on every projection), on the RTX 3090 beside a
  running queue job (allocator capped at 1–1.8 GiB). Run folders in this directory:

| run | shapes | path | forward s | fwd+bwd s | tokens/s | peak GiB | calls (per path) |
|---|---|---|---:|---:|---:|---:|---|
| `kernel-smoke-shapes-v1` | the 0.8B host's layers (width 1024, 16 × 128 linear heads, 8 × 256 attention heads), 1 × 1024 | reference | 0.0371 | 0.1109 | 9,233 | 1.37 | conv 18, delta rule 18 |
| | | **fast** | 0.0229 | 0.0487 | 21,036 | 1.00 | conv 18, delta rule 18 |
| `kernel-smoke-tiny-v1` | width 256, 4 × 64 heads, 2 × 512 | reference | 0.0144 | 0.0381 | 26,880 | 0.20 | conv 18, delta rule 18 |
| | | **fast** | 0.0118 | 0.0241 | 42,512 | 0.15 | conv 18, delta rule 18 |

  Speed-up of forward+backward 2.3× at the real layer shapes (1.6× at the toy size, launch-bound); the memory probe
  measures it for the full hosts. Agreement fast vs reference (real shapes): |Δloss| 5.5e-4, LoRA gradient cosine 0.99985
  (min per tensor 0.9985), final hidden states 1.9% relative RMS apart — the reference computes the delta rule in fp32,
  fla in bf16 with fp32 accumulation; training and every evaluation of these hosts run on CUDA, so on one path. The first
  fast step compiles and autotunes the Triton kernels (≈ 55 s, cached in `~/.triton`). On CPU only the reference path
  exists.
- **End-to-end (CPU, real Qwen3.5-0.8B-Base, reference path):** the `e9_plan` C5 config (T5 Qwen3.5 corpus, LoRA r = 64,
  `scale_to_host`) for 2 optimizer steps of 1 × 1024 tokens through `training.lm.train` — the corpus fingerprint check
  passes; 186 adapters (8 in each of the 18 Gated DeltaNet layers, 7 in each of the 6 full-attention layers; the
  default targets would give 96 and leave all 18 linear-attention mixers frozen); host row norm 0.631, channel scale 0.164
  (unscaled injection 1.8× an embedding row); loss on 2 windows 2.679 → 2.484 — then `channel_probes.load_run` with
  INT8 (186 adapters merged, 186 linear layers quantized, none skipped) and a scoring forward (80 s wall, peak RSS 19 GB).
- **INT8 / INT4 PTQ** (`e4_quant`'s path, `channel_probes.quantize_model`) on a tiny Qwen3.5 with 16-row `in_proj_a/b`
  (as the real hosts): all 31 linear layers quantized, none skipped (group size 128); bf16 loss 8.3659 → INT8 8.3655,
  INT4 (tile-packed tinygemm) 8.3714; the quantized model runs on the fast kernels.
- **Text-only loading** of the real hosts: `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` (no missing keys; vision tower
  and MTP head ignored); `hidden_states[-1]` equals the post-norm `last_hidden_state` (the probes' logits path);
  `use_cache` is switched off for these hosts (no recurrent/KV cache objects in training or scoring forwards).

## Compatibility notes

- `from_pretrained(dtype=…)` under transformers ≥ 5, `torch_dtype=…` under 4.x (`training.lm.dtype_kwargs`).
- **Tokenizer fingerprints depend on the transformers/tokenizers version**: under 5.18 SmolLM2's is `7e0b8f35…` (4.54:
  `2225e8fb…`) and Qwen3's `41e00ecc…` (4.54: `a63080b4…`); GPT-2's is the same. A corpus records the fingerprint of
  the environment that built it and the trainer checks it in the environment that trains, so **Qwen3.5 corpora are
  built and used in the Qwen3.5 environment, and SmolLM2/Qwen3 corpora in the pinned one** (Qwen3.5: `4d765ac5…`).
- Job interpreters: `cpt_plan.HOSTS[...]["python"]` (Qwen3.5 hosts) → `e9_plan`/`cpt_plan` queue those hosts' jobs with
  the venv interpreter and every other host's with the pinned one (`cpt_plan.pinned_python`, also when planned from
  the venv).
