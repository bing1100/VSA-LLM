"""Post-training weight-only quantization with torchao (B8 / E4.4).

`quantize_weight_only(model, bits)` quantizes every `nn.Linear` of the host transformer to INT8 or
INT4 (group-wise) in place. The span channel's dictionary and projector stay in full precision
unless `include_channel=True` (the variant that quantizes them too). GPT-2's `Conv1D` layers are
converted to equivalent `nn.Linear` first so torchao can see them.

Group-wise round-to-nearest helpers (D4.3 / E4.5) for tensors torchao does not quantize (the
channel's dictionary tables, embedding rows, compressed-table parameters):

- `group_fake_quantize(x, bits, group_size)` quantizes and dequantizes along the last dimension in
  groups of `group_size` (None = one group per row): asymmetric min–max codes `0 … 2^b − 1` with a
  scale and a zero point per group, or symmetric codes `−2^(b−1) … 2^(b−1) − 1` with a scale
  `absmax / ((2^b − 1)/2)` only (torchao's INT8 weight-only convention). Scales and zero points are
  rounded to FP16, the format they are stored in. `clip_search=True` shrinks each
  group's range by the factor in `CLIP_GRID` with the smallest squared error (a standard RTN
  refinement); `straight_through=True` passes gradients to `x` unchanged (quantization-aware
  finetuning). `bits ≥ 32` is the identity, `bits = 16` rounds to FP16.
- `group_quantized_bytes(rows, columns, bits, group_size)` is the exact storage of that format:
  packed codes `⌈rows·columns·bits/8⌉` plus 2 bytes per group for the scale and (asymmetric) 2 for
  the zero point; `bits = 16` / `32` are 2 / 4 bytes per value.
- `index_bytes(bound, count)` is the exact storage of packed integer indices (frame schedules,
  row → entry maps).
- `tensor_storage_bytes(t)` measures a torchao-quantized weight by its inner tensors (codes, scales,
  zero points), so INT8/INT4 bytes are those of the actual layout.
- `auto_group_size(model)` is the largest INT4 group size (128 / 64 / 32) dividing every linear
  layer's input width, so no layer is skipped for its shape.
- `merge_lora(model)` folds LoRA adapters into their base weights before quantization.
- `fake_quantize_module_(module, bits, group_size)` quantizes every floating parameter or buffer
  with ≥ 2 dimensions of a module in place (variant B of D4.3: the channel's dictionary, relation
  parameters, composer and projector) and returns the bytes of each tensor.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn


def conv1d_to_linear(model: nn.Module) -> int:
    """Replace HF GPT-2 `Conv1D` (weight in × out) by `nn.Linear` (weight out × in); returns count."""
    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if type(child).__name__ == "Conv1D":
                linear = nn.Linear(child.weight.shape[0], child.weight.shape[1], bias=child.bias is not None,
                                   device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    linear.weight.copy_(child.weight.T)
                    if child.bias is not None:
                        linear.bias.copy_(child.bias)
                setattr(module, name, linear)
                replaced += 1
    return replaced


def quantize_weight_only(model: nn.Module, bits: int, *, group_size: int = 128, include_channel: bool = False,
                         channel_prefix: str = "channel") -> dict[str, int]:
    """Quantize linear layers in place; returns counts. INT4 needs a CUDA device and bf16 weights."""
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    if bits not in (8, 4):
        raise ValueError("bits must be 8 or 4")
    converted = conv1d_to_linear(model)
    # INT4 uses the tile-packed tinygemm layout: the default "plain" layout needs the mslk package,
    # which has no usable release.
    config = Int8WeightOnlyConfig() if bits == 8 else Int4WeightOnlyConfig(group_size=group_size,
                                                                          int4_packing_format="tile_packed_to_4d")
    embedding = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    tied = {id(embedding.weight)} if embedding is not None else set()

    def keep(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        # The output head (often tied to the input embedding) stays in full precision, as in
        # standard GPTQ/AWQ practice; quantizing a tied head also corrupts the embedding lookup.
        if fqn.endswith("lm_head") or id(module.weight) in tied:
            return False
        if not include_channel and (fqn.startswith(channel_prefix) or ".channel." in fqn):
            return False
        if bits == 4 and module.in_features % group_size:
            return False
        return True

    quantize_(model, config, filter_fn=keep)
    return {"conv1d_converted": converted, "bits": bits}


# -- group-wise round-to-nearest quantization (channel tensors, embedding tables) ---------------

CLIP_GRID = tuple(1.0 - 0.05 * i for i in range(11))     # range shrink factors 1.0 … 0.5


def _groups(x: Tensor, group_size: int | None) -> tuple[Tensor, int]:
    """`x` as (rows, groups, group) along its last dimension; the last group is padded by
    repeating its final value, which leaves its min/max (and so its codes) unchanged."""
    columns = x.shape[-1]
    size = columns if not group_size or group_size >= columns else int(group_size)
    flat = x.reshape(-1, columns)
    padded = -(-columns // size) * size
    if padded != columns:
        flat = torch.cat([flat, flat[:, -1:].expand(-1, padded - columns)], 1)
    return flat.reshape(flat.shape[0], padded // size, size), columns


def _dequantized(groups: Tensor, bits: int, symmetric: bool, shrink: float) -> Tensor:
    if symmetric:
        # torchao's convention: scale = absmax / ((2^b − 1) / 2), codes in [−2^(b−1), 2^(b−1) − 1].
        scale = (groups.abs().amax(-1, keepdim=True) * shrink / ((2 ** bits - 1) / 2)).half().float()
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (groups / safe).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1) * scale
    lo, hi = groups.amin(-1, keepdim=True), groups.amax(-1, keepdim=True)
    middle, half = (hi + lo) / 2, (hi - lo) / 2 * shrink
    levels = 2 ** bits - 1
    scale, zero = (2 * half / levels).half().float(), (middle - half).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return ((groups - zero) / safe).round().clamp(0, levels) * scale + zero


def group_fake_quantize(x: Tensor, bits: int, group_size: int | None = None, *, symmetric: bool = False,
                        clip_search: bool = False, straight_through: bool = False) -> Tensor:
    """Quantize–dequantize `x` group-wise along its last dimension (see the module docstring)."""
    if bits >= 32:
        return x
    if bits < 1:
        raise ValueError("bits must be ≥ 1")
    with torch.no_grad():
        if bits == 16:
            rounded = x.detach().half().to(x.dtype)
        else:
            groups, columns = _groups(x.detach().float(), group_size)
            best = _dequantized(groups, bits, symmetric, 1.0)
            if clip_search:
                error = (best - groups).square().sum(-1, keepdim=True)
                for shrink in CLIP_GRID[1:]:
                    candidate = _dequantized(groups, bits, symmetric, shrink)
                    candidate_error = (candidate - groups).square().sum(-1, keepdim=True)
                    better = candidate_error < error
                    best = torch.where(better, candidate, best)
                    error = torch.where(better, candidate_error, error)
            rounded = best.reshape(best.shape[0], -1)[:, :columns].reshape(x.shape).to(x.dtype)
    return x + (rounded - x).detach() if straight_through else rounded


def group_quantized_bytes(rows: int, columns: int, bits: int, group_size: int | None = None, *,
                          symmetric: bool = False) -> int:
    """Exact bytes of a (rows × columns) tensor in the format of `group_fake_quantize`."""
    if bits >= 32:
        return 4 * rows * columns
    if bits == 16:
        return 2 * rows * columns
    size = columns if not group_size or group_size >= columns else int(group_size)
    groups = rows * -(-columns // size)
    return math.ceil(rows * columns * bits / 8) + groups * (2 if symmetric else 4)


def index_bytes(bound: int, count: int) -> int:
    """Bytes of `count` packed unsigned indices in `[0, bound)`: `⌈count · ⌈log2 bound⌉ / 8⌉`
    (at least one bit per index)."""
    return math.ceil(count * max(1, math.ceil(math.log2(max(int(bound), 2)))) / 8)


def is_quantized(tensor: Tensor) -> bool:
    """A torchao tensor subclass (codes plus scales), not a plain tensor."""
    return type(tensor) not in (Tensor, nn.Parameter) and hasattr(tensor, "__tensor_flatten__")


def tensor_storage_bytes(tensor: Tensor, *, float_bytes: int | None = None) -> int:
    """Bytes of a tensor as stored: a torchao-quantized weight is measured by its inner tensors
    (codes, scales, zero points); a plain floating tensor counts `float_bytes` per value (its own
    element size when None)."""
    if is_quantized(tensor):
        names, _ = tensor.__tensor_flatten__()
        return sum(tensor_storage_bytes(getattr(tensor, name)) for name in names)
    size = float_bytes if float_bytes and tensor.is_floating_point() else tensor.element_size()
    return tensor.numel() * size


def auto_group_size(model: nn.Module, candidates: Iterable[int] = (128, 64, 32)) -> int:
    """Largest INT4 group size dividing the input width of every linear layer (GPT-2 `Conv1D`
    included), so that no layer is skipped for its shape (SmolLM2-135M's 576 needs 64)."""
    widths = {m.in_features for m in model.modules() if isinstance(m, nn.Linear)}
    widths |= {m.weight.shape[0] for m in model.modules() if type(m).__name__ == "Conv1D"}
    candidates = list(candidates)
    for size in candidates:
        if all(width % size == 0 for width in widths):
            return size
    return candidates[-1]


def merge_lora(model: nn.Module) -> int:
    """Fold every `LoRALinear` into its base weight (`W + (α/r)·B A`) and unwrap it; returns the count."""
    from ..integrations.transformers import LoRALinear
    merged = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                with torch.no_grad():
                    update = child.scale * (child.lora_b.float() @ child.lora_a.float())     # out × in
                    base.weight.add_((update.T if child.transposed else update).to(base.weight.dtype))
                setattr(module, name, base)
                merged += 1
    return merged


def fake_quantize_module_(module: nn.Module, bits: int, group_size: int | None = None, *,
                          symmetric: bool = False) -> dict[str, int]:
    """Quantize every floating parameter/buffer with ≥ 2 dimensions in place (round-to-nearest, no
    clip search, like torchao's weight-only schemes); returns `{name: bytes}` for them."""
    sizes: dict[str, int] = {}
    seen: set[int] = set()
    for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
        if id(tensor) in seen or not tensor.is_floating_point() or tensor.ndim < 2:
            continue
        seen.add(id(tensor))
        with torch.no_grad():
            tensor.copy_(group_fake_quantize(tensor.detach().float(), bits, group_size, symmetric=symmetric).to(tensor.dtype))
        sizes[name] = group_quantized_bytes(tensor.numel() // tensor.shape[-1], tensor.shape[-1], bits, group_size,
                                            symmetric=symmetric)
    return sizes
