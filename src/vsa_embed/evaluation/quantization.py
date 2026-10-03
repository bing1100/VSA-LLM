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

Simulated weight-only schemes beyond torchao RTN (WP-PQ1 claim-B controls; `simulate_weight_only_`:
`rtn`, `hqq`, `nf4`, `gptq`, `awq`) and the input-embedding quantization of the `-emb` variants
(`quantize_input_embedding_`) are documented at their section below; `experiments.e9_rescore` uses them.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Sequence

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


# -- simulated weight-only schemes beyond torchao RTN (WP-PQ1, claim-B controls) -------------------------------------
#
# Every scheme below replaces a host linear weight `W` by its dequantized value (bf16 when it runs), which is exactly
# what a weight-only kernel computes with (it dequantizes on the fly); only speed and memory are not simulated, and
# bytes are reported nominally (`scheme_bits`). All schemes skip the output head and the input embedding (kept 16-bit,
# as for torchao's `int4-A`), unless `quantize_input_embedding_` is applied as well (`-emb` variants).
#
# - `rtn`  — group-wise asymmetric round-to-nearest with an integer zero point (the AutoGPTQ grid), no calibration;
# - `hqq`  — half-quadratic quantization (torchao's `_choose_qparams_and_quantize_affine_hqq`, proximal solver on the
#            zero point; float scale and zero rounded to FP16), no calibration;
# - `nf4`  — QLoRA NormalFloat-4: 64-element absmax blocks, the 16-value NF4 code book, absmax double-quantized to int8 in
#            blocks of 256 (torchao `NF4Tensor`'s algorithm; the last scaler block is padded, so any shape works);
# - `gptq` — GPTQ (Frantar et al. 2023) on the `rtn` grid: Hessians `X Xᵀ` from calibration inputs, 1% dampening,
#            act-order with static groups, 128-column blocks; block-sequential (transformer block `i` sees the
#            already-quantized blocks `< i`);
# - `awq`  — activation-aware scaling (Lin et al. 2024) on the `rtn` grid: per input channel `s = s̄_x^α`, normalized,
#            shared by the linears that read the same input (q/k/v; gate/up), `α ∈ {0, 0.05, …, 0.95}` chosen by the
#            output error on sampled calibration inputs, `W ← Q(W·diag(s))·diag(1/s)`; block-sequential; no clip search.

SIMULATED_SCHEMES = ("rtn", "hqq", "nf4", "gptq", "awq")
CALIBRATED_SCHEMES = frozenset({"gptq", "awq"})
NF4_CODE = (-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626,
            0.7230, 1.0)


def scheme_bits(scheme: str, group_size: int) -> float:
    """Nominal bits per weight: 4-bit codes plus a 16-bit scale and zero per group (NF4: an 8-bit double-quantized
    absmax per 64 values plus a 32-bit factor per 256 absmax values)."""
    if scheme == "nf4":
        return 4 + 8 / 64 + 32 / (64 * 256)
    return 4 + 32 / group_size


def int4_fake_quantize(w: Tensor, group_size: int, *, scale_mult: Tensor | None = None) -> Tensor:
    """The `rtn` grid: asymmetric 4-bit codes per group of `group_size` input columns, integer zero point, scale in FP16.
    `scale_mult` (AWQ) multiplies the columns before quantization and divides them afterwards."""
    x = w.float() if scale_mult is None else w.float() * scale_mult
    groups, columns = _groups(x, group_size)
    lo = groups.amin(-1, keepdim=True).clamp(max=0)
    hi = groups.amax(-1, keepdim=True).clamp(min=0)
    scale = ((hi - lo) / 15).half().float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = (-lo / scale).round().clamp(0, 15)
    q = ((groups / scale).round() + zero).clamp(0, 15)
    out = ((q - zero) * scale).reshape(groups.shape[0], -1)[:, :columns].reshape(x.shape)
    return out if scale_mult is None else out / scale_mult


def hqq_fake_quantize(w: Tensor, group_size: int) -> Tensor:
    from torchao.quantization.quant_primitives import _choose_qparams_and_quantize_affine_hqq
    rows, columns = w.shape
    if columns % group_size:
        raise ValueError(f"HQQ needs the input width {columns} divisible by the group size {group_size}")
    q, scale, zero, _ = _choose_qparams_and_quantize_affine_hqq(w.detach().float(), nbits=4, group_size=group_size, axis=1,
                                                                compute_dtype=torch.float32, device=str(w.device), raw_output=True)
    scale, zero = scale.half().float().reshape(rows, -1, 1), zero.half().float().reshape(rows, -1, 1)
    return ((q.float().reshape(rows, -1, group_size) - zero) * scale).reshape(rows, columns)


def nf4_fake_quantize(w: Tensor, block_size: int = 64, scaler_block_size: int = 256) -> Tensor:
    """NF4 quantize–dequantize of `w`: torchao `NF4Tensor`'s algorithm and arithmetic (in bf16, the dtype QLoRA
    quantizes), equal to `NF4Tensor.from_tensor(w.bfloat16(), 64, 256).get_original_weight()` wherever torchao accepts
    the shape; the last weight and scaler blocks are padded otherwise. Returned in float32."""
    dtype = torch.bfloat16
    code = torch.tensor(NF4_CODE, device=w.device, dtype=dtype)
    flat = w.detach().to(dtype).flatten()
    padded = -(-flat.numel() // block_size) * block_size
    if padded != flat.numel():
        flat = torch.cat([flat, flat[-1:].expand(padded - flat.numel())])
    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().max(dim=1).values
    mean = absmax.mean()
    centered = absmax - mean
    count = centered.numel()
    padded_scalers = -(-count // scaler_block_size) * scaler_block_size
    if padded_scalers != count:
        centered = torch.cat([centered, centered[-1:].expand(padded_scalers - count)])
    scaler_blocks = centered.view(-1, scaler_block_size)
    factor = 256 / (2 * scaler_blocks.abs().max(dim=1, keepdim=True).values)
    quantized = (scaler_blocks * factor).round().clamp(-128, 127).to(torch.int8)
    scalers = (quantized / factor).flatten().to(dtype)[:count] + mean
    index = ((blocks / absmax[:, None]).unsqueeze(-1) - code).abs().min(dim=-1).indices
    out = code[index] * scalers[:, None]
    return out.flatten()[:w.numel()].reshape(w.shape).float()


def gptq_quantize_weight(w: Tensor, hessian: Tensor, group_size: int, *, damp: float = 0.01, block: int = 128,
                         act_order: bool = True) -> Tensor:
    """GPTQ on the `rtn` grid (static groups: each group's scale and zero from the original weights)."""
    W = w.detach().float().clone()
    H = hessian.float().clone()
    columns = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    groups, _ = _groups(W, group_size)
    lo = groups.amin(-1).clamp(max=0)
    hi = groups.amax(-1).clamp(min=0)
    scale = ((hi - lo) / 15).half().float()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = (-lo / scale).round().clamp(0, 15)                       # rows × groups
    group_of = torch.arange(columns, device=W.device) // group_size
    perm = torch.argsort(torch.diag(H), descending=True) if act_order else torch.arange(columns, device=W.device)
    W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(columns, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, columns, block):
        i2 = min(i1 + block, columns)
        W1, Err1, Hinv1 = W[:, i1:i2].clone(), torch.zeros_like(W[:, i1:i2]), Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            g = group_of[perm[i1 + i]]
            s, z = scale[:, g], zero[:, g]
            col = W1[:, i]
            q = ((col / s).round() + z).clamp(0, 15)
            deq = (q - z) * s
            Q[:, i1 + i] = deq
            err = (col - deq) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)].to(w.dtype)


def _block_key(name: str) -> str:
    """Transformer block of a linear layer's qualified name (`…layers.<i>.…`, GPT-2 `…h.<i>.…`), else the name."""
    match = re.search(r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)\.", name)
    return f"block{int(match[1]):05d}" if match else name


class _StopForward(Exception):
    pass


def host_linears(host: nn.Module) -> list[tuple[str, nn.Linear]]:
    """The linear layers weight-only quantization touches: every `nn.Linear` except the output head and a layer tied
    to the input embedding (GPT-2 `Conv1D` must be converted first, `conv1d_to_linear`)."""
    head = host.get_output_embeddings() if hasattr(host, "get_output_embeddings") else None
    embedding = host.get_input_embeddings() if hasattr(host, "get_input_embeddings") else None
    tied = {id(embedding.weight)} if embedding is not None else set()
    return [(name, m) for name, m in host.named_modules()
            if isinstance(m, nn.Linear) and m is not head and id(m.weight) not in tied and not name.endswith("lm_head")]


def _collect(linears: list[tuple[str, nn.Linear]], forward: Callable, scheme: str, samples: int,
             seed: int) -> dict[str, dict[str, Any]]:
    """Run the calibration batches (`forward`) with hooks on `linears`: GPTQ accumulates `X Xᵀ`; AWQ the mean |x| per
    input channel, up to `samples` sampled input rows, and the identity of the first input (linears reading the same
    tensor share AWQ scales). Each batch is cut short once every hooked linear has run."""
    stats: dict[str, dict[str, Any]] = {name: {"n": 0} for name, _ in linears}
    fired: set[str] = set()
    generator = torch.Generator().manual_seed(seed)

    def hook(name: str) -> Callable:
        def run(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
            x = inputs[0].detach()
            entry = stats[name]
            if "input_key" not in entry:
                entry["input_key"] = (x.data_ptr(), tuple(x.shape))
            x = x.reshape(-1, x.shape[-1]).float()
            if scheme == "gptq":
                entry["H"] = entry.get("H", 0) + x.T @ x
            else:
                entry["abs"] = entry.get("abs", 0) + x.abs().sum(0)
                rows = entry.setdefault("rows", [])
                if sum(r.shape[0] for r in rows) < samples:
                    keep = min(x.shape[0], max(1, samples // 8))
                    rows.append(x[torch.randperm(x.shape[0], generator=generator)[:keep].to(x.device)])
            entry["n"] += x.shape[0]
            fired.add(name)
            if len(fired) == len(linears):
                raise _StopForward
        return run

    def step(call: Callable) -> None:
        fired.clear()
        try:
            call()
        except _StopForward:
            pass

    handles = [module.register_forward_hook(hook(name)) for name, module in linears]
    try:
        forward(step)
    finally:
        for handle in handles:
            handle.remove()
    return stats


def simulate_weight_only_(host: nn.Module, scheme: str, *, group_size: int, forward: Callable | None = None, samples: int = 2048,
                          seed: int = 0, damp: float = 0.01, act_order: bool = True,
                          awq_grid: Sequence[float] = tuple(i / 20 for i in range(20))) -> dict[str, Any]:
    """Quantize–dequantize the host's linear weights in place with a simulated scheme (see the section comment).

    `forward(step)` runs the calibration batches (GPTQ, AWQ): for each batch it calls `step(run)`, where `run()` runs
    the model on that batch; it is called once per transformer block (block-sequential)."""
    if scheme not in SIMULATED_SCHEMES:
        raise ValueError(f"scheme must be one of {SIMULATED_SCHEMES}")
    if scheme in CALIBRATED_SCHEMES and forward is None:
        raise ValueError(f"{scheme} needs calibration batches")
    conv1d_to_linear(host)
    linears = host_linears(host)
    skipped = [name for name, m in linears if m.in_features % group_size and scheme in {"hqq"}]
    linears = [(name, m) for name, m in linears if name not in skipped]
    info: dict[str, Any] = {"scheme": scheme, "group_size": None if scheme == "nf4" else group_size,
                            "bits_per_weight": scheme_bits(scheme, group_size), "linear_quantized": len(linears),
                            "linear_skipped": skipped}
    with torch.no_grad():
        if scheme not in CALIBRATED_SCHEMES:
            for _, module in linears:
                w = module.weight
                new = (int4_fake_quantize(w, group_size) if scheme == "rtn" else hqq_fake_quantize(w, group_size)
                       if scheme == "hqq" else nf4_fake_quantize(w))
                w.copy_(new.to(w.dtype))
            return info
        blocks: dict[str, list[tuple[str, nn.Linear]]] = {}
        for name, module in linears:
            blocks.setdefault(_block_key(name), []).append((name, module))
        chosen: dict[str, float] = {}
        for _, members in sorted(blocks.items()):
            stats = _collect(members, forward, scheme, samples, seed)
            if scheme == "gptq":
                for name, module in members:
                    if stats[name]["n"]:
                        hessian = 2 * stats[name]["H"] / stats[name]["n"]
                        module.weight.copy_(gptq_quantize_weight(module.weight, hessian, group_size, damp=damp,
                                                                 act_order=act_order).to(module.weight.dtype))
                continue
            shared: dict[Any, list[tuple[str, nn.Linear]]] = {}
            for name, module in members:
                if stats[name]["n"]:
                    shared.setdefault(stats[name]["input_key"], []).append((name, module))
            for group in shared.values():
                first = stats[group[0][0]]
                mean_abs = first["abs"] / first["n"]
                x = torch.cat(first["rows"])[:samples]
                references = [x @ m.weight.float().T for _, m in group]
                best = best_alpha = best_scales = None
                for alpha in awq_grid:
                    s = mean_abs.clamp_min(1e-4) ** alpha
                    s = s / (s.max() * s.min()).sqrt()
                    error = sum(float((x @ int4_fake_quantize(m.weight, group_size, scale_mult=s).T - ref).square().sum())
                                for (_, m), ref in zip(group, references))
                    if best is None or error < best:
                        best, best_alpha, best_scales = error, alpha, s
                for name, module in group:
                    module.weight.copy_(int4_fake_quantize(module.weight, group_size, scale_mult=best_scales).to(module.weight.dtype))
                    chosen[name] = float(best_alpha)
    info.update(calibration_rows_per_linear=samples if scheme == "awq" else None, blocks=len(blocks))
    if scheme == "gptq":
        info.update(damp=damp, act_order=act_order)
    else:
        info["awq_alpha_mean"] = float(sum(chosen.values()) / len(chosen)) if chosen else None
    return info


def quantize_input_embedding_(host: nn.Module, group_size: int, *, include_head: bool = False) -> dict[str, Any]:
    """`-emb` variants: the input-embedding rows quantize–dequantized on the `rtn` grid (4 bits, groups along the width).
    A tied output head keeps the 16-bit table (the lookup gets an untied copy) unless `include_head`, which quantizes
    the shared table, i.e. both (GGUF-style)."""
    embedding = host.get_input_embeddings()
    head = host.get_output_embeddings()
    tied = head is not None and head.weight is embedding.weight
    with torch.no_grad():
        quantized = int4_fake_quantize(embedding.weight, group_size).to(embedding.weight.dtype)
        if tied and not include_head:
            copy = nn.Embedding(embedding.num_embeddings, embedding.embedding_dim, padding_idx=embedding.padding_idx,
                                device=embedding.weight.device, dtype=embedding.weight.dtype)
            copy.weight.copy_(quantized)
            copy.weight.requires_grad_(False)
            host.set_input_embeddings(copy)
        else:
            embedding.weight.copy_(quantized)
            if include_head and not tied:
                head.weight.copy_(int4_fake_quantize(head.weight, group_size).to(head.weight.dtype))
    return {"embedding_group_size": group_size, "embedding_bits_per_weight": scheme_bits("rtn", group_size),
            "head_tied": bool(tied), "head_quantized": bool(include_head)}
