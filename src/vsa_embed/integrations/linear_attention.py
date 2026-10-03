"""Hybrid linear-attention hosts (Qwen3.5: Gated DeltaNet + gated full attention; WP-Qwen35).

Transformers (≥ 5) binds the Gated DeltaNet token mixer of `qwen3_5` to the fast kernels when their packages are
installed — `flash-linear-attention` (`fla`: chunked and fused-recurrent gated delta rule, Triton) and `causal-conv1d`
(CUDA) — and otherwise to PyTorch reference implementations ("correct but much slower"). The binding is made once,
at import, whatever the device, so with the kernels installed a CPU forward fails (Triton cannot read CPU tensors).

`install_device_dispatch()` replaces the four module-level functions of `modeling_qwen3_5` by dispatchers: the
fast kernel for CUDA inputs, the reference implementation otherwise. It counts the calls of each implementation
(`kernel_calls()`), so a run can show which path actually ran; `kernel_status()` records what transformers bound
(implementation module, package versions). The reference path is the one transformers runs without the packages,
so CPU results are unchanged; on CUDA the fast kernels compute the same function (checked by
`tests/test_qwen35_hosts.py` and the GPU smoke in `experiments/e9-retrofit/env/`).

`is_linear_attention_host(model)` tells such hosts apart (`config.layer_types` holds `linear_attention`);
`lora_layer_coverage(model)` counts LoRA adapters per decoder layer (every layer must have its token mixer adapted).
"""

from __future__ import annotations

import collections
import contextlib
import functools
import importlib
from typing import Any, Callable, Iterator

import torch
from torch import nn

# module-level functions of `transformers.models.qwen3_5.modeling_qwen3_5` → the kernel each binds when installed
FAST_FUNCTIONS = {"torch_chunk_gated_delta_rule": ("fla", "chunk_gated_delta_rule"),
                  "torch_recurrent_gated_delta_rule": ("fla", "fused_recurrent_gated_delta_rule"),
                  "causal_conv1d_fn": ("causal_conv1d", "causal_conv1d_fn"),
                  "causal_conv1d_update": ("causal_conv1d", "causal_conv1d_update")}
MODELING_MODULES = ("transformers.models.qwen3_5.modeling_qwen3_5",)
DISTRIBUTIONS = {"fla": "flash-linear-attention", "causal_conv1d": "causal-conv1d"}
_CALLS: collections.Counter = collections.Counter()


def is_linear_attention_host(model: nn.Module) -> bool:
    """A host whose decoder mixes linear-attention layers in (Qwen3.5's `layer_types`)."""
    config = getattr(model, "config", None)
    layer_types = getattr(config, "layer_types", None) or getattr(getattr(config, "text_config", None), "layer_types", None)
    return bool(layer_types) and "linear_attention" in layer_types


def _modeling(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:              # transformers without qwen3_5 (the pinned 4.54)
        return None


def _resolved(function: Callable) -> tuple[Callable | None, Callable | None]:
    """(implementation transformers bound, reference implementation) of a `use_kernel_func_from_hub_with_fallback`
    function, read from the wrapper's closure; (None, None) if the wrapper has another shape."""
    target = getattr(function, "_vsa_dispatch_of", function)
    code, closure = getattr(target, "__code__", None), getattr(target, "__closure__", None)
    if code is None or closure is None:
        return None, None
    cells = dict(zip(code.co_freevars, (c.cell_contents for c in closure)))
    return cells.get("implementation"), getattr(target, "__wrapped__", None)


def _version(package: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(DISTRIBUTIONS.get(package, package))
    except metadata.PackageNotFoundError:
        return None


def kernel_status() -> dict[str, Any]:
    """What transformers bound for each Gated DeltaNet function: `fast` (the kernel package's function) or
    `reference` (the PyTorch fallback), the implementation's module, the package versions and whether the CUDA
    dispatch is installed; `{}` without a linear-attention modeling module."""
    out: dict[str, Any] = {}
    for module_name in MODELING_MODULES:
        module = _modeling(module_name)
        if module is None:
            continue
        functions = {}
        for name, (package, kernel) in FAST_FUNCTIONS.items():
            function = getattr(module, name, None)
            implementation, reference = _resolved(function) if function is not None else (None, None)
            fast = implementation is not None and reference is not None and implementation is not reference
            functions[name] = {"kernel": f"{package}.{kernel}", "bound": "fast" if fast else "reference" if implementation else "unknown",
                               "implementation": f"{getattr(implementation, '__module__', '?')}.{getattr(implementation, '__name__', '?')}"
                               if implementation is not None else None,
                               "dispatch": bool(getattr(function, "_vsa_dispatch_of", None))}
        out[module_name.rsplit(".", 1)[-1]] = {"functions": functions,
                                               "packages": {p: _version(p) for p in ("fla", "causal_conv1d")},
                                               "fast_path_bound": all(f["bound"] == "fast" for f in functions.values())}
    return out


def _first_tensor(args: tuple, kwargs: dict) -> torch.Tensor | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value
    return None


_REFERENCE_ONLY = [False]


@contextlib.contextmanager
def reference_only() -> Iterator[None]:
    """Within the block every dispatcher runs the reference implementation, on any device (speed and agreement
    checks of the fast path)."""
    previous = _REFERENCE_ONLY[0]
    _REFERENCE_ONLY[0] = True
    try:
        yield
    finally:
        _REFERENCE_ONLY[0] = previous


def _dispatcher(name: str, wrapped: Callable, reference: Callable) -> Callable:
    @functools.wraps(reference)
    def dispatch(*args: Any, **kwargs: Any) -> Any:
        tensor = _first_tensor(args, kwargs)
        if tensor is not None and tensor.is_cuda and not _REFERENCE_ONLY[0]:
            _CALLS[(name, "fast")] += 1
            return wrapped(*args, **kwargs)           # transformers' wrapper: filters kwargs for the kernel
        _CALLS[(name, "reference")] += 1
        return reference(*args, **kwargs)
    dispatch._vsa_dispatch_of = wrapped               # type: ignore[attr-defined]
    return dispatch


def install_device_dispatch() -> dict[str, Any]:
    """Make every bound fast kernel CUDA-only (reference implementation for CPU tensors); idempotent. Returns
    `kernel_status()`. Functions already bound to the reference implementation are left as they are."""
    for module_name in MODELING_MODULES:
        module = _modeling(module_name)
        if module is None:
            continue
        for name in FAST_FUNCTIONS:
            function = getattr(module, name, None)
            if function is None or getattr(function, "_vsa_dispatch_of", None) is not None:
                continue
            implementation, reference = _resolved(function)
            if implementation is None or reference is None or implementation is reference:
                continue
            setattr(module, name, _dispatcher(name, function, reference))
    return kernel_status()


def kernel_calls(reset: bool = False) -> dict[str, dict[str, int]]:
    """Calls per function and implementation (`fast` / `reference`) since the last reset, through the dispatchers;
    a function bound to its reference implementation without a dispatcher is not counted."""
    out: dict[str, dict[str, int]] = {}
    for (name, kind), count in sorted(_CALLS.items()):
        out.setdefault(name, {})[kind] = count
    if reset:
        _CALLS.clear()
    return out


def lora_layer_coverage(model: nn.Module) -> dict[str, Any]:
    """LoRA adapters per decoder layer and layer type: `{"layers": n, "by_type": {type: {"layers", "adapters",
    "per_layer"}}, "uncovered_mixers": [layer indices whose token mixer has no adapter]}`."""
    from .transformers import LoRALinear
    config = getattr(model, "config", None)
    layer_types = list(getattr(config, "layer_types", None) or [])
    per_layer: dict[int, set[str]] = collections.defaultdict(set)
    mixers: dict[int, int] = collections.Counter()
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue
        index = int(parts[parts.index("layers") + 1])
        per_layer[index].add(parts[-1])
        if any(p in {"self_attn", "linear_attn", "attn", "attention"} for p in parts):
            mixers[index] += 1
    count = len(layer_types) or (max(per_layer) + 1 if per_layer else 0)
    by_type: dict[str, dict[str, Any]] = {}
    for index in range(count):
        kind = layer_types[index] if index < len(layer_types) else "layer"
        entry = by_type.setdefault(kind, {"layers": 0, "adapters": 0, "per_layer": set()})
        entry["layers"] += 1
        entry["adapters"] += len(per_layer.get(index, ()))
        entry["per_layer"].add(len(per_layer.get(index, ())))
    for entry in by_type.values():
        entry["per_layer"] = sorted(entry["per_layer"])
    return {"layers": count, "adapters": sum(len(v) for v in per_layer.values()), "by_type": by_type,
            "targets": sorted(set().union(*per_layer.values())) if per_layer else [],
            "uncovered_mixers": [i for i in range(count) if not mixers.get(i)]}
