"""Post-training weight-only quantization with torchao (B8 / E4.4).

`quantize_weight_only(model, bits)` quantizes every `nn.Linear` of the host transformer to INT8 or
INT4 (group-wise) in place. The span channel's dictionary and projector stay in full precision
unless `include_channel=True` (the variant that quantizes them too). GPT-2's `Conv1D` layers are
converted to equivalent `nn.Linear` first so torchao can see them.
"""

from __future__ import annotations

import torch
from torch import nn


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
    config = Int8WeightOnlyConfig() if bits == 8 else Int4WeightOnlyConfig(group_size=group_size)
    names = {id(m): n for n, m in model.named_modules()}

    def keep(module: nn.Module, fqn: str) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if not include_channel and (fqn.startswith(channel_prefix) or ".channel." in fqn):
            return False
        if bits == 4 and module.in_features % group_size:
            return False
        return True

    quantize_(model, config, filter_fn=keep)
    return {"conv1d_converted": converted, "bits": bits}
