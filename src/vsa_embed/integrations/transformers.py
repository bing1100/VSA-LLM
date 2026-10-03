"""Hugging Face causal-LM integration for the span channel (B6).

`ChannelLM` wraps any HF causal LM (GPT-2, Llama/SmolLM2, Qwen2/3, Qwen3.5 text-only): it embeds the tokens,
adds the span channel at linked positions (optionally with a P1 causal context query), runs the
base model on `inputs_embeds`, and computes the LM loss with a chunked cross-entropy that never
materialises the full `(tokens × vocabulary)` logits (Qwen2.5 has 151,936 entries). The host can
be trained, frozen, or frozen with LoRA adapters on its attention and MLP projections.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..context import CausalLocalContext
from ..span_channel import SpanChannel


class LoRALinear(nn.Module):
    """`y = W x + (α / r) · B A x` with `B` zero-initialised (identity at start)."""

    def __init__(self, base: nn.Module, rank: int = 16, alpha: float = 32.0, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        weight = base.weight
        # GPT-2 uses Conv1D with weight (in, out); nn.Linear uses (out, in).
        self.transposed = type(base).__name__ == "Conv1D"
        in_features, out_features = (weight.shape if self.transposed else weight.shape[::-1])
        self.base, self.rank, self.scale = base, rank, alpha / rank
        dtype = dtype or weight.dtype       # `dtype`: adapters in another precision than a 16-bit host (float32)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features, dtype=dtype, device=weight.device) / math.sqrt(in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=weight.device))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj", "c_fc")
# Opt-in target sets (`model.lora_targets`: a set name or a list of projection names). Qwen3.5's Gated DeltaNet layers
# (18 of 24) mix tokens through in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj, which the default set
# misses (it reaches only the 6 full-attention layers' q/k/v/o and every MLP); `linear_attention` adds them.
LINEAR_ATTENTION_TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
LORA_TARGET_SETS = {"default": LORA_TARGETS, "linear_attention": LORA_TARGETS + LINEAR_ATTENTION_TARGETS}


def lora_targets(spec: str | Iterable[str] | None) -> tuple[str, ...]:
    """Projection names of a `model.lora_targets` value: None → the default set; a set name; or a list of names."""
    if spec is None:
        return LORA_TARGETS
    if isinstance(spec, str):
        if spec not in LORA_TARGET_SETS:
            raise ValueError(f"unknown LoRA target set {spec!r}; choose from {sorted(LORA_TARGET_SETS)} or give a list")
        return LORA_TARGET_SETS[spec]
    names = tuple(spec)
    if not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError("model.lora_targets must be a set name or a non-empty list of projection names")
    return names


def add_lora(model: nn.Module, *, rank: int = 16, alpha: float = 32.0,
             targets: Iterable[str] = LORA_TARGETS, dtype: torch.dtype | None = None) -> list[nn.Parameter]:
    """Replace target projections by `LoRALinear`; returns the new trainable parameters (`dtype`: the
    adapters' dtype, default the host weights')."""
    targets = set(targets)
    params: list[nn.Parameter] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and hasattr(child, "weight") and child.weight.ndim == 2 \
                    and not isinstance(child, LoRALinear):
                wrapped = LoRALinear(child, rank, alpha, dtype=dtype)
                setattr(module, child_name, wrapped)
                params += [wrapped.lora_a, wrapped.lora_b]
    return params


def chunked_causal_lm_loss(hidden: Tensor, output_weight: Tensor, labels: Tensor, *, chunk: int = 2048,
                           output_bias: Tensor | None = None, reduction: str = "mean") -> Tensor:
    """Next-token cross-entropy from final hidden states without building full logits.

    `hidden`: (batch, time, d); `labels`: (batch, time) with −100 ignored; position `t` predicts
    `labels[t + 1]`. `reduction="none"` returns per-position losses (batch, time − 1) with 0 at
    ignored positions.
    """
    h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    y = labels[:, 1:].reshape(-1)
    losses = []
    for start in range(0, h.shape[0], chunk):
        logits = F.linear(h[start:start + chunk], output_weight, output_bias).float()
        losses.append(F.cross_entropy(logits, y[start:start + chunk], ignore_index=-100, reduction="none"))
    per_token = torch.cat(losses)
    if reduction == "none":
        return per_token.view(labels.shape[0], labels.shape[1] - 1)
    valid = (y != -100).sum().clamp_min(1)
    return per_token.sum() / valid


class ChannelLM(nn.Module):
    """A causal LM with an optional span channel on its input embeddings."""

    def __init__(self, model: nn.Module, channel: SpanChannel | None = None, *,
                 context: CausalLocalContext | None = None, host_mode: str = "train",
                 lora_rank: int = 16, loss_chunk: int = 2048, adapter_dtype: torch.dtype | None = None,
                 lora_targets: Iterable[str] = LORA_TARGETS) -> None:
        super().__init__()
        if host_mode not in {"train", "frozen", "lora"}:
            raise ValueError("host_mode must be train, frozen or lora")
        self.model, self.channel, self.context = model, channel, context
        self.host_mode, self.loss_chunk = host_mode, loss_chunk
        if host_mode in {"frozen", "lora"}:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        if host_mode == "lora":
            add_lora(self.model, rank=lora_rank, dtype=adapter_dtype, targets=lora_targets)

    @property
    def base(self) -> nn.Module:
        return self.model.base_model

    def embed(self, input_ids: Tensor, spans: dict[str, Tensor] | None) -> Tensor:
        embeddings = self.model.get_input_embeddings()(input_ids)
        if self.channel is None or spans is None:
            return embeddings
        context = None
        if self.context is not None and spans["entry"].numel():
            # P1 context: causal pooling of the token embeddings up to the injection position.
            pooled = self.context(embeddings.detach() if self.host_mode != "train" else embeddings)
            context = pooled[spans["batch"].to(input_ids.device), spans["inject"].to(input_ids.device)]
        return self.channel(embeddings, spans, input_ids=input_ids, context=context)

    def hidden_states(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                      spans: dict[str, Tensor] | None = None) -> Tensor:
        embeddings = self.embed(input_ids, spans)
        outputs = self.base(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                spans: dict[str, Tensor] | None = None, labels: Tensor | None = None,
                reduction: str = "mean") -> dict[str, Tensor]:
        hidden = self.hidden_states(input_ids, attention_mask, spans)
        result: dict[str, Tensor] = {"hidden": hidden}
        if labels is not None:
            head = self.model.get_output_embeddings()
            result["loss"] = chunked_causal_lm_loss(hidden, head.weight, labels, chunk=self.loss_chunk,
                                                    output_bias=getattr(head, "bias", None), reduction=reduction)
        return result

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
