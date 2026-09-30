"""Context queries for attentive composition (formulation §1.3).

P1 — `CausalLocalContext`: `q_t` from the token embeddings `E[x_{t−w+1..t}]`, as a windowed
mean or a depthwise causal convolution, then projected. Position `t` never sees tokens after
`t`, so it is safe to inject at a span's last subtoken in a causal LM.

P2 — `FrameSidecar`: a cross-attention head in which a mid-network hidden state `h_t^{(ℓ)}` is the
query and the concept's bound edges are keys/values; its output is added to the residual stream
at layer `ℓ+1` at the span's last position. The model-specific hook lives in
`integrations/transformers.py`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compose import segment_softmax


class CausalLocalContext(nn.Module):
    """`q_t = W · pool(E[x_{t−w+1..t}])` with `pool` = mean or depthwise causal convolution."""

    def __init__(self, embedding_dimension: int, output_dimension: int, *, window: int = 8,
                 mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "conv"}:
            raise ValueError("mode must be 'mean' or 'conv'")
        if window < 1:
            raise ValueError("window must be positive")
        self.window, self.mode = window, mode
        if mode == "conv":
            self.conv = nn.Conv1d(embedding_dimension, embedding_dimension, window,
                                  groups=embedding_dimension, bias=False)
            nn.init.constant_(self.conv.weight, 1.0 / window)
        self.projection = nn.Linear(embedding_dimension, output_dimension, bias=False)

    def forward(self, embeddings: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """`embeddings`: (batch, time, dim) → (batch, time, output_dimension)."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape (batch, time, dim)")
        x = embeddings if attention_mask is None else embeddings * attention_mask[..., None].to(embeddings.dtype)
        padded = F.pad(x.transpose(1, 2), (self.window - 1, 0))  # left padding only: causal
        if self.mode == "conv":
            pooled = self.conv(padded)
        else:
            summed = F.avg_pool1d(padded, self.window, stride=1) * self.window
            ones = torch.ones_like(x[..., :1]) if attention_mask is None else attention_mask[..., None].to(x.dtype)
            counts = F.avg_pool1d(F.pad(ones.transpose(1, 2), (self.window - 1, 0)), self.window, stride=1) * self.window
            pooled = summed / counts.clamp_min(1.0)
        return self.projection(pooled.transpose(1, 2))


class FrameSidecar(nn.Module):
    """Cross-attention from host hidden states to concept frames (P2)."""

    def __init__(self, hidden_dimension: int, frame_dimension: int, *, key_dimension: int = 64,
                 init_gate: float = 0.0) -> None:
        super().__init__()
        self.key_dimension = key_dimension
        self.query = nn.Linear(hidden_dimension, key_dimension, bias=False)
        self.key = nn.Linear(frame_dimension, key_dimension, bias=False)
        self.value = nn.Linear(frame_dimension, hidden_dimension, bias=False)
        # A zero gate makes the sidecar the identity at initialization (host unchanged).
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, hidden: Tensor, bound_edges: Tensor, segments: Tensor) -> tuple[Tensor, Tensor]:
        """`hidden`: (occurrences, hidden_dim) at span ends; `bound_edges`: (edges, frame_dim) with
        `segments` mapping each edge to its occurrence. Returns (injection, attention weights)."""
        queries = self.query(hidden)
        keys = self.key(bound_edges)
        scores = (queries[segments] * keys).sum(-1) / math.sqrt(self.key_dimension)
        weights = segment_softmax(scores, segments, hidden.shape[0])
        values = self.value(bound_edges) * weights[:, None]
        injection = hidden.new_zeros(hidden.shape).index_add(0, segments, values)
        return torch.tanh(self.gate) * injection, weights
