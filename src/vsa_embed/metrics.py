"""Retrieval and calibration metrics without external ML dependencies."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def retrieval_metrics(queries: Tensor, candidates: Tensor, targets: Tensor) -> dict[str, float]:
    similarities = F.normalize(queries, dim=-1) @ F.normalize(candidates, dim=-1).T
    order = similarities.argsort(dim=-1, descending=True)
    ranks = order.eq(targets.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1
    target_score = similarities.gather(1, targets.unsqueeze(1)).squeeze(1)
    masked = similarities.clone()
    masked.scatter_(1, targets.unsqueeze(1), -torch.inf)
    margin = target_score - masked.max(dim=1).values
    return {
        "top1": ranks.eq(1).float().mean().item(),
        "mrr": ranks.float().reciprocal().mean().item(),
        "mean_rank": ranks.float().mean().item(),
        "cosine_margin": margin.mean().item(),
    }
