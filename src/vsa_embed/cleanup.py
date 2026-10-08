"""Cleanup memories for unbinding (decision 60, the binding and unbinding program).

An unbound vector is a noisy estimate of a filler; *cleanup* maps it back to the dictionary.

- `cosine_scores` / `nearest`: hard cleanup, cosine nearest neighbour over a dictionary (the trained atomic vectors),
  optionally restricted per query to allowed rows (`relation_candidates`: the type-constrained variant, the atomics
  observed under that relation in the ontology frames).
- `filtered_ranks`: the rank of each query's gold row among its allowed rows, other gold rows of the same query removed
  (the "filtered" setting of link prediction), ties counted at half weight (the expected rank under random
  tie-breaking, as `statistics.positive_ranks(ties="mid")`), so a store that cannot separate candidates scores chance.
- `SoftCleanup`: the differentiable version (one or more modern-Hopfield / attention steps over the dictionary with a
  learned inverse temperature and an optional candidate mask), used by the unbinding readout arm.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NEG = float("-inf")


def relation_candidates(relations: Tensor, fillers: Tensor, relation_count: int, atomic_count: int) -> Tensor:
    """Boolean `(relations, atomics)` mask: atomic `a` is a candidate for relation `r` when some frame edge is `(r, a)`."""
    mask = torch.zeros(relation_count, atomic_count, dtype=torch.bool, device=relations.device)
    if relations.numel():
        mask[relations.long(), fillers.long()] = True
    return mask


def cosine_scores(queries: Tensor, dictionary: Tensor, *, normalized_dictionary: bool = False) -> Tensor:
    """`(n, m)` cosine similarities (float32) of `queries` `(n, d)` to dictionary rows `(m, d)`."""
    keys = dictionary.float() if normalized_dictionary else F.normalize(dictionary.float(), dim=-1)
    return F.normalize(queries.float(), dim=-1) @ keys.T


def _chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


def nearest(queries: Tensor, dictionary: Tensor, allowed: Tensor | None = None, *, chunk: int = 4096) -> tuple[Tensor, Tensor]:
    """Hard cleanup: per query the index and cosine of the nearest allowed dictionary row (`allowed`: `(n, m)` bool)."""
    keys = F.normalize(dictionary.float(), dim=-1)
    index, score = [], []
    for part in _chunks(queries.shape[0], chunk):
        scores = cosine_scores(queries[part], keys, normalized_dictionary=True)
        if allowed is not None:
            scores = scores.masked_fill(~allowed[part], NEG)
        best = scores.max(-1)
        index.append(best.indices); score.append(best.values)
    if not index:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0)
    return torch.cat(index), torch.cat(score)


def filtered_ranks(scores: Tensor, gold: Tensor, *, allowed: Tensor | None = None, exclude: Tensor | None = None,
                   return_hits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """Rank (1 = best; float, ties at half weight) of `gold[i]` among row `i`'s allowed candidates of `scores` `(n, m)`.

    `allowed` `(n, m)` restricts the candidates (the gold is always kept); `exclude` `(n, m)` removes other correct
    answers of the same query (filtered ranking; the gold is never excluded). A non-finite gold score ranks last.
    `return_hits`: also the expected top-1 under random tie-breaking (`1 / (1 + ties)` when no candidate scores above
    the gold, else 0)."""
    n, m = scores.shape
    rows = torch.arange(n, device=scores.device)
    keep = torch.ones_like(scores, dtype=torch.bool) if allowed is None else allowed.clone()
    if exclude is not None:
        keep &= ~exclude
    keep[rows, gold] = True
    target = scores[rows, gold]
    above = ((scores > target[:, None]) & keep).sum(-1).float()
    tied = ((scores == target[:, None]) & keep).sum(-1).float() - 1.0          # the gold itself is not a tie
    rank = 1.0 + above + 0.5 * tied
    worst = keep.sum(-1).float()
    finite = torch.isfinite(target)
    rank = torch.where(finite, rank, worst)
    if not return_hits:
        return rank
    hits = torch.where(finite & (above == 0), 1.0 / (1.0 + tied), torch.zeros_like(rank))
    return rank, hits


def chunked_filtered_ranks(queries: Tensor, dictionary: Tensor, gold: Tensor, *, allowed_rows: Tensor | None = None,
                           allowed_index: Tensor | None = None, exclude_pairs: tuple[Tensor, Tensor] | None = None,
                           chunk: int = 4096) -> Tensor:
    """`filtered_ranks` of cosine cleanup over a large dictionary, in query chunks.

    `allowed_rows` `(k, m)` with `allowed_index` `(n,)` gives each query the candidate mask of its row (e.g. the
    relation's `relation_candidates` row); `exclude_pairs` = (query index, dictionary index) pairs removed from that
    query's candidates (other gold fillers of the same frame and relation)."""
    keys = F.normalize(dictionary.float(), dim=-1)
    out = torch.empty(queries.shape[0], dtype=torch.float32, device=queries.device)
    if exclude_pairs is not None:
        order = torch.argsort(exclude_pairs[0])
        ex_q, ex_d = exclude_pairs[0][order], exclude_pairs[1][order]
    for part in _chunks(queries.shape[0], chunk):
        scores = cosine_scores(queries[part], keys, normalized_dictionary=True)
        allowed = allowed_rows[allowed_index[part]] if allowed_rows is not None else None
        exclude = None
        if exclude_pairs is not None and ex_q.numel():
            lo, hi = torch.searchsorted(ex_q, torch.tensor([part.start, part.stop], device=ex_q.device))
            if hi > lo:
                exclude = torch.zeros_like(scores, dtype=torch.bool)
                exclude[ex_q[lo:hi] - part.start, ex_d[lo:hi]] = True
        out[part] = filtered_ranks(scores, gold[part], allowed=allowed, exclude=exclude)
    return out


class SoftCleanup(nn.Module):
    """Differentiable cleanup: `steps` modern-Hopfield updates `u ← softmax(β · cos(u, A) + log mask) A` over a dictionary
    `A` (rows normalized), with a learned inverse temperature `β = exp(log_beta)` (initial `beta`). With `steps = 1` it
    is one attention read of the dictionary; a large `β` approaches the hard nearest neighbour. Returns the cleaned
    vectors (in the dictionary's space) and, with `return_weights`, the last step's attention weights."""

    def __init__(self, *, beta: float = 16.0, steps: int = 1, learn_beta: bool = True) -> None:
        super().__init__()
        if steps < 1 or beta <= 0:
            raise ValueError("steps must be ≥ 1 and beta positive")
        self.steps = int(steps)
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta)), requires_grad=learn_beta)

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    def forward(self, queries: Tensor, dictionary: Tensor, mask: Tensor | None = None, *,
                return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """`queries` `(..., d)`; `dictionary` `(m, d)`; `mask` `(..., m)` bool (True = allowed; a row with no allowed
        entry returns zeros)."""
        keys = F.normalize(dictionary, dim=-1)
        state = queries
        weights = None
        for _ in range(self.steps):
            logits = self.beta * (F.normalize(state, dim=-1) @ keys.T)
            if mask is not None:
                logits = logits.masked_fill(~mask, NEG)
            weights = torch.softmax(logits, dim=-1)
            if mask is not None:
                weights = torch.nan_to_num(weights, nan=0.0)              # a fully masked row → zero weights
            state = weights @ keys
        return (state, weights) if return_weights else state
