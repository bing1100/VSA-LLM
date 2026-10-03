"""New-word frame inference (E10.0 (e)): infer a frame from a few observations, everything frozen.

A new concept is observed `k` times (noisy targets). With the dictionary frozen, its frame is the
sparse set of `(relation option, filler)` edges whose bound vectors best explain the mean
observation: non-negative orthogonal matching pursuit over the candidate dictionary
`{P T_r(a) : r ∈ relations (seed relations + crystallized slots), a ∈ fillers}` with a correlation
stopping rule. The inferred frame is then composed zero-shot (uniform masses, as a new ontology
entry would be) and scored on observations not used for inference.

Baselines: the frame of the nearest training concept (nearest composed row to the mean
observation), random frames of the gold degree, and the oracle dictionary (teacher roles and
atomics) as an upper bound on identifiability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class InferredFrame:
    columns: list[int]
    fillers: list[int]
    coefficients: list[float]


def candidate_dictionary(bind: callable, columns: Sequence[int], fillers: Sequence[int],
                         project: callable | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """All `(column, filler)` pairs and their (projected) bound vectors."""
    cols = torch.tensor(list(columns), dtype=torch.long)
    fills = torch.tensor(list(fillers), dtype=torch.long)
    grid_c = cols.repeat_interleave(fills.numel()); grid_f = fills.repeat(cols.numel())
    with torch.no_grad():
        vectors = bind(grid_c, grid_f)
        if project is not None:
            vectors = project(vectors)
    return grid_c, grid_f, vectors


def _nnls(design: Tensor, target: Tensor, iterations: int = 200) -> Tensor:
    """Small non-negative least squares by projected gradient (design: n × s)."""
    gram = design.T @ design
    rhs = design.T @ target
    step = 1.0 / float(torch.linalg.matrix_norm(gram, ord=2).clamp_min(1e-8))
    beta = torch.linalg.lstsq(gram + 1e-6 * torch.eye(gram.shape[0]), rhs[:, None]).solution[:, 0].clamp_min(0)
    for _ in range(iterations):
        beta = (beta - step * (gram @ beta - rhs)).clamp_min(0)
    return beta


@torch.no_grad()
def omp_frame(target: Tensor, vectors: Tensor, *, max_edges: int = 12, threshold: float = 0.2) -> tuple[list[int], Tensor]:
    """Non-negative OMP: indices of selected dictionary rows and their coefficients."""
    y = F.normalize(target, dim=-1)
    units = F.normalize(vectors, dim=-1)
    residual = y.clone()
    support: list[int] = []
    beta = torch.zeros(0)
    for _ in range(max_edges):
        scores = units @ residual
        if support:
            scores[torch.tensor(support)] = float("-inf")
        best = int(scores.argmax())
        if float(scores[best]) < threshold:
            break
        support.append(best)
        design = vectors[torch.tensor(support)].T
        beta = _nnls(design, y)
        keep = (beta > 1e-6).nonzero().flatten()
        support = [support[i] for i in keep.tolist()]; beta = beta[keep]
        residual = y - vectors[torch.tensor(support, dtype=torch.long)].T @ beta if support else y.clone()
    return support, beta


def infer_frames(observations: Tensor, grid_columns: Tensor, grid_fillers: Tensor, vectors: Tensor, *,
                 max_edges: int = 12, threshold: float = 0.2) -> list[InferredFrame]:
    """Infer one frame per new concept from its observations (words × k × d)."""
    frames = []
    for rows in observations:
        support, beta = omp_frame(rows.mean(0), vectors, max_edges=max_edges, threshold=threshold)
        frames.append(InferredFrame([int(grid_columns[i]) for i in support], [int(grid_fillers[i]) for i in support],
                                    [float(b) for b in beta]))
    return frames


def frame_scores(predicted: set[tuple], gold: set[tuple]) -> dict[str, float]:
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "predicted": len(predicted), "gold": len(gold)}


def compose_frame(bind: callable, columns: Sequence[int], fillers: Sequence[int], *, weights: Sequence[float] | None = None,
                  project: callable | None = None, dimension: int) -> Tensor:
    """Zero-shot composition of a frame (uniform masses unless `weights`)."""
    if not len(columns):
        return torch.zeros(dimension)
    with torch.no_grad():
        vectors = bind(torch.tensor(list(columns)), torch.tensor(list(fillers)))
        w = torch.ones(len(columns)) if weights is None else torch.tensor(list(weights), dtype=vectors.dtype)
        row = F.normalize((w[:, None] * vectors).sum(0), dim=-1)
        return project(row[None])[0] if project is not None else row
