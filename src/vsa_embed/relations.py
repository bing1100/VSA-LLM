"""Globally shared relation transformations for structured embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .algebra import HRRAlgebra


class RelationTransform(nn.Module, ABC):
    """Map vectors through a relation selected by integer IDs."""

    family: str

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__()
        if relation_count < 1 or dimension < 1:
            raise ValueError("relation_count and dimension must be positive")
        self.relation_count = relation_count
        self.dimension = dimension

    def _validate(self, relation_ids: Tensor, vectors: Tensor) -> None:
        if relation_ids.shape != vectors.shape[:-1] or vectors.shape[-1] != self.dimension:
            raise ValueError("relation_ids must match vector leading dimensions")

    @abstractmethod
    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor: ...

    def complexity(self) -> dict[str, int | str]:
        return {"family": self.family, "parameters": sum(p.numel() for p in self.parameters())}


class AdditiveRelation(RelationTransform):
    """Relation-agnostic identity control."""

    family = "additive"

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return vectors


class HRRRelation(RelationTransform):
    """Relation vectors interpreted as circulant HRR operators."""

    family = "hrr"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.roles = nn.Parameter(torch.randn(relation_count, dimension) / dimension**0.5)
        self.algebra = HRRAlgebra()

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.algebra.bind(self.roles[relation_ids], vectors)


class MAPRelation(RelationTransform):
    """Relation vectors interpreted as diagonal elementwise operators."""

    family = "map"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.roles = nn.Parameter(torch.ones(relation_count, dimension))

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.roles[relation_ids] * vectors


class DiagonalRelation(RelationTransform):
    """Unconstrained learned diagonal operators."""

    family = "diagonal"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.diagonal = nn.Parameter(torch.ones(relation_count, dimension))

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.diagonal[relation_ids] * vectors


class LowRankRelation(RelationTransform):
    """Identity plus a relation-specific rank-r update."""

    family = "low_rank"

    def __init__(self, relation_count: int, dimension: int, rank: int = 8) -> None:
        super().__init__(relation_count, dimension)
        if not 1 <= rank <= dimension:
            raise ValueError("rank must be in [1, dimension]")
        self.rank = rank
        scale = dimension**-0.5
        self.left = nn.Parameter(torch.randn(relation_count, dimension, rank) * scale)
        self.right = nn.Parameter(torch.randn(relation_count, rank, dimension) * scale)

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        right = self.right[relation_ids]
        left = self.left[relation_ids]
        hidden = torch.einsum("...rd,...d->...r", right, vectors)
        return vectors + torch.einsum("...dr,...r->...d", left, hidden)


class OrthogonalRelation(RelationTransform):
    """Relation-specific orthogonal operators via differentiable QR."""

    family = "orthogonal"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        eye = torch.eye(dimension).expand(relation_count, -1, -1)
        self.raw = nn.Parameter(eye + 0.01 * torch.randn_like(eye))

    def matrices(self) -> Tensor:
        q, r = torch.linalg.qr(self.raw)
        signs = torch.diagonal(r, dim1=-2, dim2=-1).sign().clamp(min=-1, max=1)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return q * signs.unsqueeze(-2)

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return torch.einsum("...de,...e->...d", self.matrices()[relation_ids], vectors)


def create_relation_transform(
    family: str, relation_count: int, dimension: int, *, rank: int = 8
) -> RelationTransform:
    factories = {
        "additive": lambda: AdditiveRelation(relation_count, dimension),
        "hrr": lambda: HRRRelation(relation_count, dimension),
        "map": lambda: MAPRelation(relation_count, dimension),
        "diagonal": lambda: DiagonalRelation(relation_count, dimension),
        "low_rank": lambda: LowRankRelation(relation_count, dimension, rank),
        "orthogonal": lambda: OrthogonalRelation(relation_count, dimension),
    }
    try:
        return factories[family]()
    except KeyError as error:
        raise ValueError(f"unsupported relation family {family!r}; choose from {sorted(factories)}") from error
