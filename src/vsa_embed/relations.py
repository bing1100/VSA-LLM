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

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        """Apply `T_rᵀ`, so `⟨T_r x, y⟩ = ⟨x, T_rᵀ y⟩`.

        Every family here is linear in its input, so the vector-Jacobian product at any point is
        the adjoint; subclasses override it with a closed form where one is cheap.
        """
        self._validate(relation_ids, vectors)
        probe = torch.zeros_like(vectors, requires_grad=True)
        with torch.enable_grad():
            output = self.forward(relation_ids, probe)
            (result,) = torch.autograd.grad(output, probe, grad_outputs=vectors, create_graph=vectors.requires_grad)
        return result

    def complexity(self) -> dict[str, int | str]:
        return {"family": self.family, "parameters": sum(p.numel() for p in self.parameters())}


class AdditiveRelation(RelationTransform):
    """Relation-agnostic identity control."""

    family = "additive"

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return vectors

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        return self.forward(relation_ids, vectors)


class HRRRelation(RelationTransform):
    """Relation vectors interpreted as circulant HRR operators.

    The historical `hrr` family starts from random roles (`cos(T x, x) ≈ 0`), while every
    other family starts at the identity. `identity_init=True` (family `hrr_identity`) starts
    from the HRR identity element, a unit impulse whose spectrum is all ones, plus small noise,
    so family and initialization can be separated.
    """

    family = "hrr"

    def __init__(self, relation_count: int, dimension: int, *, identity_init: bool = False,
                 init_noise: float = 0.01) -> None:
        super().__init__(relation_count, dimension)
        if identity_init:
            self.family = "hrr_identity"
            roles = init_noise * torch.randn(relation_count, dimension) / dimension**0.5
            roles[:, 0] += 1.0
        else:
            roles = torch.randn(relation_count, dimension) / dimension**0.5
        self.roles = nn.Parameter(roles)
        self.algebra = HRRAlgebra()

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.algebra.bind(self.roles[relation_ids], vectors)

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        # The adjoint of circular convolution is circular correlation (HRR unbinding).
        self._validate(relation_ids, vectors)
        return self.algebra.unbind(vectors, self.roles[relation_ids])


class MAPRelation(RelationTransform):
    """Relation vectors interpreted as diagonal elementwise operators."""

    family = "map"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.roles = nn.Parameter(torch.ones(relation_count, dimension))

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.roles[relation_ids] * vectors

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        return self.forward(relation_ids, vectors)


class DiagonalRelation(RelationTransform):
    """Unconstrained learned diagonal operators."""

    family = "diagonal"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.diagonal = nn.Parameter(torch.ones(relation_count, dimension))

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.diagonal[relation_ids] * vectors

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        return self.forward(relation_ids, vectors)


class LowRankRelation(RelationTransform):
    """Identity plus a relation-specific rank-r update.

    The historical `low_rank` family initializes both factors randomly, so it is not the
    identity at initialization. `identity_init=True` (family `low_rank_identity`) zeroes the
    left factor, which makes `T(x) = x` at step 0 while keeping gradients alive.
    """

    family = "low_rank"

    def __init__(self, relation_count: int, dimension: int, rank: int = 8, *,
                 identity_init: bool = False) -> None:
        super().__init__(relation_count, dimension)
        if not 1 <= rank <= dimension:
            raise ValueError("rank must be in [1, dimension]")
        self.rank = rank
        scale = dimension**-0.5
        left = torch.randn(relation_count, dimension, rank) * scale
        right = torch.randn(relation_count, rank, dimension) * scale
        if identity_init:
            self.family = "low_rank_identity"
            left = torch.zeros_like(left)
        self.left = nn.Parameter(left)
        self.right = nn.Parameter(right)

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        right = self.right[relation_ids]
        left = self.left[relation_ids]
        hidden = torch.einsum("...rd,...d->...r", right, vectors)
        return vectors + torch.einsum("...dr,...r->...d", left, hidden)

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        hidden = torch.einsum("...dr,...d->...r", self.left[relation_ids], vectors)
        return vectors + torch.einsum("...rd,...r->...d", self.right[relation_ids], hidden)


class TiedLowRankRelation(RelationTransform):
    """Identity plus a scaled symmetric update `x + U diag(σ) Uᵀ x` per relation.

    Costs `rank · (dimension + 1)` parameters per relation, so rank 1 matches a `d`-parameter
    family such as `hrr` or `diagonal` to within one scalar; the untied `low_rank` family
    cannot go below `2d`. `σ` starts at zero, so the transform starts at the identity.
    """

    family = "low_rank_tied"

    def __init__(self, relation_count: int, dimension: int, rank: int = 1) -> None:
        super().__init__(relation_count, dimension)
        if not 1 <= rank <= dimension:
            raise ValueError("rank must be in [1, dimension]")
        self.rank = rank
        self.basis = nn.Parameter(torch.randn(relation_count, dimension, rank) * dimension**-0.5)
        self.scales = nn.Parameter(torch.zeros(relation_count, rank))

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        basis = self.basis[relation_ids]
        hidden = torch.einsum("...dr,...d->...r", basis, vectors) * self.scales[relation_ids]
        return vectors + torch.einsum("...dr,...r->...d", basis, hidden)

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        return self.forward(relation_ids, vectors)  # symmetric by construction


def matched_tied_rank(target_parameters: int, relation_count: int, dimension: int) -> int:
    """Largest tied low-rank rank whose parameter count does not exceed the target (min 1)."""
    return max(1, min(dimension, target_parameters // (relation_count * (dimension + 1))))


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

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return torch.einsum("...ed,...e->...d", self.matrices()[relation_ids], vectors)


def create_relation_transform(
    family: str, relation_count: int, dimension: int, *, rank: int = 8
) -> RelationTransform:
    factories = {
        "additive": lambda: AdditiveRelation(relation_count, dimension),
        "hrr": lambda: HRRRelation(relation_count, dimension),
        "hrr_identity": lambda: HRRRelation(relation_count, dimension, identity_init=True),
        "map": lambda: MAPRelation(relation_count, dimension),
        "diagonal": lambda: DiagonalRelation(relation_count, dimension),
        "low_rank": lambda: LowRankRelation(relation_count, dimension, rank),
        "low_rank_identity": lambda: LowRankRelation(relation_count, dimension, rank, identity_init=True),
        "low_rank_tied": lambda: TiedLowRankRelation(relation_count, dimension, rank),
        "orthogonal": lambda: OrthogonalRelation(relation_count, dimension),
    }
    try:
        return factories[family]()
    except KeyError as error:
        raise ValueError(f"unsupported relation family {family!r}; choose from {sorted(factories)}") from error
