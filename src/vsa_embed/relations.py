"""Globally shared relation transformations for structured embeddings.

**Unbinding** (decision 60, the binding and unbinding program): every family has `unbind(relation_ids, vectors,
method=None)`, an estimate of the filler `x` from `T_r(x)` or from a bundle of bound pairs that contains it. The first
entry of `unbind_methods` is the family's primary method:

| family | primary | other methods |
|---|---|---|
| `hrr`, `hrr_identity` | `correlation` (circular correlation, the involution / adjoint; approximate for non-unitary roles) | `inverse` (regularized FFT division) |
| `unitary_hrr` | `conjugate` (multiplication by the conjugate spectrum: exact) | — |
| `orthogonal` | `transpose` (exact) | — |
| `diagonal` | `inverse` (regularized division) | — |
| `map` | `auto` (self-inverse where the role is bipolar, else regularized division) | `self`, `inverse` |
| `low_rank`, `low_rank_identity`, `low_rank_tied` | `woodbury` (the exact inverse of `I + L R`) | — |
| `translation` | `subtract` (`x = v − t_r`, exact for one pair) | — |
| `additive` (the composer's `untyped`) | none: `unbind()` raises `UnbindingError` | `bundle` (the bundle readout: returns the vector, ignoring the role) |

Regularized divisions use `λ = ridge · mean power` of the relation's spectrum or diagonal (`UNBIND_RIDGE`), so a role
whose spectrum has a near-zero bin does not blow up the noise of a bundle.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .algebra import HRRAlgebra

# Ridge of the regularized divisions (`inverse` of hrr / diagonal / map), relative to the mean power of the relation's
# spectrum or diagonal: λ = UNBIND_RIDGE · mean |R_k|² (≈ 1 % of a typical bin's power).
UNBIND_RIDGE = 1e-2


class UnbindingError(ValueError):
    """A relation family that does not bind (the additive / untyped bundle) has no unbinding."""


class RelationTransform(nn.Module, ABC):
    """Map vectors through a relation selected by integer IDs."""

    family: str
    # Unbinding methods (module docstring); the first is the primary. Empty: no unbinding.
    unbind_methods: tuple[str, ...] = ()

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

    def _unbind_method(self, method: str | None) -> str:
        if not self.unbind_methods:
            raise UnbindingError(f"the {self.family!r} family has no unbinding")
        method = self.unbind_methods[0] if method is None else method
        if method not in self.unbind_methods:
            raise ValueError(f"unbinding method {method!r} is not one of {self.family!r}'s {list(self.unbind_methods)}")
        return method

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        """Estimate the filler `x` of `T_r(x)` (or of a bundle holding it); `method` defaults to the family's primary
        (module docstring)."""
        self._unbind_method(method)
        raise NotImplementedError(f"{type(self).__name__} does not implement unbind")

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
    unbind_methods = ("bundle",)

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return vectors

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        return self.forward(relation_ids, vectors)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        """No binding, so no unbinding: a bundle of fillers keeps no role. Only the explicitly requested `bundle`
        readout is defined — the vector itself, whatever the role (what an untyped store can give back)."""
        if method is None:
            raise UnbindingError("the additive (untyped) family does not bind, so it has no unbinding; pass "
                                 "method='bundle' for the bundle readout, which ignores the role")
        self._unbind_method(method)
        self._validate(relation_ids, vectors)
        return vectors


class HRRRelation(RelationTransform):
    """Relation vectors interpreted as circulant HRR operators.

    The historical `hrr` family starts from random roles (`cos(T x, x) ≈ 0`), while every
    other family starts at the identity. `identity_init=True` (family `hrr_identity`) starts
    from the HRR identity element, a unit impulse whose spectrum is all ones, plus small noise,
    so family and initialization can be separated.
    """

    family = "hrr"
    # Learned HRR roles are not unitary, so correlation (the classic HRR decoder) is approximate; `inverse` divides by
    # the role spectrum (regularized), which undoes the role's spectral colouring at the price of amplifying noise in
    # weak bins.
    unbind_methods = ("correlation", "inverse")

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

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        method = self._unbind_method(method)
        self._validate(relation_ids, vectors)
        roles = self.roles[relation_ids]
        if method == "correlation":
            return self.algebra.unbind(vectors, roles)
        return spectral_inverse(vectors, roles, ridge)


def spectral_inverse(vectors: Tensor, roles: Tensor, ridge: float = UNBIND_RIDGE) -> Tensor:
    """Regularized FFT division `F⁻¹[V · conj(R) / (|R|² + λ)]`, `λ = ridge · mean_k |R_k|²` per role: the least-squares
    inverse of circular convolution by `roles` (equal to correlation when the roles are unitary and `ridge` = 0)."""
    dimension = vectors.shape[-1]
    spectrum = torch.fft.rfft(roles.float())
    power = spectrum.real.square() + spectrum.imag.square()
    damping = float(ridge) * power.mean(-1, keepdim=True)
    out = torch.fft.irfft(torch.fft.rfft(vectors.float()) * spectrum.conj() / (power + damping).clamp_min(1e-12), n=dimension)
    return out.to(vectors.dtype)


def regularized_division(vectors: Tensor, diagonal: Tensor, ridge: float = UNBIND_RIDGE) -> Tensor:
    """`v · d / (d² + λ)`, `λ = ridge · mean d²` per relation: the regularized inverse of an elementwise operator."""
    power = diagonal.square()
    damping = float(ridge) * power.mean(-1, keepdim=True)
    return vectors * diagonal / (power + damping).clamp_min(1e-12)


class MAPRelation(RelationTransform):
    """Relation vectors interpreted as diagonal elementwise operators."""

    family = "map"
    # MAP binding is self-inverse when the role is bipolar (±1); learned roles drift away from ±1, where only division
    # inverts them (`auto` decides per relation).
    unbind_methods = ("auto", "self", "inverse")
    BIPOLAR_TOLERANCE = 1e-6

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.roles = nn.Parameter(torch.ones(relation_count, dimension))

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.roles[relation_ids] * vectors

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        return self.forward(relation_ids, vectors)

    def bipolar(self) -> Tensor:
        """Per relation: whether every coordinate of its role is ±1 (within `BIPOLAR_TOLERANCE`)."""
        return ((self.roles.detach().abs() - 1).abs() <= self.BIPOLAR_TOLERANCE).all(-1)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        method = self._unbind_method(method)
        self._validate(relation_ids, vectors)
        roles = self.roles[relation_ids]
        if method == "self":
            return roles * vectors
        divided = regularized_division(vectors, roles, ridge)
        if method == "inverse":
            return divided
        return torch.where(self.bipolar()[relation_ids].unsqueeze(-1), roles * vectors, divided)


class DiagonalRelation(RelationTransform):
    """Unconstrained learned diagonal operators."""

    family = "diagonal"
    unbind_methods = ("inverse",)

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.diagonal = nn.Parameter(torch.ones(relation_count, dimension))

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.diagonal[relation_ids] * vectors

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        return self.forward(relation_ids, vectors)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        self._unbind_method(method)
        self._validate(relation_ids, vectors)
        return regularized_division(vectors, self.diagonal[relation_ids], ridge)


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

    unbind_methods = ("woodbury",)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        """`(I + L R)⁻¹ v = v − L (I_r + R L)⁻¹ R v` (Woodbury; exact whenever `I + L R` is invertible)."""
        self._unbind_method(method)
        self._validate(relation_ids, vectors)
        left, right = self.left[relation_ids].float(), self.right[relation_ids].float()
        small = torch.eye(self.rank, device=vectors.device) + right @ left
        solved = torch.linalg.solve(small, torch.einsum("...rd,...d->...r", right, vectors.float()).unsqueeze(-1)).squeeze(-1)
        return (vectors.float() - torch.einsum("...dr,...r->...d", left, solved)).to(vectors.dtype)


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

    unbind_methods = ("woodbury",)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        """`(I + U S Uᵀ)⁻¹ v = v − U S (I + Uᵀ U S)⁻¹ Uᵀ v` (Woodbury in the form that stays defined at `S = 0`)."""
        self._unbind_method(method)
        self._validate(relation_ids, vectors)
        basis, scales = self.basis[relation_ids].float(), self.scales[relation_ids].float()
        gram = basis.transpose(-1, -2) @ basis
        small = torch.eye(self.rank, device=vectors.device) + gram * scales.unsqueeze(-2)
        projected = torch.einsum("...dr,...d->...r", basis, vectors.float())
        solved = torch.linalg.solve(small, projected.unsqueeze(-1)).squeeze(-1)
        return (vectors.float() - torch.einsum("...dr,...r->...d", basis, scales * solved)).to(vectors.dtype)


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

    unbind_methods = ("transpose",)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        """The transpose of an orthogonal operator is its inverse (exact)."""
        self._unbind_method(method)
        return self.adjoint(relation_ids, vectors)


class TranslationRelation(RelationTransform):
    """TransE-style translation `T_r(x) = x + t_r` (WP-PQ1 operator ablation, C5tr).

    Affine, not a binding: a bundle `Σ_e (a_e + t_{r_e})` separates into a bag of fillers plus a bag of
    relation offsets, so which filler went with which relation is lost. Offsets start like HRR roles
    (`N(0, 1/d)`, norm ≈ 1, the scale of a normalized atomic), so relations are distinguishable at step 0.
    Not linear, so it is built by `create_composition_operator`, not `create_relation_transform`.
    """

    family = "translation"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        self.offsets = nn.Parameter(torch.randn(relation_count, dimension) / dimension**0.5)

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return vectors + self.offsets[relation_ids]

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        # The linear part of an affine map is the identity (its vector-Jacobian product).
        self._validate(relation_ids, vectors)
        return vectors

    unbind_methods = ("subtract",)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        """`v − t_r`: exact for one translated filler. In a bundle every offset adds to the same bag of fillers, so
        the role read out is the bag minus one offset — which filler went with the relation is not recoverable."""
        self._unbind_method(method)
        self._validate(relation_ids, vectors)
        return vectors - self.offsets[relation_ids]


class UnitaryHRRRelation(RelationTransform):
    """HRR binding with unitary roles: every role's spectrum has unit magnitude, so binding is an exactly
    orthogonal circulant operator (norm-preserving; its inverse is the involution).

    Parametrized by the phases of the interior frequencies, drawn uniformly at random; the DC and (even
    dimension) Nyquist bins must be real, so they carry fixed random signs. `random_fixed:unitary_hrr`
    freezes the phases: a fixed random orthogonal operator per relation (WP-PQ1, C5rf).
    """

    family = "unitary_hrr"

    def __init__(self, relation_count: int, dimension: int) -> None:
        super().__init__(relation_count, dimension)
        bins = dimension // 2 + 1
        real_bins = 2 if dimension % 2 == 0 else 1          # DC, and Nyquist for an even dimension
        self.phases = nn.Parameter((torch.rand(relation_count, bins - real_bins) * 2 - 1) * math.pi)
        self.register_buffer("edge_signs", torch.where(torch.rand(relation_count, real_bins) < 0.5, -1.0, 1.0))
        self.algebra = HRRAlgebra()

    def role_vectors(self) -> Tensor:
        interior = torch.polar(torch.ones_like(self.phases), self.phases)
        dc = self.edge_signs[:, :1].to(interior.dtype)
        parts = [dc, interior] + ([self.edge_signs[:, 1:].to(interior.dtype)] if self.edge_signs.shape[1] > 1 else [])
        return torch.fft.irfft(torch.cat(parts, -1), n=self.dimension)

    def forward(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.algebra.bind(self.role_vectors()[relation_ids], vectors)

    def adjoint(self, relation_ids: Tensor, vectors: Tensor) -> Tensor:
        self._validate(relation_ids, vectors)
        return self.algebra.unbind(vectors, self.role_vectors()[relation_ids])

    # The learned unitary family of decision 60 is this class: trainable phases, unit magnitude, hence exactly
    # invertible by the conjugate spectrum whatever the phases learn (`random_fixed:unitary_hrr` freezes them).
    unbind_methods = ("conjugate",)

    def unbind(self, relation_ids: Tensor, vectors: Tensor, *, method: str | None = None,
               ridge: float = UNBIND_RIDGE) -> Tensor:
        self._unbind_method(method)
        return self.adjoint(relation_ids, vectors)


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
        "unitary_hrr": lambda: UnitaryHRRRelation(relation_count, dimension),
    }
    try:
        return factories[family]()
    except KeyError as error:
        raise ValueError(f"unsupported relation family {family!r}; choose from {sorted(factories)}") from error


# Affine (not linear) operator families: usable as composition operators, kept out of `create_relation_transform`,
# whose families are all linear (the adjoint identity `⟨T x, y⟩ = ⟨x, Tᵀ y⟩` holds for each of them).
AFFINE_FAMILIES = {"translation": TranslationRelation}


def create_composition_operator(family: str, relation_count: int, dimension: int, *, rank: int = 8) -> RelationTransform:
    """The relation operator of a frame composer: a linear family (`create_relation_transform`) or an affine one
    (`AFFINE_FAMILIES`)."""
    if family in AFFINE_FAMILIES:
        return AFFINE_FAMILIES[family](relation_count, dimension)
    return create_relation_transform(family, relation_count, dimension, rank=rank)


def readout_method(transform: RelationTransform) -> str:
    """The unbinding method a probe or a readout uses for `transform`: its primary method, or the `bundle` readout
    for the additive (untyped) family, which has no unbinding."""
    if isinstance(transform, AdditiveRelation):
        return "bundle"
    if not transform.unbind_methods:
        raise UnbindingError(f"the {transform.family!r} family has no unbinding")
    return transform.unbind_methods[0]
