"""Binding algebras with batch-first, differentiable PyTorch operations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import functional as F


def normalize(vectors: Tensor, eps: float = 1e-8) -> Tensor:
    """L2-normalize each vector along its final dimension."""
    return F.normalize(vectors, p=2, dim=-1, eps=eps)


class BindingAlgebra(ABC):
    """Minimal protocol shared by experiment and future model adapters."""

    name: str

    @abstractmethod
    def bind(self, role: Tensor, filler: Tensor) -> Tensor:
        """Bind role and filler vectors, broadcasting leading dimensions."""

    @abstractmethod
    def unbind(self, bound: Tensor, role: Tensor) -> Tensor:
        """Approximately recover a filler from a bound pair or bundle."""

    def bundle(self, vectors: Tensor, weights: Tensor | None = None, *, unit: bool = True) -> Tensor:
        """Weighted superposition over the penultimate (item) dimension."""
        if vectors.ndim < 2:
            raise ValueError("vectors must have shape (..., items, dimension)")
        if weights is not None:
            if weights.shape != vectors.shape[:-1]:
                raise ValueError("weights must match vectors without the final dimension")
            vectors = vectors * weights.unsqueeze(-1)
        result = vectors.sum(dim=-2)
        return normalize(result) if unit else result


def naive_circular_convolution(left: Tensor, right: Tensor) -> Tensor:
    """Reference O(D²) circular convolution used for correctness tests."""
    if left.shape != right.shape:
        raise ValueError("naive convolution requires equal shapes")
    dimension = left.shape[-1]
    return torch.stack(
        [sum(left[..., index] * right[..., (offset - index) % dimension] for index in range(dimension))
         for offset in range(dimension)],
        dim=-1,
    )


class HRRAlgebra(BindingAlgebra):
    """Real-valued HRR using FFT circular convolution and correlation."""

    name = "real_hrr"

    def bind(self, role: Tensor, filler: Tensor) -> Tensor:
        if role.shape[-1] != filler.shape[-1]:
            raise ValueError("role and filler dimensions differ")
        dimension = role.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(role) * torch.fft.rfft(filler), n=dimension)

    def unbind(self, bound: Tensor, role: Tensor) -> Tensor:
        if bound.shape[-1] != role.shape[-1]:
            raise ValueError("bound and role dimensions differ")
        dimension = bound.shape[-1]
        # Circular correlation: multiplication by the conjugated role spectrum.
        return torch.fft.irfft(torch.fft.rfft(bound) * torch.fft.rfft(role).conj(), n=dimension)


class UnitaryHRRAlgebra(HRRAlgebra):
    """HRR backend intended for roles projected to unitary Fourier phase."""

    name = "unitary_hrr"

    @staticmethod
    def make_role(vector: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(vector)
        phase = spectrum / spectrum.abs().clamp_min(1e-8)
        # DC and Nyquist bins must be real for irfft to represent a real signal.
        phase[..., 0] = torch.ones_like(phase[..., 0])
        if vector.shape[-1] % 2 == 0:
            phase[..., -1] = torch.ones_like(phase[..., -1])
        return normalize(torch.fft.irfft(phase, n=vector.shape[-1]))


class MAPAlgebra(BindingAlgebra):
    """Multiply-add-permute style binding using elementwise multiplication."""

    name = "map"

    def bind(self, role: Tensor, filler: Tensor) -> Tensor:
        return role * filler

    def unbind(self, bound: Tensor, role: Tensor) -> Tensor:
        return bound * role


def create_algebra(name: str) -> BindingAlgebra:
    aliases = {
        "real_hrr": HRRAlgebra,
        "unitary_hrr": UnitaryHRRAlgebra,
        # Generalized HRR is reserved until its matrix/block parameters are explicit.
        "map": MAPAlgebra,
    }
    try:
        return aliases[name]()
    except KeyError as error:
        raise ValueError(f"Unsupported algebra {name!r}; choose from {sorted(aliases)}") from error
