"""Atomic hypervector generation."""

from __future__ import annotations

import torch
from torch import Tensor

from .algebra import UnitaryHRRAlgebra, normalize


def random_hypervectors(
    count: int,
    dimension: int,
    *,
    distribution: str = "normal",
    unitary: bool = False,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create independent unit-norm atomic vectors."""
    if count <= 0 or dimension <= 1:
        raise ValueError("count must be positive and dimension must exceed one")
    if distribution == "normal":
        vectors = torch.randn(count, dimension, generator=generator, device=device, dtype=dtype)
    elif distribution == "bipolar":
        bits = torch.randint(0, 2, (count, dimension), generator=generator, device=device)
        vectors = bits.to(dtype).mul_(2).sub_(1)
    else:
        raise ValueError("distribution must be 'normal' or 'bipolar'")
    vectors = normalize(vectors)
    return UnitaryHRRAlgebra.make_role(vectors) if unitary else vectors


def correlated_hypervectors(
    count: int,
    dimension: int,
    correlation: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate atomics with approximately constant positive pairwise cosine.

    Before final normalization, each vector is ``sqrt(rho) * common +
    sqrt(1-rho) * independent``. In high dimension the expected pairwise
    cosine approaches ``rho``. This is a controlled stressor, not a model of
    every semantic embedding covariance structure.
    """
    if not 0.0 <= correlation < 1.0:
        raise ValueError("correlation must satisfy 0 <= correlation < 1")
    independent = torch.randn(count, dimension, generator=generator, device=device, dtype=dtype)
    if correlation == 0.0:
        return normalize(independent)
    common = torch.randn(1, dimension, generator=generator, device=device, dtype=dtype)
    vectors = correlation**0.5 * common + (1.0 - correlation) ** 0.5 * independent
    return normalize(vectors)
