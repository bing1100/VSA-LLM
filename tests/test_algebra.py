import pytest
import torch

from vsa_embed.algebra import HRRAlgebra, MAPAlgebra, UnitaryHRRAlgebra, naive_circular_convolution
from vsa_embed.atomics import random_hypervectors


@pytest.mark.parametrize("dimension", [7, 8, 17])
def test_fft_hrr_matches_naive_convolution(dimension: int) -> None:
    generator = torch.Generator().manual_seed(3)
    left = torch.randn(2, dimension, generator=generator, dtype=torch.float64)
    right = torch.randn(2, dimension, generator=generator, dtype=torch.float64)
    expected = naive_circular_convolution(left, right)
    actual = HRRAlgebra().bind(left, right)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_fft_hrr_gradient_matches_naive() -> None:
    generator = torch.Generator().manual_seed(9)
    left = torch.randn(2, 9, generator=generator, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 9, generator=generator, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 9, generator=generator, dtype=torch.float64)
    fft_loss = (HRRAlgebra().bind(left, right) * weights).sum()
    fft_grad = torch.autograd.grad(fft_loss, (left, right))
    naive_loss = (naive_circular_convolution(left, right) * weights).sum()
    naive_grad = torch.autograd.grad(naive_loss, (left, right))
    torch.testing.assert_close(fft_grad[0], naive_grad[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(fft_grad[1], naive_grad[1], rtol=1e-10, atol=1e-10)


def test_unitary_role_round_trip_is_exact_up_to_numeric_error() -> None:
    generator = torch.Generator().manual_seed(12)
    role = random_hypervectors(4, 128, unitary=True, generator=generator, dtype=torch.float64)
    filler = random_hypervectors(4, 128, generator=generator, dtype=torch.float64)
    algebra = UnitaryHRRAlgebra()
    recovered = algebra.unbind(algebra.bind(role, filler), role)
    torch.testing.assert_close(recovered, filler, rtol=1e-9, atol=1e-9)


def test_bundle_normalizes_rows_not_columns() -> None:
    vectors = torch.tensor([[[3.0, 4.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])
    bundled = HRRAlgebra().bundle(vectors)
    torch.testing.assert_close(torch.linalg.vector_norm(bundled, dim=-1), torch.ones(2))


def test_weight_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="weights"):
        MAPAlgebra().bundle(torch.ones(2, 3, 4), torch.ones(2, 4))


def test_map_bipolar_role_is_its_own_inverse() -> None:
    role = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    filler = torch.tensor([[0.5, 0.25, -0.5, 1.0]])
    algebra = MAPAlgebra()
    torch.testing.assert_close(algebra.unbind(algebra.bind(role, filler), role), filler)
