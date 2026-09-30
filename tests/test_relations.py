import pytest
import torch

from vsa_embed.relations import OrthogonalRelation, create_relation_transform


@pytest.mark.parametrize("family", ["additive", "hrr", "map", "diagonal", "low_rank", "orthogonal"])
def test_relation_transforms_are_batched_and_differentiable(family: str) -> None:
    transform = create_relation_transform(family, 3, 8, rank=2)
    relation_ids = torch.tensor([0, 1, 2, 1])
    vectors = torch.randn(4, 8, requires_grad=True)
    output = transform(relation_ids, vectors)
    assert output.shape == vectors.shape
    output.square().mean().backward()
    assert vectors.grad is not None
    assert transform.complexity()["family"] == family


def test_orthogonal_relation_preserves_norm() -> None:
    transform = OrthogonalRelation(2, 8)
    vectors = torch.randn(10, 8)
    relation_ids = torch.arange(10) % 2
    actual = transform(relation_ids, vectors)
    assert torch.allclose(actual.norm(dim=-1), vectors.norm(dim=-1), atol=1e-5)


def test_invalid_relation_shapes_fail() -> None:
    transform = create_relation_transform("hrr", 2, 8)
    with pytest.raises(ValueError):
        transform(torch.tensor([0]), torch.randn(2, 8))



@pytest.mark.parametrize("family", [
    "additive", "hrr", "hrr_identity", "map", "diagonal", "low_rank", "low_rank_identity",
    "low_rank_tied", "orthogonal",
])
def test_adjoint_satisfies_the_inner_product_identity(family: str) -> None:
    torch.manual_seed(0)
    transform = create_relation_transform(family, 3, 16, rank=4)
    with torch.no_grad():
        for parameter in transform.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
    ids = torch.tensor([0, 1, 2, 1])
    x, y = torch.randn(4, 16), torch.randn(4, 16)
    left = (transform(ids, x) * y).sum(-1)
    right = (x * transform.adjoint(ids, y)).sum(-1)
    torch.testing.assert_close(left.detach(), right.detach(), atol=1e-4, rtol=1e-4)


def test_residual_adjoint_is_the_jacobian_transpose() -> None:
    from vsa_embed.residual_relations import create_residual_relation
    torch.manual_seed(1)
    transform = create_residual_relation("basis_offset_residual_hrr", 2, 8)
    with torch.no_grad():
        transform.residual_scale_logits.fill_(1.0)
    x, v = torch.randn(3, 8), torch.randn(3, 8)
    ids = torch.tensor([0, 1, 0])
    jacobian = torch.autograd.functional.jacobian(lambda z: transform(z, ids), x)
    expected = torch.einsum("aibj,ai->bj", jacobian, v)
    torch.testing.assert_close(transform.adjoint_at(x, ids, v), expected, atol=1e-5, rtol=1e-4)
