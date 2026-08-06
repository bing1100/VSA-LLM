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
