import torch

from vsa_embed.factorization import OntologyFactorizer, fit_factorizer, geometry_metrics, nearest_recipe_predictions


def test_factorizer_shapes_gradients_and_missing_roles() -> None:
    recipes = torch.tensor([[0, 2], [1, -1], [0, 3]])
    model = OntologyFactorizer(4, 2, 8, 5)
    output = model(recipes)
    assert output.shape == (3, 5)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fit_uses_only_declared_training_rows() -> None:
    torch.manual_seed(4)
    recipes = torch.tensor([[0, 2], [1, 2], [0, 3], [1, 3]])
    targets = torch.randn(4, 6)
    altered = targets.clone(); altered[3] = 10_000
    train = torch.tensor([0, 1, 2])
    torch.manual_seed(9); first = OntologyFactorizer(4, 2, 8, 6)
    fit_factorizer(first, recipes, targets, train, steps=10)
    torch.manual_seed(9); second = OntologyFactorizer(4, 2, 8, 6)
    fit_factorizer(second, recipes, altered, train, steps=10)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_fit_reduces_training_objective() -> None:
    torch.manual_seed(2)
    recipes = torch.tensor([[0, 3], [1, 3], [2, 4], [0, 4], [1, 5], [2, 5]])
    teacher = OntologyFactorizer(6, 2, 10, 7)
    targets = teacher(recipes).detach()
    torch.manual_seed(3); student = OntologyFactorizer(6, 2, 10, 7)
    result = fit_factorizer(student, recipes, targets, torch.arange(5), steps=80)
    assert result.final_loss < result.initial_loss * 0.2


def test_nearest_recipe_and_geometry_metrics() -> None:
    recipes = torch.tensor([[0, 2], [1, 3], [0, 2]])
    targets = torch.eye(3)
    prediction = nearest_recipe_predictions(recipes, targets, torch.tensor([0, 1]), torch.tensor([2]))
    assert torch.equal(prediction, targets[[0]])
    metrics = geometry_metrics(targets, targets, k=1)
    assert metrics["row_cosine"] == 1.0
    assert metrics["knn_overlap"] == 1.0