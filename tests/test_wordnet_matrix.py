import torch
from torch import nn

from vsa_embed.experiments.wordnet_matrix import Concept, encode_recipes, immutable_embedding_rows, supported_holdout


def test_immutable_embedding_rows_clones_without_mutation() -> None:
    embedding = nn.Embedding.from_pretrained(torch.arange(20, dtype=torch.float32).reshape(5, 4), freeze=False)
    before = embedding.weight.detach().clone()
    rows = immutable_embedding_rows(embedding, torch.tensor([1, 3]))
    rows.zero_()
    assert torch.equal(embedding.weight, before)


def test_recipe_encoding_is_stable_and_shared() -> None:
    concepts = [
        Concept("a.n.01", "a", 1, ("type:noun", "parent:x"), 2),
        Concept("b.n.01", "b", 2, ("type:noun", "parent:y"), 1),
    ]
    recipes, values = encode_recipes(concepts)
    assert values == ["parent:x", "parent:y", "type:noun"]
    assert recipes[:, 0].unique().numel() == 1


def test_supported_holdout_preserves_every_role_value() -> None:
    recipes = torch.tensor([[0, 2], [0, 2], [0, 3], [1, 2], [1, 3], [1, 3]])
    train, test = supported_holdout(recipes, 2, 7)
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert set(recipes[test].flatten().tolist()) <= set(recipes[train].flatten().tolist())
    assert {tuple(row) for row in recipes[train].tolist()}.isdisjoint({tuple(row) for row in recipes[test].tolist()})


def test_fixture_recipes_are_unique() -> None:
    concepts = [
        Concept("a.n.01", "a", 1, ("type:noun", "parent:x"), 2),
        Concept("b.n.01", "b", 2, ("type:noun", "parent:y"), 1),
    ]
    recipes, _ = encode_recipes(concepts)
    assert recipes.unique(dim=0).shape[0] == recipes.shape[0]