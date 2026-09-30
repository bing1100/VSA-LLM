"""Synthetic global-local relational factorization data and models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relations import RelationTransform, create_relation_transform


@dataclass(frozen=True)
class RelationalDataset:
    sources: Tensor
    relation_ids: Tensor
    edge_features: Tensor
    concept_features: Tensor
    targets: Tensor
    true_weights: Tensor

    def subset(self, indices: Tensor) -> "RelationalDataset":
        return RelationalDataset(*(field[indices] for field in (
            self.sources, self.relation_ids, self.edge_features,
            self.concept_features, self.targets, self.true_weights,
        )))

    def __len__(self) -> int:
        return self.sources.shape[0]


class GlobalLocalRelationalModel(nn.Module):
    """Global relation transform plus transferable salience and residual heads."""

    def __init__(
        self, relation_count: int, dimension: int, *, family: str,
        rank: int = 8, edge_feature_dimension: int = 4,
        concept_feature_dimension: int = 8, residual_dimension: int = 0,
    ) -> None:
        super().__init__()
        self.transform = create_relation_transform(family, relation_count, dimension, rank=rank)
        self.salience = nn.Linear(edge_feature_dimension, relation_count, bias=True)
        self.residual_dimension = residual_dimension
        if residual_dimension:
            self.residual_encoder = nn.Linear(concept_feature_dimension, residual_dimension, bias=False)
            self.residual_basis = nn.Parameter(torch.randn(residual_dimension, dimension) / dimension**0.5)
        else:
            self.register_parameter("residual_basis", None)

    def components(self, data: RelationalDataset) -> tuple[Tensor, Tensor, Tensor]:
        transformed = self.transform(data.relation_ids, data.sources)
        weights = self.weights_for_relations(data.edge_features, data.relation_ids)
        if self.residual_dimension:
            residual = self.residual_encoder(data.concept_features) @ self.residual_basis
        else:
            residual = torch.zeros_like(transformed)
        return transformed, weights, residual

    def weights_for_relations(self, edge_features: Tensor, relation_ids: Tensor) -> Tensor:
        all_weights = torch.sigmoid(self.salience(edge_features))
        return all_weights.gather(1, relation_ids[:, None]).squeeze(1)

    def forward(self, data: RelationalDataset) -> Tensor:
        transformed, weights, residual = self.components(data)
        return weights[:, None] * transformed + residual

    def parameter_groups(self) -> dict[str, int]:
        return {
            "global_relation": sum(p.numel() for p in self.transform.parameters()),
            "global_salience": sum(p.numel() for p in self.salience.parameters()),
            "global_residual": sum(p.numel() for n, p in self.named_parameters() if n.startswith("residual")),
            "edge_specific": 0,
            "concept_specific": 0,
        }


def make_synthetic_relations(
    *, family: str, relation_count: int, dimension: int, train_per_relation: int,
    test_per_relation: int, rank: int = 8, edge_feature_dimension: int = 4,
    concept_feature_dimension: int = 8, residual_dimension: int = 0,
    noise_std: float = 0.0, seed: int = 0,
) -> tuple[RelationalDataset, RelationalDataset, RelationTransform]:
    """Generate nested, relation-balanced train/test edges from a hidden teacher."""
    g = torch.Generator().manual_seed(seed)
    total_per_relation = train_per_relation + test_per_relation
    relation_ids = torch.arange(relation_count).repeat_interleave(total_per_relation)
    n = relation_ids.numel()
    sources = F.normalize(torch.randn(n, dimension, generator=g), dim=-1)
    edge_features = torch.randn(n, edge_feature_dimension, generator=g)
    concept_features = torch.randn(n, concept_feature_dimension, generator=g)
    # Teacher initialization draws from the global RNG; fork it so callers' streams are untouched.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 10_000)
        teacher = create_relation_transform(family, relation_count, dimension, rank=rank)
    # Learners start near identity where useful, but synthetic teachers must be
    # non-degenerate and relation-specific or the identity control can explain them.
    with torch.no_grad():
        if family in {"map", "diagonal"}:
            parameter = teacher.roles if family == "map" else teacher.diagonal
            parameter.copy_(0.25 + 1.5 * torch.rand(parameter.shape, generator=g))
        elif family == "low_rank":
            teacher.left.mul_(2.0)
            teacher.right.mul_(2.0)
    teacher.eval()
    salience_parameters = torch.randn(relation_count, edge_feature_dimension, generator=g)
    true_weights = torch.sigmoid((edge_features * salience_parameters[relation_ids]).sum(-1))
    if residual_dimension:
        encoder = torch.randn(concept_feature_dimension, residual_dimension, generator=g) / concept_feature_dimension**0.5
        basis = torch.randn(residual_dimension, dimension, generator=g) / dimension**0.5
        residual = concept_features @ encoder @ basis
    else:
        residual = torch.zeros_like(sources)
    with torch.no_grad():
        targets = true_weights[:, None] * teacher(relation_ids, sources) + residual
        targets += noise_std * torch.randn(targets.shape, generator=g)
    train_mask = torch.zeros(n, dtype=torch.bool)
    for relation in range(relation_count):
        start = relation * total_per_relation
        train_mask[start:start + train_per_relation] = True
    train = RelationalDataset(sources[train_mask], relation_ids[train_mask], edge_features[train_mask],
                              concept_features[train_mask], targets[train_mask], true_weights[train_mask])
    test = RelationalDataset(sources[~train_mask], relation_ids[~train_mask], edge_features[~train_mask],
                             concept_features[~train_mask], targets[~train_mask], true_weights[~train_mask])
    return train, test, teacher


@dataclass(frozen=True)
class GlobalLocalFitResult:
    initial_loss: float
    final_loss: float
    threshold_step: int | None
    steps: int


def fit_global_local(
    model: GlobalLocalRelationalModel, train: RelationalDataset, *, steps: int = 300,
    learning_rate: float = 0.02, weight_supervision: float = 0.1,
    threshold: float = 0.02,
) -> GlobalLocalFitResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def objective() -> Tensor:
        prediction = model(train)
        _, weights, _ = model.components(train)
        return F.mse_loss(prediction, train.targets) + weight_supervision * F.mse_loss(weights, train.true_weights)

    with torch.no_grad(): initial = float(objective())
    threshold_step = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(); loss.backward(); optimizer.step()
        if threshold_step is None and float(loss.detach()) <= threshold:
            threshold_step = step
    with torch.no_grad(): final = float(objective())
    return GlobalLocalFitResult(initial, final, threshold_step, steps)


def relational_metrics(model: GlobalLocalRelationalModel, data: RelationalDataset) -> dict[str, float]:
    with torch.no_grad():
        prediction = model(data)
        _, weights, residual = model.components(data)
        candidates = []
        for relation in range(model.transform.relation_count):
            candidate_ids = torch.full_like(data.relation_ids, relation)
            candidate_weights = model.weights_for_relations(data.edge_features, candidate_ids)
            candidates.append(candidate_weights[:, None] * model.transform(candidate_ids, data.sources) + residual)
        candidates = torch.stack(candidates, dim=1)
        relation_accuracy = (candidates.sub(data.targets[:, None]).square().mean(-1).argmin(1) == data.relation_ids).float().mean()
        wc = torch.stack((weights, data.true_weights)).corrcoef()[0, 1] if len(data) > 1 else torch.tensor(1.0)
        return {
            "mse": float(F.mse_loss(prediction, data.targets)),
            "cosine": float(F.cosine_similarity(prediction, data.targets, dim=-1).mean()),
            "relation_accuracy": float(relation_accuracy),
            "weight_correlation": float(torch.nan_to_num(wc)),
        }


def balanced_prefix(data: RelationalDataset, per_relation: int, relation_count: int) -> RelationalDataset:
    """Select a nested prefix for every relation without touching test examples."""
    indices = []
    for relation in range(relation_count):
        candidates = (data.relation_ids == relation).nonzero().flatten()
        if candidates.numel() < per_relation:
            raise ValueError(f"relation {relation} has fewer than {per_relation} examples")
        indices.append(candidates[:per_relation])
    return data.subset(torch.cat(indices))
