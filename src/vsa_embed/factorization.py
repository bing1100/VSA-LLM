"""Ontology-constrained factorization of embedding matrices."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .algebra import BindingAlgebra, UnitaryHRRAlgebra, create_algebra, normalize


class OntologyFactorizer(nn.Module):
    """Compose typed concept recipes and align them to a host embedding space.

    ``recipes[n, r]`` stores the atomic value used by node ``n`` in role ``r``.
    A value of ``-1`` denotes a missing role. Target rows are never parameters,
    which permits strict synthesis of nodes omitted from the fitting loss.
    """

    def __init__(
        self,
        value_count: int,
        role_count: int,
        vsa_dimension: int,
        host_dimension: int,
        *,
        algebra: str = "real_hrr",
        typed: bool = True,
    ) -> None:
        super().__init__()
        if min(value_count, role_count, vsa_dimension, host_dimension) < 1:
            raise ValueError("all dimensions and counts must be positive")
        self.algebra_name = algebra
        self.algebra: BindingAlgebra = create_algebra(algebra)
        self.typed = typed
        self.values = nn.Parameter(torch.randn(value_count, vsa_dimension) / vsa_dimension**0.5)
        self.roles = nn.Parameter(torch.randn(role_count, vsa_dimension) / vsa_dimension**0.5)
        self.alignment = nn.Linear(vsa_dimension, host_dimension, bias=True)

    def role_vectors(self) -> Tensor:
        roles = normalize(self.roles)
        if self.algebra_name == "unitary_hrr":
            roles = UnitaryHRRAlgebra.make_role(roles)
        elif self.algebra_name == "map":
            # Self-inverse bipolar roles make MAP composition identifiable.
            roles = roles.sign().detach() + roles - roles.detach()
        return roles

    def compose(self, recipes: Tensor) -> Tensor:
        if recipes.ndim != 2 or recipes.shape[1] != self.roles.shape[0]:
            raise ValueError("recipes must have shape (nodes, role_count)")
        mask = recipes.ge(0)
        safe = recipes.clamp_min(0)
        fillers = normalize(self.values)[safe]
        if self.typed:
            roles = self.role_vectors().unsqueeze(0).expand(recipes.shape[0], -1, -1)
            terms = self.algebra.bind(roles, fillers)
        else:
            terms = fillers
        terms = terms * mask.unsqueeze(-1)
        return normalize(terms.sum(dim=1))

    def forward(self, recipes: Tensor) -> Tensor:
        return self.alignment(self.compose(recipes))


@dataclass(frozen=True)
class FitResult:
    initial_loss: float
    final_loss: float
    steps: int


def fit_factorizer(
    model: OntologyFactorizer,
    recipes: Tensor,
    targets: Tensor,
    train_indices: Tensor,
    *,
    steps: int = 500,
    learning_rate: float = 0.03,
    cosine_weight: float = 1.0,
) -> FitResult:
    """Fit only rows named by ``train_indices`` and return loss diagnostics."""
    if train_indices.ndim != 1 or train_indices.numel() == 0:
        raise ValueError("train_indices must be a non-empty vector")
    if targets.shape[0] != recipes.shape[0]:
        raise ValueError("recipes and targets must contain the same nodes")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def objective() -> Tensor:
        predicted = model(recipes[train_indices])
        expected = targets[train_indices]
        mse = F.mse_loss(predicted, expected)
        cosine = 1 - F.cosine_similarity(predicted, expected, dim=-1).mean()
        return mse + cosine_weight * cosine

    with torch.no_grad():
        initial = float(objective())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(objective())
    return FitResult(initial, final, steps)


def nearest_recipe_predictions(
    recipes: Tensor, targets: Tensor, train_indices: Tensor, query_indices: Tensor
) -> Tensor:
    """Copy the host row of the training concept with maximal recipe overlap."""
    train = recipes[train_indices]
    query = recipes[query_indices]
    valid = query.unsqueeze(1).ge(0) & train.unsqueeze(0).ge(0)
    overlap = ((query.unsqueeze(1) == train.unsqueeze(0)) & valid).float().sum(dim=-1)
    return targets[train_indices[overlap.argmax(dim=1)]]


def geometry_metrics(predicted: Tensor, target: Tensor, *, k: int = 5) -> dict[str, float]:
    """Row reconstruction and within-split neighborhood preservation metrics."""
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must be equal 2-D matrices")
    n = predicted.shape[0]
    effective_k = min(k, n - 1)
    cosine = F.cosine_similarity(predicted, target, dim=-1)
    metrics = {
        "row_cosine": float(cosine.mean()),
        "row_mse": float(F.mse_loss(predicted, target)),
    }
    if effective_k < 1:
        metrics["knn_overlap"] = 1.0
        return metrics
    pred_sim = F.normalize(predicted, dim=-1) @ F.normalize(predicted, dim=-1).T
    true_sim = F.normalize(target, dim=-1) @ F.normalize(target, dim=-1).T
    pred_sim.fill_diagonal_(-torch.inf)
    true_sim.fill_diagonal_(-torch.inf)
    pred_knn = pred_sim.topk(effective_k, dim=1).indices
    true_knn = true_sim.topk(effective_k, dim=1).indices
    overlap = (pred_knn.unsqueeze(-1) == true_knn.unsqueeze(-2)).any(dim=-1).float().mean()
    metrics["knn_overlap"] = float(overlap)
    return metrics