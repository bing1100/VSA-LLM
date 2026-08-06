"""Relation transfer models for frozen host representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relations import create_relation_transform
from .residual_relations import ResidualHRRRelation, create_residual_relation


@dataclass(frozen=True)
class HostEdges:
    sources: Tensor
    targets: Tensor
    relation_ids: Tensor
    source_node_ids: Tensor | None = None
    target_node_ids: Tensor | None = None

    def subset(self, indices: list[int] | Tensor) -> "HostEdges":
        index = torch.as_tensor(indices, dtype=torch.long)
        return HostEdges(
            self.sources[index], self.targets[index], self.relation_ids[index],
            None if self.source_node_ids is None else self.source_node_ids[index],
            None if self.target_node_ids is None else self.target_node_ids[index],
        )


class HostRelationModel(nn.Module):
    def __init__(
        self, relation_count: int, dimension: int, family: str, *, rank: int = 8,
        max_residual_scale: float = 0.25, max_log_basis_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.family = family
        residual_families = {
            "residual_hrr", "offset_residual_hrr", "gated_offset_residual_hrr",
            "basis_offset_residual_hrr",
            "basis_offset_diagonal_control",
        }
        self.residual_transform: ResidualHRRRelation | None = (
            create_residual_relation(
                family, relation_count, dimension, max_residual_scale=max_residual_scale,
                max_log_basis_scale=max_log_basis_scale,
            ) if family in residual_families else None
        )
        self.transform = (
            create_relation_transform("additive" if family == "offset" else family,
                                      relation_count, dimension, rank=rank)
            if self.residual_transform is None else None
        )
        self.offset = nn.Parameter(torch.zeros(relation_count, dimension)) if family == "offset" else None

    @property
    def relation_count(self) -> int:
        transform = self.residual_transform if self.residual_transform is not None else self.transform
        assert transform is not None
        return transform.relation_count

    def forward(self, sources: Tensor, relation_ids: Tensor) -> Tensor:
        if self.residual_transform is not None:
            output = self.residual_transform(sources, relation_ids)
        else:
            assert self.transform is not None
            output = self.transform(relation_ids, sources)
        if self.offset is not None: output = output + self.offset[relation_ids]
        return F.normalize(output, dim=-1)

    def diagnostics(self, sources: Tensor, relation_ids: Tensor) -> dict[str, float]:
        if self.residual_transform is None:
            return {}
        return self.residual_transform.diagnostics(sources, relation_ids)

    def regularization(
        self, sources: Tensor, relation_ids: Tensor, *, correction_weight: float = 0.0,
        offset_weight: float = 0.0, basis_weight: float = 0.0,
    ) -> Tensor:
        if self.residual_transform is None:
            return sources.new_zeros(())
        return self.residual_transform.regularization(
            sources, relation_ids, correction_weight=correction_weight,
            offset_weight=offset_weight, basis_weight=basis_weight,
        )


def fit_host_relations(
    model: HostRelationModel, train: HostEdges, *, steps: int, learning_rate: float,
    cosine_weight: float, correction_weight: float = 0.0, offset_weight: float = 0.0,
    basis_weight: float = 0.0, rank_weight: float = 0.0,
    rank_temperature: float = 0.07,
) -> tuple[float, float]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        loss = float(1 - F.cosine_similarity(model(train.sources, train.relation_ids), train.targets).mean())
        return loss, loss
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    def objective() -> Tensor:
        prediction = model(train.sources, train.relation_ids)
        data_loss = F.mse_loss(prediction, train.targets) + cosine_weight * (
            1 - F.cosine_similarity(prediction, train.targets).mean()
        )
        if rank_weight:
            if rank_temperature <= 0:
                raise ValueError("rank_temperature must be positive")
            scores = F.normalize(prediction, dim=-1) @ F.normalize(train.targets, dim=-1).T
            scores = scores / rank_temperature
            if train.source_node_ids is None:
                positives = torch.eye(len(train.sources), dtype=torch.bool, device=scores.device)
            else:
                positives = (
                    (train.source_node_ids[:, None] == train.source_node_ids[None, :])
                    & (train.relation_ids[:, None] == train.relation_ids[None, :])
                )
            positive_scores = scores.masked_fill(~positives, float("-inf"))
            rank_loss = -(torch.logsumexp(positive_scores, dim=1) - torch.logsumexp(scores, dim=1)).mean()
            data_loss = data_loss + rank_weight * rank_loss
        return data_loss + model.regularization(
            train.sources, train.relation_ids, correction_weight=correction_weight,
            offset_weight=offset_weight, basis_weight=basis_weight,
        )
    with torch.no_grad(): initial = float(objective())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True); loss = objective(); loss.backward(); optimizer.step()
    with torch.no_grad(): final = float(objective())
    return initial, final


def host_relation_metrics(model: HostRelationModel, data: HostEdges) -> dict[str, float]:
    with torch.no_grad():
        prediction = model(data.sources, data.relation_ids)
        cosine = F.cosine_similarity(prediction, data.targets, dim=-1)
        candidates = torch.stack([
            model(data.sources, torch.full_like(data.relation_ids, relation))
            for relation in range(model.relation_count)
        ], dim=1)
        scores = F.cosine_similarity(candidates, data.targets[:, None, :], dim=-1)
        return {
            "cosine": float(cosine.mean()),
            "mse": float(F.mse_loss(prediction, data.targets)),
            "relation_accuracy": float((scores.argmax(1) == data.relation_ids).float().mean()),
        }


def prediction_metrics(prediction: Tensor, targets: Tensor) -> dict[str, float]:
    return {
        "cosine": float(F.cosine_similarity(prediction, targets, dim=-1).mean()),
        "mse": float(F.mse_loss(prediction, targets)),
    }


def target_retrieval_metrics(
    prediction: Tensor, target_node_ids: Tensor, all_anchors: Tensor,
    candidate_node_ids: Tensor | None = None,
) -> dict[str, float]:
    """Rank the exact target among unique held-out target nodes."""
    candidates = (target_node_ids if candidate_node_ids is None else candidate_node_ids).unique(sorted=True)
    scores = F.normalize(prediction, dim=-1) @ F.normalize(all_anchors[candidates], dim=-1).T
    matches = candidates[None, :] == target_node_ids[:, None]
    positive = scores.masked_fill(~matches, float("-inf")).max(1).values
    ranks = 1 + (scores > positive[:, None]).sum(1)
    return {
        "target_mrr": float((1 / ranks.float()).mean()),
        "target_recall_at_10": float((ranks <= 10).float().mean()),
        "target_candidates": int(candidates.numel()),
    }


def multi_positive_target_retrieval_metrics(
    prediction: Tensor, source_node_ids: Tensor, target_node_ids: Tensor,
    relation_ids: Tensor, all_anchors: Tensor,
) -> dict[str, float]:
    """Rank any target observed for the same held-out (source, relation) as positive."""
    candidates = target_node_ids.unique(sorted=True)
    scores = F.normalize(prediction, dim=-1) @ F.normalize(all_anchors[candidates], dim=-1).T
    positive_targets = (
        (source_node_ids[:, None] == source_node_ids[None, :])
        & (relation_ids[:, None] == relation_ids[None, :])
    )
    candidate_matches = target_node_ids[:, None] == candidates[None, :]
    positives = positive_targets.to(torch.float32) @ candidate_matches.to(torch.float32) > 0
    positive_score = scores.masked_fill(~positives, float("-inf")).max(1).values
    ranks = 1 + (scores > positive_score[:, None]).sum(1)
    return {
        "distribution_mrr": float((1 / ranks.float()).mean()),
        "distribution_recall_at_10": float((ranks <= 10).float().mean()),
        "mean_positive_targets": float(positives.sum(1).float().mean()),
    }


def source_neighborhood_metrics(
    prediction: Tensor, source_node_ids: Tensor, target_node_ids: Tensor,
    relation_ids: Tensor, all_anchors: Tensor, *, temperature: float = 0.07,
) -> dict[str, float]:
    """Evaluate one set-valued retrieval query per unique `(source, relation)`."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    candidates = target_node_ids.unique(sorted=True)
    keys = torch.stack((source_node_ids, relation_ids), dim=1)
    unique_keys = torch.unique(keys, dim=0)
    reciprocal_ranks, recalls, hits, nlls, centroid_cosines, sizes = [], [], [], [], [], []
    candidate_anchors = F.normalize(all_anchors[candidates], dim=-1)
    for key in unique_keys:
        selected = (keys == key).all(dim=1)
        query_prediction = F.normalize(prediction[selected].mean(0), dim=-1)
        positives_ids = target_node_ids[selected].unique()
        positives = (candidates[:, None] == positives_ids[None, :]).any(1)
        scores = candidate_anchors @ query_prediction
        positive_best = scores.masked_fill(~positives, float("-inf")).max()
        rank = 1 + (scores > positive_best).sum()
        top = scores.topk(min(10, len(candidates))).indices
        found = positives[top].sum()
        log_probs = F.log_softmax(scores / temperature, dim=0)
        reciprocal_ranks.append(1 / rank.float())
        recalls.append(found.float() / positives.sum())
        hits.append((found > 0).float())
        nlls.append(-torch.logsumexp(log_probs[positives], dim=0))
        centroid = F.normalize(all_anchors[positives_ids].mean(0), dim=-1)
        centroid_cosines.append(query_prediction @ centroid)
        sizes.append(positives.sum().float())
    mean = lambda values: float(torch.stack(values).mean())
    return {
        "query_mrr": mean(reciprocal_ranks), "set_recall_at_10": mean(recalls),
        "set_hit_at_10": mean(hits), "set_mass_nll": mean(nlls),
        "target_centroid_cosine": mean(centroid_cosines),
        "mean_targets_per_query": mean(sizes), "queries": int(len(unique_keys)),
        "target_candidates": int(len(candidates)),
    }