"""Relation transfer models for frozen host representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relations import create_relation_transform, matched_tied_rank
from .residual_relations import ResidualHRRRelation, create_residual_relation
from .statistics import positive_ranks


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
    """Relation model over frozen host vectors.

    `low_rank_matched` is the tied low-rank family (`x + U diag(σ) Uᵀ x`) at the largest rank
    whose parameter count does not exceed `match_parameters`, the parameter count of the family
    it is compared with.
    """

    def __init__(
        self, relation_count: int, dimension: int, family: str, *, rank: int = 8,
        max_residual_scale: float = 0.25, max_log_basis_scale: float = 0.25,
        match_parameters: int | None = None,
    ) -> None:
        super().__init__()
        self.family = family
        residual_families = {
            "residual_hrr", "offset_residual_hrr", "gated_offset_residual_hrr",
            "basis_offset_residual_hrr",
            "basis_offset_diagonal_control", "basis_offset_rotated_diagonal_control",
        }
        if family == "low_rank_matched":
            if match_parameters is None:
                raise ValueError("low_rank_matched requires match_parameters")
            rank = matched_tied_rank(match_parameters, relation_count, dimension)
        self.residual_transform: ResidualHRRRelation | None = (
            create_residual_relation(
                family, relation_count, dimension, max_residual_scale=max_residual_scale,
                max_log_basis_scale=max_log_basis_scale,
            ) if family in residual_families else None
        )
        transform_family = {"offset": "additive", "low_rank_matched": "low_rank_tied"}.get(family, family)
        self.transform = (
            create_relation_transform(transform_family, relation_count, dimension, rank=rank)
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


def rank_positive_mask(train: HostEdges, mode: str) -> Tensor:
    """Which training rows count as positives for each query row in the InfoNCE rank loss.

    `source_relation` (historical) marks rows with the same (source, relation); a different
    source that shares the target node is then a negative, so a perfect predictor keeps a loss
    of `log k` on targets of multiplicity `k`. `target_set` marks every row whose target node is
    among the query's (source, relation) targets, so duplicate targets are never negatives.
    """
    if train.source_node_ids is None:
        return torch.eye(len(train.sources), dtype=torch.bool, device=train.sources.device)
    same_query = (
        (train.source_node_ids[:, None] == train.source_node_ids[None, :])
        & (train.relation_ids[:, None] == train.relation_ids[None, :])
    )
    if mode == "source_relation":
        return same_query
    if mode != "target_set":
        raise ValueError("rank_positives must be 'source_relation' or 'target_set'")
    if train.target_node_ids is None:
        raise ValueError("target_set positives require target_node_ids")
    same_target = train.target_node_ids[:, None] == train.target_node_ids[None, :]
    return (same_query.float() @ same_target.float()) > 0


def info_nce_rank_loss(prediction: Tensor, targets: Tensor, positives: Tensor, temperature: float) -> Tensor:
    """Multi-positive InfoNCE of each prediction against every training target."""
    if temperature <= 0:
        raise ValueError("rank_temperature must be positive")
    scores = F.normalize(prediction, dim=-1) @ F.normalize(targets, dim=-1).T
    scores = scores / temperature
    positive_scores = scores.masked_fill(~positives, float("-inf"))
    return -(torch.logsumexp(positive_scores, dim=1) - torch.logsumexp(scores, dim=1)).mean()


def fit_host_relations(
    model: HostRelationModel, train: HostEdges, *, steps: int, learning_rate: float,
    cosine_weight: float, correction_weight: float = 0.0, offset_weight: float = 0.0,
    basis_weight: float = 0.0, rank_weight: float = 0.0,
    rank_temperature: float = 0.07, rank_positives: str = "source_relation",
    parameter_free_objective: bool = False,
) -> tuple[float, float]:
    """Fit and return (initial, final) objective.

    For a parameter-free model the historical return value is `1 − cosine`, not the objective;
    `parameter_free_objective=True` returns the objective itself so losses are comparable.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters and not parameter_free_objective:
        loss = float(1 - F.cosine_similarity(model(train.sources, train.relation_ids), train.targets).mean())
        return loss, loss
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    def objective() -> Tensor:
        prediction = model(train.sources, train.relation_ids)
        data_loss = F.mse_loss(prediction, train.targets) + cosine_weight * (
            1 - F.cosine_similarity(prediction, train.targets).mean()
        )
        if rank_weight:
            rank_loss = info_nce_rank_loss(
                prediction, train.targets, rank_positive_mask(train, rank_positives), rank_temperature,
            )
            data_loss = data_loss + rank_weight * rank_loss
        return data_loss + model.regularization(
            train.sources, train.relation_ids, correction_weight=correction_weight,
            offset_weight=offset_weight, basis_weight=basis_weight,
        )
    with torch.no_grad(): initial = float(objective())
    if not parameters:
        return initial, initial
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True); loss = objective(); loss.backward(); optimizer.step()
    with torch.no_grad(): final = float(objective())
    return initial, final


def host_relation_metrics(
    model: HostRelationModel, data: HostEdges, *, ties: str = "optimistic",
) -> dict[str, float]:
    """Reconstruction metrics and relation accuracy.

    With `ties="optimistic"` (historical) `argmax` breaks ties toward relation 0, so a
    parameter-free model scores the share of relation 0. With `ties="mid"` accuracy is the
    expected accuracy under random tie-breaking (`1/R` for a constant model), and it is
    reported as NaN for models without trainable parameters.
    """
    with torch.no_grad():
        prediction = model(data.sources, data.relation_ids)
        cosine = F.cosine_similarity(prediction, data.targets, dim=-1)
        candidates = torch.stack([
            model(data.sources, torch.full_like(data.relation_ids, relation))
            for relation in range(model.relation_count)
        ], dim=1)
        scores = F.cosine_similarity(candidates, data.targets[:, None, :], dim=-1)
        if ties == "optimistic":
            accuracy = float((scores.argmax(1) == data.relation_ids).float().mean())
        elif ties == "mid":
            if not any(p.requires_grad for p in model.parameters()):
                accuracy = float("nan")
            else:
                best = scores.max(1, keepdim=True).values
                winners = scores == best
                hit = winners.gather(1, data.relation_ids[:, None]).squeeze(1).float()
                accuracy = float((hit / winners.sum(1).float()).mean())
        else:
            raise ValueError("ties must be 'mid' or 'optimistic'")
        return {
            "cosine": float(cosine.mean()),
            "mse": float(F.mse_loss(prediction, data.targets)),
            "relation_accuracy": accuracy,
        }


def prediction_metrics(prediction: Tensor, targets: Tensor) -> dict[str, float]:
    return {
        "cosine": float(F.cosine_similarity(prediction, targets, dim=-1).mean()),
        "mse": float(F.mse_loss(prediction, targets)),
    }


def target_retrieval_metrics(
    prediction: Tensor, target_node_ids: Tensor, all_anchors: Tensor,
    candidate_node_ids: Tensor | None = None, *, ties: str = "optimistic",
) -> dict[str, float]:
    """Rank the exact target among unique held-out target nodes (see `positive_ranks`)."""
    candidates = (target_node_ids if candidate_node_ids is None else candidate_node_ids).unique(sorted=True)
    scores = F.normalize(prediction, dim=-1) @ F.normalize(all_anchors[candidates], dim=-1).T
    matches = candidates[None, :] == target_node_ids[:, None]
    ranks = positive_ranks(scores, matches, ties=ties)
    return {
        "target_mrr": float((1 / ranks.float()).mean()),
        "target_recall_at_10": float((ranks <= 10).float().mean()),
        "target_candidates": int(candidates.numel()),
    }


def multi_positive_target_retrieval_metrics(
    prediction: Tensor, source_node_ids: Tensor, target_node_ids: Tensor,
    relation_ids: Tensor, all_anchors: Tensor, candidate_node_ids: Tensor | None = None, *,
    ties: str = "optimistic",
) -> dict[str, float]:
    """Rank any target observed for the same held-out (source, relation) as positive.

    Pass the split's full target set as `candidate_node_ids` when scoring a subset (e.g. one
    relation); otherwise the candidate universe shrinks to the subset's own targets.
    """
    candidates = (target_node_ids if candidate_node_ids is None else candidate_node_ids).unique(sorted=True)
    scores = F.normalize(prediction, dim=-1) @ F.normalize(all_anchors[candidates], dim=-1).T
    positive_targets = (
        (source_node_ids[:, None] == source_node_ids[None, :])
        & (relation_ids[:, None] == relation_ids[None, :])
    )
    candidate_matches = target_node_ids[:, None] == candidates[None, :]
    positives = positive_targets.to(torch.float32) @ candidate_matches.to(torch.float32) > 0
    ranks = positive_ranks(scores, positives, ties=ties)
    return {
        "distribution_mrr": float((1 / ranks.float()).mean()),
        "distribution_recall_at_10": float((ranks <= 10).float().mean()),
        "mean_positive_targets": float(positives.sum(1).float().mean()),
    }


def source_neighborhood_metrics(
    prediction: Tensor, source_node_ids: Tensor, target_node_ids: Tensor,
    relation_ids: Tensor, all_anchors: Tensor, *, temperature: float = 0.07,
    ties: str = "optimistic", seen_node_ids: Tensor | None = None,
    per_query: bool = False,
) -> dict[str, Any]:
    """Evaluate one set-valued retrieval query per unique `(source, relation)`.

    With `seen_node_ids` (training endpoints), queries are also reported in two strata:
    `seen` if any positive target was a training endpoint, `unseen` otherwise. With
    `per_query=True` the per-query reciprocal ranks are returned for paired statistics.
    """
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
        rank = positive_ranks(scores[None, :], positives[None, :], ties=ties)[0]
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
    result: dict[str, Any] = {
        "query_mrr": mean(reciprocal_ranks), "set_recall_at_10": mean(recalls),
        "set_hit_at_10": mean(hits), "set_mass_nll": mean(nlls),
        "target_centroid_cosine": mean(centroid_cosines),
        "mean_targets_per_query": mean(sizes), "queries": int(len(unique_keys)),
        "target_candidates": int(len(candidates)),
    }
    if seen_node_ids is not None:
        seen = torch.tensor([
            bool(torch.isin(target_node_ids[(keys == key).all(dim=1)], seen_node_ids).any())
            for key in unique_keys
        ])
        rr = torch.stack(reciprocal_ranks)
        for name, mask in (("seen", seen), ("unseen", ~seen)):
            result[f"query_mrr_{name}"] = float(rr[mask].mean()) if mask.any() else float("nan")
            result[f"queries_{name}"] = int(mask.sum())
    if per_query:
        result["per_query_reciprocal_rank"] = [float(value) for value in reciprocal_ranks]
    return result


def derange_labels(labels: Tensor, generator: torch.Generator) -> Tensor:
    """Shuffle labels so that as many as possible change, preserving the label multiset.

    A plain permutation leaves many labels unchanged when a few labels dominate (with two
    relation types it is a coin flip per edge). Items are ordered randomly, grouped by label,
    and each takes the label `m` positions later in that order, where `m` is the size of the
    largest label group; every label changes whenever `m ≤ n/2`.
    """
    labels = labels.cpu()
    n = labels.numel()
    if n < 2:
        return labels.clone()
    order = torch.randperm(n, generator=generator)
    grouped = order[torch.sort(labels[order], stable=True).indices]
    shift = int(torch.bincount(labels).max())
    shuffled = torch.empty_like(labels)
    shuffled[grouped] = labels[grouped.roll(-shift)]
    return shuffled