"""Frame composition: static (M0), attentive (M1) and factored (M2) VSA concept vectors.

A concept's frame is a list of edges `(relation r_e, atomic a_e)`. Each edge is bound through a
relation transform, `v_e = T_{r_e}(a_e)`, and the concept vector is a weighted bundle

    c_i(q) = N( Σ_e w_{i,e}(q) · v_e ).

M0 uses `w = 1`. M1 computes `w = |F_i| · softmax_e(⟨u_i(q), k_e⟩ / (τ √d_k))` with edge keys
`k_e = K_r[r_e] + W_K^a a_e` and query `u_i(q) = ū_i + W_C q`. M2 chooses the concept factor
`ū_i`: induced from the frame (`W_Q c̄_i`, no per-concept parameters, transfers to unseen
concepts), free (one learnable row per concept, cannot transfer) or hybrid (induced plus a
penalized residual `δ_i` that stays zero for concepts never trained). See
`resources/plan-improvement/formulation.md` §0–§2.

Only the concepts requested in a batch are generated. Accumulation uses `index_add_`, so an
edge listed twice in a frame counts twice (indexed `+=` silently drops duplicates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .algebra import normalize
from .relations import create_relation_transform


@dataclass(frozen=True)
class FrameSchedule:
    """CSR frame table: edges of concept `i` are `offsets[i]:offsets[i + 1]`."""

    offsets: Tensor
    relations: Tensor
    fillers: Tensor

    def __post_init__(self) -> None:
        if self.offsets.ndim != 1 or self.offsets.numel() < 1 or int(self.offsets[0]) != 0:
            raise ValueError("offsets must be a 1-D tensor starting at 0")
        if bool((self.offsets[1:] < self.offsets[:-1]).any()):
            raise ValueError("offsets must be non-decreasing")
        if self.relations.shape != self.fillers.shape or self.relations.numel() != int(self.offsets[-1]):
            raise ValueError("relations and fillers must hold offsets[-1] edges")

    @property
    def concept_count(self) -> int:
        return self.offsets.numel() - 1

    @property
    def degrees(self) -> Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    @classmethod
    def from_frames(cls, frames: Sequence[Iterable[tuple[int, int]]]) -> "FrameSchedule":
        """Build from one list of `(relation, filler)` pairs per concept."""
        relations: list[int] = []
        fillers: list[int] = []
        offsets = [0]
        for frame in frames:
            for relation, filler in frame:
                relations.append(int(relation)); fillers.append(int(filler))
            offsets.append(len(relations))
        return cls(torch.tensor(offsets), torch.tensor(relations, dtype=torch.long),
                   torch.tensor(fillers, dtype=torch.long))

    @classmethod
    def from_recipes(cls, recipes: Tensor) -> "FrameSchedule":
        """Convert `OntologyFactorizer` recipes (`recipes[n, r]` = value or −1) to frames."""
        if recipes.ndim != 2:
            raise ValueError("recipes must have shape (nodes, role_count)")
        return cls.from_frames([
            [(role, int(value)) for role, value in enumerate(row.tolist()) if value >= 0]
            for row in recipes
        ])

    def to(self, device: torch.device | str) -> "FrameSchedule":
        return FrameSchedule(self.offsets.to(device), self.relations.to(device), self.fillers.to(device))


def segment_softmax(scores: Tensor, segments: Tensor, segment_count: int) -> Tensor:
    """Softmax of `scores` within each segment id (segments need not be sorted)."""
    maxima = scores.new_full((segment_count,), float("-inf"))
    maxima = maxima.scatter_reduce(0, segments, scores, reduce="amax", include_self=True)
    exponents = (scores - maxima[segments]).exp()
    totals = scores.new_zeros(segment_count).index_add(0, segments, exponents)
    return exponents / totals[segments]


def _occurrence_edges(schedule: FrameSchedule, concept_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Edge indices of each requested concept and the occurrence each edge belongs to."""
    degrees = schedule.degrees[concept_ids]
    if bool((degrees == 0).any()):
        raise ValueError("every composed concept needs at least one edge")
    segments = torch.repeat_interleave(torch.arange(concept_ids.numel(), device=concept_ids.device), degrees)
    starts = torch.cumsum(degrees, 0) - degrees
    local = torch.arange(segments.numel(), device=concept_ids.device) - starts[segments]
    return schedule.offsets[concept_ids][segments] + local, segments


class FrameComposer(nn.Module):
    """Compose concept vectors from atomics, relations and a frame schedule.

    `mode="bundle"` is M0. `mode="attentive"` is M1 with the concept factor chosen by
    `concept_factor` ∈ {"induced", "free", "hybrid"} (M2). `operator` is any relation family of
    `create_relation_transform` (`hrr`, `hrr_identity`, `diagonal`, `low_rank_identity`,
    `orthogonal`, ...), `untyped` (no binding) or `random_fixed:<family>` (bound, never trained).
    """

    def __init__(
        self, schedule: FrameSchedule, atomic_count: int, relation_count: int, dimension: int, *,
        operator: str = "hrr", mode: str = "bundle", concept_factor: str = "induced",
        key_dimension: int = 16, context_dimension: int = 0, temperature: float = 1.0,
        learn_temperature: bool = False, weight_mode: str = "softmax", output_dimension: int | None = None,
        normalize_atomics: bool = True, rank: int = 8,
    ) -> None:
        super().__init__()
        if mode not in {"bundle", "attentive"}:
            raise ValueError("mode must be 'bundle' or 'attentive'")
        if concept_factor not in {"induced", "free", "hybrid"}:
            raise ValueError("concept_factor must be 'induced', 'free' or 'hybrid'")
        if weight_mode not in {"softmax", "sigmoid", "raw"}:
            raise ValueError("weight_mode must be 'softmax', 'sigmoid' or 'raw'")
        self.atomic_count, self.relation_count = atomic_count, relation_count
        self.register_buffer("frame_offsets", schedule.offsets.clone())
        self.register_buffer("frame_relations", schedule.relations.clone())
        self.register_buffer("frame_fillers", schedule.fillers.clone())
        self._check_schedule(schedule)
        self.mode, self.concept_factor, self.weight_mode = mode, concept_factor, weight_mode
        self.normalize_atomics = normalize_atomics
        self.operator = operator
        self.atomics = nn.Parameter(torch.randn(atomic_count, dimension) / dimension**0.5)
        family, frozen = operator, False
        if operator.startswith("random_fixed"):
            family, frozen = (operator.split(":", 1)[1] if ":" in operator else "hrr"), True
        self.transform = create_relation_transform("additive" if family == "untyped" else family,
                                                   relation_count, dimension, rank=rank)
        if frozen:
            self.transform.requires_grad_(False)
        self.key_dimension = key_dimension
        if mode == "attentive":
            self.relation_keys = nn.Parameter(torch.randn(relation_count, key_dimension) / key_dimension**0.5)
            self.atomic_key = nn.Linear(dimension, key_dimension, bias=False)
            self.query = nn.Linear(dimension, key_dimension, bias=False)
            self.context = nn.Linear(context_dimension, key_dimension, bias=False) if context_dimension else None
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)), requires_grad=learn_temperature)
            concepts = schedule.concept_count
            if concept_factor == "free":
                self.free_factor = nn.Parameter(torch.randn(concepts, key_dimension) / key_dimension**0.5)
            elif concept_factor == "hybrid":
                self.delta = nn.Parameter(torch.zeros(concepts, key_dimension))
        self.projector = nn.Linear(dimension, output_dimension, bias=False) if output_dimension else None
        # When True, the next compositions record `z_i` (pre-normalization rows, with gradients
        # retained) and their schedule so M3 can read per-usage gradients after backward.
        self.capture_usage = False
        self.captured: list[dict[str, Tensor]] = []

    @property
    def schedule(self) -> FrameSchedule:
        return FrameSchedule(self.frame_offsets, self.frame_relations, self.frame_fillers)

    def _check_schedule(self, schedule: FrameSchedule) -> None:
        if schedule.fillers.numel() and int(schedule.fillers.max()) >= self.atomic_count:
            raise ValueError("schedule references an atomic beyond atomic_count")
        if schedule.relations.numel() and int(schedule.relations.max()) >= self.relation_count:
            raise ValueError("schedule references a relation beyond relation_count")

    def set_schedule(self, schedule: FrameSchedule) -> None:
        """Replace the frame table (same concept count), e.g. after an ontology update."""
        if schedule.concept_count != self.schedule.concept_count:
            raise ValueError("set_schedule keeps the concept count; per-concept factors are sized by it")
        self._check_schedule(schedule)
        device = self.frame_offsets.device
        self.frame_offsets = schedule.offsets.to(device)
        self.frame_relations = schedule.relations.to(device)
        self.frame_fillers = schedule.fillers.to(device)

    # -- pieces -------------------------------------------------------------------------------

    def atomic_vectors(self) -> Tensor:
        return normalize(self.atomics) if self.normalize_atomics else self.atomics

    def bound_edges(self, edge_index: Tensor) -> Tensor:
        """`v_e = T_{r_e}(a_e)` for the given schedule edges."""
        fillers = self.atomic_vectors()[self.schedule.fillers[edge_index]]
        return self.transform(self.schedule.relations[edge_index], fillers)

    def edge_weights(
        self, edge_index: Tensor, segments: Tensor, concept_ids: Tensor, bound: Tensor,
        context: Tensor | None,
    ) -> Tensor:
        count = concept_ids.numel()
        if self.mode == "bundle":
            return bound.new_ones(edge_index.numel())
        static = bound.new_zeros(count, bound.shape[-1]).index_add(0, segments, bound)
        if self.concept_factor == "free":
            factor = self.free_factor[concept_ids]
        else:
            factor = self.query(static)
            if self.concept_factor == "hybrid":
                factor = factor + self.delta[concept_ids]
        if context is not None:
            if self.context is None:
                raise ValueError("composer was built without context_dimension")
            factor = factor + self.context(context)
        keys = self.relation_keys[self.schedule.relations[edge_index]] + self.atomic_key(
            self.atomic_vectors()[self.schedule.fillers[edge_index]])
        scores = (factor[segments] * keys).sum(-1) / math.sqrt(self.key_dimension)
        if self.weight_mode == "raw":
            return scores
        if self.weight_mode == "sigmoid":
            return 2 * torch.sigmoid(scores)
        degrees = self.schedule.degrees[concept_ids].to(scores.dtype)
        return degrees[segments] * segment_softmax(scores / self.log_temperature.exp(), segments, count)

    # -- composition --------------------------------------------------------------------------

    def compose(self, concept_ids: Tensor, context: Tensor | None = None, *,
                return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Concept vectors for `concept_ids` (one row per occurrence), before projection."""
        if concept_ids.ndim != 1:
            raise ValueError("concept_ids must be a vector")
        if context is not None and context.shape[0] != concept_ids.numel():
            raise ValueError("context needs one row per occurrence")
        if context is None and not return_weights:
            # Static composition: build each distinct concept once, then gather.
            unique, inverse = torch.unique(concept_ids, return_inverse=True)
            return self._compose(unique, None)[0][inverse]
        rows, weights, edge_index = self._compose(concept_ids, context)
        return (rows, weights, edge_index) if return_weights else rows

    def _compose(self, concept_ids: Tensor, context: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        edge_index, segments = _occurrence_edges(self.schedule, concept_ids)
        needed, position = torch.unique(edge_index, return_inverse=True)
        bound = self.bound_edges(needed)[position]
        weights = self.edge_weights(edge_index, segments, concept_ids, bound, context)
        summed = bound.new_zeros(concept_ids.numel(), bound.shape[-1]).index_add(
            0, segments, weights.unsqueeze(-1) * bound)
        if self.capture_usage and summed.requires_grad:
            summed.retain_grad()
            self.captured.append({"summed": summed, "segments": segments, "edge_index": edge_index,
                                  "weights": weights.detach(), "concept_ids": concept_ids})
        return normalize(summed), weights, edge_index

    def forward(self, concept_ids: Tensor, context: Tensor | None = None) -> Tensor:
        rows = self.compose(concept_ids, context)
        return self.projector(rows) if self.projector is not None else rows

    # -- growth (used by the developmental dictionary, M3) --------------------------------------

    def add_atomics(self, vectors: Tensor) -> Tensor:
        """Append atomic rows; returns their ids. The `atomics` Parameter object is replaced."""
        start = self.atomics.shape[0]
        self.atomics = nn.Parameter(torch.cat([self.atomics.detach(), vectors.to(self.atomics)], 0))
        self.atomic_count = self.atomics.shape[0]
        return torch.arange(start, self.atomic_count, device=self.atomics.device)

    def add_relation_copies(self, sources: Tensor, offsets: Tensor | None = None) -> Tensor:
        """Append relations copied from `sources` (plus optional `offsets` on their vector).

        Supported for families with one vector per relation (`hrr`, `hrr_identity`, `diagonal`,
        `map`), whose relation parameter is `roles` or `diagonal`.
        """
        name = "roles" if hasattr(self.transform, "roles") else "diagonal" if hasattr(self.transform, "diagonal") else None
        if name is None:
            raise ValueError(f"relation splitting is not supported for operator {self.operator!r}")
        current = getattr(self.transform, name)
        new = current.detach()[sources].clone()
        if offsets is not None:
            new = new + offsets.to(new)
        setattr(self.transform, name, nn.Parameter(torch.cat([current.detach(), new], 0),
                                                   requires_grad=current.requires_grad))
        start = self.relation_count
        self.relation_count += sources.numel()
        self.transform.relation_count = self.relation_count
        if self.mode == "attentive":
            self.relation_keys = nn.Parameter(torch.cat([self.relation_keys.detach(),
                                                         self.relation_keys.detach()[sources]], 0))
        return torch.arange(start, self.relation_count, device=current.device)

    def relation_vectors(self) -> Tensor:
        return getattr(self.transform, "roles", None) if hasattr(self.transform, "roles") else self.transform.diagonal

    # -- inspection ---------------------------------------------------------------------------

    @torch.no_grad()
    def explain(self, concept: int, context: Tensor | None = None) -> list[dict[str, float | int]]:
        """Edges of one concept with their weights, largest first."""
        ids = torch.tensor([concept], device=self.atomics.device)
        ctx = None if context is None else context.reshape(1, -1)
        _, weights, edge_index = self._compose(ids, ctx)
        entries = [
            {"edge": int(edge), "relation": int(self.schedule.relations[edge]),
             "filler": int(self.schedule.fillers[edge]), "weight": float(weight)}
            for edge, weight in zip(edge_index.tolist(), weights.tolist())
        ]
        return sorted(entries, key=lambda item: -item["weight"])

    def parameter_groups(self) -> dict[str, int]:
        """Parameter counts by group, as the shared protocols require."""
        groups = {"atomic": 0, "relation": 0, "global": 0, "concept_local": 0}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "atomics":
                group = "atomic"
            elif name.startswith("transform.") or name == "relation_keys":
                group = "relation"
            elif name in {"free_factor", "delta"}:
                group = "concept_local"
            else:
                group = "global"
            groups[group] += parameter.numel()
        return groups

    def delta_penalty(self) -> Tensor:
        """`Σ ‖δ_i‖²` for the hybrid factor (zero otherwise)."""
        delta = getattr(self, "delta", None)
        return delta.square().sum() if delta is not None else self.atomics.new_zeros(())


def uniform_bundle(composer: FrameComposer, concept_ids: Tensor) -> Tensor:
    """Reference M0 composition by explicit loops (for tests)."""
    atomics = composer.atomic_vectors()
    rows = []
    for concept in concept_ids.tolist():
        start, end = int(composer.schedule.offsets[concept]), int(composer.schedule.offsets[concept + 1])
        relations = composer.schedule.relations[start:end]
        fillers = atomics[composer.schedule.fillers[start:end]]
        rows.append(normalize(composer.transform(relations, fillers).sum(0)))
    return torch.stack(rows)
