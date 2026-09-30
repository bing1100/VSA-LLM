"""Synthetic teachers for experiment E0 (identifiability of M1, M2 and M3).

Every teacher draws only from its own `torch.Generator`, so building one never changes the
global RNG. Teachers bind with HRR (random unit-norm roles) and bundle with mass-preserving
weights, matching `FrameComposer`'s parameterization so recovery is possible in principle.

- `make_frame_teacher`: concepts with frames of degree 3–8; edge weights either uniform
  (static teacher, the no-harm control), a rank-`k*` bilinear function of the concept's own
  bundle (E0.3), or additionally conditioned on one of `K` context prototypes (E0.1).
- `make_polysemy_teacher`: some atomics (and relations) are two hidden vectors used by disjoint
  concept subsets under one visible label (E0.2); the ground-truth usage partition is returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F

from .algebra import HRRAlgebra
from .compose import FrameSchedule, segment_softmax


@dataclass
class FrameTeacherData:
    schedule: FrameSchedule
    atomics: Tensor                 # (A, d) hidden atomic vectors
    roles: Tensor                   # (R, d) hidden HRR roles
    train_concepts: Tensor          # concept ids used for training
    test_concepts: Tensor           # composition-disjoint held-out concepts
    prototypes: Tensor              # (K, context_dim) context prototypes (empty if static)
    train_contexts: Tensor          # prototype ids seen in training
    test_contexts: Tensor           # held-out prototype ids
    samples: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    # samples[split] = {"concepts", "context_ids", "contexts", "targets", "weights", "edge_index"}


def _random_frames(
    concept_count: int, atomic_count: int, relation_count: int, degree_range: tuple[int, int],
    generator: torch.Generator,
) -> list[list[tuple[int, int]]]:
    frames = []
    low, high = degree_range
    for _ in range(concept_count):
        degree = int(torch.randint(low, high + 1, (), generator=generator))
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < degree:
            pairs.add((int(torch.randint(relation_count, (), generator=generator)),
                       int(torch.randint(atomic_count, (), generator=generator))))
        frames.append(sorted(pairs))
    return frames


def _supported_holdout(
    frames: list[list[tuple[int, int]]], holdout: int, generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Held-out concepts whose every atomic and relation also occurs in a training frame."""
    n = len(frames)
    for _ in range(200):
        order = torch.randperm(n, generator=generator)
        test, train = order[:holdout], order[holdout:]
        seen_atoms = {a for i in train.tolist() for _, a in frames[i]}
        seen_rels = {r for i in train.tolist() for r, _ in frames[i]}
        if all(a in seen_atoms and r in seen_rels for i in test.tolist() for r, a in frames[i]):
            return train.sort().values, test.sort().values
    raise RuntimeError("could not find an atomic-supported composition-disjoint holdout")


def make_frame_teacher(
    *, concept_count: int = 400, atomic_count: int = 64, relation_count: int = 6, dimension: int = 64,
    degree_range: tuple[int, int] = (3, 8), key_dimension: int = 8, weighting: str = "contextual",
    context_count: int = 8, context_dimension: int = 16, heldout_contexts: int = 2,
    context_strength: float = 2.0, query_strength: float = 1.0, samples_per_concept: int = 4,
    holdout_concepts: int = 80, noise_std: float = 0.0, seed: int = 0,
) -> FrameTeacherData:
    """Build a teacher; `weighting` ∈ {"static", "bilinear", "contextual"}."""
    if weighting not in {"static", "bilinear", "contextual"}:
        raise ValueError("weighting must be 'static', 'bilinear' or 'contextual'")
    g = torch.Generator().manual_seed(seed)
    frames = _random_frames(concept_count, atomic_count, relation_count, degree_range, g)
    schedule = FrameSchedule.from_frames(frames)
    atomics = F.normalize(torch.randn(atomic_count, dimension, generator=g), dim=-1)
    roles = F.normalize(torch.randn(relation_count, dimension, generator=g), dim=-1)
    relation_keys = torch.randn(relation_count, key_dimension, generator=g)
    atomic_key = torch.randn(dimension, key_dimension, generator=g) / dimension**0.5
    query = torch.randn(dimension, key_dimension, generator=g) / dimension**0.5 * query_strength * math.sqrt(dimension)
    context_map = torch.randn(context_dimension, key_dimension, generator=g) / context_dimension**0.5 * context_strength
    prototypes = torch.randn(context_count, context_dimension, generator=g) if weighting == "contextual" \
        else torch.zeros(0, context_dimension)
    train_concepts, test_concepts = _supported_holdout(frames, holdout_concepts, g)
    if weighting == "contextual":
        order = torch.randperm(context_count, generator=g)
        test_contexts, train_contexts = order[:heldout_contexts].sort().values, order[heldout_contexts:].sort().values
    else:
        train_contexts = test_contexts = torch.zeros(0, dtype=torch.long)
    algebra = HRRAlgebra()
    bound_all = algebra.bind(roles[schedule.relations], atomics[schedule.fillers])
    keys_all = relation_keys[schedule.relations] + atomics[schedule.fillers] @ atomic_key
    data = FrameTeacherData(schedule, atomics, roles, train_concepts, test_concepts, prototypes,
                            train_contexts, test_contexts)

    def draw(concepts: Tensor, context_pool: Tensor) -> dict[str, Tensor]:
        ids = concepts.repeat_interleave(samples_per_concept)
        if weighting == "contextual":
            context_ids = context_pool[torch.randint(context_pool.numel(), (ids.numel(),), generator=g)]
            contexts = prototypes[context_ids] + 0.1 * torch.randn(ids.numel(), context_dimension, generator=g)
        else:
            context_ids = torch.full((ids.numel(),), -1)
            contexts = torch.zeros(ids.numel(), context_dimension)
        degrees = schedule.degrees[ids]
        segments = torch.repeat_interleave(torch.arange(ids.numel()), degrees)
        starts = torch.cumsum(degrees, 0) - degrees
        edge_index = schedule.offsets[ids][segments] + torch.arange(segments.numel()) - starts[segments]
        bound = bound_all[edge_index]
        if weighting == "static":
            weights = torch.ones(edge_index.numel())
        else:
            static = torch.zeros(ids.numel(), dimension).index_add(0, segments, bound)
            factor = static @ query
            if weighting == "contextual":
                factor = factor + contexts @ context_map
            scores = (factor[segments] * keys_all[edge_index]).sum(-1) / math.sqrt(key_dimension)
            weights = degrees[segments].float() * segment_softmax(scores, segments, ids.numel())
        rows = torch.zeros(ids.numel(), dimension).index_add(0, segments, weights[:, None] * bound)
        targets = F.normalize(rows, dim=-1)
        if noise_std:
            targets = F.normalize(targets + noise_std * torch.randn(targets.shape, generator=g) / dimension**0.5, dim=-1)
        return {"concepts": ids, "context_ids": context_ids, "contexts": contexts, "targets": targets,
                "weights": weights, "edge_index": edge_index}

    pools = {"train": train_contexts, "test_seen_context": train_contexts, "test": test_contexts}
    data.samples["train"] = draw(train_concepts, pools["train"])
    data.samples["test"] = draw(test_concepts, pools["test"] if weighting == "contextual" else train_contexts)
    if weighting == "contextual":
        data.samples["test_seen_context"] = draw(test_concepts, pools["test_seen_context"])
    return data


@dataclass
class PolysemyTeacherData:
    schedule: FrameSchedule          # collapsed (visible) labels
    targets: Tensor                  # (concepts, d)
    train_concepts: Tensor
    test_concepts: Tensor
    polysemous_atomics: Tensor       # visible atomic ids that hide two senses
    polysemous_relations: Tensor     # visible relation ids that hide two sub-operators
    edge_atomic_sense: Tensor        # (edges,) hidden sense (0/1) of the edge's atomic; -1 if monosemous
    edge_relation_sense: Tensor      # (edges,) hidden sub-operator (0/1) of the edge's relation; -1 otherwise


def make_polysemy_teacher(
    *, concept_count: int = 600, atomic_count: int = 48, relation_count: int = 6, dimension: int = 64,
    polysemous_atomics: int = 8, polysemous_relations: int = 2, degree_range: tuple[int, int] = (3, 6),
    holdout_concepts: int = 100, noise_std: float = 0.0, seed: int = 0,
) -> PolysemyTeacherData:
    """Planted senses: each polysemous label is two hidden vectors used by disjoint concepts."""
    g = torch.Generator().manual_seed(seed)
    frames = _random_frames(concept_count, atomic_count, relation_count, degree_range, g)
    schedule = FrameSchedule.from_frames(frames)
    sense_atomics = F.normalize(torch.randn(atomic_count, 2, dimension, generator=g), dim=-1)
    sense_roles = F.normalize(torch.randn(relation_count, 2, dimension, generator=g), dim=-1)
    poly_atoms = torch.randperm(atomic_count, generator=g)[:polysemous_atomics].sort().values
    poly_rels = torch.randperm(relation_count, generator=g)[:polysemous_relations].sort().values
    # Each concept belongs to one "domain" (0/1); a polysemous label takes the sense of the
    # concept's domain, so senses are used by disjoint concept subsets.
    domain = torch.randint(0, 2, (concept_count,), generator=g)
    concept_of_edge = torch.repeat_interleave(torch.arange(concept_count), schedule.degrees)
    edge_domain = domain[concept_of_edge]
    atom_poly = torch.isin(schedule.fillers, poly_atoms)
    rel_poly = torch.isin(schedule.relations, poly_rels)
    atom_sense = torch.where(atom_poly, edge_domain, torch.zeros_like(edge_domain))
    rel_sense = torch.where(rel_poly, edge_domain, torch.zeros_like(edge_domain))
    bound = HRRAlgebra().bind(sense_roles[schedule.relations, rel_sense], sense_atomics[schedule.fillers, atom_sense])
    rows = torch.zeros(concept_count, dimension).index_add(0, concept_of_edge, bound)
    targets = F.normalize(rows, dim=-1)
    if noise_std:
        targets = F.normalize(targets + noise_std * torch.randn(targets.shape, generator=g) / dimension**0.5, dim=-1)
    train, test = _supported_holdout(frames, holdout_concepts, g)
    return PolysemyTeacherData(
        schedule, targets, train, test, poly_atoms, poly_rels,
        torch.where(atom_poly, atom_sense, torch.full_like(atom_sense, -1)),
        torch.where(rel_poly, rel_sense, torch.full_like(rel_sense, -1)),
    )
