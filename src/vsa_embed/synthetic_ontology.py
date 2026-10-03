"""Planted logical ontology for E10 (self-learned semantics, hypothesis H-H).

A synthetic world in which every relation has a known logical character, so that relations can be
hidden on purpose and their recovery scored against gold *after* learning:

| relation        | tails      | planted structure                                    |
|-----------------|------------|------------------------------------------------------|
| `is_a`          | concepts   | a 4-level tree (functional, antisymmetric)           |
| `has_attribute` | attributes | inherited down the tree with mutations               |
| `part_of`       | concepts   | parts → wholes (functional)                          |
| `has_part`      | concepts   | the inverse of `part_of`                             |
| `member_of`     | concepts   | members → groups (functional)                        |
| `similar_to`    | concepts   | sibling pairs, both directions (symmetric)           |
| `located_in`    | concepts   | transitive closure of a city → country → region tree |

Every concept is also an atomic (its identity filler), plus `attributes` attribute atomics. The
teacher composes `y_i = N(Σ_e w*_e · r*_{r_e} ⊛ a*_{a_e})` with HRR and per-relation salience times
per-edge jitter; each concept is observed through `observations` noisy views of `y_i`
(`y_{i,o} = N(y_i + σ ε)`), split into train / validation / audit observations. Optionally each
observation comes from one of a pool of *sources* that add a source-specific artifact edge (a fixed
`(relation, attribute)` pair bound in with weight `artifact_strength`) — the E10.7 self-confirmation
setting, where low source variety lets a learner confirm a source artifact as if it were a fact.

The learner never sees gold relation labels for hidden relations; `OntologyWorld` keeps them for
evaluation only. Builders draw only from their own `torch.Generator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import torch
from torch import Tensor
from torch.nn import functional as F

from .algebra import HRRAlgebra

RELATIONS = ("is_a", "has_attribute", "part_of", "has_part", "member_of", "similar_to", "located_in")

# Designed logical character of each planted relation (gold; for evaluation only).
GOLD_PROPERTIES: dict[str, tuple[str, ...]] = {
    "is_a": ("functional", "antisymmetric"),
    "has_attribute": (),
    "part_of": ("inverse_of:has_part", "functional", "antisymmetric"),
    "has_part": ("inverse_of:part_of", "antisymmetric"),
    "member_of": ("functional", "antisymmetric"),
    "similar_to": ("symmetric",),
    "located_in": ("transitive", "antisymmetric"),
}


@dataclass
class OntologyWorld:
    """A gold ontology with observations. Edges are `(head concept, relation, filler atomic)`."""

    name: str
    concept_count: int
    atomic_count: int
    relation_names: list[str]
    heads: Tensor                     # (E,) head concept of each gold edge
    relations: Tensor                 # (E,) world relation id
    fillers: Tensor                   # (E,) filler atomic id
    teacher_weights: Tensor           # (E,) teacher edge weight (synthetic) or ones
    filler_node: list[Hashable]       # node key of each atomic (concept id for concept atomics)
    concept_node: list[Hashable]      # node key of each concept
    concept_level: Tensor             # (N,) depth in the is_a tree (−1 if unknown)
    observations: Tensor              # (N, n_obs, target_dim)
    observation_sources: Tensor       # (N, n_obs) source id (−1 = none)
    splits: dict[str, Tensor]         # concept ids: train / validation / test / new_word
    observation_splits: dict[str, list[int]]   # observation indices: train / val / audit (+ infer / eval)
    dimension: int                    # composition dimension of the learner
    atomic_prior: Tensor | None = None
    relation_prior: Tensor | None = None
    gold_properties: dict[str, tuple[str, ...]] = field(default_factory=dict)
    truth: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_dimension(self) -> int:
        return int(self.observations.shape[-1])

    def relation_id(self, name: str) -> int:
        return self.relation_names.index(name)

    def edge_list(self) -> list[tuple[int, int, int]]:
        return list(zip(self.heads.tolist(), self.relations.tolist(), self.fillers.tolist()))

    def pairs(self, relation: int | str, heads: set[int] | None = None) -> set[tuple[Hashable, Hashable]]:
        """Gold (head node, tail node) pairs of one relation, optionally restricted to head concepts."""
        rid = self.relation_id(relation) if isinstance(relation, str) else int(relation)
        mask = self.relations == rid
        return {(self.concept_node[h], self.filler_node[a])
                for h, a in zip(self.heads[mask].tolist(), self.fillers[mask].tolist())
                if heads is None or h in heads}

    def relations_of_pair(self) -> dict[tuple[int, int], set[int]]:
        """(head concept, filler atomic) → set of gold world relations."""
        table: dict[tuple[int, int], set[int]] = {}
        for h, r, a in self.edge_list():
            table.setdefault((h, a), set()).add(r)
        return table


def _unit(generator: torch.Generator, count: int, dimension: int) -> Tensor:
    return F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)


def noisy_prior(truth: Tensor, cosine: float, generator: torch.Generator) -> Tensor:
    """Unit vectors with expected cosine `cosine` to the (unit) truth: a stand-in for a pretrained prior."""
    noise = torch.randn(truth.shape, generator=generator)
    noise = noise - (noise * truth).sum(-1, keepdim=True) * truth
    noise = F.normalize(noise, dim=-1)
    return F.normalize(cosine * truth + (1 - cosine**2) ** 0.5 * noise, dim=-1)


def make_ontology_world(
    *, seed: int = 0, dimension: int = 64, roots: int = 4, branching: tuple[int, ...] = (3, 4, 5),
    attributes: int = 40, wholes: int = 40, parts: int = 100, groups: int = 15, members: int = 60,
    similar_pairs_per_family: int = 2, regions: int = 4, countries_per_region: int = 3,
    cities_per_country: int = 3, salience_std: float = 0.3, jitter_std: float = 0.15,
    observations: int = 12, observation_noise: float = 0.6, prior_cosine: float = 0.8,
    split_fractions: tuple[float, float, float] = (0.13, 0.13, 0.13),
    observation_split: dict[str, list[int]] | None = None,
    sources: int = 0, sources_per_concept: int = 1, artifact_strength: float = 0.0,
    audit_observations: int = 4,
) -> OntologyWorld:
    """Build the planted world (see module docstring). `split_fractions` = (validation, test, new_word)."""
    g = torch.Generator().manual_seed(seed)
    rand = lambda n: int(torch.randint(n, (), generator=g))  # noqa: E731
    # --- is_a tree ------------------------------------------------------------------------------
    level, parent = [0] * roots, [-1] * roots
    frontier = list(range(roots))
    for depth, fan in enumerate(branching, start=1):
        nxt = []
        for p in frontier:
            for _ in range(fan):
                level.append(depth); parent.append(p); nxt.append(len(level) - 1)
        frontier = nxt
    n = len(level)
    children: dict[int, list[int]] = {}
    for c, p in enumerate(parent):
        if p >= 0:
            children.setdefault(p, []).append(c)
    edges: set[tuple[int, int, int]] = set()
    rel = {name: i for i, name in enumerate(RELATIONS)}
    for c, p in enumerate(parent):
        if p >= 0:
            edges.add((c, rel["is_a"], p))
    # --- attributes inherited down the tree -----------------------------------------------------
    attr_base = n
    attrs: dict[int, set[int]] = {}
    for c in range(n):   # parents precede children by construction
        own = set()
        if parent[c] < 0:
            while len(own) < 2:
                own.add(rand(attributes))
        else:
            own = {a for a in attrs[parent[c]] if float(torch.rand((), generator=g)) < 0.5}
            if float(torch.rand((), generator=g)) < 0.8 or not own:
                own.add(rand(attributes))
        attrs[c] = own
        for a in own:
            edges.add((c, rel["has_attribute"], attr_base + a))
    deep = [c for c in range(n) if level[c] >= len(branching) - 1]
    leaves = [c for c in range(n) if level[c] == len(branching)]
    order = lambda items: [items[i] for i in torch.randperm(len(items), generator=g).tolist()]  # noqa: E731
    # --- part_of / has_part ---------------------------------------------------------------------
    whole_set = order(deep)[:wholes]
    part_set = [c for c in order(leaves) if c not in set(whole_set)][:parts]
    for p in part_set:
        w = whole_set[rand(len(whole_set))]
        edges.add((p, rel["part_of"], w)); edges.add((w, rel["has_part"], p))
    # --- member_of --------------------------------------------------------------------------------
    mid = [c for c in range(n) if level[c] == len(branching) - 1]
    group_set = order(mid)[:groups]
    for m in order(leaves)[:members]:
        edges.add((m, rel["member_of"], group_set[rand(len(group_set))]))
    # --- similar_to (symmetric sibling pairs) ------------------------------------------------------
    for p in sorted(children):
        kids = [c for c in children[p] if level[c] == len(branching)]
        kids = order(kids)
        for k in range(min(similar_pairs_per_family, len(kids) // 2)):
            a, b = kids[2 * k], kids[2 * k + 1]
            edges.add((a, rel["similar_to"], b)); edges.add((b, rel["similar_to"], a))
    # --- located_in (transitive closure of a place tree) ----------------------------------------
    place_count = regions * (1 + countries_per_region * (1 + cities_per_country))
    places = order([c for c in range(n) if level[c] >= 2])[:place_count]
    it = iter(places)
    for _ in range(regions):
        region = next(it)
        for _ in range(countries_per_region):
            country = next(it)
            edges.add((country, rel["located_in"], region))
            for _ in range(cities_per_country):
                city = next(it)
                edges.add((city, rel["located_in"], country)); edges.add((city, rel["located_in"], region))
    edge_list = sorted(edges)
    heads = torch.tensor([e[0] for e in edge_list]); relations = torch.tensor([e[1] for e in edge_list])
    fillers = torch.tensor([e[2] for e in edge_list])
    atomic_count = n + attributes
    # --- teacher ---------------------------------------------------------------------------------
    atomics = _unit(g, atomic_count, dimension)
    roles = _unit(g, len(RELATIONS), dimension)
    salience = salience_std * torch.randn(len(RELATIONS), generator=g)
    weights = torch.exp(salience[relations] + jitter_std * torch.randn(len(edge_list), generator=g))
    bound = HRRAlgebra().bind(roles[relations], atomics[fillers])
    clean = F.normalize(torch.zeros(n, dimension).index_add(0, heads, weights[:, None] * bound), dim=-1)
    # --- sources (E10.7) -------------------------------------------------------------------------
    source_ids = torch.full((n, observations), -1, dtype=torch.long)
    artifacts = torch.zeros(n, observations, dimension)
    source_rel = torch.zeros(max(sources, 1), dtype=torch.long)
    source_attr = torch.zeros(max(sources, 1), dtype=torch.long)
    if sources and artifact_strength > 0:
        source_rel = torch.full((sources,), rel["has_attribute"], dtype=torch.long)
        source_attr = torch.randint(attributes, (sources,), generator=g) + attr_base
        source_vectors = HRRAlgebra().bind(roles[source_rel], atomics[source_attr])
        audit_from = observations - audit_observations
        for c in range(n):
            pool = torch.randperm(sources, generator=g)
            own, fresh = pool[:sources_per_concept], pool[sources_per_concept:]
            for o in range(observations):
                s = int(own[o % sources_per_concept]) if o < audit_from else int(fresh[(o - audit_from) % len(fresh)])
                source_ids[c, o] = s
                artifacts[c, o] = artifact_strength * source_vectors[s]
    noise = torch.randn(n, observations, dimension, generator=g) * observation_noise / dimension**0.5
    raw = clean[:, None, :] + artifacts + noise
    observed = F.normalize(raw, dim=-1)
    # --- splits ----------------------------------------------------------------------------------
    perm = torch.randperm(n, generator=g)
    counts = [int(round(f * n)) for f in split_fractions]
    val, test, new = (perm[:counts[0]], perm[counts[0]:counts[0] + counts[1]],
                      perm[counts[0] + counts[1]:sum(counts)])
    train = perm[sum(counts):]
    splits = {"train": train.sort().values, "validation": val.sort().values, "test": test.sort().values,
              "new_word": new.sort().values}
    if observation_split is None:
        audit_from = observations - audit_observations
        half = audit_from // 2
        observation_split = {"train": list(range(half)), "val": list(range(half, audit_from)),
                             "audit": list(range(audit_from, observations)),
                             "infer": list(range(audit_from)), "eval": list(range(audit_from, observations))}
    filler_node: list[Hashable] = list(range(n)) + [("attr", a) for a in range(attributes)]
    return OntologyWorld(
        name="synthetic-ontology", concept_count=n, atomic_count=atomic_count, relation_names=list(RELATIONS),
        heads=heads, relations=relations, fillers=fillers, teacher_weights=weights, filler_node=filler_node,
        concept_node=list(range(n)), concept_level=torch.tensor(level), observations=observed,
        observation_sources=source_ids, splits=splits, observation_splits=observation_split, dimension=dimension,
        atomic_prior=noisy_prior(atomics, prior_cosine, g), relation_prior=noisy_prior(roles, prior_cosine, g),
        gold_properties=dict(GOLD_PROPERTIES),
        truth={"atomics": atomics, "roles": roles, "salience": salience, "clean_targets": clean,
               "parent": torch.tensor(parent), "source_relation": source_rel, "source_attribute": source_attr},
        metadata={"probes": {"isa": "is_a", "transitive": "located_in", "inverse": ["has_part", "part_of"],
                             "symmetric": "similar_to"},
                  "seed": seed, "edges": len(edge_list), "concepts": n, "atomics": atomic_count,
                  "edges_per_relation": {name: int((relations == i).sum()) for i, name in enumerate(RELATIONS)},
                  "sources": sources, "sources_per_concept": sources_per_concept,
                  "artifact_strength": artifact_strength},
    )
