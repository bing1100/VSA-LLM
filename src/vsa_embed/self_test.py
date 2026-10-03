"""Self-constructed held-out tests for ontology hypotheses (E10.0 (d), E10.5–E10.7).

The learner accepts a hypothesis — an edge, a discovered relation (slot), a structural rule, a
revision — only if it improves fit on **held-out observations** with a bootstrap lower bound > 0.
The tests read a `HeldOutData` object (validation observations of training concepts and of
never-trained validation concepts) and nothing else: no gold edges, no relation names, no audit
observations. Audit data (`HeldOutData` built from the audit observations) and gold labels are used
only by evaluation code, through the same effect functions.

Edits are expressed as changes `δ` of a concept's pre-normalization row `z` (bundle mode, where
removing or adding an edge changes `z` by exactly its weighted bound vector), so every candidate is
scored from one composition pass: `Δ_o = cos(P N(z + δ), y_o) − cos(P N(z), y_o)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .learnable_ontology import LearnableOntologyComposer
from .ontology_hypotheses import Pair, StructuralHypothesis


@dataclass
class HeldOutData:
    """Observations a test may read: one row of `k` observations per concept."""

    concepts: Tensor          # (n,) concept ids
    observations: Tensor      # (n, k, d)
    name: str = "heldout"

    def __post_init__(self) -> None:
        self._row = {int(c): i for i, c in enumerate(self.concepts.tolist())}

    def rows(self, concepts: Sequence[int] | Tensor) -> Tensor:
        values = concepts.tolist() if isinstance(concepts, Tensor) else list(concepts)
        return torch.tensor([self._row.get(int(c), -1) for c in values], dtype=torch.long)

    def has(self, concept: int) -> bool:
        return int(concept) in self._row


@dataclass
class NodeMap:
    """Node keys of concepts and atomics (pairs are expressed over nodes)."""

    node_of_concept: list[Hashable]
    node_of_atomic: list[Hashable]
    concept_of_node: dict[Hashable, int] = field(default_factory=dict)
    atomic_of_node: dict[Hashable, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.concept_of_node = {n: i for i, n in enumerate(self.node_of_concept)}
        self.atomic_of_node = {n: i for i, n in enumerate(self.node_of_atomic)}

    def edge_pair(self, head: int, filler: int) -> Pair:
        return (self.node_of_concept[head], self.node_of_atomic[filler])


@dataclass
class Decision:
    kind: str
    key: Any
    mean: float
    lower: float
    n: int
    accept: bool
    extra: dict[str, Any] = field(default_factory=dict)


def _check_bundle(composer: LearnableOntologyComposer) -> None:
    if composer.mode != "bundle":
        raise ValueError("self-test edits assume bundle composition (exact add/remove of an edge)")


def base_rows(composer: LearnableOntologyComposer, concepts: Tensor) -> Tensor:
    """Pre-normalization rows `z` (zeros for concepts without edges)."""
    degrees = composer.schedule.degrees[concepts]
    z = torch.zeros(concepts.numel(), composer.atomics.shape[1], device=composer.atomics.device)
    live = (degrees > 0).nonzero().flatten()
    if live.numel():
        summed, _, _, _ = composer.compose_raw(concepts[live])
        z = z.index_copy(0, live, summed)
    return z


@torch.no_grad()
def edit_effects(composer: LearnableOntologyComposer, data: HeldOutData, concepts: Tensor, deltas: Tensor) -> Tensor:
    """Per-observation change of cosine fit (edits × k) for row edits `z_c ← z_c + δ`."""
    _check_bundle(composer)
    rows = data.rows(concepts)
    if bool((rows < 0).any()):
        raise ValueError("an edited concept has no held-out observations")
    unique, inverse = torch.unique(rows, return_inverse=True)
    z = base_rows(composer, data.concepts[unique])
    observations = data.observations[unique]
    base = F.cosine_similarity(composer.predict_from_raw(z)[:, None, :], observations, dim=-1)
    edited = composer.predict_from_raw(z[inverse] + deltas)
    after = F.cosine_similarity(edited[:, None, :], observations[inverse], dim=-1)
    return after - base[inverse]


@torch.no_grad()
def edge_contributions(composer: LearnableOntologyComposer, edge_index: Tensor) -> Tensor:
    """Weighted bound vectors `m_e s_e v_e` of existing edges (bundle mode)."""
    _check_bundle(composer)
    bound, scale, _ = composer._edge_values(edge_index)
    weight = composer.edge_masses()[edge_index]
    if scale is not None:
        weight = weight * scale
    return weight[:, None] * bound


@torch.no_grad()
def hypothetical_contributions(composer: LearnableOntologyComposer, columns: Tensor, fillers: Tensor,
                               mass: float | Tensor) -> Tensor:
    vectors = composer.bind_options(columns, fillers)
    weight = torch.as_tensor(mass, dtype=vectors.dtype).expand(columns.numel())
    if composer.relation_scale is not None:
        weight = weight * composer.option_scales(columns)
    return weight[:, None] * vectors


def bootstrap_lower(values: Tensor, *, resamples: int = 1000, seed: int = 0, alpha: float = 0.025) -> tuple[Tensor, Tensor]:
    """Row-wise mean and percentile lower bound of the mean, resampling columns (rows × n)."""
    if values.ndim == 1:
        values = values[None]
    n = values.shape[1]
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(n, (resamples, n), generator=generator)
    means = values[:, index].mean(-1)
    return values.mean(1), torch.quantile(means, alpha, dim=1)


def cluster_lower(values: Tensor, clusters: Tensor, *, resamples: int = 1000, seed: int = 0,
                  alpha: float = 0.025) -> tuple[float, float, int]:
    """Mean of per-cluster means and its cluster-bootstrap lower bound."""
    labels, inverse = torch.unique(clusters, return_inverse=True)
    sums = torch.zeros(labels.numel(), dtype=values.dtype).index_add(0, inverse, values)
    counts = torch.zeros(labels.numel(), dtype=values.dtype).index_add(0, inverse, torch.ones_like(values))
    per = sums / counts
    if per.numel() < 2:
        return float(per.mean()) if per.numel() else float("nan"), float("-inf"), int(per.numel())
    mean, lower = bootstrap_lower(per, resamples=resamples, seed=seed, alpha=alpha)
    return float(mean[0]), float(lower[0]), int(per.numel())


# -- edge hypotheses ---------------------------------------------------------------------------------

@torch.no_grad()
def accept_edges(composer: LearnableOntologyComposer, edge_index: Tensor, data: HeldOutData, *,
               resamples: int = 1000, seed: int = 0, alpha: float = 0.025) -> list[Decision]:
    """Accept an existing edge if removing it worsens held-out fit (utility lower bound > 0)."""
    heads = composer.edge_heads()[edge_index]
    keep = torch.tensor([data.has(int(h)) for h in heads.tolist()], dtype=torch.bool)
    decisions: list[Decision] = []
    if not bool(keep.any()):
        return decisions
    edges = edge_index[keep]; heads = heads[keep]
    effects = edit_effects(composer, data, heads, -edge_contributions(composer, edges))
    mean, lower = bootstrap_lower(-effects, resamples=resamples, seed=seed, alpha=alpha)
    for e, m, lo in zip(edges.tolist(), mean.tolist(), lower.tolist()):
        decisions.append(Decision("edge", int(e), m, lo, effects.shape[1], lo > 0))
    return decisions


@torch.no_grad()
def removal_gains(composer: LearnableOntologyComposer, edge_index: Tensor, data: HeldOutData, *,
                  resamples: int = 1000, seed: int = 0, alpha: float = 0.025) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-edge held-out gain of removing it: (kept edges, heads, mean gain, lower bound)."""
    heads = composer.edge_heads()[edge_index]
    keep = torch.tensor([data.has(int(h)) for h in heads.tolist()], dtype=torch.bool)
    edges, heads = edge_index[keep], heads[keep]
    if not edges.numel():
        empty = torch.zeros(0)
        return edges, heads, empty, empty
    effects = edit_effects(composer, data, heads, -edge_contributions(composer, edges))
    mean, lower = bootstrap_lower(effects, resamples=resamples, seed=seed, alpha=alpha)
    return edges, heads, mean, lower


@torch.no_grad()
def relabel_gains(composer: LearnableOntologyComposer, edge_index: Tensor, columns: Tensor, data: HeldOutData, *,
                  resamples: int = 1000, seed: int = 0, alpha: float = 0.025) -> tuple[Tensor, Tensor]:
    """Held-out gain of moving each edge to another relation option (same mass)."""
    heads = composer.edge_heads()[edge_index]
    old = edge_contributions(composer, edge_index)
    mass = composer.edge_masses()[edge_index]
    new = hypothetical_contributions(composer, columns, composer.schedule.fillers[edge_index], 1.0) * mass[:, None]
    effects = edit_effects(composer, data, heads, new - old)
    return bootstrap_lower(effects, resamples=resamples, seed=seed, alpha=alpha)


# -- slot (relation) hypotheses ----------------------------------------------------------------------

@torch.no_grad()
def slot_self_test(composer: LearnableOntologyComposer, members: Tensor, column: int, data: HeldOutData, *,
                   resamples: int = 1000, seed: int = 0, alpha: float = 0.025, control_fillers: Tensor | None = None,
                   control_per_head: int = 2) -> Decision:
    """Accept a discovered relation if (i) its edges improve held-out fit of their heads (should hold)
    and (ii) beat the same heads with random fillers under the same operator (should not hold).
    Both are cluster-bootstrapped over heads; the decision needs both lower bounds > 0."""
    heads_all = composer.edge_heads()[members]
    keep = torch.tensor([data.has(int(h)) for h in heads_all.tolist()], dtype=torch.bool)
    members, heads_all = members[keep], heads_all[keep]
    if members.numel() < 2:
        return Decision("slot", column, float("nan"), float("-inf"), int(members.numel()), False,
                        {"reason": "fewer than 2 testable edges"})
    contributions = edge_contributions(composer, members)
    heads, inverse = torch.unique(heads_all, return_inverse=True)
    delta = torch.zeros(heads.numel(), contributions.shape[1]).index_add(0, inverse, contributions)
    utility = -edit_effects(composer, data, heads, -delta).mean(1)          # per head
    mean, lower, n = cluster_lower(utility, heads, resamples=resamples, seed=seed, alpha=alpha)
    fillers = composer.schedule.fillers[members]
    pool = control_fillers if control_fillers is not None else torch.unique(fillers)
    generator = torch.Generator().manual_seed(seed + 17)
    control_heads = heads.repeat_interleave(control_per_head)
    control_fill = pool[torch.randint(pool.numel(), (control_heads.numel(),), generator=generator)]
    mass = float(composer.edge_masses()[members].median())
    control = hypothetical_contributions(composer, torch.full_like(control_fill, column), control_fill, mass)
    control_utility = edit_effects(composer, data, control_heads, control).mean(1)
    control_per = torch.zeros(heads.numel()).index_add(0, torch.arange(heads.numel()).repeat_interleave(control_per_head),
                                                       control_utility) / control_per_head
    spec_mean, spec_lower, _ = cluster_lower(utility - control_per, heads, resamples=resamples, seed=seed + 1, alpha=alpha)
    accept = lower > 0 and spec_lower > 0
    return Decision("slot", column, mean, lower, n, accept,
                    {"specificity_mean": spec_mean, "specificity_lower": spec_lower, "edges": int(members.numel())})


# -- structural hypotheses (riddle, E10.6) -----------------------------------------------------------

@dataclass
class HypothesisScore:
    hypothesis: StructuralHypothesis
    positives: int
    negatives: int
    mean: float
    lower: float
    positive_hit_rate: float
    passed: bool
    predicted: set[Pair] = field(default_factory=set)


@torch.no_grad()
def score_hypotheses(composer: LearnableOntologyComposer, column: int, hypotheses: Sequence[StructuralHypothesis],
                     pairs: set[Pair], relations: dict[str, set[Pair]], data: HeldOutData, nodes: NodeMap, *,
                     mass: float, tail_pool: Sequence[Hashable], limit: int = 400, resamples: int = 1000, seed: int = 0,
                     alpha: float = 0.025) -> list[HypothesisScore]:
    """Test each hypothesis by its predictions on unseen pairs, scored on held-out observations.

    Each predicted pair `(h, t)` gets the held-out utility of adding the edge `(h, slot, t)`; a matched
    control pair `(h, t')` with a random tail gives the chance level. A prediction's contrast is
    `±(u − u_control)` (+ for "should hold", − for "should not hold"); the score is the mean contrast,
    cluster-bootstrapped over heads. A hypothesis passes if its lower bound is > 0.
    """
    heads_ok = {nodes.node_of_concept[int(c)] for c in data.concepts.tolist()}
    tails_ok = set(nodes.atomic_of_node)
    pool = [t for t in tail_pool if t in nodes.atomic_of_node]
    generator = torch.Generator().manual_seed(seed + 29)
    scores = []
    for index, hypothesis in enumerate(hypotheses):
        positive, negative = hypothesis.predict(pairs, relations, heads=heads_ok, tails=tails_ok, limit=limit, seed=seed + index)
        items = [(p, 1.0) for p in sorted(positive, key=repr)] + [(p, -1.0) for p in sorted(negative, key=repr)]
        items = [(p, s) for p, s in items if p[0] in nodes.concept_of_node and p[1] in nodes.atomic_of_node]
        if not items or not pool:
            scores.append(HypothesisScore(hypothesis, len(positive), len(negative), 0.0, float("-inf"), float("nan"),
                                          False, set(positive)))
            continue
        heads = torch.tensor([nodes.concept_of_node[p[0]] for p, _ in items])
        fillers = torch.tensor([nodes.atomic_of_node[p[1]] for p, _ in items])
        signs = torch.tensor([s for _, s in items])
        control_fill = torch.tensor([nodes.atomic_of_node[pool[i]] for i in
                                     torch.randint(len(pool), (len(items),), generator=generator).tolist()])
        columns = torch.full_like(fillers, column)
        u = edit_effects(composer, data, heads, hypothetical_contributions(composer, columns, fillers, mass)).mean(1)
        u0 = edit_effects(composer, data, heads, hypothetical_contributions(composer, columns, control_fill, mass)).mean(1)
        contrast = signs * (u - u0)
        mean, lower, _ = cluster_lower(contrast, heads, resamples=resamples, seed=seed, alpha=alpha)
        hits = ((u > u0) & (signs > 0)).sum() / max(1, int((signs > 0).sum()))
        scores.append(HypothesisScore(hypothesis, len(positive), len(negative), mean, lower,
                                      float(hits) if bool((signs > 0).any()) else float("nan"), lower > 0, set(positive)))
    return scores


def adopt(scores: Sequence[HypothesisScore]) -> HypothesisScore | None:
    """Best explanation: the passing hypothesis with the largest lower bound (None if none passes)."""
    passing = [s for s in scores if s.passed and s.hypothesis.kind != "unstructured"]
    return max(passing, key=lambda s: (s.lower, s.mean)) if passing else None
