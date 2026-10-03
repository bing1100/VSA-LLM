"""Learnable ontology composer (E10, hypothesis H-H): learnable edge masses, blank relation slots.

Extends `FrameComposer` (M0–M2) so that the ontology itself — which edges a concept has, and which
relation each edge carries — is a learnable, regularized object:

    c_i = N( Σ_{e ∈ F_i ∪ K_i} m_e · s_e · T_{ρ_e}(a_e) )

- `F_i` are the concept's **asserted** edges (the curated prior), `K_i` its **candidate** edges
  (e.g. erased edges plus distractors). `m_e ≥ 0` is a learnable edge mass with prior 1 on asserted
  and 0 on candidate edges: L2-to-prior on asserted masses (the "mapping"), L1 (+ optional L2) on
  candidate masses (the "frames"), projected to `[0, gate_max]` after every step. A candidate at mass
  0 contributes nothing, and the first gradient on its mass is minus its first-order utility
  `−⟨∂L/∂z_i, v_e⟩` (formulation §5.2), so L1 acts as a utility threshold.
- `ρ_e` is the edge's relation. Ordinary edges carry a fixed known relation. **Open** edges carry a
  soft assignment `π_e` over their allowed options — known relations and/or **blank slots** (K extra
  vector-per-relation operators with no prior) — and are bound with the mixed role
  `Σ_k π_{e,k} r_k` (exact for HRR / diagonal / MAP, which are linear in the role). Assignments are
  free per-edge logits (softmax or sparsemax) or amortized by a router on the filler (the minimal
  01d "stem-cell" pool). An entropy prior pushes edges to commit; a group-L1 prior keeps surplus
  slots empty.
- `crystallize(slot)` freezes a slot's operator and hard-assigns its edges (they become asserted
  edges at their learned mass); `reopen(slot)` reverses it; `add_blank_slot()` opens the next slot;
  `add_edges()` asserts new edges (e.g. implied by an adopted logical rule).
- Per-component learnability {atomics, relations, mapping, frames} ∈ {fixed, l2, free}.

With no candidate or open edges, all masses fixed at 1 and no relation scaling, the composition is
exactly `FrameComposer`'s (bundle, salience and attentive modes; unit-tested to 1e-6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .algebra import normalize
from .compose import FrameComposer, FrameSchedule, _occurrence_edges
from .relations import DiagonalRelation, HRRRelation, MAPRelation

SETTINGS = ("fixed", "l2", "free")
ORIGIN = {"asserted": 0, "candidate": 1, "open_asserted": 2, "rule": 3, "injected": 4}
_MASK = -1e4


@dataclass
class LearnabilityConfig:
    """Which ontology components are learnable, and how they are regularized toward the prior."""

    atomics: str = "l2"
    relations: str = "l2"
    mapping: str = "l2"
    frames: str = "l2"
    atomic_l2: float = 1.0
    relation_l2: float = 1.0
    mapping_l2: float = 0.1
    frame_l1: float = 0.05
    frame_l2: float = 0.0
    assignment_entropy: float = 0.0
    slot_group_l1: float = 0.0
    gate_max: float = 2.0
    mass_floor: float = 0.0           # per-concept prior relu(floor − Σ_e m_e)²: anchors the scale of
    mass_floor_weight: float = 1.0    # concepts made only of candidates (zero when asserted mass ≥ floor)

    def __post_init__(self) -> None:
        for name in ("atomics", "relations", "mapping", "frames"):
            if getattr(self, name) not in SETTINGS:
                raise ValueError(f"{name} must be one of {SETTINGS}")


@dataclass
class EdgeTable:
    """Union of asserted and candidate edges, sorted by head concept (CSR order)."""

    concept_count: int
    relation_count: int
    max_slots: int
    heads: Tensor
    relations: Tensor
    fillers: Tensor
    prior: Tensor
    candidate: Tensor
    open: Tensor
    options: Tensor           # (E, relation_count + max_slots) allowed relation options of open edges
    origin: Tensor

    @property
    def option_count(self) -> int:
        return self.relation_count + self.max_slots

    def schedule(self) -> FrameSchedule:
        counts = torch.bincount(self.heads, minlength=self.concept_count)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        return FrameSchedule(offsets, self.relations.clone(), self.fillers.clone())

    @classmethod
    def build(
        cls, concept_count: int, relation_count: int, *, max_slots: int = 0,
        asserted: Sequence[tuple[int, int, int]] = (), candidates: Sequence[tuple[int, int, int]] = (),
        open_asserted: Sequence[tuple[int, int, int]] = (), candidate_known_options: bool = False,
        asserted_mass: Sequence[float] | None = None,
    ) -> "EdgeTable":
        """`candidates` with relation −1 are open (assigned over slots, plus known relations if
        `candidate_known_options`); `open_asserted` edges keep their label as one option and may move
        to a slot (collapsed sub-types)."""
        rows: list[tuple[int, int, int, float, bool, bool, int]] = []
        option_rows: list[Tensor] = []
        width = relation_count + max_slots
        slot_mask = torch.zeros(width, dtype=torch.bool)
        slot_mask[relation_count:] = True
        masses = list(asserted_mass) if asserted_mass is not None else [1.0] * len(asserted)
        for (h, r, a), m in zip(asserted, masses):
            rows.append((h, r, a, float(m), False, False, ORIGIN["asserted"])); option_rows.append(torch.zeros(width, dtype=torch.bool))
        for h, r, a in candidates:
            if r < 0:
                options = slot_mask.clone()
                if candidate_known_options:
                    options[:relation_count] = True
                rows.append((h, 0, a, 0.0, True, True, ORIGIN["candidate"])); option_rows.append(options)
            else:
                rows.append((h, r, a, 0.0, True, False, ORIGIN["candidate"])); option_rows.append(torch.zeros(width, dtype=torch.bool))
        for h, r, a in open_asserted:
            options = slot_mask.clone(); options[r] = True
            rows.append((h, r, a, 1.0, False, True, ORIGIN["open_asserted"])); option_rows.append(options)
        if not rows:
            raise ValueError("the edge table is empty")
        order = sorted(range(len(rows)), key=lambda i: rows[i][0])
        rows = [rows[i] for i in order]; option_rows = [option_rows[i] for i in order]
        heads = torch.tensor([r[0] for r in rows], dtype=torch.long)
        if int(heads.max()) >= concept_count:
            raise ValueError("edge head beyond concept_count")
        return cls(concept_count, relation_count, max_slots, heads,
                   torch.tensor([r[1] for r in rows], dtype=torch.long), torch.tensor([r[2] for r in rows], dtype=torch.long),
                   torch.tensor([r[3] for r in rows], dtype=torch.float32), torch.tensor([r[4] for r in rows]),
                   torch.tensor([r[5] for r in rows]), torch.stack(option_rows), torch.tensor([r[6] for r in rows]))


def sparsemax(logits: Tensor) -> Tensor:
    """Sparsemax over the last dimension (Martins & Astudillo 2016)."""
    z, _ = torch.sort(logits, dim=-1, descending=True)
    k = torch.arange(1, logits.shape[-1] + 1, device=logits.device, dtype=logits.dtype)
    cumulative = z.cumsum(-1)
    support = (1 + k * z) > cumulative
    k_z = support.sum(-1, keepdim=True).clamp_min(1)
    tau = (cumulative.gather(-1, k_z - 1) - 1) / k_z
    return torch.clamp(logits - tau, min=0)


class LearnableOntologyComposer(FrameComposer):
    """`FrameComposer` over an `EdgeTable` with learnable edge masses, open edges and blank slots."""

    def __init__(
        self, table: EdgeTable, atomic_count: int, dimension: int, *, learn: LearnabilityConfig | None = None,
        assignment: str = "free", normalizer: str = "softmax", relation_scaling: bool = False,
        atomic_prior: Tensor | None = None, relation_prior: Tensor | None = None, slot_seed: int = 0,
        slot_init: str = "random", scale_free: bool = False, **composer_kwargs: Any,
    ) -> None:
        super().__init__(table.schedule(), atomic_count, table.relation_count, dimension, **composer_kwargs)
        if assignment not in {"free", "router"} or normalizer not in {"softmax", "sparsemax"}:
            raise ValueError("assignment ∈ {free, router}, normalizer ∈ {softmax, sparsemax}")
        if relation_scaling and self.mode != "bundle":
            raise ValueError("relation_scaling is a bundle-mode option (salience mode already weighs relations)")
        self.learn = learn or LearnabilityConfig()
        self.base_relations = table.relation_count
        self.max_slots = table.max_slots
        self.assignment_kind, self.normalizer = assignment, normalizer
        self.relation_scaling = relation_scaling
        self.slot_init = slot_init
        # `scale_free`: unit-norm relation operators and mean-centred relation log-scales, so the
        # overall scale of the asserted part cannot shrink to make candidate masses cheaper (the
        # composition is scale invariant, the L1 prior on candidate masses is not).
        self.scale_free = scale_free
        self._vector_family = isinstance(self.transform, (HRRRelation, MAPRelation, DiagonalRelation))
        if (bool(table.open.any()) or table.max_slots or scale_free) and not self._vector_family:
            raise ValueError("open edges, blank slots and scale_free need a vector-per-relation family (hrr, diagonal, map)")
        with torch.no_grad():
            if atomic_prior is not None:
                self.atomics.copy_(atomic_prior)
            if relation_prior is not None and self._vector_family:
                self.relation_vectors().copy_(relation_prior)
        self.register_buffer("atomic_prior", self.atomics.detach().clone())
        self.register_buffer("relation_prior", self.relation_vectors().detach().clone() if self._vector_family
                             else torch.zeros(0))
        self.has_atomic_prior, self.has_relation_prior = atomic_prior is not None, relation_prior is not None
        # per-edge state
        self.register_buffer("prior", table.prior.clone())
        self.register_buffer("candidate", table.candidate.clone())
        self.register_buffer("open", table.open.clone())
        self.register_buffer("options", table.options.clone())
        self.register_buffer("origin", table.origin.clone())
        self.register_buffer("pinned", torch.zeros_like(table.candidate))
        self.register_buffer("assignment_fixed", torch.full((table.heads.numel(),), -1, dtype=torch.long))
        self.register_buffer("mass_fixed", torch.zeros_like(table.candidate))
        self.register_buffer("mass_fixed_value", table.prior.clone())
        self.edge_mass = nn.Parameter(table.prior.clone())
        width = table.option_count
        self.assignment_logits = nn.Parameter(torch.zeros(table.heads.numel(), width))
        self.router = nn.Linear(dimension, width) if assignment == "router" else None
        # blank slots (pre-allocated, activated one at a time)
        generator = torch.Generator().manual_seed(slot_seed)
        self.slot_seed, self.slots_opened = slot_seed, 0
        self.slot_roles = nn.Parameter(self._fresh_roles(table.max_slots, dimension, generator))
        self.slot_scale = nn.Parameter(torch.zeros(table.max_slots), requires_grad=False)
        self.register_buffer("slot_active", torch.zeros(table.max_slots, dtype=torch.bool))
        self.register_buffer("slot_frozen", torch.zeros(table.max_slots, dtype=torch.bool))
        self.register_buffer("slot_frozen_roles", torch.zeros(table.max_slots, dimension))
        if self.mode == "salience":
            self.slot_salience = nn.Parameter(torch.zeros(table.max_slots))
        if self.mode == "attentive":
            self.slot_keys = nn.Parameter(torch.randn(table.max_slots, self.key_dimension, generator=generator)
                                          / self.key_dimension**0.5)
        self.relation_scale = nn.Parameter(torch.zeros(table.relation_count)) if relation_scaling else None
        self._mapping_init = {name: p.detach().clone() for name, p in self._mapping_parameters()}
        self.slot_records: dict[int, dict[str, Any]] = {}
        self.cards: list[dict[str, Any]] = []
        self.apply_learnability()

    # -- configuration ----------------------------------------------------------------------------

    def _fresh_roles(self, count: int, dimension: int, generator: torch.Generator) -> Tensor:
        roles = torch.randn(count, dimension, generator=generator) / dimension**0.5
        if self.slot_init == "identity":
            roles = 0.01 * roles
            roles[:, 0] += 1.0
        return roles

    def _mapping_parameters(self) -> list[tuple[str, nn.Parameter]]:
        names = ("relation_salience", "relation_keys", "query", "atomic_key", "context", "delta", "free_factor",
                 "relation_scale")
        out = []
        for name in names:
            module_or_param = getattr(self, name, None)
            if isinstance(module_or_param, nn.Parameter):
                out.append((name, module_or_param))
            elif isinstance(module_or_param, nn.Module):
                out.extend((f"{name}.{n}", p) for n, p in module_or_param.named_parameters())
        return out

    def apply_learnability(self) -> None:
        """(Re)apply component learnability: requires_grad flags and which edge masses are fixed."""
        learn = self.learn
        self.atomics.requires_grad_(learn.atomics != "fixed")
        if not self.operator.startswith("random_fixed"):
            for parameter in self.transform.parameters():
                parameter.requires_grad_(learn.relations != "fixed")
        for _, parameter in self._mapping_parameters():
            parameter.requires_grad_(learn.mapping != "fixed")
        asserted = ~self.candidate
        fixed = (asserted & (learn.mapping == "fixed")) | (self.candidate & (learn.frames == "fixed")) | self.pinned
        self.mass_fixed = fixed
        self.mass_fixed_value = self.prior.clone()
        with torch.no_grad():
            self.edge_mass.copy_(torch.where(fixed, self.prior, self.edge_mass))

    @property
    def option_count(self) -> int:
        return self.base_relations + self.max_slots

    def slot_column(self, slot: int) -> int:
        return self.base_relations + int(slot)

    # -- effective quantities ---------------------------------------------------------------------

    def edge_masses(self) -> Tensor:
        return torch.where(self.mass_fixed, self.mass_fixed_value, self.edge_mass)

    def edge_heads(self) -> Tensor:
        schedule = self.schedule
        return torch.repeat_interleave(torch.arange(schedule.concept_count, device=schedule.offsets.device),
                                       schedule.degrees)

    def slot_role_vectors(self) -> Tensor:
        """Unit-norm slot operators (a free norm would trade off against the L1 prior on masses)."""
        return torch.where(self.slot_frozen[:, None], self.slot_frozen_roles, F.normalize(self.slot_roles, dim=-1))

    def known_role_vectors(self) -> Tensor:
        roles = self.relation_vectors()
        return F.normalize(roles, dim=-1) if self.scale_free else roles

    def relation_log_scales(self) -> Tensor:
        scales = self.relation_scale
        return scales - scales[:self.base_relations].mean() if self.scale_free else scales

    def option_roles(self) -> Tensor:
        return torch.cat([self.known_role_vectors()[:self.base_relations], self.slot_role_vectors()], 0)

    def option_log_scales(self) -> Tensor:
        """Per-option log-scales: learned for seed relations (with relation scaling), 0 for slots
        (a slot's magnitude lives in its edge masses only)."""
        known = self.relation_log_scales()[:self.base_relations] if self.relation_scale is not None \
            else self.slot_scale.new_zeros(self.base_relations)
        return torch.cat([known, self.slot_scale.new_zeros(self.max_slots)], 0)

    def _closed_bound(self, relations: Tensor, atoms: Tensor) -> Tensor:
        if self.scale_free:
            return self._bind_roles(self.known_role_vectors()[relations], atoms)
        return self.transform(relations, atoms)

    def active_columns(self) -> Tensor:
        return torch.cat([torch.ones(self.base_relations, dtype=torch.bool, device=self.slot_active.device),
                          self.slot_active])

    def assignment(self, edge_index: Tensor) -> Tensor:
        """Relation-option distribution `π_e` (rows of zeros when an edge has no active option)."""
        allowed = self.options[edge_index] & self.active_columns()[None]
        if self.router is not None:
            logits = self.router(self.atomic_vectors()[self.schedule.fillers[edge_index]])
        else:
            logits = self.assignment_logits[edge_index]
        masked = logits.masked_fill(~allowed, _MASK)
        probabilities = torch.softmax(masked, -1) if self.normalizer == "softmax" else sparsemax(masked)
        probabilities = probabilities * allowed * allowed.any(-1, keepdim=True)
        fixed = self.assignment_fixed[edge_index]
        hard = F.one_hot(fixed.clamp_min(0), self.option_count).to(probabilities.dtype)
        return torch.where((fixed >= 0)[:, None], hard, probabilities)

    def _bind_roles(self, roles: Tensor, fillers: Tensor) -> Tensor:
        if isinstance(self.transform, HRRRelation):
            return self.transform.algebra.bind(roles, fillers)
        return roles * fillers

    def bind_options(self, columns: Tensor, fillers: Tensor) -> Tensor:
        """Bound vectors `T_k(a)` for explicit (option column, filler atomic) pairs (hypothetical edges)."""
        roles = self.option_roles()[columns]
        return self._bind_roles(roles, self.atomic_vectors()[fillers])

    def option_scales(self, columns: Tensor) -> Tensor:
        return self.option_log_scales()[columns].exp()

    def _edge_values(self, edge_index: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Bound vectors, multiplicative scales (or None) and the assignment of open edges."""
        schedule = self.schedule
        relations, fillers = schedule.relations[edge_index], schedule.fillers[edge_index]
        is_open = self.open[edge_index]
        atoms = self.atomic_vectors()[fillers]
        if not bool(is_open.any()):
            bound = self._closed_bound(relations, atoms)
            scale = self.relation_log_scales()[relations].exp() if self.relation_scale is not None else None
            return bound, scale, None
        closed = (~is_open).nonzero().flatten(); opened = is_open.nonzero().flatten()
        bound = atoms.new_zeros(atoms.shape)
        if closed.numel():
            bound = bound.index_copy(0, closed, self._closed_bound(relations[closed], atoms[closed]))
        pi = self.assignment(edge_index[opened])
        mixed = pi @ self.option_roles()
        bound = bound.index_copy(0, opened, self._bind_roles(mixed, atoms[opened]))
        scale = None
        if self.relation_scale is not None:
            log_scale = atoms.new_zeros(edge_index.numel())
            if closed.numel():
                log_scale = log_scale.index_copy(0, closed, self.relation_log_scales()[relations[closed]])
            log_scale = log_scale.index_copy(0, opened, pi @ self.option_log_scales())
            scale = log_scale.exp()
        return bound, scale, pi

    # -- composition ------------------------------------------------------------------------------

    def _gated_softmax(self, scores: Tensor, mass: Tensor, segments: Tensor, count: int) -> Tensor:
        maxima = scores.new_full((count,), float("-inf")).scatter_reduce(0, segments, scores, reduce="amax",
                                                                         include_self=True)
        exponents = mass * (scores - maxima[segments]).exp()
        totals = scores.new_zeros(count).index_add(0, segments, exponents)
        mass_total = scores.new_zeros(count).index_add(0, segments, mass)
        return mass_total[segments] * exponents / totals[segments].clamp_min(1e-30)

    def _weights(self, edge_index: Tensor, segments: Tensor, concept_ids: Tensor, bound: Tensor, mass: Tensor,
                 pi: Tensor | None, context: Tensor | None) -> Tensor:
        count = concept_ids.numel()
        if self.mode == "bundle":
            return mass
        relations = self.schedule.relations[edge_index]
        opened = self.open[edge_index].nonzero().flatten() if pi is not None else None
        if self.mode == "salience":
            logits = self.relation_salience[relations]
            if opened is not None and opened.numel():
                table = torch.cat([self.relation_salience[:self.base_relations], self.slot_salience])
                logits = logits.index_copy(0, opened, pi @ table)
            return self._gated_softmax(logits, mass, segments, count)
        static = bound.new_zeros(count, bound.shape[-1]).index_add(0, segments, mass.unsqueeze(-1) * bound)
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
        relation_keys = self.relation_keys[relations]
        if opened is not None and opened.numel():
            table = torch.cat([self.relation_keys[:self.base_relations], self.slot_keys])
            relation_keys = relation_keys.index_copy(0, opened, pi @ table)
        keys = relation_keys + self.atomic_key(self.atomic_vectors()[self.schedule.fillers[edge_index]])
        scores = (factor[segments] * keys).sum(-1) / math.sqrt(self.key_dimension)
        if self.weight_mode == "raw":
            return scores * mass
        if self.weight_mode == "sigmoid":
            return 2 * torch.sigmoid(scores) * mass
        return self._gated_softmax(scores / self.log_temperature.exp(), mass, segments, count)

    def compose_raw(self, concept_ids: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pre-normalization rows `z`, effective weights, edge indices and weighted edge vectors."""
        edge_index, segments = _occurrence_edges(self.schedule, concept_ids)
        needed, position = torch.unique(edge_index, return_inverse=True)
        bound_needed, scale_needed, pi_needed = self._edge_values(needed)
        bound = bound_needed[position]
        mass = self.edge_masses()[edge_index]
        if scale_needed is not None:
            mass = mass * scale_needed[position]
        pi = None
        if pi_needed is not None:
            # π rows of the open occurrences of `edge_index`, in order (`_weights` expects that).
            rank = torch.cumsum(self.open[needed].long(), 0) - 1
            pi = pi_needed[rank[position[self.open[edge_index]]]]
        weights = self._weights(edge_index, segments, concept_ids, bound, mass, pi, context)
        contributions = weights.unsqueeze(-1) * bound
        summed = bound.new_zeros(concept_ids.numel(), bound.shape[-1]).index_add(0, segments, contributions)
        return summed, weights, edge_index, contributions

    def _compose(self, concept_ids: Tensor, context: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        live = self.schedule.degrees[concept_ids] > 0
        if not bool(live.all()):
            # concepts without any edge (e.g. an empty seed ontology) compose to the zero row
            index = live.nonzero().flatten()
            rows = self.atomics.new_zeros(concept_ids.numel(), self.atomics.shape[1])
            if not index.numel():
                empty = torch.zeros(0, dtype=torch.long, device=concept_ids.device)
                return rows, rows.new_zeros(0), empty
            sub_rows, weights, edge_index = self._compose(concept_ids[index],
                                                          None if context is None else context[index])
            return rows.index_copy(0, index, sub_rows), weights, edge_index
        summed, weights, edge_index, _ = self.compose_raw(concept_ids, context)
        if self.capture_usage and summed.requires_grad:
            summed.retain_grad()
            segments = _occurrence_edges(self.schedule, concept_ids)[1]
            self.captured.append({"summed": summed, "segments": segments, "edge_index": edge_index,
                                  "weights": weights.detach(), "concept_ids": concept_ids})
        return normalize(summed), weights, edge_index

    def predict_from_raw(self, summed: Tensor) -> Tensor:
        rows = normalize(summed)
        return self.projector(rows) if self.projector is not None else rows

    def project_(self) -> None:
        """Projection after an optimizer step: masses into `[0, gate_max]`."""
        with torch.no_grad():
            self.edge_mass.clamp_(0.0, self.learn.gate_max)

    # -- regularization ---------------------------------------------------------------------------

    def penalty(self) -> Tensor:
        learn = self.learn
        total = self.edge_mass.new_zeros(())
        if learn.atomics == "l2" and self.has_atomic_prior:
            total = total + learn.atomic_l2 * (self.atomics - self.atomic_prior).square().sum()
        if learn.relations == "l2" and self.has_relation_prior and self._vector_family:
            current = self.relation_vectors()[:self.base_relations]
            total = total + learn.relation_l2 * (current - self.relation_prior[:self.base_relations]).square().sum()
        mass = self.edge_masses()
        learnable = ~self.mass_fixed
        asserted = learnable & ~self.candidate
        if learn.mapping == "l2":
            total = total + learn.mapping_l2 * (mass - self.prior)[asserted].square().sum()
            for name, parameter in self._mapping_parameters():
                total = total + learn.mapping_l2 * (parameter - self._mapping_init[name].to(parameter)).square().sum()
        candidates = learnable & self.candidate
        if learn.frames == "l2":
            total = total + learn.frame_l1 * mass[candidates].sum() + learn.frame_l2 * mass[candidates].square().sum()
        if learn.mass_floor > 0 and bool(learnable.any()):
            heads = self.edge_heads()
            totals = mass.new_zeros(self.schedule.concept_count).index_add(0, heads, mass)
            has_learnable = torch.zeros(self.schedule.concept_count, dtype=torch.bool, device=mass.device)
            has_learnable[heads[learnable]] = True
            deficit = (learn.mass_floor - totals).clamp_min(0)[has_learnable]
            total = total + learn.mass_floor_weight * deficit.square().sum()
        soft = (self.open & (self.assignment_fixed < 0)).nonzero().flatten()
        if soft.numel() and (learn.assignment_entropy or learn.slot_group_l1):
            pi = self.assignment(soft)
            if learn.assignment_entropy:
                entropy = -(pi * pi.clamp_min(1e-12).log()).sum(-1)
                total = total + learn.assignment_entropy * (mass[soft] * entropy).sum()
            if learn.slot_group_l1:
                usage = mass[soft, None] * pi[:, self.base_relations:]
                live = self.slot_active & ~self.slot_frozen
                total = total + learn.slot_group_l1 * (usage.square().sum(0) + 1e-12).sqrt()[live].sum()
        return total

    def parameter_groups(self) -> dict[str, int]:
        groups = {"atomic": 0, "relation": 0, "global": 0, "concept_local": 0, "edge_mass": 0, "assignment": 0,
                  "slot": 0}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "atomics":
                group = "atomic"
            elif name.startswith("transform.") or name in {"relation_keys", "relation_salience", "relation_scale"}:
                group = "relation"
            elif name in {"free_factor", "delta"}:
                group = "concept_local"
            elif name == "edge_mass":
                group = "edge_mass"
            elif name == "assignment_logits" or name.startswith("router."):
                group = "assignment"
            elif name.startswith("slot_"):
                group = "slot"
            else:
                group = "global"
            groups[group] += parameter.numel()
        return groups

    # -- slots ------------------------------------------------------------------------------------

    def seed_candidate_masses(self, init_mass: float, column: int | None = None) -> int:
        """Give soft open candidates (optionally only those allowing `column`) at least `init_mass`.

        A blank relation has a random operator and its candidates have mass 0, so neither receives an
        informative gradient; a small initial mass on every candidate starts the slot as an
        over-inclusive hypothesis that the L1 prior then prunes."""
        rows = self.open & self.candidate & (self.assignment_fixed < 0) & ~self.mass_fixed
        if column is not None:
            rows = rows & self.options[:, column]
        with torch.no_grad():
            self.edge_mass[rows] = self.edge_mass[rows].clamp_min(init_mass)
        return int(rows.sum())

    def seed_empty_concepts(self, init_mass: float) -> int:
        """Give the learnable candidates of concepts without any asserted mass at least `init_mass`.

        A concept whose edges are all candidates at mass 0 composes to `z = 0`, where the row
        normalization has no usable gradient; a small uniform start makes `z ≠ 0`."""
        heads = self.edge_heads()
        masses = self.edge_masses().detach()
        asserted = torch.zeros(self.schedule.concept_count, device=masses.device).index_add(
            0, heads, torch.where(self.candidate, torch.zeros_like(masses), masses))
        rows = self.candidate & ~self.mass_fixed & (asserted[heads] <= 0)
        with torch.no_grad():
            self.edge_mass[rows] = self.edge_mass[rows].clamp_min(init_mass)
        return int(rows.sum())

    def add_blank_slot(self, *, step: int | None = None, init_mass: float = 0.0) -> int:
        """Activate the next pre-allocated slot with a fresh random operator; returns its index."""
        free = (~self.slot_active).nonzero().flatten()
        if not free.numel():
            raise RuntimeError("no blank slot left (raise max_slots)")
        slot = int(free[0])
        self.slots_opened += 1
        generator = torch.Generator().manual_seed(self.slot_seed * 7919 + self.slots_opened)
        with torch.no_grad():
            self.slot_roles[slot] = self._fresh_roles(1, self.slot_roles.shape[1], generator)[0].to(self.slot_roles)
            self.slot_scale[slot] = 0.0
            self.assignment_logits[:, self.slot_column(slot)] = 0.0
        self.slot_active[slot] = True
        self.slot_frozen[slot] = False
        if init_mass > 0:
            self.seed_candidate_masses(init_mass, self.slot_column(slot))
        self.cards.append({"event": "open_slot", "slot": slot, "step": step})
        return slot

    def slot_members(self, slot: int, *, min_assignment: float = 0.5, min_mass: float = 0.25,
                     hard_only: bool = False) -> Tensor:
        """Edge indices assigned to a slot: hard-assigned ones plus committed soft ones."""
        column = self.slot_column(slot)
        hard = (self.assignment_fixed == column).nonzero().flatten()
        if hard_only:
            return hard
        soft = (self.open & (self.assignment_fixed < 0)).nonzero().flatten()
        if not soft.numel():
            return hard
        with torch.no_grad():
            pi = self.assignment(soft)
            mass = self.edge_masses()[soft]
        keep = (pi.argmax(-1) == column) & (pi[:, column] >= min_assignment) & (mass >= min_mass)
        return torch.cat([hard, soft[keep]]).sort().values

    def crystallize(self, slot: int, *, min_assignment: float = 0.5, min_mass: float = 0.25,
                    record: dict[str, Any] | None = None, step: int | None = None) -> dict[str, Any]:
        """Freeze the slot's operator and hard-assign its committed edges (they become asserted)."""
        members = self.slot_members(slot, min_assignment=min_assignment, min_mass=min_mass)
        column = self.slot_column(slot)
        with torch.no_grad():
            self.slot_frozen_roles[slot] = F.normalize(self.slot_roles[slot].detach(), dim=-1)
            mass = self.edge_masses()[members].detach()
        self.slot_frozen[slot] = True
        self.assignment_fixed[members] = column
        self.candidate[members] = False
        self.prior[members] = mass
        self.apply_learnability()
        self.slot_records[slot] = dict(record or {}, slot=slot, edges=int(members.numel()), step=step)
        card = {"event": "crystallize", "slot": slot, "edges": int(members.numel()), "step": step,
                **{k: v for k, v in (record or {}).items() if k in {"hypothesis", "description", "utility"}}}
        self.cards.append(card)
        return card

    def absorb(self, slot: int, *, min_assignment: float = 0.5, min_mass: float = 0.25,
               step: int | None = None) -> int:
        """Consolidate late members: soft candidates now committed to a crystallized slot become its
        hard-assigned edges (the slot's operator stays frozen)."""
        if not bool(self.slot_frozen[slot]):
            return 0
        column = self.slot_column(slot)
        members = self.slot_members(slot, min_assignment=min_assignment, min_mass=min_mass)
        late = members[self.assignment_fixed[members] != column]
        if not late.numel():
            return 0
        with torch.no_grad():
            mass = self.edge_masses()[late].detach()
        self.assignment_fixed[late] = column
        self.candidate[late] = False
        self.prior[late] = mass
        self.apply_learnability()
        self.cards.append({"event": "absorb", "slot": slot, "edges": int(late.numel()), "step": step})
        return int(late.numel())

    def reopen(self, slot: int, *, step: int | None = None, reason: str = "") -> dict[str, Any]:
        """Unfreeze a crystallized slot and release its edges back to soft assignment."""
        column = self.slot_column(slot)
        members = (self.assignment_fixed == column).nonzero().flatten()
        self.slot_frozen[slot] = False
        self.assignment_fixed[members] = -1
        from_candidates = self.origin[members] == ORIGIN["candidate"]
        self.candidate[members[from_candidates]] = True
        self.prior[members[from_candidates]] = 0.0
        with torch.no_grad():
            self.assignment_logits[members] = 0.0
            self.assignment_logits[members, column] = 3.0
        self.apply_learnability()
        self.slot_records.pop(slot, None)
        card = {"event": "reopen", "slot": slot, "edges": int(members.numel()), "step": step, "reason": reason}
        self.cards.append(card)
        return card

    def deactivate_slot(self, slot: int, *, step: int | None = None, reason: str = "") -> dict[str, Any]:
        """Close a slot: release its edges and remove it from every option set."""
        if bool(self.slot_frozen[slot]):
            self.reopen(slot, step=step, reason=reason)
        self.slot_active[slot] = False
        with torch.no_grad():
            self.assignment_logits[:, self.slot_column(slot)] = 0.0
        card = {"event": "close_slot", "slot": slot, "step": step, "reason": reason}
        self.cards.append(card)
        return card

    def edge_options(self, edge_index: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Current relation option (column) of each edge and its confidence (1 for fixed relations)."""
        edge_index = torch.arange(self.edge_mass.numel(), device=self.edge_mass.device) if edge_index is None else edge_index
        columns = self.schedule.relations[edge_index].clone()
        confidence = torch.ones(edge_index.numel(), device=columns.device)
        opened = self.open[edge_index].nonzero().flatten()
        if opened.numel():
            with torch.no_grad():
                pi = self.assignment(edge_index[opened])
            best, arg = pi.max(-1)
            columns[opened] = torch.where(best > 0, arg, torch.full_like(arg, -1))
            confidence[opened] = best
        return columns, confidence

    # -- structural edits -------------------------------------------------------------------------

    def add_edges(self, heads: Sequence[int], columns: Sequence[int], fillers: Sequence[int], *,
                  mass: float | Sequence[float] = 1.0, origin: str = "rule", candidate: bool = False) -> Tensor:
        """Append asserted (or candidate) edges with a fixed relation option; returns their edge ids.

        Known-relation columns become ordinary edges, slot columns open edges hard-assigned to the slot.
        The `edge_mass` and `assignment_logits` Parameters are replaced: rebuild the optimizer.
        """
        if not len(heads):
            return torch.zeros(0, dtype=torch.long)
        device = self.edge_mass.device
        heads_t = torch.as_tensor(list(heads), dtype=torch.long, device=device)
        columns_t = torch.as_tensor(list(columns), dtype=torch.long, device=device)
        fillers_t = torch.as_tensor(list(fillers), dtype=torch.long, device=device)
        count = heads_t.numel()
        masses = (torch.full((count,), float(mass), device=device) if isinstance(mass, (int, float))
                  else torch.as_tensor(mass, dtype=torch.float32, device=device).reshape(count))
        is_slot = columns_t >= self.base_relations
        relations = torch.where(is_slot, torch.zeros_like(columns_t), columns_t)
        options = torch.zeros(count, self.option_count, dtype=torch.bool, device=device)
        options[is_slot.nonzero().flatten(), columns_t[is_slot]] = True
        fixed = torch.where(is_slot, columns_t, torch.full_like(columns_t, -1))
        old_heads = self.edge_heads()
        all_heads = torch.cat([old_heads, heads_t])
        order = torch.argsort(all_heads, stable=True)
        cat = lambda old, new: torch.cat([old, new.to(old.dtype)])[order]  # noqa: E731
        schedule = self.schedule
        counts = torch.bincount(all_heads, minlength=schedule.concept_count)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])
        new_relations = cat(schedule.relations, relations); new_fillers = cat(schedule.fillers, fillers_t)
        self.frame_offsets, self.frame_relations, self.frame_fillers = offsets, new_relations, new_fillers
        candidate_t = torch.full((count,), bool(candidate), device=device)
        self.prior = cat(self.prior, torch.zeros(count, device=device) if candidate else masses)
        self.candidate = cat(self.candidate, candidate_t)
        self.open = cat(self.open, is_slot)
        self.options = cat(self.options, options)
        self.origin = cat(self.origin, torch.full((count,), ORIGIN[origin], device=device))
        self.pinned = cat(self.pinned, torch.zeros(count, dtype=torch.bool, device=device))
        self.assignment_fixed = cat(self.assignment_fixed, fixed)
        self.mass_fixed = cat(self.mass_fixed, torch.zeros(count, dtype=torch.bool, device=device))
        self.mass_fixed_value = cat(self.mass_fixed_value, masses)
        self.edge_mass = nn.Parameter(cat(self.edge_mass.detach(), torch.zeros(count, device=device) if candidate else masses))
        self.assignment_logits = nn.Parameter(cat(self.assignment_logits.detach(),
                                                  torch.zeros(count, self.option_count, device=device)))
        self.apply_learnability()
        inverse = torch.argsort(order)
        return inverse[old_heads.numel():]

    def set_edge_option(self, edge: int, column: int) -> None:
        """Move one edge to another relation option (known relation or slot), keeping its mass."""
        edge, column = int(edge), int(column)
        if column < self.base_relations and not bool(self.open[edge]):
            relations = self.schedule.relations.clone(); relations[edge] = column
            self.frame_relations = relations
            return
        if column < self.base_relations:
            if bool(self.options[edge, column]):
                self.assignment_fixed[edge] = column
                return
            relations = self.schedule.relations.clone(); relations[edge] = column
            self.frame_relations = relations
            self.open[edge] = False
            self.assignment_fixed[edge] = -1
            return
        self.open[edge] = True
        self.options[edge, column] = True
        self.assignment_fixed[edge] = column

    def unpin(self, edge_index: Tensor) -> None:
        """Release pinned edges back to learnable candidates (e.g. evidence that arrives later)."""
        self.pinned[edge_index] = False
        self.candidate[edge_index] = True
        self.prior[edge_index] = 0.0
        self.apply_learnability()

    def pin_masses(self, edge_index: Tensor, value: float | Tensor) -> None:
        """Fix edge masses permanently (e.g. a removed edge pinned at 0)."""
        self.prior[edge_index] = torch.as_tensor(value, dtype=self.prior.dtype, device=self.prior.device)
        self.pinned[edge_index] = True
        self.candidate[edge_index] = False
        self.apply_learnability()

    def edge_records(self) -> list[dict[str, Any]]:
        """One record per edge (head, column, filler, mass, origin) for analysis."""
        heads = self.edge_heads().tolist()
        columns, confidence = self.edge_options()
        mass = self.edge_masses().detach()
        return [{"edge": i, "head": h, "column": int(c), "filler": int(a), "mass": float(m), "confidence": float(p),
                 "origin": int(o), "candidate": bool(k)}
                for i, (h, c, a, m, p, o, k) in enumerate(zip(heads, columns.tolist(), self.schedule.fillers.tolist(),
                                                               mass.tolist(), confidence.tolist(), self.origin.tolist(),
                                                               self.candidate.tolist()))]


@dataclass
class TrainResult:
    optimizer: torch.optim.Optimizer
    fit: float
    history: list[dict[str, Any]] = field(default_factory=list)


def cosine_fit(prediction: Tensor, targets: Tensor) -> Tensor:
    """Mean `1 − cos` of each prediction row against each of its observations (`targets`: n × k × d)."""
    return (1 - F.cosine_similarity(prediction[:, None, :], targets, dim=-1)).mean()


def train_composer(
    composer: LearnableOntologyComposer, concepts: Tensor, targets: Tensor, *, steps: int, lr: float,
    optimizer: torch.optim.Optimizer | None = None, tracker: Any = None, tracker_phase: Callable[[int], bool] | None = None,
    callback: Callable[[int, LearnableOntologyComposer], dict[str, Any] | None] | None = None, callback_every: int = 0,
    after_step: Callable[[int], None] | None = None,
) -> TrainResult:
    """Full-batch training on `targets` (concepts × observations × d) with the composer's penalty.

    `tracker` is an optional M3 `DevelopmentalDictionary` (begin/observe/grow around each step while
    `tracker_phase(step)` is true).
    """
    if optimizer is None:
        optimizer = torch.optim.Adam([p for p in composer.parameters() if p.requires_grad], lr=lr)
    history: list[dict[str, Any]] = []
    n = concepts.numel()
    fit = float("nan")
    for step in range(steps):
        active = tracker is not None and (tracker_phase is None or tracker_phase(step))
        if active:
            tracker.begin()
        prediction = composer(concepts)
        # rows of concepts whose composition is exactly zero have no usable gradient (the row
        # normalization is singular there); they are left out of the fit term.
        valid = prediction.detach().norm(dim=-1) > 1e-6
        fit_t = cosine_fit(prediction[valid], targets[valid]) if bool(valid.any()) else prediction.sum() * 0
        loss = fit_t + composer.penalty() / n
        optimizer.zero_grad()
        loss.backward()
        if active:
            tracker.observe()
        optimizer.step()
        composer.project_()
        if active:
            tracker.grow()
        if after_step is not None:
            after_step(step)
        fit = float(fit_t.detach())
        if callback is not None and callback_every and (step % callback_every == 0 or step == steps - 1):
            entry = callback(step, composer)
            if entry:
                history.append(entry)
    return TrainResult(optimizer, fit, history)
