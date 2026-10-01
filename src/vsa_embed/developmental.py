"""Developmental dictionary (M3): split, merge, freeze and allocate dictionary vectors.

Implements `resources/plan-improvement/formulation.md` §3 for a `FrameComposer`:

- **Screening.** EMAs of each vector's gradient `m` and absolute gradient `a` give coherence
  `κ = ‖m‖₁ / ‖a‖₁` and activity `α = ‖a‖₁`. Active vectors with low coherence (pulled hard in
  inconsistent directions) become candidates.
- **Per-usage gradients.** For atomic `a` used by concept `i` through edge `e`,
  `∂L/∂a|_i = w_{i,e} T_{r_e}ᵀ g_i` with `g_i = ∂L/∂z_i`; for a relation vector,
  `∂L/∂r|_e = w_{i,e} J_T(a_e)ᵀ g_i`. They are recorded only for candidates.
- **Split gain.** With usage momenta `m_u`, the between-usage scatter `S = Σ m_u m_uᵀ − m mᵀ/|U|`
  gives `v*` (top eigenvector, computed as the top right singular vector of the centred usage
  matrix) and `G* = Σ_u |⟨m_u, v*⟩| − |Σ_u ⟨m_u, v*⟩|`. A split is accepted only if `G*` beats a
  permutation null in which usage labels are shuffled over the recorded per-step contributions.
- **Children.** `θ± = θ ± ε‖θ‖ v*`; observed usages go to the child matching the sign of
  `⟨m_u, v*⟩`, unobserved ones to the child whose observed usages' composed vectors they most
  resemble. Optimizer state rows are duplicated. Every event is logged as a card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .algebra import HRRAlgebra
from .compose import FrameComposer, FrameSchedule


@dataclass
class DevelopmentalConfig:
    target: str = "atomics"             # "atomics" or "relations"
    beta: float = 0.9
    screen_every: int = 50
    activity_percentile: float = 0.5
    coherence_threshold: float = 0.5
    max_candidates: int = 16
    min_usages: int = 4
    min_contributions: int = 20
    max_splits_per_round: int = 4
    growth_budget: float = 0.25
    cooldown: int = 100
    epsilon: float = 0.05
    permutations: int = 200
    p_value: float = 0.01
    test: str = "permutation"           # or "anderson" (G-means-style normality test on projections)
    route_unobserved: str = "context"   # or "parent": unobserved usages keep the unsplit parent vector
    merge_cosine: float = 0.98
    freeze_activity: float = 0.0        # freeze vectors whose activity stays below this
    freeze_patience: int = 500
    seed: int = 0


def split_gain(usage_momenta: Tensor, direction: Tensor) -> float:
    """`G(v) = Σ_u |⟨m_u, v⟩| − |Σ_u ⟨m_u, v⟩|` (≥ 0; 0 iff all usages agree in sign)."""
    projections = usage_momenta @ direction
    return float(projections.abs().sum() - projections.sum().abs())


def top_between_usage_direction(usage_momenta: Tensor) -> Tensor:
    """Top eigenvector of `S = Σ m_u m_uᵀ − m mᵀ/|U|` via the centred usage matrix's SVD."""
    centred = usage_momenta - usage_momenta.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    return vh[0]


def usage_momenta(usages: Tensor, gradients: Tensor, usage_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Sum recorded contributions per usage label; returns (labels, momenta)."""
    labels, inverse = torch.unique(usages if usage_ids is None else usage_ids, return_inverse=True)
    momenta = gradients.new_zeros(labels.numel(), gradients.shape[-1]).index_add(0, inverse, gradients)
    return labels, momenta


def split_test(
    usages: Tensor, gradients: Tensor, *, permutations: int, generator: torch.Generator,
) -> dict[str, Any]:
    """Split gain along `v*` and its permutation p-value for one candidate's contributions."""
    labels, momenta = usage_momenta(usages, gradients)
    direction = top_between_usage_direction(momenta)
    gain = split_gain(momenta, direction)
    exceed = 0
    for _ in range(permutations):
        shuffled = usages[torch.randperm(usages.numel(), generator=generator)]
        _, null_momenta = usage_momenta(shuffled, gradients)
        if split_gain(null_momenta, top_between_usage_direction(null_momenta)) >= gain:
            exceed += 1
    return {"labels": labels, "momenta": momenta, "direction": direction, "gain": gain,
            "p_value": (1 + exceed) / (1 + permutations)}


def anderson_split_test(usages: Tensor, gradients: Tensor) -> dict[str, Any]:
    """G-means-style alternative: Anderson–Darling normality test of usage projections on `v*`.

    A unimodal (normal) projection means no split; rejection at 1% accepts one. Uses SciPy.
    """
    from scipy.stats import anderson
    labels, momenta = usage_momenta(usages, gradients)
    direction = top_between_usage_direction(momenta)
    projections = (momenta @ direction).double().numpy()
    result = anderson(projections, dist="norm")
    critical_1pct = float(result.critical_values[list(result.significance_level).index(1.0)])
    statistic = float(result.statistic)
    return {"labels": labels, "momenta": momenta, "direction": direction,
            "gain": split_gain(momenta, direction), "statistic": statistic,
            "p_value": 0.0 if statistic > critical_1pct else 1.0}


def _replace_parameter(optimizer: torch.optim.Optimizer | None, old: nn.Parameter, new: nn.Parameter,
                       copy_rows: Tensor) -> None:
    """Point the optimizer at `new` and duplicate per-row state for appended rows."""
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["params"] = [new if p is old else p for p in group["params"]]
    state = optimizer.state.pop(old, None)
    if state:
        grown = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.ndim == old.ndim and value.shape[0] == old.shape[0]:
                grown[key] = torch.cat([value, value[copy_rows]], 0)
            else:
                grown[key] = value
        optimizer.state[new] = grown


class DevelopmentalDictionary:
    """Track gradient statistics of a composer's dictionary and grow it by tested splits.

    Per training step: `begin()` before the forward pass, `observe()` after `backward()` (before
    `optimizer.step()`), and `grow()` after `optimizer.step()`.
    """

    def __init__(self, composer: FrameComposer, optimizer: torch.optim.Optimizer | None,
                 config: DevelopmentalConfig | None = None) -> None:
        self.composer, self.optimizer = composer, optimizer
        self.config = config or DevelopmentalConfig()
        if self.config.target not in {"atomics", "relations"}:
            raise ValueError("target must be 'atomics' or 'relations'")
        if self.config.target == "relations" and not hasattr(composer.transform, ("roles")) \
                and not hasattr(composer.transform, "diagonal"):
            raise ValueError("relation splitting needs a vector-per-relation family (hrr, diagonal, map)")
        self.generator = torch.Generator().manual_seed(self.config.seed)
        count, dim = self._parameter().shape
        self.momentum = torch.zeros(count, dim)
        self.absolute = torch.zeros(count, dim)
        self.initial_count = count
        self.step = 0
        self.splits = 0
        self.candidates: set[int] = set()
        self.records: dict[int, list[tuple[Tensor, Tensor]]] = {}   # id -> [(usages, grads)]
        self.cooldown_until: dict[int, int] = {}
        self.frozen = torch.zeros(count, dtype=torch.bool)
        self.low_activity_steps = torch.zeros(count, dtype=torch.long)
        self.provisional: set[int] = set()
        self.siblings: list[tuple[int, int]] = []
        self.cards: list[dict[str, Any]] = []

    # -- bookkeeping ----------------------------------------------------------------------------

    def _parameter(self) -> nn.Parameter:
        return self.composer.atomics if self.config.target == "atomics" else self.composer.relation_vectors()

    def _grow_state(self, rows: Tensor) -> None:
        extra = rows.numel()
        self.momentum = torch.cat([self.momentum, self.momentum[rows]], 0)
        self.absolute = torch.cat([self.absolute, self.absolute[rows]], 0)
        self.frozen = torch.cat([self.frozen, torch.zeros(extra, dtype=torch.bool)])
        self.low_activity_steps = torch.cat([self.low_activity_steps, torch.zeros(extra, dtype=torch.long)])

    def coherence(self) -> Tensor:
        return self.momentum.abs().sum(1) / self.absolute.abs().sum(1).clamp_min(1e-12)

    def activity(self) -> Tensor:
        return self.absolute.sum(1)

    # -- per-step API -----------------------------------------------------------------------------

    def begin(self) -> None:
        self.composer.capture_usage = True
        self.composer.captured.clear()

    def observe(self) -> None:
        """Update statistics from the current gradients (call after backward)."""
        parameter = self._parameter()
        grad = parameter.grad
        if grad is None:
            self.composer.captured.clear()
            return
        if bool(self.frozen.any()):
            grad[self.frozen.to(grad.device)] = 0
        g = grad.detach().float().cpu()
        beta = self.config.beta
        self.momentum.mul_(beta).add_(g, alpha=1 - beta)
        self.absolute.mul_(beta).add_(g.abs(), alpha=1 - beta)
        if self.candidates:
            for vector_id, usages, grads in self._usage_gradients(self.candidates):
                self.records.setdefault(vector_id, []).append((usages, grads))
        self.composer.captured.clear()
        self.step += 1

    def _usage_gradients(self, wanted: set[int]) -> list[tuple[int, Tensor, Tensor]]:
        """Per-(vector, usage) gradient sums for this step, for the wanted vector ids."""
        out: dict[int, list[tuple[Tensor, Tensor]]] = {}
        schedule = self.composer.schedule
        wanted_t = torch.tensor(sorted(wanted), device=schedule.fillers.device)
        for record in self.composer.captured:
            if record["summed"].grad is None:
                continue
            edge_index, segments = record["edge_index"], record["segments"]
            ids = schedule.fillers[edge_index] if self.config.target == "atomics" else schedule.relations[edge_index]
            keep = torch.isin(ids, wanted_t)
            if not bool(keep.any()):
                continue
            edge_index, segments, ids = edge_index[keep], segments[keep], ids[keep]
            g = record["summed"].grad[segments] * record["weights"][keep][:, None]
            relations = schedule.relations[edge_index]
            atomics = self.composer.atomic_vectors().detach()
            if self.config.target == "atomics":
                grads = self.composer.transform.adjoint(relations, g).detach()
                if self.composer.normalize_atomics:
                    raw = self.composer.atomics.detach()[ids]
                    unit = F.normalize(raw, dim=-1)
                    grads = (grads - unit * (unit * grads).sum(-1, keepdim=True)) / raw.norm(dim=-1, keepdim=True)
                usages = record["concept_ids"][segments]
            else:
                fillers = atomics[schedule.fillers[edge_index]]
                if hasattr(self.composer.transform, "roles"):
                    grads = HRRAlgebra().unbind(g, fillers).detach()   # ∂(r ⊛ a)/∂r adjoint
                else:
                    grads = (g * fillers).detach()                      # diagonal / map
                usages = edge_index
            for vector_id in ids.unique().tolist():
                mask = ids == vector_id
                labels, momenta = usage_momenta(usages[mask], grads[mask])
                out.setdefault(vector_id, []).append((labels.cpu(), momenta.float().cpu()))
        result = []
        for vector_id, parts in out.items():
            labels = torch.cat([p[0] for p in parts]); grads = torch.cat([p[1] for p in parts])
            labels, grads = usage_momenta(labels, grads)
            result.append((vector_id, labels, grads))
        return result

    # -- growth -------------------------------------------------------------------------------

    def budget_left(self) -> int:
        return int(self.config.growth_budget * self.initial_count) - self.splits

    def grow(self) -> list[dict[str, Any]]:
        """Every `screen_every` steps: test recorded candidates, split accepted ones, re-screen."""
        if self.step == 0 or self.step % self.config.screen_every:
            return []
        accepted = self._test_candidates()
        events = [self._split(vector_id, result) for vector_id, result in accepted]
        self._update_freezing()
        self._screen()
        return events

    def _test_candidates(self) -> list[tuple[int, dict[str, Any]]]:
        tested = []
        for vector_id, entries in self.records.items():
            usages = torch.cat([u for u, _ in entries])
            grads = torch.cat([g for _, g in entries])
            if usages.unique().numel() < self.config.min_usages or usages.numel() < self.config.min_contributions:
                continue
            if self.config.test == "anderson":
                result = anderson_split_test(usages, grads)
            else:
                result = split_test(usages, grads, permutations=self.config.permutations, generator=self.generator)
            result["contributions"] = int(usages.numel())
            tested.append((vector_id, result))
        self.records.clear()
        significant = [(v, r) for v, r in tested if r["p_value"] < self.config.p_value]
        significant.sort(key=lambda item: -item[1]["gain"])
        return significant[:max(0, min(self.config.max_splits_per_round, self.budget_left()))]

    def _screen(self) -> None:
        count = self.momentum.shape[0]
        activity, coherence = self.activity(), self.coherence()
        active = activity > 0
        if not bool(active.any()):
            self.candidates = set()
            return
        floor = torch.quantile(activity[active], self.config.activity_percentile)
        eligible = [
            i for i in range(count)
            if active[i] and activity[i] >= floor and coherence[i] < self.config.coherence_threshold
            and not self.frozen[i] and self.cooldown_until.get(i, -1) <= self.step
        ]
        eligible.sort(key=lambda i: float(coherence[i]))
        self.candidates = set(eligible[:self.config.max_candidates])

    def _split(self, vector_id: int, result: dict[str, Any]) -> dict[str, Any]:
        composer, schedule = self.composer, self.composer.schedule
        direction = result["direction"]
        labels, momenta = result["labels"], result["momenta"]
        positive = (momenta @ direction) >= 0
        old = self._parameter()
        theta = old.detach()[vector_id].float().cpu()
        offset = self.config.epsilon * theta.norm() * direction
        keep_parent = self.config.route_unobserved == "parent"
        old_keys = getattr(composer, "relation_keys", None)
        if keep_parent and self.config.target == "relations":
            ids = composer.add_relation_copies(torch.tensor([vector_id, vector_id]), torch.stack([offset, -offset]))
            _replace_parameter(self.optimizer, old, composer.relation_vectors(), torch.tensor([vector_id, vector_id]))
            if old_keys is not None:
                _replace_parameter(self.optimizer, old_keys, composer.relation_keys, torch.tensor([vector_id, vector_id]))
            positive_id, new_id = int(ids[0]), int(ids[1])
            uses = (schedule.relations == vector_id).nonzero().flatten()
            usage_of_edge = uses
        elif keep_parent:
            # Two new children for observed usages; the parent keeps its value for usages without
            # evidence (held-out or rare), for which the collapsed vector is the best guess.
            ids = composer.add_atomics(torch.stack([theta + offset, theta - offset]).to(old))
            _replace_parameter(self.optimizer, old, composer.atomics, torch.tensor([vector_id, vector_id]))
            positive_id, new_id = int(ids[0]), int(ids[1])
            uses = (schedule.fillers == vector_id).nonzero().flatten()
            usage_of_edge = torch.repeat_interleave(
                torch.arange(schedule.concept_count, device=schedule.offsets.device), schedule.degrees)[uses]
        elif self.config.target == "atomics":
            new_id = int(composer.add_atomics((theta - offset)[None].to(old))[0])
            with torch.no_grad():
                composer.atomics[vector_id] += offset.to(composer.atomics)
            _replace_parameter(self.optimizer, old, composer.atomics, torch.tensor([vector_id]))
            uses = (schedule.fillers == vector_id).nonzero().flatten()
            usage_of_edge = torch.repeat_interleave(
                torch.arange(schedule.concept_count, device=schedule.offsets.device), schedule.degrees)[uses]
        else:
            new_id = int(composer.add_relation_copies(torch.tensor([vector_id]), (-offset)[None])[0])
            new_param = composer.relation_vectors()
            with torch.no_grad():
                new_param[vector_id] += offset.to(new_param)
            _replace_parameter(self.optimizer, old, new_param, torch.tensor([vector_id]))
            if old_keys is not None:
                _replace_parameter(self.optimizer, old_keys, composer.relation_keys, torch.tensor([vector_id]))
            uses = (schedule.relations == vector_id).nonzero().flatten()
            usage_of_edge = uses
        relations, fillers = schedule.relations.clone(), schedule.fillers.clone()
        target = fillers if self.config.target == "atomics" else relations
        if keep_parent:
            self._grow_state(torch.tensor([vector_id, vector_id]))
            observed = torch.isin(usage_of_edge, labels.to(usage_of_edge.device))
            sign_of = {int(l): bool(p) for l, p in zip(labels.tolist(), positive.tolist())}
            to_positive = torch.tensor([bool(o) and sign_of.get(int(u), False) for o, u in
                                        zip(observed.tolist(), usage_of_edge.tolist())], dtype=torch.bool)
            go_negative = observed.cpu() & ~to_positive
            target[uses[to_positive.to(uses.device)]] = positive_id
            target[uses[go_negative.to(uses.device)]] = new_id
            children = (positive_id, new_id)
        else:
            self._grow_state(torch.tensor([vector_id]))
            go_negative = self._assign(uses, usage_of_edge, labels, positive)
            target[uses[go_negative]] = new_id
            children = (vector_id, new_id)
        composer.set_schedule(FrameSchedule(schedule.offsets, relations, fillers))
        self.splits += 1
        for child in children:
            self.cooldown_until[child] = self.step + self.config.cooldown
        self.siblings.append(children)
        card = {
            "event": "split", "target": self.config.target, "step": self.step, "parent": vector_id,
            "children": list(children), "parent_kept_for_unobserved": keep_parent,
            "gain": result["gain"], "p_value": result["p_value"],
            "contributions": result["contributions"], "observed_usages": int(labels.numel()),
            "usages_to_new_child": int(go_negative.sum()), "usages_kept": int((~go_negative).sum()),
            "direction": direction.tolist(),
        }
        self.cards.append(card)
        return card

    def _assign(self, uses: Tensor, usage_of_edge: Tensor, labels: Tensor, positive: Tensor) -> Tensor:
        """Which edges move to the new child: observed by sign, unobserved by similarity."""
        labels_dev = labels.to(usage_of_edge.device)
        observed = torch.isin(usage_of_edge, labels_dev)
        go_negative = torch.zeros(uses.numel(), dtype=torch.bool, device=uses.device)
        sign_of = {int(l): bool(p) for l, p in zip(labels.tolist(), positive.tolist())}
        for position in observed.nonzero().flatten().tolist():
            go_negative[position] = not sign_of[int(usage_of_edge[position])]
        unobserved = (~observed).nonzero().flatten()
        if unobserved.numel() and self.config.target == "atomics":
            with torch.no_grad():
                features = self._frame_context(usage_of_edge, int(self.composer.schedule.fillers[uses[0]]))
                observed_idx = observed.nonzero().flatten()
                negative = go_negative[observed_idx]
                if bool(negative.any()) and bool((~negative).any()):
                    centre_neg = features[observed_idx[negative]].mean(0)
                    centre_pos = features[observed_idx[~negative]].mean(0)
                    rows = features[unobserved]
                    go_negative[unobserved] = (rows @ centre_neg) > (rows @ centre_pos)
        return go_negative

    def _frame_context(self, concepts: Tensor, excluded_atomic: int) -> Tensor:
        """Mean of each concept's other atomics (the split atomic left out): the frame context.

        Unobserved usages carry no gradient; their frame's other fillers are the evidence of
        which sense they use (formulation §3.4). The concept's full composition is a poor router:
        for trained concepts it is fitted to targets that already contain the sense.
        """
        schedule = self.composer.schedule
        atomics = F.normalize(self.composer.atomic_vectors().detach(), dim=-1)
        rows = []
        for concept in concepts.tolist():
            fillers = schedule.fillers[int(schedule.offsets[concept]):int(schedule.offsets[concept + 1])]
            others = fillers[fillers != excluded_atomic]
            rows.append(F.normalize(atomics[others].mean(0), dim=0) if others.numel()
                        else atomics.new_zeros(atomics.shape[1]))
        return torch.stack(rows)

    # -- consolidation ------------------------------------------------------------------------

    def consolidate(self) -> list[dict[str, Any]]:
        """Merge sibling pairs that became near-identical; the second row is retired (frozen)."""
        events = []
        parameter = self._parameter()
        schedule = self.composer.schedule
        for first, second in list(self.siblings):
            vectors = parameter.detach()
            if self.frozen[second] or float(F.cosine_similarity(vectors[first], vectors[second], dim=0)) < self.config.merge_cosine:
                continue
            with torch.no_grad():
                parameter[first] = (vectors[first] + vectors[second]) / 2
            relations, fillers = schedule.relations.clone(), schedule.fillers.clone()
            target = fillers if self.config.target == "atomics" else relations
            target[target == second] = first
            self.composer.set_schedule(FrameSchedule(schedule.offsets, relations, fillers))
            schedule = self.composer.schedule
            self.frozen[second] = True
            self.siblings.remove((first, second))
            card = {"event": "merge", "target": self.config.target, "step": self.step,
                    "kept": first, "retired": second}
            self.cards.append(card); events.append(card)
        return events

    def _update_freezing(self) -> None:
        if self.config.freeze_activity <= 0:
            return
        low = self.activity() < self.config.freeze_activity
        self.low_activity_steps = torch.where(low, self.low_activity_steps + self.config.screen_every,
                                              torch.zeros_like(self.low_activity_steps))
        newly = (self.low_activity_steps >= self.config.freeze_patience) & ~self.frozen
        for vector_id in newly.nonzero().flatten().tolist():
            self.frozen[vector_id] = True
            self.cards.append({"event": "freeze", "target": self.config.target, "step": self.step, "id": vector_id})

    def allocate(self, vector: Tensor, *, reason: str = "") -> int:
        """Add a new atomic (e.g. a filler an ontology update introduced), flagged provisional."""
        if self.config.target != "atomics":
            raise ValueError("allocate adds atomics")
        old = self.composer.atomics
        new_id = int(self.composer.add_atomics(vector[None].to(old))[0])
        _replace_parameter(self.optimizer, old, self.composer.atomics, torch.tensor([0]))
        if self.optimizer is not None:
            state = self.optimizer.state.get(self.composer.atomics)
            if state:
                for key, value in state.items():
                    if torch.is_tensor(value) and value.ndim == 2:
                        value[new_id] = 0
        self._grow_state(torch.tensor([0]))
        self.momentum[new_id] = 0; self.absolute[new_id] = 0
        self.provisional.add(new_id)
        self.cards.append({"event": "allocate", "step": self.step, "id": new_id, "reason": reason})
        return new_id
