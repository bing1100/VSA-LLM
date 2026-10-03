"""Self-supervised relation discovery over a `LearnableOntologyComposer` (E10.0 (c), E10.4–E10.8).

Everything here reads a `LearningContext`: training targets of training concepts, held-out
observations (`HeldOutData`), node keys and the model's own names for its seed relations. No gold
relation labels, gold pairs or audit observations enter: discovered relations are interpreted
structurally, accepted by held-out self-tests and named by their adopted rule.

- `additive_curriculum`: open one blank slot, train, interpret (riddle: generate structural
  hypotheses from the captured pairs, test each by its predictions on held-out pairs, adopt the
  best), self-test, crystallize with the adopted rule enforced (e.g. symmetric / transitive
  closure) or close the slot; repeat until `patience` consecutive slots are rejected.
- `all_at_once`: K slots trained together for the same number of steps, then interpreted and
  self-tested the same way.
- `dream_pass` (E10.5): an offline revision pass over the current ontology — remove or relabel
  asserted edges, split / merge / reopen / remove relations — each proposal accepted only if it
  improves held-out fit (bootstrap lower bound > 0) relative to an equally long control refit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

import torch
from torch import Tensor
from torch.nn import functional as F

from .developmental import top_between_usage_direction
from .learnable_ontology import ORIGIN, LearnableOntologyComposer, cosine_fit, train_composer
from .ontology_hypotheses import (
    HypothesisSettings, Pair, StructuralHypothesis, describe_pairs, generate_hypotheses, pair_signatures,
)
from .self_test import (
    HeldOutData, NodeMap, adopt, cluster_lower, edit_effects, relabel_gains, removal_gains, score_hypotheses,
    slot_self_test,
)


@dataclass
class LearningContext:
    """What the learning loop may read (never gold labels, never audit observations)."""

    train_concepts: Tensor
    train_targets: Tensor              # (n, k, d) training observations
    heldout: HeldOutData               # validation observations (self-tests)
    nodes: NodeMap
    relation_names: list[str]          # the model's own names for its seed relations (by column)
    rule_heads: set[int]               # concepts that may receive rule-implied edges
    seed: int = 0


@dataclass
class DreamSettings:
    refit_steps: int = 150
    lr: float = 0.02
    edges: bool = True
    relabel: bool = True
    split: bool = True
    merge: bool = True
    reopen: bool = True
    remove_slot: bool = True
    min_split_members: int = 8
    split_epsilon: float = 0.3
    merge_cosine: float = 0.3
    merge_margin: float = 0.0
    max_relabel_edges: int = 4000


@dataclass
class DiscoverySettings:
    steps_per_round: int = 1500
    lr: float = 0.02
    min_assignment: float = 0.5
    min_mass: float = 0.25
    resamples: int = 1000
    alpha: float = 0.025
    patience: int = 1
    max_rounds: int = 5
    enforce_rule: bool = True
    hypothesis_limit: int = 400
    rule_limit: int = 5000
    trajectory_every: int = 100
    dream_between_rounds: bool = False
    slot_init_mass: float = 0.1        # initial mass of open candidates when a slot opens
    absorb: bool = True                # crystallized slots take late soft members at each round's end
    refine: bool = True                # self-tested split of a new slot before it is consolidated
    max_refinements: int = 3
    riddle_full: bool = False          # also score the full hypothesis space (for the E10.6 analysis only)
    hypotheses: HypothesisSettings = field(default_factory=HypothesisSettings)
    dream: DreamSettings = field(default_factory=DreamSettings)
    namer: Callable[[dict], str] | None = None


# -- reading the model's own ontology ---------------------------------------------------------------

def edge_pairs(composer: LearnableOntologyComposer, edges: Tensor, nodes: NodeMap) -> set[Pair]:
    heads = composer.edge_heads()[edges].tolist(); fillers = composer.schedule.fillers[edges].tolist()
    return {nodes.edge_pair(h, a) for h, a in zip(heads, fillers)}


def own_relations(composer: LearnableOntologyComposer, ctx: LearningContext, *, exclude_slot: int | None = None,
                  min_mass: float = 0.25) -> dict[str, set[Pair]]:
    """The model's relations by its own names: seed labels (asserted edges with that nominal label)
    and crystallized slots (`slot<k>`)."""
    relations: dict[str, set[Pair]] = {}
    masses = composer.edge_masses().detach()
    asserted = (~composer.candidate) & (masses >= min_mass)
    nominal = composer.schedule.relations
    seed_edges = asserted & ((~composer.open) | (composer.origin == ORIGIN["open_asserted"]))
    for column, name in enumerate(ctx.relation_names):
        edges = (seed_edges & (nominal == column)).nonzero().flatten()
        if edges.numel():
            relations[name] = edge_pairs(composer, edges, ctx.nodes)
    for slot in composer.slot_frozen.nonzero().flatten().tolist():
        if slot == exclude_slot:
            continue
        members = composer.slot_members(slot, hard_only=True)
        members = members[masses[members] >= min_mass]
        if members.numel():
            relations[f"slot{slot}"] = edge_pairs(composer, members, ctx.nodes)
    return relations


def tail_pool(composer: LearnableOntologyComposer, ctx: LearningContext) -> list[Hashable]:
    """Tails a relation could plausibly take: fillers of open edges (the candidate pool)."""
    fillers = torch.unique(composer.schedule.fillers[composer.open]).tolist()
    return [ctx.nodes.node_of_atomic[a] for a in fillers]


# -- interpreting and deciding about one slot --------------------------------------------------------

@torch.no_grad()
def interpret_slot(composer: LearnableOntologyComposer, slot: int, ctx: LearningContext,
                   settings: DiscoverySettings) -> dict[str, Any]:
    """Riddle step (E10.6) and acceptance self-test for one slot; no gold is read."""
    members = composer.slot_members(slot, min_assignment=settings.min_assignment, min_mass=settings.min_mass)
    column = composer.slot_column(slot)
    pairs = edge_pairs(composer, members, ctx.nodes)
    relations = own_relations(composer, ctx, exclude_slot=slot)
    signature = pair_signatures(pairs, relations) if pairs else {}
    pool = tail_pool(composer, ctx)
    mass = float(composer.edge_masses()[members].median()) if members.numel() else 1.0
    hypotheses = generate_hypotheses(pairs, relations, settings.hypotheses) if pairs else []
    scores = score_hypotheses(composer, column, hypotheses, pairs, relations, ctx.heldout, ctx.nodes, mass=mass,
                              tail_pool=pool, limit=settings.hypothesis_limit, resamples=settings.resamples,
                              seed=ctx.seed + slot, alpha=settings.alpha) if pairs else []
    adopted = adopt(scores)
    full_scores = []
    if settings.riddle_full and pairs:
        full = generate_hypotheses(pairs, relations, HypothesisSettings(include_all=True))
        full_scores = score_hypotheses(composer, column, full, pairs, relations, ctx.heldout, ctx.nodes, mass=mass,
                                       tail_pool=pool, limit=settings.hypothesis_limit, resamples=settings.resamples,
                                       seed=ctx.seed + slot, alpha=settings.alpha)
    control = torch.tensor([ctx.nodes.atomic_of_node[t] for t in pool]) if pool else None
    decision = slot_self_test(composer, members, column, ctx.heldout, resamples=settings.resamples,
                              seed=ctx.seed + 101 * (slot + 1), alpha=settings.alpha, control_fillers=control)
    description = describe_pairs(pairs, adopted.hypothesis if adopted else None, signature, namer=settings.namer)
    return {"slot": slot, "column": column, "members": members, "pairs": pairs, "signature": signature,
            "scores": scores, "full_scores": full_scores, "adopted": adopted, "decision": decision, "mass": mass, "description": description}


def enforce_rule(composer: LearnableOntologyComposer, slot: int, adopted: StructuralHypothesis, ctx: LearningContext,
                 settings: DiscoverySettings, *, mass: float, pairs: set[Pair], relations: dict[str, set[Pair]]) -> dict[str, int]:
    """Crystallize with the adopted rule: add implied pairs (closure) or prune violations."""
    column = composer.slot_column(slot)
    heads_ok = {ctx.nodes.node_of_concept[c] for c in ctx.rule_heads}
    tails_ok = set(ctx.nodes.atomic_of_node)
    added = pruned = 0
    if adopted.kind in {"symmetric", "transitive", "inverse_of", "composition_of"}:
        positive, _ = adopted.predict(pairs, relations, heads=heads_ok, tails=tails_ok, limit=settings.rule_limit,
                                      seed=ctx.seed)
        existing = edge_pairs(composer, composer.slot_members(slot, hard_only=True), ctx.nodes)
        new = sorted(positive - existing, key=repr)
        if new:
            composer.add_edges([ctx.nodes.concept_of_node[h] for h, _ in new], [column] * len(new),
                               [ctx.nodes.atomic_of_node[t] for _, t in new], mass=mass, origin="rule")
            added = len(new)
    elif adopted.kind in {"functional", "one_to_one", "antisymmetric"}:
        members = composer.slot_members(slot, hard_only=True)
        masses = composer.edge_masses().detach()[members]
        heads = composer.edge_heads()[members]; fillers = composer.schedule.fillers[members]
        keys = heads if adopted.kind == "functional" else fillers
        losers = []
        if adopted.kind == "antisymmetric":
            index = {(int(h), int(a)): i for i, (h, a) in enumerate(zip(heads.tolist(), fillers.tolist()))}
            for (h, a), i in index.items():
                # the reverse partner (a, h) exists only when the filler is a concept atomic
                partner_head = ctx.nodes.concept_of_node.get(ctx.nodes.node_of_atomic[a])
                partner_fill = ctx.nodes.atomic_of_node.get(ctx.nodes.node_of_concept[h])
                j = index.get((partner_head, partner_fill)) if partner_head is not None else None
                if j is not None and masses[i] < masses[j]:
                    losers.append(i)
        else:
            for key in torch.unique(keys).tolist():
                group = (keys == key).nonzero().flatten()
                if group.numel() > 1:
                    best = group[masses[group].argmax()]
                    losers.extend(int(g) for g in group.tolist() if int(g) != int(best))
        if losers:
            composer.pin_masses(members[torch.tensor(sorted(set(losers)))], 0.0)
            pruned = len(set(losers))
    return {"rule_added": added, "rule_pruned": pruned}


def decide_slot(composer: LearnableOntologyComposer, record: dict[str, Any], ctx: LearningContext,
                settings: DiscoverySettings, *, step: int) -> dict[str, Any]:
    """Crystallize an accepted slot (with its adopted rule enforced) or close a rejected one."""
    slot, decision, adopted = record["slot"], record["decision"], record["adopted"]
    out = {"accepted": bool(decision.accept), "rule_added": 0, "rule_pruned": 0}
    if decision.accept:
        relations = own_relations(composer, ctx, exclude_slot=slot)
        composer.crystallize(slot, min_assignment=settings.min_assignment, min_mass=settings.min_mass, step=step,
                             record={"hypothesis": adopted.hypothesis.name if adopted else "unstructured",
                                     "description": record["description"],
                                     "utility": {"mean": decision.mean, "lower": decision.lower}})
        if settings.enforce_rule and adopted is not None:
            out.update(enforce_rule(composer, slot, adopted.hypothesis, ctx, settings, mass=record["mass"],
                                    pairs=record["pairs"], relations=relations))
    else:
        composer.deactivate_slot(slot, step=step, reason="held-out self-test rejected")
    return out


def summarize_record(record: dict[str, Any], outcome: dict[str, Any], *, step: int, round_index: int) -> dict[str, Any]:
    """JSON-friendly summary of a slot decision (pairs kept for evaluation by the caller)."""
    d = record["decision"]
    return {
        "round": round_index, "step": step, "slot": record["slot"], "size": len(record["pairs"]),
        "accept": bool(d.accept), "utility_mean": d.mean, "utility_lower": d.lower,
        "specificity_lower": d.extra.get("specificity_lower"), "adopted": record["adopted"].hypothesis.name if record["adopted"] else None,
        "hypotheses": [{"name": s.hypothesis.name, "support": s.hypothesis.support, "positives": s.positives,
                        "negatives": s.negatives, "mean": s.mean, "lower": s.lower, "passed": s.passed}
                       for s in record["scores"]],
        "hypotheses_full": [{"name": s.hypothesis.name, "positives": s.positives, "negatives": s.negatives,
                             "mean": s.mean, "lower": s.lower, "passed": s.passed} for s in record.get("full_scores", [])],
        "description": record["description"], **outcome,
        "pairs": sorted(record["pairs"], key=repr),
        "predicted_by_adopted": sorted(record["adopted"].predicted, key=repr) if record["adopted"] else [],
    }


# -- procedures --------------------------------------------------------------------------------------

def _trajectory(ctx: LearningContext, log: list[dict[str, Any]], offset: int, *, dense_until: int = 200,
                every: int = 100) -> Callable:
    """Log each live slot over training: committed members and the soft mass `m_e π_{e,k}` it holds
    (pairs with soft mass ≥ 0.02), densely early (every call before `dense_until`) and every `every`
    steps afterwards."""
    def callback(step: int, composer: LearnableOntologyComposer) -> None:
        if step > dense_until and step % every:
            return
        soft = (composer.open & (composer.assignment_fixed < 0)).nonzero().flatten()
        with torch.no_grad():
            pi = composer.assignment(soft) if soft.numel() else None
            masses = composer.edge_masses()[soft] if soft.numel() else None
        for slot in (composer.slot_active & ~composer.slot_frozen).nonzero().flatten().tolist():
            members = composer.slot_members(slot, min_assignment=0.5, min_mass=0.25)
            column = composer.slot_column(slot)
            soft_pairs = []
            if pi is not None:
                values = masses * pi[:, column]
                keep = (values >= 0.02).nonzero().flatten()
                heads = composer.edge_heads()[soft[keep]].tolist(); fills = composer.schedule.fillers[soft[keep]].tolist()
                soft_pairs = [[ctx.nodes.edge_pair(h, a), round(float(v), 4)] for h, a, v in zip(heads, fills, values[keep].tolist())]
            log.append({"step": offset + step, "slot": slot, "members": int(members.numel()),
                        "assigned_mass": float(sum(v for _, v in soft_pairs)),
                        "pairs": sorted(edge_pairs(composer, members, ctx.nodes), key=repr), "soft": soft_pairs})
    return callback


def refine_slot(composer: LearnableOntologyComposer, slot: int, ctx: LearningContext, settings: DiscoverySettings, *,
                seed: int = 0) -> tuple[LearnableOntologyComposer, int | None, dict[str, Any]]:
    """Refinement before consolidation: try splitting an (over-inclusive) slot along its top
    between-edge role-gradient direction; keep the split only if held-out fit improves over an equally
    long control refit (cluster-bootstrap lower bound > 0). Returns the refit composer."""
    dream = settings.dream
    column = composer.slot_column(slot)
    members = composer.slot_members(slot, min_assignment=settings.min_assignment, min_mass=settings.min_mass)
    if members.numel() < dream.min_split_members or not bool((~composer.slot_active).any()):
        return composer, None, {"tried": False}
    proposal = copy.deepcopy(composer)
    for e in members.tolist():
        proposal.set_edge_option(e, column)
    info = _propose_split(proposal, ctx, column, dream, members=members)
    if info is None:
        return composer, None, {"tried": False}
    control = copy.deepcopy(composer)
    train_composer(control, ctx.train_concepts, ctx.train_targets, steps=dream.refit_steps, lr=dream.lr)
    train_composer(proposal, ctx.train_concepts, ctx.train_targets, steps=dream.refit_steps, lr=dream.lr)
    for s in info["refreeze"]:
        proposal.slot_frozen[s] = False                      # still active; decided later this round
    heads = torch.unique(composer.edge_heads()[members])
    heads = heads[torch.tensor([ctx.heldout.has(int(c)) for c in heads.tolist()], dtype=torch.bool)]
    if heads.numel() < 2:
        return control, None, {"tried": True, "accepted": False}
    diff = _heldout_fit(proposal, ctx.heldout, heads) - _heldout_fit(control, ctx.heldout, heads)
    mean, lower, n = cluster_lower(diff, heads, resamples=settings.resamples, seed=seed, alpha=settings.alpha)
    record = {"tried": True, "slot": slot, "moved": info["moved"], "kept": info["kept"], "mean": mean, "lower": lower,
              "accepted": bool(lower > 0)}
    if lower > 0:
        # release the hard split assignment so both halves stay soft until their own decision
        for s in (slot, info["new_slot"]):
            col = proposal.slot_column(s)
            hard = (proposal.assignment_fixed == col) & (proposal.origin == ORIGIN["candidate"])
            proposal.assignment_fixed[hard] = -1
            with torch.no_grad():
                proposal.assignment_logits[hard] = 0.0
                proposal.assignment_logits[hard, col] = 3.0
        proposal.cards.append({"event": "refine_split", **{k: v for k, v in record.items() if k != "tried"},
                               "new_slot": info["new_slot"]})
        return proposal, info["new_slot"], record
    return control, None, record


def additive_curriculum(composer: LearnableOntologyComposer, ctx: LearningContext, settings: DiscoverySettings, *,
                        on_stage: Callable[[int, LearnableOntologyComposer], LearnableOntologyComposer | None] | None = None,
                        dream_log: list | None = None) -> tuple[LearnableOntologyComposer, dict[str, Any]]:
    """Open → train → interpret → self-test → crystallize (rule enforced) or close; repeat."""
    rounds, trajectory = [], []
    step = rejections = 0
    for round_index in range(settings.max_rounds):
        if on_stage is not None:
            replaced = on_stage(round_index, composer)
            composer = replaced or composer
        if not bool((~composer.slot_active).any()):
            break
        slot = composer.add_blank_slot(step=step, init_mass=settings.slot_init_mass)
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=settings.steps_per_round, lr=settings.lr,
                       callback=_trajectory(ctx, trajectory, step, every=settings.trajectory_every), callback_every=25)
        step += settings.steps_per_round
        absorbed = sum(composer.absorb(s, min_assignment=settings.min_assignment, min_mass=settings.min_mass, step=step)
                       for s in composer.slot_frozen.nonzero().flatten().tolist()) if settings.absorb else 0
        round_slots, queue, refinements = [slot], [slot], []
        while queue and settings.refine and len(refinements) < settings.max_refinements:
            current = queue.pop(0)
            composer, new_slot, info = refine_slot(composer, current, ctx, settings, seed=ctx.seed + 31 * step + current)
            if info.get("tried"):
                refinements.append(info)
                step += 2 * settings.dream.refit_steps
            if new_slot is not None:
                round_slots.append(new_slot); queue += [current, new_slot]
        accepted_any = False
        for current in round_slots:
            record = interpret_slot(composer, current, ctx, settings)
            outcome = decide_slot(composer, record, ctx, settings, step=step)
            outcome["absorbed_by_earlier"] = absorbed
            outcome["refinements"] = refinements if current == slot else []
            outcome["split_from"] = slot if current != slot else None
            rounds.append(summarize_record(record, outcome, step=step, round_index=round_index))
            accepted_any |= outcome["accepted"]
        rejections = 0 if accepted_any else rejections + 1
        if settings.dream_between_rounds:
            composer, revisions = dream_pass(composer, ctx, settings.dream, seed=ctx.seed + 1000 * round_index)
            step += settings.dream.refit_steps
            if dream_log is not None:
                dream_log.append({"round": round_index, "step": step, "revisions": revisions})
        if rejections >= settings.patience:
            break
    return composer, {"rounds": rounds, "trajectory": trajectory, "steps": step}


def all_at_once(composer: LearnableOntologyComposer, ctx: LearningContext, settings: DiscoverySettings, *, slots: int,
                steps: int) -> tuple[LearnableOntologyComposer, dict[str, Any]]:
    """K slots trained together for `steps`, then interpreted / self-tested / crystallized."""
    opened = [composer.add_blank_slot(step=0, init_mass=settings.slot_init_mass) for _ in range(slots)]
    trajectory: list[dict[str, Any]] = []
    train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=steps, lr=settings.lr,
                   callback=_trajectory(ctx, trajectory, 0, every=settings.trajectory_every), callback_every=25)
    sizes = {slot: int(composer.slot_members(slot).numel()) for slot in opened}
    rounds = []
    for index, slot in enumerate(sorted(opened, key=lambda k: -sizes[k])):
        record = interpret_slot(composer, slot, ctx, settings)
        outcome = decide_slot(composer, record, ctx, settings, step=steps)
        rounds.append(summarize_record(record, outcome, step=steps, round_index=index))
    return composer, {"rounds": rounds, "trajectory": trajectory, "steps": steps}


# -- dreaming (E10.5) --------------------------------------------------------------------------------

def _heldout_fit(composer: LearnableOntologyComposer, data: HeldOutData, concepts: Tensor) -> Tensor:
    """Per-concept mean held-out cosine (concepts without edges score 0)."""
    from .self_test import base_rows
    with torch.no_grad():
        rows = data.rows(concepts)
        z = base_rows(composer, concepts)
        prediction = composer.predict_from_raw(z)
        return F.cosine_similarity(prediction[:, None, :], data.observations[rows], dim=-1).mean(1)


def _relation_members(composer: LearnableOntologyComposer, column: int) -> Tensor:
    """Asserted edges currently carried by a relation option (seed label or slot)."""
    columns, _ = composer.edge_options()
    masses = composer.edge_masses().detach()
    return ((columns == column) & ~composer.candidate & (masses > 0)).nonzero().flatten()


def _role_gradients(composer: LearnableOntologyComposer, ctx: LearningContext, members: Tensor) -> tuple[Tensor, Tensor]:
    """Per-edge role gradient (unbound residual gradient) for members headed by training concepts."""
    heads = composer.edge_heads()[members]
    index = {int(c): i for i, c in enumerate(ctx.train_concepts.tolist())}
    keep = torch.tensor([int(h) in index for h in heads.tolist()], dtype=torch.bool)
    members, heads = members[keep], heads[keep]
    if not members.numel():
        return members, torch.zeros(0, composer.atomics.shape[1])
    with torch.enable_grad():
        z, _, _, _ = composer.compose_raw(ctx.train_concepts)
        z = z.detach().requires_grad_(True)
        loss = cosine_fit(composer.predict_from_raw(z), ctx.train_targets) * ctx.train_concepts.numel()
        (grad,) = torch.autograd.grad(loss, z)
    rows = torch.tensor([index[int(h)] for h in heads.tolist()])
    atoms = composer.atomic_vectors().detach()[composer.schedule.fillers[members]]
    g = grad[rows]
    if hasattr(composer.transform, "algebra"):
        role_grad = composer.transform.algebra.unbind(g, atoms)
    else:
        role_grad = g * atoms
    return members, role_grad


def dream_pass(composer: LearnableOntologyComposer, ctx: LearningContext, settings: DreamSettings, *,
               seed: int = 0, resamples: int = 1000, alpha: float = 0.025) -> tuple[LearnableOntologyComposer, list[dict[str, Any]]]:
    """One offline revision pass; returns the revised (refit) composer and the revision log."""
    log: list[dict[str, Any]] = []
    data = ctx.heldout
    # 1. edge revisions: remove, then relabel, asserted edges whose change improves held-out fit.
    if settings.edges:
        masses = composer.edge_masses().detach()
        asserted = ((~composer.candidate) & (~composer.pinned) & (masses > 0)).nonzero().flatten()
        edges, heads, mean, lower = removal_gains(composer, asserted, data, resamples=resamples, seed=seed, alpha=alpha)
        removed = edges[lower > 0]
        if removed.numel():
            composer.pin_masses(removed, 0.0)
        log.append({"proposal": "remove_edges", "tested": int(edges.numel()), "accepted": int(removed.numel()),
                    "edges": edge_records(composer, removed, ctx)})
    if settings.relabel:
        masses = composer.edge_masses().detach()
        asserted = ((~composer.candidate) & (~composer.pinned) & (masses > 0)).nonzero().flatten()
        heads = composer.edge_heads()[asserted]
        asserted = asserted[torch.tensor([data.has(int(h)) for h in heads.tolist()], dtype=torch.bool)][:settings.max_relabel_edges]
        current, _ = composer.edge_options(asserted)
        active = composer.active_columns().nonzero().flatten()
        live = torch.tensor([c for c in active.tolist() if c < composer.base_relations or bool(composer.slot_frozen[c - composer.base_relations])])
        best_gain = torch.full((asserted.numel(),), float("-inf")); best_col = torch.full((asserted.numel(),), -1)
        for column in live.tolist():
            mask = current != column
            if not bool(mask.any()):
                continue
            mean, lower = relabel_gains(composer, asserted[mask], torch.full((int(mask.sum()),), column), data,
                                        resamples=resamples, seed=seed + column, alpha=alpha)
            idx = mask.nonzero().flatten()
            better = lower > best_gain[idx]
            best_gain[idx[better]] = lower[better]; best_col[idx[better]] = column
        moved = (best_gain > 0).nonzero().flatten()
        for i in moved.tolist():
            composer.set_edge_option(int(asserted[i]), int(best_col[i]))
        log.append({"proposal": "relabel_edges", "tested": int(asserted.numel()), "accepted": int(moved.numel()),
                    "edges": edge_records(composer, asserted[moved], ctx)})
    # 2. structural proposals against a control refit of equal length.
    control = copy.deepcopy(composer)
    train_composer(control, ctx.train_concepts, ctx.train_targets, steps=settings.refit_steps, lr=settings.lr)
    proposals: list[tuple[str, dict[str, Any], LearnableOntologyComposer, Tensor]] = []
    live_slots = (composer.slot_active).nonzero().flatten().tolist()
    columns = list(range(composer.base_relations)) + [composer.slot_column(s) for s in live_slots]
    if settings.split:
        for column in columns:
            members = _relation_members(composer, column)
            if members.numel() < settings.min_split_members or not bool((~composer.slot_active).any()):
                continue
            proposal = copy.deepcopy(composer)
            info = _propose_split(proposal, ctx, column, settings)
            if info is None:
                continue
            train_composer(proposal, ctx.train_concepts, ctx.train_targets, steps=settings.refit_steps, lr=settings.lr)
            for slot in info["refreeze"]:
                proposal.crystallize(slot, min_mass=0.0, record={"hypothesis": "split"})
            proposals.append(("split", {"column": column, **{k: v for k, v in info.items() if k != "refreeze"}},
                              proposal, composer.edge_heads()[members]))
    if settings.merge:
        for i, a in enumerate(live_slots):
            for b in live_slots[i + 1:]:
                roles = composer.slot_role_vectors().detach()
                if float(F.cosine_similarity(roles[a], roles[b], dim=0)) < settings.merge_cosine:
                    continue
                proposal = copy.deepcopy(composer)
                members_b = _relation_members(proposal, proposal.slot_column(b))
                heads = composer.edge_heads()[torch.cat([members_b, _relation_members(composer, composer.slot_column(a))])]
                for e in members_b.tolist():
                    proposal.set_edge_option(e, proposal.slot_column(a))
                proposal.slot_active[b] = False; proposal.slot_frozen[b] = False
                train_composer(proposal, ctx.train_concepts, ctx.train_targets, steps=settings.refit_steps, lr=settings.lr)
                proposals.append(("merge", {"slots": [a, b]}, proposal, heads))
    frozen = composer.slot_frozen.nonzero().flatten().tolist()
    if settings.reopen:
        for slot in frozen:
            members = _relation_members(composer, composer.slot_column(slot))
            if not members.numel():
                continue
            proposal = copy.deepcopy(composer)
            proposal.reopen(slot, reason="dream")
            train_composer(proposal, ctx.train_concepts, ctx.train_targets, steps=settings.refit_steps, lr=settings.lr)
            proposal.crystallize(slot, record={"hypothesis": "reopened"})
            proposals.append(("reopen", {"slot": slot}, proposal, composer.edge_heads()[members]))
    if settings.remove_slot:
        for slot in live_slots:
            members = _relation_members(composer, composer.slot_column(slot))
            if not members.numel():
                continue
            proposal = copy.deepcopy(composer)
            proposal.pin_masses(members, 0.0)
            proposal.deactivate_slot(slot, reason="dream: remove relation")
            train_composer(proposal, ctx.train_concepts, ctx.train_targets, steps=settings.refit_steps, lr=settings.lr)
            proposals.append(("remove_slot", {"slot": slot}, proposal, composer.edge_heads()[members]))
    best = None
    for kind, info, proposal, heads in proposals:
        concepts = torch.unique(heads)
        concepts = concepts[torch.tensor([data.has(int(c)) for c in concepts.tolist()], dtype=torch.bool)]
        if concepts.numel() < 2:
            continue
        diff = _heldout_fit(proposal, data, concepts) - _heldout_fit(control, data, concepts)
        mean, lower, n = cluster_lower(diff, concepts, resamples=resamples, seed=seed + 7, alpha=alpha)
        margin = settings.merge_margin if kind == "merge" else 0.0
        accepted = lower > -margin if kind == "merge" else lower > 0
        log.append({"proposal": kind, **info, "mean": mean, "lower": lower, "heads": n, "accepted": bool(accepted)})
        if accepted and (best is None or lower > best[0]):
            best = (lower, proposal, kind, info)
    if best is not None:
        log.append({"proposal": "applied", "kind": best[2], **best[3]})
        best[1].cards.append({"event": f"dream_{best[2]}", **{k: v for k, v in best[3].items() if k != "partition"}})
        return best[1], log
    return control, log


def _propose_split(proposal: LearnableOntologyComposer, ctx: LearningContext, column: int,
                   settings: DreamSettings, members: Tensor | None = None) -> dict[str, Any] | None:
    """Split a relation along its top between-edge role-gradient direction (M3's statistic)."""
    members = _relation_members(proposal, column) if members is None else members
    members, grads = _role_gradients(proposal, ctx, members)
    if members.numel() < settings.min_split_members:
        return None
    direction = top_between_usage_direction(grads)
    side = (grads @ direction) < 0
    if int(side.sum()) < 2 or int((~side).sum()) < 2:
        return None
    refreeze = []
    is_slot = column >= proposal.base_relations
    old_slot = column - proposal.base_relations if is_slot else None
    if is_slot and bool(proposal.slot_frozen[old_slot]):
        proposal.reopen(old_slot, reason="dream split")
        refreeze.append(old_slot)
    new_slot = proposal.add_blank_slot()
    roles = proposal.option_roles().detach()
    base = roles[column]
    offset = settings.split_epsilon * base.norm() * direction
    with torch.no_grad():
        proposal.slot_roles[new_slot] = base - offset
        if is_slot:
            proposal.slot_roles[old_slot] = base + offset
    for e in members[side].tolist():
        proposal.set_edge_option(e, proposal.slot_column(new_slot))
    if is_slot:
        for e in members[~side].tolist():
            proposal.set_edge_option(e, column)
    refreeze.append(new_slot)
    return {"new_slot": new_slot, "moved": int(side.sum()), "kept": int((~side).sum()), "refreeze": refreeze,
            "partition": {"moved": members[side].tolist(), "kept": members[~side].tolist()}}


@torch.no_grad()
def revisit_crystallized(composer: LearnableOntologyComposer, ctx: LearningContext, *, threshold: float = 0.0,
                         resamples: int = 1000, alpha: float = 0.025, step: int | None = None) -> list[dict[str, Any]]:
    """Reopen crystallized slots whose held-out utility has turned negative.

    A slot is reopened when removing it would improve held-out fit by more than `threshold` with a
    cluster-bootstrap lower bound > 0 (the relation now confidently hurts).
    """
    from .self_test import edge_contributions
    events = []
    for slot in composer.slot_frozen.nonzero().flatten().tolist():
        members = composer.slot_members(slot, hard_only=True)
        members = members[composer.edge_masses()[members] > 0]
        heads = composer.edge_heads()[members]
        keep = torch.tensor([ctx.heldout.has(int(h)) for h in heads.tolist()], dtype=torch.bool)
        if int(keep.sum()) < 2:
            continue
        contributions = edge_contributions(composer, members[keep])
        unique, inverse = torch.unique(heads[keep], return_inverse=True)
        delta = torch.zeros(unique.numel(), contributions.shape[1]).index_add(0, inverse, contributions)
        gain = edit_effects(composer, ctx.heldout, unique, -delta).mean(1)     # held-out gain of removing the slot
        mean, lower, _ = cluster_lower(gain, unique, resamples=resamples, seed=ctx.seed + slot, alpha=alpha)
        reopen = mean > threshold and lower > 0
        events.append({"slot": slot, "removal_gain_mean": mean, "removal_gain_lower": lower, "reopened": bool(reopen)})
        if reopen:
            composer.reopen(slot, step=step, reason="held-out utility turned negative")
    return events


def edge_records(composer: LearnableOntologyComposer, edges: Tensor, ctx: LearningContext) -> list[list]:
    """(head node, column, tail node) for logged edges."""
    if not edges.numel():
        return []
    columns, _ = composer.edge_options(edges)
    heads = composer.edge_heads()[edges].tolist(); fillers = composer.schedule.fillers[edges].tolist()
    return [[ctx.nodes.node_of_concept[h], int(c), ctx.nodes.node_of_atomic[a]] for h, c, a in zip(heads, columns.tolist(), fillers)]
