"""Experiment E10.9a — passive learning paired with active self-reflection (synthetic planted world, CPU).

Question (author, 2026-10-04; execution.md decision 52): passive gradient learning first, then periodic
**active self-reflection** — interpret what a blank relation slot means by testing structural property
hypotheses on held-out data, and commit the best-fit explanation as an *explicit* weight write
(crystallize the slot, enforce its rule: closure edges added, violations pruned, operator frozen) — then
continue passive learning. Does this hybrid learn much faster than passive learning alone, at matched data
and compute? In a world without planted structure (ρ = 0) a commitment can only be wrong: the null world
measures the **cost of commitment** (does a wrong commitment hurt held-out prediction, and does later
reflection retract it?).

World and scenario: the E10.0 absent-mode world (has_part, similar_to, located_in hidden; their training-head
edges offered as open candidates among as many distractors; blank slots). The structure fraction ρ grades it:
the tails of a random fraction 1 − ρ of the offered pairs are permuted (`graded_scenario`); ρ = 0 is exactly
the D-B4 `null_permuted` world of `e10_baselines`.

Data: the learner-readable observations are (training concept, observations 0–7) and (validation concept,
observations 4–7), shuffled once per seed. A budget f gives every arm the same nested prefix. Hybrid arms
train on the training-role observations (0–3) of the prefix and run their held-out tests on its
validation-role observations (4–7), as E10.0 does at f = 1; the passive arm P trains on *every* observation
of the prefix (matched data: the observations a hybrid arm reads only through its tests are gradient data for
P). Training is full batch, so every step visits every observation of the prefix once: the fixed
steps-per-observation rule is the same step budget T at every f.

Arms (same world, observation order, initialization and slot-opening schedule; one new blank slot every K
steps):

- **P** passive only: open slots and open candidates trained by gradient; no interpretation, no self-test,
  no crystallization, no rule closure. **P-split**: P on the hybrid's gradient data only (decomposition).
- **H** hybrid: every K steps `reflect` (riddle hypotheses + held-out slot self-test; commit = crystallize +
  adopted rule enforced); rejected slots stay open (no write); then passive learning continues.
- **H+R**: H plus the existing offline revision pass (`dream_pass`, restricted to reopening / removing
  committed relations against an equally long control refit) before each later round's commitments.
- **H-rand** content control: H's schedule and acceptance, but each adopted rule is replaced by a random
  write of the same size (as many random pairs added, or random members pruned).
- **H-AMIE** generator control: the explanation comes from the AMIE-style miner on the captured pairs.
- **P+compute** compute control: P plus extra gradient steps at each round, matching the composition work
  H's reflection spent in that round (rows composed with gradient × 3 + rows composed without gradient,
  converted to P's full-batch steps).
- **A-first**: the first reflection at step 0 on the untrained model (no passive phase), gated like H;
  **A-first-forced**: the same with the step-0 commitment not gated.

Self-supervision: the learning loop reads a `LearningContext` (training-role observations, held-out
observations) only. Gold relation labels, gold pairs and the audit observations are read by evaluation code
(`light_eval`, `final_eval`, `_commitment_eval`) only; learning with all audit observations set to NaN gives
bit-identical decisions (tested).

    python -m vsa_embed.experiments.e10_active_passive --config experiments/e10-self-semantics/e10.9-active-passive.yaml \
        --output experiments/e10-self-semantics/runs/e10.9-v1 [--workers 3]
    python -m vsa_embed.experiments.e10_active_passive --report experiments/e10-self-semantics/runs/e10.9-v1
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from vsa_embed import kg_baselines as kg
from vsa_embed.experiments import e10_baselines as b
from vsa_embed.experiments import e10_self_semantics as e10
from vsa_embed.frame_inference import candidate_dictionary, frame_scores, infer_frames
from vsa_embed.learnable_ontology import ORIGIN, EdgeTable, LearnableOntologyComposer
from vsa_embed.ontology_discovery import (
    DiscoverySettings, DreamSettings, LearningContext, dream_pass, edge_pairs, enforce_rule, interpret_slot,
    own_relations, tail_pool,
)
from vsa_embed.ontology_hypotheses import (
    HypothesisSettings, StructuralHypothesis, best_match, jaccard, partition_scores, property_holds,
)
from vsa_embed.provenance import apply_thread_setting, prepare_output_dir, write_run_metadata
from vsa_embed.self_test import HeldOutData, NodeMap
from vsa_embed.statistics import mean_confidence_interval, paired_bootstrap_ci
from vsa_embed.synthetic_ontology import OntologyWorld

ARM_ORDER = ("P", "P-split", "H", "P+compute", "H+R", "H-rand", "H-AMIE", "A-first", "A-first-forced")
CONTRASTS = (("H", "P"), ("H", "P+compute"), ("H", "H-rand"), ("H", "H-AMIE"), ("H+R", "P"), ("H+R", "H"),
             ("H", "P-split"), ("H+R", "P-split"), ("P", "P-split"), ("A-first", "P"), ("A-first-forced", "P"),
             ("H", "A-first"), ("H", "A-first-forced"))
HYBRID = {"H", "H+R", "H-rand", "H-AMIE", "A-first", "A-first-forced"}
PASSIVE = {"P", "P-split", "P+compute"}


# =====================================================================================================
# graded world (structure fraction ρ)
# =====================================================================================================

def graded_scenario(world: OntologyWorld, *, rho: float, hidden: list[str], seed: int, coverage: float,
                    distractor_ratio: float, max_slots: int) -> e10.Scenario:
    """The E10.0 absent scenario with the tails of a random fraction 1 − ρ of the offered pairs permuted.

    Permuted pairs are re-drawn exactly as `e10_baselines.null_permuted_scenario` draws them (shuffled tails,
    never a gold pair or a duplicate); at ρ = 0 every offered pair is permuted and the table equals
    `null_permuted_scenario`'s (tested). The fraction of offered gold pairs that survive is ≈ ρ."""
    base = e10.hidden_scenario(world, hidden=hidden, seed=seed, coverage=coverage, distractor_ratio=distractor_ratio,
                               max_slots=max_slots)
    gold_pairs = set(world.relations_of_pair())
    hidden_ids = {world.relation_id(n) for n in hidden}
    gold_hidden = {(h, a) for h, r, a in world.edge_list() if r in hidden_ids}
    t = base.table
    opened = [(int(h), int(a)) for h, a, c in zip(t.heads.tolist(), t.fillers.tolist(), t.candidate.tolist()) if c]
    before = sum(p in gold_hidden for p in opened)
    if rho >= 1.0:
        base.info.update(rho=1.0, offered_gold_before=before, offered_gold_after=before, structure_fraction=1.0)
        return base
    asserted = [(int(h), int(r), int(a)) for h, r, a, c in zip(t.heads.tolist(), t.relations.tolist(), t.fillers.tolist(),
                                                                 t.candidate.tolist()) if not c]
    n = len(opened)
    chosen = list(range(n)) if rho <= 0.0 else sorted(random.Random(seed + 31).sample(range(n), int(round((1 - rho) * n))))
    chosen_set = set(chosen)
    rng = random.Random(seed + 23)                          # null_permuted_scenario's generator
    tails = [opened[i][1] for i in chosen]
    rng.shuffle(tails)
    kept = [opened[i] for i in range(n) if i not in chosen_set]
    seen: set[tuple[int, int]] = set(kept)
    open_edges = []
    queue = iter(tails)
    for i, (h, a0) in enumerate(opened):
        if i not in chosen_set:
            open_edges.append((h, -1, a0))
            continue
        a = next(queue)
        for _attempt in range(50):
            if (h, a) not in gold_pairs and (h, a) not in seen:
                break
            a = rng.choice(tails)
        if (h, a) not in gold_pairs and (h, a) not in seen:
            seen.add((h, a)); open_edges.append((h, -1, a))
    table = EdgeTable.build(world.concept_count, t.relation_count, max_slots=max_slots, asserted=asserted, candidates=open_edges)
    after = sum((h, a) in gold_hidden for h, _, a in open_edges)
    return e10.Scenario(f"graded-{rho}", table, base.relation_names, base.column_world, base.relation_prior, base.hidden,
                        {"open_candidates": len(open_edges), "permuted": len(chosen), "rho": rho,
                         "offered_gold_before": before, "offered_gold_after": after,
                         "structure_fraction": after / before if before else float("nan")})


# =====================================================================================================
# data budgets (nested prefixes of one observation order)
# =====================================================================================================

def observation_pool(world: OntologyWorld, seed: int) -> list[tuple[int, int]]:
    """Every learner-readable (concept, observation) item in one fixed order per seed: training concepts ×
    (train + val observations) and validation concepts × val observations. Audit observations never."""
    obs = world.observation_splits
    items = [(int(c), int(o)) for c in world.splits["train"].tolist() for o in obs["train"] + obs["val"]]
    items += [(int(c), int(o)) for c in world.splits["validation"].tolist() for o in obs["val"]]
    random.Random(seed + 4099).shuffle(items)
    return items


def _stack(world: OntologyWorld, by_concept: dict[int, list[int]], *, repeat_to_lcm: bool) -> tuple[Tensor, Tensor, Tensor]:
    """(concepts, targets, mask). With `repeat_to_lcm`, each concept's observations are repeated cyclically to
    the lcm of the counts (mask all ones), so a plain mean over observations is each concept's own mean."""
    concepts = sorted(c for c, o in by_concept.items() if o)
    if not concepts:
        d = world.target_dimension
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, 1, d), torch.zeros(0, 1)
    counts = [len(by_concept[c]) for c in concepts]
    width = math.lcm(*range(1, max(counts) + 1)) if repeat_to_lcm else max(counts)
    targets = torch.zeros(len(concepts), width, world.target_dimension)
    mask = torch.zeros(len(concepts), width)
    for i, c in enumerate(concepts):
        o = sorted(by_concept[c])
        if repeat_to_lcm:
            index = [o[j % len(o)] for j in range(width)]
            targets[i] = world.observations[c, index]
            mask[i] = 1.0
        else:
            targets[i, :len(o)] = world.observations[c, o]
            mask[i, :len(o)] = 1.0
    return torch.tensor(concepts, dtype=torch.long), targets, mask


@dataclass
class BudgetData:
    """What each arm may read at data budget `f`."""

    f: float
    hybrid: tuple[Tensor, Tensor, Tensor]          # training-role observations (hybrid gradient data, P-split)
    passive: tuple[Tensor, Tensor, Tensor]         # every observation of the prefix (P, P+compute)
    ctx: LearningContext                           # training role (lcm-repeated) + held-out (validation role)
    counts: dict[str, int] = field(default_factory=dict)


def budget_data(world: OntologyWorld, scenario: e10.Scenario, pool: list[tuple[int, int]], f: float, seed: int) -> BudgetData:
    prefix = pool[:max(1, int(round(f * len(pool))))]
    obs = world.observation_splits
    train_role, val_role = set(obs["train"]), set(obs["val"])
    train_concepts = set(world.splits["train"].tolist())
    hybrid: dict[int, list[int]] = defaultdict(list)
    held: dict[int, list[int]] = defaultdict(list)
    everything: dict[int, list[int]] = defaultdict(list)
    for c, o in prefix:
        everything[c].append(o)
        if o in train_role and c in train_concepts:
            hybrid[c].append(o)
        elif o in val_role:
            held[c].append(o)
    h_concepts, h_targets, h_mask = _stack(world, hybrid, repeat_to_lcm=False)
    ctx_concepts, ctx_targets, _ = _stack(world, hybrid, repeat_to_lcm=True)
    held_concepts, held_targets, _ = _stack(world, held, repeat_to_lcm=True)
    heldout = HeldOutData(held_concepts, held_targets, name="val")
    ctx = LearningContext(ctx_concepts, ctx_targets, heldout, NodeMap(world.concept_node, world.filler_node),
                          scenario.relation_names, set(e10._framed(world)), seed)
    counts = {"prefix_observations": len(prefix), "pool_observations": len(pool),
              "hybrid_gradient_observations": sum(len(v) for v in hybrid.values()),
              "heldout_observations": sum(len(v) for v in held.values()),
              "hybrid_concepts": len(hybrid), "heldout_concepts": len(held), "passive_concepts": len(everything)}
    return BudgetData(f, (h_concepts, h_targets, h_mask), _stack(world, everything, repeat_to_lcm=False), ctx, counts)


# =====================================================================================================
# compute accounting and training
# =====================================================================================================

_COUNTER = {"on": False, "grad_rows": 0, "nograd_rows": 0}


class CountingComposer(LearnableOntologyComposer):
    """A `LearnableOntologyComposer` that counts composed rows while `counting()` is active (with gradient:
    a training forward + backward; without: a test or evaluation pass). Behaviour is unchanged; the counter
    is module state, so deep copies made by revision passes count into it too."""

    def compose_raw(self, concept_ids: Tensor, context: Tensor | None = None):
        if _COUNTER["on"]:
            _COUNTER["grad_rows" if torch.is_grad_enabled() else "nograd_rows"] += int(concept_ids.numel())
        return super().compose_raw(concept_ids, context)


@contextmanager
def counting() -> Iterator[dict[str, int]]:
    """Count the rows composed inside the block: `{"grad_rows", "nograd_rows", "units"}` where
    units = 3 × grad rows + nograd rows (a backward costs about two forwards)."""
    start = (_COUNTER["grad_rows"], _COUNTER["nograd_rows"])
    box: dict[str, int] = {}
    _COUNTER["on"] = True
    try:
        yield box
    finally:
        _COUNTER["on"] = False
        box["grad_rows"] = _COUNTER["grad_rows"] - start[0]
        box["nograd_rows"] = _COUNTER["nograd_rows"] - start[1]
        box["units"] = 3 * box["grad_rows"] + box["nograd_rows"]


def make_counting_composer(world: OntologyWorld, scenario: e10.Scenario, config: dict[str, Any], seed: int) -> CountingComposer:
    composer = e10.make_composer(world, scenario, config, seed)
    composer.__class__ = CountingComposer
    return composer


def train_masked(composer: LearnableOntologyComposer, concepts: Tensor, targets: Tensor, mask: Tensor, *, steps: int,
                 lr: float, optimizer: torch.optim.Optimizer | None = None) -> torch.optim.Optimizer | None:
    """`train_composer` with a variable number of observations per concept: the fit is the mean over concepts
    of each concept's mean over its observed observations (with a full mask exactly `train_composer`'s)."""
    if not concepts.numel() or steps <= 0:
        return optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam([p for p in composer.parameters() if p.requires_grad], lr=lr)
    weights = mask / mask.sum(1, keepdim=True)
    n = concepts.numel()
    for _ in range(steps):
        prediction = composer(concepts)
        valid = prediction.detach().norm(dim=-1) > 1e-6
        if bool(valid.any()):
            cos = F.cosine_similarity(prediction[valid][:, None, :], targets[valid], dim=-1)
            fit = ((1 - cos) * weights[valid]).sum(1).mean()
        else:
            fit = prediction.sum() * 0
        loss = fit + composer.penalty() / n
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        composer.project_()
    return optimizer


# =====================================================================================================
# the reflection step (reusable; reads only a LearningContext)
# =====================================================================================================

@dataclass
class ReflectionPolicy:
    """How a reflection round explains and commits a slot.

    - `generator`: "riddle" (E10.6 structural hypotheses scored on held-out data, best passing adopted) or
      "amie" (best AMIE-style Horn explanation of the captured pairs, PCA confidence).
    - `content`: "best" (enforce the adopted rule) or "random_matched" (a random write of the same size).
    - `revise`: run the offline revision pass over earlier commitments first (`revision` settings).
    - `force`: commit even if the held-out slot self-test rejects; `min_mass`: member threshold override.
    """

    generator: str = "riddle"
    content: str = "best"
    revise: bool = False
    force: bool = False
    min_mass: float | None = None
    amie: dict[str, float] = field(default_factory=lambda: {"min_support": 2, "min_head_coverage": 0.01, "min_pca": 0.1})
    revision: DreamSettings | None = None
    seed: int = 0


_NO_RIDDLE = HypothesisSettings(min_overlap=2.0, min_symmetry=2.0, max_antisymmetry=-1.0, min_transitivity=2.0,
                                min_two_paths=10**9, min_functional=2.0)


def hypothesis_from_name(name: str) -> StructuralHypothesis:
    kind, _, args = name.partition(":")
    return StructuralHypothesis(kind, tuple(args.split("∘")) if args else ())


def amie_explanation(pairs: set, relations: dict[str, set], thresholds: dict[str, float]) -> StructuralHypothesis | None:
    """Best AMIE-style explanation of a slot's pairs over the model's own relations (D-B1's language bias:
    symmetric, transitive, inverse-of, sub-relation-of, composition-of); highest PCA confidence among the
    rules passing AMIE's thresholds, closure-type rules first on ties. No gold is read."""
    if not pairs:
        return None
    names = sorted(relations)
    rels = {**{n: relations[n] for n in names}, "slot": set(pairs)}
    index = {r: {h for h, _ in p} for r, p in rels.items()}
    cache: dict = {}
    scored = []
    for hyp, rule in b._explanations(names):
        s = kg.score_rule(rule, rels, cache=cache, head_index=index)
        if s is not None and s.passes(min_support=int(thresholds["min_support"]),
                                      min_head_coverage=float(thresholds["min_head_coverage"]),
                                      min_pca=float(thresholds["min_pca"])):
            scored.append((s.pca_confidence, rule.head == "slot", s.support, hyp))
    scored.sort(key=lambda x: (-x[0], not x[1], -x[2], x[3]))
    return hypothesis_from_name(scored[0][3]) if scored else None


def enforce_random_matched(composer: LearnableOntologyComposer, slot: int, hypothesis: StructuralHypothesis,
                           ctx: LearningContext, settings: DiscoverySettings, *, mass: float, pairs: set,
                           relations: dict[str, set], seed: int) -> dict[str, int]:
    """Content control: measure what `enforce_rule` would write (on a copy), then write as many random pairs
    under the slot (heads among the rule heads, tails from the candidate tail pool) or prune as many random
    members. Returns the same counts as `enforce_rule`."""
    probe = copy.deepcopy(composer)
    counts = enforce_rule(probe, slot, hypothesis, ctx, settings, mass=mass, pairs=pairs, relations=relations)
    del probe
    rng = random.Random(seed)
    column = composer.slot_column(slot)
    if counts["rule_added"]:
        existing = edge_pairs(composer, composer.slot_members(slot, hard_only=True), ctx.nodes)
        heads = sorted(ctx.rule_heads)
        tails = sorted({ctx.nodes.atomic_of_node[t] for t in tail_pool(composer, ctx)})
        new: list[tuple[int, int]] = []
        seen = set(existing)
        for _attempt in range(50 * counts["rule_added"]):
            if len(new) >= counts["rule_added"]:
                break
            h, a = rng.choice(heads), rng.choice(tails)
            pair = ctx.nodes.edge_pair(h, a)
            if pair in seen or pair[0] == pair[1]:
                continue
            seen.add(pair); new.append((h, a))
        if new:
            composer.add_edges([h for h, _ in new], [column] * len(new), [a for _, a in new], mass=mass, origin="rule")
        counts["rule_added"] = len(new)
    if counts["rule_pruned"]:
        members = composer.slot_members(slot, hard_only=True)
        live = members[~composer.pinned[members]].tolist()
        pick = sorted(rng.sample(live, min(counts["rule_pruned"], len(live))))
        if pick:
            composer.pin_masses(torch.tensor(pick, dtype=torch.long), 0.0)
        counts["rule_pruned"] = len(pick)
    return counts


def reflect(composer: LearnableOntologyComposer, ctx: Any, settings: DiscoverySettings, policy: ReflectionPolicy, *,
            step: int, round_index: int, interpret: Callable[..., dict[str, Any]] = interpret_slot,
            revise: Callable[..., tuple[LearnableOntologyComposer, list[dict[str, Any]]]] = dream_pass,
            ) -> tuple[LearnableOntologyComposer, dict[str, Any]]:
    """One active self-reflection round over a learnable ontology (no LM- or world-specific code).

    The held-out evidence enters only through `interpret(composer, slot, ctx, settings) -> record` (default
    `interpret_slot`: target-vector held-out observations; record keys `pairs`, `adopted`, `decision`, `mass`,
    `description`) and `revise(composer, ctx, revision_settings, seed=, resamples=, alpha=) -> (composer, log)`
    (default `dream_pass`). Another evidence source (e.g. held-out LM windows, E10.9b) plugs in by passing
    functions with these signatures; `ctx` must provide `nodes` and `rule_heads` (and whatever they read).

    1. (`policy.revise`) offline revision of earlier commitments: `dream_pass` restricted to `policy.revision`
       (here: reopen / remove a committed relation, each against an equally long control refit, accepted iff
       the held-out cluster-bootstrap lower bound is > 0). May return a refit copy of the composer.
    2. Crystallized slots absorb late soft members.
    3. For every open (active, not frozen) slot: interpret it (captured pairs → hypotheses → held-out scores;
       held-out slot self-test), and if accepted (or `policy.force`), **commit**: crystallize (freeze the
       operator, hard-assign the members) and enforce the adopted explanation as an explicit write (closure
       edges added / violations pruned). Rejected slots stay open (no write).

    Returns the (possibly replaced) composer and a log; each slot record keeps its captured `pairs` (for the
    caller's evaluation; strip before serializing). Rebuild the optimizer afterwards (`add_edges` replaces
    Parameters and the revision pass may return a copy)."""
    log: dict[str, Any] = {"round": round_index, "step": step, "revisions": [], "slots": [], "absorbed": 0}
    if policy.revise and bool(composer.slot_frozen.any()) and policy.revision is not None:
        composer, revisions = revise(composer, ctx, policy.revision, seed=policy.seed + 1000 * round_index + 7,
                                     resamples=settings.resamples, alpha=settings.alpha)
        log["revisions"] = [{k: v for k, v in r.items() if k not in {"edges", "partition"}} for r in revisions]
    if settings.absorb:
        log["absorbed"] = sum(composer.absorb(s, min_assignment=settings.min_assignment, min_mass=settings.min_mass,
                                              step=step) for s in composer.slot_frozen.nonzero().flatten().tolist())
    local = settings
    if policy.min_mass is not None:
        local = dataclasses.replace(local, min_mass=float(policy.min_mass))
    if policy.generator == "amie":
        local = dataclasses.replace(local, hypotheses=_NO_RIDDLE, riddle_full=False)
    for slot in (composer.slot_active & ~composer.slot_frozen).nonzero().flatten().tolist():
        record = interpret(composer, slot, ctx, local)
        decision = record["decision"]
        relations = own_relations(composer, ctx, exclude_slot=slot)
        if policy.generator == "amie":
            hypothesis = amie_explanation(record["pairs"], relations, policy.amie)
        else:
            hypothesis = record["adopted"].hypothesis if record["adopted"] is not None else None
        commit = bool(record["pairs"]) and (bool(decision.accept) or policy.force)
        entry = {"slot": slot, "size": len(record["pairs"]), "accept": bool(decision.accept), "forced": bool(policy.force),
                 "commit": commit, "utility_mean": decision.mean, "utility_lower": decision.lower,
                 "specificity_lower": decision.extra.get("specificity_lower"),
                 "adopted": hypothesis.name if hypothesis is not None else None,
                 "rule_added": 0, "rule_pruned": 0, "pairs": record["pairs"]}
        if commit:
            composer.crystallize(slot, min_assignment=local.min_assignment, min_mass=local.min_mass, step=step,
                                 record={"hypothesis": entry["adopted"] or "unstructured", "description": record["description"],
                                         "utility": {"mean": decision.mean, "lower": decision.lower}})
            if settings.enforce_rule and hypothesis is not None:
                if policy.content == "random_matched":
                    out = enforce_random_matched(composer, slot, hypothesis, ctx, local, mass=record["mass"],
                                                 pairs=record["pairs"], relations=relations,
                                                 seed=policy.seed + 97 * round_index + slot)
                else:
                    out = enforce_rule(composer, slot, hypothesis, ctx, local, mass=record["mass"], pairs=record["pairs"],
                                       relations=relations)
                entry.update(out)
        log["slots"].append(entry)
    return composer, log


# =====================================================================================================
# evaluation (gold and audit observations: evaluation only)
# =====================================================================================================

def _hidden_gold_pairs(world: OntologyWorld, scenario: e10.Scenario) -> set[tuple[int, int]]:
    hidden = set(scenario.hidden)
    return {(h, a) for h, r, a in world.edge_list() if r in hidden}


@torch.no_grad()
def _edge_auc(composer: LearnableOntologyComposer, gold: set[tuple[int, int]]) -> float:
    """AUC of offered open candidates (score: mass × assignment confidence) against gold hidden pairs."""
    cand = ((composer.origin == ORIGIN["candidate"]) & composer.open).nonzero().flatten()
    if not cand.numel():
        return float("nan")
    _, conf = composer.edge_options(cand)
    scores = (composer.edge_masses().detach()[cand] * conf).tolist()
    labels = [(h, a) in gold for h, a in zip(composer.edge_heads()[cand].tolist(), composer.schedule.fillers[cand].tolist())]
    if any(not math.isfinite(s) for s in scores):          # e10._auc's tie loop does not terminate on NaN
        return float("nan")
    return e10._auc(scores, labels)


@torch.no_grad()
def light_eval(world: OntologyWorld, scenario: e10.Scenario, composer: LearnableOntologyComposer, audit: HeldOutData,
               gold_hidden: set[tuple[int, int]]) -> dict[str, float]:
    rule = composer.origin == ORIGIN["rule"]
    return {"test_fit": e10.eval_fit(composer, audit, world.splits["test"]),
            "train_audit_fit": e10.eval_fit(composer, audit, world.splits["train"]),
            "edge_auc": _edge_auc(composer, gold_hidden), "frozen": int(composer.slot_frozen.sum()),
            "rule_edges": int((rule & (composer.edge_masses().detach() > 0)).sum())}


@torch.no_grad()
def frame_f1(world: OntologyWorld, scenario: e10.Scenario, composer: LearnableOntologyComposer, *, ks: list[int],
             max_edges: int, threshold: float) -> dict[str, float]:
    """New-word frame F1 (E10.0 (e), non-negative OMP, everything frozen) over the seed relations plus every
    active slot with ≥ 2 members (crystallized or open, so the passive arms are not penalized)."""
    nodes = NodeMap(world.concept_node, world.filler_node)
    columns_gold = e10._gold_columns(world, scenario, composer, nodes)
    slots = [composer.slot_column(s) for s in composer.slot_active.nonzero().flatten().tolist()
             if composer.slot_members(s).numel() >= 2]
    columns = list(range(composer.base_relations)) + slots
    words = world.splits["new_word"]
    gold_frames = {int(w): set() for w in words.tolist()}
    for h, r, a in world.edge_list():
        if h in gold_frames:
            gold_frames[h].add((world.relation_names[r], a))
    obs = world.observation_splits
    infer_obs = world.observations[words][:, obs["infer"]]
    grid_c, grid_f, vectors = candidate_dictionary(lambda c, f: composer.bind_options(c, f).detach(), columns,
                                                   list(range(world.atomic_count)), project=composer.projector)
    out = {}
    for k in ks:
        frames = infer_frames(infer_obs[:, :k], grid_c, grid_f, vectors, max_edges=max_edges, threshold=threshold)
        f1 = [frame_scores({(columns_gold.get(c), a) for c, a in zip(fr.columns, fr.fillers)}, gold_frames[w])["f1"]
              for w, fr in zip(words.tolist(), frames)]
        out[f"frame_f1_k{k}"] = float(np.nanmean(f1)) if f1 else float("nan")
    return out


def _commitment_eval(world: OntologyWorld, scenario: e10.Scenario, commitments: list[dict[str, Any]]) -> None:
    """Gold verdict of every commitment (in place): the slot is `good` if its captured pairs are mostly
    offered gold pairs of one hidden relation (precision ≥ 0.5, purity ≥ 0.7, as `_slot_decision_eval`);
    its adopted rule is `true` if it holds of that relation (`property_holds`). Evaluation only."""
    offered = e10._offered_pairs(world, scenario)
    gold_rel = {n: world.pairs(n) for n in world.relation_names}
    alias = {name: world.relation_names[w] for name, w in zip(scenario.relation_names, scenario.column_world) if w is not None}
    for c in commitments:
        pairs = c["pairs"]
        name, score = best_match(pairs, offered) if pairs and offered else (None, 0.0)
        real = [p for p in pairs if any(p in g for g in offered.values())]
        precision = len(real) / len(pairs) if pairs else float("nan")
        dominant = max((sum(p in g for p in real) for g in offered.values()), default=0)
        purity = dominant / len(real) if real else float("nan")
        c.update(best=name, jaccard=score, precision=precision, purity=purity,
                 good=bool(precision == precision and precision >= 0.5 and purity == purity and purity >= 0.7))
        if name is not None:
            alias[f"slot{c['slot']}"] = name
    for c in commitments:
        if c["adopted"] is None:
            c["rule_true"] = None
            continue
        name = c["best"]
        c["rule_true"] = bool(name is not None and property_holds(hypothesis_from_name(c["adopted"]), gold_rel[name],
                                                                   gold_rel, alias))


@torch.no_grad()
def final_eval(world: OntologyWorld, scenario: e10.Scenario, composer: LearnableOntologyComposer, audit: HeldOutData,
               config: dict[str, Any], seed: int) -> dict[str, Any]:
    nodes = NodeMap(world.concept_node, world.filler_node)
    out: dict[str, Any] = {}
    gold_l, pred_l, pairs = e10._partition(world, scenario, composer, mode="absent")
    if any(g is not None for g in gold_l):
        scores = partition_scores(gold_l, pred_l, pairs, permutations=0, seed=seed)
        out.update(ari=scores.ari_gold_edges, mean_best_jaccard=scores.mean_best_jaccard,
                   detection_precision=scores.detection_precision, detection_recall=scores.detection_recall)
    else:
        out.update(ari=float("nan"), mean_best_jaccard=float("nan"), detection_precision=float("nan"),
                   detection_recall=float("nan"))
    framed = e10._framed(world)
    masses = composer.edge_masses().detach()
    slot_pairs = {}
    for s in composer.slot_active.nonzero().flatten().tolist():
        members = composer.slot_members(s)
        members = members[masses[members] >= 0.25]
        slot_pairs[s] = edge_pairs(composer, members, nodes)
    recovery = {}
    for r in scenario.hidden:
        name = world.relation_names[r]
        gold = world.pairs(name, heads=framed)
        recovery[name] = max((jaccard(p, gold) for p in slot_pairs.values()), default=0.0)
    out["relation_jaccard_framed"] = recovery
    out["relation_jaccard_framed_mean"] = float(np.mean(list(recovery.values()))) if recovery else float("nan")
    # rule-implied edges: gold hidden-relation pairs the learner never saw as candidates
    hidden = set(scenario.hidden)
    test_heads = set(world.splits["test"].tolist())
    offered = {(int(h), int(a)) for h, a, o in zip(scenario.table.heads.tolist(), scenario.table.fillers.tolist(),
                                                    scenario.table.open.tolist()) if o}
    gold_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    for h, r, a in world.edge_list():
        if r in hidden and h in framed:
            gold_by_pair[(h, a)].add(world.relation_names[r])
    unseen_test = {p for p in gold_by_pair if p[0] in test_heads}
    unseen_all = {p for p in gold_by_pair if p not in offered}
    columns_gold = e10._gold_columns(world, scenario, composer, nodes)
    columns, _ = composer.edge_options()
    rule = ((composer.origin == ORIGIN["rule"]) & (masses >= 0.25)).nonzero().flatten()
    rule_pairs = {(int(composer.edge_heads()[e]), int(composer.schedule.fillers[e])): columns_gold.get(int(columns[e]))
                  for e in rule.tolist()}
    out["rule_edges"] = len(rule_pairs)
    out["rule_edge_precision"] = (sum(p in gold_by_pair for p in rule_pairs) / len(rule_pairs)) if rule_pairs else float("nan")
    out["unobserved_test_recall"] = (sum(p in rule_pairs for p in unseen_test) / len(unseen_test)) if unseen_test else float("nan")
    out["unobserved_test_recall_matched"] = (sum(p in rule_pairs and rule_pairs[p] in gold_by_pair[p] for p in unseen_test)
                                             / len(unseen_test)) if unseen_test else float("nan")
    out["unobserved_recall"] = (sum(p in rule_pairs for p in unseen_all) / len(unseen_all)) if unseen_all else float("nan")
    out["unobserved_test_gold"] = len(unseen_test)
    out["test_fit"] = e10.eval_fit(composer, audit, world.splits["test"])
    probe = copy.deepcopy(composer)
    all_rule = (probe.origin == ORIGIN["rule"]).nonzero().flatten()
    if all_rule.numel():
        probe.pin_masses(all_rule, 0.0)
    out["test_fit_without_rule_edges"] = e10.eval_fit(probe, audit, world.splits["test"])
    out["validation_concept_fit"] = e10.eval_fit(composer, audit, world.splits["validation"])
    frames = config["frames"]
    out.update(frame_f1(world, scenario, composer, ks=list(frames["k"]), max_edges=int(frames["max_edges"]),
                        threshold=float(frames["threshold"])))
    return out


# =====================================================================================================
# arms
# =====================================================================================================

def arm_policy(arm: str, config: dict[str, Any], seed: int, round_index: int) -> ReflectionPolicy:
    lr = float(config["model"]["lr"])
    revision = DreamSettings(**{**config.get("dream_settings", {}), "lr": lr, **config["revision"]})
    policy = ReflectionPolicy(amie=dict(config["amie"]), revision=revision, seed=seed)
    if arm == "H+R":
        policy.revise = True
    elif arm == "H-rand":
        policy.content = "random_matched"
    elif arm == "H-AMIE":
        policy.generator = "amie"
    elif arm in {"A-first", "A-first-forced"} and round_index == 0:
        policy.min_mass = float(config["discovery"].get("slot_init_mass", 0.1))
        policy.force = arm == "A-first-forced"
    return policy


def run_arm(arm: str, world: OntologyWorld, scenario: e10.Scenario, data: BudgetData, config: dict[str, Any], seed: int, *,
            reflect_every: int, evaluate: bool = True, compute_plan: list[int] | None = None) -> tuple[LearnableOntologyComposer, dict[str, Any]]:
    """Train one arm on one budget; returns the composer and its log (curve, reflections, compute)."""
    sched = config["schedule"]
    total, every = int(sched["total_steps"]), int(sched["eval_every"])
    lr = float(config["model"]["lr"])
    settings = e10.discovery_settings(config)
    first = 0 if arm in {"A-first", "A-first-forced"} else reflect_every
    rounds = list(range(first, total, reflect_every))
    concepts, targets, mask = data.passive if arm in {"P", "P+compute"} else data.hybrid
    n_ref = int(world.splits["train"].numel())                 # compute unit: one step over all training concepts
    unit = 3.0 * n_ref
    composer: LearnableOntologyComposer = make_counting_composer(world, scenario, config, seed)
    composer.add_blank_slot(step=0, init_mass=settings.slot_init_mass)
    audit = e10.audit_data(world) if evaluate else None
    gold_hidden = _hidden_gold_pairs(world, scenario)
    marks = sorted(set(range(0, total + 1, every)) | set(rounds) | {total})
    step, compute, optimizer = 0, 0.0, None
    curve, reflections, commitments = [], [], []
    extra_total, passive_seconds = 0, 0.0
    for mark in marks:
        if mark > step:
            tick = time.perf_counter()
            optimizer = train_masked(composer, concepts, targets, mask, steps=mark - step, lr=lr, optimizer=optimizer)
            passive_seconds += time.perf_counter() - tick
            compute += (mark - step) * 3.0 * concepts.numel() / unit
            step = mark
        if mark in rounds:
            r = rounds.index(mark)
            if arm in HYBRID and data.ctx.heldout.concepts.numel():
                before = light_eval(world, scenario, composer, audit, gold_hidden)["test_fit"] if evaluate else None
                tick = time.perf_counter()
                with counting() as cost:
                    composer, log = reflect(composer, data.ctx, settings, arm_policy(arm, config, seed, r), step=mark,
                                            round_index=r)
                cost["seconds"] = time.perf_counter() - tick
                compute += cost["units"] / unit
                log.update(cost=dict(cost), test_fit_before=before,
                           test_fit_after=light_eval(world, scenario, composer, audit, gold_hidden)["test_fit"] if evaluate else None)
                for entry in log["slots"]:
                    if entry["commit"]:
                        commitments.append({"round": r, "step": mark, "slot": entry["slot"], "adopted": entry["adopted"],
                                            "size": entry["size"], "forced": entry["forced"], "rule_added": entry["rule_added"],
                                            "rule_pruned": entry["rule_pruned"], "pairs": entry["pairs"],
                                            "retracted": False, "revised": False})
                for rev in log["revisions"]:
                    if rev.get("proposal") == "applied" and rev.get("kind") in {"remove_slot", "reopen"}:
                        target = [c for c in commitments if c["slot"] == rev.get("slot") and c["round"] < r]
                        if target:
                            target[-1]["retracted" if rev["kind"] == "remove_slot" else "revised"] = True
                            target[-1].setdefault("retracted_round", r)
                reflections.append({k: v for k, v in log.items() if k != "slots"} |
                                   {"slots": [{k: v for k, v in s.items() if k != "pairs"} for s in log["slots"]]})
            elif arm == "P+compute" and compute_plan is not None and r < len(compute_plan):
                extra = int(round(compute_plan[r] / (3.0 * max(1, concepts.numel()))))
                optimizer = train_masked(composer, concepts, targets, mask, steps=extra, lr=lr, optimizer=optimizer)
                compute += extra * 3.0 * concepts.numel() / unit
                extra_total += extra
            composer.add_blank_slot(step=mark, init_mass=settings.slot_init_mass)
            optimizer = None                      # a fresh optimizer per round (E10.0); Parameters may be replaced
        if evaluate and (mark % every == 0 or mark == total):
            curve.append({"step": mark, "compute": compute, **light_eval(world, scenario, composer, audit, gold_hidden)})
    log = {"arm": arm, "rounds": rounds, "curve": curve, "reflections": reflections, "commitments": commitments,
           "compute": compute, "extra_steps": extra_total, "batch_concepts": int(concepts.numel()),
           "reflection_units": [r["cost"]["units"] for r in reflections],
           "reflection_seconds": [r["cost"]["seconds"] for r in reflections],
           "passive_seconds_per_step": passive_seconds / max(1, total + extra_total)}
    return composer, log


def run_cell(config: dict[str, Any], rho: float, seed: int, f: float, reflect_every: int, arms: list[str], *,
             evaluate: bool = True) -> list[dict[str, Any]]:
    """Every requested arm on one (ρ, seed, budget, K) cell; P+compute matches H's measured reflection work."""
    c = config["c"]
    world = e10.build_world(config, seed)
    scenario = graded_scenario(world, rho=rho, hidden=c["hidden"], seed=seed, coverage=c["coverage"],
                               distractor_ratio=c["distractor_ratio"], max_slots=c["max_slots"])
    pool = observation_pool(world, seed)
    data = budget_data(world, scenario, pool, f, seed)
    audit = e10.audit_data(world)
    rows, plans = [], {}
    for arm in [a for a in ARM_ORDER if a in arms]:
        start = time.time()
        plan = plans.get("H") if arm == "P+compute" else None
        if arm == "P+compute" and plan is None:
            raise ValueError("P+compute needs H in the same cell")
        composer, log = run_arm(arm, world, scenario, data, config, seed, reflect_every=reflect_every, evaluate=evaluate,
                                compute_plan=plan)
        if arm == "H":
            plans["H"] = log["reflection_units"]
        row = {"part": "e10.9", "rho": rho, "seed": seed, "f": f, "K": reflect_every, "arm": arm,
               "scenario": {k: v for k, v in scenario.info.items()}, "data": data.counts}
        if evaluate:
            _commitment_eval(world, scenario, log["commitments"])
            row["final"] = final_eval(world, scenario, composer, audit, config, seed)
        commitments = [{k: v for k, v in cm.items() if k != "pairs"} for cm in log["commitments"]]
        row.update(curve=log["curve"], reflections=log["reflections"], commitments=commitments, compute=log["compute"],
                   extra_steps=log["extra_steps"], batch_concepts=log["batch_concepts"],
                   reflection_units=log["reflection_units"], reflection_seconds=log["reflection_seconds"],
                   passive_seconds_per_step=log["passive_seconds_per_step"], seconds=time.time() - start)
        rows.append(row)
    return rows


# =====================================================================================================
# orchestration
# =====================================================================================================

def resolve(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    return b.resolve(config, root)


def _selected(spec: Any, value: float, universe: list[float]) -> bool:
    if spec == "all" or spec is None:
        return True
    return any(abs(float(v) - value) < 1e-9 for v in spec)


def jobs(config: dict[str, Any], *, seeds: list[int] | None = None) -> list[tuple]:
    """(ρ, seed, f, K, arms) cells: the main grid at K, and the K-sensitivity cells."""
    seeds = [int(s) for s in (seeds if seeds is not None else config["seeds"])]
    rhos = [float(r) for r in config["rhos"]]
    budgets = [float(f) for f in config["budgets"]]
    k_main = int(config["schedule"]["reflect_every"])
    out = []
    for seed in seeds:
        for rho in rhos:
            for f in budgets:
                arms = [a for a in ARM_ORDER if a in config["arms"]
                        and _selected((config["arms"][a] or {}).get("rho", "all"), rho, rhos)
                        and _selected((config["arms"][a] or {}).get("f", "all"), f, budgets)]
                for extra in config.get("extra_cells") or []:
                    if _selected(extra.get("rho", "all"), rho, rhos) and _selected(extra.get("f", "all"), f, budgets):
                        arms += [a for a in extra["arms"] if a not in arms]
                if "P+compute" in arms and "H" not in arms:
                    arms.append("H")
                if arms:
                    out.append((rho, seed, f, k_main, tuple(a for a in ARM_ORDER if a in arms)))
        sens = config.get("k_sensitivity") or {}
        for k in sens.get("K", []):
            for rho in sens.get("rho", []):
                for f in sens.get("f", []):
                    out.append((float(rho), seed, float(f), int(k), tuple(a for a in ARM_ORDER if a in sens["arms"])))
    return out


def _run_job(job: tuple, config: dict[str, Any]) -> list[dict[str, Any]]:
    torch.set_num_threads(int(config.get("num_threads", 1)))
    rho, seed, f, k, arms = job
    start = time.time()
    rows = run_cell(config, rho, seed, f, k, list(arms))
    for row in rows:
        row["job_seconds"] = time.time() - start
        row["job_key"] = f"rho={rho},seed={seed},f={f},K={k}"
    return rows


def _print(message: str) -> None:
    print(message, flush=True)


def run(config: dict[str, Any], output_dir: Path, *, workers: int | None = None, log=_print) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir)
    apply_thread_setting(config)
    todo = jobs(config)
    workers = workers or int(config.get("workers", 1))
    started = time.time()
    partial = output_dir / "metrics.partial.jsonl"
    results: dict[tuple, list[dict[str, Any]]] = {}

    def done(job, rows):
        results[job] = rows
        with partial.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(e10._jsonable(row)) + "\n")
        log(f"[e10.9] done {job[:4]} ({len(results)}/{len(todo)}, {time.time() - started:.0f}s)")

    if workers <= 1:
        for job in todo:
            done(job, _run_job(job, config))
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        try:
            futures = {pool.submit(_run_job, job, config): job for job in todo}
            for future in as_completed(futures):
                done(futures[future], future.result())
        except BaseException:
            # fail fast: a plain `with` block would wait for every queued job before re-raising
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
    rows = [json.loads(json.dumps(e10._jsonable(row))) for job in todo for row in results[job]]
    with (output_dir / "metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    partial.unlink()
    summary = summarize(rows, config)
    summary["wall_seconds"] = time.time() - started
    summary["cpu_seconds"] = float(sum({r["job_key"]: r["job_seconds"] for r in rows}.values()))
    (output_dir / "summary.json").write_text(json.dumps(e10._jsonable(summary), indent=2) + "\n")
    (output_dir / "report.md").write_text(render_report(summary, config))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", jobs=len(todo), workers=workers,
                       wall_seconds=summary["wall_seconds"])
    return summary


# =====================================================================================================
# summary and report (rows only)
# =====================================================================================================

def _nan(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _ci(values: list[float]) -> dict[str, Any]:
    values = [v for v in (_nan(x) for x in values) if v == v]
    if not values:
        return {"mean": float("nan"), "ci_low": None, "ci_high": None, "n": 0}
    return mean_confidence_interval(values)


def _paired(a: list[float], b_: list[float], *, resamples: int, seed: int = 0) -> dict[str, Any]:
    """Paired difference over seeds: 95% t-interval and paired percentile bootstrap."""
    pairs = [(x, y) for x, y in zip(a, b_) if x == x and y == y]
    if not pairs:
        return {"mean": float("nan"), "ci_low": None, "ci_high": None, "n": 0}
    out = mean_confidence_interval([x - y for x, y in pairs])
    if len(pairs) > 1:
        boot = paired_bootstrap_ci([x for x, _ in pairs], [y for _, y in pairs], resamples=resamples, seed=seed)
        out["boot_low"], out["boot_high"] = boot["ci_low"], boot["ci_high"]
    out["per_seed"] = [x - y for x, y in pairs]
    return out


def aulc(budgets: list[float], fits: list[float]) -> float:
    """Area under the learning curve over ln f, normalized by the ln-range (= mean fit over the log budget axis)."""
    if len(budgets) < 2 or any(v != v for v in fits):
        return float("nan")
    x = np.log(np.asarray(budgets, dtype=float))
    y = np.asarray(fits, dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def aulc_linear(budgets: list[float], fits: list[float]) -> float:
    if len(budgets) < 2 or any(v != v for v in fits):
        return float("nan")
    x = np.asarray(budgets, dtype=float)
    return float(np.trapezoid(np.asarray(fits, dtype=float), x) / (x[-1] - x[0]))


def to_criterion(xs: list[float], ys: list[float], target: float, *, log_x: bool = True) -> float:
    """Smallest x where the curve reaches `target` (linear interpolation, on ln x if `log_x`); inf if never,
    the first x if already reached there."""
    if not xs or target != target:
        return float("nan")
    if ys[0] >= target:
        return float(xs[0])
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y1 >= target:
            if y1 == y0:
                return float(x1)
            w = (target - y0) / (y1 - y0)
            if log_x:
                return float(math.exp(math.log(x0) + w * (math.log(x1) - math.log(x0))))
            return float(x0 + w * (x1 - x0))
    return float("inf")


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in rows if r.get("part") == "e10.9"]
    stats = config.get("statistics", {})
    resamples = int(stats.get("bootstrap_resamples", 10000))
    margin = float(stats.get("noninferiority_margin", 0.01))
    much = float(stats.get("much_faster_ratio", 0.5))
    k_main = int(config["schedule"]["reflect_every"])
    budgets = sorted({float(r["f"]) for r in rows if r["K"] == k_main})
    rhos = sorted({float(r["rho"]) for r in rows}, reverse=True)
    seeds = sorted({int(r["seed"]) for r in rows})
    main = [r for r in rows if r["K"] == k_main]
    cell = {(r["arm"], float(r["rho"]), int(r["seed"]), float(r["f"])): r for r in main}
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in main)]
    final = lambda a, rho, s, f, key="test_fit": _nan(cell[(a, rho, s, f)]["final"].get(key)) if (a, rho, s, f) in cell else float("nan")  # noqa: E731

    def curve(a, rho, s, key="test_fit"):
        return [final(a, rho, s, f, key) for f in budgets]

    summary: dict[str, Any] = {"budgets": budgets, "rhos": rhos, "seeds": seeds, "arms": arms, "K": k_main}
    # fit by arm × budget × rho
    table = {}
    for a in arms:
        for rho in rhos:
            for f in budgets:
                vals = [final(a, rho, s, f) for s in seeds]
                if any(v == v for v in vals):
                    table[f"{a}|{rho}|{f}"] = _ci(vals)
    summary["test_fit"] = table
    # AULC per arm × rho (arms with every budget): primary = linear f axis, secondary = ln f axis
    au, au_log = {}, {}
    for a in arms:
        for rho in rhos:
            per = {s: aulc_linear(budgets, curve(a, rho, s)) for s in seeds}
            if all(v == v for v in per.values()):
                au[f"{a}|{rho}"] = per
                au_log[f"{a}|{rho}"] = {s: aulc(budgets, curve(a, rho, s)) for s in seeds}
    summary["aulc"] = {k: _ci(list(v.values())) for k, v in au.items()}
    summary["aulc_log"] = {k: _ci(list(v.values())) for k, v in au_log.items()}

    def diff(x, y, rho, source=au):
        kx, ky = f"{x}|{rho}", f"{y}|{rho}"
        if kx not in source or ky not in source:
            return None
        return _paired([source[kx][s] for s in seeds], [source[ky][s] for s in seeds], resamples=resamples)

    contrasts = {}
    for rho in rhos:
        for x, y in CONTRASTS:
            d = diff(x, y, rho)
            if d is not None:
                contrasts[f"{x}-{y}|{rho}"] = d
            dl = diff(x, y, rho, au_log)
            if dl is not None:
                contrasts[f"{x}-{y}|{rho}|log"] = dl
    summary["aulc_contrasts"] = contrasts
    # fit contrasts at each budget
    at_f = {}
    for rho in rhos:
        for f in budgets:
            for x, y in CONTRASTS:
                a_vals = [final(x, rho, s, f) for s in seeds]; b_vals = [final(y, rho, s, f) for s in seeds]
                if any(v == v for v in a_vals) and any(v == v for v in b_vals):
                    at_f[f"{x}-{y}|{rho}|{f}"] = _paired(a_vals, b_vals, resamples=resamples)
    summary["fit_contrasts"] = at_f
    # observations to criterion: smallest f where H reaches P's f = 1 fit
    otc = {}
    for rho in rhos:
        for x in ("H", "H+R", "H-rand", "H-AMIE", "P+compute"):
            if f"{x}|{rho}" not in au or f"P|{rho}" not in au:
                continue
            per = {}
            for s in seeds:
                target = final("P", rho, s, budgets[-1])
                per[s] = to_criterion(budgets, curve(x, rho, s), target) / budgets[-1]
            mean_x = [float(np.mean([final(x, rho, s, f) for s in seeds])) for f in budgets]
            target = float(np.mean([final("P", rho, s, budgets[-1]) for s in seeds]))
            otc[f"{x}|{rho}"] = {"per_seed": per, "mean_curve": to_criterion(budgets, mean_x, target) / budgets[-1],
                                 "seeds_at_or_below": sum(v <= much for v in per.values()), "seeds": len(per)}
    summary["obs_to_criterion"] = otc
    # step efficiency at f = 1: steps (and compute) to P's final fit
    ste = {}
    for rho in rhos:
        f1 = budgets[-1]
        if ("P", rho, seeds[0], f1) not in cell:
            continue
        p_final = [cell[("P", rho, s, f1)]["final"]["test_fit"] for s in seeds if ("P", rho, s, f1) in cell]
        target = float(np.mean(p_final))
        for x in arms:
            if (x, rho, seeds[0], f1) not in cell:
                continue
            curves = [cell[(x, rho, s, f1)]["curve"] for s in seeds if (x, rho, s, f1) in cell]
            steps = [c["step"] for c in curves[0]]
            mean_fit = [float(np.mean([c[i]["test_fit"] for c in curves])) for i in range(len(steps))]
            mean_compute = [float(np.mean([c[i]["compute"] for c in curves])) for i in range(len(steps))]
            ste[f"{x}|{rho}"] = {"steps": to_criterion(steps[1:], mean_fit[1:], target, log_x=False),
                                 "compute": to_criterion(mean_compute[1:], mean_fit[1:], target, log_x=False),
                                 "total_steps": steps[-1], "total_compute": mean_compute[-1],
                                 "curve": {"step": steps, "test_fit": mean_fit, "compute": mean_compute}}
    summary["step_efficiency"] = ste
    # per arm × rho at f = 1 (and every budget): structure readouts and commitments
    readouts = {}
    keys = ("test_fit", "test_fit_without_rule_edges", "validation_concept_fit", "edge_auc_final", "ari", "mean_best_jaccard",
            "relation_jaccard_framed_mean", "rule_edges", "rule_edge_precision", "unobserved_test_recall",
            "unobserved_test_recall_matched", "unobserved_recall", "frame_f1_k1", "frame_f1_k2", "frame_f1_k4")
    for a in arms:
        for rho in rhos:
            for f in budgets:
                rs = [cell[(a, rho, s, f)] for s in seeds if (a, rho, s, f) in cell]
                if not rs:
                    continue
                entry = {}
                for key in keys:
                    if key == "edge_auc_final":
                        entry[key] = _ci([r["curve"][-1]["edge_auc"] if r["curve"] else None for r in rs])
                    else:
                        entry[key] = _ci([r["final"].get(key) for r in rs])
                cms = [c for r in rs for c in r["commitments"]]
                entry["commitments_per_run"] = _ci([len(r["commitments"]) for r in rs])
                entry["false_commitments_per_run"] = _ci([sum(not c.get("good", False) for c in r["commitments"]) for r in rs])
                entry["commitments"] = len(cms)
                entry["false_commitments"] = sum(not c.get("good", False) for c in cms)
                entry["retracted"] = sum(bool(c.get("retracted")) for c in cms)
                entry["false_retracted"] = sum(bool(c.get("retracted")) and not c.get("good", False) for c in cms)
                entry["true_retracted"] = sum(bool(c.get("retracted")) and c.get("good", False) for c in cms)
                entry["revised"] = sum(bool(c.get("revised")) for c in cms)
                entry["rules_adopted"] = sum(c.get("adopted") is not None for c in cms)
                entry["rules_true"] = sum(bool(c.get("rule_true")) for c in cms)
                kinds: dict[str, int] = defaultdict(int)
                for c in cms:
                    if c.get("adopted"):
                        kinds[c["adopted"]] += 1
                entry["rule_kinds"] = dict(sorted(kinds.items()))
                entry["compute"] = _ci([r["compute"] for r in rs])
                entry["seconds"] = _ci([r["seconds"] for r in rs])
                readouts[f"{a}|{rho}|{f}"] = entry
    summary["readouts"] = readouts
    # reflection write effect (immediate test-fit change of each reflection round, hybrid arms)
    writes = {}
    for a in arms:
        for rho in rhos:
            rs = [cell[(a, rho, s, budgets[-1])] for s in seeds if (a, rho, s, budgets[-1]) in cell]
            deltas = [x["test_fit_after"] - x["test_fit_before"] for r in rs for x in r["reflections"]
                      if x.get("test_fit_after") is not None and x.get("test_fit_before") is not None
                      and any(sl["commit"] for sl in x["slots"])]
            if deltas:
                writes[f"{a}|{rho}"] = {"rounds_with_commit": len(deltas), "mean_delta": float(np.mean(deltas)),
                                        "positive": sum(d > 0 for d in deltas)}
    summary["write_effects"] = writes
    wall = {}
    for a in arms:
        rs = [r for r in main if r["arm"] == a and r.get("reflection_seconds")]
        if rs:
            per_step = float(np.mean([r["passive_seconds_per_step"] for r in rs]))
            per_round = [x for r in rs for x in r["reflection_seconds"]]
            units = [x for r in rs for x in r["reflection_units"]]
            wall[a] = {"reflection_seconds_per_round": float(np.mean(per_round)), "passive_seconds_per_step": per_step,
                       "reflection_step_equivalents_wall": float(np.mean(per_round)) / per_step if per_step else float("nan"),
                       "reflection_units_per_round": float(np.mean(units)),
                       "reflection_step_equivalents_units": float(np.mean([u / (3.0 * r["batch_concepts"])
                                                                           for r in rs for u in r["reflection_units"]]))}
    summary["reflection_cost"] = wall
    # pre-registered readings
    pre: dict[str, Any] = {}
    top = rhos[0] if rhos else 1.0
    hp = contrasts.get(f"H-P|{top}")
    hc = contrasts.get(f"H-P+compute|{top}")
    hr = contrasts.get(f"H-H-rand|{top}")
    excl = lambda d: bool(d and d.get("ci_low") is not None and d["ci_low"] > 0)  # noqa: E731
    pre["primary_aulc_H_minus_P"] = hp
    pre["faster"] = excl(hp)
    o = otc.get(f"H|{top}")
    pre["obs_to_criterion"] = o
    pre["much_faster"] = bool(excl(hp) and o and o["mean_curve"] <= much and o["seeds_at_or_below"] >= max(1, o["seeds"] - 1))
    pre["H_minus_Pcompute"] = hc
    pre["primary_aulc_log_H_minus_P"] = contrasts.get(f"H-P|{top}|log")
    pre["H_minus_Psplit"] = contrasts.get(f"H-P-split|{top}")
    pre["H_minus_Hamie"] = contrasts.get(f"H-H-AMIE|{top}")
    pre["unobserved_test_recall_H"] = readouts.get(f"H|{top}|{budgets[-1]}", {}).get("unobserved_test_recall")
    cost = {}
    for rho in rhos:
        for x, y in (("H", "P"), ("H+R", "P"), ("H", "P-split"), ("H+R", "P-split")):
            d = at_f.get(f"{x}-{y}|{rho}|{budgets[-1]}")
            if d is not None:
                cost[f"{x}-{y}|{rho}"] = d
    pre["fit_at_full_data"] = cost
    pre["rho_curve_aulc_H_minus_P"] = {str(rho): contrasts.get(f"H-P|{rho}") for rho in rhos}
    pre["e10_9b_gate"] = {"aulc_H_minus_P_ci_excludes_0": excl(hp), "aulc_H_minus_Pcompute_ci_excludes_0": excl(hc),
                          "positive": bool(excl(hp) and excl(hc))}
    pre["refutation_H_approx_Pcompute"] = bool(hc is not None and not excl(hc))
    pre["refutation_H_approx_Hrand"] = bool(hr is not None and not excl(hr))
    nonin = {}
    for rho in rhos:
        if rho > 0.25:
            continue
        d = at_f.get(f"H+R-P|{rho}|{budgets[-1]}")
        if d is not None:
            nonin[str(rho)] = {"diff": d, "noninferior": bool(d.get("ci_low") is not None and d["ci_low"] > -margin)}
    pre["noninferiority_HR_vs_P"] = nonin
    pre["refutation_HR_harmful_weak_structure"] = any(not v["noninferior"] for v in nonin.values())
    null = readouts.get(f"H+R|0.0|{budgets[-1]}") if budgets else None
    if null:
        pre["null_false_commitments_retracted"] = {"false": null["false_commitments"], "retracted": null["false_retracted"],
                                                   "fraction": null["false_retracted"] / null["false_commitments"]
                                                   if null["false_commitments"] else float("nan")}
    summary["preregistered"] = pre
    # K sensitivity: the sensitivity cells and the matching main-K cells
    ks = config.get("k_sensitivity") or {}
    sens: dict[str, list[float]] = {}
    for r in rows:
        if r["arm"] not in ks.get("arms", []) or not _selected(ks.get("rho"), float(r["rho"]), rhos) \
                or not _selected(ks.get("f"), float(r["f"]), budgets):
            continue
        sens.setdefault(f"{r['arm']}|{r['rho']}|{r['f']}|K={r['K']}", []).append(r["final"]["test_fit"])
    summary["k_sensitivity"] = {k: _ci(v) for k, v in sens.items()} if ks.get("K") else {}
    return summary


def _fmt(ci: Any, digits: int = 3) -> str:
    if ci is None:
        return "—"
    if isinstance(ci, (int, float)):
        if ci != ci:
            return "—"
        return "∞" if ci == float("inf") else f"{ci:.{digits}f}"
    mean = ci.get("mean")
    if mean is None or mean != mean:
        return "—"
    if ci.get("ci_low") is None:
        return f"{mean:.{digits}f}"
    return f"{mean:+.{digits}f} [{ci['ci_low']:+.{digits}f}, {ci['ci_high']:+.{digits}f}]" if ci.get("per_seed") is not None \
        else f"{mean:.{digits}f} [{ci['ci_low']:.{digits}f}, {ci['ci_high']:.{digits}f}]"


def _boot(ci: dict[str, Any] | None, digits: int = 3) -> str:
    if not ci or ci.get("boot_low") is None:
        return "—"
    return f"[{ci['boot_low']:+.{digits}f}, {ci['boot_high']:+.{digits}f}]"


def render_report(summary: dict[str, Any], config: dict[str, Any]) -> str:
    budgets, rhos, arms = summary["budgets"], summary["rhos"], summary["arms"]
    f1 = budgets[-1] if budgets else 1.0
    top = rhos[0] if rhos else 1.0
    margin = config.get("statistics", {}).get("noninferiority_margin", 0.01)
    lines = ["# E10.9a — passive learning paired with active self-reflection", "",
             f"Seeds {summary['seeds']}; budgets f ∈ {budgets}; structure ρ ∈ {rhos}; reflection every K = {summary['K']} "
             f"steps of T = {config['schedule']['total_steps']}. Mean [95% t-interval over seeds]; paired differences also with a "
             "paired bootstrap over seeds. Test fit = mean cosine of composed test-concept rows (never observed, composition "
             "only) with their audit observations. AULC = area under the test-fit curve over the budget grid divided by its "
             "width (primary: linear f; secondary: ln f).", ""]
    pre = summary["preregistered"]

    def row(name: str, ci: Any, verdict: str, d: int = 4) -> str:
        return f"| {name} | {_fmt(ci, d)}; bootstrap {_boot(ci, d)} | {verdict} |"

    lines += ["## Pre-registered readings", "", "| Reading | Value | Verdict |", "|---|---|---|",
              row(f"**primary** AULC (linear f) H − P, ρ = {top}", pre.get("primary_aulc_H_minus_P"),
                  "faster" if pre["faster"] else "not faster")]
    o = pre.get("obs_to_criterion")
    if o:
        per = ", ".join(_fmt(v, 2) for v in o["per_seed"].values())
        lines.append(f"| observations to P's f = {f1} fit, H (fraction of P's data) | mean curve {_fmt(o['mean_curve'], 2)}; "
                     f"per seed {per} | {'much faster (≤ 0.5)' if pre['much_faster'] else 'not much faster'} |")
    lines.append(row(f"AULC (ln f) H − P, ρ = {top}", pre.get("primary_aulc_log_H_minus_P"), "secondary axis"))
    hc, hr = pre.get("H_minus_Pcompute"), summary["aulc_contrasts"].get(f"H-H-rand|{top}")
    lines.append(row(f"AULC H − P+compute, ρ = {top}", hc, "not run" if hc is None else
                     "refutation reading (H ≈ P+compute)" if pre["refutation_H_approx_Pcompute"] else "H ahead of P+compute"))
    lines.append(row(f"AULC H − H-rand, ρ = {top}", hr, "not run" if hr is None else
                     "refutation reading (H ≈ H-rand)" if pre["refutation_H_approx_Hrand"] else "content matters"))
    lines.append(row(f"AULC H − H-AMIE, ρ = {top}", pre.get("H_minus_Hamie"), "generator control"))
    lines.append(row(f"AULC H − P-split (same gradient data), ρ = {top}", pre.get("H_minus_Psplit"), "decomposition"))
    lines.append(f"| rule-implied recovery of unobserved test-head edges, H (P = 0 by construction), f = {f1} | "
                 f"{_fmt(pre.get('unobserved_test_recall_H'))} | — |")
    for rho, v in pre.get("noninferiority_HR_vs_P", {}).items():
        lines.append(row(f"H+R − P test fit, f = {f1}, ρ = {rho} (margin −{margin})", v["diff"],
                         "non-inferior" if v["noninferior"] else "not non-inferior (harmful reading)"))
    for key, v in pre.get("fit_at_full_data", {}).items():
        name, rho = key.split("|")
        if float(rho) <= 0.25 and "P-split" in name:
            lines.append(row(f"{name} test fit, f = {f1}, ρ = {rho} (cost of commitment at equal gradient data)", v, "—"))
    nr = pre.get("null_false_commitments_retracted")
    if nr:
        lines.append(f"| false commitments retracted by H+R, ρ = 0, f = {f1} | {nr['retracted']} of {nr['false']} | {_fmt(nr['fraction'], 2)} |")
    gate = pre["e10_9b_gate"]
    lines += [f"| E10.9b gate: AULC H − P and H − P+compute, 95% t-intervals exclude 0 | {gate['aulc_H_minus_P_ci_excludes_0']} / "
              f"{gate['aulc_H_minus_Pcompute_ci_excludes_0']} | {'POSITIVE' if gate['positive'] else 'not met'} |", ""]
    lines += ["## ρ curve of the H − P gain", "",
              "| ρ | AULC linear H − P | AULC ln f H − P | test fit H − P at f = 1 | H+R − P at f = 1 |", "|---|---|---|---|---|"]
    for rho in rhos:
        lines.append(f"| {rho} | {_fmt(summary['aulc_contrasts'].get(f'H-P|{rho}'), 4)} | "
                     f"{_fmt(summary['aulc_contrasts'].get(f'H-P|{rho}|log'), 4)} | "
                     f"{_fmt(summary['fit_contrasts'].get(f'H-P|{rho}|{f1}'), 4)} | "
                     f"{_fmt(summary['fit_contrasts'].get(f'H+R-P|{rho}|{f1}'), 4)} |")
    lines.append("")
    for rho in rhos:
        lines += [f"## Test-concept fit by data budget, ρ = {rho}", "",
                  "| Arm | " + " | ".join(f"f = {f}" for f in budgets) + " | AULC linear | AULC ln f |",
                  "|---|" + "---|" * (len(budgets) + 2)]
        for a in arms:
            vals = [summary["test_fit"].get(f"{a}|{rho}|{f}") for f in budgets]
            if not any(vals):
                continue
            lines.append(f"| {a} | " + " | ".join(_fmt(v) for v in vals) + f" | {_fmt(summary['aulc'].get(f'{a}|{rho}'), 4)} | "
                         f"{_fmt(summary['aulc_log'].get(f'{a}|{rho}'), 4)} |")
        lines.append("")
    lines += ["## AULC contrasts (paired over seeds)", "", "| Contrast | ρ | AULC linear [t] | bootstrap | AULC ln f [t] |",
              "|---|---|---|---|---|"]
    for key, v in summary["aulc_contrasts"].items():
        if key.endswith("|log"):
            continue
        name, rho = key.split("|")
        lines.append(f"| {name} | {rho} | {_fmt(v, 4)} | {_boot(v, 4)} | {_fmt(summary['aulc_contrasts'].get(key + '|log'), 4)} |")
    lines += ["", f"## Test-fit contrasts by budget (paired over seeds), ρ = {top}", "",
              "| Contrast | " + " | ".join(f"f = {f}" for f in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for x, y in CONTRASTS:
        vals = [summary["fit_contrasts"].get(f"{x}-{y}|{top}|{f}") for f in budgets]
        if any(vals):
            lines.append(f"| {x} − {y} | " + " | ".join(_fmt(v, 3) for v in vals) + " |")
    lines += ["", f"## Observations to criterion (fraction of the data P needs for its f = {f1} fit)", "",
              "| Arm | ρ | mean curve | per seed | seeds ≤ 0.5 |", "|---|---|---|---|---|"]
    for key, v in summary["obs_to_criterion"].items():
        a, rho = key.split("|")
        lines.append(f"| {a} | {rho} | {_fmt(v['mean_curve'], 2)} | {', '.join(_fmt(x, 2) for x in v['per_seed'].values())} | "
                     f"{v['seeds_at_or_below']} / {v['seeds']} |")
    lines += ["", f"## Step efficiency at f = {f1} (seed-mean curves): steps and compute to P's final fit", "",
              "| Arm | ρ | steps to criterion | compute to criterion (reference steps) | total compute |", "|---|---|---|---|---|"]
    for key, v in summary["step_efficiency"].items():
        a, rho = key.split("|")
        lines.append(f"| {a} | {rho} | {_fmt(v['steps'], 0)} | {_fmt(v['compute'], 0)} | {_fmt(v['total_compute'], 0)} |")
    if summary.get("reflection_cost"):
        lines += ["", "## Cost of one reflection round", "",
                  "| Arm | rows composed (3 × with gradient + without) | step equivalents (rows) | wall s / round | step equivalents (wall) |",
                  "|---|---|---|---|---|"]
        for a, v in summary["reflection_cost"].items():
            lines.append(f"| {a} | {v['reflection_units_per_round']:.0f} | {v['reflection_step_equivalents_units']:.2f} | "
                         f"{v['reflection_seconds_per_round']:.3f} | {v['reflection_step_equivalents_wall']:.1f} |")
    lines += ["", f"## Structure readouts at f = {f1}", "",
              "| Arm | ρ | test fit | without rule edges | hidden-edge AUC | ARI | mean best J | framed J (incl. rules) | rule edges | "
              "rule-edge gold precision | unobserved test-head recall | frame F1 k=1/2/4 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        for rho in rhos:
            e = summary["readouts"].get(f"{a}|{rho}|{f1}")
            if not e:
                continue
            lines.append(f"| {a} | {rho} | {_fmt(e['test_fit'])} | {_fmt(e['test_fit_without_rule_edges'])} | {_fmt(e['edge_auc_final'])} | "
                         f"{_fmt(e['ari'])} | {_fmt(e['mean_best_jaccard'])} | {_fmt(e['relation_jaccard_framed_mean'])} | "
                         f"{_fmt(e['rule_edges'], 0)} | {_fmt(e['rule_edge_precision'])} | {_fmt(e['unobserved_test_recall'])} | "
                         f"{_fmt(e['frame_f1_k1'], 2)} / {_fmt(e['frame_f1_k2'], 2)} / {_fmt(e['frame_f1_k4'], 2)} |")
    lines += ["", f"## Commitments at f = {f1} (gold verdicts: evaluation only)", "",
              "| Arm | ρ | commitments / run | false / run | rules adopted (true) | retracted (false / true) | revised | rule kinds |",
              "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        for rho in rhos:
            e = summary["readouts"].get(f"{a}|{rho}|{f1}")
            if not e or a in PASSIVE:
                continue
            kinds = ", ".join(f"{k} ×{v}" for k, v in e["rule_kinds"].items()) or "—"
            lines.append(f"| {a} | {rho} | {_fmt(e['commitments_per_run'], 2)} | {_fmt(e['false_commitments_per_run'], 2)} | "
                         f"{e['rules_adopted']} ({e['rules_true']}) | {e['retracted']} ({e['false_retracted']} / {e['true_retracted']}) | "
                         f"{e['revised']} | {kinds} |")
    lines += ["", f"## Commitments and rules by budget, H, ρ = {top}", "",
              "| f | commitments / run | false / run | rules adopted (true) | rule edges | unobserved test-head recall |",
              "|---|---|---|---|---|---|"]
    for f in budgets:
        e = summary["readouts"].get(f"H|{top}|{f}")
        if e:
            lines.append(f"| {f} | {_fmt(e['commitments_per_run'], 2)} | {_fmt(e['false_commitments_per_run'], 2)} | "
                         f"{e['rules_adopted']} ({e['rules_true']}) | {_fmt(e['rule_edges'], 0)} | {_fmt(e['unobserved_test_recall'])} |")
    if summary.get("write_effects"):
        lines += ["", f"## Immediate effect of a committing reflection round on test fit (f = {f1})", "",
                  "| Arm | ρ | rounds with a commitment | mean Δ test fit | rounds with Δ > 0 |", "|---|---|---|---|---|"]
        for key, v in summary["write_effects"].items():
            a, rho = key.split("|")
            lines.append(f"| {a} | {rho} | {v['rounds_with_commit']} | {v['mean_delta']:+.4f} | {v['positive']} |")
    if summary.get("k_sensitivity"):
        lines += ["", "## K sensitivity (test fit)", "", "| Arm | ρ | f | K | test fit |", "|---|---|---|---|---|"]
        for key in sorted(summary["k_sensitivity"]):
            a, rho, f, k = key.split("|")
            lines.append(f"| {a} | {rho} | {f} | {k[2:]} | {_fmt(summary['k_sensitivity'][key])} |")
    lines += ["", f"Wall time {summary.get('wall_seconds', float('nan')):.0f} s; job CPU time "
              f"{summary.get('cpu_seconds', float('nan')) / 3600:.2f} h.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None, help="re-render report.md of a finished run folder")
    args = parser.parse_args(argv)
    if args.report is not None:
        config = yaml.safe_load((args.report / "resolved_config.yaml").read_text())
        rows = [json.loads(line) for line in (args.report / "metrics.jsonl").read_text().splitlines() if line.strip()]
        summary = summarize(rows, config)
        old = json.loads((args.report / "summary.json").read_text()) if (args.report / "summary.json").exists() else {}
        for key in ("wall_seconds", "cpu_seconds"):
            if key in old:
                summary[key] = old[key]
        (args.report / "summary.json").write_text(json.dumps(e10._jsonable(summary), indent=2) + "\n")
        (args.report / "report.md").write_text(render_report(summary, config))
        print((args.report / "report.md").read_text())
        return
    config = resolve(yaml.safe_load(args.config.read_text()))
    if args.seeds:
        config["seeds"] = args.seeds
    summary = run(config, args.output, workers=args.workers)
    print(json.dumps({"wall_seconds": summary.get("wall_seconds"), "preregistered": summary.get("preregistered")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
